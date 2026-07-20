"""
The nodes of the agentic decision graph. Ported directly from
Agentic_RAG_ready.ipynb (cells 14-19): retrieve, grade_documents, web_search,
generate, and the groundedness / answers-question graders that decide the
routing after "generate".
"""
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

from .config import get_llm, vectorstore, MAX_RETRIES
from .graph_state import GraphState


# ---------- structured-output graders ----------

class RelevanceGrade(BaseModel):
    """Is a single retrieved chunk relevant to the question?"""
    binary_score: str = Field(description="'yes' or 'no'")


class GroundednessGrade(BaseModel):
    """Is the answer grounded in / supported by the given facts?"""
    binary_score: str = Field(description="'yes' = grounded, 'no' = fabricated")


class AnswersQuestionGrade(BaseModel):
    """Does the answer actually resolve the question?"""
    binary_score: str = Field(description="'yes' or 'no'")


# ---------- 1) RETRIEVE ----------

def retrieve(state: GraphState) -> dict:
    question = state["question"]
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)

    return {
        "documents": [d.page_content for d in docs],
        "sources": [
            {
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page"),
                "type": d.metadata.get("type", "text"),
            }
            for d in docs
        ],
        "web_used": False,
        "retries": 0,
        "steps": ["retrieve"],
    }


# ---------- 2) GRADE DOCUMENTS ----------

_relevance_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing relevance of a retrieved document chunk to a "
               "user question. Give a binary score 'yes' or 'no'. 'yes' means the chunk "
               "contains keywords or semantic meaning related to the question, even partially."),
    ("human", "Retrieved chunk:\n\n{document}\n\nUser question: {question}"),
])


def grade_documents(state: GraphState) -> dict:
    llm = get_llm()
    grader = _relevance_prompt | llm.with_structured_output(RelevanceGrade)

    question = state["question"]
    docs = state["documents"]
    sources = state.get("sources", [])

    kept_docs, kept_sources = [], []
    for doc, src in zip(docs, sources):
        result = grader.invoke({"document": doc, "question": question})
        if result.binary_score.strip().lower().startswith("y"):
            kept_docs.append(doc)
            kept_sources.append(src)

    return {
        "documents": kept_docs,
        "sources": kept_sources,
        "steps": state.get("steps", []) + ["grade_documents"],
    }


def route_after_grade(state: GraphState) -> str:
    """At least one relevant chunk left -> go straight to generate.
    Otherwise (everything filtered out) -> web search."""
    if len(state.get("documents", [])) == 0:
        return "web_search"
    return "generate"


# ---------- 3) WEB SEARCH (Tavily fallback) ----------

def web_search(state: GraphState) -> dict:
    question = state["question"]
    tool = TavilySearchResults(max_results=4)
    items = tool.invoke({"query": question})  # list of {"content":.., "url":..}

    new_docs = [item.get("content", "") for item in items]
    new_sources = [{"source": item.get("url", "web"), "page": None, "type": "web"} for item in items]

    return {
        "documents": state.get("documents", []) + new_docs,
        "sources": state.get("sources", []) + new_sources,
        "web_used": True,
        "steps": state.get("steps", []) + ["web_search"],
    }


# ---------- 4) GENERATE ----------

_generate_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant answering questions using ONLY the provided context. "
               "If the context does not contain the answer, say you don't know — do not "
               "make anything up. Answer in the same language as the question. Be concise "
               "and cite which piece of context you used when relevant."),
    ("human", "Context:\n\n{context}\n\nQuestion: {question}"),
])


def generate(state: GraphState) -> dict:
    llm = get_llm(temperature=0.2)
    chain = _generate_prompt | llm

    context = "\n\n---\n\n".join(state.get("documents", []))
    question = state["question"]

    response = chain.invoke({"context": context, "question": question})

    return {
        "generation": response.content,
        "retries": state.get("retries", 0) + 1,
        "steps": state.get("steps", []) + ["generate"],
    }


# ---------- 5) GRADE GENERATION (groundedness + answers-question) ----------

_groundedness_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing whether an answer is grounded in / supported by "
               "a given set of facts. Give a binary score 'yes' or 'no'. 'yes' means the "
               "answer is supported by the facts, with no fabricated claims."),
    ("human", "Facts:\n\n{documents}\n\nAnswer: {generation}"),
])

_answers_question_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing whether an answer actually resolves a user "
               "question. Give a binary score 'yes' or 'no'."),
    ("human", "Question: {question}\n\nAnswer: {generation}"),
])


def route_after_generate(state: GraphState) -> str:
    if state.get("retries", 0) >= MAX_RETRIES:
        return "useful"  # retry cap guarantees the graph terminates

    llm = get_llm()
    documents = "\n\n".join(state.get("documents", []))
    generation = state["generation"]
    question = state["question"]

    grounded = (
        (_groundedness_prompt | llm.with_structured_output(GroundednessGrade))
        .invoke({"documents": documents, "generation": generation})
        .binary_score.strip().lower().startswith("y")
    )
    if not grounded:
        return "not_grounded"

    answers = (
        (_answers_question_prompt | llm.with_structured_output(AnswersQuestionGrade))
        .invoke({"question": question, "generation": generation})
        .binary_score.strip().lower().startswith("y")
    )
    if not answers:
        return "not_useful"

    return "useful"
