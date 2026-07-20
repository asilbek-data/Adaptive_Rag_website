"""
Wires the nodes together into the decision graph (ported from
Agentic_RAG_ready.ipynb cell 21). This is the same graph shown in the
project guide's "Visual 1 — The agentic decision graph".
"""
from langgraph.graph import StateGraph, END

from .graph_state import GraphState
from .nodes import (
    retrieve,
    grade_documents,
    web_search,
    generate,
    route_after_grade,
    route_after_generate,
)


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("web_search", web_search)
    g.add_node("generate", generate)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade_documents")

    g.add_conditional_edges(
        "grade_documents", route_after_grade,
        {"web_search": "web_search", "generate": "generate"},
    )
    g.add_edge("web_search", "generate")

    g.add_conditional_edges(
        "generate", route_after_generate,
        {"useful": END, "not_grounded": "generate", "not_useful": "web_search"},
    )

    return g.compile()


rag_app = build_graph()
