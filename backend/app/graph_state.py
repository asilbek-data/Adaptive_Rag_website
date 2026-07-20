from typing import List, TypedDict


class GraphState(TypedDict, total=False):
    question: str          # user's question
    generation: str         # the LLM-written answer
    documents: List[str]     # text chunks currently in context
    sources: List[dict]       # for citations: {"source":.., "page":.., "type":..}
    web_used: bool               # whether the web fallback was used
    retries: int                  # how many times "generate" has re-run
    steps: List[str]                # ordered list of visited nodes (debug/UI)
