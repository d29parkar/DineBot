from typing_extensions import TypedDict

class State(TypedDict):
    input: str
    intent: str
    structured_results: list
    entities : dict
    faiss_results: list
    google_results: list
    graph_results: list
    llm_made_graph_results: list
    response: str
