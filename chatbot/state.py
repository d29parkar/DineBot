from typing_extensions import TypedDict

class State(TypedDict):
    input: str
    intent: str
    structured_results: list
    faiss_results: list
    google_results: list
    response: str
