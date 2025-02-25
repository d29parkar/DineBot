from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from chatbot.intent_recognition import detect_intent
from chatbot.ingredient_search import query_database
from chatbot.faiss_search import search_faiss
from chatbot.google_search import google_search
from chatbot.response_generator import generate_response
from chatbot.state import State


# **2️⃣ Define Gate Functions for Conditional Transitions**
def check_structured_results(state: State):
    """Gate function to check structured search results."""
    return "Found" if state.get("structured_results") else "Not Found"


def check_faiss_results(state: State):
    """Gate function to check FAISS search results."""
    return "Found" if state.get("faiss_results") else "Not Found"


# **3️⃣ Initialize LangGraph Workflow**
graph = StateGraph(State)

# **4️⃣ Add Nodes to the Graph**
graph.add_node("intent_recognition", detect_intent)
graph.add_node("structured_search", query_database)
graph.add_node("faiss_search", search_faiss)
graph.add_node("google_fallback", google_search)
graph.add_node("generate_response", generate_response)

# **5️⃣ Define Edges and Conditional Branching**
graph.add_edge(START, "intent_recognition")
graph.add_edge("intent_recognition", "structured_search")

graph.add_conditional_edges(
    "structured_search",
    check_structured_results,
    {
        "Found": "google_fallback",
        "Not Found": "faiss_search",
    }
)

graph.add_conditional_edges(
    "faiss_search",
    check_faiss_results,
    {
        "Found": "generate_response",
        "Not Found": "google_fallback",
    }
)

graph.add_edge("google_fallback", "generate_response")
graph.add_edge("generate_response", END)

# **6️⃣ Compile the Workflow**
app = graph.compile()
