from chatbot.langgraph_workflow import app

def get_response(user_input):
    """Correctly initializes the state to match `TypedDict` in `langgraph_workflow.py`."""
    state = {
        "input": user_input,
        "intent": "",
        "structured_results": [],
        "faiss_results": [],
        "google_results": [],
        "response": "",
    }
    return app.invoke(state)["response"]
