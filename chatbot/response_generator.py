from chatbot.state import State  # ✅ Use the correct state structure
from chatbot.config import llm  # Import LLM from config.py

def generate_response(state: State) -> State:
    """Generates a response using LLM by verifying and refining structured, FAISS, and Google results."""

    user_query = state["input"]
    structured_results = state.get("structured_results", [])
    faiss_results = state.get("faiss_results", [])
    google_results = state.get("google_results", [])

    # Format results as text for LLM
    results_summary = []

    if structured_results:
        structured_text = "\n".join([f"- {r['restaurant_name']}" for r in structured_results])
        results_summary.append(f"**Highly relevant restaurant matches:**\n{structured_text}")

    if faiss_results:
        faiss_text = "\n".join([f"- {r['restaurant_name']}" for r in faiss_results])
        results_summary.append(f"**Similarity-based recommendations:**\n{faiss_text}")

    if google_results:
        google_text = "\n".join([f"- {url}" for url in google_results])
        results_summary.append(f"**External references:**\n{google_text}")

    # If no results found, add a default message
    if not results_summary:
        response_text = "No directly relevant results were found based on the query."
    else:
        response_text = "\n\n".join(results_summary)

    # **Construct a powerful LLM prompt**
    prompt = f"""
    You are an intelligent assistant that can answer a variety of questions about restaurants, their 
    menus, and their ingredients, leveraging both an internal proprietary dataset and external public
    datasets.

    ## User Query:
    "{user_query}"

    ## Search Results:
    {response_text}

    ## Task:
    1. **Verify the relevance** of the provided results to the user's query. Look if the results are exact matches, similar recommendations, or external references.
    2. If results are relevant, generate a **natural, engaging, and informative response** summarizing them.
    3. If results are **not relevant**, politely inform the user and suggest alternative ways they might refine their query.
    4. Be **concise yet helpful**, keeping the response succinct unless more details are necessary.
    5. Give more preference to highly relevant restaurant matches than the others.
    
    Respond conversationally and make sure the final response is user-friendly.
    """

    # Invoke LLM and generate a response
    state["response"] = llm.invoke(prompt).content.strip()

    return state
