from chatbot.state import State  # ✅ Use the correct state structure
from chatbot.config import llm  # Import LLM from config.py

def generate_llm_response(user_query, intent, results):
    prompt = f"""
    You are an intelligent assistant that can answer a variety of questions about restaurants, their 
    menus, and their ingredients, leveraging both an internal proprietary dataset and external public
    datasets.

    ## User Query:
    "{user_query}"

    ## Intent:
    "{intent}"

    ## Search Results:
    {results}

    ## Task:
    1. **Verify the relevance** of the provided results to the user's query. Look if the results are exact matches, similar recommendations, or external references.
    2. If results are relevant, generate a **natural, engaging, and informative response** summarizing them.
    3. If results are **not relevant**, politely inform the user and suggest alternative ways they might refine their query.
    4. Be **concise yet helpful**, keeping the response succinct unless more details are necessary.
    5. Give more preference to highly relevant restaurant matches than the others.
    6. Do not mention any kind of errors that had occurred during the search process.
    7. First properly vet the results and then generate a response.
    ## Response Formatting:
    - Start with a direct response to the user query.
    - List the most relevant restaurants first, with dish names and key details.
    - Offer alternatives if exact matches are unavailable.
    - Keep it **concise, friendly, and informative**.

    
    Respond conversationally and make sure the final response is user-friendly.
    """

    # Invoke LLM and generate a response
    return llm.invoke(prompt[:6000]).content.strip()


def generate_response(state: State, result_key: str) -> State:
    """Generates a response using LLM by verifying and refining structured, FAISS, and Google results."""

    user_query = state["input"]
    results = state.get(result_key, [])
    intent = state.get("intent", "")

    if not results:
        return state
    
    # Format results as text for LLM
    results_summary = []
    llm_response = ""  # Initialize before appending responses

    if result_key == "structured_results":
        structured_prompt = f"**Highly relevant restaurant matches:**\n{results}"
        llm_response = generate_llm_response(user_query, intent, structured_prompt)

    elif result_key == "faiss_results":
        faiss_prompt = f"**Similarity-based recommendations:**\n{results}"
        llm_response = generate_llm_response(user_query, intent, faiss_prompt)

    elif result_key == "google_results":
        google_prompt = f"**External references:**\n{results}"
        llm_response = generate_llm_response(user_query, intent, google_prompt)
    
    elif result_key == "graph_results":
        graph_prompt = f"**Internal dataset matches:**\n{results}"
        llm_response = generate_llm_response(user_query, intent, graph_prompt)

    elif result_key == "llm_made_graph_results":
        llm_made_graph_results = f"**Internal dataset matches:**\n{results}"
        llm_response = generate_llm_response(user_query, intent, llm_made_graph_results) 
    
    else:
        return state

    # Append to state["response"] instead of overwriting
    if "response" in state and state["response"]:
        state["response"] += f"\n\n{llm_response}"
    else:
        state["response"] = llm_response


    return state
