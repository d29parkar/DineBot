from langgraph.graph import StateGraph, START, END
from chatbot.intent_recognition import detect_intent
from chatbot.entity_extraction import extract_entities
from chatbot.structured_db_search import query_database
from chatbot.faiss_search import search_faiss
from chatbot.google_search import google_search
from chatbot.llm_graph_search import query_knowledge_graph
from chatbot.response_generator import generate_response
from chatbot.state import State
import pandas as pd
from chatbot.config import llm

def introduce_chatbot(state):
    """Generate an introduction when a user greets or says something generic."""
    return {"response": "Hello! 👋 I'm your AI restaurant assistant. I can help you discover restaurants, explore menus, and find trending food items. How can I assist you today?"}

# Define intent detection with a fallback
def get_intent(state):
    intent = state.get("intent", "fallback")
    valid_intents = {
        "ingredient_discovery", "trending_insights", "historical_context",
        "comparative_analysis", "menu_innovation", "fallback"
    }
    if intent not in valid_intents:
        print(f"WARNING: Unknown intent detected: {intent}")
        return "fallback"
    return intent

def refine_final_response(state):
    """
    Calls an LLM to properly structure and align the responses present in state["response"].
    This function ensures a coherent and structured final output.
    """
    responses = state.get("response", [])
    
    if not responses:
        return {"final_response": "I'm sorry, but I couldn't find relevant information. How else can I assist you?"}

    # Combine responses into a single input for the LLM
    input_text = "\n".join(responses)
    user_input = state["input"]
    user_intent = state["intent"]

    # Call the LLM to refine the final response

    prompt = f"""
    You are an AI assistant specializing in food, dining, and restaurant recommendations. Your task is to refine and synthesize multiple responses retrieved from different sources into a well-structured, contextually accurate, and engaging final response.
    Original User Query: "{user_input}"
    User Intent: "{user_intent}"
    **Key Refinement Guidelines:**
    1. **Logical Flow & Clarity**: Ensure a smooth transition between ideas, removing contradictions and redundant statements.
    2. **Precision & Accuracy**: If different sources provide conflicting information, resolve inconsistencies logically.
    3. **Professional & Engaging Tone**: Use a polished, informative, and engaging tone suitable for professional or expert readers.
    4. **Relevance**: Focus on the most relevant and critical information based on user query and user intent.
    5. **Structured Format**:
    - **Brief Introduction**: Summarize the key responses in 1-2 sentences.
    - **Main Insights**: Present the key findings in a structured way (bullets or short paragraphs for readability).
    - **Expert Opinions (only if applicable)**: Integrate expert perspectives naturally without abrupt shifts.
    - **Conclusion**: Provide a brief, insightful wrap-up and Offer a summary and guidance for the user’s next action (e.g., reservation links, alternative options)..
    6. No need to explicitly mention section headers (e.g., “Introduction” or “Conclusion”)—just ensure a natural, flowing response.
    7. External references should be explicitly cited, but internal references can be integrated seamlessly.

    **Here are the responses:**
    {input_text}

    Now, synthesize them into a single refined response that is well-structured, professional, and compelling.

    """
    structured_response = llm.invoke(prompt[:6000]).content.strip()
    state["response"] = structured_response
    return state

# **Initialize LangGraph**
graph = StateGraph(State)

# **Define Nodes (Agents)**
graph.add_node("intent_recognition", detect_intent)
graph.add_node("entity_extraction", extract_entities)
graph.add_node("structured_search", query_database)
graph.add_node("faiss_search", search_faiss)
graph.add_node("google_search", google_search)
graph.add_node("llm_graph_search", query_knowledge_graph)

# **Create Custom Response Nodes to Handle Different Result Keys**
graph.add_node("generate_structured_response", lambda state: generate_response(state, result_key="structured_results"))
graph.add_node("generate_faiss_response", lambda state: generate_response(state, result_key="faiss_results"))
graph.add_node("generate_google_response", lambda state: generate_response(state, result_key="google_results"))
graph.add_node("generate_llm_graph_response", lambda state: generate_response(state, result_key="llm_made_graph_results"))
# Add the final response node to the graph
graph.add_node("refine_response", refine_final_response)

# **Define Workflow**
graph.add_edge(START, "intent_recognition")
graph.add_edge("intent_recognition", "entity_extraction")

# **Conditional Routing Based on Intent**
# Add an introduction response node
graph.add_node("introduce_chatbot", introduce_chatbot)

graph.add_conditional_edges(
    "entity_extraction",
    get_intent,
    {
        "ingredient_discovery": "structured_search",
        "trending_insights": "llm_graph_search",
        "historical_context": "google_search",
        "comparative_analysis": "faiss_search",
        "menu_innovation": "faiss_search",
        "fallback": "introduce_chatbot"  # Send to introduction instead of Google Search
    }
)

# **Handling Structured Search Results**
graph.add_conditional_edges(
    "structured_search",
    lambda state: "Found" if state.get("structured_results") else "Not Found",
    {
        "Found": "generate_structured_response",
        "Not Found": "llm_graph_search"
    }
)

# **Handling LLM Graph Search Results**
graph.add_conditional_edges(
    "llm_graph_search",
    lambda state: "Found" if state.get("llm_made_graph_results") else "Not Found",
    {
        "Found": "generate_llm_graph_response",
        "Not Found": "faiss_search"
    }
)

# **Handling FAISS and Google Search**
graph.add_edge("faiss_search", "generate_faiss_response")
graph.add_edge("generate_faiss_response", "google_search")
graph.add_edge("generate_llm_graph_response", "google_search")
graph.add_edge("generate_structured_response", "google_search")
graph.add_edge("google_search", "generate_google_response")
graph.add_edge("introduce_chatbot", END)


# **End the Workflow**
#graph.add_edge("generate_structured_response", END)
#graph.add_edge("generate_faiss_response", END)
graph.add_edge("generate_google_response", "refine_response")
#graph.add_edge("generate_llm_graph_response", END)
graph.add_edge("refine_response", END)

# **Compile Workflow**
app = graph.compile()
