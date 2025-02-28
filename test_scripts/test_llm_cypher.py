from chatbot.llm_graph_search import query_knowledge_graph
from chatbot.state import State

def test_full_pipeline():
    """Runs a complete pipeline test by extracting entities, detecting intent, and querying the database."""

    test_queries = [
        "Which restaurants in San Francisco offer dishes with Impossible Meat?",
        "Find restaurants in San Francisco that serve gluten-free pizza.",
        "Give me a summary of the latest trends around desserts in San Francisco.",
        "Compare the average menu price of vegan restaurants in San Francisco vs. Mexican restaurants.",
        "How has the use of saffron in desserts changed over the last year, according to restaurant menus or news articles?"
    ]

    for query in test_queries:
        # Step 1: Initialize state
        state = {
        "input": query,
        "intent": "",
        "structured_results": [],
        "entities": {},
        "faiss_results": [],
        "google_results": [],
        "graph_results": [],
        "llm_made_graph_results": [],
        "response": ""
        }


        # Step 2: Extract entities
        print(f"🟢 Extracting entities for: {query}")
        state = query_knowledge_graph(state)
        print("🔹 LLM Made Graph Results:", state["llm_made_graph_results"])
        print("=" * 120)

        

if __name__ == "__main__":
    test_full_pipeline()
