import json
from chatbot.google_search import google_search  # Ensure this function is correctly imported
from chatbot.state import State


def test_google_search():
    test_cases = [
        # Ingredient-Based Discovery
        {"input": "Which restaurants in Los Angeles offer dishes with Impossible Meat?", "intent": "entity_extraction"},
        {"input": "Find restaurants near me that serve gluten-free pizza.", "intent": "entity_extraction"},
        
        # Trending Insights & Explanations
        {"input": "Give me a summary of the latest trends around desserts in San Francisco.", "intent": "trending_insights"},
        
        # Historical or Cultural Context
        {"input": "What is the history of sushi, and which restaurants in my area are known for it?", "intent": "historical_context"},
        
        # Comparative Analysis
        {"input": "Compare the average menu price of vegan restaurants in San Francisco vs. Mexican restaurants.", "intent": "comparative_analysis"},
        
        # Menu Innovation & Flavor Trends
        {"input": "How has the use of saffron in desserts changed over the last year, according to restaurant menus or news articles?", "intent": "menu_innovation"},
        
        # Additional Edge Cases
        {"input": "Find the cheapest three-star Michelin restaurants in New York.", "intent": "comparative_analysis"},
        {"input": "What are the best vegan restaurants in Paris according to customer reviews?", "intent": "entity_extraction"},
        {"input": "Tell me the cultural significance of ramen in Japan.", "intent": "historical_context"},
        {"input": "Compare how fast-food chains have adjusted their prices over the last five years.", "intent": "comparative_analysis"},
        {"input": "What are the latest health food trends in Los Angeles?", "intent": "trending_insights"},
    ]

    for i, test_case in enumerate(test_cases, start=1):
        print(f"\nRunning Test Case {i}: {test_case['input']}")

        # Initialize state
        state = State()
        state["input"] = test_case["input"]
        state["intent"] = test_case["intent"]

        # Run the function directly (no async needed)
        try:
            updated_state = google_search(state)  # Call the sync function
            output = updated_state.get("google_results", {})

            # Pretty print the results
            print(json.dumps(output, indent=2, ensure_ascii=False))

        except Exception as e:
            print(f"Test Case {i} Failed: {str(e)}")


# Run tests
if __name__ == "__main__":
    test_google_search()  # Directly call the function without asyncio
