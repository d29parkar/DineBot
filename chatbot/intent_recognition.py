import re
from chatbot.config import slm  # Use smaller LLM for intent extraction
from chatbot.state import State

def detect_intent(state: State) -> State:
    """Uses LLM to identify user intent and updates state."""
    user_input = state["input"]
    
    intent_prompt = f"""
    You are an intent detection AI assistant. Classify the given user query into one of the following categories:
    
    1. **Ingredient Discovery** → Questions about finding restaurants with specific ingredients or dietary options.
       - Example: "Which restaurants serve dishes with Impossible Meat?"
       - Example: "Find gluten-free pizza near me."
    
    2. **Trending Insights** → Queries about recent food trends, popularity, and emerging dishes.
       - Example: "What are the latest trends in desserts?"
       - Example: "Popular dishes in New York restaurants?"
    
    3. **Historical Context** → Requests for cultural or historical information about food.
       - Example: "What is the history of sushi?"
       - Example: "Origin of ramen and where to eat it?"
    
    4. **Comparative Analysis** → Questions about comparing menu prices, cuisines, or food statistics.
       - Example: "Compare the average price of vegan and Mexican restaurants in San Francisco."
       - Example: "Which city has cheaper fine dining: LA or NYC?"
    
    5. **Menu Innovation** → Questions about changes in food trends over time.
       - Example: "How has the use of saffron in desserts changed in the last year?"
       - Example: "New ingredient trends in cocktails?"
    
    **User Query:** "{user_input}"
    
    **Task:** Identify the best-matching category from the list above. Return only the category name.
    """
    
    llm_response = slm.invoke(intent_prompt).content.strip().lower()
    
    # Extract category name using regex (in case LLM outputs extra text)
    match = re.search(r"(ingredient discovery|trending insights|historical context|comparative analysis|menu innovation)", llm_response)
    extracted_intent = match.group(1) if match else "default"
    
    # Mapping extracted intent to predefined categories
    intent_mapping = {
        "ingredient discovery": "ingredient_discovery",
        "trending insights": "trending_insights",
        "historical context": "historical_context",
        "comparative analysis": "comparative_analysis",
        "menu innovation": "menu_innovation"
    }
    
    state["intent"] = intent_mapping.get(extracted_intent, "default")
    print("\n🔍 Detected Intent:", state["intent"])
    return state
