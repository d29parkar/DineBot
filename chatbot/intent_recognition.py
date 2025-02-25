import re
from chatbot.config import llm
from chatbot.state import State # ✅ Use correct `State` definition

INTENT_CATEGORIES = {
    "ingredient_discovery": ["which restaurants serve", "dishes with", "gluten-free", "vegan"],
    "trending_insights": ["latest trends", "popular dishes"],
    "historical_context": ["history of", "origin of"],
    "comparative_analysis": ["compare prices", "cost of"],
    "menu_innovation": ["new trends", "how has * changed"]
}

def detect_intent(state: State) -> State:
    """Identifies user intent based on predefined categories and updates state."""
    user_input = state["input"]
    for intent, patterns in INTENT_CATEGORIES.items():
        for pattern in patterns:
            if pattern.lower() in user_input.lower():
                state["intent"] = intent
                return state
    state["intent"] = "default"
    return state
