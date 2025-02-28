import json
from chatbot.config import llm
from chatbot.state import State
from langchain.schema.runnable import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

# **🔍 Step 1: Define LLM-Powered Entity Extraction**
entity_extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """Extract relevant entities from the user query and return them in a structured JSON format.

    **Output Format (MUST BE VALID JSON)**
    ```json
    {{
        "location": ["list of location synonyms"],
        "menu_item": ["list of dish synonyms"],
        "ingredient_name": ["list of ingredient synonyms"],
        "menu_category": ["list of category synonyms"],
        "price": ["list of price terms"],
        "rating": ["list of rating terms"],
        "review_count": ["list of review-related terms"]
    }}
    ```

    **Rules:**
    - The response **MUST always be valid JSON**. Do not include extra text before or after the JSON.
    - Always include **all keys**, even if some are empty (`[]`).
    - Extract **synonyms** and **alternative phrasings** for each entity.

    **Example Query:**  
    _"Which restaurants serve gluten-free pizza in New York?"_

    **Expected JSON Output:**  
    ```json
    {{
        "location": ["New York", "NYC", "Big Apple"],
        "menu_item": ["pizza", "flatbread", "Neapolitan pizza"],
        "ingredient_name": ["gluten-free", "GF", "wheat-free"],
        "menu_category": [],
        "price": [],
        "rating": [],
        "review_count": []
    }}
    ```
    """),
    ("human", "{input}")
])

# **🔍 Step 2: Define LLM Chain for Entity Extraction**
extract_entities_chain = entity_extraction_prompt | RunnableLambda(llm.invoke)

def validate_json(response):
    """
    Ensures the LLM response is a valid JSON and contains all required keys.
    
    Parameters:
        response (str): The JSON response string from the LLM.

    Returns:
        dict: A dictionary containing the structured entities if valid, else None.
    """
    try:
        json_str = response.strip().replace("```json", "").replace("```", "")
        entities = json.loads(json_str)

        required_keys = ["location", "menu_item", "ingredient_name", "menu_category", "price", "rating", "review_count"]
        for key in required_keys:
            if key not in entities:
                entities[key] = []  # Default to empty list if missing

        return entities
    except json.JSONDecodeError:
        return None  # Return None if JSON parsing fails

def extract_entities(state: State) -> State:
    """
    Extracts entities from the user query using LLM and stores them in the state.

    Parameters:
        state (State): The chatbot state containing user input.

    Returns:
        State: Updated chatbot state with extracted entities.
    """
    user_input = state["input"]

    # Step 1: Extract Entities Using LLM
    llm_response = extract_entities_chain.invoke({"input": user_input})
    
    # Step 2: Validate JSON Response
    entities = validate_json(llm_response.content)

    if not entities:
        print("⚠️ Entity extraction failed. Setting empty entities.")
        entities = {key: [] for key in ["location", "menu_item", "ingredient_name", "menu_category", "price", "rating", "review_count"]}

    # Step 3: Store Entities in State
    state["entities"] = entities
    return state
