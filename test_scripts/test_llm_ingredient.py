import pandas as pd
import json
from chatbot.config import llm
from chatbot.state import State
from langchain.schema.runnable import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

# Load dataset
df = pd.read_csv("menudata_internal_data.csv")

# **Step 1: Define the LLM-Powered Prompt (Ensuring JSON Response)**
entity_extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """Extract relevant entities from the user query and return them in a structured JSON format.

    **Output Format (MUST BE VALID JSON)**
    ```
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
    ```
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

# **Step 2: Define LLM Chain**
extract_entities_chain = entity_extraction_prompt | RunnableLambda(llm.invoke)

def validate_json(response):
    """Ensures valid JSON response from LLM."""
    try:
        # **Fix common JSON issues**
        json_str = response.strip()  # Remove whitespace
        json_str = json_str.replace("```json", "").replace("```", "")  # Remove Markdown formatting

        # **Attempt JSON parsing**
        entities = json.loads(json_str)

        # **Ensure all required keys exist**
        required_keys = ["location", "menu_item", "ingredient_name", "menu_category", "price", "rating", "review_count"]
        for key in required_keys:
            if key not in entities:
                entities[key] = []  # Ensure missing keys default to empty list

        return entities

    except json.JSONDecodeError:
        print("❌ Error: Invalid JSON response from LLM")
        return None

def filter_df(df, entities):
    """Filters the DataFrame based on extracted entities."""
    
    # **Ensure entity values exist before filtering**
    location_values = entities.get("location", [])
    menu_item_values = entities.get("menu_item", [])
    ingredient_values = entities.get("ingredient_name", [])
    category_values = entities.get("menu_category", [])
    price_values = entities.get("price", [])
    rating_values = entities.get("rating", [])
    review_values = entities.get("review_count", [])

    print("\n🔍 Filtering DataFrame with Entities:")
    print("📍 Locations:", location_values)
    print("🍽️ Menu Items:", menu_item_values)
    print("🧂 Ingredients:", ingredient_values)
    print("📖 Categories:", category_values)
    print("💲 Price Ranges:", price_values)
    print("⭐ Ratings:", rating_values)
    print("📊 Review Counts:", review_values)
    # **Apply Filters Efficiently**
    if location_values:
        df = df[df["city"].str.lower().str.strip().isin([loc.lower().strip() for loc in location_values])]
        print("📍 Filtered Locations:", df["city"].unique())

    if menu_item_values:
        df = df[df["menu_item"].str.contains('|'.join(menu_item_values), case=False, na=False)]
        print("🍽️ Filtered Menu Items:", df["menu_item"].unique())

    if ingredient_values:
        df = df[df["ingredient_name"].str.contains('|'.join(ingredient_values), case=False, na=False)]
        print("🧂 Filtered Ingredients:", df["ingredient_name"].unique())

    if category_values:
        df = df[df["menu_category"].str.contains('|'.join(category_values), case=False, na=False)]
        print("📖 Filtered Categories:", df["menu_category"].unique())

    if price_values:
        df = df[df["price"].astype(str).str.contains('|'.join(price_values), case=False, na=False)]
        print("💲 Filtered Price Ranges:", df["price"].unique())

    if rating_values:
        df = df[df["rating"].astype(str).str.contains('|'.join(rating_values), case=False, na=False)]
        print("⭐ Filtered Ratings:", df["rating"].unique())

    if review_values:
        df = df[df["review_count"].astype(str).str.contains('|'.join(review_values), case=False, na=False)]
        print("📊 Filtered Review Counts:", df["review_count"].unique())

    return df


def query_database(state: State) -> State:
    """Uses LLM to extract entities, then queries the dataset."""
    
    user_input = state["input"]

    # **Step 1: Extract Entities Using LLM**
    llm_response = extract_entities_chain.invoke({"input": user_input})
    
    # **Step 2: Validate JSON Response**
    entities = validate_json(llm_response.content)

    if not entities:
        print("❌ LLM returned invalid JSON. No results found.")
        state["structured_results"] = []
        return state

    print("\n✅ Valid Entities Extracted:", entities)

    # **Step 3: Apply Optimized Filtering**
    filtered_df = filter_df(df, entities)
    

    state["structured_results"] = filtered_df.to_dict(orient="records")
    return state

    # if not filtered_df.empty:
    #     state["structured_results"] = filtered_df.to_dict(orient="records")
    #     return state
    
    # # **Step 4: If SQL Fails, Run FAISS Vector Search**
    # query_text = f"{' '.join(entities.get('menu_item', []))} {' '.join(entities.get('ingredient_name', []))}"
    # faiss_results = search_faiss(query_text)

    # state["structured_results"] = faiss_results  # Return FAISS matches if no SQL results found
    # return state



def test(user_input):
    # **Step 1: Extract Entities Using LLM**
    llm_response = extract_entities_chain.invoke({"input": user_input})
    
    # **Step 2: Validate JSON Response**
    entities = validate_json(llm_response.content)

    if not entities:
        print("❌ LLM returned invalid JSON. No results found.")
        return []

    print("\n✅ Valid Entities Extracted:", entities)

    # **Step 3: Apply Optimized Filtering**
    filtered_df = filter_df(df, entities)
    #print(filtered_df)
    print(filtered_df.to_dict(orient="records"))

    return filtered_df.to_dict(orient="records")

if __name__ == "__main__":
    print("\n🔹 Testing Ingredient Extraction and Structured Search..." )
    print("\n🚀 Testing Input: 'Which restaurants serve gluten-free pizza in SF?'")
    results = test("Which restaurants serve impossible burger in SF?")
    print("\n✅ Structured Search Results:")
    print(results)  # Print the structured search results
