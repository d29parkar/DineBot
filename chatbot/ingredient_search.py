import pandas as pd
import json
from chatbot.config import llm
from chatbot.state import State
from langchain.schema.runnable import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

# Load dataset
df = pd.read_csv("menudata_internal_data.csv")

# **Step 1: Define the LLM-Powered Prompt for Entity Extraction**
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

# **Step 2: Define LLM Chain for Entity Extraction**
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
        # Clean common JSON formatting issues
        json_str = response.strip()
        json_str = json_str.replace("```json", "").replace("```", "")

        # Parse JSON response
        entities = json.loads(json_str)

        # Ensure all required keys exist
        required_keys = ["location", "menu_item", "ingredient_name", "menu_category", "price", "rating", "review_count"]
        for key in required_keys:
            if key not in entities:
                entities[key] = []  # Default to empty list if missing

        return entities

    except json.JSONDecodeError:
        return None  # Return None if JSON parsing fails


def aggregate_restaurant_data(df):
    """
    Aggregates restaurant data into a single row per restaurant.
    Merges duplicate menu items while collecting ingredients and other relevant details.

    Parameters:
        df (pd.DataFrame): The raw DataFrame containing restaurant menu data.

    Returns:
        pd.DataFrame: A DataFrame with one row per restaurant, aggregating all menu details.
    """
    # Drop unnecessary columns
    df = df.drop(columns=["item_id", "confidence"], errors="ignore")

    # First aggregation: Group by restaurant and menu item
    first_aggregation_rules = {
        "menu_description": "first",  # Retain first occurrence
        "menu_category": "first",  # Retain first occurrence
        "ingredient_name": lambda x: list(set(x.dropna())),  # Collect unique ingredients
        "address1": "first",
        "city": "first",
        "zip_code": "first",
        "country": "first",
        "state": "first",
        "rating": "first",
        "review_count": "first",
        "price": "first"
    }

    df_grouped = df.groupby(["restaurant_name", "menu_item"]).agg(first_aggregation_rules).reset_index()

    # Second aggregation: Group by restaurant name
    final_aggregation_rules = {
        "menu_item": lambda x: list(set(x.dropna())),  # Collect all unique menu items
        "menu_description": lambda x: list(set(x.dropna())),  # Collect unique descriptions
        "ingredient_name": lambda x: list(x),  # Keep lists of ingredients as is (nested list)
        "menu_category": lambda x: list(set(x.dropna())),  # Collect unique categories
        "address1": "first",
        "city": "first",
        "zip_code": "first",
        "country": "first",
        "state": "first",
        "rating": "first",
        "review_count": "first",
        "price": "first"
    }

    final_grouped_df = df_grouped.groupby("restaurant_name").agg(final_aggregation_rules).reset_index()

    return final_grouped_df

def filter_df(df, entities):
    """
    Filters the DataFrame based on extracted entities from the user query.
    
    Parameters:
        df (pd.DataFrame): The dataset containing restaurant and menu information.
        entities (dict): Extracted entities from the LLM in a structured format.

    Returns:
        pd.DataFrame: A filtered DataFrame based on user query parameters.
    """
    
    # Retrieve entity values for filtering
    location_values = entities.get("location", [])
    menu_item_values = entities.get("menu_item", [])
    ingredient_values = entities.get("ingredient_name", [])
    category_values = entities.get("menu_category", [])
    price_values = entities.get("price", [])
    rating_values = entities.get("rating", [])
    review_values = entities.get("review_count", [])

    df = df.apply(lambda x: x.astype(str).str.lower().str.strip() if x.dtype == "object" else x)


    # Apply filters efficiently
    if location_values:
        df = df[df["city"].str.lower().str.strip().isin([loc.lower().strip() for loc in location_values])]
        #print("Filtered Locations:", df["city"].unique())

    if menu_item_values:
        columns_to_search = ["ingredient_name", "menu_item", "menu_description", "menu_category"]

        df = df[df[columns_to_search].apply(
            lambda row: any(row.str.contains('|'.join(menu_item_values), case=False, na=False)), axis=1
        )]
        #print("\n\n\n")
        #print("Filtered Menu Items:", df["menu_item"].unique())

    # Ensure that at least one entity exists before filtering
    if ingredient_values:
        columns_to_search = ["ingredient_name", "menu_item", "menu_description", "menu_category"]

        df = df[df[columns_to_search].apply(
            lambda row: any(row.str.contains('|'.join(ingredient_values), case=False, na=False)), axis=1
        )]
        #print("\n\n\n")
        #print("Filtered Ingredients:", df)

    # if price_values:
    #     df = df[df["price"].astype(str).str.contains('|'.join(price_values), case=False, na=False)]

    # if rating_values:
    #     df = df[df["rating"].astype(str).str.contains('|'.join(rating_values), case=False, na=False)]

    # if review_values:
    #     df = df[df["review_count"].astype(str).str.contains('|'.join(review_values), case=False, na=False)]
    #print('\n\n\n')
    #print("Filtered DataFrame Shape:", df.shape)

    return df


def query_database(state: State) -> State:
    """
    Queries the restaurant dataset based on user input by extracting structured entities via LLM.

    Workflow:
    1. Extract entities from user input using the LLM.
    2. Validate the extracted entities to ensure JSON integrity.
    3. Apply structured filtering to the dataset using `filter_df()`.
    4. Update the state with the structured query results.

    Parameters:
        state (State): The chatbot state containing user input.

    Returns:
        State: Updated chatbot state with structured results.
    """
    
    user_input = state["input"]
    #print("User Input: ", user_input)

    # Step 1: Extract Entities Using LLM
    llm_response = extract_entities_chain.invoke({"input": user_input})
    print("LLM Response: ", llm_response.content)
    
    # Step 2: Validate JSON Response
    entities = validate_json(llm_response.content)
    print("Validated Entities: ", entities)

    if not entities:
        state["structured_results"] = []  # Return empty results if JSON is invalid
        return state

    # Step 3: Apply Optimized Filtering
    filtered_df = filter_df(df, entities)
    #print("Structured Query Results: ", filtered_df.to_dict(orient="records"))
    
    # Step 4: Group results by restaurant
    grouped_df = aggregate_restaurant_data(filtered_df)

    #print("Grouped Results: ", grouped_df.to_dict(orient="records"))

    # Step 5: Store the results in state
    state["structured_results"] = grouped_df.to_dict(orient="records")

    return state
