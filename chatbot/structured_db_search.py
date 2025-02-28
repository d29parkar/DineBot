import pandas as pd
import json
from chatbot.config import llm
from chatbot.state import State
from langchain.schema.runnable import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

# Load dataset
df = pd.read_csv("cleaned_menu_data.csv")

def aggregate_restaurant_data(df):
    """
    Aggregates restaurant data into a single row per restaurant.
    Merges duplicate menu items while collecting ingredients and other relevant details.

    Returns:
        pd.DataFrame: Aggregated DataFrame with one row per restaurant.
    """
    aggregation_rules = {
        "menu_item": lambda x: list(set(x.dropna())),
        "menu_description": lambda x: list(set(x.dropna())),
        "menu_category": lambda x: list(set(x.dropna())),
        "categories": lambda x: list(set(x.dropna())),
        "ingredient_name": lambda x: list(set(x.dropna())),
        "address1": "first",
        "city": "first",
        "zip_code": "first",
        "country": "first",
        "state": "first",
        "rating": "mean",
        "review_count": "sum",
        "price": lambda x: list(set(x.dropna()))
    }

    return df.groupby("restaurant_name").agg(aggregation_rules).reset_index()


def compute_price_comparison(df, category_1, category_2, city):
    """
    Compares the average price of two cuisines in a given city.
    
    Returns:
        dict: Comparison of average prices.
    """
    filtered_df = df[df["city"].str.lower() == city.lower()]
    category_prices = filtered_df.groupby("menu_category")["price"].apply(lambda x: x.astype(str).mode().values)
    
    return {
        category_1: category_prices.get(category_1, "N/A"),
        category_2: category_prices.get(category_2, "N/A")
    }


def filter_df(df, entities, intent):
    """
    Filters the DataFrame based on extracted entities from the user query and intent.
    
    Returns:
        pd.DataFrame or dict: A filtered DataFrame or comparative analytics data.
    """
    
    location_values = entities.get("location", [])
    menu_item_values = entities.get("menu_item", [])
    ingredient_values = entities.get("ingredient_name", [])
    category_values = entities.get("menu_category", [])
    price_values = entities.get("price", [])
    rating_values = entities.get("rating", [])
    review_values = entities.get("review_count", [])

    df = df.apply(lambda x: x.astype(str).str.lower().str.strip() if x.dtype == "object" else x)

    if location_values:
        df = df[df["city"].str.lower().isin([loc.lower() for loc in location_values])]

    if intent == "ingredient_discovery":
        # Ensure menu_item_values is not empty before filtering
        # Step 1: Filter based on menu_item_values (anywhere in the dataframe)
        if menu_item_values:
            search_pattern = '|'.join(menu_item_values)  # Create regex pattern for OR search
            df = df[
                df.apply(lambda row: row.astype(str).str.contains(search_pattern, case=False, na=False).any(), axis=1)
            ]

        # Step 2: Filter the already filtered df based on ingredient_values (anywhere in the dataframe)
        if ingredient_values:
            search_pattern = '|'.join(ingredient_values)  # Create regex pattern
            df = df[
                df.apply(lambda row: row.astype(str).str.contains(search_pattern, case=False, na=False).any(), axis=1)
            ]


        return df
    else:
        return None


def query_database(state: State) -> State:
    """
    Queries the restaurant dataset based on user input by extracting structured entities via LLM.
    Returns:
        State: Updated chatbot state with structured results.
    """
    
    entities = state["entities"]
    intent = state["intent"]
    print("Validated Entities: ", entities)

    if not entities:
        state["structured_results"] = None
        return state

    # Apply optimized filtering based on intent
    filtered_df = filter_df(df, entities, intent)

    if filtered_df is None or filtered_df.empty:
        print("WARNING: No matching results after filtering. Skipping aggregation.")
        state["structured_results"] = None
        return state

    if isinstance(filtered_df, dict):  
        state["structured_results"] = filtered_df if filtered_df else None
    elif isinstance(filtered_df, pd.DataFrame):  
        if "restaurant_name" not in filtered_df.columns:
            print("ERROR: 'restaurant_name' column missing in filtered DataFrame")
            print("DEBUG: Available columns:", filtered_df.columns)
            state["structured_results"] = None
            return state  # ✅ Prevents KeyError

        grouped_df = aggregate_restaurant_data(filtered_df)
        state["structured_results"] = grouped_df.to_dict(orient="records") if not grouped_df.empty else None
    else:
        state["structured_results"] = None  

    return state

