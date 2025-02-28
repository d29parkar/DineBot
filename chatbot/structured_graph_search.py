import json
from chatbot.config import llm, slm  # Use smaller LLM for query validation
from chatbot.state import State
from py2neo import Graph
from langchain.schema.runnable import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

# **🔗 Connect to Neo4j Database**
graph = Graph("neo4j+s://fdd1303c.databases.neo4j.io", auth=("neo4j", "1f8bgEco73so8nVug9mfFTlEjNfatT8cnOE_Ee8hDKc"))

QUERY_DICTIONARY = {
    "ingredient_discovery": {
        # 🌍 Find restaurants serving a specific cuisine in given cities
        "restaurant_search_based_on_cuisine_in_cities": """
            MATCH (r:Restaurant)-[:SERVES]->(m:MenuCategory)
            WHERE toLower(m.name) IN {menu_categories} AND toLower(r.city) IN {locations}
            RETURN r.name AS restaurant, r.city AS city, collect(m.name) AS cuisine 
            ORDER BY r.rating DESC
            LIMIT 15
        """,

        # 🔎 Find restaurants that have dishes containing a specific ingredient
        "restaurant_search_based_on_ingredient": """
            MATCH (r:Restaurant)-[:SERVES]->(m:MenuCategory)-[:HAS_ITEM]->(mi:MenuItem)
            OPTIONAL MATCH (mi)-[:CONTAINS]->(i:Ingredient)
            WHERE apoc.text.containsAll(toLower(mi.name), {menu_items}) 
                  OR apoc.text.containsAll(toLower(i.name), {ingredients})
                  OR apoc.text.containsAll(toLower(m.name), {menu_categories})
                  OR apoc.text.containsAll(toLower(mi.description), {ingredients})
            RETURN r.name AS restaurant, collect(DISTINCT mi.name) AS matched_items
            ORDER BY r.rating DESC
            LIMIT 15
        """,

        # 📍 Find dishes available in a city
        "dish_search_in_city": """
            MATCH (r:Restaurant)-[:SERVES]->(m:MenuCategory)-[:HAS_ITEM]->(mi:MenuItem)
            WHERE toLower(mi.name) IN {menu_items} AND toLower(r.city) IN {locations}
            RETURN r.name AS restaurant, collect(mi.name) AS dishes 
            ORDER BY r.rating DESC
            LIMIT 15
        """,
    },

    "menu_innovation": {
        # 📈 Find how often an ingredient is used in dishes
        "ingredient_use": """
            MATCH (i:Ingredient)<-[:CONTAINS]-(mi:MenuItem)
            WHERE toLower(i.name) IN {ingredients}
            RETURN i.name AS ingredient, COUNT(mi) AS mentions
            ORDER BY mentions DESC
            LIMIT 15
        """,

        # 🔥 Identify trending ingredients across menus
        "trending_ingredients": """
            MATCH (i:Ingredient)<-[:CONTAINS]-(mi:MenuItem)
            WHERE i.name IS NOT NULL
            RETURN i.name AS ingredient, COUNT(mi) AS usage_count
            ORDER BY usage_count DESC
            LIMIT 15
        """
    },

    "trending_insights": {
        # 🌟 Popular dishes based on the number of menu occurrences
        "popular_dishes": """
            MATCH (r:Restaurant)-[:SERVES]->(m:MenuCategory)-[:HAS_ITEM]->(mi:MenuItem)
            RETURN mi.name AS dish, COUNT(*) AS appearances
            ORDER BY appearances DESC
            LIMIT 15
        """,

        # 📊 Identify trends in ingredient usage
        "ingredient_trends": """
            MATCH (i:Ingredient)<-[:CONTAINS]-(mi:MenuItem)
            WITH i.name AS ingredient, COUNT(mi) AS mentions
            ORDER BY mentions DESC
            RETURN ingredient, mentions
            LIMIT 15
        """,

        # 💰 Compare menu prices for a cuisine across different locations
        "price_comparison": """
            MATCH (r:Restaurant)-[:SERVES]->(m:MenuCategory)-[:HAS_ITEM]->(mi:MenuItem)
            WHERE toLower(m.name) IN {menu_categories} AND toLower(r.city) IN {locations}
            RETURN m.name AS cuisine, r.city AS city, 
                   ROUND(AVG(mi.price), 2) AS avg_price
            ORDER BY avg_price DESC
            LIMIT 10
        """,
        
        # 📍 Compare the popularity of two cuisines across cities
        "cuisine_popularity_comparison": """
            MATCH (r:Restaurant)-[:SERVES]->(m:MenuCategory)
            WHERE toLower(m.name) IN {menu_categories} AND toLower(r.city) IN {locations}
            RETURN m.name AS cuisine, r.city AS city, COUNT(r) AS restaurant_count
            ORDER BY restaurant_count DESC
            LIMIT 10
        """,
    },

    "reviews_analysis": {
        # 🌟 Find top-rated restaurants in a city
        "top_rated_restaurants": """
            MATCH (r:Restaurant)
            WHERE toLower(r.city) IN {locations}
            RETURN r.name AS restaurant, r.rating AS rating
            ORDER BY rating DESC
            LIMIT 10
        """,

        # 🏆 Find the most reviewed restaurants
        "most_reviewed_restaurants": """
            MATCH (r:Restaurant)
            WHERE toLower(r.city) IN {locations}
            RETURN r.name AS restaurant, r.review_count AS reviews
            ORDER BY reviews DESC
            LIMIT 10
        """
    }
}


def select_subcategory(intent, user_query):
    """
    Uses SLM to determine the best subcategory for the intent.
    """
    subcategory_prompt = f"""
    You are an expert in database querying.
    The user intent is **{intent}**.

    Based on the user query: "{user_query}", select the **most appropriate subcategory** to use.
    
    **Available Subcategories:**
    {list(QUERY_DICTIONARY.get(intent, {}).keys())}

    **Rules:**
    - Return ONLY the subcategory name.
    - Do NOT add explanations.

    **Example Output:**
    "restaurant_search_based_on_ingredient"
    """

    response = slm.invoke(subcategory_prompt).content.strip()
    cleaned_response = response.replace('"', '').replace("'", "").strip()
    return response


def construct_query(intent, subcategory, extracted_entities):
    """Constructs the correct Cypher query based on intent and extracted entities."""
    
    # Normalize subcategory key
    subcategory = subcategory.replace('"', '').replace("'", "").strip().lower()
    
    # Debug: Print available subcategories before lookup
    available_subcategories = QUERY_DICTIONARY.get(intent, {}).keys()
    print(f"\n🔍 Available Subcategories for '{intent}':", list(available_subcategories))

    # Retrieve query template
    query_template = QUERY_DICTIONARY.get(intent, {}).get(subcategory, None)
    
    # Debug: Check if query template was found
    if query_template is None:
        print(f"\n❌ ERROR: No query template found for subcategory '{subcategory}' under intent '{intent}'.")
        return None
    else:
        print("\n✅ Selected Query Template:", query_template)

    # Inject extracted entities into the query
    query = query_template.format(
        locations=json.dumps(extracted_entities.get("location", [])),
        menu_items=json.dumps(extracted_entities.get("menu_item", [])),
        ingredients=json.dumps(extracted_entities.get("ingredient_name", [])),
        menu_categories=json.dumps(extracted_entities.get("menu_category", []))
    )

    return query


def verify_query_with_slm(intent, query_options):
    """
    Uses SLM to verify whether a generated Cypher query is correct.
    If incorrect, it suggests a corrected query.
    """
    query_verification_prompt = f"""
    You are an AI expert in database query validation.
    Below is a set of predefined Cypher queries based on the intent **{intent}**.
    
    Verify if any of these queries correctly retrieve the intended information. 
    If one is correct, respond with **"correct query"**.  
    If not, suggest a modified query that better fits the intent.

    **Queries to Verify:**
    ```cypher
    {query_options}
    ```

    **Response Format:**
    - If a query is correct, return: `"correct query"`
    - If a query needs modification, return the modified query.
    """

    response = slm.invoke(query_verification_prompt).content.strip()
    return response

def query_knowledge_graph(state: State) -> State:
    """
    Queries the Neo4j knowledge graph based on user input after selecting the correct subcategory.
    """
    user_input = state["input"]
    intent = state["intent"]

    # Step 1: Extract Entities Using LLM
    entities = state["entities"]
    print("\n🔍 Extracted Entities:")
    print(json.dumps(entities, indent=2))

    if not entities:
        state["graph_results"] = []
        return state

    # Step 2: Select Best Subcategory Using SLM
    selected_subcategory = select_subcategory(intent, user_input)
    print("\n🔍 Selected Subcategory:", selected_subcategory)
    
    print("\n🔍 Selected Subcategory:", selected_subcategory)
    

    # Step 3: Construct the Correct Cypher Query
    cypher_query = construct_query(intent, selected_subcategory, entities)
    print("\n🔍 Generated Cypher Query:")
    print(cypher_query)

    if not cypher_query:
        state["graph_results"] = []
        return state

    # Step 4: Execute the Query in Neo4j
    results = graph.run(cypher_query).data()

    # Step 5: Store Results in State
    state["graph_results"] = results

    print("\n🔍 Graph Query Results:")
    print(json.dumps(results, indent=2))
    return state
