from typing_extensions import TypedDict
from py2neo import Graph
from chatbot.config import llm
from langchain.schema.runnable import RunnableLambda
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Define the state structure
class State(TypedDict):
    input: str
    intent: str
    structured_results: list
    entities: dict
    faiss_results: list
    google_results: list
    graph_results: list
    llm_made_graph_results: list
    response: str

# Connect to Neo4j Database
neo4j_url = os.getenv("NEO4J_URL")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")
graph = Graph(neo4j_url, auth=(neo4j_user, neo4j_password))

# Define LLM Prompt for Generating Cypher Queries
query_generation_prompt = """
You are an expert in writing Cypher queries for Neo4j. Given the user's natural language query, generate the most **efficient, structured, and optimized** Cypher query based on the structured knowledge graph.

---

## **💡 Knowledge Graph Schema:**
- **Restaurant (Node)**  
  - Properties: `name`, `address`, `city`, `zip_code`, `country`, `state`, `rating`, `review_count`, `price`
- **MenuCategory (Node)**
  - Properties: `name`
- **MenuItem (Node)**
  - Properties: `name`, `description`
- **Ingredient (Node)**
  - Properties: `name`
- **Relationships:**
  - `(Restaurant)-[:SERVES]->(MenuCategory)`
  - `(MenuCategory)-[:HAS_ITEM]->(MenuItem)`
  - `(MenuItem)-[:CONTAINS]->(Ingredient)`

---

## **📝 Query Generation Rules**
- **Always use case-insensitive matching** (`toLower()`) when searching for text-based properties.
- **Always use `OPTIONAL MATCH`** when dealing with `Ingredient` nodes, as some `MenuItems` may not contain ingredients.
- **Ensure queries aggregate results properly** to return structured output.
- **Include synonym-based searches** for improved keyword matching (e.g., "gluten-free" → "GF", "pizza" → "flatbread").
- **Optimize queries for performance**, ensuring proper indexing and minimal unnecessary computations.
- **Only return the Cypher query,** do not include any explanations or extra text.
- If the query **cannot be generated**, return `"NO_QUERY"`.

---

## **🔍 Query Types & Example Cypher Queries**

### **1️⃣ Find all menu items from a specific restaurant, ensuring menu categories exist**
```cypher
MATCH (r:Restaurant {name: "20 Spot"})-[:SERVES]->(c:MenuCategory)-[:HAS_ITEM]->(m:MenuItem)
RETURN c.name AS category, m.name AS menu_item, m.description AS description
ORDER BY c.name
```

---

### **2️⃣ Find restaurants that serve a specific menu category and include ratings and menu items**
```cypher
MATCH (r:Restaurant)-[:SERVES]->(c:MenuCategory {name: "dessert"})-[:HAS_ITEM]->(m:MenuItem)
RETURN r.name AS restaurant_name, r.city AS city, r.state AS state, r.rating AS rating, collect(m.name) AS desserts
ORDER BY r.rating DESC
```

---

### **3️⃣ Find the top-rated restaurants serving a specific dish**
```cypher
MATCH (r:Restaurant)-[:SERVES]->(:MenuCategory)-[:HAS_ITEM]->(m:MenuItem)
WHERE toLower(m.name) CONTAINS "pizza"
RETURN r.name AS restaurant_name, r.city AS city, r.rating AS rating
ORDER BY r.rating DESC
LIMIT 10;
```

---

### **4️⃣ Find restaurants, menu items, and ingredients related to a keyword (e.g., "Impossible")**
```cypher
MATCH (r:Restaurant)-[:SERVES]->(:MenuCategory)-[:HAS_ITEM]->(m:MenuItem)
WHERE toLower(m.name) CONTAINS "impossible" OR toLower(m.description) CONTAINS "impossible"
WITH r, collect({type: "menu_item", name: m.name, description: m.description}) AS menu_items

MATCH (r)-[:SERVES]->(:MenuCategory)-[:HAS_ITEM]->(m2:MenuItem)-[:CONTAINS]->(i:Ingredient)
WHERE toLower(i.name) CONTAINS "impossible"
WITH r, menu_items, collect({type: "ingredient", name: i.name}) AS ingredient_items

RETURN r.name AS restaurant_name, r.address AS address, r.city AS city, menu_items + ingredient_items AS combined_items
LIMIT 50;
```
---

### ** Find trending ingredients based on menu occurrences**
```cypher
MATCH (i:Ingredient)<-[:CONTAINS]-(m:MenuItem)
RETURN i.name AS ingredient, COUNT(m) AS mentions
ORDER BY mentions DESC
LIMIT 10;
```
---

## **⚠️ Edge Cases to Handle**
- If a query involves **ingredients**, always use `OPTIONAL MATCH` since some `MenuItem` nodes may not have ingredients.
- If filtering by city, ensure the query uses **case-insensitive filtering (`toLower()`)** to avoid mismatches.
- If filtering by price, **cast `price` to `FLOAT` (`toFloat()`)** to prevent type errors.
- **Leverage synonyms** to ensure broader search coverage.
- **Ensure queries return aggregated results to avoid duplication**.

---

## **🎡 Output Format**
- **Only return the Cypher query, nothing else.**  
- The query should be well-optimized, structured, and formatted properly.  
- If no valid query can be created, return `"NO_QUERY"`.  
"""



# LLM-based Cypher Query Generator
generate_cypher_query = RunnableLambda(lambda state: 
    llm.invoke(f"{query_generation_prompt}\nUser Query: {state}").content.strip())

def query_knowledge_graph(state: State) -> State:
    """
    Queries Neo4j based on dynamically generated Cypher queries from LLM.
    """
    user_input = state["input"]

    # Step 1: LLM generates Cypher query
    cypher_query = generate_cypher_query.invoke(user_input).replace("```cypher", "").replace("```", "").strip()
    print("\n🔍 Generated Cypher Query:", cypher_query)


    # Step 2: Check if a valid query was generated
    if cypher_query == "NO_QUERY":
        state["llm_made_graph_results"] = []
        return state

    # Step 3: Execute the query in Neo4j
    try:
        results = graph.run(cypher_query).data()
        state["llm_made_graph_results"] = results
    except Exception as e:
        print(f"❌ Neo4j Query Failed: {e}")
        state["llm_made_graph_results"] = []

    return state
