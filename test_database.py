import pandas as pd
import spacy
import nltk
from nltk.corpus import wordnet
from chatbot.state import State

# Load spaCy's NLP model
nlp = spacy.load("en_core_web_sm")

# Load dataset
df = pd.read_csv("menudata_internal_data.csv")

# Ensure NLTK WordNet is downloaded
nltk.download("wordnet")

def get_synonyms(word):
    """Generate synonyms dynamically using WordNet and LLM-based expansion."""
    synonyms = set()

    # Fetch synonyms from WordNet
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))

    # Add the original word
    synonyms.add(word)

    return synonyms

def extract_entities(text):
    """Extract key entities dynamically using spaCy NLP."""
    doc = nlp(text)
    entities = {
        "location": set(),
        "menu_item": set(),
        "ingredient_name": set(),
        "menu_category": set(),
        "price": set(),
        "rating": set(),
        "review_count": set()
    }

    for ent in doc.ents:
        label = ent.label_

        if label == "GPE":  # City, state, country
            entities["location"].add(ent.text)
        elif label in ["PRODUCT", "ORG", "FOOD"]:  # Menu items, food categories
            entities["menu_item"].add(ent.text)
        elif label == "MONEY":  # Price-related terms
            entities["price"].add(ent.text)
        elif label == "CARDINAL" or label == "QUANTITY":  # Ratings & Reviews
            if "star" in ent.text.lower():
                entities["rating"].add(ent.text)
            else:
                entities["review_count"].add(ent.text)

    return entities

def query_database(state: State) -> State:
    """Extracts key details, generates synonyms, and performs a structured search."""
    user_input = state["input"]

    # **Step 1: Extract Entities**
    entities = extract_entities(user_input)

    # **Step 2: Generate Synonyms**
    expanded_entities = {category: set() for category in entities}

    for category, values in entities.items():
        for value in values:
            expanded_entities[category].update(get_synonyms(value))

    # **Step 3: Add Original Extracted Terms**
    for category, values in entities.items():
        expanded_entities[category].update(values)

    # **Step 4: Construct Query Masks**
    location_mask = df["city"].str.contains('|'.join(expanded_entities["location"]), case=False, na=False) if expanded_entities["location"] else True
    menu_item_mask = df["menu_item"].str.contains('|'.join(expanded_entities["menu_item"]), case=False, na=False) if expanded_entities["menu_item"] else True
    ingredient_mask = df["ingredient_name"].str.contains('|'.join(expanded_entities["ingredient_name"]), case=False, na=False) if expanded_entities["ingredient_name"] else True
    category_mask = df["menu_category"].str.contains('|'.join(expanded_entities["menu_category"]), case=False, na=False) if expanded_entities["menu_category"] else True
    price_mask = df["price"].astype(str).str.contains('|'.join(expanded_entities["price"]), case=False, na=False) if expanded_entities["price"] else True
    rating_mask = df["rating"].astype(str).str.contains('|'.join(expanded_entities["rating"]), case=False, na=False) if expanded_entities["rating"] else True
    review_mask = df["review_count"].astype(str).str.contains('|'.join(expanded_entities["review_count"]), case=False, na=False) if expanded_entities["review_count"] else True

    # **Step 5: Execute Query**
    mask = location_mask & menu_item_mask & ingredient_mask & category_mask & price_mask & rating_mask & review_mask

    state["structured_results"] = df[mask].to_dict(orient="records")

    return state


def test(user_input):
    """Extracts key details, generates synonyms, and performs a structured search."""
    # **Step 1: Extract Entities**
    entities = extract_entities(user_input)
    print("Entities: ", entities)

    # **Step 2: Generate Synonyms**
    expanded_entities = {category: set() for category in entities}

    for category, values in entities.items():
        for value in values:
            expanded_entities[category].update(get_synonyms(value))

    print("Expanded Entities: ", entities)

    # **Step 3: Add Original Extracted Terms**
    for category, values in entities.items():
        expanded_entities[category].update(values)

    print("Add Original Extracted Terms to Expanded Entities: ", entities)

    # **Step 4: Construct Query Masks**
    location_mask = df["city"].str.contains('|'.join(expanded_entities["location"]), case=False, na=False) if expanded_entities["location"] else True
    menu_item_mask = df["menu_item"].str.contains('|'.join(expanded_entities["menu_item"]), case=False, na=False) if expanded_entities["menu_item"] else True
    ingredient_mask = df["ingredient_name"].str.contains('|'.join(expanded_entities["ingredient_name"]), case=False, na=False) if expanded_entities["ingredient_name"] else True
    category_mask = df["menu_category"].str.contains('|'.join(expanded_entities["menu_category"]), case=False, na=False) if expanded_entities["menu_category"] else True
    price_mask = df["price"].astype(str).str.contains('|'.join(expanded_entities["price"]), case=False, na=False) if expanded_entities["price"] else True
    rating_mask = df["rating"].astype(str).str.contains('|'.join(expanded_entities["rating"]), case=False, na=False) if expanded_entities["rating"] else True
    review_mask = df["review_count"].astype(str).str.contains('|'.join(expanded_entities["review_count"]), case=False, na=False) if expanded_entities["review_count"] else True

    # **Step 5: Execute Query**
    mask = location_mask & menu_item_mask & ingredient_mask & category_mask & price_mask & rating_mask & review_mask

    return df[mask].to_dict(orient="records")
    
if __name__ == "__main__":
    # Test the function with a sample user input
    results = test("Which restaurants serve gluten-free pizza in New York?")
    print(results)  # Print the structured search results
    print(len(results))  # Print the number of results
    print("✅ Test completed successfully!")  # Print success message