# import os
# import pandas as pd
# import faiss
# import numpy as np
# from dotenv import load_dotenv
# from langchain_core.documents import Document
# import pickle
# from langchain.embeddings import HuggingFaceEmbeddings
# from tqdm import tqdm

# # Load environment variables (if needed)
# load_dotenv()

# # Load restaurant dataset
# data_path = "menudata_internal_data.csv"
# df = pd.read_csv(data_path)

# # Load embedding model (HuggingFaceEmbeddings from Langchain)
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# def generate_embedding(text):
#     """Generate embeddings using HuggingFaceEmbeddings"""
#     return embedding_model.embed_query(text)

# # Convert dataset rows into documents and embeddings
# documents = []
# embeddings = []
# metadata_list = []

# for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
#     text = f"{row['restaurant_name']} - {row['menu_category']} - {row['menu_item']} - {row['menu_description']} - Ingredients: {row['ingredient_name']}."
#     embedding_vector = generate_embedding(text)
    
#     metadata = {
#         "restaurant_name": row["restaurant_name"],
#         "menu_category": row["menu_category"],
#         "menu_item": row["menu_item"],
#         "menu_description": row["menu_description"],
#         "ingredient_name": row["ingredient_name"],
#         "confidence": row["confidence"],
#         "address": f"{row['address1']}, {row['city']}, {row['state']}, {row['country']} - {row['zip_code']}",
#         "rating": row["rating"],
#         "review_count": row["review_count"],
#         "price": row["price"]
#     }
    
#     doc = Document(page_content=text, metadata=metadata)
#     documents.append(doc)
#     embeddings.append(embedding_vector)
#     metadata_list.append(metadata)

# # Convert embeddings to numpy array
# embedding_dim = len(embeddings[0])  # Get embedding dimension
# embedding_matrix = np.array(embeddings, dtype=np.float32)

# # Create FAISS index
# faiss_index = faiss.IndexFlatL2(embedding_dim)
# faiss_index.add(embedding_matrix)

# # Save FAISS index and metadata
# faiss.write_index(faiss_index, "faiss_index.bin")
# with open("metadata.pkl", "wb") as f:
#     pickle.dump(metadata_list, f)

# print("Internal dataset stored in FAISS!")


import os
import pandas as pd
import faiss
import numpy as np
from langchain_core.documents import Document
import pickle
from langchain.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

# Load cleaned restaurant dataset
data_path = "cleaned_menu_data.csv"
df = pd.read_csv(data_path)

# Load HuggingFace embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def generate_embedding(text):
    """Generate embeddings using HuggingFaceEmbeddings"""
    return embedding_model.embed_query(text)

# Group by restaurant and menu_category
grouped_data = df.groupby(['restaurant_name', 'menu_category']).agg(list).reset_index()

documents = []
embeddings = []
metadata_list = []

# Generate structured embeddings for each restaurant-category
for _, row in tqdm(grouped_data.iterrows(), total=len(grouped_data), desc="Processing Categories"):

    restaurant = row['restaurant_name']
    category = row['menu_category']
    
    # Create structured format for embedding
    structured_text = f"Restaurant: {restaurant}\nMenu Category: {category}\nItems:\n"
    
    for item, desc, ingredients in zip(row['menu_item'], row['menu_description'], row['ingredient_name']):
        structured_text += f"  - {item}\n    - Description: {desc}\n    - Ingredients: {ingredients}\n"

    # Generate embedding
    embedding_vector = generate_embedding(structured_text)

    # Metadata
    metadata = {
        "restaurant_name": restaurant,
        "menu_category": category,
        "menu_items": row["menu_item"],
        "menu_descriptions": row["menu_description"],
        "ingredients": row["ingredient_name"],
        "categories": row["categories"][0],  # Take the first as representative
        "address": f"{row['address1'][0]}, {row['city'][0]}, {row['state'][0]}, {row['country'][0]} - {row['zip_code'][0]}",
        "rating": row["rating"][0],
        "review_count": row["review_count"][0],
        "price": row["price"][0]
    }
    
    doc = Document(page_content=structured_text, metadata=metadata)
    documents.append(doc)
    embeddings.append(embedding_vector)
    metadata_list.append(metadata)

# Convert embeddings to numpy array
embedding_dim = len(embeddings[0])  # Get embedding dimension
embedding_matrix = np.array(embeddings, dtype=np.float32)

# Create FAISS index
faiss_index = faiss.IndexFlatL2(embedding_dim)
faiss_index.add(embedding_matrix)

# Save FAISS index and metadata
faiss.write_index(faiss_index, "faiss_index_2.bin")
with open("metadata_2.pkl", "wb") as f:
    pickle.dump(metadata_list, f)

print("Optimized FAISS index stored successfully!")