import os
import pandas as pd
import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
import pickle
from langchain.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

# Load environment variables (if needed)
load_dotenv()

# Load restaurant dataset
data_path = "menudata_internal_data.csv"
df = pd.read_csv(data_path)

# Load embedding model (HuggingFaceEmbeddings from Langchain)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def generate_embedding(text):
    """Generate embeddings using HuggingFaceEmbeddings"""
    return embedding_model.embed_query(text)

# Convert dataset rows into documents and embeddings
documents = []
embeddings = []
metadata_list = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    text = f"{row['restaurant_name']} - {row['menu_category']} - {row['menu_item']} - {row['menu_description']} - Ingredients: {row['ingredient_name']}."
    embedding_vector = generate_embedding(text)
    
    metadata = {
        "restaurant_name": row["restaurant_name"],
        "menu_category": row["menu_category"],
        "menu_item": row["menu_item"],
        "menu_description": row["menu_description"],
        "ingredient_name": row["ingredient_name"],
        "confidence": row["confidence"],
        "address": f"{row['address1']}, {row['city']}, {row['state']}, {row['country']} - {row['zip_code']}",
        "rating": row["rating"],
        "review_count": row["review_count"],
        "price": row["price"]
    }
    
    doc = Document(page_content=text, metadata=metadata)
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
faiss.write_index(faiss_index, "faiss_index.bin")
with open("metadata.pkl", "wb") as f:
    pickle.dump(metadata_list, f)

print("Internal dataset stored in FAISS!")
