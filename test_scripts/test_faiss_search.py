import faiss
import pickle
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from chatbot.state import State  # Ensure your chatbot state implementation is accessible

# Load FAISS index and metadata
faiss_index = faiss.read_index("faiss_index_2.bin")
with open("metadata_2.pkl", "rb") as f:
    metadata_list = pickle.load(f)

# Load embedding model for FAISS search
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load cross-encoder model for reranking
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def search_faiss(state: State) -> State:
    """Performs FAISS-based similarity search and applies reranking."""
    
    # Generate query embedding
    query_vector = np.array([embedding_model.embed_query(state["input"])], dtype=np.float32)
    
    # Search FAISS index for nearest neighbors
    distances, indices = faiss_index.search(query_vector, k=10)  # Retrieve top 10 candidates

    # Retrieve metadata for top results
    retrieved_docs = [metadata_list[idx] for idx in indices[0] if idx != -1]

    # Apply cross-encoder reranking
    query_text = state["input"]
    reranker_inputs = [(query_text, doc["restaurant_name"] + " " + doc["menu_category"]) for doc in retrieved_docs]
    
    reranker_scores = reranker.predict(reranker_inputs)

    # Sort retrieved docs by reranker score (higher is better)
    sorted_results = sorted(zip(retrieved_docs, reranker_scores), key=lambda x: x[1], reverse=True)

    # Keep top 5 reranked results
    state["faiss_results"] = [doc for doc, _ in sorted_results[:5]]

    return state


# **TESTING THE FAISS SEARCH FUNCTION**

def run_test_query(query: str):
    """Runs a test query and prints retrieved results before and after reranking."""
    
    # Create initial state
    state = {"input": query}

    # Perform FAISS search with reranking
    updated_state = search_faiss(state)

    # Retrieve final results
    results = updated_state["faiss_results"]

    print("\n===== TEST QUERY =====")
    print(f"Query: {query}\n")

    print("🔍 **Retrieved Results (Top 5 After Reranking):**\n")
    for idx, doc in enumerate(results):
        print(f"{idx+1}. Restaurant: {doc['restaurant_name']} | Category: {doc['menu_category']}")
        print(f"   - Address: {doc['address']}")
        print(f"   - Rating: {doc['rating']} ⭐ | Reviews: {doc['review_count']} reviews")
        print(f"   - Price: {doc['price']}")
        print(f"   - Sample Menu Item: {doc['menu_items'][0]} (if available)\n")

    print("✅ Test completed!\n")


# **RUN TEST QUERIES**
if __name__ == "__main__":
    test_queries = [
        "Best sushi in New York",  
        "Healthy vegan salads",  
        "Mexican tacos with spicy sauce",
        "Italian pasta with cheese",
        "Budget-friendly Indian restaurants in Chicago"
    ]

    for test_query in test_queries:
        run_test_query(test_query)
