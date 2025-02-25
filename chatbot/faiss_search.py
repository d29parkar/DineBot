import faiss
import pickle
import numpy as np
from chatbot.state import State  # ✅ Use the correct state structure
from langchain_community.embeddings import HuggingFaceEmbeddings


faiss_index = faiss.read_index("faiss_index.bin")
with open("metadata.pkl", "rb") as f:
    metadata_list = pickle.load(f)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def search_faiss(state: State) -> State:
    """Performs FAISS-based similarity search and updates state."""
    query_vector = np.array([embedding_model.embed_query(state["input"])], dtype=np.float32)
    distances, indices = faiss_index.search(query_vector, k=5)

    state["faiss_results"] = [metadata_list[idx] for idx in indices[0] if idx != -1]
    return state
