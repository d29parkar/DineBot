import os
import pickle
import faiss
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from config import llm

# Memory for storing conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define conversational prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that answers questions about restaurants, menus, and food ingredients."),
    ("human", "{input}")
])

# Create an LLM chain with memory
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Load FAISS index and metadata
faiss_index = faiss.read_index("faiss_index.bin")
with open("metadata.pkl", "rb") as f:
    metadata_list = pickle.load(f)

# Load embedding model (HuggingFaceEmbeddings from Langchain)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def generate_embedding(text):
    """Generate embeddings using HuggingFaceEmbeddings"""
    return embedding_model.embed_query(text)

def search_faiss(query):
    """Search FAISS for relevant restaurants"""
    query_vector = np.array([generate_embedding(query)], dtype=np.float32)
    distances, indices = faiss_index.search(query_vector, k=5)
    
    restaurants = []
    for idx in indices[0]:
        if idx != -1:
            restaurants.append(metadata_list[idx])
    return restaurants

def get_response(user_input):
    """Fetch restaurant data + LLM-generated response"""
    restaurant_results = search_faiss(user_input)

    if restaurant_results:
        response_text = "**Here are some matching restaurants and menu items:**\n\n"
        for res in restaurant_results:
            response_text += f"🍽️ **{res['restaurant_name']}**\n"
            response_text += f"📌 **Menu Item:** {res['menu_item']}\n"
            response_text += f"📖 **Description:** {res['menu_description']}\n"
            response_text += f"🧂 **Ingredients:** {res['ingredient_name']}\n"
            response_text += f"⭐ **Rating:** {res['rating']} ({res['review_count']} reviews)\n"
            response_text += f"💲 **Price Range:** {res['price']}\n"
            response_text += f"📍 **Location:** {res['address']}\n"
            response_text += f"🛠️ **Confidence:** {res['confidence']}\n\n"
        return response_text
    else:
        # Default LLM response if no restaurants found
        response = chain.invoke({"input": user_input})
        return response.get("text", "⚠️ No matching restaurant found.")