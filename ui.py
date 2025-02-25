import streamlit as st
from chatbot.get_response import get_response  # ✅ Ensure proper import

def setup_ui():
    """Initialize Streamlit UI"""
    st.set_page_config(page_title="Restaurant AI Chatbot", page_icon="🍽️🤖", layout="centered")
    st.title("🍽️🤖 Restaurant AI Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, message in st.session_state.chat_history:
        with st.chat_message("user" if role == "User" else "assistant"):
            st.markdown(message)

    return st.chat_input("Type your message here...")

def handle_chat(user_query):
    """Processes user input, updates chat history, and displays response"""
    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)

        ai_response = get_response(user_query)  # ✅ Calls function from chatbot.get_response

        st.session_state.chat_history.append(("User", user_query))
        st.session_state.chat_history.append(("AI", ai_response))

        with st.chat_message("assistant"):
            st.markdown(ai_response)

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
