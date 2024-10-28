import streamlit as st
from rag_backend import get_llm_response

# Streamlit UI configuration
st.set_page_config(page_title="Enhanced Chatbot", page_icon="🧠")  # Update page title and icon
st.title("Enhanced RAG-Enabled Chatbot 🧠")

# Conversation history storage
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Mode selector for RAG or Non-RAG
mode = st.radio("Choose Query Mode:", ["Non-RAG Mode", "RAG Mode"])  # Radio button to switch modes

# Input field for user queries
user_query = st.text_input("Enter your query:")

# Model selector (Llama3 or Mistral)
model_name = st.selectbox("Choose Model:", ["llama3", "mistral"])

# Display conversation history
st.subheader("Conversation History")
for message in st.session_state.conversation_history:
    st.write(message)

# If the user submits a query, call the LLM
if st.button("Send"):
    if user_query:
        # Determine if RAG mode is enabled based on the selected mode
        use_rag = mode == "RAG Mode"
        
        # Call the backend to get the response from the LLM
        response = get_llm_response(model_name, user_query, use_rag=use_rag)
        
        # Store conversation in session state
        st.session_state.conversation_history.append(f"User: {user_query}")
        st.session_state.conversation_history.append(f"Bot: {response}")

        # Display the latest response
        st.write(f"Bot: {response}")

