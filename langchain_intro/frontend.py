import streamlit as st
from backend import stream_llm


# --- Streamlit App ---

st.set_page_config(page_title="Bank Chatbot", layout="centered")

st.title("🏦 Bank Chatbot")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Configuration")
    model = st.selectbox("Select model:", ["Gemini", "Open AI"])
    language = st.selectbox("Select language:", ["English", "Hebrew"])

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Chat Interface ---

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages from the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input from the chat input box
if prompt := st.chat_input("What is your question?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response and display it
    with st.chat_message("assistant"):

        # stream
        response = st.write_stream(stream_llm(prompt, st.session_state.messages, language=language, model_type=model))


    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

