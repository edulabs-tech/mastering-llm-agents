import streamlit as st
from income_tax_tool_basic import invoke_llm
# --- Streamlit App ---

st.set_page_config(page_title="Tax Chatbot", layout="centered")

st.title("🏦 Tax Chatbot")

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

        # invoke
        response = invoke_llm(prompt, st.session_state.messages)
        st.markdown(response)

    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

