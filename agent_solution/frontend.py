import streamlit as st
from react_agent import invoke_llm
# --- Streamlit App ---

st.set_page_config(page_title="Tavily + Exchange rate Agent", layout="centered")

# --- Sidebar for Configuration ---
# --- Sidebar for Configuration ---
with st.sidebar:
    st.title("⚙️ Configuration")
    # Input box for the user to enter a thread_id
    # We can provide a default value and store the input in session_state
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = "aaa"

    # Store the current thread_id to detect if it changes
    previous_thread_id = st.session_state.thread_id

    st.session_state.thread_id = st.text_input(
        "Thread ID",
        value=st.session_state.thread_id
    )
    st.info("Enter a Thread ID to manage separate conversation histories.")

    # If the thread_id has been changed, clear the message history
    if st.session_state.thread_id != previous_thread_id:
        st.session_state.messages = []
        # Rerun the app to immediately reflect the cleared messages
        st.rerun()

st.title("Tavily + Exchange rate Agent")

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
        response = invoke_llm(prompt, thread_id=st.session_state.thread_id)
        st.markdown(response)

    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

