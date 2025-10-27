import streamlit as st
from backend import invoke_llm, stream_llm, invoke_with_trim
# --- Streamlit App ---

st.set_page_config(page_title="Bank Chatbot", layout="centered")

st.title("🏦 Bank Chatbot")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Configuration")
    model = st.selectbox("Select model:", ["Gemini", "Open AI"])
    language = st.selectbox("Select language:", ["English", "Hebrew"])

# --- Chat Interface (Stateless) ---

# Get user input from the chat input box
if prompt := st.chat_input("What is your question?"):
    # Display the user's message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display assistant response
    with st.chat_message("assistant"):
        # We now only send the current prompt, not the whole history.
        # The message is wrapped in a list to match the expected format.
        current_message = [{"role": "user", "content": prompt}]

        # INVOKING
        # response = invoke_llm(prompt, [], language, model)
        # st.write(response)

        # STREAMING
        response_stream = stream_llm(prompt, [], language, model)
        # st.write_stream displays the streamed response
        st.write_stream(response_stream)