import streamlit as st

from backend import call_llm

st.title("🤖 Bank Hapoalim Assistant")

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    
    # Dropdown for language
    language = st.selectbox(
        "Select language:",
        ("English", "Hebrew", "Russian")
    )

    # Input to insert assistant's name
    assistant_name = st.text_input("Enter assistant's name:", "Moshe")

st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What can I help you with?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- This is the "fix" ---
    # Get the response from your backend function
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = call_llm(
                user_input=prompt,
                assistant_name=assistant_name,
                language=language,
                history=st.session_state.messages
            )
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})