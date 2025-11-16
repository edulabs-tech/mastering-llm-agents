import streamlit as st
from backend_human_in_the_loop import invoke_llm, confirm

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Custom songs SQL agent", layout="centered")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "123"

# Track if we are currently waiting for user confirmation (Interruption state)
if "pending_interruption" not in st.session_state:
    st.session_state.pending_interruption = None

# --- Sidebar for Configuration ---
with st.sidebar:
    st.title("⚙️ Configuration")
    
    previous_thread_id = st.session_state.thread_id

    st.session_state.thread_id = st.text_input(
        "Thread ID",
        value=st.session_state.thread_id
    )
    st.info("Enter a Thread ID to manage separate conversation histories.")

    # If the thread_id has been changed, clear the message history and interrupt state
    if st.session_state.thread_id != previous_thread_id:
        st.session_state.messages = []
        st.session_state.pending_interruption = None
        st.rerun()

st.title("Custom songs SQL agent")

# --- Chat Interface ---

# 1. Display past messages from the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Handle Pending Interruptions (Display Buttons instead of Input)
if st.session_state.pending_interruption:
    
    # Display the interruption message (The query to approve)
    with st.chat_message("assistant"):
        st.warning("⚠️ Human Approval Required")
        st.markdown(st.session_state.pending_interruption)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✅ Confirm", type="primary", use_container_width=True):
                with st.spinner("Running query..."):
                    # Call the confirm backend function
                    response = confirm(thread_id=st.session_state.thread_id)
                    
                    # Append the confirmation result to history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Clear the interruption state
                    st.session_state.pending_interruption = None
                    st.rerun()
        
        with col2:
            if st.button("✏️ Edit", use_container_width=True):
                st.toast("Edit functionality is not implemented yet.", icon="🚧")
                
        with col3:
            if st.button("❌ Deny", use_container_width=True):
                # Just clear the state and let the user type something else
                st.session_state.pending_interruption = None
                st.rerun()

# 3. Standard Chat Input (Only show if NOT waiting for confirmation)
else:
    if prompt := st.chat_input("What is your question?"):
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = invoke_llm(prompt, thread_id=st.session_state.thread_id)

            # Check for Interruption signal
            if response.startswith("INTERRUPTED!"):
                # Store the interruption text in session state
                st.session_state.pending_interruption = response
                # Rerun immediately to hide input and show buttons
                st.rerun()
            else:
                # Normal response
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# Which genre on average has the longest tracks?