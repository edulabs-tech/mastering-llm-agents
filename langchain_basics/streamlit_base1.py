from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

import streamlit as st

# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

from langchain_ollama import OllamaLLM

# Load your local model
llm = OllamaLLM(model="gemma3:1b")

# --- Streamlit UI ---
st.title("Minimal Movie Review Analyzer")

# Get movie review input
review_input = st.text_area("Enter your movie review:")

# Button to trigger the model call
if st.button("Analyze Review"):
    if review_input:
        with st.spinner("Analyzing..."):
            try:
                response = llm.invoke(
                    f"{review_input}"
                )

                # Print the returned content response
                st.subheader("Analysis:")
                st.write(response)
                # st.write(response.content)

            except Exception as e:
                st.error(f"An error occurred while calling the API: {e}")
    else:
        st.warning("Please enter a review first.")


