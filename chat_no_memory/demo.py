# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# create model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

SYSTEM_TEMPLATE = """
    You are a friendly customer assistant at Bank Hapoalim.
    Your name is {assistant_name}.
    All your answers should be in {language}.
    Be friendly, greet the customer, present yourself, and answer their questions related to bank operations.
"""

template = ChatPromptTemplate(
    [
        ("system", SYSTEM_TEMPLATE),
        ("human", "{user_input}"),
    ]
)

prompt_value = template.invoke(
    {
        "assistant_name": "David",
        "language": "Hebrew",
        "user_input": "Hi, what can you do?",
    }
)
print(f"Prompt value:\n{prompt_value}")

# "Manual" chaining
response = llm.invoke(prompt_value)
print(f"LLM response:\n{response}")


# Building langchain chain
chain = template | llm | StrOutputParser()
response = chain.invoke(
    {
        "assistant_name": "David",
        "language": "Hebrew",
        "user_input": "Hi, what can you do?",
    }
)
print(f"Response:\n{response}")
