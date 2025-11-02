# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
        MessagesPlaceholder(variable_name="history"),
        ("human", "{user_input}"),
    ]
)


# Building langchain chain
chain = template | llm | StrOutputParser()

def call_llm(user_input: str, assistant_name: str, language: str, history: list):
    response = chain.invoke(
        {
            "assistant_name": assistant_name,
            "language": language,
            "user_input": user_input,
            "history": history
        }
    )
    return response

