from typing import List

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage


# Load environment variables from .env file
load_dotenv()

system_template = """
    You are a friendly customer assistant at Bank Hapoalim.
    All your answers should be in {language}.
    Be friendly, greet the customer, and answer their questions related to bank operations.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{text}")
])

open_ai_model = ChatOpenAI(model="gpt-4o-mini")
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# llama

parser = StrOutputParser()

open_ai_chain = prompt_template | open_ai_model | parser
gemini_chain = prompt_template | gemini_model | parser


def pretty_print_messages(messages: List[BaseMessage]):
    for msg in messages:
        msg.pretty_print()


def stream_llm(text, history, language, model_type="Gemini"):
    chain = open_ai_chain if model_type == "Open AI" else gemini_chain
    for chunk in chain.stream(
            {
                "language": language, 
                "text": text, 
                "history": history
            },
            stream_mode="messages"):
        yield chunk




