import os
from pprint import pprint

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from openai import api_key

load_dotenv()


model = ChatOpenAI(model="gpt-4o-mini")
# model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")



response = model.invoke("Hi")
print(response)
response.pretty_print()