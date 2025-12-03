from dotenv import load_dotenv
load_dotenv()

import pprint

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI

# Load your local model
ollama_llm = OllamaLLM(model="gemma3:1b")
openai_llm = ChatOpenAI(model="gpt-4o-mini")
gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Run a simple prompt
response = gemini_llm.invoke("Write a haiku about LLMs running locally.")
pprint.pprint(response)

print("---------------------")

# Run a simple prompt
response = openai_llm.invoke("Write a haiku about LLMs running locally.")
pprint.pprint(response)

print("---------------------")

# Run a simple prompt
response = ollama_llm.invoke("Write a haiku about LLMs running locally.")
pprint.pprint(response)


#
# response = gemini_model.invoke("Tell me  a joke")
#
# print(response)