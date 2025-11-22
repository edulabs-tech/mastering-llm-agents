from pprint import pprint

from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Add memory to the process
memory = MemorySaver()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")


songs_db = SQLDatabase.from_uri("/Users/valeria/src/mastering-llm-agents/docs/Chinook.db")
songs_toolkit = SQLDatabaseToolkit(db=songs_db, llm=llm)
songs_db_tools = songs_toolkit.get_tools()

movies_db = SQLDatabase.from_uri("/Users/valeria/src/mastering-llm-agents/docs/netflixdb.sqlite")
movies_toolkit = SQLDatabaseToolkit(db=movies_db, llm=llm)
movies_db_tools = movies_toolkit.get_tools()


# Nodes

SQL_PROMPT = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database."""

def songs_node(state: MessagesState):
    # response = llm.bind(songs_db_tools).invoke()
    return {"messages": [response]} 