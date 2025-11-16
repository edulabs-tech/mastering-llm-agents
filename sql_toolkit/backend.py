from pprint import pprint

from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit



load_dotenv()

# Add memory to the process
memory = MemorySaver()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
llm_for_query_checker = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


db = SQLDatabase.from_uri("sqlite:///../docs/demo.db")
# db = SQLDatabase.from_uri("sqlite:///../docs/netflixdb.sqlite")
# print(db.dialect)
# print(db.get_usable_table_names())


toolkit = SQLDatabaseToolkit(db=db, llm=llm_for_query_checker)
db_tools = toolkit.get_tools()
# pprint(db_tools)


SYSTEM_PROMPT = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables."""



sql_agent = create_agent(
    model=llm,
    tools=db_tools,
    checkpointer=memory,
    system_prompt=SYSTEM_PROMPT
)

def invoke_llm(user_input: str, thread_id: str):
    response = sql_agent.invoke(
        {"messages": [("user", user_input)]},
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )
    last_message = response["messages"][-1].content
    text = ""
    if isinstance(last_message, str):
        text = last_message
    elif isinstance(last_message, list):
        text = last_message[0]['text']
    return text



# More tools:
# https://python.langchain.com/docs/integrations/tools/
