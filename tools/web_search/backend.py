from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
tavily_tool = TavilySearch(max_results=3)

tools = [
    tavily_tool
]

memory = MemorySaver()
agent_executor = create_agent(
    llm,
    tools,
    checkpointer=memory
)

def invoke_llm(user_input: str, thread_id: str):
    response = agent_executor.invoke(
        {"messages": [("user", user_input)]},
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )
    return response["messages"][-1].content