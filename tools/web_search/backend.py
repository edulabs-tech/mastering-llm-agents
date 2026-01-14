from pprint import pprint
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

from langchain.agents.middleware import SummarizationMiddleware
from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_for_summarization = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
tavily_tool = TavilySearch(max_results=3)


# pprint(tavily_tool.invoke("latest ollama version"))


tools = [
    tavily_tool
]

memory = MemorySaver()
agent_executor = create_agent(
    llm,
    tools,
    checkpointer=memory,
    middleware=[
        SummarizationMiddleware(
            model=llm_for_summarization,
            max_tokens_before_summary=500,  # Trigger summarization at 500 tokens
            messages_to_keep=1,  # Keep last 1 messages after summary
            # summary_prompt="Custom prompt for summarization...",  # Optional
        )
    ]
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