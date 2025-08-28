import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from prompts import SYSTEM_PROMPT
from tools import TOOLS

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
checkpointer = MemorySaver()


agent = create_react_agent(
    llm,
    TOOLS,
    prompt=SYSTEM_PROMPT.format(current_date=datetime.date.today()),
    checkpointer=checkpointer
)

def invoke_llm(user_input: str, thread_id: str):
    response = agent.invoke(
        {"messages": [("user", user_input)]},
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )
    return response["messages"][-1].content