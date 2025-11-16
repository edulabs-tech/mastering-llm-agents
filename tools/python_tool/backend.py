from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

from python_ast_repl_tool import PythonAstREPLTool

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
python_tool = PythonAstREPLTool()

tools = [
    python_tool
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