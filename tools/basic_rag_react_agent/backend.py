from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_classic.tools.retriever import create_retriever_tool
from langchain.agents import AgentState
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser

from data_preprocessing import retriever

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")




memory = MemorySaver()

retriever_tool = create_retriever_tool(
    retriever,
    name="retrieve_mortgage_info",
    description="Searches and returns documents regarding mortgage information"
)
tools = [retriever_tool]



agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=memory
)


def invoke_llm(user_input: str, thread_id: str):
    response: AgentState = agent.invoke(
        # {"messages": [("user", user_input)]},
        {"messages": [HumanMessage(user_input)]},
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )
    last_message = response["messages"][-1]
    message_content = StrOutputParser().invoke(last_message)
    return message_content



def save_graph_png():
    agent.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == '__main__':
    from IPython.display import Image, display
    save_graph_png()