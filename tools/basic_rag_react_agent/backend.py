from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_classic.tools.retriever import create_retriever_tool
from langchain.agents import AgentState


from data_preprocessing import retriever

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


class MyAgentState(AgentState):
    user_id: str



memory = MemorySaver()

retriever_tool = create_retriever_tool(
    retriever,
    name="retrieve_mortgage_info",
    description="Searches and returns documents regarding mortgage information"
)
tools = [retriever_tool]



agent = create_agent(
    state_schema=MyAgentState,
    llm=llm,
    tools=tools,
    checkpointer=memory
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



def save_graph_png():
    agent.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == '__main__':
    from IPython.display import Image, display
    save_graph_png()