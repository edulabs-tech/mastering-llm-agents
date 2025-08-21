from dotenv import load_dotenv

from typing import Annotated

from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI


load_dotenv()


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

memory = MemorySaver()
graph = graph_builder.compile(
    checkpointer=memory
)

# without memory
# def invoke_llm(user_input: str, history):
#     print(history)
#     response: State = graph.invoke({"messages": [(m["role"], m["content"]) for m in history] + [("user", user_input)]})
#     return response["messages"][-1].content


def invoke_llm(user_input: str, thread_id: str):
    response: State = graph.invoke(
        {"messages": [("user", user_input)]},
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )
    return response["messages"][-1].content



# from IPython.display import Image, display
# def save_graph_png():
#     graph.get_graph().draw_mermaid_png(output_file_path="graph.png")