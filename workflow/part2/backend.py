import pprint

from dotenv import load_dotenv

from typing import Annotated

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

from langchain_community.tools.tavily_search import TavilySearchResults


import json

from langchain_core.messages import ToolMessage


load_dotenv()


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]



graph_builder = StateGraph(State)

# tool = TavilySearchResults(max_results=2)


@tool
def calculate_income_tax(annual_income):
    """Calculate annual income tax based on annual income"""
    tax = 0
    brackets = [
        (84120, 0.10),
        (120720, 0.14),
        (193800, 0.20),
        (269280, 0.31),
        (560280, 0.35),
        (721560, 0.47),
    ]

    remaining_income = annual_income

    for i, (limit, rate) in enumerate(brackets):
        if remaining_income > limit:
            tax += limit * rate
            remaining_income -= limit
        else:
            tax += remaining_income * rate
            return tax

    # Additional tax for incomes above 721,560 ₪
    tax += remaining_income * 0.50
    if annual_income > 721560:
        tax += (annual_income - 721560) * 0.03

    return tax

tools = [calculate_income_tax]

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
llm = llm.bind_tools(tools)


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        print(f"*** Calling Tool Node... ***")
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            print(f"Tool result:")
            pprint.pprint(tool_result)
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


def route_tools(state: State):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        print(f"*** Routing to Tools Node ***")
        return "tools"
    print(f"*** Routing to END Node ***")
    return END


def chatbot(state: State):
    print(f"*** Invoking Chatbot Node ***")
    return {"messages": [llm.invoke(state["messages"])]}


tool_node = BasicToolNode(tools)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node) # Adding Tools node
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "chatbot") # Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_conditional_edges("chatbot", route_tools)

memory = MemorySaver()
graph = graph_builder.compile(
    checkpointer=memory
)


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

