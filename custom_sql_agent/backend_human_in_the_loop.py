from langchain_core.runnables import RunnableConfig
from langchain.tools import tool
from langgraph.types import interrupt
from nodes import *
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

@tool(
    run_query_tool.name,
    description=run_query_tool.description,
    args_schema=run_query_tool.args_schema
)
def run_query_tool_with_interrupt(config: RunnableConfig, **tool_input):
    request = {
        "action": run_query_tool.name,
        "args": tool_input,
        "description": "Please review the tool call"
    }
    response = interrupt([request]) 
    # approve the tool call
    if response["type"] == "accept":
        tool_response = run_query_tool.invoke(tool_input, config)
    # update tool call args
    elif response["type"] == "edit":
        tool_input = response["args"]["args"]
        tool_response = run_query_tool.invoke(tool_input, config)
    # respond to the LLM with user feedback
    elif response["type"] == "response":
        user_feedback = response["args"]
        tool_response = user_feedback
    else:
        raise ValueError(f"Unsupported interrupt response type: {response['type']}")

    return tool_response


run_query_node = ToolNode([run_query_tool_with_interrupt], name="run_query")

def should_continue(state: MessagesState) -> Literal[END, "run_query"]:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    else:
        return "run_query"

builder = StateGraph(MessagesState)
builder.add_node(list_tables)
builder.add_node(call_get_schema)
builder.add_node(get_schema_node, "get_schema")
builder.add_node(generate_query)
builder.add_node(run_query_node, "run_query")

builder.add_edge(START, "list_tables")
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")
builder.add_conditional_edges(
    "generate_query",
    should_continue,
)
builder.add_edge("run_query", "generate_query")

checkpointer = InMemorySaver() 
agent = builder.compile(checkpointer=checkpointer)


def invoke_llm(user_input: str, thread_id: str):
    response = agent.invoke(
        {"messages": [("user", user_input)]},
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )
    if '__interrupt__' in response:
        suggested_sql_query = response["__interrupt__"][0].value[0]['args']['query']
        return f"INTERRUPTED! Do you want to run the following query:\n{suggested_sql_query}"
    else:
        last_message = response["messages"][-1].content
        text = ""
        if isinstance(last_message, str):
            text = last_message
        elif isinstance(last_message, list):
            text = last_message[0]['text']
        return text
    
def confirm(thread_id: str):

    response = agent.invoke(
        Command(resume={"type": "accept"}),
        # Command(resume={"type": "edit", "args": {"query": "..."}}),
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


def save_graph_png():
    agent.get_graph().draw_mermaid_png(output_file_path="graph-human-in-the-loop.png")

if __name__ == '__main__':
    save_graph_png()