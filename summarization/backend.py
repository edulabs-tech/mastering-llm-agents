
from typing import Any

from dotenv import load_dotenv

from langchain_core.messages.utils import count_tokens_approximately

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langmem.short_term import SummarizationNode
from langchain_tavily import TavilySearch


load_dotenv()

# Add memory to the process
memory = MemorySaver()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

class State(AgentState):
    # NOTE: we're adding this key to keep track of previous summary information
    # to make sure we're not summarizing on every LLM call
    context: dict[str, Any]

summarization_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")


summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=summarization_model,
    max_tokens=384,
    max_summary_tokens=128,
    output_messages_key="llm_input_messages",
)


SYSTEM_PROMPT = """
You are a friendly assistant.
Be polite, and answer long answers.
"""


agent = create_react_agent(
    llm,
    tools=[TavilySearch(max_results=2)],
    checkpointer=memory,
    prompt=SYSTEM_PROMPT,
    pre_model_hook=summarization_node,
    state_schema=State,
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


# hi
# what can you do?
# tell me a joke


