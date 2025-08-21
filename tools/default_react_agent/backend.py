from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_experimental.tools import PythonAstREPLTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

python_tool = PythonAstREPLTool()

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


tools = [
    python_tool,
    calculate_income_tax
]

memory = MemorySaver()
agent_executor = create_react_agent(
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