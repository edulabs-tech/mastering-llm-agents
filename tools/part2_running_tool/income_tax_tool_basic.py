import pprint

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# https://python.langchain.com/docs/concepts/tool_calling/

@tool
def calculate_income_tax(annual_income: int):
    """
    Calculate Israeli annual income tax based on annual income in ILS
    params:
        annual_income: annual income in ILS
    """
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


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
tools = [
    calculate_income_tax,
]

llm_with_tools = llm.bind_tools(tools)

prompt_template = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ("human", "{text}")
])

chain = prompt_template | llm_with_tools


def invoke_llm(prompt, history):
    print(history)
    response: AIMessage = chain.invoke({"history": history, "text": prompt})
    print(response)

    if len(response.tool_calls) > 0:
        single_tool_call = response.tool_calls[0]
        # now run function
        tool_call_id = single_tool_call["id"]
        tool_name = single_tool_call['name']
        tool_args = single_tool_call['args']
        print(f"Requested to run tool {tool_name} with args {tool_args}")
        if tool_name == "calculate_income_tax":
            tax = calculate_income_tax.invoke(tool_args)
            print(f"Calculated income tax: {tax}")
            # lets call LLM with this response so it could generate better reply
            tool_message: ToolMessage = ToolMessage(content=str(tax), tool_call_id=tool_call_id)
            response = llm_with_tools.invoke(
                [(m["role"], m["content"]) for m in history] + [("human", prompt)] + [response]  + [tool_message])
            return response.content
    else:
        return response.content

# calculate income tax if my annual income is 1234567
