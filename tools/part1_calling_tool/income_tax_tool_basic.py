import pprint

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# https://python.langchain.com/docs/concepts/tool_calling/

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


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
tools = [
    calculate_income_tax,
]

llm_with_tools = llm.bind_tools(tools)

prompt_template = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ("human", "{text}")
])

chain = prompt_template | llm_with_tools | StrOutputParser()

def invoke_llm(prompt, history):
    response =  chain.invoke({"history": history, "text": prompt})
    pprint.pprint(response)
    return response

if __name__ == '__main__':
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        ai_response = llm_with_tools.invoke(user_input)
        ai_response.pretty_print()
        print("---------------------------------------------")

