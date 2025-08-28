import os

from langchain_core.tools import tool
from langchain_tavily import TavilySearch
import requests

from dotenv import load_dotenv

load_dotenv()

tavily_tool = TavilySearch(max_results=3)

API_KEY = os.environ.get("EXCHANGE_RATE_API_KEY")

@tool
def get_exchange_rate(currency_from: str, currency_to: str):
    """
    Receives two currencies and returns excheange rate between them
    :param currency_from: Currency code FROM which we need to convert - ISO 4217 Three Letter Currency Codes - e.g. USD for US Dollars, EUR for Euro, etc.
    :param currency_to: Currency code TO which we need to convert - ISO 4217 Three Letter Currency Codes - e.g. USD for US Dollars, EUR for Euro, etc.
    :return: exchange rate from currency_from to currency_to
    """
    url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/latest/{currency_from}"
    response = requests.get(url)
    response.raise_for_status()

    data = response.json()
    conversion_rates = data["conversion_rates"]
    rate = conversion_rates[currency_to]

    return rate

TOOLS = [
    tavily_tool, get_exchange_rate
]


if __name__ == "__main__":
    print(get_exchange_rate.invoke({"currency_from": 'ILS', "currency_to": "USD"}))