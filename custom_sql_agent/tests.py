from dotenv import load_dotenv
from nodes import call_get_schema, list_tables

load_dotenv()

print(list_tables({"messages": []}))
print(call_get_schema(
    {
        "messages": [
            ("user", "how many songs are there?")
        ]
    }
))