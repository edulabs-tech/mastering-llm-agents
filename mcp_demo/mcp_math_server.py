from typing import Tuple

from langchain_core.tools import tool
from langchain_mcp_adapters.tools import to_fastmcp
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel


class ToolResponse(BaseModel):
    content: str
    artifact: int

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> dict:
    """Add two numbers"""
    return {"content": f"the result is {a + b}", "artifact": a + b}


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")