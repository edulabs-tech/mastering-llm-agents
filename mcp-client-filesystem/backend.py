# https://mcpservers.org/servers/modelcontextprotocol/filesystem

from mcp import ClientSession, StdioServerParameters
import os
from dotenv import load_dotenv
load_dotenv()
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

SERVER_PARAMS = StdioServerParameters(
    command="npx",  # Executable
    args=[
        "-y", 
        "@modelcontextprotocol/server-filesystem", # The package name
        os.path.join(os.getcwd(), 'docs') # Argument: The directory we want to expose
    ],
    env=None
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

memory = MemorySaver()


async def main():

    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools from MCP server
            mcp_tools = await load_mcp_tools(session)
            for mcp_tool in mcp_tools:
                print(f"{mcp_tool.name}: {mcp_tool.description}\n\n")

            agent = create_agent(
                llm,
                mcp_tools
            )

            # agent_response = await agent.ainvoke({"messages": "what can you do?"})
            agent_response = await agent.ainvoke({"messages": "how many files are in directory docs/?"})

            for m in agent_response["messages"]:
                m.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())