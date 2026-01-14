# Create server parameters for stdio connection
import json

from langchain_core.messages import ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent
import asyncio

from dotenv import load_dotenv
load_dotenv()


async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_math_server.py"],  # make sure path is correct
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools from MCP server

            mcp_tools = await load_mcp_tools(session)
            print(f"Detected tools: {mcp_tools}")


            # # # Create and run the agent
            gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
            agent = create_agent(
                gemini_model,
                mcp_tools
            )
            agent_response = await agent.ainvoke({"messages": "what's 3 + 5 x 12?"})
            # agent_response = await agent.ainvoke({"messages": "solve the following: 5+4 and 2+3?"})

            for m in agent_response["messages"]:
                m.pretty_print()

if __name__ == "__main__":
    asyncio.run(main())