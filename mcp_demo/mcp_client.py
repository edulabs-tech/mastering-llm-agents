# Create server parameters for stdio connection
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
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
            tools = await load_mcp_tools(session)
            print(f"Detected tools: {tools}")

            # Create and run the agent
            gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            agent = create_react_agent(gemini_model, tools)
            agent_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
            # agent_response = await agent.ainvoke({"messages": "solve the following: 5+4 and 2+3?"})

            for m in agent_response["messages"]:
                m.pretty_print()

if __name__ == "__main__":
    asyncio.run(main())