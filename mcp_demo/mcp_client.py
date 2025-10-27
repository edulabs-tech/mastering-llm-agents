# Create server parameters for stdio connection
import json

from langchain_core.messages import ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools, convert_mcp_tool_to_langchain_tool
from langgraph.prebuilt import create_react_agent
import asyncio

from dotenv import load_dotenv
load_dotenv()



def pre_model_hook(state):
    messages = state["messages"]
    if len(messages) > 0:
        last_message = messages[-1]
        if isinstance(last_message, ToolMessage):
            tool_result = json.loads(last_message.content)
            last_message.content = tool_result["content"]
            last_message.artifact = tool_result["artifact"]
    return {
        "messages": messages
    }

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


            # Create and run the agent
            gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            agent = create_react_agent(
                gemini_model,
                mcp_tools,
                # IMPORTANT - WE WILL USE THIS HOOK TO POPULATE TOOL_MESSAGE
                # with content and artifact
                pre_model_hook=pre_model_hook,
            )
            # agent_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
            agent_response = await agent.ainvoke({"messages": "solve the following: 5+4 and 2+3?"})

            for m in agent_response["messages"]:
                m.pretty_print()
                if isinstance(m, ToolMessage):
                    print(f"Artifact: {m.artifact}")
                    print(f"Content (to LLM): {m.content}")

if __name__ == "__main__":
    asyncio.run(main())