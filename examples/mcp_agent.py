"""MCP agent — DuraMCPSession makes MCP calls durable.

Note: MCP tools are not standard LangChain tools, so they use
DuraMCPSession for durable call_tool() instead of create_agent.
"""

import asyncio

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, ToolMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from duralang import DuraMCPSession, dura


@dura
async def fs_agent(messages: list, fs) -> list:
    """Agent that uses an MCP filesystem server."""
    # fs is a DuraMCPSession — same API as ClientSession
    tools_result = await fs.list_tools()
    tool_schemas = [
        {
            "name": t.name,
            "description": t.description,
            "parameters": t.inputSchema,
        }
        for t in tools_result.tools
    ]

    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools(tool_schemas)

    while True:
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            result = await fs.call_tool(tc["name"], tc["args"])  # -> Temporal Activity
            messages.append(
                ToolMessage(content=str(result.content), tool_call_id=tc["id"])
            )

    return messages


async def main():
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            fs = DuraMCPSession(session, "filesystem")  # <- one line

            result = await fs_agent(
                [HumanMessage(content="List all files in /tmp")],
                fs=fs,
            )
    print(result[-1].content)


if __name__ == "__main__":
    asyncio.run(main())
