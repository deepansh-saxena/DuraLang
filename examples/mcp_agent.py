"""MCP agent — MCP tools work with dura_agent like any other tool.

langchain-mcp-adapters converts MCP tools to standard LangChain BaseTool
instances. dura_agent() wraps them with DuraTool automatically — each
tool call becomes a durable Temporal Activity.

The MCP client must be created OUTSIDE the @dura function because MCP
stdio transport spawns subprocesses, which Temporal's event loop doesn't
support. The tools are stored at module level and accessed inside @dura.

Prerequisites:
    pip install langchain-mcp-adapters
    temporal server start-dev
    ANTHROPIC_API_KEY set
"""

import asyncio

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from duralang import dura, dura_agent

# Module-level tools — populated before calling @dura function.
# MCP tools are standard BaseTool instances, so dura_agent wraps them normally.
_mcp_tools: list = []


@dura
async def fs_agent(messages: list) -> list:
    """Agent with MCP filesystem tools — fully durable."""
    agent = dura_agent(
        model="claude-sonnet-4-6",
        tools=_mcp_tools,
        system_prompt="You have access to filesystem tools. Use them to complete the user's request.",
    )
    result = await agent.ainvoke({"messages": messages})
    return result["messages"]


async def main():
    global _mcp_tools

    # Create MCP client and fetch tools BEFORE entering @dura.
    # The tools are BaseTool instances — dura_agent wraps them with DuraTool.
    client = MultiServerMCPClient(
        {
            "filesystem": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            },
        }
    )
    _mcp_tools = await client.get_tools()
    print(f"MCP tools loaded: {[t.name for t in _mcp_tools]}")

    result = await fs_agent(
        [HumanMessage(content="List all files in /tmp and tell me how many there are")],
    )
    print(result[-1].content)


if __name__ == "__main__":
    asyncio.run(main())
