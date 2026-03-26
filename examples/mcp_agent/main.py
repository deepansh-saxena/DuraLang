"""MCP Agent example — LangForge with an MCP filesystem server."""

import asyncio

from langforge import ForgeConfig, ForgeRuntime, LLMConfig


async def main():
    config = ForgeConfig(temporal_host="localhost:7233")

    # In a real scenario, you'd connect to an MCP server:
    # from mcp import ClientSession, StdioServerParameters
    # from mcp.client.stdio import stdio_client
    #
    # server_params = StdioServerParameters(
    #     command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    # )
    # async with stdio_client(server_params) as (read, write):
    #     async with ClientSession(read, write) as session:
    #         await session.initialize()
    #
    #         async with ForgeRuntime(config) as runtime:
    #             graph_id = runtime.register_graph(graph)
    #             await runtime.register_mcp_session(session, "filesystem")
    #             ...

    print("MCP agent example — see comments in source for usage pattern")


if __name__ == "__main__":
    asyncio.run(main())
