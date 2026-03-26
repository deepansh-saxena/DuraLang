"""Basic LangForge example — ReAct agent with tools, powered by Temporal."""

import asyncio

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent

from langforge import ForgeConfig, ForgeRuntime, LLMConfig


async def main():
    config = ForgeConfig(temporal_host="localhost:7233")
    tools = [TavilySearchResults(max_results=3)]

    # Compile with NO checkpointer — LangForge owns state
    graph = create_react_agent(ChatAnthropic(model="claude-sonnet-4-6"), tools)

    async with ForgeRuntime(config) as runtime:
        graph_id = runtime.register_graph(graph)
        runtime.register_tools(tools)

        result = await runtime.run(
            graph_id=graph_id,
            initial_state={
                "messages": [{"role": "user", "content": "What is the weather in NYC?"}]
            },
            llm_config=LLMConfig(provider="anthropic", model="claude-sonnet-4-6"),
        )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
