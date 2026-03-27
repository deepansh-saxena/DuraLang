"""Basic agent — identical LangChain code with @dura for durability.

This is a standard LangChain agent using create_agent. The only DuraLang
addition is @dura on the function. Every LLM call and tool call inside
becomes a Temporal Activity — automatically retried, heartbeated, and durable.
"""

import asyncio

from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage

from duralang import dura


@dura
async def research_agent(messages: list) -> list:
    agent = create_agent(
        model="claude-sonnet-4-6",
        tools=[TavilySearchResults(max_results=3)],
    )
    result = await agent.ainvoke({"messages": messages})
    return result["messages"]


async def main():
    result = await research_agent(
        [HumanMessage(content="What are the latest developments in AI agents?")]
    )
    print(result[-1].content)


if __name__ == "__main__":
    asyncio.run(main())
