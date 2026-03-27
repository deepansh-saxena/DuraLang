"""Multi-tool agent — multiple tools with @dura.

create_agent handles tool dispatch and parallel execution automatically.
Each tool call becomes its own durable Temporal Activity.
"""

import asyncio

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from duralang import dura


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"72F and sunny in {city}"


@tool
def get_time(timezone: str) -> str:
    """Get the current time in a timezone."""
    return f"3:00 PM in {timezone}"


@dura
async def multi_tool_agent(messages: list) -> list:
    agent = create_agent(
        model="claude-sonnet-4-6",
        tools=[get_weather, get_time],
    )
    result = await agent.ainvoke({"messages": messages})
    return result["messages"]


async def main():
    result = await multi_tool_agent(
        [HumanMessage(content="What's the weather in NYC and the time in EST?")]
    )
    print(result[-1].content)


if __name__ == "__main__":
    asyncio.run(main())
