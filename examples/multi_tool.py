"""Multi-tool agent — parallel tool calls with @dura."""

import asyncio

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, ToolMessage
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


tools = [get_weather, get_time]
tools_by_name = {t.name: t for t in tools}


@dura
async def multi_tool_agent(messages: list) -> list:
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools(tools)

    while True:
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        # Parallel tool execution — each becomes its own Temporal Activity
        tasks = [
            tools_by_name[tc["name"]].ainvoke(tc["args"])
            for tc in response.tool_calls
        ]
        results = await asyncio.gather(*tasks)

        for tc, result in zip(response.tool_calls, results):
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )

    return messages


async def main():
    result = await multi_tool_agent(
        [HumanMessage(content="What's the weather in NYC and the time in EST?")]
    )
    print(result[-1].content)


if __name__ == "__main__":
    asyncio.run(main())
