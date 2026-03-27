"""Basic agent — identical LangChain code with @dura for durability."""

import asyncio

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, ToolMessage

from duralang import dura

tools = [TavilySearchResults(max_results=3)]
tools_by_name = {t.name: t for t in tools}


@dura
async def research_agent(messages: list) -> list:
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools(tools)

    while True:
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            result = await tools_by_name[tc["name"]].arun(tc["args"])
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )

    return messages


async def main():
    result = await research_agent(
        [HumanMessage(content="What are the latest developments in AI agents?")]
    )
    print(result[-1].content)


if __name__ == "__main__":
    asyncio.run(main())
