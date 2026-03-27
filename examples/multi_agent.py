"""Multi-agent — @@dura calling @dura becomes Temporal Child Workflows."""

import asyncio

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, ToolMessage

from duralang import dura

search_tool = TavilySearchResults()


@dura
async def researcher(messages: list) -> list:
    """Subagent — becomes a child workflow when called from orchestrator."""
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools([search_tool])

    while True:
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            result = await search_tool.arun(tc["args"])
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )

    return messages


@dura
async def orchestrator(task: str) -> str:
    """Top-level agent — calls researcher as a child workflow."""
    llm = ChatAnthropic(model="claude-sonnet-4-6")

    # Calling @dura from @dura -> Temporal Child Workflow automatically
    research_result = await researcher(
        [HumanMessage(content=f"Research this topic thoroughly: {task}")]
    )

    # Use research to generate final response
    response = await llm.ainvoke(
        [
            HumanMessage(
                content=(
                    f"Based on this research: {research_result[-1].content}\n\n"
                    f"Write a summary of: {task}"
                )
            )
        ]
    )
    return response.content


async def main():
    result = await orchestrator("The current state of multi-agent AI systems")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
