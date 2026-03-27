"""Multi-agent — @dura functions calling @dura functions become Child Workflows.

This shows two patterns for multi-agent calls:
  1. Direct calls: @dura calling @dura (simple, works for fixed delegation)
  2. Agent tools: dura_agent_tool() wraps @dura as BaseTool (flexible, LLM decides)

Both patterns produce Temporal Child Workflows with independent event histories.
"""

import asyncio

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, ToolMessage

from duralang import dura, dura_agent_tool

search_tool = TavilySearchResults()


@dura
async def researcher(query: str) -> str:
    """Research agent — searches the web and summarizes findings."""
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools([search_tool])

    messages = [HumanMessage(content=query)]
    for _ in range(10):
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            result = await search_tool.ainvoke(tc["args"])
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )

    return response.content


# ── Pattern 1: Direct @dura calls ────────────────────────────────────────────
# Use this when your code decides which agents to call.


@dura
async def report_agent_direct(task: str) -> str:
    """Agent that calls researcher directly — code decides the delegation."""
    # Calling @dura from @dura → Temporal Child Workflow automatically
    research = await researcher(f"Research this topic thoroughly: {task}")

    llm = ChatAnthropic(model="claude-sonnet-4-6")
    response = await llm.ainvoke(
        [HumanMessage(content=f"Based on this research:\n{research}\n\nWrite a summary of: {task}")]
    )
    return response.content


# ── Pattern 2: Agent tools via dura_agent_tool() ─────────────────────────────
# Use this when the LLM should decide which agents to call.

all_tools = [
    dura_agent_tool(researcher),  # LLM can call this like any tool → Child Workflow
]
tools_by_name = {t.name: t for t in all_tools}


@dura
async def report_agent_flexible(task: str) -> str:
    """Agent that lets the LLM decide when to call the researcher."""
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools(all_tools)

    messages = [HumanMessage(content=task)]
    for _ in range(10):
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            result = await tools_by_name[tc["name"]].ainvoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return response.content


async def main():
    result = await report_agent_flexible(
        "The current state of multi-agent AI systems"
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
