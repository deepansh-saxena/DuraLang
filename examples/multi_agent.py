"""Multi-agent — @dura functions calling @dura functions become Child Workflows.

This shows two patterns for multi-agent calls:
  1. Direct calls: @dura calling @dura (simple, works for fixed delegation)
  2. Agent tools: dura_agent_tool() wraps @dura as BaseTool (flexible, LLM decides)

Both patterns produce Temporal Child Workflows with independent event histories.
"""

import asyncio

from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage

from duralang import dura, dura_agent_tool


@dura
async def researcher(query: str) -> str:
    """Research agent — searches the web and summarizes findings."""
    agent = create_agent(
        model="claude-sonnet-4-6",
        tools=[TavilySearchResults()],
        system_prompt="Search the web and summarize your findings concisely.",
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content


# ── Pattern 1: Direct @dura calls ────────────────────────────────────────────
# Use this when your code decides which agents to call.


@dura
async def report_agent_direct(task: str) -> str:
    """Agent that calls researcher directly — code decides the delegation."""
    # Calling @dura from @dura → Temporal Child Workflow automatically
    research = await researcher(f"Research this topic thoroughly: {task}")

    agent = create_agent(model="claude-sonnet-4-6")
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=f"Based on this research:\n{research}\n\nWrite a summary of: {task}")]}
    )
    return result["messages"][-1].content


# ── Pattern 2: Agent tools via dura_agent_tool() ─────────────────────────────
# Use this when the LLM should decide which agents to call.

all_tools = [
    dura_agent_tool(researcher),  # LLM can call this like any tool → Child Workflow
]


@dura
async def report_agent_flexible(task: str) -> str:
    """Agent that lets the LLM decide when to call the researcher."""
    agent = create_agent(
        model="claude-sonnet-4-6",
        tools=all_tools,
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})
    return result["messages"][-1].content


async def main():
    result = await report_agent_flexible(
        "The current state of multi-agent AI systems"
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
