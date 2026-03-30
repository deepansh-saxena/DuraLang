"""Simple agent — single agent with one tool, minimal API calls.

This is the simplest possible DuraLang example:
  - One @dura agent with one tool
  - LLM call → Temporal Activity
  - Tool call → Temporal Activity
"""

import asyncio

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from duralang import dura, dura_agent


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@dura
async def math_agent(task: str) -> str:
    """Simple math agent with a calculator tool."""
    agent = dura_agent(
        model="claude-sonnet-4-6",
        tools=[calculator],
        system_prompt="You are a math assistant. Use the calculator tool to evaluate expressions. Be concise.",
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})
    return result["messages"][-1].content


async def main():
    print("=" * 60)
    print("SIMPLE AGENT — one tool, fully durable")
    print("=" * 60)
    print()

    result = await math_agent("What is 15 * 37 + 42?")

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
