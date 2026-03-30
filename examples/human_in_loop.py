"""Human-in-the-loop — placeholder for Temporal signal-based human input.

This example demonstrates the pattern for pausing an agent to wait for
human input via Temporal Signals. Full implementation is planned for v2.
"""

import asyncio

from langchain_core.messages import HumanMessage

from duralang import dura, dura_agent


@dura
async def agent_with_confirmation(messages: list) -> list:
    """Agent that would pause for human confirmation before acting.

    In v2, this will use Temporal Signals to pause and wait for input.
    For now, this demonstrates the pattern.
    """
    agent = dura_agent(
        model="claude-sonnet-4-6",
        system_prompt="You are a helpful assistant that drafts content for review.",
    )
    result = await agent.ainvoke({"messages": messages})
    return result["messages"]


async def main():
    result = await agent_with_confirmation(
        [HumanMessage(content="Draft an email to the team about the project update")]
    )
    print(result[-1].content)


if __name__ == "__main__":
    asyncio.run(main())
