"""Multi-model — same agent code, different LLM providers."""

import asyncio

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from duralang import dura


@dura
async def chat_agent(messages: list, provider: str = "anthropic") -> list:
    """Agent that works with any LLM provider."""
    models = {
        "anthropic": "claude-sonnet-4-6",
        "openai": "gpt-4o",
    }
    model = models.get(provider)
    if model is None:
        raise ValueError(f"Unknown provider: {provider}")

    agent = create_agent(model=model)
    result = await agent.ainvoke({"messages": messages})
    return result["messages"]


async def main():
    question = [HumanMessage(content="What is 2 + 2?")]

    # Same agent, different providers — each becomes its own Temporal workflow
    anthropic_result = await chat_agent(question, provider="anthropic")
    print(f"Anthropic: {anthropic_result[-1].content}")

    openai_result = await chat_agent(question, provider="openai")
    print(f"OpenAI: {openai_result[-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
