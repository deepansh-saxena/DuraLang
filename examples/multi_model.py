"""Multi-model — same agent code, different LLM providers."""

import asyncio

from langchain_core.messages import HumanMessage

from duralang import dura


@dura
async def chat_agent(messages: list, provider: str = "anthropic") -> list:
    """Agent that works with any LLM provider."""
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model="claude-sonnet-4-6")
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o")
    else:
        raise ValueError(f"Unknown provider: {provider}")

    response = await llm.ainvoke(messages)
    messages.append(response)
    return messages


async def main():
    question = [HumanMessage(content="What is 2 + 2?")]

    # Same agent, different providers — each becomes its own Temporal workflow
    anthropic_result = await chat_agent(question, provider="anthropic")
    print(f"Anthropic: {anthropic_result[-1].content}")

    openai_result = await chat_agent(question, provider="openai")
    print(f"OpenAI: {openai_result[-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
