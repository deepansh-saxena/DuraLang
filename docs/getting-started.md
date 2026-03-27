# Getting Started

This guide walks you through installing DuraLang and running your first durable agent.

---

## Prerequisites

- Python 3.11+
- A running Temporal server (local or cloud)
- An LLM API key (Anthropic, OpenAI, Google, or Ollama)

## Installation

```bash
# With Anthropic
pip install "duralang[anthropic]"

# With OpenAI
pip install "duralang[openai]"

# All providers
pip install "duralang[all-models]"

# Start local Temporal (if not already running)
temporal server start-dev
```

---

## Your First Durable Agent

```python
import asyncio
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, ToolMessage
from duralang import dura

tools = [TavilySearchResults(max_results=3)]
tools_by_name = {t.name: t for t in tools}

@dura
async def my_agent(messages: list) -> list:
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools(tools)

    while True:
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            result = await tools_by_name[tc["name"]].arun(tc["args"])
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tc["id"]
            ))

    return messages

async def main():
    result = await my_agent([
        HumanMessage(content="What is the weather in NYC?")
    ])
    print(result[-1].content)

asyncio.run(main())
```

This is **identical LangChain code**. The only addition is `@dura` on line 10.

---

## What Just Happened

1. `@dura` wrapped your function
2. When called, it started a `DuraLangWorkflow` on Temporal
3. Inside the workflow, proxy objects intercepted `llm.ainvoke()` and `tool.arun()`
4. Each call became a Temporal Activity (`dura__llm`, `dura__tool`)
5. Temporal retried failures, heartbeated long-running calls, and persisted state
6. Your function returned normally, as if nothing happened

---

## Next Steps

- [Core Concepts](core-concepts.md) — understand how interception works
- [Configuration](configuration.md) — customize timeouts, retries, and host
- [Examples](examples.md) — multi-tool, MCP, multi-agent examples
