# DuraLang Documentation

Welcome to DuraLang — **write normal LangChain code, get Temporal durability with one decorator**.

DuraLang adds `@dura` to your existing LangChain agent. Every `llm.ainvoke()`, `tool.arun()`, and `session.call_tool()` becomes a Temporal Activity — automatically retried, heartbeated, and visible in the Temporal UI. `@dura` functions calling other `@dura` functions become Temporal Child Workflows. Your code doesn't change.

---

## Quick Navigation

| Topic | Description |
|---|---|
| [Getting Started](getting-started.md) | Install, prerequisites, first agent |
| [Core Concepts](core-concepts.md) | How `@dura`, proxies, and context work |
| [Configuration](configuration.md) | `DuraConfig`, `ActivityConfig` |
| [Activities](activities.md) | `dura__llm`, `dura__tool`, `dura__mcp` |
| [Tools & MCP](tools-and-mcp.md) | LangChain tools and MCP servers |
| [Human-in-the-Loop](human-in-the-loop.md) | Temporal signals for human input |
| [Error Handling](error-handling.md) | Retryable vs non-retryable errors |
| [API Reference](api-reference.md) | `dura`, `DuraConfig`, `DuraMCPSession` |
| [Examples](examples.md) | Working code examples |
| [FAQ](faq.md) | Common questions |

---

## The Entire API

```python
from duralang import dura

@dura
async def my_agent(messages):
    # your existing LangChain code here, unchanged
    ...

result = await my_agent([HumanMessage(content="hello")])
```
