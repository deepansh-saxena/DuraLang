# DuraLang

**Keep your LangChain code. Add durability with one decorator.**

DuraLang makes agent execution resilient in production by routing LLM calls, tool calls, and MCP calls through Temporal. You write normal LangChain loops. DuraLang handles retries, crash recovery, and observability.

## Why Teams Use DuraLang

- Stop losing full runs to one flaky API or timeout.
- Avoid rewriting agents into graph-specific patterns.
- Keep your existing LangChain mental model and code style.
- Get per-call visibility in Temporal without custom tracing code.

## The 10-Second Mental Model

Without DuraLang:

- One failure can force a full rerun.
- You pay for repeated LLM calls after crashes.

With DuraLang:

- Each `ainvoke()` or MCP call is durable.
- A failed step retries or resumes from the last successful point.

## The Entire API Surface

```python
from duralang import dura, dura_agent_tool

@dura
async def my_agent(messages):
    # Your existing async LangChain code stays the same.
    ...

result = await my_agent(messages)
```

`dura_agent_tool()` is optional and used when you want multi-agent orchestration through normal tool calling.

## Quick Start

```bash
pip install "duralang[anthropic]"
temporal server start-dev
```

```python
from duralang import dura

@dura
async def research_agent(messages):
    # Standard LangChain loop with llm.bind_tools(...)
    # and tool.ainvoke(...) calls
    ...
```

For a complete runnable first example, go to [docs/getting-started.md](docs/getting-started.md).

## What You Get

| Capability | Outcome |
|---|---|
| Per-step durability | LLM/tool/MCP calls recover from transient failure |
| Automatic retries | Backoff and retry without manual boilerplate |
| Crash recovery | Resume from recorded history instead of full restart |
| Temporal visibility | Inputs, outputs, timings, retry history in one place |
| Multi-agent support | `@dura` calling `@dura` becomes child workflows |
| Tool compatibility | Sub-agents and regular tools coexist in `bind_tools()` |

## Who This Is For

- Teams shipping LLM apps to production.
- Founders who need reliability before open-source launch.
- Platform teams standardizing resilient agent execution.

## Documentation

| | |
|---|---|
| [Getting Started](docs/getting-started.md) | [Core Concepts](docs/core-concepts.md) |
| [Configuration](docs/configuration.md) | [Activities](docs/activities.md) |
| [Tools & MCP](docs/tools-and-mcp.md) | [Architecture](docs/architecture.md) |
| [Error Handling](docs/error-handling.md) | [API Reference](docs/api-reference.md) |
| [Examples](docs/examples.md) | [FAQ](docs/faq.md) |

## Requirements

- Python 3.11+
- [Temporal Server](https://docs.temporal.io/cli#install) (local or cloud)
- LLM provider credentials (Anthropic, OpenAI, Google, or Ollama)

## License

MIT
