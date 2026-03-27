# DuraLang Documentation

DuraLang lets you keep normal LangChain agent code while adding production durability through Temporal.

If you are new, start here: [Getting Started](getting-started.md)

## Start By Goal

| I want to... | Go here |
|---|---|
| Get my first durable run in minutes | [Getting Started](getting-started.md) |
| Understand what `@dura` intercepts | [Core Concepts](core-concepts.md) |
| Tune retries, timeouts, and worker settings | [Configuration](configuration.md) |
| Use tools, MCP, and multi-agent patterns | [Tools & MCP](tools-and-mcp.md) |
| Build human approval flows | [Human-in-the-Loop](human-in-the-loop.md) |
| See concrete runnable samples | [Examples](examples.md) |
| Debug failures and classify retryability | [Error Handling](error-handling.md) |
| Look up APIs quickly | [API Reference](api-reference.md) |

## Product Snapshot

- One primary API: `@dura`
- Optional multi-agent bridge: `dura_agent_tool()`
- Three durable activity routes: `dura__llm`, `dura__tool`, `dura__mcp`
- Child workflows when one `@dura` function calls another

## Suggested Reading Order

1. [Getting Started](getting-started.md)
2. [Core Concepts](core-concepts.md)
3. [Configuration](configuration.md)
4. [Tools & MCP](tools-and-mcp.md)
5. [Examples](examples.md)
