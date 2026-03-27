# DuraLang Documentation

**Write normal LangChain code. Get Temporal durability. One decorator.**

DuraLang is the missing durability layer for LangChain. It intercepts every `llm.ainvoke()`, `tool.ainvoke()`, and `session.call_tool()` inside a `@dura`-decorated function and routes each call through Temporal — giving you automatic retries, heartbeating, crash recovery, and full observability without changing your agent logic.

---

## Navigate by Goal

| I want to... | Start here |
|---|---|
| Get my first durable agent running in 5 minutes | [Getting Started](getting-started.md) |
| Understand how `@dura` intercepts calls transparently | [Core Concepts](core-concepts.md) |
| See the full system architecture and request flows | [Architecture](architecture.md) |
| Tune retries, timeouts, heartbeats, and task queues | [Configuration](configuration.md) |
| Use LangChain tools, agent tools, and MCP servers together | [Tools & MCP](tools-and-mcp.md) |
| Build human approval and review flows | [Human-in-the-Loop](human-in-the-loop.md) |
| Look up every public class and function | [API Reference](api-reference.md) |
| Understand how the three activities work internally | [Activities](activities.md) |
| Debug failures and understand retry vs non-retry errors | [Error Handling](error-handling.md) |
| See complete runnable examples with walkthroughs | [Examples](examples.md) |
| Get answers to common questions | [FAQ](faq.md) |

---

## What DuraLang Gives You

| Capability | How |
|---|---|
| **One public API** | `@dura` — decorate your function, get durability |
| **Multi-agent bridge** | `dura_agent_tool()` — wrap `@dura` as a `BaseTool` |
| **Three durable routes** | `dura__llm` · `dura__tool` · `dura__mcp` |
| **Hierarchical execution** | Child workflows when `@dura` calls `@dura` |
| **Zero code change** | Your LangChain loop stays identical |
| **Full observability** | Every call visible in Temporal UI |

---

## Recommended Reading Order

For a complete understanding, read the docs in this order:

1. **[Getting Started](getting-started.md)** — Install, prerequisites, first durable run
2. **[Core Concepts](core-concepts.md)** — The three layers, DuraContext, LLMIdentity, proxies
3. **[Architecture](architecture.md)** — System diagrams, request flows, module dependencies
4. **[Configuration](configuration.md)** — DuraConfig, ActivityConfig, retry policies, task queues
5. **[Tools & MCP](tools-and-mcp.md)** — LangChain tools, agent tools, MCP servers, routing
6. **[Activities](activities.md)** — How `dura__llm`, `dura__tool`, `dura__mcp` work internally
7. **[Error Handling](error-handling.md)** — Exception hierarchy, retryable vs non-retryable errors
8. **[API Reference](api-reference.md)** — Complete public API documentation
9. **[Examples](examples.md)** — Runnable demos with detailed walkthroughs
10. **[FAQ](faq.md)** — Troubleshooting and answers to common questions
