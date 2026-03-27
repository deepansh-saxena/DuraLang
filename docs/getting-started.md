# Getting Started

This guide gets you from install to first durable run fast.

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

At the code level, the concept is simple: decorate your existing async LangChain agent function with `@dura`.

```python
from duralang import dura

@dura
async def my_agent(messages):
    # Your existing LangChain loop remains unchanged.
    # llm.ainvoke(...), tool.ainvoke(...), mcp.call_tool(...)
    ...
```

Then call it like a normal async function.

---

## Run A Complete Example

Use one of the included examples for a full runnable script.

```bash
python examples/basic_agent.py
```

More examples:

- `examples/multi_tool.py`
- `examples/mcp_agent.py`
- `examples/stochastic_agents.py`

---

## What Happens Under The Hood

1. `@dura` wrapped your function
2. When called, it started a Temporal Workflow
3. Inside the workflow, DuraLang intercepted every `llm.ainvoke()` and `tool.ainvoke()` call
4. Each call became an individually retryable Temporal Activity
5. If anything failed, Temporal would have retried it automatically
6. Your function returned normally, as if nothing happened

---

## Next Steps

- [Core Concepts](core-concepts.md) — understand how interception works
- [Tools & MCP](tools-and-mcp.md) — mix sub-agents, regular tools, and MCP servers
- [Configuration](configuration.md) — customize timeouts, retries, and host
- [Examples](examples.md) — multi-tool, MCP, multi-agent examples
