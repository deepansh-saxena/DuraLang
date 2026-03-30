# Getting Started

Get from zero to your first durable LLM agent in under 5 minutes.

---

## Prerequisites

| Requirement | Why |
|---|---|
| **Python 3.11+** | DuraLang uses modern Python features (`ContextVar`, type unions, `asyncio`) |
| **Temporal Server** | The durable execution engine. Runs locally or in the cloud |
| **LLM API Key** | At least one of: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY` |

---

## Step 1: Install DuraLang

```bash
pip install duralang
```

With a specific LLM provider:

```bash
pip install "duralang[anthropic]"     # Claude models
pip install "duralang[openai]"        # GPT models
pip install "duralang[google]"        # Gemini models
pip install "duralang[ollama]"        # Local models via Ollama
pip install "duralang[all-models]"    # All providers
```

---

## Step 2: Start Temporal

DuraLang needs a running Temporal server. The fastest way to get one:

```bash
# Install Temporal CLI (macOS)
brew install temporal

# Start the development server
# Includes the Temporal UI at http://localhost:8233
temporal server start-dev
```

For other operating systems, see the [Temporal CLI quickstart](https://docs.temporal.io/cli).

> **Note:** Leave this running in a separate terminal. DuraLang connects to it automatically on `localhost:7233`.

---

## Step 3: Write Your First Durable Agent

Create a file called `my_agent.py`:

```python
import asyncio
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from duralang import dura, dura_agent

# Standard LangChain tool
tools = [TavilySearchResults(max_results=3)]

@dura  # ← This is the only DuraLang-specific line
async def research_agent(messages: list) -> list:
    agent = dura_agent(
        model="claude-sonnet-4-6",
        tools=tools,
    )
    result = await agent.ainvoke({"messages": messages})   # → Temporal Activities
    return result["messages"]

async def main():
    result = await research_agent(
        [HumanMessage(content="What are the latest developments in AI agents?")]
    )
    print(result[-1].content)

asyncio.run(main())
```

Run it:

```bash
python my_agent.py
```

---

## Step 4: Inspect in Temporal UI

Open [http://localhost:8233](http://localhost:8233) in your browser. You'll see your workflow with a complete timeline of every operation:

- **Each LLM call** — with input messages, output response, latency, and attempt count
- **Each tool call** — with tool name, arguments, result, and retry history
- **Workflow state** — start time, completion, and return value

Every operation is individually visible, searchable, and replayable.

---

## What Just Happened

Behind the scenes, `@dura` did the following — without you writing any of it:

1. **Wrapped your function** as a Temporal Workflow
2. **Set a `DuraContext`** via Python's `contextvars.ContextVar`
3. **Intercepted every `ainvoke()` call** — LLM calls routed to `dura__llm` Activity, tool calls routed to `dura__tool` Activity
4. **Checkpointed each operation** in Temporal's event history
5. **Applied retry policies** — transient failures (timeouts, rate limits) would be retried automatically with exponential backoff
6. **Emitted heartbeats** — if any operation hung, Temporal would detect it and reschedule
7. **Returned the result** as if nothing happened — your calling code sees a normal async function

If any call had failed, Temporal would have retried it. If the entire process had crashed, restarting with the same workflow ID would resume from the last checkpoint — no completed calls re-executed, no API costs wasted.

---

## Next Steps

You now have a working durable agent. Here's where to go from here:

| Goal | Doc |
|---|---|
| Understand how interception works at the code level | [Core Concepts](core-concepts.md) |
| See the full architecture with sequence diagrams | [Architecture](architecture.md) |
| Build multi-agent systems with sub-agents as tools | [Tools & MCP](tools-and-mcp.md) |
| Customize retries, timeouts, and heartbeats | [Configuration](configuration.md) |
| Run the crash recovery demo | [Examples](examples.md) |
| Connect MCP servers for file/database/API access | [Tools & MCP](tools-and-mcp.md) |
