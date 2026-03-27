<div align="center">

<h1>duralang</h1>

<p><strong>Agents that cannot fail. One decorator.</strong></p>

<p>
<code>duralang</code> is the missing durability layer for LangChain.<br>
Write the same LangChain code you already know — add <code>@dura</code> — and every LLM call,<br> tool call, MCP call, and agent-to-agent call becomes individually recoverable,<br> automatically retried, and fully observable through Temporal.
</p>

<p><strong>No new framework. No graph DSL. No code rewrite. Just durability.</strong></p>

[![PyPI](https://img.shields.io/pypi/v/duralang)](https://pypi.org/project/duralang)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Temporal](https://img.shields.io/badge/powered%20by-Temporal-black)](https://temporal.io)

</div>

---

## The Problem

**Most AI agent failures are infrastructure failures, not intelligence failures.**

The model picked the right tool. The reasoning was correct. But a network timeout at minute 47 of a 60-minute run killed the entire pipeline — and you lost every completed step along with it.

This is the reality of production agent systems today:

- **LangChain** gives you the best composability layer for LLM applications. But it has no built-in answer for what happens when a call fails mid-run. Your agent restarts from scratch. Previous LLM calls (and their costs) are wasted.

- **LangGraph** solves this with graph-based checkpointing. But it requires restructuring your code into explicit nodes and edges. Free-form agent loops — where the LLM decides what to call, in what order, how many times — don't map cleanly to static graphs. You end up choosing between graph complexity or coarse checkpoints that don't protect individual operations.

- **Temporal** is the gold standard for durable execution. But using it with LLM agents requires you to manually define workflows, activities, serialization boundaries, and retry policies for every operation. It's powerful but high-ceremony — the opposite of what you want when iterating on agent logic.

**The gap:** there is no way to take an existing LangChain agent and make it durable without rewriting it.

**duralang fills that gap.**

---

## The Solution

```python
from duralang import dura  # ← only new import

@dura                       # ← only code change
async def my_agent(messages):
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools(tools)

    while True:
        response = await llm_with_tools.ainvoke(messages)   # → Temporal Activity
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            result = await tools_by_name[tc["name"]].ainvoke(tc["args"])  # → Temporal Activity
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return messages
```

**That's it.** The code above is identical to standard LangChain — except it cannot fail permanently. Every `ainvoke()` call is now a durable Temporal Activity with automatic retries, heartbeating, and state checkpointing.

> **The LLM is stochastic and decides everything.**
> duralang does not change that.
> It just makes sure whatever the LLM decides to do cannot fail permanently.
>
> **Nondeterminism in the model. Durability in Temporal.**

---

## What Happens When Something Fails

```mermaid
graph TD
    A["① llm.ainvoke()"] -->|"✓ success"| B["state checkpointed"]
    B --> C["② tool.ainvoke()"]
    C -->|"✗ timeout"| D["automatic retry"]
    D -->|"backoff + heartbeat"| E["② tool.ainvoke()"]
    E -->|"✓ retry succeeds"| F["state checkpointed"]
    F --> G["③ llm.ainvoke()"]
    G -->|"✓ success"| H["✅ DONE"]

    style A fill:#22c55e,color:#fff
    style B fill:#16a34a,color:#fff
    style C fill:#ef4444,color:#fff
    style D fill:#f59e0b,color:#fff
    style E fill:#22c55e,color:#fff
    style F fill:#16a34a,color:#fff
    style G fill:#22c55e,color:#fff
    style H fill:#3b82f6,color:#fff
```

**Only the failed operation retries.** Step ① is replayed from Temporal's event history — not re-executed. No wasted API calls. No wasted money. No lost progress.

### Process Crash Recovery

If the entire worker process dies (OOM, hardware failure, deployment), Temporal still holds the workflow. Restart the process, and execution resumes from the exact point of failure:

```bash
# First run — crashes at step 4
python examples/crash_recovery.py --crash
# Process killed ☠️

# Second run — resumes from checkpoint, steps 1-3 NOT re-executed
python examples/crash_recovery.py --crash
# ✓ Completed (no LLM calls re-made, no money wasted)
```

See [`crash_recovery.py`](examples/crash_recovery.py) for the full working demo.

---

## Features

### 🧭 Durable Stochastic Workflows

The model decides the path at runtime, and **every chosen step is durable.** There is no predefined graph and no hardcoded route. Whatever branch the LLM takes is captured as durable execution state in Temporal's event history.

```python
@dura
async def research_agent(messages):
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools([web_search, calculator])

    while True:  # ← No fixed graph. LLM decides the path.
        response = await llm_with_tools.ainvoke(messages)  # → durable
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            result = await tools_by_name[tc["name"]].ainvoke(tc["args"])  # → durable
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return messages
```

---

### 🤖 Durable Multi-Agent Systems

Sub-agents run as **independent durable units** with their own Temporal event history, retry boundaries, and timeouts. Two patterns are supported:

**Pattern 1: Direct calls** — Your code decides which agents to call:

```python
@dura
async def researcher(query: str) -> str:
    """Research agent with web search tools."""
    ...

@dura
async def orchestrator(task: str) -> str:
    # @dura calling @dura → Temporal Child Workflow automatically
    research = await researcher(f"Research: {task}")
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    response = await llm.ainvoke([HumanMessage(content=f"Summarize: {research}")])
    return response.content
```

**Pattern 2: Agent tools** — The LLM decides which agents to call:

```python
from duralang import dura, dura_agent_tool

all_tools = [
    dura_agent_tool(researcher),   # sub-agent (own event history)
    dura_agent_tool(analyst),      # sub-agent (own event history)
    calculator,                     # regular tool
]

@dura
async def orchestrator(task: str) -> str:
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools(all_tools)  # mix agents + tools freely
    tools_by_name = {t.name: t for t in all_tools}

    messages = [HumanMessage(content=task)]
    while True:
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            # Same ainvoke() for agents and tools. Routing is automatic.
            result = await tools_by_name[tc["name"]].ainvoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return response.content
```

**`dura_agent_tool()`** wraps any `@dura` function as a real LangChain `BaseTool` — auto-generating the Pydantic schema from the function signature, type hints, and docstring. The LLM sees a flat tool list. duralang routes each call to the right Temporal primitive.

```
orchestrator
├── llm.ainvoke()                       ← durable
├── researcher (independent sub-agent)
│    ├── llm.ainvoke()                  ← durable
│    └── web_search.ainvoke()           ← durable
├── analyst (independent sub-agent)
│    ├── llm.ainvoke()                  ← durable
│    └── calculator.ainvoke()           ← durable
└── llm.ainvoke()                       ← durable
```

If the analyst crashes, **only the analyst retries**. The researcher's completed work is preserved. Nesting works to any depth.

---

### 🕸️ Native MCP Support

MCP (Model Context Protocol) servers are first-class citizens. Wrap once with `DuraMCPSession`, and every `call_tool()` becomes a durable Temporal Activity:

```python
from duralang import dura, DuraMCPSession
from mcp import ClientSession

async with ClientSession(read, write) as session:
    await session.initialize()
    fs = DuraMCPSession(session, "filesystem")  # ← one line

    @dura
    async def my_agent(messages):
        result = await fs.call_tool("read_file", {"path": "/tmp/data.csv"})
        # ← intercepted, durable (retried, heartbeated)
        return result
```

`DuraMCPSession` proxies all methods to the underlying session. Only `call_tool()` is intercepted — everything else (like `list_tools()`) passes through normally.

---

### 🧱 Durability Stack

Every operation gets the full durability stack — not as separate features, but as one integrated system:

| Layer | What It Does | Default |
|---|---|---|
| **Retries** | Exponential backoff on transient failures | 3 attempts, 2× backoff |
| **Timeouts** | Bounded execution per operation | 10 min (LLM), 2 min (tool), 5 min (MCP) |
| **Heartbeating** | Detects hung operations | 5 min (LLM), 30s (tool/MCP) |
| **State** | Durable checkpoint after each operation | Automatic via Temporal history |

Non-retryable errors (e.g., `ValueError`, `TypeError`) fail immediately. Transient errors (timeouts, rate limits, network failures) are retried automatically with backoff.

All defaults are configurable per activity type:

```python
from datetime import timedelta
from temporalio.common import RetryPolicy
from duralang import dura, DuraConfig, ActivityConfig

config = DuraConfig(
    task_queue="agents-prod",
    llm_config=ActivityConfig(
        start_to_close_timeout=timedelta(minutes=3),
        heartbeat_timeout=timedelta(seconds=30),
        retry_policy=RetryPolicy(maximum_attempts=5),
    ),
    tool_config=ActivityConfig(
        start_to_close_timeout=timedelta(minutes=1),
        retry_policy=RetryPolicy(maximum_attempts=4),
    ),
)

@dura(config=config)
async def my_agent(messages):
    ...
```

---

### 💓 Heartbeating

Hung operations are detected automatically. Every LLM call, tool call, and MCP call emits heartbeat signals while running. If heartbeats stop (dead process, infinite loop, stuck API), Temporal marks the operation as unhealthy and reschedules it.

This matters for LLM calls that can take 30–120 seconds — a timeout alone can't distinguish between "still thinking" and "stuck." Heartbeats can.

---

### 🔍 Built-in Observability

Every execution is fully inspectable in the Temporal UI at `http://localhost:8233`:

- **Per-call timeline:** See every LLM call, tool call, and MCP call with inputs, outputs, latency, and attempt number
- **Retry history:** See exactly which calls failed, when, and how many attempts were needed
- **Workflow hierarchy:** Parent → child workflow nesting is visible as a tree
- **State progression:** Full durable state at every checkpoint
- **Replayable:** Temporal's event history is a deterministic record of the entire execution

---

### 🌐 Model-Agnostic

duralang works with any LangChain-compatible `BaseChatModel`. Same code, any provider:

| Provider | Class | Status |
|---|---|---|
| Anthropic | `ChatAnthropic` | ✅ Supported |
| OpenAI | `ChatOpenAI` | ✅ Supported |
| Google | `ChatGoogleGenerativeAI` | ✅ Supported |
| Ollama | `ChatOllama` | ✅ Supported |

Switch providers by changing one line. duralang automatically detects the provider and handles everything needed to make it durable.

---

### ⚡ Parallel Tool Execution

If the LLM returns multiple tool calls, execute them in parallel. `asyncio.gather` works as expected — each call becomes its own Temporal Activity, scheduled concurrently:

```python
@dura
async def my_agent(messages):
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools(tools)

    while True:
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break

        # Parallel execution — each call is an independent durable Activity
        tasks = [tools_by_name[tc["name"]].ainvoke(tc["args"]) for tc in response.tool_calls]
        results = await asyncio.gather(*tasks)

        for tc, result in zip(response.tool_calls, results):
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return messages
```

---

## Compared to Alternatives

**duralang vs LangGraph** — they solve different problems:

| | LangGraph | duralang |
|---|---|---|
| **Execution** | Graph nodes + edges | Free-form async loops |
| **Durability** | Per-node checkpoint | Per-operation (`ainvoke()` level) |
| **Code change** | Restructure into graph | Add `@dura` |
| **Recovery** | Re-execute entire node | Retry only the failed call |
| **Best for** | Known workflow topology | Stochastic, LLM-driven loops |

**duralang vs Temporal directly** — duralang is built on Temporal, but eliminates the boilerplate: no manual workflow/activity definitions, no custom serializers, no worker lifecycle management. You get Temporal's full power behind `@dura`.

---

## How It Works

You write normal LangChain code. duralang intercepts it transparently.

When you `import duralang`, it patches `BaseChatModel` and `BaseTool` so that every instance created afterward has its `ainvoke()` method wrapped. Inside a `@dura` function, those calls are routed to Temporal. Outside `@dura`, they work exactly as normal LangChain.

```python
@dura
async def my_agent(task):
    llm = ChatAnthropic(model="claude-sonnet-4-6")       # normal LangChain
    llm_with_tools = llm.bind_tools(all_tools)             # normal LangChain

    response = await llm_with_tools.ainvoke(messages)      # ← intercepted, durable
    result = await tools_by_name[tc["name"]].ainvoke(args)  # ← intercepted, durable
```

That's the entire mental model:

- **`@dura`** on your function → makes it a Temporal Workflow
- **`ainvoke()` calls inside** → each becomes a retryable, checkpointed Temporal Activity
- **`@dura` calling `@dura`** → becomes a Child Workflow with its own state
- **Remove `@dura`** → everything runs as vanilla LangChain

For the full architecture (proxy mechanism, serialization, activity internals), see [Architecture](docs/architecture.md).

---

## Quickstart

### 1. Install duralang

```bash
pip install duralang
```

With a specific LLM provider:

```bash
pip install "duralang[anthropic]"   # or openai, google, ollama, all-models
```

### 2. Start Temporal

duralang requires a running Temporal server. Fastest setup via [Temporal CLI](https://docs.temporal.io/cli):

```bash
# Install Temporal CLI (macOS)
brew install temporal

# Start the development server (includes UI at localhost:8233)
temporal server start-dev
```

### 3. Write your agent

```python
import asyncio
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, ToolMessage
from duralang import dura

tools = [TavilySearchResults(max_results=3)]
tools_by_name = {t.name: t for t in tools}

@dura
async def research_agent(messages: list) -> list:
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools(tools)

    while True:
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            result = await tools_by_name[tc["name"]].ainvoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return messages

async def main():
    result = await research_agent([HumanMessage(content="What is the weather in NYC?")])
    print(result[-1].content)

asyncio.run(main())
```

### 4. Inspect in Temporal UI

Open `http://localhost:8233` to see the full execution timeline — every LLM call, tool call, retry, latency, and input/output payload.

---

## Examples

The [`examples/`](examples/) directory contains runnable demos:

| Example | What It Shows |
|---|---|
| [`basic_agent.py`](examples/basic_agent.py) | Standard LangChain agent with `@dura` |
| [`multi_tool.py`](examples/multi_tool.py) | Parallel tool execution with `asyncio.gather` |
| [`multi_model.py`](examples/multi_model.py) | Same agent code with different LLM providers |
| [`multi_agent.py`](examples/multi_agent.py) | Direct `@dura` → `@dura` calls and `dura_agent_tool()` |
| [`stochastic_agents.py`](examples/stochastic_agents.py) | Fully stochastic orchestrator with mixed agent/tool dispatch |
| [`multiagent_system.py`](examples/multiagent_system.py) | Multi-agent pipeline: research → analyze → write |
| [`mcp_agent.py`](examples/mcp_agent.py) | MCP filesystem server with `DuraMCPSession` |
| [`crash_recovery.py`](examples/crash_recovery.py) | Automatic retry + process crash recovery demo |
| [`human_in_loop.py`](examples/human_in_loop.py) | Human-in-the-loop pattern (v2 preview) |

---

## Architecture

```
duralang/
├── __init__.py              # Exports: dura, dura_agent_tool, DuraConfig, DuraMCPSession
├── decorator.py             # @dura — the entire public API
├── proxy.py                 # DuraLLMProxy, DuraToolProxy, DuraMCPProxy
├── agent_tool.py            # dura_agent_tool() — wraps @dura as BaseTool
├── context.py               # DuraContext — ContextVar-based workflow context
├── workflow.py              # DuraLangWorkflow — Temporal workflow definition
├── runner.py                # DuraRunner — Temporal client + worker lifecycle
├── activities/
│   ├── llm.py               # dura__llm — LLM inference activity
│   ├── tool.py              # dura__tool — tool execution activity
│   └── mcp.py               # dura__mcp — MCP call activity
├── graph_def.py             # Payload/Result dataclasses for Temporal
├── state.py                 # MessageSerializer + ArgSerializer
├── config.py                # DuraConfig, ActivityConfig, LLMIdentity
├── registry.py              # ToolRegistry, MCPSessionRegistry
├── exceptions.py            # Exception hierarchy
└── cli.py                   # duralang CLI (worker management)
```

---

## API Reference

### `@dura`

The primary public API. Decorates an async function to make it durable.

```python
@dura
async def my_agent(messages): ...

@dura(config=DuraConfig(...))
async def my_agent(messages): ...
```

- Supports `@dura` (no parentheses) and `@dura(config=...)` (with config)
- Functions must be `async`, module-level, and importable
- When called from within another `@dura` function → becomes a Child Workflow
- When called from normal code → starts a new Temporal Workflow

### `dura_agent_tool(fn)`

Wraps a `@dura` function as a LangChain `BaseTool`.

```python
tool = dura_agent_tool(researcher)
tool = dura_agent_tool(researcher, name="search", description="Search the web.")
```

- Auto-generates Pydantic `args_schema` from function signature
- Returns a real `BaseTool` — compatible with `bind_tools()` and `ainvoke()`
- Sub-agent calls get their own event history and retry boundaries

### `DuraMCPSession(session, server_name)`

Wraps an MCP `ClientSession` to enable durable `call_tool()`.

```python
fs = DuraMCPSession(session, "filesystem")
result = await fs.call_tool("read_file", {"path": "/tmp/data.csv"})
```

### `DuraConfig`

Top-level configuration.

```python
config = DuraConfig(
    temporal_host="localhost:7233",
    temporal_namespace="default",
    task_queue="duralang",
    max_iterations=50,
    child_workflow_timeout=timedelta(hours=1),
    llm_config=ActivityConfig(...),
    tool_config=ActivityConfig(...),
    mcp_config=ActivityConfig(...),
)
```

### `ActivityConfig`

Per-activity type configuration.

```python
config = ActivityConfig(
    start_to_close_timeout=timedelta(minutes=5),
    heartbeat_timeout=timedelta(seconds=30),
    retry_policy=RetryPolicy(
        initial_interval=timedelta(seconds=1),
        backoff_coefficient=2.0,
        maximum_attempts=3,
        non_retryable_error_types=["ValueError", "TypeError"],
    ),
)
```

---

## Documentation

**Start here:**

- [Getting Started](docs/getting-started.md) — Installation, prerequisites, first agent
- [Core Concepts](docs/core-concepts.md) — The three layers, DuraContext, LLMIdentity, agent tools
- [Architecture](docs/architecture.md) — Full system diagrams, request flows, module dependencies

**Reference:**

- [API Reference](docs/api-reference.md) — Complete API documentation
- [Configuration](docs/configuration.md) — DuraConfig, ActivityConfig, retry policies
- [Activities](docs/activities.md) — dura__llm, dura__tool, dura__mcp internals
- [Tools & MCP](docs/tools-and-mcp.md) — Tool types, MCP integration, routing
- [Error Handling](docs/error-handling.md) — Exception hierarchy, retry vs non-retry errors

**Examples & help:**

- [Examples](docs/examples.md) — Walkthroughs for every example
- [FAQ](docs/faq.md) — Troubleshooting and common questions

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and PR guidelines.

## License

MIT — see [LICENSE](LICENSE).
