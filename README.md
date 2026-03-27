# DuraLang

**Durable LangChain. One decorator.**

---

## The Problem: LLM Agents Are Stochastic

LLM agents are fundamentally different from traditional software. Every call to an LLM is **non-deterministic** — the same prompt can produce different outputs, trigger different tool calls, and take wildly different amounts of time. This makes agent systems inherently unpredictable:

- An LLM call **times out** after running for 90 seconds on a complex reasoning step
- A tool call **fails** because an external API is down
- The LLM returns a **malformed tool call** that crashes your parsing logic
- Your worker **dies** mid-execution — 15 LLM calls and 8 tool calls, all lost
- You hit a **rate limit** on call 12 of a 20-call agent loop

Traditional software handles this with try/catch and retry logic. But LLM agents are **multi-step, stateful, and long-running**. A research agent might make 30+ LLM calls, each one a potential failure point. Writing manual retry and checkpointing logic for every step is untenable — and it buries your agent logic under infrastructure code.

### Why a graph doesn't solve this

Frameworks like LangGraph give you a **deterministic graph with stochastic nodes**. You define the path upfront — node A goes to node B or C based on a condition you wrote. Checkpointing happens at graph-node boundaries.

But real agent behavior isn't a fixed graph. **The LLM decides what happens next** — which tools to call, how many times to loop, whether to delegate to another agent. The flow itself is stochastic. Forcing this into a graph means either you over-specify edges to cover every possible path, or you shove all the real logic into one giant node and lose granular durability.

```
LangGraph approach:
  [Node A] → edge → [Node B] → edge → [Node C]
  Checkpoints at node boundaries. The path is fixed.
  Stochasticity is trapped inside each node.

  Problem: What if the LLM decides to call 3 tools, then loop twice,
  then delegate to another agent? You'd need a graph edge for every
  possible decision. The graph becomes the complexity you were avoiding.

DuraLang approach:
  Your code IS the control flow. The LLM decides at runtime.
  Every individual LLM call and tool call is independently durable.
  No graph to define. No edges to maintain.
```

**The core insight:** LLM calls are like microservice calls — unreliable, variable-latency, and side-effectful. The infrastructure world solved this problem years ago with **durable execution**. DuraLang brings that same solution to LangChain — without forcing your agent logic into a graph.

---

## The Solution: Durable Execution for LangChain

DuraLang makes every operation in your agent **individually durable**. Each LLM call, tool call, and agent-to-agent call is recorded in Temporal's event history. If anything fails at any point, execution resumes from the last successful step — not from the beginning.

```
Without DuraLang:
  LLM call 1 ✓ → Tool call ✓ → LLM call 2 ✓ → Tool call ✗ CRASH
  Result: Everything lost. Start over.

With DuraLang:
  LLM call 1 ✓ → Tool call ✓ → LLM call 2 ✓ → Tool call ✗ CRASH
  Worker restarts...
  LLM call 1 (replayed) → Tool call (replayed) → LLM call 2 (replayed) → Tool call ✓ RETRY
  Result: Resumed from where it left off. Zero wasted LLM calls.
```

And you get this by adding **one decorator**:

```python
from duralang import dura

@dura
async def my_agent(messages):
    ...  # your exact same LangChain code
```

Your code doesn't change. DuraLang intercepts LLM and tool calls at the method level and routes them through Temporal Activities — automatically retried, heartbeated, and observable.

---

## Quick Start

```bash
pip install "duralang[anthropic]"
temporal server start-dev  # Start local Temporal
```

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
            result = await tools_by_name[tc["name"]].arun(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return messages

async def main():
    result = await research_agent([
        HumanMessage(content="What are the latest developments in AI agents?")
    ])
    print(result[-1].content)

asyncio.run(main())
```

**The code above is identical LangChain code.** The only addition is `@dura`.

---

## What DuraLang Handles For You

| Failure | Without DuraLang | With DuraLang |
|---|---|---|
| LLM call times out | Agent crashes, all state lost | Temporal retries with backoff |
| Tool throws an error | Unhandled exception kills the run | Temporal retries or returns error gracefully |
| Worker crashes mid-run | Start over from scratch | Temporal replays from last completed step |
| Rate limited on call 15 of 20 | You write retry logic by hand | Temporal backs off automatically |
| Tool hangs forever | Agent hangs forever | Heartbeat detects it, Temporal retries |
| LLM returns malformed output | Crash, no recovery | Temporal retries the LLM call |

---

## How It Works

| What You Write | What Happens Under the Hood |
|---|---|
| `llm.ainvoke(messages)` | Temporal Activity: `dura__llm` |
| `tool.arun(input)` | Temporal Activity: `dura__tool` |
| `session.call_tool(...)` | Temporal Activity: `dura__mcp` |
| `@dura` calling `@dura` | Temporal Child Workflow |

DuraLang intercepts calls at the method level using proxy objects. Outside a `@dura` function, LangChain works exactly as normal. Inside `@dura`, every LLM, tool, and MCP call is independently:

- **Retryable** — with configurable retry policies
- **Observable** — visible in the Temporal UI with input/output
- **Timeoutable** — with per-activity timeouts and heartbeats
- **Durable** — survives worker crashes via Temporal replay

---

## Multi-Agent — Durability All the Way Down

`@dura` functions calling other `@dura` functions become Temporal Child Workflows. This nests to any depth — and **every level is independently durable**.

```python
@dura
async def search_agent(query: str) -> str:
    """Leaf agent — does web search + LLM summary."""
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    results = await search_tool.arun(query)           # dura__tool activity
    response = await llm.ainvoke([...])                # dura__llm activity
    return response.content

@dura
async def researcher(topic: str) -> str:
    """Mid-level agent — delegates to search agents."""
    background = await search_agent(f"{topic} background")    # child workflow
    recent = await search_agent(f"{topic} recent news")       # child workflow
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    return (await llm.ainvoke([...])).content                  # dura__llm activity

@dura
async def orchestrator(task: str) -> str:
    """Top-level agent — coordinates researchers."""
    research = await researcher(task)                  # child workflow (which spawns its own children)
    analysis = await analyst(research)                 # another child workflow
    return await writer(research, analysis)            # another child workflow
```

What Temporal sees:

```
orchestrator (workflow)
├── researcher (child workflow)                    ← own event history, own retries
│   ├── search_agent "background" (grandchild)     ← own event history, own retries
│   │   ├── dura__tool: search
│   │   └── dura__llm
│   └── search_agent "recent" (grandchild)         ← own event history, own retries
│       ├── dura__tool: search
│       └── dura__llm
│   └── dura__llm
├── analyst (child workflow)
│   └── ...
└── writer (child workflow)
    └── ...
```

If `search_agent("recent news")` fails, only that grandchild retries. The orchestrator, researcher, and the completed `search_agent("background")` are untouched. Every `@dura` boundary is a durability boundary.

---

## Features

- **Zero API** — `@dura` is the only new concept
- **Transparent interception** — LLM, tool, and MCP calls routed through Temporal automatically
- **Automatic retries** — transient failures retried with configurable backoff
- **Heartbeating** — long-running calls monitored, hung operations detected
- **Crash recovery** — Temporal replays from last completed step on worker restart
- **Parallel tool calls** — `asyncio.gather` in user code works correctly
- **Multi-agent** — `@dura` calling `@dura` = Temporal Child Workflows
- **Model-agnostic** — Anthropic, OpenAI, Google, Ollama
- **MCP support** — MCP servers via `DuraMCPSession`
- **Full observability** — every activity visible in the Temporal UI

---

## Documentation

| Topic | Link |
|---|---|
| Getting Started | [docs/getting-started.md](docs/getting-started.md) |
| Core Concepts | [docs/core-concepts.md](docs/core-concepts.md) |
| Configuration | [docs/configuration.md](docs/configuration.md) |
| Activities | [docs/activities.md](docs/activities.md) |
| Tools & MCP | [docs/tools-and-mcp.md](docs/tools-and-mcp.md) |
| Human-in-the-Loop | [docs/human-in-the-loop.md](docs/human-in-the-loop.md) |
| Error Handling | [docs/error-handling.md](docs/error-handling.md) |
| API Reference | [docs/api-reference.md](docs/api-reference.md) |
| Examples | [docs/examples.md](docs/examples.md) |
| FAQ | [docs/faq.md](docs/faq.md) |

---

## Requirements

- Python 3.11+
- Temporal Server (local dev or cloud)
- LLM API key (Anthropic, OpenAI, Google, or Ollama)

## License

MIT
