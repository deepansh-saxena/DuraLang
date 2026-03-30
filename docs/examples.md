# Examples

Complete runnable examples demonstrating every DuraLang capability. All examples are in the [`examples/`](../examples/) directory.

---

## Quick Reference

| Example | What It Demonstrates | Key Concept |
|---|---|---|
| [`basic_agent.py`](../examples/basic_agent.py) | Standard LangChain agent with `@dura` | One decorator, zero code change |
| [`multi_tool.py`](../examples/multi_tool.py) | Parallel tool execution | `asyncio.gather` with durable Activities |
| [`multi_model.py`](../examples/multi_model.py) | Same agent, different LLM providers | Model-agnostic durability |
| [`multiagent_system.py`](../examples/multiagent_system.py) | Multi-agent orchestrator with mixed agent/tool dispatch | Child Workflows, LLM-driven delegation |
| [`sequential_agents.py`](../examples/sequential_agents.py) | Sequential pipeline: research → analyze → write | Fixed pipeline with independent sub-agents |
| [`mcp_agent.py`](../examples/mcp_agent.py) | MCP filesystem server integration | `DuraMCPSession` for durable MCP calls |
| [`crash_recovery.py`](../examples/crash_recovery.py) | Automatic retry + process crash recovery | Temporal replay, deterministic workflow IDs |
| [`human_in_loop.py`](../examples/human_in_loop.py) | Human-in-the-loop pattern | Signal-based pause/resume (v2 preview) |

### Prerequisites for All Examples

```bash
# 1. Install DuraLang with Anthropic
pip install "duralang[anthropic]"

# 2. Start Temporal server
temporal server start-dev

# 3. Set your API key
export ANTHROPIC_API_KEY="sk-..."

# 4. Run any example
python examples/basic_agent.py
```

---

## Basic Agent

**File:** [`examples/basic_agent.py`](../examples/basic_agent.py)

The simplest possible DuraLang agent. A standard LangChain agent loop — the only addition is `@dura` on the function definition.

```python
from duralang import dura, dura_agent

tools = [TavilySearchResults(max_results=3)]

@dura
async def research_agent(messages: list) -> list:
    agent = dura_agent(
        model="claude-sonnet-4-6",
        tools=tools,
    )
    result = await agent.ainvoke({"messages": messages})   # → dura__llm + dura__tool Activities
    return result["messages"]
```

**What to observe in Temporal UI (`localhost:8233`):**
- Each LLM call appears as a `dura__llm` Activity with full input/output payloads
- Each tool call appears as a `dura__tool` Activity with tool name and arguments
- The workflow timeline shows the complete execution narrative

---

## Multi-Tool with Parallel Execution

**File:** [`examples/multi_tool.py`](../examples/multi_tool.py)

Demonstrates parallel tool execution. When the LLM returns multiple tool calls, `dura_agent` handles parallel dispatch automatically — each tool call runs as its own Temporal Activity, scheduled concurrently:

```python
from duralang import dura_agent

agent = dura_agent(
    model="claude-sonnet-4-6",
    tools=tools,
)
# dura_agent handles parallel tool calls automatically
result = await agent.ainvoke({"messages": messages})
```

**What to observe:** In the Temporal UI, parallel activities overlap in the timeline. Both complete independently — if one fails and retries, the other's result is already checkpointed. Note: `dura_agent` handles parallel tool dispatch automatically — no manual `asyncio.gather` needed.

---

## Multi-Model

**File:** [`examples/multi_model.py`](../examples/multi_model.py)

Same agent code, different LLM providers. Shows that DuraLang auto-detects the provider from any `BaseChatModel` instance:

```python
@dura
async def chat_agent(messages: list, provider: str = "anthropic") -> list:
    if provider == "anthropic":
        llm = ChatAnthropic(model="claude-sonnet-4-6")
    elif provider == "openai":
        llm = ChatOpenAI(model="gpt-4o")

    response = await llm.ainvoke(messages)  # → dura__llm (auto-detected provider)
    messages.append(response)
    return messages
```

**What to observe:** The `dura__llm` Activity payload shows the `LLMIdentity` with the detected provider and model. Same activity, different LLM behind it.

---

## Multi-Agent System

**File:** [`examples/multiagent_system.py`](../examples/multiagent_system.py)

**This is the flagship example.** An orchestrator agent with both sub-agents and regular tools in the same list. The LLM freely decides:

- Which agent or tool to call
- How many times to call each
- What order to call them in
- When it has enough information to respond

```python
all_tools = [
    researcher,    # @dura → Child Workflow (auto-wrapped by dura_agent)
    analyst,       # @dura → Child Workflow (auto-wrapped by dura_agent)
    writer,        # @dura → Child Workflow (auto-wrapped by dura_agent)
    calculator,    # @tool → dura__tool Activity (auto-wrapped by dura_agent)
]

@dura
async def orchestrator(task: str) -> str:
    agent = dura_agent(
        model="claude-sonnet-4-6",
        tools=all_tools,
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})
    return result["messages"][-1].content
```

**What to observe:** Run this example multiple times. The execution path is different every run — but every operation is individually durable. If any call fails, only that call retries. In the Temporal UI, child workflows appear nested under the parent, each with its own event history.

---

## Sequential Agents

**File:** [`examples/sequential_agents.py`](../examples/sequential_agents.py)

A fixed three-stage pipeline where each stage is a `@dura` agent with its own tools:

```
pipeline_agent → researcher → analyst → writer
```

Each sub-agent has its own tools and Temporal event history. If the analyst fails, only the analyst retries — the researcher's completed work is preserved.

```python
@dura
async def pipeline_agent(topic: str) -> str:
    research = await researcher(topic)                    # Child Workflow
    analysis = await analyst(research, question=...)      # Child Workflow
    report = await writer(research, analysis, topic)      # Child Workflow
    return report
```

---

## MCP Agent

**File:** [`examples/mcp_agent.py`](../examples/mcp_agent.py)

Integrates an MCP filesystem server using `langchain-mcp-adapters`. MCP tools are converted to standard `BaseTool` instances and passed to `dura_agent()` — they go through `DuraTool` and the `dura__tool` Activity like any other tool:

```python
from langchain_mcp_adapters.tools import load_mcp_tools

async with mcp_server("filesystem", args=["path/to/dir"]) as session:
    mcp_tools = await load_mcp_tools(session)

    @dura
    async def fs_agent(messages: list) -> list:
        agent = dura_agent(
            model="claude-sonnet-4-6",
            tools=mcp_tools,       # MCP tools as BaseTool → dura__tool Activity
        )
        result = await agent.ainvoke({"messages": messages})
        return result["messages"]
```

**What to observe:** MCP tool calls appear as `dura__tool` Activities in the Temporal UI — same as any other tool.

---

## Crash Recovery

**File:** [`examples/crash_recovery.py`](../examples/crash_recovery.py)

**The most impressive demo.** Shows two failure modes and how DuraLang handles each:

### Mode 1: Automatic Retry

A flaky tool (`get_stock_price`) fails on its first attempt with a `TimeoutError`. Temporal retries it automatically with backoff. The completed LLM call is NOT re-executed:

```bash
python examples/crash_recovery.py
```

### Mode 2: Process Crash Recovery

The worker process is killed mid-execution (via `os._exit(1)`). The Temporal server still holds the workflow. Re-run the script — Temporal replays all completed steps from event history and continues from the exact point of failure:

```bash
# First run — crashes mid-execution
python examples/crash_recovery.py --crash
# Process killed ☠️

# Second run — Temporal resumes from checkpoint
python examples/crash_recovery.py --crash
# ✓ Completed (completed steps NOT re-executed)
```

The demo uses a fixed `_workflow_id` so the second run reconnects to the existing workflow:

```python
result = await market_analyst(
    [HumanMessage(content=prompt)],
    _workflow_id=WORKFLOW_ID if crash_mode else None,
)
```

**What to observe:** On the second run, you'll see output confirming that completed LLM calls were replayed from history — no API calls made, no money wasted.

```bash
# Clean up after the demo
python examples/crash_recovery.py --clean
```

---

## Human-in-the-Loop

**File:** [`examples/human_in_loop.py`](../examples/human_in_loop.py)

Preview of the v2 human-in-the-loop pattern using Temporal Signals. Currently demonstrates the pattern structure — full implementation is planned for v2.

See [Human-in-the-Loop](human-in-the-loop.md) for details on the design.
