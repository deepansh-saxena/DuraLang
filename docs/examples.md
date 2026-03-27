# Examples

Complete runnable examples demonstrating every DuraLang capability. All examples are in the [`examples/`](../examples/) directory.

---

## Quick Reference

| Example | What It Demonstrates | Key Concept |
|---|---|---|
| [`basic_agent.py`](../examples/basic_agent.py) | Standard LangChain agent with `@dura` | One decorator, zero code change |
| [`multi_tool.py`](../examples/multi_tool.py) | Parallel tool execution | `asyncio.gather` with durable Activities |
| [`multi_model.py`](../examples/multi_model.py) | Same agent, different LLM providers | Model-agnostic durability |
| [`multi_agent.py`](../examples/multi_agent.py) | Direct `@dura` → `@dura` calls + `dura_agent_tool()` | Child Workflows, LLM-driven delegation |
| [`stochastic_agents.py`](../examples/stochastic_agents.py) | Fully stochastic orchestrator | Mixed agent/tool dispatch, runtime decisions |
| [`multiagent_system.py`](../examples/multiagent_system.py) | Multi-agent pipeline: research → analyze → write | Fixed pipeline with independent sub-agents |
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
from duralang import dura

tools = [TavilySearchResults(max_results=3)]
tools_by_name = {t.name: t for t in tools}

@dura
async def research_agent(messages: list) -> list:
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools(tools)

    while True:
        response = await llm_with_tools.ainvoke(messages)   # → dura__llm Activity
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            result = await tools_by_name[tc["name"]].ainvoke(tc["args"])  # → dura__tool Activity
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return messages
```

**What to observe in Temporal UI (`localhost:8233`):**
- Each LLM call appears as a `dura__llm` Activity with full input/output payloads
- Each tool call appears as a `dura__tool` Activity with tool name and arguments
- The workflow timeline shows the complete execution narrative

---

## Multi-Tool with Parallel Execution

**File:** [`examples/multi_tool.py`](../examples/multi_tool.py)

Demonstrates parallel tool execution with `asyncio.gather`. When the LLM returns multiple tool calls, each runs as its own Temporal Activity — scheduled concurrently:

```python
# Parallel tool execution — each becomes its own Temporal Activity
tasks = [tools_by_name[tc["name"]].ainvoke(tc["args"]) for tc in response.tool_calls]
results = await asyncio.gather(*tasks)
```

**What to observe:** In the Temporal UI, parallel activities overlap in the timeline. Both complete independently — if one fails and retries, the other's result is already checkpointed.

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

## Multi-Agent

**File:** [`examples/multi_agent.py`](../examples/multi_agent.py)

Shows both patterns for multi-agent calls:

### Pattern 1: Direct `@dura` → `@dura` calls

Your code decides which agent to call. The `@dura` decorator detects the existing context and routes as a Child Workflow:

```python
@dura
async def orchestrator(task: str) -> str:
    research = await researcher(f"Research: {task}")  # → Child Workflow
    ...
```

### Pattern 2: `dura_agent_tool()` — LLM decides

The LLM decides which agents to call. Sub-agents are wrapped as tools:

```python
all_tools = [dura_agent_tool(researcher)]  # → Child Workflow when called

@dura
async def orchestrator(task: str) -> str:
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools(all_tools)
    ...
```

**What to observe:** In the Temporal UI, child workflows appear nested under the parent. Each has its own event history, timing, and retry state.

---

## Stochastic Agents

**File:** [`examples/stochastic_agents.py`](../examples/stochastic_agents.py)

**This is the flagship example.** An orchestrator agent with both sub-agents and regular tools in the same list. The LLM freely decides:

- Which agent or tool to call
- How many times to call each
- What order to call them in
- When it has enough information to respond

```python
all_tools = [
    dura_agent_tool(researcher),       # → Temporal Child Workflow
    dura_agent_tool(analyst),          # → Temporal Child Workflow
    dura_agent_tool(writer),           # → Temporal Child Workflow
    calculator,                         # → Temporal Activity
]

@dura
async def orchestrator(task: str) -> str:
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools(all_tools)
    # ... standard dispatch loop ...
```

**What to observe:** Run this example multiple times. The execution path is different every run — but every operation is individually durable. If any call fails, only that call retries.

---

## Multi-Agent Pipeline

**File:** [`examples/multiagent_system.py`](../examples/multiagent_system.py)

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

Integrates an MCP filesystem server using `DuraMCPSession`:

```python
fs = DuraMCPSession(session, "filesystem")  # ← one line

@dura
async def fs_agent(messages, fs):
    tools_result = await fs.list_tools()           # passes through to MCP
    result = await fs.call_tool("read_file", ...)  # → dura__mcp Activity
```

**What to observe:** `dura__mcp` Activities appear in the Temporal UI with server name, tool name, and arguments.

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
