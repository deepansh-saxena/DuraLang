# FAQ

Answers to common questions about DuraLang — how it works, when to use it, and how to troubleshoot issues.

---

## General

### What does `@dura` actually do?

It wraps your async function so that when called, it starts a Temporal Workflow. Inside the workflow, proxy objects intercept `llm.ainvoke()` and `tool.ainvoke()` — routing each call through a Temporal Activity with automatic retries, heartbeating, and state checkpointing. Outside of a `@dura` function, LangChain works exactly as normal — the proxies check for context and take the pass-through path.

### Do I need to change my LangChain code?

No. You add `@dura` to your function, use `dura_agent()` instead of `create_agent()`, and `from duralang import dura, dura_agent`. Everything else stays the same — same tool loop, same message handling, same `ainvoke()` calls. For multi-agent systems, pass `@dura` functions directly in the `tools` list — `dura_agent()` wraps them automatically.

### Does `@dura` work with any LLM provider?

Yes. Any LangChain-compatible `BaseChatModel` works:

| Provider | Class | Status |
|---|---|---|
| Anthropic | `ChatAnthropic` | ✅ Supported |
| OpenAI | `ChatOpenAI` | ✅ Supported |
| Google | `ChatGoogleGenerativeAI` | ✅ Supported |
| Ollama | `ChatOllama` | ✅ Supported |

DuraLang extracts `LLMIdentity` (provider + model + kwargs) from the instance and reconstructs a fresh LLM inside the Activity. The LLM object itself never crosses the Temporal boundary.

### What happens if my agent crashes mid-execution?

Temporal replays the workflow from its event history. **Completed activities are not re-executed** — they are replayed from the stored results. Only the currently running or failed activity is retried. This means:

- No wasted LLM API calls
- No wasted money
- No duplicate tool side-effects (for idempotent tools)
- Execution resumes from the exact point of failure

### What's the difference between `@dura` and `dura_agent()`?

`@dura` is the decorator that makes a function durable — it wraps the function as a Temporal Workflow. `dura_agent()` is the factory that creates a LangChain agent with durable model and tool wrappers.

Use `@dura` on every agent function. Use `dura_agent()` inside `@dura` functions to create agents with automatic durable dispatch. Pass `@dura` functions directly in the `tools` list to use sub-agents as tools.

### Is duralang a replacement for LangChain?

No. duralang is a **complement** to LangChain. You keep using LangChain for LLM calls, tools, messages, and agent loops. duralang adds the durability layer underneath — intercepting calls transparently and routing them through Temporal. If you remove `@dura`, your code runs as standard LangChain.

### Is duralang a replacement for LangGraph?

No. They solve different problems. LangGraph is for designing explicit graph topologies with defined nodes and edges. duralang is for making free-form agent loops durable without restructuring your code. If your workflow has a known, designable graph shape, LangGraph is the right tool. If your agent is fully stochastic (LLM decides the path), duralang gives you per-operation durability without a graph.

### Do I need to learn Temporal?

Not to use duralang. The `@dura` decorator abstracts away all Temporal concepts. You never write `@workflow.defn`, `@activity.defn`, or set up workers manually. You do need a running Temporal server (`temporal server start-dev`), and the Temporal UI is useful for observability, but no Temporal knowledge is required to use duralang.

---

## Architecture

### How does interception work?

`dura_agent()` wraps your model with `DuraModel` and your tools with `DuraTool` — these are `BaseChatModel` and `BaseTool` subclasses that check `DuraContext.get()` on every call:

- `None` → outside `@dura` → calls the original method (standard LangChain behavior)
- Context exists → inside `@dura` → routes to the appropriate Temporal Activity

The per-call cost is a single `ContextVar` lookup (nanoseconds).

### What is DuraContext?

A `contextvars.ContextVar` that carries workflow state — the workflow ID, config, and callback functions for scheduling activities and child workflows. It's set by `DuraLangWorkflow` before calling your function and read by proxy objects on every interception.

`ContextVar` is the correct mechanism for async Python — `asyncio` tasks propagate ContextVars, so concurrent activities within the same workflow all see the correct context.

### How do multi-agent calls work?

Two mechanisms:

1. **Direct calls:** If a `@dura` function calls another `@dura` function, the wrapper detects the existing `DuraContext` and routes to `workflow.execute_child_workflow()` — a Temporal Child Workflow with its own event history.

2. **Agent tools:** `dura_agent()` auto-wraps `@dura` functions as `BaseTool` instances. When called via `ainvoke()`, `DuraTool` detects the `__dura_agent_tool__` flag and calls the `@dura` function directly — which routes as a Child Workflow (same as pattern 1).

### Can I mix agent tools, regular tools, and MCP in the same agent?

Yes. Pass `@dura` functions, `@tool` functions, MCP tools (via `langchain-mcp-adapters`), and `BaseTool` instances in the same `tools` list — `dura_agent()` wraps each automatically. The routing to the correct Temporal primitive (Child Workflow or `dura__tool` Activity) happens automatically:

```python
from langchain_mcp_adapters.tools import load_mcp_tools

mcp_tools = await load_mcp_tools(session)  # MCP tools as BaseTool instances

all_tools = [
    researcher,    # @dura → Child Workflow
    writer,        # @dura → Child Workflow
    calculator,    # @tool → dura__tool Activity
    *mcp_tools,    # MCP → dura__tool Activity (same as any BaseTool)
]
```

### Do nested agent calls work?

Yes, to any depth. Each `@dura` function at any nesting level gets its own Temporal workflow with its own event history. If agent A calls agent B, and agent B calls agent C, each level is independently durable. A failure in C retries only C — A and B are unaffected.

Child workflow IDs are deterministic and prevent unbounded ID growth:

```
root-workflow-id--child--agent_b-1--run_id_prefix
root-workflow-id--child--agent_c-1--run_id_prefix
```

### What serialization formats are used?

Everything crossing a Temporal boundary must be JSON-serializable. DuraLang handles this with two serializers:

| Data | Serialization |
|---|---|
| LangChain messages | `MessageSerializer`: message class, content, tool_calls, tool_call_id, name, id → JSON dict |
| LLM instances | `LLMIdentity`: provider + model + kwargs → JSON dict. Reconstructed on the other side. |
| Function arguments | `ArgSerializer`: primitives pass through, lists/dicts recurse, tuples get tagged, messages get tagged |
| Tool instances | Not serialized. Registered by name in `ToolRegistry`. Looked up by name inside activities. |

---

## Troubleshooting

### "Tool 'X' not in registry" error

The `dura__tool` Activity couldn't find the tool by name in `ToolRegistry`. This means the tool wasn't created before or inside your `@dura` function.

**Fix:** Ensure the tool is instantiated at module level or inside the `@dura` function before it's used:

```python
# ✓ Module-level — registered at import time
tools = [TavilySearchResults(max_results=3)]

@dura
async def my_agent(messages):
    llm.bind_tools(tools)
    ...
```

### "Cannot determine LLM provider" error

DuraLang needs to identify your LLM provider to reconstruct it inside the Activity. This error means you're using a `BaseChatModel` subclass that DuraLang doesn't recognize.

**Supported:** `ChatAnthropic`, `ChatOpenAI`, `ChatGoogleGenerativeAI`, `ChatOllama`

### "not a @dura function" error when passing a function as a tool

Functions passed in the `tools` list to `dura_agent()` that have the `__dura__` flag must be decorated with `@dura`. If you pass a plain async function that looks like an agent, decorate it first:

```python
# ✗ Wrong — not decorated
async def my_agent(query: str) -> str: ...

# ✓ Correct — decorated
@dura
async def my_agent(query: str) -> str: ...

# Then pass directly to dura_agent
agent = dura_agent(model="claude-sonnet-4-6", tools=[my_agent])
```

### "@dura cannot wrap lambda functions"

`@dura` functions must be importable by the Temporal worker via `module:function_name`. Lambdas and closures cannot be resolved by import path.

**Fix:** Define your function at module top level:

```python
# ✗ Won't work
agent = dura(lambda messages: ...)

# ✓ Works
@dura
async def my_agent(messages): ...
```

### My function works outside `@dura` but fails inside

Check these three things:

1. **All arguments are serializable** — Only primitives (`str`, `int`, `float`, `bool`, `None`), lists, dicts, and LangChain messages are supported. Custom objects or dataclasses will raise `StateSerializationError`.

2. **Function is at module level** — Not nested inside another function, class, or method.

3. **Temporal server is running** — `temporal server start-dev` must be active and reachable at the configured `temporal_host` (default: `localhost:7233`).

### Activities are timing out

The default `start_to_close_timeout` may be too short for your use case:

```python
from datetime import timedelta
from duralang import DuraConfig, ActivityConfig

config = DuraConfig(
    llm_config=ActivityConfig(
        start_to_close_timeout=timedelta(minutes=15),    # Increase for slow models
        heartbeat_timeout=timedelta(minutes=10),          # Match the timeout increase
    ),
)
```

### How do I see what's happening?

Open the Temporal UI at `http://localhost:8233`. Every workflow shows its complete activity timeline with inputs, outputs, retries, timing, and errors. This is the single best debugging tool for DuraLang agents.
