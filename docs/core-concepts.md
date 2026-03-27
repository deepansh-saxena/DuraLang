# Core Concepts

This page explains how DuraLang makes LangChain agents durable without changing your code.

---

## Three Layers

### Layer 1: The `@dura` Decorator

The decorator wraps your async function. When called:
1. Serializes function arguments
2. Starts a Temporal Workflow
3. Blocks until the workflow completes
4. Returns the deserialized result

Callers see a normal async function. They don't know Temporal is involved.

### Layer 2: Call Interception

Before your function executes inside the Temporal worker, DuraLang sets a context flag (`DuraContext`) using Python's `contextvars.ContextVar`. Every `BaseChatModel` and `BaseTool` instance has proxy methods installed at import time that check this flag:

- **Inside `@dura`**: `llm.ainvoke()` routes to a Temporal Activity (retried, heartbeated, durable)
- **Outside `@dura`**: `llm.ainvoke()` calls the original method normally (standard LangChain)

This means your code works identically with or without `@dura`.

### Layer 3: Temporal Activities

Every intercepted call maps to one of three Temporal Activities:

| What You Call | What Runs |
|---|---|
| `llm.ainvoke(messages)` | Activity: `dura__llm` |
| `tool.ainvoke(input)` | Activity: `dura__tool` |
| `mcp_session.call_tool(...)` | Activity: `dura__mcp` |
| `@dura` calling `@dura` | Child Workflow |
| `dura_agent_tool(fn).ainvoke(args)` | Child Workflow (via `BaseTool`) |

---

## DuraContext

`DuraContext` is stored in a `contextvars.ContextVar`. It carries:

- `workflow_id` — the current Temporal workflow ID
- `config` — the `DuraConfig` for this execution
- `execute_activity` — a function that schedules Temporal Activities
- `execute_child_agent` — a function that starts Child Workflows

Proxy objects call `DuraContext.get()` to decide whether to intercept or pass through. If it returns `None` (not inside `@dura`), the original method runs. If it returns a context, the call routes to Temporal.

---

## LLMIdentity

When DuraLang intercepts `llm.ainvoke()`, it extracts an `LLMIdentity` from the LLM instance — the provider name (e.g., "anthropic"), model name, and kwargs. This lightweight descriptor is what crosses the Temporal boundary. The `dura__llm` Activity reconstructs a fresh LLM instance from it.

The LLM object itself never touches Temporal's serializer.

---

## Auto-Registration

Tools are auto-registered in an internal `ToolRegistry` when proxy methods are installed. When `bind_tools()` is called and the proxy extracts schemas, it registers each `BaseTool` so `dura__tool` can find it later. You never call `register_tools()` — it happens automatically.

---

## Child Workflows

When a `@dura` function is called from within another `@dura` function, DuraLang detects the existing context and routes to a Temporal Child Workflow instead of starting a new top-level workflow. Each child gets its own event history and retry boundaries.

```python
@dura
async def researcher(query: str) -> str:
    ...  # This agent's calls are in its own event history

@dura
async def orchestrator(task: str) -> str:
    result = await researcher(task)  # ← becomes a Child Workflow
```

Child workflow IDs are deterministic — they include the parent workflow ID and function name. This nests to any depth.

---

## Agent Tools

`dura_agent_tool()` wraps a `@dura` function as a real LangChain `BaseTool`. This lets you put sub-agents and regular tools in the same list, use the same `bind_tools()`, and dispatch with the same `ainvoke()` loop.

```python
from duralang import dura, dura_agent_tool

@dura
async def researcher(query: str) -> str:
    """Research agent — gathers information."""
    ...

# Mix sub-agents and regular tools in one list
all_tools = [
    dura_agent_tool(researcher),   # → Child Workflow
    calculator,                     # → dura__tool Activity
]
```

How it works under the hood:
1. `dura_agent_tool()` reads the function's signature, type hints, and docstring
2. It generates a Pydantic `args_schema` and tool schema automatically
3. It returns a `BaseTool` whose `_arun()` calls the `@dura` function
4. The proxy skips `dura__tool` routing for agent tools — the call goes directly to the `@dura` function, which becomes a Child Workflow

The result: one dispatch pattern for everything. The LLM sees a flat list of tools. DuraLang routes each call to the right Temporal primitive.
