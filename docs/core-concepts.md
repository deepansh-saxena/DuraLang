# Core Concepts

This page explains the foundational ideas behind DuraLang — how it makes LangChain agents durable without changing your code, and what happens at each layer of the system.

---

## The Three Layers

DuraLang operates in three layers. Each layer is invisible to the one above it — your code never knows it's being intercepted.

### Layer 1: The `@dura` Decorator

The decorator wraps your async function. When called:

1. **Serializes** all function arguments using `ArgSerializer`
2. **Starts** a `DuraLangWorkflow` on the Temporal server
3. **Blocks** until the workflow completes
4. **Deserializes** and returns the result

From the caller's perspective, it's just an async function:

```python
from duralang import dura

@dura
async def my_agent(messages):
    ...  # Your unchanged LangChain code

# Called like any normal async function
result = await my_agent([HumanMessage(content="Hello")])
```

The caller doesn't know Temporal is involved. The function signature doesn't change. Return types don't change. Error handling doesn't change (except now transient failures are retried automatically).

### Layer 2: Call Interception via Proxies

Before your function body executes inside the Temporal worker, DuraLang sets a `DuraContext` flag using Python's `contextvars.ContextVar`. `dura_agent()` wraps your model and tools with durable subclasses (`DuraModel`, `DuraTool`) that check this flag on every call:

- **Inside `@dura`** → `llm.ainvoke()` routes to a Temporal Activity (retried, heartbeated, durable)
- **Outside `@dura`** → `llm.ainvoke()` calls the original method normally (standard LangChain behavior)

This is the key guarantee: **your code works identically with or without `@dura`**. Remove the decorator, and everything runs as vanilla LangChain. Add it back, and every operation becomes durable.

### Layer 3: Temporal Activities

Every intercepted call maps to one of three Temporal Activities:

| What You Call | Temporal Primitive | What Happens |
|---|---|---|
| `llm.ainvoke(messages)` | `dura__llm` Activity | LLM instance reconstructed from `LLMIdentity`, messages deserialized, inference executed, result checkpointed |
| `tool.ainvoke(input)` | `dura__tool` Activity | Tool looked up in `ToolRegistry` by name, executed, result checkpointed |
| MCP tool (via `langchain-mcp-adapters`) | `dura__tool` Activity | MCP tools converted to `BaseTool` by `langchain-mcp-adapters`, then wrapped by `dura_agent()` as `DuraTool` — same path as any other tool |
| `@dura` calling `@dura` | Child Workflow | Sub-agent gets its own event history, timeouts, and retry boundaries |
| `@dura` fn as tool (auto-wrapped) | Child Workflow | Same as above, but triggered through the LangChain `BaseTool` interface |
| `session.call_tool(...)` *(legacy)* | `dura__mcp` Activity | Legacy path via `DuraMCPProxy`. Prefer `langchain-mcp-adapters` instead |

Each activity is individually retryable, heartbeated, and checkpointed. If any operation fails, only that operation retries — everything before it is preserved in Temporal's event history.

---

## DuraContext

`DuraContext` is the bridge between your code and Temporal. It's stored in a `contextvars.ContextVar` — Python's built-in mechanism for task-local state in async code.

When `DuraLangWorkflow` starts executing your function, it sets a `DuraContext` containing:

| Field | Purpose |
|---|---|
| `workflow_id` | The current Temporal workflow ID |
| `config` | The `DuraConfig` for this execution |
| `execute_activity` | A function that schedules Temporal Activities |
| `execute_child_agent` | A function that starts Child Workflows |

Every proxy object calls `DuraContext.get()` on every method invocation:

- **Returns `None`** → no `@dura` context → call the original method (standard LangChain)
- **Returns a context** → inside `@dura` → route through Temporal

This is why DuraLang has zero runtime cost outside of `@dura` functions — the proxies check a single ContextVar and take the fast path.

---

## LLMIdentity

LLM objects (like `ChatAnthropic` or `ChatOpenAI`) can't be serialized and sent through Temporal. DuraLang solves this with `LLMIdentity` — a lightweight descriptor that captures the minimum information needed to reconstruct the LLM:

| Field | Example |
|---|---|
| `provider` | `"anthropic"`, `"openai"`, `"google"`, `"ollama"` |
| `model` | `"claude-sonnet-4-6"`, `"gpt-4o"` |
| `kwargs` | `{"temperature": 0.7}` |

When DuraLang intercepts `llm.ainvoke()`:

1. **Proxy side** (inside your workflow): Extracts `LLMIdentity` from the instance → serializes messages → sends payload to Temporal
2. **Activity side** (inside `dura__llm`): Reconstructs a fresh LLM from `LLMIdentity` → rebinds tool schemas → deserializes messages → calls the real `ainvoke()` → returns the result

The LLM object itself never touches Temporal's serializer. Only the identity crosses the boundary.

---

## Auto-Registration

DuraLang automatically registers tools so that the `dura__tool` Activity can find them by name. This happens when `DuraTool` wraps a tool — `dura_agent()` does this automatically for every tool in its `tools` list.

You never call `register_tools()`. It doesn't exist. Registration is fully automatic.

For MCP tools, use [`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters) to convert MCP server tools into standard `BaseTool` instances, then pass them to `dura_agent()` like any other tool:

```python
from langchain_mcp_adapters.tools import load_mcp_tools

mcp_tools = await load_mcp_tools(session)
agent = dura_agent(model="claude-sonnet-4-6", tools=mcp_tools)
# ↑ MCP tools are BaseTool instances — auto-wrapped as DuraTool → dura__tool Activity
```

> **Legacy:** `DuraMCPSession` and `MCPSessionRegistry` still exist but are superseded by the `langchain-mcp-adapters` pattern above.

---

## Child Workflows

When a `@dura` function calls another `@dura` function, DuraLang detects the existing `DuraContext` and routes the call as a **Temporal Child Workflow** — not a new top-level workflow. This gives each sub-agent:

- **Its own event history** — independent from the parent
- **Its own retry boundaries** — a failure in the child doesn't restart the parent
- **Its own timeout** — configurable via `child_workflow_timeout`

```python
@dura
async def researcher(query: str) -> str:
    ...  # This agent's calls live in its own event history

@dura
async def orchestrator(task: str) -> str:
    # Calling @dura from @dura → Temporal Child Workflow automatically
    result = await researcher(task)
    # ↑ If researcher fails, only researcher retries
    # orchestrator's completed work is preserved
```

### Deterministic Workflow IDs

Child workflow IDs are deterministic — they include the root workflow ID, function name, and a counter:

```
duralang-orchestrator-a1b2c3d4          ← root workflow
├── duralang-orchestrator-a1b2c3d4--child--researcher-1--f8e9d0c1
└── duralang-orchestrator-a1b2c3d4--child--analyst-2--f8e9d0c1
```

This prevents unbounded ID growth in deep nesting and makes child workflows easy to find in the Temporal UI.

---

## Agent Tools (Sub-Agents as Tools)

`dura_agent()` automatically detects `@dura` functions in its `tools` list and wraps them as LangChain `BaseTool` instances. This means sub-agents and regular tools coexist in the same list — no extra wrapping needed.

```python
from duralang import dura, dura_agent

@dura
async def researcher(query: str) -> str:
    """Research agent — gathers information."""
    ...

# Mix sub-agents and regular tools in one list — dura_agent handles wrapping
all_tools = [
    researcher,    # @dura → auto-wrapped as agent tool → Child Workflow
    calculator,    # @tool → auto-wrapped as DuraTool → dura__tool Activity
]
```

### How It Works Under the Hood

1. `dura_agent()` detects the `__dura__` flag on the function
2. It calls `dura_agent_tool()` internally, which reads the function's **signature**, **type hints**, and **docstring**
3. It generates a Pydantic `args_schema` and tool schema **automatically** (no manual schema authoring)
4. It returns a `BaseTool` subclass whose `_arun()` calls the `@dura` function
5. `DuraTool` detects the `__dura_agent_tool__` flag and **skips** `dura__tool` routing — the call goes directly to the `@dura` function, which the decorator routes as a Child Workflow

The result: **one dispatch pattern for everything.** The LLM sees a flat list of tools. DuraLang routes each call to the right Temporal primitive automatically.

---

## How Interception Works

`dura_agent()` wraps your model and tools with durable subclasses at agent creation time:

- `BaseChatModel` → `DuraModel` (delegates `_agenerate` through `dura__llm` Activity)
- `BaseTool` → `DuraTool` (delegates `_arun` through `dura__tool` Activity)
- `@dura` functions → agent tool via `dura_agent_tool()` (→ Child Workflow)

Each wrapper checks `DuraContext.get()` on every call and routes accordingly.

Key design decisions:

| Decision | Why |
|---|---|
| **Subclassing, not monkey-patching** | Clean composition — `DuraModel` wraps the inner LLM, `DuraTool` wraps the inner tool |
| **Wrapping at `dura_agent()` time** | Explicit — only tools/models passed to `dura_agent()` are wrapped |
| **ContextVar, not thread-local** | Correct behavior in async Python (`asyncio` tasks propagate ContextVars) |
| **Check on every call** | Zero overhead outside `@dura` (single ContextVar lookup), full routing inside |

---

## Serialization Boundaries

Data must cross Temporal's serialization boundary at two points: when scheduling operations and when returning results. DuraLang handles this with two serializers:

| Serializer | What It Handles |
|---|---|
| `MessageSerializer` | `HumanMessage`, `AIMessage`, `ToolMessage`, `SystemMessage` ↔ JSON dicts |
| `ArgSerializer` | Function arguments and return values ↔ JSON-safe primitives, lists, dicts, messages |

### Supported Types

| Type | Serialized As |
|---|---|
| `str`, `int`, `float`, `bool`, `None` | Passed through as-is |
| `list` | Recursively serialized |
| `dict` | Recursively serialized |
| `tuple` | Tagged dict with `__dura_type__: "tuple"` |
| LangChain `BaseMessage` | Tagged dict with `__dura_type__: "message"` and full message fields |

Unsupported types raise `StateSerializationError` with a clear message explaining what types are supported.
