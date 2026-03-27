# Core Concepts

This page explains how DuraLang makes LangChain agents durable without changing your code.

---

## Three Layers

### Layer 1: `@dura` Decorator

The decorator wraps your async function. When called:
1. Serializes function arguments via `ArgSerializer`
2. Starts a `DuraLangWorkflow` on Temporal
3. Blocks until the workflow completes
4. Returns the deserialized result

Callers see a normal async function.

### Layer 2: Proxy Injection

Before your function executes inside the Temporal worker, DuraLang sets a `DuraContext` via Python's `contextvars.ContextVar`. Proxy objects installed on `BaseChatModel` and `BaseTool` at import time check for this context:

- **Inside `@dura`**: `llm.ainvoke()` routes to `dura__llm` Activity
- **Outside `@dura`**: `llm.ainvoke()` calls the original method normally

This means LangChain works identically outside of a `@dura` function.

### Layer 3: Temporal Primitives

| Intercepted Call | Temporal Primitive |
|---|---|
| `BaseChatModel.ainvoke()` | Activity: `dura__llm` |
| `BaseTool.ainvoke()` | Activity: `dura__tool` |
| `ClientSession.call_tool()` | Activity: `dura__mcp` |
| `@dura` calling `@dura` | Child Workflow |

---

## DuraContext

`DuraContext` is stored in a `contextvars.ContextVar`. It carries:

- `workflow_id` — the current Temporal workflow ID
- `config` — the `DuraConfig` for this execution
- `execute_activity` — closure that calls `workflow.execute_activity()`
- `execute_child_agent` — closure that calls `workflow.execute_child_workflow()`

Proxy objects read `DuraContext.get()` to decide whether to intercept or pass through.

---

## LLMIdentity

When a proxy intercepts `llm.ainvoke()`, it extracts an `LLMIdentity` from the LLM instance (provider, model, kwargs). This serializable identity crosses the Temporal boundary. The `dura__llm` Activity reconstructs the LLM from it.

The LLM object itself never touches Temporal's serializer.

---

## Auto-Registration

Tools are auto-registered in `ToolRegistry` when proxies are installed. When `bind_tools()` is called and the proxy extracts schemas, it also registers each `BaseTool` so `dura__tool` can find it. No explicit `register_tools()` call is needed.

---

## Child Workflows

When a `@dura`-decorated function is called from within an active `DuraContext`, the wrapper detects the context and routes to a Temporal Child Workflow instead of starting a new top-level workflow.

```python
@dura
async def researcher(messages): ...  # Child workflow when called from orchestrator

@dura
async def orchestrator(task):
    result = await researcher(messages)  # <- becomes Child Workflow
```

Child workflow IDs are deterministic — they include the parent workflow ID and function name.
