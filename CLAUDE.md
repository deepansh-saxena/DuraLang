# DuraLang — Specifications (CLAUDE.md)

> **Save this file as both `SPECIFICATIONS.md` and `CLAUDE.md` at the repo root.**
> Claude Code reads `CLAUDE.md` automatically as its primary instruction file.

> **Powered by Temporal** | Write normal LangChain code. Get Temporal durability.
> One decorator. Zero API to learn.

---

## 0. North Star

DuraLang makes LangChain agents durable with a single decorator.

The user writes **identical LangChain code**. DuraLang intercepts LLM calls,
tool calls, and MCP calls at runtime and routes them through Temporal Activities.
Agent calls become Temporal Child Workflows. State lives in Temporal event history.

```python
# Before DuraLang — normal LangChain
async def my_agent(messages):
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    tools = [TavilySearchResults(), calculator]
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

# After DuraLang — identical code, fully durable
from duralang import dura

@dura                                          # <- the only change
async def my_agent(messages):
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    tools = [TavilySearchResults(), calculator]
    llm_with_tools = llm.bind_tools(tools)

    while True:
        response = await llm_with_tools.ainvoke(messages)  # -> Temporal Activity
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            result = await tools_by_name[tc["name"]].arun(tc["args"])  # -> Temporal Activity
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return messages

# Call it like a normal async function
result = await my_agent([HumanMessage(content="What is the weather in NYC?")])
```

**The decorator is the entire public API.**
Everything inside the function is unchanged LangChain code.
DuraLang intercepts at the method level — the user never knows.

---

## 1. How It Works — Three Layers

### Layer 1: `@dura` Decorator
Wraps the user's async function. When called:
1. Serializes arguments
2. Starts a `DuraLangWorkflow` on Temporal
3. Blocks until workflow completes
4. Returns deserialized result

The function signature is unchanged. Callers see a normal async function.

### Layer 2: Proxy Injection
Before the user's function body executes inside the workflow/worker,
DuraLang installs proxy objects via `contextvars.ContextVar`:

- Every `BaseChatModel` instance the user creates gets transparently
  replaced by a `DuraLLMProxy` — same interface, `ainvoke` routes to `dura__llm` Activity
- Every `BaseTool` instance gets transparently replaced by a `DuraToolProxy`
  — same interface, `arun` routes to `dura__tool` Activity
- Every `mcp.ClientSession` gets transparently replaced by a `DuraMCPProxy`
  — same interface, `call_tool` routes to `dura__mcp` Activity
- Every `@dura`-decorated function called from within a dura context
  becomes a child workflow call

### Layer 3: Temporal Primitives
Underneath, the same three Activities and child workflow pattern:

| Intercepted call | Temporal primitive |
|---|---|
| `BaseChatModel.ainvoke()` | Activity: `dura__llm` |
| `BaseTool.arun()` / `.run()` | Activity: `dura__tool` |
| `ClientSession.call_tool()` | Activity: `dura__mcp` |
| `@dura` function call from within dura | Child Workflow |

---

## 2. Goals & Non-Goals

### Goals
- Zero new API — user writes identical LangChain code
- `@dura` decorator is the **only** new concept the user learns
- LLM calls, tool calls, MCP calls, agent calls all durably intercepted
- Model-agnostic — any `BaseChatModel`
- MCP-native — any `ClientSession`
- Multi-agent — `@dura` functions calling other `@dura` functions = child workflows
- Parallel tool calls preserved — `asyncio.gather` in user code works correctly
- Python-only (v1)

### Non-Goals (v1)
- LangGraph support (v2)
- Intercepting sync `BaseTool.run()` — async only in v1; raise clear error
- TypeScript
- Managed cloud hosting

---

## 3. Module Structure

```
duralang/
├── __init__.py              # exports: dura, DuraConfig
├── decorator.py             # @dura implementation
├── proxy.py                 # DuraLLMProxy, DuraToolProxy, DuraMCPProxy
├── context.py               # DuraContext — ContextVar-based workflow context
├── workflow.py              # DuraLangWorkflow (@workflow.defn)
├── runner.py                # DuraRunner — Temporal client + worker lifecycle
├── activities/
│   ├── __init__.py
│   ├── llm.py               # dura__llm
│   ├── tool.py              # dura__tool
│   └── mcp.py               # dura__mcp
├── state.py                 # MessageSerializer + ArgSerializer
├── config.py                # DuraConfig, ActivityConfig
├── registry.py              # ToolRegistry, MCPSessionRegistry
├── exceptions.py
└── py.typed
```

---

## 4. Non-Negotiable Rules for Claude Code

1. **`@dura` is the only public API.** No `register_tools()`. No `runtime.run()`.
   No `LLMConfig`. The user writes LangChain code. DuraLang intercepts it.

2. **Proxy passthrough when no context.** Every proxy method checks
   `DuraContext.get()`. If `None`, calls the original method normally.
   LangChain must work identically outside a dura context.

3. **Proxy installed at `__init__` time via method wrapping.** Not via
   subclassing. Not via metaclass. Wrap the instance methods directly after
   `__init__` completes using `BaseChatModel.__init__` patching.

4. **`DuraContext` via `contextvars.ContextVar`.** Never pass context
   explicitly through function arguments. Proxy objects read it directly.

5. **Three activities only.** `dura__llm`, `dura__tool`, `dura__mcp`.

6. **Only the Workflow schedules activities and child workflows.**

7. **`@dura` calling `@dura` = Child Workflow.**

8. **Child workflow IDs are deterministic.** Include parent workflow ID,
   function name, and a stable identifier. Never random inside workflow code.

9. **`LLMIdentity` not LLM object crosses serialization.**

10. **Auto-register tools.** When `_extract_bound_tool_schemas` runs,
    it registers every `BaseTool` in `ToolRegistry`. The user never
    calls `register_tools()`.

11. **`workflow.now()` not `datetime.now()`** in all workflow code.

12. **User functions must be module-level importable.** Raise `ConfigurationError`
    with a clear message if a lambda or non-importable closure is decorated
    with `@dura`.
