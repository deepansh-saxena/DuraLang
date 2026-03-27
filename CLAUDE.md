# DuraLang — Specifications (CLAUDE.md)

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
from langchain.agents import create_agent

async def my_agent(messages):
    agent = create_agent(
        model="claude-sonnet-4-6",
        tools=[TavilySearchResults(), calculator],
    )
    result = await agent.ainvoke({"messages": messages})
    return result["messages"]

# After DuraLang — identical code, fully durable
from langchain.agents import create_agent
from duralang import dura

@dura                                          # <- the only change
async def my_agent(messages):
    agent = create_agent(
        model="claude-sonnet-4-6",
        tools=[TavilySearchResults(), calculator],
    )
    result = await agent.ainvoke({"messages": messages})  # LLM + tool calls → Temporal Activities
    return result["messages"]

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
  — same interface, `ainvoke` routes to `dura__tool` Activity
  — **except** agent tools (created by `dura_agent_tool()`), which bypass
    `dura__tool` and call the `@dura` function directly → Child Workflow
- Every `mcp.ClientSession` gets transparently replaced by a `DuraMCPProxy`
  — same interface, `call_tool` routes to `dura__mcp` Activity
- Every `@dura`-decorated function called from within a dura context
  becomes a child workflow call

### Layer 3: Temporal Primitives
Underneath, the same three Activities and child workflow pattern:

| Intercepted call | Temporal primitive |
|---|---|
| `BaseChatModel.ainvoke()` | Activity: `dura__llm` |
| `BaseTool.ainvoke()` | Activity: `dura__tool` |
| `dura_agent_tool(fn).ainvoke()` | Child Workflow (bypasses `dura__tool`) |
| `ClientSession.call_tool()` | Activity: `dura__mcp` |
| `@dura` function call from within dura | Child Workflow |

---

## 2. Multi-Agent via `dura_agent_tool()`

`dura_agent_tool()` wraps a `@dura` function as a real LangChain `BaseTool`.
This lets sub-agents and regular tools coexist in the same `create_agent(tools=...)`
call. Any agent becomes an orchestrator the moment you add sub-agent tools
to its toolkit.

```python
from langchain.agents import create_agent
from duralang import dura, dura_agent_tool

@dura
async def researcher(query: str) -> str:
    """Research agent — gathers information via web search."""
    agent = create_agent(model="claude-sonnet-4-6", tools=[TavilySearchResults()])
    result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content

@dura
async def analyst(data: str, question: str) -> str:
    """Analysis agent — runs calculations."""
    ...

# Sub-agents + regular tools — same list, same dispatch
all_tools = [
    dura_agent_tool(researcher),   # → Child Workflow
    dura_agent_tool(analyst),      # → Child Workflow
    calculator,                     # → dura__tool Activity
]

@dura
async def orchestrator(task: str) -> str:
    agent = create_agent(
        model="claude-sonnet-4-6",
        tools=all_tools,
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})
    return result["messages"][-1].content
```

How `dura_agent_tool()` works:
1. Reads function signature, type hints, and docstring
2. Generates Pydantic `args_schema` and tool schema automatically
3. Returns a `BaseTool` with `__dura_agent_tool__ = True`
4. `DuraToolProxy` detects the flag and skips `dura__tool` routing
5. The tool's `_arun()` calls the `@dura` function directly → Child Workflow

---

## 3. Goals & Non-Goals

### Goals
- Zero new API — user writes identical LangChain code using `create_agent`
- `@dura` decorator + `dura_agent_tool()` are the only new concepts
- LLM calls, tool calls, MCP calls, agent calls all durably intercepted
- Works with LangChain's `create_agent` — tool dispatch handled by LangChain
- Model-agnostic — any `BaseChatModel`
- MCP-native — any `ClientSession`
- Multi-agent — `@dura` functions calling other `@dura` functions = child workflows
- `dura_agent_tool()` lets sub-agents and regular tools coexist in same list
- Parallel tool calls preserved — handled by `create_agent` internally
- Python-only (v1)

### Non-Goals (v1)
- Intercepting sync `BaseTool.run()` — async only in v1; raise clear error
- TypeScript
- Managed cloud hosting

---

## 4. Module Structure

```
duralang/
├── __init__.py              # exports: dura, dura_agent_tool, DuraConfig, DuraMCPSession
├── decorator.py             # @dura implementation
├── proxy.py                 # DuraLLMProxy, DuraToolProxy, DuraMCPProxy
├── agent_tool.py            # dura_agent_tool() — wraps @dura as BaseTool
├── context.py               # DuraContext — ContextVar-based workflow context
├── workflow.py              # DuraLangWorkflow (@workflow.defn)
├── runner.py                # DuraRunner — Temporal client + worker lifecycle
├── activities/
│   ├── __init__.py
│   ├── llm.py               # dura__llm
│   ├── tool.py              # dura__tool
│   └── mcp.py               # dura__mcp
├── graph_def.py             # Payload/Result dataclasses for Temporal
├── state.py                 # MessageSerializer + ArgSerializer
├── config.py                # DuraConfig, ActivityConfig, LLMIdentity
├── registry.py              # ToolRegistry, MCPSessionRegistry
├── exceptions.py
├── cli.py                   # duralang CLI (worker start)
└── py.typed
```

---

## 5. Non-Negotiable Rules for Claude Code

1. **`@dura` is the primary public API.** `dura_agent_tool()` is the only
   other public function. No `register_tools()`. No `runtime.run()`.
   No `LLMConfig`. The user writes LangChain code. DuraLang intercepts it.

2. **Proxy passthrough when no context.** Every proxy method checks
   `DuraContext.get()`. If `None`, calls the original method normally.
   LangChain must work identically outside a dura context.

3. **Agent tools bypass `dura__tool`.** When `DuraToolProxy` detects
   `__dura_agent_tool__` on the instance, it calls the original `ainvoke`
   (which calls `_arun`, which calls the `@dura` function → Child Workflow).
   Agent tools never go through `dura__tool` activity.

4. **Proxy installed at `__init__` time via method wrapping.** Not via
   subclassing. Not via metaclass. Wrap the instance methods directly after
   `__init__` completes using `BaseChatModel.__init__` patching.

5. **`DuraContext` via `contextvars.ContextVar`.** Never pass context
   explicitly through function arguments. Proxy objects read it directly.

6. **Three activities only.** `dura__llm`, `dura__tool`, `dura__mcp`.

7. **Only the Workflow schedules activities and child workflows.**

8. **`@dura` calling `@dura` = Child Workflow.**

9. **Child workflow IDs are deterministic.** Include parent workflow ID,
   function name, and a stable identifier. Never random inside workflow code.

10. **`LLMIdentity` not LLM object crosses serialization.**

11. **Auto-register tools.** When `_extract_bound_tool_schemas` runs,
    it registers every `BaseTool` in `ToolRegistry`. The user never
    calls `register_tools()`.

12. **`workflow.now()` not `datetime.now()`** in all workflow code.

13. **User functions must be module-level importable.** Raise `ConfigurationError`
    with a clear message if a lambda or non-importable closure is decorated
    with `@dura`.

14. **`dura_agent_tool()` generates schemas from function signatures.**
    It reads type hints and docstrings. It returns a real `BaseTool`
    subclass via `pydantic.create_model`. No manual schema authoring.

15. **Use `create_agent` for tool-calling agents.** Examples and docs use
    LangChain's `create_agent` for tool dispatch — never manual tool loops.
    DuraLang intercepts the internal `ainvoke()` calls automatically.
