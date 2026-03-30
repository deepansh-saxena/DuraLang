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
from duralang import dura, dura_agent

@dura                                          # <- the only change
async def my_agent(messages):
    agent = dura_agent(
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

### Layer 2: Durable Wrapping via `dura_agent()`
`dura_agent()` wraps the user's model and tools with durable subclasses:

- `BaseChatModel` → `DuraModel` — same interface, `_agenerate` routes to `dura__llm` Activity
- `BaseTool` → `DuraTool` — same interface, `_arun` routes to `dura__tool` Activity
  — **except** agent tools (`@dura` functions passed as tools), which bypass
    `dura__tool` and call the `@dura` function directly → Child Workflow
- MCP tools: use `langchain-mcp-adapters` to convert MCP tools to `BaseTool` →
  passed to `dura_agent()` → wrapped as `DuraTool` → `dura__tool` Activity
- Every `@dura`-decorated function called from within a dura context
  becomes a child workflow call

### Layer 3: Temporal Primitives
Underneath, the same three Activities and child workflow pattern:

| Intercepted call | Temporal primitive |
|---|---|
| `DuraModel._agenerate()` | Activity: `dura__llm` |
| `DuraTool._arun()` | Activity: `dura__tool` |
| MCP tool (via `langchain-mcp-adapters`) | Activity: `dura__tool` (same as any `BaseTool`) |
| `@dura` function as tool | Child Workflow (bypasses `dura__tool`) |
| `@dura` function call from within dura | Child Workflow |
| `DuraMCPProxy.call_tool()` *(legacy)* | Activity: `dura__mcp` |

---

## 2. Multi-Agent via `dura_agent()`

`dura_agent()` automatically wraps `@dura` functions passed in `tools` as
LangChain `BaseTool` instances. Sub-agents and regular tools coexist in
the same list. Any agent becomes an orchestrator the moment you add
sub-agent functions to its toolkit.

```python
from duralang import dura, dura_agent

@dura
async def researcher(query: str) -> str:
    """Research agent — gathers information via web search."""
    agent = dura_agent(model="claude-sonnet-4-6", tools=[TavilySearchResults()])
    result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content

@dura
async def analyst(data: str, question: str) -> str:
    """Analysis agent — runs calculations."""
    ...

# Sub-agents + regular tools — same list, dura_agent wraps each automatically
all_tools = [
    researcher,    # @dura → Child Workflow (auto-wrapped)
    analyst,       # @dura → Child Workflow (auto-wrapped)
    calculator,    # @tool → dura__tool Activity (auto-wrapped)
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

How it works internally:
1. `dura_agent()` detects `@dura` functions via the `__dura__` flag
2. Calls `dura_agent_tool()` internally — reads signature, type hints, docstring
3. Generates Pydantic `args_schema` and tool schema automatically
4. Returns a `BaseTool` with `__dura_agent_tool__ = True`
5. `DuraTool` detects the flag and skips `dura__tool` routing
6. The tool's `_arun()` calls the `@dura` function directly → Child Workflow

---

## 3. Goals & Non-Goals

### Goals
- Zero new API — user writes identical LangChain code using `dura_agent`
- `@dura` decorator + `dura_agent()` are the only new concepts
- LLM calls, tool calls, MCP calls, agent calls all durably intercepted
- Works with `dura_agent` — tool dispatch handled by LangChain internally
- Model-agnostic — any `BaseChatModel`
- MCP-native — via `langchain-mcp-adapters` (MCP tools become `BaseTool` → `dura__tool`)
- Multi-agent — `@dura` functions calling other `@dura` functions = child workflows
- `dura_agent()` auto-wraps sub-agents and regular tools in same list
- Parallel tool calls preserved — handled by `dura_agent` internally
- Python-only (v1)

### Non-Goals (v1)
- Intercepting sync `BaseTool.run()` — async only in v1; raise clear error
- TypeScript
- Managed cloud hosting

---

## 4. Module Structure

```
duralang/
├── __init__.py              # exports: dura, dura_agent, DuraConfig
├── decorator.py             # @dura implementation
├── dura_agent.py            # dura_agent() — wraps model+tools for durable dispatch
├── dura_model.py            # DuraModel — BaseChatModel subclass for durable LLM calls
├── dura_tool.py             # DuraTool — BaseTool subclass for durable tool calls
├── agent_tool.py            # dura_agent_tool() — wraps @dura as BaseTool (internal)
├── proxy.py                 # DuraMCPProxy (legacy — MCP tools now use langchain-mcp-adapters → BaseTool → DuraTool)
├── context.py               # DuraContext — ContextVar-based workflow context
├── workflow.py              # DuraLangWorkflow (@workflow.defn)
├── runner.py                # DuraRunner — Temporal client + worker lifecycle
├── activities/
│   ├── __init__.py
│   ├── llm.py               # dura__llm
│   ├── tool.py              # dura__tool
│   └── mcp.py               # dura__mcp (legacy — MCP tools now go through dura__tool)
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

1. **`@dura` + `dura_agent()` are the public API.** No `register_tools()`.
   No `runtime.run()`. No `LLMConfig`. The user writes LangChain code.
   DuraLang wraps and intercepts it.

2. **Passthrough when no context.** `DuraModel` and `DuraTool` check
   `DuraContext.get()`. If `None`, call the inner model/tool normally.
   LangChain must work identically outside a dura context.

3. **Agent tools bypass `dura__tool`.** When `DuraTool` detects
   `__dura_agent_tool__` on the inner tool, it calls the original `ainvoke`
   (which calls `_arun`, which calls the `@dura` function → Child Workflow).
   Agent tools never go through `dura__tool` activity.

4. **Wrapping via subclassing, not monkey-patching.** `DuraModel` wraps
   `BaseChatModel`, `DuraTool` wraps `BaseTool`. `dura_agent()` creates
   these wrappers at agent creation time.

5. **`DuraContext` via `contextvars.ContextVar`.** Never pass context
   explicitly through function arguments. Wrapper objects read it directly.

6. **Two primary activities.** `dura__llm` and `dura__tool`. (`dura__mcp` is legacy —
   MCP tools now go through `langchain-mcp-adapters` → `BaseTool` → `dura__tool`.)

7. **Only the Workflow schedules activities and child workflows.**

8. **`@dura` calling `@dura` = Child Workflow.**

9. **Child workflow IDs are deterministic.** Include parent workflow ID,
   function name, and a stable identifier. Never random inside workflow code.

10. **`LLMIdentity` not LLM object crosses serialization.**

11. **Auto-register tools.** `DuraTool.__init__` registers the inner tool
    in `ToolRegistry`. The user never calls `register_tools()`.

12. **`workflow.now()` not `datetime.now()`** in all workflow code.

13. **User functions must be module-level importable.** Raise `ConfigurationError`
    with a clear message if a lambda or non-importable closure is decorated
    with `@dura`.

14. **`dura_agent_tool()` generates schemas from function signatures.**
    It reads type hints and docstrings. It returns a real `BaseTool`
    subclass via `pydantic.create_model`. No manual schema authoring.
    Called internally by `dura_agent()`.

15. **Use `dura_agent` for tool-calling agents.** Examples and docs use
    `dura_agent` for tool dispatch — never manual tool loops.
    DuraLang intercepts the internal `ainvoke()` calls automatically.
