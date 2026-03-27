# API Reference

Complete reference for all public classes and functions in DuraLang.

---

## `dura`

The decorator that makes a LangChain agent function durable via Temporal.

```python
from duralang import dura

# No arguments — uses default DuraConfig
@dura
async def my_agent(messages: list) -> list:
    ...

# With custom config
@dura(config=DuraConfig(temporal_host="prod:7233"))
async def my_agent(messages: list) -> list:
    ...
```

**Behavior:**
- Outside a `@dura` context: starts a new `DuraLangWorkflow` on Temporal
- Inside a `@dura` context: executes as a Temporal Child Workflow
- Returns whatever the decorated function returns

---

## `dura_agent_tool`

Wraps a `@dura` function as a real LangChain `BaseTool` for use in `bind_tools()`.

```python
from duralang import dura, dura_agent_tool

@dura
async def researcher(query: str) -> str:
    """Research agent — gathers info via web search."""
    ...

@dura
async def analyst(data: str, question: str) -> str:
    """Analysis agent — runs calculations."""
    ...

# Create agent tools — these are real BaseTool instances
researcher_tool = dura_agent_tool(researcher)
analyst_tool = dura_agent_tool(analyst)

# Mix with regular @tool functions in the same list
all_tools = [researcher_tool, analyst_tool, calculator]
```

**Parameters:**
- `fn` — A `@dura`-decorated async function (required)
- `name` — Override tool name (default: `"call_{fn.__name__}"`)
- `description` — Override description (default: function docstring)

**Returns:** A `BaseTool` instance with:
- `name` — derived from function name or overridden
- `description` — derived from docstring or overridden
- `args_schema` — Pydantic model generated from function signature and type hints

**Behavior:**
- Schema auto-generated from function signature, type hints, and docstring
- When called via `ainvoke()` inside a `@dura` context, routes as a **Temporal Child Workflow** (not a `dura__tool` activity)
- Each sub-agent gets its own event history, timeouts, and retry boundaries
- Can be mixed with regular `@tool` functions in the same `bind_tools()` call

**Example — same dispatch loop for both agent tools and regular tools:**

```python
all_tools = [
    dura_agent_tool(researcher),   # → Child Workflow
    dura_agent_tool(analyst),      # → Child Workflow
    calculator,                     # → dura__tool Activity
]
tools_by_name = {t.name: t for t in all_tools}

@dura
async def orchestrator(task: str) -> str:
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools(all_tools)

    messages = [HumanMessage(content=task)]
    while True:
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            result = await tools_by_name[tc["name"]].ainvoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return response.content
```

---

## `DuraConfig`

Top-level configuration.

```python
@dataclass
class DuraConfig:
    temporal_host: str = "localhost:7233"
    temporal_namespace: str = "default"
    task_queue: str = "duralang"
    max_iterations: int = 50
    child_workflow_timeout: timedelta = timedelta(hours=1)
    llm_config: ActivityConfig = ...    # dura__llm settings
    tool_config: ActivityConfig = ...   # dura__tool settings
    mcp_config: ActivityConfig = ...    # dura__mcp settings
```

---

## `ActivityConfig`

Per-activity Temporal configuration.

```python
@dataclass
class ActivityConfig:
    start_to_close_timeout: timedelta = timedelta(minutes=5)
    heartbeat_timeout: timedelta = timedelta(seconds=30)
    retry_policy: RetryPolicy = RetryPolicy(
        initial_interval=timedelta(seconds=1),
        backoff_coefficient=2.0,
        maximum_attempts=3,
        non_retryable_error_types=["ValueError", "TypeError", "KeyError"],
    )
```

---

## `DuraMCPSession`

Thin wrapper around `mcp.ClientSession` that attaches a server name for routing.

```python
from duralang import DuraMCPSession

fs = DuraMCPSession(session, "filesystem")
# Use fs exactly like a normal ClientSession
result = await fs.call_tool("read_file", {"path": "/tmp/data.csv"})
```

All methods pass through to the underlying session. Inside a `@dura` context, `call_tool()` routes to `dura__mcp` Activity.

---

## Exceptions

| Exception | Description |
|---|---|
| `DuraLangError` | Base exception |
| `ConfigurationError` | Unknown provider, non-importable function, non-`@dura` function passed to `dura_agent_tool()` |
| `LLMActivityError` | LLM inference failed |
| `ToolActivityError` | Tool not registered or execution failed |
| `MCPActivityError` | MCP server not registered or call failed |
| `WorkflowFailedError` | Unrecoverable workflow failure |
| `StateSerializationError` | Serialization failure |

---

## Internal Classes (not public API)

These are implementation details, documented for contributor reference:

- `DuraContext` — ContextVar bridge between proxies and workflow
- `LLMIdentity` — serializable LLM identifier extracted from instances
- `DuraRunner` — Temporal client/worker lifecycle singleton
- `MessageSerializer` — LangChain message serialization
- `ArgSerializer` — function argument serialization
- `ToolRegistry` — auto-populated tool registry
- `MCPSessionRegistry` — MCP session registry
