# API Reference

Complete reference for all public classes and functions in DuraLang.

---

## `dura`

```python
from duralang import dura
```

The primary public API. Decorates an async function to make it durable via Temporal.

### Signatures

```python
# Without arguments — uses default DuraConfig
@dura
async def my_agent(messages: list) -> list:
    ...

# With custom configuration
@dura(config=DuraConfig(temporal_host="prod:7233"))
async def my_agent(messages: list) -> list:
    ...
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `DuraConfig \| None` | `None` | Temporal runtime configuration. If `None`, uses `DuraConfig()` with all defaults |

### Behavior

| Context | What Happens |
|---|---|
| Called from **normal code** | Starts a new `DuraLangWorkflow` on Temporal. Blocks until workflow completes. Returns deserialized result. |
| Called from **inside another `@dura` function** | Executes as a Temporal Child Workflow with its own event history, timeouts, and retry boundaries |

### Requirements

- Function must be `async`
- Function must be defined at **module level** (importable by the Temporal worker)
- Function must not be a lambda or closure
- All arguments must be serializable (primitives, lists, dicts, LangChain messages)

### Special Keyword Arguments

| Kwarg | Type | Description |
|---|---|---|
| `_workflow_id` | `str \| None` | Optional fixed workflow ID. If provided and a workflow with this ID already exists, DuraLang reconnects to it — enabling crash recovery. See [`crash_recovery.py`](../examples/crash_recovery.py) |

---

## `dura_agent_tool`

```python
from duralang import dura_agent_tool
```

Wraps a `@dura`-decorated function as a real LangChain `BaseTool` for use in `bind_tools()` and `ainvoke()`.

### Signature

```python
def dura_agent_tool(
    fn,
    *,
    name: str | None = None,
    description: str | None = None,
) -> BaseTool
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fn` | Callable | *required* | A `@dura`-decorated async function. Raises `ConfigurationError` if the function doesn't have the `@dura` decorator. |
| `name` | `str \| None` | `"call_{fn.__name__}"` | Override the tool name |
| `description` | `str \| None` | Function docstring | Override the tool description |

### Returns

A `BaseTool` instance with:

| Attribute | Source |
|---|---|
| `name` | `"call_{fn.__name__}"` or overridden |
| `description` | Function docstring or overridden |
| `args_schema` | Pydantic model auto-generated from function signature and type hints |

### Schema Generation

`dura_agent_tool()` automatically generates the tool schema from the function's signature:

```python
@dura
async def researcher(query: str) -> str:
    """Research agent — gathers info via web search."""
    ...

tool = dura_agent_tool(researcher)
# tool.name → "call_researcher"
# tool.description → "Research agent — gathers info via web search."
# tool.args_schema → Pydantic model with: query: str (required)
```

For functions with multiple parameters and defaults:

```python
@dura
async def analyst(data: str, question: str, depth: int = 3) -> str:
    """Analysis agent — processes data."""
    ...

tool = dura_agent_tool(analyst)
# tool.args_schema → Pydantic model with:
#   data: str (required)
#   question: str (required)
#   depth: int = 3 (optional)
```

### Routing

When called via `ainvoke()` inside a `@dura` context:

1. `DuraToolProxy` detects the `__dura_agent_tool__` flag on the tool instance
2. **Skips** `dura__tool` activity routing
3. Calls the original `ainvoke()` → `_arun()` → the `@dura` function
4. The `@dura` wrapper detects the existing `DuraContext` and routes as a **Child Workflow**

The sub-agent gets its own event history, timeouts, and retry boundaries. Failures in the sub-agent don't affect the parent.

### Example

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
            # Same ainvoke() for agents and tools — routing is automatic
            result = await tools_by_name[tc["name"]].ainvoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return response.content
```

---

## `DuraMCPSession`

```python
from duralang import DuraMCPSession
```

Thin wrapper around `mcp.ClientSession` that enables durable `call_tool()` inside `@dura` functions.

### Signature

```python
class DuraMCPSession:
    def __init__(self, session: ClientSession, server_name: str): ...
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `session` | `mcp.ClientSession` | An initialized MCP client session |
| `server_name` | `str` | Unique name for this server (used for registry lookup in `dura__mcp` activity) |

### Behavior

- **`call_tool()`** → Inside `@dura`: routed to `dura__mcp` Temporal Activity. Outside `@dura`: calls the original method.
- **All other methods** (e.g., `list_tools()`, `initialize()`) → passed through to the underlying session unchanged.

### Example

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from duralang import dura, DuraMCPSession

server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        fs = DuraMCPSession(session, "filesystem")  # ← registers + installs proxy

        @dura
        async def my_agent(messages):
            # list_tools() passes through to the original session
            tools = await fs.list_tools()

            # call_tool() is intercepted → dura__mcp Activity
            result = await fs.call_tool("read_file", {"path": "/tmp/data.csv"})
            return result
```

---

## `DuraConfig`

```python
from duralang import DuraConfig
```

Top-level configuration for the DuraLang runtime.

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `temporal_host` | `str` | `"localhost:7233"` | Temporal server address |
| `temporal_namespace` | `str` | `"default"` | Temporal namespace |
| `task_queue` | `str` | `"duralang"` | Worker task queue name |
| `max_iterations` | `int` | `50` | Maximum LLM loop iterations per agent run |
| `child_workflow_timeout` | `timedelta` | `1 hour` | Maximum execution time for child workflows |
| `llm_config` | `ActivityConfig` | 10 min timeout, 5 min heartbeat, 3 retries | Config for `dura__llm` activities |
| `tool_config` | `ActivityConfig` | 2 min timeout, 30s heartbeat, 3 retries | Config for `dura__tool` activities |
| `mcp_config` | `ActivityConfig` | 5 min timeout, 30s heartbeat, 3 retries | Config for `dura__mcp` activities |

### Example

```python
from datetime import timedelta
from duralang import DuraConfig

config = DuraConfig(
    temporal_host="prod.temporal:7233",
    task_queue="agents-prod",
    child_workflow_timeout=timedelta(hours=4),
)

@dura(config=config)
async def my_agent(messages):
    ...
```

---

## `ActivityConfig`

```python
from duralang import ActivityConfig
```

Per-activity Temporal configuration controlling timeouts, heartbeats, and retry behavior.

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `start_to_close_timeout` | `timedelta` | `5 min` | Maximum time for a single activity attempt |
| `heartbeat_timeout` | `timedelta` | `30s` | Window for heartbeat liveness checks |
| `retry_policy` | `RetryPolicy` | 3 attempts, 2× backoff | Temporal retry policy |

### Default Retry Policy

```python
RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_attempts=3,
    maximum_interval=timedelta(seconds=30),
    non_retryable_error_types=["ValueError", "TypeError", "KeyError"],
)
```

### Example

```python
from datetime import timedelta
from temporalio.common import RetryPolicy
from duralang import DuraConfig, ActivityConfig

config = DuraConfig(
    llm_config=ActivityConfig(
        start_to_close_timeout=timedelta(minutes=3),
        heartbeat_timeout=timedelta(seconds=30),
        retry_policy=RetryPolicy(
            maximum_attempts=5,
            initial_interval=timedelta(seconds=2),
        ),
    ),
)
```

---

## Exceptions

All exceptions inherit from `DuraLangError` and can be imported from `duralang`:

```python
from duralang import (
    DuraLangError,
    ConfigurationError,
    LLMActivityError,
    ToolActivityError,
    MCPActivityError,
    WorkflowFailedError,
    StateSerializationError,
)
```

| Exception | When It's Raised |
|---|---|
| `DuraLangError` | Base class — catch this to handle all DuraLang errors |
| `ConfigurationError` | Unknown LLM provider, lambda decorated with `@dura`, non-`@dura` function passed to `dura_agent_tool()` |
| `LLMActivityError` | LLM inference failed after exhausting all retry attempts |
| `ToolActivityError` | Tool not found in `ToolRegistry`, or tool execution failed after retries |
| `MCPActivityError` | MCP server not found in `MCPSessionRegistry`, or call failed after retries |
| `WorkflowFailedError` | The `@dura` function raised an unhandled exception, or a child workflow failed |
| `StateSerializationError` | Function argument or LangChain message couldn't be serialized (unsupported type) |

---

## Internal Classes

These are implementation details not part of the public API. Documented here for contributor reference and debugging:

| Class | Role |
|---|---|
| `DuraContext` | `ContextVar`-based bridge between proxy objects and the Temporal workflow. Set when entering a `@dura` function, read by proxies on every call |
| `LLMIdentity` | Serializable identifier for a `BaseChatModel` — captures provider, model, and kwargs |
| `DuraRunner` | Temporal client/worker lifecycle manager. Singleton per `(host, task_queue)` pair |
| `DuraLangWorkflow` | The Temporal workflow definition. Resolves functions, sets context, executes user code |
| `MessageSerializer` | Converts LangChain `BaseMessage` objects ↔ JSON-safe dicts |
| `ArgSerializer` | Converts function arguments and return values ↔ JSON-safe representations |
| `ToolRegistry` | Module-level singleton mapping tool names → `BaseTool` instances |
| `MCPSessionRegistry` | Module-level singleton mapping server names → `ClientSession` instances |
