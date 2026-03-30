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

## `dura_agent`

```python
from duralang import dura_agent
```

Factory function that wraps a model and tools for durable dispatch. Returns a standard LangChain agent.

### Signature

```python
def dura_agent(
    model: str | BaseChatModel,
    tools: list | None = None,
    **kwargs,
) -> CompiledGraph
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str \| BaseChatModel` | *required* | Model string (e.g. `"claude-sonnet-4-6"`) or a `BaseChatModel` instance. Wrapped with `DuraModel` automatically. |
| `tools` | `list \| None` | `None` | Mix of `@tool` functions, `@dura` functions, or `BaseTool` instances. Each is wrapped automatically. |
| `**kwargs` | | | Passed through to LangChain's `create_agent()` |

### Tool Wrapping

`dura_agent()` detects each tool type and wraps accordingly:

| Input | Wrapped As | Temporal Primitive |
|---|---|---|
| `@tool` function / `BaseTool` | `DuraTool` | `dura__tool` Activity |
| `@dura` function | Agent tool (via `dura_agent_tool()` internally) | Child Workflow |
| Plain callable | `@tool` → `DuraTool` | `dura__tool` Activity |

### Returns

A compiled LangChain agent (`CompiledGraph`) — use `agent.ainvoke({"messages": ...})`.

### Example

```python
from duralang import dura, dura_agent

@dura
async def researcher(query: str) -> str:
    """Research sub-agent."""
    ...

all_tools = [
    researcher,    # @dura → Child Workflow (auto-wrapped)
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

---

## MCP Integration

The recommended way to use MCP servers with duralang is via [`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters). This converts MCP tools into standard LangChain `BaseTool` instances, which `dura_agent()` wraps automatically -- no special MCP handling needed.

### Recommended Pattern

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from duralang import dura, dura_agent

client = MultiServerMCPClient({
    "filesystem": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    },
})
tools = await client.get_tools()

@dura
async def my_agent(messages):
    agent = dura_agent("claude-sonnet-4-6", tools=tools)
    result = await agent.ainvoke({"messages": messages})
    return result["messages"]
```

MCP tools returned by `MultiServerMCPClient.get_tools()` are standard `BaseTool` instances. `dura_agent()` wraps them with `DuraTool` like any other tool, routing calls through the `dura__tool` Activity.

---

## `DuraMCPSession` (Legacy)

> **Legacy.** The recommended approach for MCP integration is `langchain-mcp-adapters` (see above). `DuraMCPSession` is retained for backward compatibility but may be removed in a future release.

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
| `ConfigurationError` | Unknown LLM provider, lambda decorated with `@dura`, non-`@dura` function passed as agent tool |
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
