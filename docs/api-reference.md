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
| `ConfigurationError` | Unknown provider, non-importable function |
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
