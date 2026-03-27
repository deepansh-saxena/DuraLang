# Configuration

DuraLang uses two configuration dataclasses: `DuraConfig` for the runtime and `ActivityConfig` for per-activity settings.

---

## DuraConfig

```python
from duralang import DuraConfig

config = DuraConfig(
    temporal_host="localhost:7233",     # Temporal server address
    temporal_namespace="default",       # Temporal namespace
    task_queue="duralang",             # Worker task queue
    max_iterations=50,                  # Max LLM calls per agent run
)
```

Pass it to the decorator:

```python
@dura(config=DuraConfig(temporal_host="prod.temporal:7233"))
async def my_agent(messages):
    ...
```

Or use defaults:

```python
@dura  # uses DuraConfig() defaults
async def my_agent(messages):
    ...
```

---

## ActivityConfig

Each activity type has its own `ActivityConfig` with timeouts and retry policies:

```python
from datetime import timedelta
from temporalio.common import RetryPolicy
from duralang import DuraConfig, ActivityConfig

config = DuraConfig(
    llm_config=ActivityConfig(
        start_to_close_timeout=timedelta(minutes=10),
        heartbeat_timeout=timedelta(seconds=60),
        retry_policy=RetryPolicy(
            initial_interval=timedelta(seconds=2),
            backoff_coefficient=2.0,
            maximum_attempts=3,
            non_retryable_error_types=["ValueError", "TypeError"],
        ),
    ),
    tool_config=ActivityConfig(
        start_to_close_timeout=timedelta(minutes=2),
    ),
    mcp_config=ActivityConfig(
        start_to_close_timeout=timedelta(minutes=5),
    ),
)
```

### Defaults

| Activity | Timeout | Heartbeat | Max Retries |
|---|---|---|---|
| `dura__llm` | 10 min | 60s | 3 |
| `dura__tool` | 2 min | 30s | 3 |
| `dura__mcp` | 5 min | 30s | 3 |

---

## Child Workflow Timeout

For `@dura` functions calling other `@dura` functions:

```python
config = DuraConfig(
    child_workflow_timeout=timedelta(hours=2),  # default: 1 hour
)
```
