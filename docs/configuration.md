# Configuration

DuraLang's behavior is controlled through two dataclasses: `DuraConfig` for runtime settings and `ActivityConfig` for per-activity tuning. All settings have sensible defaults — you can use `@dura` with zero configuration and customize later.

---

## DuraConfig

Top-level configuration for the DuraLang runtime.

```python
from duralang import DuraConfig

config = DuraConfig(
    temporal_host="localhost:7233",     # Temporal server address
    temporal_namespace="default",       # Temporal namespace
    task_queue="duralang",             # Worker task queue name
    max_iterations=50,                  # Safety limit on LLM iterations
    child_workflow_timeout=timedelta(hours=1),  # Timeout for child workflows
)
```

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `temporal_host` | `str` | `"localhost:7233"` | Temporal server address. Use `temporal server start-dev` for local development |
| `temporal_namespace` | `str` | `"default"` | Temporal namespace. Use different namespaces to isolate environments |
| `task_queue` | `str` | `"duralang"` | Worker task queue. Multiple agents on the same queue share workers |
| `max_iterations` | `int` | `50` | Maximum LLM loop iterations per agent run. Safety valve against infinite loops |
| `child_workflow_timeout` | `timedelta` | `1 hour` | Maximum execution time for child workflows (`@dura` calling `@dura`) |
| `llm_config` | `ActivityConfig` | See below | Configuration for `dura__llm` activities |
| `tool_config` | `ActivityConfig` | See below | Configuration for `dura__tool` activities |
| `mcp_config` | `ActivityConfig` | See below | Configuration for `dura__mcp` activities |

### Usage

Pass config to the decorator:

```python
@dura(config=DuraConfig(temporal_host="prod.temporal:7233", task_queue="agents-prod"))
async def my_agent(messages):
    ...
```

Or use defaults (recommended for development):

```python
@dura  # DuraConfig() with all defaults
async def my_agent(messages):
    ...
```

---

## ActivityConfig

Per-activity Temporal configuration. Controls timeouts, heartbeats, and retry behavior for each activity type independently.

```python
from datetime import timedelta
from temporalio.common import RetryPolicy
from duralang import ActivityConfig

config = ActivityConfig(
    start_to_close_timeout=timedelta(minutes=5),
    heartbeat_timeout=timedelta(seconds=30),
    retry_policy=RetryPolicy(
        initial_interval=timedelta(seconds=1),
        backoff_coefficient=2.0,
        maximum_attempts=3,
        maximum_interval=timedelta(seconds=30),
        non_retryable_error_types=["ValueError", "TypeError", "KeyError"],
    ),
)
```

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `start_to_close_timeout` | `timedelta` | `5 min` | Maximum time for a single activity execution attempt |
| `heartbeat_timeout` | `timedelta` | `30s` | If no heartbeat is received within this window, the activity is considered hung |
| `retry_policy` | `RetryPolicy` | See below | Temporal retry policy with backoff, max attempts, and error classification |

### Default Retry Policy

```python
RetryPolicy(
    initial_interval=timedelta(seconds=1),      # First retry after 1s
    backoff_coefficient=2.0,                     # Each retry waits 2× longer
    maximum_attempts=3,                          # Give up after 3 attempts
    maximum_interval=timedelta(seconds=30),      # Never wait more than 30s
    non_retryable_error_types=[
        "ValueError",    # Logic errors — retrying won't help
        "TypeError",     # Type mismatches — retrying won't help
        "KeyError",      # Missing keys — retrying won't help
    ],
)
```

---

## Activity-Specific Defaults

Each activity type has different default settings optimized for its typical workload:

| Activity | Timeout | Heartbeat | Max Retries | Rationale |
|---|---|---|---|---|
| `dura__llm` | 10 min | 5 min | 3 | LLM calls can take 30–120s; generous timeout and heartbeat to avoid premature kills |
| `dura__tool` | 2 min | 30s | 3 | Tools are typically fast; shorter timeout catches stuck operations sooner |
| `dura__mcp` | 5 min | 30s | 3 | MCP calls vary; moderate timeout balances responsiveness and reliability |

### Customizing Per Activity Type

```python
from datetime import timedelta
from temporalio.common import RetryPolicy
from duralang import DuraConfig, ActivityConfig

config = DuraConfig(
    llm_config=ActivityConfig(
        start_to_close_timeout=timedelta(minutes=3),
        heartbeat_timeout=timedelta(seconds=30),
        retry_policy=RetryPolicy(maximum_attempts=5),
    ),
    tool_config=ActivityConfig(
        start_to_close_timeout=timedelta(minutes=1),
        retry_policy=RetryPolicy(maximum_attempts=4),
    ),
    mcp_config=ActivityConfig(
        start_to_close_timeout=timedelta(minutes=10),
    ),
)

@dura(config=config)
async def my_agent(messages):
    ...
```

---

## Child Workflow Timeout

When `@dura` functions call other `@dura` functions, each sub-agent runs as a Temporal Child Workflow. The `child_workflow_timeout` controls how long a child workflow can run before Temporal considers it timed out:

```python
config = DuraConfig(
    child_workflow_timeout=timedelta(hours=2),  # Default: 1 hour
)
```

This applies to all child workflows started by the decorated function, including those triggered by `dura_agent_tool()`.

---

## Environment-Based Configuration

A common pattern is to use different configurations for development and production:

```python
import os
from datetime import timedelta
from duralang import DuraConfig

def get_config() -> DuraConfig:
    if os.getenv("ENV") == "production":
        return DuraConfig(
            temporal_host="temporal.mycompany.com:7233",
            temporal_namespace="production",
            task_queue="agents-prod",
            child_workflow_timeout=timedelta(hours=4),
        )
    return DuraConfig()  # localhost defaults for development

@dura(config=get_config())
async def my_agent(messages):
    ...
```

---

## Task Queue Strategy

Task queues determine which workers process which workflows. Use separate task queues to:

- **Isolate workloads**: Critical agents on dedicated workers
- **Scale independently**: More workers for high-throughput queues
- **Version safely**: Route new code to a separate queue during deployment

```python
# High-priority agent on its own queue
@dura(config=DuraConfig(task_queue="critical-agents"))
async def billing_agent(messages): ...

# Standard agents on the default queue
@dura(config=DuraConfig(task_queue="general-agents"))
async def research_agent(messages): ...
```
