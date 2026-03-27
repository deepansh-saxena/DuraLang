# Error Handling

DuraLang uses Temporal's retry mechanism for transient failures and returns non-retryable errors to the caller for graceful handling. Understanding how errors flow through the system is essential for building reliable agents.

---

## Exception Hierarchy

All DuraLang exceptions inherit from `DuraLangError`:

```
DuraLangError                    ← Base exception for all DuraLang errors
├── ConfigurationError           ← Setup-time errors
│   • Unknown LLM provider (not ChatAnthropic/ChatOpenAI/ChatGoogleGenerativeAI/ChatOllama)
│   • Lambda or non-importable function decorated with @dura
│   • Non-@dura function passed to dura_agent_tool()
│
├── LLMActivityError             ← LLM inference failed after max retries
│
├── ToolActivityError            ← Tool not found in registry, or execution failed after retries
│
├── MCPActivityError             ← MCP server not registered, or call failed after retries
│
├── WorkflowFailedError          ← Unrecoverable failure in the @dura function itself
│   • Raised when the user's function raises an exception
│   • Raised when a child workflow fails
│
└── StateSerializationError      ← Argument or message could not be serialized
    • Unsupported argument type (not a primitive, list, dict, or LangChain message)
    • Unknown message type during deserialization
```

---

## Retryable vs Non-Retryable Errors

DuraLang classifies errors into two categories. This classification determines whether Temporal retries the operation or fails it immediately:

### Retryable Errors (Temporal retries automatically)

These are transient — the operation might succeed on the next attempt:

| Error | Typical Cause | Retry Behavior |
|---|---|---|
| `httpx.TimeoutException` | LLM provider slow or overloaded | Backoff: 1s → 2s → 4s |
| `ConnectionError` | Network issue, DNS failure, provider down | Backoff: 1s → 2s → 4s |
| Rate limit (HTTP 429) | Too many requests to the LLM provider | Backoff: 2s → 4s → 8s |
| `OSError` / socket errors | Transient network failure | Backoff: 1s → 2s → 4s |

### Non-Retryable Errors (fail immediately)

These are logic errors — retrying with the same input will produce the same failure:

| Error | Typical Cause | Behavior |
|---|---|---|
| `ValueError` | Invalid argument value | For tools: returned as error string. For LLM: activity fails permanently. |
| `TypeError` | Wrong argument type | Activity fails permanently |
| `KeyError` | Missing expected key | For tools: returned as error string |
| `StateSerializationError` | Unsupported argument type | Raised immediately at serialization time |
| `ConfigurationError` | Setup error (wrong provider, lambda, etc.) | Raised immediately at decoration or import time |

---

## Tool Error Recovery

`dura__tool` has a unique error handling pattern designed for agent self-correction:

### Logic errors → returned as feedback (not raised)

When a tool raises `ValueError`, `TypeError`, or `KeyError`, the error message is **returned as a string** in `ToolActivityResult.error` instead of being raised as an exception. The LLM receives this error and can:

- Adjust its arguments and try again
- Choose a different tool
- Ask the user for clarification

This prevents wasted retry attempts on errors that will never succeed with the same input.

### Infrastructure errors → re-raised for retry

When a tool raises any other exception (network timeout, connection error, etc.), it's **re-raised** — Temporal catches it and retries the activity according to the retry policy. This is the right behavior for transient failures.

```
Tool raises ValueError("invalid date format")
  → ToolActivityResult(error="invalid date format", output="")
  → LLM sees the error as text feedback
  → LLM adjusts and calls the tool with correct format

Tool raises ConnectionError("API unreachable")
  → Exception re-raised to Temporal
  → Temporal retries with backoff: 1s → 2s → 4s
  → On success: ToolActivityResult with output
  → After max attempts: activity fails permanently
```

---

## Workflow-Level Errors

If the user's `@dura`-decorated function itself raises an exception (not an activity failure, but an exception in the orchestration logic), DuraLang catches it and:

1. Wraps the exception message in `WorkflowResult(error=str(e))`
2. Returns the result to `DuraRunner`
3. `DuraRunner` raises `WorkflowFailedError` with the original message

```python
@dura
async def my_agent(messages):
    raise RuntimeError("something went wrong")

try:
    await my_agent(messages)
except WorkflowFailedError as e:
    print(e)  # "something went wrong"
```

For child workflow failures, the same pattern applies — a child workflow failure propagates as `WorkflowFailedError` to the parent.

---

## Error Flow Diagram

```
User's @dura function
  │
  ├── llm.ainvoke() ──→ DuraLLMProxy ──→ dura__llm Activity
  │                                           │
  │                                    ┌──────┴──────┐
  │                                    │             │
  │                              Retryable      Non-retryable
  │                              (timeout,      (ValueError,
  │                               rate limit)    TypeError)
  │                                    │             │
  │                              Temporal        Activity fails
  │                              retries with    permanently
  │                              backoff
  │
  ├── tool.ainvoke() ──→ DuraToolProxy ──→ dura__tool Activity
  │                                           │
  │                                    ┌──────┼──────┐
  │                                    │      │      │
  │                              Retryable  Logic   Tool not
  │                              (network)  error   found
  │                                    │    (V/T/K)    │
  │                              Temporal  Error    ToolActivity
  │                              retries   string   Error raised
  │                                        returned
  │                                        to LLM
  │
  └── Function exception ──→ Caught by DuraLangWorkflow
                                    │
                              WorkflowResult(error=...)
                                    │
                              WorkflowFailedError raised to caller
```

---

## Debugging Guide

### 1. Check the Temporal UI First

Open `http://localhost:8233`. Every workflow shows:

- **Activity list** — each LLM call, tool call, MCP call with status
- **Input/output payloads** — full serialized data for each activity
- **Retry history** — how many attempts, what error on each attempt
- **Timing** — start time, duration, and gaps between activities

### 2. Common Error Patterns

| Symptom | Likely Cause | Fix |
|---|---|---|
| `"Tool 'X' not in registry"` | Tool wasn't created inside/before the `@dura` function | Ensure the tool is instantiated at module level or inside the function |
| `"Cannot determine LLM provider"` | Using an unsupported `BaseChatModel` subclass | Use `ChatAnthropic`, `ChatOpenAI`, `ChatGoogleGenerativeAI`, or `ChatOllama` |
| Activity times out repeatedly | `start_to_close_timeout` too short for the operation | Increase the timeout in `ActivityConfig` |
| Activity marked unhealthy | `heartbeat_timeout` too short for the operation | Increase `heartbeat_timeout`, especially for long LLM calls |
| `"@dura cannot wrap lambda functions"` | Lambda or closure passed to `@dura` | Define the function at module top level |
| `StateSerializationError` | Unsupported argument type | Use primitives, lists, dicts, or LangChain messages |
| `WorkflowFailedError` | Your function raised an unhandled exception | Check the error message — it contains the original exception text |

### 3. Adjusting Retry Policies

If you're seeing too many retries (wasting time) or too few (giving up too early):

```python
from datetime import timedelta
from temporalio.common import RetryPolicy
from duralang import DuraConfig, ActivityConfig

config = DuraConfig(
    llm_config=ActivityConfig(
        retry_policy=RetryPolicy(
            maximum_attempts=5,           # More attempts for unreliable providers
            initial_interval=timedelta(seconds=5),  # Longer wait between retries
            backoff_coefficient=3.0,      # More aggressive backoff
        ),
    ),
)
```
