# Error Handling

DuraLang uses Temporal's retry mechanism for transient failures and returns non-retryable errors to the caller for graceful handling.

---

## Exception Hierarchy

```
DuraLangError
├── ConfigurationError        # Unknown provider, non-importable function
├── LLMActivityError          # LLM inference failed after max retries
├── ToolActivityError         # Tool not registered or failed after retries
├── MCPActivityError          # MCP server not registered or call failed
├── WorkflowFailedError       # Unrecoverable workflow failure
└── StateSerializationError   # Argument or message serialization failed
```

---

## Retryable vs Non-Retryable

| Error | Retryable | Behavior |
|---|---|---|
| `httpx.TimeoutException` | Yes | Temporal retries with backoff |
| `ConnectionError` | Yes | Temporal retries with backoff |
| Rate limit errors | Yes | Temporal retries with longer backoff |
| `ValueError`, `TypeError` | No | Returned as error string to caller |
| `KeyError` | No | Returned as error string to caller |
| `StateSerializationError` | No | Raised immediately |

---

## Tool Error Handling

When a tool raises a non-retryable error (like `ValueError`), the error message is returned as a string instead of being raised. The calling code receives the error string and can decide how to handle it.

When a tool raises a retryable error (like a network timeout), Temporal retries the `dura__tool` Activity automatically according to the retry policy.

---

## Function Errors

If the user's `@dura`-decorated function raises an exception, it is captured and returned as a `WorkflowResult` with `error` set. The `DuraRunner` then raises `WorkflowFailedError`.

---

## Debugging Tips

1. **Check the Temporal UI** — every activity shows its input, output, and any retries
2. **Look at the error field** — `WorkflowResult.error` contains the exception message
3. **Check activity timeouts** — if an LLM call times out, increase `start_to_close_timeout`
4. **Check heartbeat timeouts** — if a long tool execution fails, increase `heartbeat_timeout`
