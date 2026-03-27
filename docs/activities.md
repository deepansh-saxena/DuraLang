# Activities

DuraLang registers exactly **three Temporal Activities**. Every piece of non-deterministic work inside a `@dura` function runs through one of these. Understanding what they do and how they handle errors is essential for debugging and performance tuning.

---

## Overview

| Activity Name | Handles | Input | Output |
|---|---|---|---|
| `dura__llm` | All LLM inference calls | `LLMActivityPayload` | `LLMActivityResult` |
| `dura__tool` | All LangChain tool calls | `ToolActivityPayload` | `ToolActivityResult` |
| `dura__mcp` | All MCP server calls | `MCPActivityPayload` | `MCPActivityResult` |

Three activities. That's it. Every agent operation maps to one of these, regardless of LLM provider, tool type, or MCP server.

---

## `dura__llm` — LLM Inference

The most complex activity. Handles every intercepted `BaseChatModel.ainvoke()` call.

### Execution Steps

1. **Emit heartbeat** — signals the activity has started
2. **Reconstruct LLM** — builds a fresh `BaseChatModel` instance from `LLMIdentity`
   - Supported providers: `ChatAnthropic`, `ChatOpenAI`, `ChatGoogleGenerativeAI`, `ChatOllama`
   - If the original `llm` had `bind_tools()` called, the tool schemas are rebound here
3. **Deserialize messages** — converts JSON dicts back to LangChain `BaseMessage` objects
4. **Call `llm.ainvoke(messages)`** — the actual inference call to the LLM provider
5. **Emit heartbeat** — signals completion
6. **Serialize and return** — converts the `AIMessage` response to a JSON-safe dict

### Why the LLM Is Reconstructed

The LLM object (`ChatAnthropic`, `ChatOpenAI`, etc.) contains API clients, connection pools, and other non-serializable state. Instead of trying to serialize the full object, DuraLang captures the minimum needed to reconstruct it:

```
LLMIdentity:
  provider: "anthropic"
  model: "claude-sonnet-4-6"
  kwargs: {"temperature": 0.7}
```

This identity crosses the Temporal boundary. Inside the activity, `build_llm_from_identity()` creates a fresh instance and rebinds any tool schemas. The result is a clean, isolated inference call.

### Retry Behavior

| Error Type | Retryable? | What Happens |
|---|---|---|
| `httpx.TimeoutException` | Yes | Temporal retries with exponential backoff |
| `ConnectionError` | Yes | Temporal retries — the LLM provider may be temporarily down |
| Rate limit (429) errors | Yes | Temporal retries with longer backoff |
| `ValueError` | No | Activity fails permanently — likely a configuration issue |
| `TypeError` | No | Activity fails permanently — likely a type mismatch |

### Content Normalization

LLM providers return content in different formats — some as strings, some as lists of content blocks. The `_normalize_content()` function converts all formats to a consistent string representation before returning.

---

## `dura__tool` — Tool Execution

Handles every intercepted `BaseTool.ainvoke()` call (except agent tools — those bypass this activity entirely).

### Execution Steps

1. **Emit heartbeat** — signals the activity has started
2. **Look up tool** — finds the `BaseTool` instance in `ToolRegistry` by name
   - If not found: raises `ToolActivityError` with a clear message
3. **Call `tool.ainvoke(input)`** — executes the tool
4. **Emit heartbeat** — signals completion
5. **Return result** — the tool output as a string

### Error Handling Strategy

`dura__tool` uses a **two-tier error strategy** that separates logic errors from infrastructure errors:

| Error Type | Behavior | Rationale |
|---|---|---|
| `ValueError`, `TypeError`, `KeyError` | Caught, returned as error string in `ToolActivityResult.error` | These are logic errors — the input was wrong, not the infrastructure. Retrying won't help. The LLM receives the error as feedback and can try a different input. |
| All other exceptions (network, timeout) | Re-raised, triggering Temporal retry | These are transient — the request might succeed on the next attempt |

This means the LLM can recover from its own mistakes (bad tool arguments) without wasting retry attempts, while genuine infrastructure failures (API timeouts, connection drops) are retried automatically.

### Agent Tool Bypass

When `DuraToolProxy` detects the `__dura_agent_tool__` flag on a tool instance, it **skips** `dura__tool` routing entirely. The proxy calls the original `ainvoke()` directly, which calls `_arun()`, which calls the `@dura` function — routing it as a Temporal Child Workflow instead.

This means agent tools never appear in `dura__tool` activity logs. They appear as child workflows in the Temporal UI.

---

## `dura__mcp` — MCP Server Call

Handles every intercepted `ClientSession.call_tool()` call.

### Execution Steps

1. **Emit heartbeat** — signals the activity has started
2. **Look up session** — finds the MCP `ClientSession` in `MCPSessionRegistry` by server name
   - If not found: raises `MCPActivityError` with the server name
3. **Call `session.call_tool(tool_name, arguments)`** — executes the MCP tool call
4. **Emit heartbeat** — signals completion
5. **Serialize and return** — converts MCP content objects to JSON-safe dicts

### MCP Result Reconstruction

MCP `call_tool()` returns a result object with `.content` (a list of content blocks) and `.isError`. The activity serializes these to JSON dicts. On the proxy side, `_reconstruct_mcp_result()` rebuilds a compatible result object using `SimpleNamespace` — preserving the same `.content` and `.isError` interface that your code expects.

---

## Heartbeating in Detail

All three activities emit heartbeats at start and completion. Heartbeats serve two purposes:

1. **Liveness detection** — If Temporal doesn't receive a heartbeat within the `heartbeat_timeout` window, it marks the activity as unhealthy and reschedules it. This catches hung processes, dead workers, and stuck API calls.

2. **Progress signaling** — The heartbeat payload includes a human-readable message (e.g., `"llm: starting inference"`, `"tool:web_search complete"`), visible in the Temporal UI for debugging.

For LLM calls that can take 30–120 seconds, heartbeats are critical. A simple timeout can't distinguish between "the model is still thinking" and "the process is dead." Heartbeats can.

---

## Activity Payloads and Results

Every activity uses typed dataclasses for its input and output:

### LLM Activity

```python
@dataclass
class LLMActivityPayload:
    messages: list[dict]        # Serialized message history
    llm_identity: dict          # Provider + model + kwargs
    tool_schemas: list[dict]    # Bound tool schemas (if any)
    invoke_kwargs: dict         # Additional kwargs for ainvoke()

@dataclass
class LLMActivityResult:
    ai_message: dict            # Serialized AIMessage
    content: str                # Normalized response content
```

### Tool Activity

```python
@dataclass
class ToolActivityPayload:
    tool_name: str              # Tool name (looked up in ToolRegistry)
    tool_input: dict | str      # Tool input arguments
    tool_call_id: str           # ID from the LLM's tool_calls

@dataclass
class ToolActivityResult:
    output: str                 # Tool output as string
    tool_call_id: str           # Preserved tool_call_id
    error: str | None           # Error message if tool raised ValueError/TypeError/KeyError
```

### MCP Activity

```python
@dataclass
class MCPActivityPayload:
    server_name: str            # MCP server name (looked up in MCPSessionRegistry)
    tool_name: str              # MCP tool name
    arguments: dict             # Tool arguments
    tool_call_id: str           # ID from the LLM's tool_calls

@dataclass
class MCPActivityResult:
    content: list[dict]         # Serialized MCP content blocks
    tool_call_id: str           # Preserved tool_call_id
    is_error: bool              # Whether the MCP call returned an error
```
