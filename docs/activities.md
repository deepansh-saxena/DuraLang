# Activities

DuraLang registers exactly three Temporal Activities. Every piece of non-deterministic work runs through one of these.

---

## `dura__llm` — LLM Inference

Handles all LLM calls intercepted from `BaseChatModel.ainvoke()`.

**What it does:**
1. Reconstructs the LLM from `LLMIdentity` (provider + model + kwargs)
2. Rebinds tool schemas if the LLM had `bind_tools()` called
3. Deserializes message history
4. Calls `llm.ainvoke(messages)`
5. Returns the serialized `AIMessage`

**Retry behavior:** Rate limit errors and connection errors are retryable. `ValueError` and `TypeError` are not.

---

## `dura__tool` — Tool Execution

Handles all tool calls intercepted from `BaseTool.ainvoke()`.

**What it does:**
1. Looks up the tool in `ToolRegistry` by name
2. Calls `tool.ainvoke(input)`
3. Returns the string output

**Error handling:**
- `ValueError`, `TypeError`, `KeyError` — returned as error string (non-retryable)
- Network errors, timeouts — re-raised for Temporal retry

---

## `dura__mcp` — MCP Server Call

Handles MCP tool calls intercepted from `ClientSession.call_tool()`.

**What it does:**
1. Looks up the MCP session in `MCPSessionRegistry` by server name
2. Calls `session.call_tool(tool_name, arguments)`
3. Returns the raw MCP content blocks

---

## Activity Naming

| Activity | Registered Name |
|---|---|
| LLM inference | `dura__llm` |
| Tool execution | `dura__tool` |
| MCP server call | `dura__mcp` |

Three activities. That's it.
