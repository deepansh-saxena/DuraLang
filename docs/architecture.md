# Architecture

A complete technical walkthrough of how DuraLang makes LangChain agents durable — from the `@dura` decorator down to Temporal Activities.

---

## High-Level Overview

DuraLang sits between your LangChain code and Temporal. It intercepts LLM calls and tool calls at the method level and routes each one through a Temporal Activity or Child Workflow.

> **Note:** MCP tools from `langchain-mcp-adapters` are converted to standard `BaseTool` instances and go through the same `DuraTool` → `dura__tool` path as regular tools. No special MCP handling is needed.

```mermaid
graph TB
    subgraph "Your Code (unchanged LangChain)"
        A["@dura<br/>async def my_agent(messages)"]
        B["llm.ainvoke(messages)"]
        C["tool.ainvoke(input)<br/>(regular tools + MCP tools)"]
        E["await other_dura_fn(...)"]
        AT["@dura fn as tool → ainvoke(args)"]
    end

    subgraph "DuraLang (transparent interception layer)"
        F["DuraRunner<br/>starts workflow"]
        G["DuraLangWorkflow<br/>sets DuraContext"]
        H["DuraModel"]
        I["DuraTool"]
        K["Child Workflow routing"]
        AG["AgentTool<br/>(BaseTool → @dura)"]
    end

    subgraph "Temporal (durable execution engine)"
        L["dura__llm Activity"]
        M["dura__tool Activity"]
        O["Child DuraLangWorkflow"]
        P["Event History<br/>durable state"]
    end

    A -->|"called"| F
    F -->|"start_workflow"| G
    G -->|"sets ContextVar"| H
    G -->|"sets ContextVar"| I

    B -->|"intercepted"| H
    C -->|"intercepted"| I
    E -->|"detected"| K
    AT -->|"calls @dura fn"| AG
    AG -->|"detected"| K

    H -->|"execute_activity"| L
    I -->|"execute_activity"| M
    K -->|"execute_child_workflow"| O

    L --> P
    M --> P
    O --> P
```

**Key insight:** Your code (top layer) is unchanged LangChain. The DuraLang layer (middle) is completely transparent. Temporal (bottom) provides the durable execution guarantees. MCP tools are handled identically to regular tools — `langchain-mcp-adapters` converts them to `BaseTool` instances, and `dura_agent()` wraps them as `DuraTool` like any other tool.

---

## Request Flow — Single LLM Call

What happens, step by step, when a `@dura` function calls `llm.ainvoke(messages)`:

```mermaid
sequenceDiagram
    participant User as User Code
    participant Decorator as @dura wrapper
    participant Runner as DuraRunner
    participant Temporal as Temporal Server
    participant Workflow as DuraLangWorkflow
    participant Proxy as DuraModel
    participant Activity as dura__llm

    User->>Decorator: await my_agent(messages)
    Decorator->>Runner: get_or_create(config)
    Runner->>Temporal: start_workflow(DuraLangWorkflow)
    Temporal->>Workflow: run(payload)

    Note over Workflow: Set DuraContext in ContextVar

    Workflow->>Workflow: Resolve & call user function

    Note over Workflow: User code runs normally

    Workflow->>Proxy: llm.ainvoke(messages)
    Proxy->>Proxy: DuraContext.get() → context exists

    Note over Proxy: Extract LLMIdentity<br/>Serialize messages<br/>Extract tool schemas

    Proxy->>Workflow: ctx.execute_activity("dura__llm", payload)
    Workflow->>Temporal: workflow.execute_activity()
    Temporal->>Activity: dura__llm(payload)

    Note over Activity: Reconstruct LLM from LLMIdentity<br/>Rebind tools<br/>Deserialize messages<br/>Call real llm.ainvoke()

    Activity->>Temporal: LLMActivityResult
    Temporal->>Workflow: result
    Workflow->>Proxy: result
    Proxy->>Proxy: Deserialize AIMessage

    Note over Proxy: Return real AIMessage to user code

    Proxy->>Workflow: AIMessage
    Workflow->>Temporal: WorkflowResult
    Temporal->>Runner: result
    Runner->>Decorator: deserialized return value
    Decorator->>User: result
```

**The user code sees:** `response = await llm.ainvoke(messages)` returning an `AIMessage` — exactly as if DuraLang wasn't there. But behind the scenes, the call was routed through Temporal with full retry, heartbeat, and checkpoint guarantees.

---

## Tool Call Flow — Parallel Execution

When the LLM returns multiple tool calls and the user runs them with `asyncio.gather`:

```mermaid
sequenceDiagram
    participant UserFn as User Function
    participant LLMProxy as DuraModel
    participant ToolProxy1 as DuraTool (tool_1)
    participant ToolProxy2 as DuraTool (tool_2)
    participant WF as DuraLangWorkflow
    participant T as Temporal Server
    participant A1 as dura__tool (tool_1)
    participant A2 as dura__tool (tool_2)

    UserFn->>LLMProxy: llm.ainvoke(messages)
    LLMProxy->>WF: execute_activity("dura__llm")
    WF->>T: schedule dura__llm
    T-->>WF: AIMessage with tool_calls=[tc_1, tc_2]
    WF-->>LLMProxy: AIMessage
    LLMProxy-->>UserFn: AIMessage

    Note over UserFn: asyncio.gather(<br/>  tool_1.ainvoke(tc_1.args),<br/>  tool_2.ainvoke(tc_2.args)<br/>)

    par Parallel Tool Execution
        UserFn->>ToolProxy1: tool_1.ainvoke(args)
        ToolProxy1->>WF: execute_activity("dura__tool")
        WF->>T: schedule dura__tool
        T->>A1: dura__tool(payload_1)
        A1-->>T: ToolActivityResult
        T-->>WF: result_1
        WF-->>ToolProxy1: result_1
        ToolProxy1-->>UserFn: output_1
    and
        UserFn->>ToolProxy2: tool_2.ainvoke(args)
        ToolProxy2->>WF: execute_activity("dura__tool")
        WF->>T: schedule dura__tool
        T->>A2: dura__tool(payload_2)
        A2-->>T: ToolActivityResult
        T-->>WF: result_2
        WF-->>ToolProxy2: result_2
        ToolProxy2-->>UserFn: output_2
    end

    Note over UserFn: Both results ready — continue loop
```

**Key point:** If `tool_1` fails and retries, `tool_2`'s result is already checkpointed. Only the failed operation retries.

---

## Multi-Agent Flow — Child Workflows

When a `@dura` function calls another `@dura` function (either directly or as an agent tool):

```mermaid
sequenceDiagram
    participant User as User
    participant Parent as @dura orchestrator
    participant PWF as DuraLangWorkflow (parent)
    participant T as Temporal Server
    participant CWF as DuraLangWorkflow (child)
    participant Child as @dura researcher

    User->>Parent: await orchestrator(task)
    Parent->>PWF: start workflow

    Note over PWF: DuraContext set for parent

    PWF->>PWF: Call orchestrator() body

    Note over PWF: orchestrator calls researcher()

    PWF->>PWF: @dura wrapper detects DuraContext exists
    PWF->>T: execute_child_workflow(DuraLangWorkflow)

    Note over T: Child workflow starts with<br/>its own event history

    T->>CWF: run(child_payload)

    Note over CWF: DuraContext set for child

    CWF->>Child: Call researcher() body
    Child->>Child: LLM calls → dura__llm activities
    Child->>Child: Tool calls → dura__tool activities
    Child-->>CWF: return result
    CWF-->>T: WorkflowResult
    T-->>PWF: child result

    Note over PWF: orchestrator continues

    PWF->>PWF: More LLM calls if needed
    PWF-->>T: WorkflowResult
    T-->>Parent: result
    Parent-->>User: final answer
```

**Key point:** The child workflow has its own event history. If the child fails, only the child retries. The parent's completed work is preserved.

---

## Durable Wrapping — Decision Flow

How `dura_agent()` wraps and how wrappers route calls:

```mermaid
graph LR
    subgraph "Agent Creation (dura_agent)"
        A["dura_agent(model, tools)"] --> B["Wrap model → DuraModel"]
        A --> C["Wrap each tool"]
        C --> D{"Tool type?"}
        D -->|"@dura function"| E["dura_agent_tool() → BaseTool"]
        D -->|"@tool / BaseTool"| F["DuraTool wrapper"]
        D -->|"plain callable"| G["@tool → DuraTool"]
    end

    subgraph "Call Time (every ainvoke)"
        H["DuraModel._agenerate()"] --> I{"DuraContext.get()"}
        I -->|"None (outside @dura)"| J["Call inner LLM<br/>(standard LangChain)"]
        I -->|"Context exists"| K["Route to dura__llm<br/>(Temporal Activity)"]

        L["DuraTool._arun()"] --> M{"DuraContext.get()"}
        M -->|"None"| N["Call inner tool<br/>(standard LangChain)"]
        M -->|"Context exists"| O{"__dura_agent_tool__?"}
        O -->|"No"| P["Route to dura__tool<br/>(Temporal Activity)"]
        O -->|"Yes"| Q["Call @dura fn directly<br/>(Child Workflow)"]
    end

    style J fill:#d4edda
    style N fill:#d4edda
    style K fill:#cce5ff
    style P fill:#cce5ff
    style Q fill:#fff3cd
```

**Three outcomes:**
- **Green:** Outside `@dura` — vanilla LangChain behavior (zero overhead)
- **Blue:** Regular tool/LLM inside `@dura` — routed to Temporal Activity
- **Yellow:** Agent tool inside `@dura` — routed to Child Workflow

---

## Serialization Boundaries

Data must be JSON-serializable to cross Temporal's boundary. This diagram shows what gets serialized and how:

```mermaid
graph TD
    subgraph "User Code (Python objects)"
        A["list[BaseMessage]"]
        B["BaseChatModel instance"]
        C["BaseTool instance"]
        D["Function args (any)"]
    end

    subgraph "Serialized (JSON-safe dicts)"
        E["list[dict] — MessageSerializer"]
        F["LLMIdentity dict"]
        G["tool_name string"]
        H["list/dict — ArgSerializer"]
    end

    subgraph "Temporal Event History"
        I["WorkflowPayload"]
        J["LLMActivityPayload"]
        K["ToolActivityPayload"]
    end

    A -->|"MessageSerializer.serialize_many()"| E
    B -->|"LLMIdentity.from_instance()"| F
    C -->|"tool.name"| G
    D -->|"ArgSerializer.serialize()"| H

    E --> J
    F --> J
    G --> K
    H --> I

    I --> J
    I --> K
```

**Key design decision:** LLM objects and tool objects are never serialized directly. Instead, lightweight identifiers cross the boundary — `LLMIdentity` for models, tool name strings for tools. The real objects are reconstructed or looked up on the Activity side.

---

## Failure & Retry Flow

How errors are classified and handled:

```mermaid
graph TD
    A["dura__llm Activity starts"] --> B{"LLM call succeeds?"}
    B -->|"Yes"| C["Return LLMActivityResult<br/>✓ Checkpointed"]
    B -->|"No"| D{"Error type?"}

    D -->|"Timeout / ConnectionError<br/>RateLimitError"| E["Temporal retries<br/>with backoff"]
    E --> A

    D -->|"ValueError / TypeError"| F["Non-retryable<br/>Activity fails permanently"]

    G["dura__tool Activity starts"] --> H{"Tool call succeeds?"}
    H -->|"Yes"| I["Return ToolActivityResult<br/>✓ Checkpointed"]
    H -->|"No"| J{"Error type?"}

    J -->|"ValueError / TypeError<br/>KeyError"| K["Return error string<br/>in ToolActivityResult<br/>(LLM can self-correct)"]
    J -->|"Network / Timeout"| L["Temporal retries<br/>with backoff"]
    L --> G

    style E fill:#fff3cd
    style L fill:#fff3cd
    style F fill:#f8d7da
    style K fill:#f8d7da
    style C fill:#d4edda
    style I fill:#d4edda
```

**Note the difference:** Tool logic errors (`ValueError`, `TypeError`, `KeyError`) are returned as error strings — the LLM receives them as feedback and can self-correct. LLM logic errors fail permanently (they're typically configuration issues).

---

## Module Dependency Graph

How the modules depend on each other:

```mermaid
graph TD
    A["__init__.py"] --> B["decorator.py"]
    A --> C["proxy.py"]
    A --> D["config.py"]
    A --> E["exceptions.py"]
    A --> R["registry.py"]

    B --> F["context.py"]
    B --> D
    B --> G["runner.py"]

    G --> H["workflow.py"]
    G --> I["activities/"]
    G --> J["state.py"]

    C --> F
    C --> D
    C --> R
    C --> J
    C --> K["graph_def.py"]

    H --> F
    H --> K
    H --> J
    H --> G

    I --> J
    I --> K
    I --> R
    I --> D
    I --> E

    J --> E

    style A fill:#cce5ff
    style B fill:#cce5ff
    style C fill:#d4edda
    style H fill:#fff3cd
    style I fill:#fff3cd
```

**Color coding:**
- 🔵 **Blue:** Entry points — what the user imports
- 🟢 **Green:** Interception layer — proxy routing
- 🟡 **Yellow:** Temporal integration — workflows and activities

---

## Component Summary

| Component | File | Role |
|---|---|---|
| `@dura` | `decorator.py` | Entry point — wraps user function, starts workflow or child workflow |
| `dura_agent()` | `dura_agent.py` | Factory — wraps model+tools, delegates to `create_agent()` |
| `DuraModel` | `dura_model.py` | `BaseChatModel` subclass — routes `_agenerate` to `dura__llm` |
| `DuraTool` | `dura_tool.py` | `BaseTool` subclass — routes `_arun` to `dura__tool` (skips agent tools) |
| `dura_agent_tool()` | `agent_tool.py` | Wraps `@dura` function as `BaseTool` (internal, called by `dura_agent`) |
| `DuraContext` | `context.py` | `ContextVar` bridge — wrappers read it to decide routing |
| `DuraMCPProxy` | `proxy.py` | **Legacy.** Intercepts `call_tool()` on MCP sessions → routes to `dura__mcp`. Superseded by `langchain-mcp-adapters` which converts MCP tools to `BaseTool` — they now go through `DuraTool` like regular tools. |
| `DuraRunner` | `runner.py` | Temporal client/worker lifecycle — singleton per `(host, task_queue)` |
| `DuraLangWorkflow` | `workflow.py` | Temporal workflow — sets context, resolves function, executes user code |
| `dura__llm` | `activities/llm.py` | Activity — reconstructs LLM from identity, calls `ainvoke()` |
| `dura__tool` | `activities/tool.py` | Activity — looks up tool in registry, calls `ainvoke()` |
| `dura__mcp` | `activities/mcp.py` | **Legacy.** Activity — looks up MCP session, calls `call_tool()`. MCP tools now go through `dura__tool` via `langchain-mcp-adapters`. |
| `LLMIdentity` | `config.py` | Serializable LLM descriptor — crosses Temporal boundary |
| `ArgSerializer` | `state.py` | Serializes function args for workflow payload |
| `MessageSerializer` | `state.py` | Serializes LangChain messages for activity payloads |
| `ToolRegistry` | `registry.py` | Auto-populated tool registry — maps names to `BaseTool` instances |
| `MCPSessionRegistry` | `registry.py` | MCP session registry — maps server names to `ClientSession` instances |
| `DuraConfig` | `config.py` | Top-level Temporal configuration |
| `ActivityConfig` | `config.py` | Per-activity timeout, heartbeat, and retry configuration |

---

## File Structure

```
duralang/
├── __init__.py              # Exports: dura, dura_agent, DuraConfig, DuraMCPSession
├── decorator.py             # @dura — the primary public API
├── dura_agent.py            # dura_agent() — wraps model+tools for durable dispatch
├── dura_model.py            # DuraModel — BaseChatModel subclass for durable LLM calls
├── dura_tool.py             # DuraTool — BaseTool subclass for durable tool calls
├── agent_tool.py            # dura_agent_tool() — wraps @dura as BaseTool (internal)
├── proxy.py                 # DuraMCPProxy (legacy — MCP tools now use DuraTool path)
├── context.py               # DuraContext — ContextVar-based workflow context
├── workflow.py              # DuraLangWorkflow — Temporal workflow definition
├── runner.py                # DuraRunner — Temporal client + worker lifecycle
├── activities/
│   ├── __init__.py          # Exports: llm_activity, tool_activity, mcp_activity
│   ├── llm.py               # dura__llm — LLM inference activity
│   ├── tool.py              # dura__tool — tool execution activity
│   └── mcp.py               # dura__mcp — MCP call activity (legacy — MCP tools now use dura__tool)
├── graph_def.py             # Payload/Result dataclasses for Temporal
├── state.py                 # MessageSerializer + ArgSerializer
├── config.py                # DuraConfig, ActivityConfig, LLMIdentity
├── registry.py              # ToolRegistry, MCPSessionRegistry
├── exceptions.py            # Exception hierarchy
├── cli.py                   # duralang CLI (worker management)
└── py.typed                 # PEP 561 marker for type checking
```
