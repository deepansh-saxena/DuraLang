# Architecture

How DuraLang makes LangChain agents durable — from decorator to Temporal Activity.

---

## High-Level Overview

```mermaid
graph TB
    subgraph "Your Code"
        A["@dura<br/>async def my_agent(messages)"]
        B["llm.ainvoke(messages)"]
        C["tool.arun(input)"]
        D["session.call_tool(...)"]
        E["await other_dura_fn(...)"]
    end

    subgraph "DuraLang Layer"
        F["DuraRunner<br/>starts workflow"]
        G["DuraLangWorkflow<br/>sets DuraContext"]
        H["DuraLLMProxy"]
        I["DuraToolProxy"]
        J["DuraMCPProxy"]
        K["Child Workflow"]
    end

    subgraph "Temporal"
        L["dura__llm Activity"]
        M["dura__tool Activity"]
        N["dura__mcp Activity"]
        O["Child DuraLangWorkflow"]
        P["Event History<br/>durable state"]
    end

    A -->|"called"| F
    F -->|"start_workflow"| G
    G -->|"sets ContextVar"| H
    G -->|"sets ContextVar"| I
    G -->|"sets ContextVar"| J

    B -->|"intercepted"| H
    C -->|"intercepted"| I
    D -->|"intercepted"| J
    E -->|"detected"| K

    H -->|"execute_activity"| L
    I -->|"execute_activity"| M
    J -->|"execute_activity"| N
    K -->|"execute_child_workflow"| O

    L --> P
    M --> P
    N --> P
    O --> P
```

---

## Request Flow — Single LLM Call

What happens when a `@dura` function calls `llm.ainvoke(messages)`:

```mermaid
sequenceDiagram
    participant User as User Code
    participant Decorator as @dura wrapper
    participant Runner as DuraRunner
    participant Temporal as Temporal Server
    participant Workflow as DuraLangWorkflow
    participant Proxy as DuraLLMProxy
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

    Note over Proxy: Extract LLMIdentity from instance<br/>Serialize messages<br/>Extract bound tool schemas

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

---

## Tool Call Flow — With Parallel Execution

When the LLM returns multiple tool calls and the user runs them with `asyncio.gather`:

```mermaid
sequenceDiagram
    participant UserFn as User Function
    participant LLMProxy as DuraLLMProxy
    participant ToolProxy1 as DuraToolProxy (tool_1)
    participant ToolProxy2 as DuraToolProxy (tool_2)
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

    Note over UserFn: asyncio.gather(<br/>  tool_1.arun(tc_1.args),<br/>  tool_2.arun(tc_2.args)<br/>)

    par Parallel Tool Execution
        UserFn->>ToolProxy1: tool_1.arun(args)
        ToolProxy1->>WF: execute_activity("dura__tool")
        WF->>T: schedule dura__tool
        T->>A1: dura__tool(payload_1)
        A1-->>T: ToolActivityResult
        T-->>WF: result_1
        WF-->>ToolProxy1: result_1
        ToolProxy1-->>UserFn: output_1
    and
        UserFn->>ToolProxy2: tool_2.arun(args)
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

---

## Multi-Agent Flow — Child Workflows

When a `@dura` function calls another `@dura` function:

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
    Child-->>CWF: return messages
    CWF-->>T: WorkflowResult
    T-->>PWF: child result

    Note over PWF: orchestrator continues with result

    PWF->>PWF: More LLM calls if needed
    PWF-->>T: WorkflowResult
    T-->>Parent: result
    Parent-->>User: final answer
```

---

## Proxy Interception — How It Works

```mermaid
graph LR
    subgraph "Import Time"
        A["import duralang"] --> B["install_patches()"]
        B --> C["Patch BaseChatModel.__init__"]
        B --> D["Patch BaseTool.__init__"]
    end

    subgraph "Instance Creation"
        E["ChatAnthropic(model=...)"] --> F["Original __init__ runs"]
        F --> G["_install_llm_proxy(instance)"]
        G --> H["instance.ainvoke = proxy_fn"]
    end

    subgraph "Call Time"
        H --> I{"DuraContext.get()"}
        I -->|"None"| J["Call original ainvoke<br/>(normal LangChain)"]
        I -->|"Context exists"| K["Route to dura__llm<br/>(Temporal Activity)"]
    end

    style J fill:#d4edda
    style K fill:#cce5ff
```

---

## Serialization Boundaries

Data must cross serialization boundaries at the Temporal API:

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

---

## Failure & Retry

```mermaid
graph TD
    A["dura__llm Activity starts"] --> B{"LLM call succeeds?"}
    B -->|"Yes"| C["Return LLMActivityResult"]
    B -->|"No"| D{"Error type?"}

    D -->|"Timeout / ConnectionError<br/>RateLimitError"| E["Temporal retries<br/>with backoff"]
    E --> A

    D -->|"ValueError / TypeError"| F["Non-retryable<br/>Activity fails permanently"]

    G["dura__tool Activity starts"] --> H{"Tool call succeeds?"}
    H -->|"Yes"| I["Return ToolActivityResult"]
    H -->|"No"| J{"Error type?"}

    J -->|"ValueError / TypeError<br/>KeyError"| K["Return error string<br/>in ToolActivityResult"]
    J -->|"Network / Timeout"| L["Temporal retries<br/>with backoff"]
    L --> G

    style E fill:#fff3cd
    style L fill:#fff3cd
    style F fill:#f8d7da
    style K fill:#f8d7da
    style C fill:#d4edda
    style I fill:#d4edda
```

---

## Module Dependency Graph

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

---

## Component Summary

| Component | Role |
|---|---|
| `@dura` | Entry point — wraps user function, starts workflow |
| `DuraContext` | ContextVar bridge — proxies read it to decide routing |
| `DuraLLMProxy` | Intercepts `ainvoke()` — routes to `dura__llm` |
| `DuraToolProxy` | Intercepts `ainvoke()` — routes to `dura__tool` |
| `DuraMCPProxy` | Intercepts `call_tool()` — routes to `dura__mcp` |
| `DuraRunner` | Temporal client/worker lifecycle — singleton per config |
| `DuraLangWorkflow` | Temporal workflow — sets context, runs user function |
| `dura__llm` | Activity — reconstructs LLM, calls `ainvoke()` |
| `dura__tool` | Activity — looks up tool, calls `ainvoke()` |
| `dura__mcp` | Activity — looks up session, calls `call_tool()` |
| `LLMIdentity` | Serializable LLM descriptor — crosses Temporal boundary |
| `ArgSerializer` | Serializes function args for workflow payload |
| `MessageSerializer` | Serializes LangChain messages for activity payloads |
