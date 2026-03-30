# Tools & MCP

DuraLang supports three types of tool integrations: **LangChain tools**, **agent tools** (sub-agents as tools), and **MCP tools** (via `langchain-mcp-adapters`). All are intercepted automatically inside `@dura` functions, and all can be mixed freely in the same agent.

---

## LangChain Tools

Standard LangChain `@tool` functions work inside `@dura` with zero changes. Every `tool.ainvoke()` call becomes a `dura__tool` Temporal Activity — automatically retried, heartbeated, and checkpointed.

```python
from langchain_core.tools import tool
from duralang import dura, dura_agent

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"72°F and sunny in {city}"

@tool
def get_time(timezone: str) -> str:
    """Get the current time in a timezone."""
    return f"3:00 PM in {timezone}"

tools = [get_weather, get_time]

@dura
async def my_agent(messages):
    agent = dura_agent(
        model="claude-sonnet-4-6",
        tools=tools,
    )
    result = await agent.ainvoke({"messages": messages})   # → dura__llm + dura__tool Activities
    return result["messages"]
```

### Auto-Registration

Tools are auto-registered in DuraLang's internal `ToolRegistry` when `DuraTool` wraps them. `dura_agent()` does this automatically for every tool in its `tools` list.

You never call `register_tools()`. There is no such function. Registration is fully automatic.

### Parallel Tool Calls

When the LLM returns multiple tool calls, `dura_agent` handles parallel dispatch automatically. Each tool call becomes its own `dura__tool` Activity, scheduled concurrently by Temporal:

```python
from duralang import dura_agent

@dura
async def my_agent(messages):
    agent = dura_agent(
        model="claude-sonnet-4-6",
        tools=tools,
    )
    # dura_agent handles parallel tool calls automatically —
    # each becomes an independent durable Activity
    result = await agent.ainvoke({"messages": messages})
    return result["messages"]
```

---

## Agent Tools — Sub-Agents as Tools

`dura_agent()` automatically wraps `@dura` functions passed in the `tools` list as LangChain `BaseTool` instances. This is the key to building dynamic multi-agent systems where the LLM decides which agents to invoke, how many times, and in what order.

### Why Agent Tools Exist

Without agent tools, you have two patterns for multi-agent:

1. **Direct `@dura` → `@dura` calls** — your code decides which agent to call. Simple, but rigid.
2. **Manual tool wrapping** — write boilerplate to expose each agent as a tool. Tedious and error-prone.

`dura_agent()` gives you a third pattern: **LLM-driven agent delegation** with zero boilerplate. Pass `@dura` functions alongside regular tools — `dura_agent()` handles the wrapping automatically.

### Usage

```python
from duralang import dura, dura_agent

@dura
async def researcher(query: str) -> str:
    """Research agent — gathers information via web search."""
    agent = dura_agent(
        model="claude-sonnet-4-6",
        tools=[web_search, wikipedia_lookup],
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content

@dura
async def analyst(data: str, question: str) -> str:
    """Analysis agent — processes data with calculations."""
    ...

# Mix sub-agents and regular tools — dura_agent wraps each automatically
all_tools = [
    researcher,    # @dura → auto-wrapped → Child Workflow
    analyst,       # @dura → auto-wrapped → Child Workflow
    calculator,    # @tool → auto-wrapped → dura__tool Activity
]

@dura
async def orchestrator(task: str) -> str:
    agent = dura_agent(
        model="claude-sonnet-4-6",
        tools=all_tools,
    )
    # dura_agent handles dispatch — DuraLang routes automatically
    result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})
    return result["messages"][-1].content
```

### How It Works Under the Hood

1. **Detection** — `dura_agent()` checks each tool for the `__dura__` flag (set by `@dura`)

2. **Schema generation** — For `@dura` functions, it calls `dura_agent_tool()` internally, which reads the function's signature, type hints, and docstring, then generates a Pydantic `args_schema` automatically. No manual schema authoring.

3. **Tool creation** — Returns a `BaseTool` subclass with `name`, `description`, and `args_schema` set. The tool's `_arun()` method calls the `@dura` function.

4. **Routing bypass** — The tool instance is marked with `__dura_agent_tool__ = True`. When `DuraTool` detects this flag, it **skips** `dura__tool` routing and calls the original `ainvoke()` directly. This triggers `_arun()`, which calls the `@dura` function, which the decorator routes as a Child Workflow.

5. **Independent durability** — Each sub-agent gets its own Temporal workflow with its own event history. A failure in one sub-agent retries only that sub-agent.

### Naming Convention

By default, a `@dura` function named `researcher` creates a tool named `call_researcher`. The tool description comes from the function's docstring.

### Nesting

Agent tools nest to any depth. If an orchestrator calls a researcher, and the researcher calls a fact-checker, each level is independently durable:

```
orchestrator (Workflow)
├── dura__llm [Activity]
├── call_researcher (Child Workflow)
│    ├── dura__llm [Activity]
│    ├── call_fact_checker (Child Workflow)
│    │    ├── dura__llm [Activity]
│    │    └── dura__tool: web_search [Activity]
│    └── dura__tool: wikipedia [Activity]
└── dura__llm [Activity]
```

---

## MCP Tools

MCP (Model Context Protocol) tools are supported via [`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters). The `MultiServerMCPClient` converts MCP servers into standard LangChain `BaseTool` instances, which `dura_agent()` wraps with `DuraTool` like any other tool. No special MCP plumbing is needed — MCP tools are just regular tools.

### Setup

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from duralang import dura, dura_agent

client = MultiServerMCPClient({
    "filesystem": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    },
    "weather": {
        "transport": "http",
        "url": "http://localhost:8000/mcp",
    },
})
tools = await client.get_tools()

@dura
async def my_agent(messages):
    agent = dura_agent("claude-sonnet-4-6", tools=tools)
    result = await agent.ainvoke({"messages": messages})
    return result["messages"]
```

### How It Works

1. **`MultiServerMCPClient`** connects to one or more MCP servers and calls `get_tools()` to produce standard `BaseTool` instances
2. **`dura_agent()`** wraps each `BaseTool` with `DuraTool`, exactly as it does for `@tool` functions
3. **Each MCP tool call** becomes a `dura__tool` Temporal Activity — retried, heartbeated, and checkpointed automatically

There is no separate `dura__mcp` routing for this pattern. MCP tools go through the same `dura__tool` Activity as every other tool.

### Transport Considerations

**Stdio servers** spawn a subprocess. Temporal workflows run inside a sandboxed event loop where subprocess creation is not available. Create the `MultiServerMCPClient` **outside** the `@dura` function and pass the tools in:

```python
# Stdio — create client at module level, outside @dura
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

**HTTP, SSE, and WebSocket servers** do not have this limitation. They communicate over the network and work anywhere, including inside `@dura` functions.

### Multiple MCP Servers

`MultiServerMCPClient` supports multiple servers in a single dictionary. All tools from all servers are returned as a flat list:

```python
client = MultiServerMCPClient({
    "filesystem": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    },
    "database": {
        "transport": "http",
        "url": "http://localhost:9000/mcp",
    },
    "github": {
        "transport": "sse",
        "url": "http://localhost:9001/sse",
    },
})
mcp_tools = await client.get_tools()

@dura
async def orchestrator(task: str) -> str:
    agent = dura_agent(
        model="claude-sonnet-4-6",
        tools=mcp_tools + [calculator, researcher],  # MCP + regular + agent tools
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})
    return result["messages"][-1].content
```

---

## Complete Routing Table

All three tool types can coexist in the same agent. DuraLang routes each call to the correct Temporal primitive automatically:

| What You Pass | Wrapped As | Temporal Primitive |
|---|---|---|
| `@tool` function / `BaseTool` | `DuraTool` | `dura__tool` Activity |
| `@dura` function | Agent tool (via `dura_agent_tool()` internally) | Child Workflow |
| MCP tools (via `langchain-mcp-adapters`) | `DuraTool` | `dura__tool` Activity |

`dura_agent` handles dispatch for all tool types — DuraLang decides whether each call becomes a `dura__tool` Activity or a Child Workflow automatically.

---

## Mixing Everything Together

A single agent can use all three tool types simultaneously:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from duralang import dura, dura_agent

# MCP tools
client = MultiServerMCPClient({
    "filesystem": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    },
})
mcp_tools = await client.get_tools()

# Regular tools
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    ...

# Agent tools
@dura
async def researcher(query: str) -> str:
    """Research agent."""
    ...

# Everything in one list — dura_agent wraps each automatically
all_tools = mcp_tools + [
    researcher,    # @dura → Child Workflow
    calculator,    # @tool → dura__tool Activity
]
# MCP tools from mcp_tools → dura__tool Activity

@dura
async def orchestrator(task: str) -> str:
    agent = dura_agent(
        model="claude-sonnet-4-6",
        tools=all_tools,
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})
    return result["messages"][-1].content
```

The LLM sees a flat list of tools. DuraLang routes each call to the right Temporal primitive automatically. Every operation is individually durable.
