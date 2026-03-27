# Tools & MCP

DuraLang supports three types of tool integrations: **LangChain tools**, **agent tools** (sub-agents as tools), and **MCP servers**. All are intercepted automatically inside `@dura` functions, and all can be mixed freely in the same agent.

---

## LangChain Tools

Standard LangChain `@tool` functions work inside `@dura` with zero changes. Every `tool.ainvoke()` call becomes a `dura__tool` Temporal Activity — automatically retried, heartbeated, and checkpointed.

```python
from langchain_core.tools import tool
from duralang import dura

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"72°F and sunny in {city}"

@tool
def get_time(timezone: str) -> str:
    """Get the current time in a timezone."""
    return f"3:00 PM in {timezone}"

tools = [get_weather, get_time]
tools_by_name = {t.name: t for t in tools}

@dura
async def my_agent(messages):
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools(tools)

    while True:
        response = await llm_with_tools.ainvoke(messages)   # → dura__llm Activity
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            result = await tools_by_name[tc["name"]].ainvoke(tc["args"])  # → dura__tool Activity
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return messages
```

### Auto-Registration

Tools are auto-registered in DuraLang's internal `ToolRegistry` at two points:

1. **At instantiation** — `BaseTool.__init__` is patched to register every tool when it's created
2. **At `bind_tools()` time** — the proxy extracts tool schemas and registers any `BaseTool` instances it finds

You never call `register_tools()`. There is no such function. Registration is fully automatic.

### Parallel Tool Calls

If the LLM returns multiple tool calls, you can execute them in parallel with `asyncio.gather`. Each tool call becomes its own `dura__tool` Activity, scheduled concurrently by Temporal:

```python
import asyncio

@dura
async def my_agent(messages):
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools(tools)

    while True:
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break

        # Parallel — each becomes an independent durable Activity
        tasks = [
            tools_by_name[tc["name"]].ainvoke(tc["args"])
            for tc in response.tool_calls
        ]
        results = await asyncio.gather(*tasks)

        for tc, result in zip(response.tool_calls, results):
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return messages
```

---

## Agent Tools — Sub-Agents as Tools

`dura_agent_tool()` wraps a `@dura` function as a real LangChain `BaseTool`. This is the key to building dynamic multi-agent systems where the LLM decides which agents to invoke, how many times, and in what order.

### Why Agent Tools Exist

Without `dura_agent_tool()`, you have two patterns for multi-agent:

1. **Direct `@dura` → `@dura` calls** — your code decides which agent to call. Simple, but rigid.
2. **Manual tool wrapping** — write boilerplate to expose each agent as a tool. Tedious and error-prone.

`dura_agent_tool()` gives you a third pattern: **LLM-driven agent delegation** with zero boilerplate. Sub-agents and regular tools go in the same list, the same `bind_tools()` call, and the same `ainvoke()` dispatch loop.

### Usage

```python
from duralang import dura, dura_agent_tool

@dura
async def researcher(query: str) -> str:
    """Research agent — gathers information via web search."""
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools([web_search, wikipedia_lookup])

    messages = [HumanMessage(content=query)]
    for _ in range(10):
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            result = await tools_by_name[tc["name"]].ainvoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return response.content

@dura
async def analyst(data: str, question: str) -> str:
    """Analysis agent — processes data with calculations."""
    ...

# Mix sub-agents and regular tools in the SAME list
all_tools = [
    dura_agent_tool(researcher),   # → Temporal Child Workflow
    dura_agent_tool(analyst),      # → Temporal Child Workflow
    calculator,                     # → dura__tool Activity
]
tools_by_name = {t.name: t for t in all_tools}

@dura
async def orchestrator(task: str) -> str:
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools(all_tools)

    messages = [HumanMessage(content=task)]
    while True:
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            # Same ainvoke() — DuraLang routes automatically
            result = await tools_by_name[tc["name"]].ainvoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return response.content
```

### How It Works Under the Hood

1. **Schema generation** — `dura_agent_tool()` reads the function's signature, type hints, and docstring, then generates a Pydantic `args_schema` automatically. No manual schema authoring.

2. **Tool creation** — Returns a `BaseTool` subclass with `name`, `description`, and `args_schema` set. The tool's `_arun()` method calls the `@dura` function.

3. **Routing bypass** — The tool instance is marked with `__dura_agent_tool__ = True`. When `DuraToolProxy` detects this flag, it **skips** `dura__tool` routing and calls the original `ainvoke()` directly. This triggers `_arun()`, which calls the `@dura` function, which the decorator routes as a Child Workflow.

4. **Independent durability** — Each sub-agent gets its own Temporal workflow with its own event history. A failure in one sub-agent retries only that sub-agent.

### Naming Convention

By default, `dura_agent_tool(researcher)` creates a tool named `call_researcher`. Override with:

```python
dura_agent_tool(researcher, name="search", description="Search the web for information.")
```

### Nesting

Agent tools nest to any depth. If an orchestrator calls a researcher via `dura_agent_tool()`, and the researcher calls a fact-checker via `dura_agent_tool()`, each level is independently durable:

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

## MCP Servers

MCP (Model Context Protocol) servers are first-class citizens in DuraLang. Wrap a `ClientSession` with `DuraMCPSession`, and every `call_tool()` becomes a `dura__mcp` Temporal Activity.

### Setup

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from duralang import dura, DuraMCPSession

server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        fs = DuraMCPSession(session, "filesystem")  # ← one line

        @dura
        async def my_agent(messages, fs):
            # list_tools() passes through to the original session
            tools_result = await fs.list_tools()

            # call_tool() is intercepted → dura__mcp Activity
            result = await fs.call_tool("read_file", {"path": "/tmp/data.csv"})
            return result
```

### What `DuraMCPSession` Does

1. **Registers** the session in `MCPSessionRegistry` with the given server name
2. **Installs** a proxy on `call_tool()` that routes through Temporal when inside `@dura`
3. **Passes through** all other methods (`list_tools()`, `initialize()`, etc.) unchanged

### Multiple MCP Servers

You can use multiple MCP servers in the same agent. Each gets its own server name:

```python
fs = DuraMCPSession(session1, "filesystem")
db = DuraMCPSession(session2, "database")
git = DuraMCPSession(session3, "github")

@dura
async def my_agent(messages):
    files = await fs.call_tool("list_files", {"path": "/tmp"})     # → dura__mcp
    data = await db.call_tool("query", {"sql": "SELECT * ..."})    # → dura__mcp
    issues = await git.call_tool("list_issues", {"repo": "..."})   # → dura__mcp
```

---

## Complete Routing Table

All three tool types can coexist in the same agent. DuraLang routes each call to the correct Temporal primitive automatically:

| Call Type | Proxy | Temporal Primitive | Event History |
|---|---|---|---|
| `@tool` function via `.ainvoke()` | `DuraToolProxy` | `dura__tool` Activity | Parent workflow |
| `dura_agent_tool(fn).ainvoke()` | Skips proxy → calls `@dura` fn | Child Workflow | Own workflow |
| `DuraMCPSession.call_tool()` | `DuraMCPProxy` | `dura__mcp` Activity | Parent workflow |
| `@dura` calling `@dura` directly | Decorator detects context | Child Workflow | Own workflow |

The dispatch loop is identical for all tool types:

```python
for tc in response.tool_calls:
    result = await tools_by_name[tc["name"]].ainvoke(tc["args"])
    # ↑ DuraLang decides: dura__tool Activity or Child Workflow
    messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
```

---

## Mixing Everything Together

A single agent can use all three tool types simultaneously:

```python
from duralang import dura, dura_agent_tool, DuraMCPSession

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

# Build the tool list
all_tools = [
    dura_agent_tool(researcher),   # → Child Workflow
    calculator,                     # → dura__tool Activity
]
tools_by_name = {t.name: t for t in all_tools}

# MCP server
fs = DuraMCPSession(mcp_session, "filesystem")

@dura
async def orchestrator(task: str) -> str:
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools(all_tools)

    messages = [HumanMessage(content=task)]
    while True:
        response = await llm_with_tools.ainvoke(messages)       # → dura__llm
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            result = await tools_by_name[tc["name"]].ainvoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    # MCP call alongside everything else
    data = await fs.call_tool("read_file", {"path": "/tmp/report.csv"})  # → dura__mcp
    return response.content
```

The LLM sees a flat list of tools. DuraLang routes each call to the right Temporal primitive automatically. Every operation is individually durable.
