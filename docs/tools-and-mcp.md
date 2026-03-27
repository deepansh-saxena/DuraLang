# Tools & MCP

DuraLang supports three types of tool integrations: **LangChain tools**, **agent tools** (sub-agents as tools), and **MCP (Model Context Protocol) servers**. All are intercepted automatically inside `@dura` functions and can be mixed freely.

---

## LangChain Tools

Tools are auto-registered when they are used inside a `@dura` function. No explicit registration needed.

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"72F and sunny in {city}"

@dura
async def my_agent(messages):
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools([get_weather])

    response = await llm_with_tools.ainvoke(messages)
    # If the LLM calls get_weather, it becomes a dura__tool Activity
    for tc in response.tool_calls:
        result = await get_weather.ainvoke(tc["args"])  # -> Temporal Activity
        ...
```

### Parallel Tool Calls

If the LLM returns multiple tool calls, you can execute them in parallel. `asyncio.gather` works as expected:

```python
import asyncio

tasks = [tools_by_name[tc["name"]].ainvoke(tc["args"]) for tc in response.tool_calls]
results = await asyncio.gather(*tasks)
```

Each tool call becomes its own `dura__tool` Activity, scheduled in parallel by Temporal.

---

## Agent Tools — Sub-Agents as Tools

`dura_agent_tool()` wraps a `@dura` function as a real LangChain `BaseTool`. This lets you mix sub-agents and regular tools in the **same list**, **same `bind_tools()` call**, and **same `ainvoke()` dispatch loop**.

Under the hood, agent tool calls become **Temporal Child Workflows** (not `dura__tool` activities), giving each sub-agent its own event history, timeouts, and retry boundaries.

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

# Mix agent tools and regular tools in the same list
all_tools = [
    dura_agent_tool(researcher),   # → Child Workflow
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
            # Same ainvoke() — routing happens automatically
            result = await tools_by_name[tc["name"]].ainvoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return response.content
```

### How `dura_agent_tool()` works

1. Reads the `@dura` function's **signature**, **type hints**, and **docstring**
2. Generates a Pydantic `args_schema` and tool schema automatically
3. Returns a real `BaseTool` whose `_arun()` calls the `@dura` function
4. When called inside a `@dura` context, the call routes as a Child Workflow (not a `dura__tool` activity), giving the sub-agent its own event history

### Naming

By default, `dura_agent_tool(researcher)` creates a tool named `call_researcher`. Override with:

```python
dura_agent_tool(researcher, name="search", description="Search the web for info.")
```

---

## MCP Servers

MCP sessions are wrapped with `DuraMCPSession` to enable interception:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from duralang import dura, DuraMCPSession

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        fs = DuraMCPSession(session, "filesystem")  # <- one line

        @dura
        async def my_agent(messages, fs):
            result = await fs.call_tool("read_file", {"path": "/tmp/data.csv"})
            # -> dura__mcp Activity
```

`DuraMCPSession` is a thin wrapper that:
1. Registers the session in `MCPSessionRegistry` with the server name
2. Installs a proxy on `call_tool()` that routes through Temporal when inside `@dura`
3. Passes all other methods through to the underlying session

---

## How Routing Works

| Call Type | Proxy | Temporal Primitive |
|---|---|---|
| `@tool` function `.ainvoke()` | `DuraToolProxy` | `dura__tool` Activity |
| `dura_agent_tool(fn).ainvoke()` | Skips proxy, calls `@dura` directly | Child Workflow |
| `mcp_session.call_tool()` | `DuraMCPProxy` | `dura__mcp` Activity |

The `dura__tool` Activity looks up the tool by name in `ToolRegistry`. The `dura__mcp` Activity looks up the session by server name in `MCPSessionRegistry`. Agent tools bypass both registries — they call the `@dura` function directly, which the decorator routes as a Child Workflow.

---

## Mixing Everything Together

All three tool types can coexist in the same agent:

```python
all_tools = [
    dura_agent_tool(researcher),   # → Child Workflow
    dura_agent_tool(writer),       # → Child Workflow
    calculator,                     # → dura__tool Activity
]

@dura
async def orchestrator(task: str) -> str:
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools(all_tools)
    tools_by_name = {t.name: t for t in all_tools}

    # MCP server for file access
    fs = DuraMCPSession(mcp_session, "filesystem")

    messages = [HumanMessage(content=task)]
    while True:
        response = await llm_with_tools.ainvoke(messages)      # → dura__llm
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            result = await tools_by_name[tc["name"]].ainvoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    # MCP calls work alongside everything else
    data = await fs.call_tool("read_file", {"path": "/tmp/report.csv"})  # → dura__mcp
    return response.content
```

The LLM sees a flat list of tools. DuraLang routes each call to the right Temporal primitive automatically.
