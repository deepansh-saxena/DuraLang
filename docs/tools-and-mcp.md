# Tools & MCP

DuraLang supports two types of tool integrations: **LangChain tools** and **MCP (Model Context Protocol) servers**. Both are intercepted automatically inside `@dura` functions.

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
        result = await get_weather.arun(tc["args"])  # -> Temporal Activity
        ...
```

### Parallel Tool Calls

If the LLM returns multiple tool calls, you can execute them in parallel. `asyncio.gather` works as expected:

```python
import asyncio

tasks = [tools_by_name[tc["name"]].arun(tc["args"]) for tc in response.tool_calls]
results = await asyncio.gather(*tasks)
```

Each tool call becomes its own `dura__tool` Activity, scheduled in parallel by Temporal.

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

- **LangChain tools**: Proxy intercepts `BaseTool.ainvoke()` -> `dura__tool` Activity
- **MCP tools**: Proxy intercepts `ClientSession.call_tool()` -> `dura__mcp` Activity

The `dura__tool` Activity looks up the tool by name in `ToolRegistry`. The `dura__mcp` Activity looks up the session by server name in `MCPSessionRegistry`.
