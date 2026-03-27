# Examples

Complete working examples demonstrating DuraLang's capabilities.

---

## Basic Agent

A standard LangChain agent loop with `@dura`. Every LLM call and tool call becomes a Temporal Activity.

```python
# examples/basic_agent.py
import asyncio
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, ToolMessage
from duralang import dura

tools = [TavilySearchResults(max_results=3)]
tools_by_name = {t.name: t for t in tools}

@dura
async def research_agent(messages: list) -> list:
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools(tools)

    while True:
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            result = await tools_by_name[tc["name"]].ainvoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return messages

async def main():
    result = await research_agent([HumanMessage(content="AI agent news?")])
    print(result[-1].content)

asyncio.run(main())
```

---

## Multi-Tool with Parallel Execution

Multiple tools, parallel execution via `asyncio.gather`. Each tool call runs as its own Temporal Activity in parallel.

See `examples/multi_tool.py`.

---

## Multi-Agent with Agent Tools

`dura_agent_tool()` wraps `@dura` functions as real `BaseTool` instances. Sub-agents and regular tools go in the same list, same `bind_tools()`, same `ainvoke()` loop. Any agent becomes an orchestrator the moment you add sub-agent tools to its toolkit.

```python
# examples/stochastic_agents.py
from duralang import dura, dura_agent_tool

@dura
async def researcher(query: str) -> str:
    """Research agent — gathers information via web search."""
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools([web_search, wikipedia_lookup])
    # ... standard agent loop ...
    return response.content

@dura
async def analyst(data: str, question: str) -> str:
    """Analysis agent — runs calculations and identifies trends."""
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools([calculator])
    # ... standard agent loop ...
    return response.content

# Sub-agents + regular tools in the SAME list
all_tools = [
    dura_agent_tool(researcher),   # → Child Workflow
    dura_agent_tool(analyst),      # → Child Workflow
    calculator,                     # → Temporal Activity
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
            # Same ainvoke() for both — routing is automatic
            result = await tools_by_name[tc["name"]].ainvoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return response.content
```

The LLM decides which agents and tools to call, in what order, and how many times. Every operation is individually durable. See `examples/stochastic_agents.py` for the full working example.

---

## MCP Agent

Using `DuraMCPSession` to make MCP server calls durable.

See `examples/mcp_agent.py`.

---

## Multi-Model

Same agent code, different LLM providers. DuraLang auto-detects the provider.

See `examples/multi_model.py`.
