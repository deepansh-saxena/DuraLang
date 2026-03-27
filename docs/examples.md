# Examples

Complete working examples demonstrating DuraLang's capabilities.

---

## Basic Agent

A simple agent with one tool. Identical LangChain code with `@dura`.

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
            result = await tools_by_name[tc["name"]].arun(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return messages

async def main():
    result = await research_agent([HumanMessage(content="AI agent news?")])
    print(result[-1].content)

asyncio.run(main())
```

---

## Multi-Tool with Parallel Execution

Multiple tools, parallel execution via `asyncio.gather`.

See `examples/multi_tool.py`.

---

## Multi-Agent

`@dura` functions calling other `@dura` functions become Child Workflows.

```python
@dura
async def researcher(messages: list) -> list:
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    llm_with_tools = llm.bind_tools([search_tool])
    # ... agent loop ...
    return messages

@dura
async def orchestrator(task: str) -> str:
    research = await researcher([HumanMessage(content=f"Research: {task}")])
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    response = await llm.ainvoke([HumanMessage(content=f"Summarize: {research[-1].content}")])
    return response.content
```

See `examples/multi_agent.py`.

---

## MCP Agent

Using `DuraMCPSession` to make MCP calls durable.

See `examples/mcp_agent.py`.

---

## Multi-Model

Same agent code, different LLM providers.

See `examples/multi_model.py`.
