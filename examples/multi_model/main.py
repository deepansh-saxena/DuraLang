"""Multi-model example — same graph, different LLM providers."""

import asyncio
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from langforge import ForgeConfig, ForgeRuntime, LLMConfig


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


async def agent_node(state: AgentState, llm=None) -> dict:
    """Simple agent node — LLM is injected by LangForge."""
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}


async def main():
    config = ForgeConfig(temporal_host="localhost:7233")

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.set_entry_point("agent")
    builder.add_edge("agent", END)
    graph = builder.compile()

    initial_state = {
        "messages": [HumanMessage(content="What is 2 + 2?")]
    }

    configs = [
        LLMConfig(provider="anthropic", model="claude-sonnet-4-6"),
        LLMConfig(provider="openai", model="gpt-4o"),
    ]

    async with ForgeRuntime(config) as runtime:
        graph_id = runtime.register_graph(graph)

        for llm_config in configs:
            print(f"\n--- {llm_config.provider}/{llm_config.model} ---")
            result = await runtime.run(
                graph_id=graph_id,
                initial_state=initial_state,
                llm_config=llm_config,
            )
            print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
