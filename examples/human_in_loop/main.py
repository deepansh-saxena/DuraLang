"""Human-in-the-loop example — pauses workflow for human input via Temporal signals."""

import asyncio

from langforge import ForgeConfig, ForgeRuntime, LLMConfig


async def main():
    config = ForgeConfig(temporal_host="localhost:7233")

    # Example: start a workflow, then send a human input signal
    # async with ForgeRuntime(config) as runtime:
    #     graph_id = runtime.register_graph(graph)
    #     workflow_id = await runtime.start(
    #         graph_id=graph_id,
    #         initial_state={"messages": [...]},
    #         llm_config=LLMConfig(provider="anthropic", model="claude-sonnet-4-6"),
    #     )
    #
    #     # ... workflow pauses at human-in-the-loop node ...
    #
    #     await runtime.send_signal(
    #         workflow_id, "human_input", "Yes, proceed with the plan"
    #     )
    #
    #     result = await runtime.get_result(workflow_id)

    print("Human-in-the-loop example — see comments in source for usage pattern")


if __name__ == "__main__":
    asyncio.run(main())
