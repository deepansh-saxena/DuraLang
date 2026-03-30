"""DuraTool — context-aware tool wrapper that routes through Temporal activities."""

from __future__ import annotations

from typing import Any, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from duralang.context import DuraContext
from duralang.registry import ToolRegistry


class DuraTool(BaseTool):
    """A BaseTool that routes _arun() through Temporal when inside @dura."""

    inner_tool: BaseTool
    """The actual tool to delegate to."""

    name: str = ""
    description: str = ""
    args_schema: Type[BaseModel] | None = None

    def __init__(self, tool: BaseTool, **kwargs):
        super().__init__(
            inner_tool=tool,
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            **kwargs,
        )
        # Register in ToolRegistry for activity-side lookup
        ToolRegistry.register(tool)

    def _run(self, *args, **kwargs):
        raise NotImplementedError("DuraTool only supports async. Use ainvoke().")

    async def _arun(self, *args, **kwargs) -> str:
        ctx = DuraContext.get()
        if ctx is None:
            # Outside dura context — passthrough via ainvoke to preserve
            # LangChain's full invocation chain (config, callbacks, etc.)
            return await self.inner_tool.ainvoke(kwargs if kwargs else (args[0] if args else {}))

        # Inside dura context — route through Temporal activity
        from langchain_core.messages import ToolMessage

        from duralang.graph_def import ToolActivityPayload

        tool_input = kwargs if kwargs else (args[0] if args else {})

        # Extract tool_call_id from input (ToolCall dict) or kwargs
        tool_call_id = ""
        if isinstance(tool_input, dict) and "id" in tool_input:
            tool_call_id = tool_input["id"]
        if not tool_call_id:
            tool_call_id = kwargs.get("tool_call_id", "")

        payload = ToolActivityPayload(
            tool_name=self.name,
            tool_input=tool_input,
            tool_call_id=tool_call_id,
        )

        result = await ctx.execute_activity("dura__tool", payload, ctx.config.tool_config)

        content = result.error if result.error else result.output
        status = "error" if result.error else "success"

        # Return ToolMessage to match what BaseTool.ainvoke normally returns.
        # LangGraph's ToolNode expects ToolMessage, not raw strings.
        if tool_call_id:
            return ToolMessage(
                content=content,
                tool_call_id=tool_call_id,
                name=self.name,
                status=status,
            )
        return content
