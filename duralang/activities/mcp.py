"""dura__mcp activity — executes an MCP server tool call."""

from __future__ import annotations

from temporalio import activity

from duralang.exceptions import MCPActivityError
from duralang.graph_def import MCPActivityPayload, MCPActivityResult
from duralang.registry import MCPSessionRegistry


@activity.defn(name="dura__mcp")
async def mcp_activity(payload: MCPActivityPayload) -> MCPActivityResult:
    """Execute a single MCP tool call."""
    activity.heartbeat(f"mcp:{payload.server_name}:{payload.tool_name} starting")

    session = MCPSessionRegistry.get(payload.server_name)
    if session is None:
        raise MCPActivityError(
            f"MCP server '{payload.server_name}' not registered. "
            f"Set session.dura_server_name = '{payload.server_name}'."
        )

    result = await session.call_tool(payload.tool_name, payload.arguments)

    activity.heartbeat(f"mcp:{payload.server_name}:{payload.tool_name} complete")

    return MCPActivityResult(
        content=[c.model_dump() for c in result.content],
        tool_call_id=payload.tool_call_id,
        is_error=getattr(result, "isError", False) or False,
    )
