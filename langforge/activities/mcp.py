"""forge__mcp activity — executes an MCP server tool call."""

from __future__ import annotations

from temporalio import activity

from langforge.exceptions import MCPActivityError
from langforge.graph_def import MCPActivityPayload, MCPActivityResult
from langforge.registry import MCPSessionRegistry


@activity.defn(name="forge__mcp")
async def mcp_activity(payload: MCPActivityPayload) -> MCPActivityResult:
    """Execute a single MCP tool call.

    Routes to the registered MCP session for the given server name.
    """
    activity.heartbeat(f"mcp:{payload.server_name}:{payload.tool_name}")

    session = MCPSessionRegistry.get(payload.server_name)
    if session is None:
        raise MCPActivityError(f"MCP server '{payload.server_name}' not registered")

    result = await session.call_tool(payload.tool_name, payload.arguments)

    activity.heartbeat(f"mcp:{payload.server_name}:{payload.tool_name} complete")

    return MCPActivityResult(
        content=[c.model_dump() for c in result.content],
        tool_call_id=payload.tool_call_id,
        is_error=getattr(result, "isError", False) or False,
    )
