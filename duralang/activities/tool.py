"""dura__tool activity — executes a LangChain tool."""

from __future__ import annotations

from temporalio import activity

from duralang.exceptions import ToolActivityError
from duralang.graph_def import ToolActivityPayload, ToolActivityResult
from duralang.registry import ToolRegistry


@activity.defn(name="dura__tool")
async def tool_activity(payload: ToolActivityPayload) -> ToolActivityResult:
    """Execute a single tool call.

    Non-retryable errors (ValueError, TypeError, KeyError) are caught and returned as error results.
    Retryable errors (network, timeout) are re-raised for Temporal retry.
    """
    activity.heartbeat(f"tool:{payload.tool_name} starting")

    tool = ToolRegistry.get(payload.tool_name)
    if tool is None:
        raise ToolActivityError(
            f"Tool '{payload.tool_name}' not in registry. "
            f"Ensure the tool is created inside your @dura function."
        )

    try:
        output = await tool.ainvoke(payload.tool_input)
        activity.heartbeat(f"tool:{payload.tool_name} complete")
        return ToolActivityResult(
            output=str(output),
            tool_call_id=payload.tool_call_id,
        )
    except (ValueError, TypeError, KeyError) as e:
        return ToolActivityResult(
            output="",
            tool_call_id=payload.tool_call_id,
            error=str(e),
        )
    except Exception:
        raise
