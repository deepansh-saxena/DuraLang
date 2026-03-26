"""forge__tool activity — executes a LangChain tool."""

from __future__ import annotations

from temporalio import activity

from langforge.exceptions import ToolActivityError
from langforge.graph_def import ToolActivityPayload, ToolActivityResult
from langforge.registry import ToolRegistry


@activity.defn(name="forge__tool")
async def tool_activity(payload: ToolActivityPayload) -> ToolActivityResult:
    """Execute a single tool call.

    Non-retryable errors (ToolException, ValueError) are caught and returned as error results.
    Retryable errors (network, timeout) are re-raised for Temporal retry.
    """
    activity.heartbeat(f"tool:{payload.tool_name}")

    tool = ToolRegistry.get(payload.tool_name)
    if tool is None:
        raise ToolActivityError(f"Tool '{payload.tool_name}' not registered")

    try:
        output = await tool.ainvoke(payload.tool_input)
        activity.heartbeat(f"tool:{payload.tool_name} complete")
        return ToolActivityResult(
            output=str(output),
            tool_call_id=payload.tool_call_id,
        )
    except (ValueError, TypeError, KeyError) as e:
        # Non-retryable — logic error in tool invocation
        return ToolActivityResult(
            output="",
            tool_call_id=payload.tool_call_id,
            error=str(e),
        )
    except Exception:
        # Retryable — network error, timeout, etc.
        raise
