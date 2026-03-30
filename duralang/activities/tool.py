"""dura__tool activity — executes a LangChain tool."""

from __future__ import annotations

import asyncio

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

    # Wait for the tool to appear in the registry. After a crash, Temporal
    # retries pending activities immediately — potentially before the workflow
    # replay has called dura_agent() which registers the tools. The workflow
    # task may take up to 10s to be dispatched (sticky task timeout), so we
    # wait generously.
    tool = ToolRegistry.get(payload.tool_name)
    if tool is None:
        for _ in range(60):
            await asyncio.sleep(0.5)
            activity.heartbeat(f"tool:{payload.tool_name} waiting for registry")
            tool = ToolRegistry.get(payload.tool_name)
            if tool is not None:
                break

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
    except asyncio.CancelledError:
        activity.heartbeat(f"tool:{payload.tool_name} cancelled — cleaning up")
        raise
    except (ValueError, TypeError, KeyError) as e:
        # These are returned as tool output (not raised) so the LLM can
        # self-correct — e.g. fix a malformed expression and retry.
        return ToolActivityResult(
            output="",
            tool_call_id=payload.tool_call_id,
            error=str(e),
        )
    except Exception:
        raise
