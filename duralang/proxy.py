"""Proxy objects — MCP proxy and Temporal compatibility patches.

DuraMCPProxy wraps mcp.ClientSession.call_tool() to route through Temporal
activities when inside a @dura context.

_install_eager_task_patch fixes a Python 3.12+ incompatibility between
LangGraph's task creation and Temporal's workflow event loop.
"""

from __future__ import annotations

import functools

from duralang.context import DuraContext
from duralang.graph_def import (
    MCPActivityPayload,
    MCPActivityResult,
)


class DuraMCPProxy:
    """Intercepts mcp.ClientSession.call_tool() inside a dura context."""

    @staticmethod
    def install(session, server_name: str) -> None:
        """Install proxy on an MCP ClientSession instance."""
        original_call_tool = session.call_tool

        @functools.wraps(original_call_tool)
        async def call_tool_proxy(tool_name, arguments=None, *args, **kwargs):
            ctx = DuraContext.get()
            if ctx is None:
                return await original_call_tool(tool_name, arguments, *args, **kwargs)

            result: MCPActivityResult = await ctx.execute_activity(
                "dura__mcp",
                MCPActivityPayload(
                    server_name=server_name,
                    tool_name=tool_name,
                    arguments=arguments or {},
                    tool_call_id=kwargs.get("tool_call_id", ""),
                ),
                ctx.config.mcp_config,
            )
            return _reconstruct_mcp_result(result)

        session.call_tool = call_tool_proxy


def _reconstruct_mcp_result(result: MCPActivityResult):
    """Reconstruct an MCP-like result object from MCPActivityResult."""
    from types import SimpleNamespace

    content_items = []
    for c in result.content:
        content_items.append(SimpleNamespace(**c))

    return SimpleNamespace(
        content=content_items,
        isError=result.is_error,
    )


# ── Temporal compatibility patches ───────────────────────────────────────────

_eager_task_patched = False


def _install_eager_task_patch() -> None:
    """Patch asyncio.eager_task_factory for Temporal compatibility.

    On Python 3.12+, LangGraph uses asyncio.eager_task_factory() to create
    tasks. This bypasses loop.create_task(), so tasks are NOT registered in
    Temporal's _tasks set. Without a strong reference, these tasks get
    garbage-collected while still pending — causing "Task was destroyed but
    it is pending!" warnings and silently lost work.

    This patch detects Temporal's workflow event loop (via the
    __temporal_workflow_runtime attribute) and routes through
    loop.create_task() instead, ensuring proper task registration.
    Non-Temporal event loops are unaffected.
    """
    global _eager_task_patched
    if _eager_task_patched:
        return
    _eager_task_patched = True

    import asyncio
    import sys

    if sys.version_info < (3, 12):
        return  # eager_task_factory doesn't exist before 3.12

    _original_eager_factory = asyncio.eager_task_factory

    def _temporal_safe_eager_factory(loop, coro, *, name=None, context=None):
        if hasattr(loop, "__temporal_workflow_runtime"):
            return loop.create_task(coro, name=name, context=context)
        return _original_eager_factory(loop, coro, name=name, context=context)

    asyncio.eager_task_factory = _temporal_safe_eager_factory
