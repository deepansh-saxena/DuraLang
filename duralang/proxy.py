"""Proxy objects — intercept LLM, Tool, and MCP calls inside @dura context.

Proxies are installed at __init__ time via method wrapping on BaseChatModel
and BaseTool. When DuraContext is set (inside @dura workflow), calls are
routed to Temporal Activities. When no context is set, calls pass through
to the original methods normally.
"""

from __future__ import annotations

import functools
from typing import Any

from duralang.config import LLMIdentity
from duralang.context import DuraContext
from duralang.graph_def import (
    LLMActivityPayload,
    LLMActivityResult,
    MCPActivityPayload,
    MCPActivityResult,
    ToolActivityPayload,
    ToolActivityResult,
)
from duralang.registry import ToolRegistry
from duralang.state import MessageSerializer


def _extract_bound_tool_schemas(llm_instance) -> list[dict]:
    """Extracts tool schemas from a BaseChatModel that has had bind_tools() called.

    Also auto-registers each BaseTool in ToolRegistry so dura__tool can find it.
    """
    tools = getattr(llm_instance, "tools", None) or getattr(
        llm_instance, "_tools", None
    ) or []

    schemas = []
    for tool in tools:
        if hasattr(tool, "name") and hasattr(tool, "args_schema"):
            # It's a BaseTool — register it and get its schema
            ToolRegistry.register(tool)
            if tool.args_schema:
                schemas.append(tool.args_schema.model_json_schema())
        elif isinstance(tool, dict):
            schemas.append(tool)

    return schemas


def _safe_kwargs(kwargs: dict) -> dict:
    """Filter kwargs to only JSON-serializable values."""
    safe = {}
    for k, v in kwargs.items():
        if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
            safe[k] = v
    return safe


class DuraLLMProxy:
    """Intercepts BaseChatModel.ainvoke() inside a dura context."""

    @staticmethod
    def make_ainvoke(original_ainvoke, original_instance):
        @functools.wraps(original_ainvoke)
        async def ainvoke_proxy(messages, *args, **kwargs):
            ctx = DuraContext.get()
            if ctx is None:
                return await original_ainvoke(messages, *args, **kwargs)

            llm_identity = LLMIdentity.from_instance(original_instance)
            tool_schemas = _extract_bound_tool_schemas(original_instance)

            serialized_messages = MessageSerializer.serialize_many(
                messages if isinstance(messages, list) else [messages]
            )

            result: LLMActivityResult = await ctx.execute_activity(
                "dura__llm",
                LLMActivityPayload(
                    messages=serialized_messages,
                    llm_identity={
                        "provider": llm_identity.provider,
                        "model": llm_identity.model,
                        "kwargs": llm_identity.kwargs,
                    },
                    tool_schemas=tool_schemas,
                    invoke_kwargs=_safe_kwargs(kwargs),
                ),
                ctx.config.llm_config,
            )
            return MessageSerializer.deserialize(result.ai_message)

        return ainvoke_proxy


class DuraToolProxy:
    """Intercepts BaseTool.ainvoke() inside a dura context."""

    @staticmethod
    def make_ainvoke(original_ainvoke, original_instance):
        @functools.wraps(original_ainvoke)
        async def ainvoke_proxy(input, *args, **kwargs):
            ctx = DuraContext.get()
            if ctx is None:
                return await original_ainvoke(input, *args, **kwargs)

            # Agent tools handle their own durability via child workflows —
            # skip dura__tool activity routing and call _arun directly.
            if getattr(original_instance, "__dura_agent_tool__", False):
                return await original_ainvoke(input, *args, **kwargs)

            tool_input = input if isinstance(input, (dict, str)) else str(input)

            result: ToolActivityResult = await ctx.execute_activity(
                "dura__tool",
                ToolActivityPayload(
                    tool_name=original_instance.name,
                    tool_input=tool_input,
                    tool_call_id=kwargs.get("tool_call_id", ""),
                ),
                ctx.config.tool_config,
            )
            if result.error:
                return result.error
            return result.output

        return ainvoke_proxy


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


def _install_llm_proxy(instance) -> None:
    """Replace ainvoke on a BaseChatModel instance with proxy version."""
    original_ainvoke = instance.ainvoke
    # Use object.__setattr__ to bypass Pydantic's field validation
    object.__setattr__(
        instance, "ainvoke", DuraLLMProxy.make_ainvoke(original_ainvoke, instance)
    )


def _install_tool_proxy(instance) -> None:
    """Replace ainvoke on a BaseTool instance with proxy version."""
    original_ainvoke = instance.ainvoke
    # Use object.__setattr__ to bypass Pydantic's field validation
    object.__setattr__(
        instance, "ainvoke", DuraToolProxy.make_ainvoke(original_ainvoke, instance)
    )
    # Auto-register tool
    ToolRegistry.register(instance)


# ── Monkey-patching at import time ──────────────────────────────────────────

_patched = False


def install_patches() -> None:
    """Patch BaseChatModel.__init__ and BaseTool.__init__ to install proxies.

    Called at duralang import time. Safe to call multiple times.
    """
    global _patched
    if _patched:
        return
    _patched = True

    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool

    _original_chat_model_init = BaseChatModel.__init__

    @functools.wraps(_original_chat_model_init)
    def _patched_chat_model_init(self, *args, **kwargs):
        _original_chat_model_init(self, *args, **kwargs)
        _install_llm_proxy(self)

    BaseChatModel.__init__ = _patched_chat_model_init

    _original_tool_init = BaseTool.__init__

    @functools.wraps(_original_tool_init)
    def _patched_tool_init(self, *args, **kwargs):
        _original_tool_init(self, *args, **kwargs)
        _install_tool_proxy(self)

    BaseTool.__init__ = _patched_tool_init
