"""Tool and MCP session registries — module-level singletons."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from mcp import ClientSession


class ToolRegistry:
    """Global registry for LangChain tools. Accessed inside dura__tool activity.

    Auto-populated by proxy installation — no explicit register_tools() needed.
    """

    _registry: dict[str, BaseTool] = {}

    @classmethod
    def register(cls, tool: BaseTool) -> None:
        cls._registry[tool.name] = tool

    @classmethod
    def get(cls, name: str) -> BaseTool | None:
        return cls._registry.get(name)

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()


class MCPSessionRegistry:
    """Global registry for MCP client sessions. Accessed inside dura__mcp activity."""

    _registry: dict[str, ClientSession] = {}

    @classmethod
    def register(cls, server_name: str, session: ClientSession) -> None:
        """Register an MCP session. Called by DuraMCPSession."""
        cls._registry[server_name] = session

    @classmethod
    def get(cls, server_name: str) -> ClientSession | None:
        return cls._registry.get(server_name)

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()
