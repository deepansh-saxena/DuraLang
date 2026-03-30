"""Tool and MCP session registries — module-level singletons."""

from __future__ import annotations

import threading
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from mcp import ClientSession


class ToolRegistry:
    """Global registry for LangChain tools. Accessed inside dura__tool activity.

    Auto-populated by proxy installation — no explicit register_tools() needed.
    Thread-safe via RLock (Temporal runs activities in a thread pool).
    """

    _registry: dict[str, BaseTool] = {}
    _schema_cache: dict[str, dict] = {}
    _lock = threading.RLock()

    @classmethod
    def register(cls, tool: BaseTool) -> None:
        with cls._lock:
            if tool.name in cls._registry and cls._registry[tool.name] is not tool:
                warnings.warn(
                    f"Tool '{tool.name}' already registered, overwriting.",
                    stacklevel=2,
                )
            cls._registry[tool.name] = tool
            if tool.args_schema and tool.name not in cls._schema_cache:
                cls._schema_cache[tool.name] = tool.args_schema.model_json_schema()

    @classmethod
    def get(cls, name: str) -> BaseTool | None:
        with cls._lock:
            return cls._registry.get(name)

    @classmethod
    def get_schema(cls, name: str) -> dict | None:
        with cls._lock:
            return cls._schema_cache.get(name)

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._registry.clear()
            cls._schema_cache.clear()


class MCPSessionRegistry:
    """Global registry for MCP client sessions. Accessed inside dura__mcp activity."""

    _registry: dict[str, ClientSession] = {}
    _lock = threading.RLock()

    @classmethod
    def register(cls, server_name: str, session: ClientSession) -> None:
        """Register an MCP session. Called by DuraMCPSession."""
        with cls._lock:
            cls._registry[server_name] = session

    @classmethod
    def get(cls, server_name: str) -> ClientSession | None:
        with cls._lock:
            return cls._registry.get(server_name)

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._registry.clear()
