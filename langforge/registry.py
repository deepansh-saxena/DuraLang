"""Tool and MCP session registries — module-level, populated at worker startup."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from mcp import ClientSession


class ToolRegistry:
    """Global registry for LangChain tools. Accessed inside forge__tool activity."""

    _registry: dict[str, BaseTool] = {}

    @classmethod
    def register(cls, tool: BaseTool) -> None:
        cls._registry[tool.name] = tool

    @classmethod
    def get(cls, name: str) -> BaseTool | None:
        return cls._registry.get(name)

    @classmethod
    def get_all(cls) -> dict[str, BaseTool]:
        return dict(cls._registry)

    @classmethod
    def get_schemas(cls, tool_names: list[str] | None = None) -> list[dict]:
        """Returns JSON tool schemas for the specified tools (or all if None)."""
        tools = cls._registry.values() if tool_names is None else [
            cls._registry[n] for n in tool_names if n in cls._registry
        ]
        schemas = []
        for tool in tools:
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.args_schema.model_json_schema() if tool.args_schema else {},
                },
            }
            schemas.append(schema)
        return schemas

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()


class MCPSessionRegistry:
    """Global registry for MCP client sessions. Accessed inside forge__mcp activity."""

    _registry: dict[str, ClientSession] = {}
    _tool_schemas: dict[str, list[dict]] = {}  # server_name → tool schemas

    @classmethod
    def register(cls, server_name: str, session: ClientSession) -> None:
        cls._registry[server_name] = session

    @classmethod
    def get(cls, server_name: str) -> ClientSession | None:
        return cls._registry.get(server_name)

    @classmethod
    def set_tool_schemas(cls, server_name: str, schemas: list[dict]) -> None:
        cls._tool_schemas[server_name] = schemas

    @classmethod
    def get_tool_schemas(cls, server_name: str) -> list[dict]:
        return cls._tool_schemas.get(server_name, [])

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()
        cls._tool_schemas.clear()
