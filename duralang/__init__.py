"""DuraLang — Write normal LangChain code. Get Temporal durability. One decorator."""

from duralang.config import ActivityConfig, DuraConfig
from duralang.decorator import dura
from duralang.exceptions import (
    ConfigurationError,
    DuraLangError,
    LLMActivityError,
    MCPActivityError,
    StateSerializationError,
    ToolActivityError,
    WorkflowFailedError,
)
from duralang.proxy import install_patches
from duralang.registry import MCPSessionRegistry

# Install proxy patches at import time
install_patches()


class DuraMCPSession:
    """Thin wrapper around mcp.ClientSession that attaches a server_name.

    Usage:
        async with ClientSession(read, write) as session:
            await session.initialize()
            fs = DuraMCPSession(session, "filesystem")
            result = await fs.call_tool("read_file", {"path": "/tmp/data.csv"})
    """

    def __init__(self, session, server_name: str):
        self._session = session
        self._server_name = server_name
        MCPSessionRegistry.register(server_name, session)
        from duralang.proxy import DuraMCPProxy

        DuraMCPProxy.install(session, server_name)

    def __getattr__(self, name):
        return getattr(self._session, name)


__all__ = [
    "dura",
    "DuraConfig",
    "ActivityConfig",
    "DuraMCPSession",
    "DuraLangError",
    "ConfigurationError",
    "LLMActivityError",
    "ToolActivityError",
    "MCPActivityError",
    "WorkflowFailedError",
    "StateSerializationError",
]
