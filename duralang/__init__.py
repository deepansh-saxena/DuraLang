"""DuraLang — Write normal LangChain code. Get Temporal durability. One decorator."""

from duralang.config import ActivityConfig, DuraConfig
from duralang.decorator import dura
from duralang.dura_agent import dura_agent
from duralang.dura_model import DuraModel
from duralang.dura_tool import DuraTool
from duralang.exceptions import (
    ConfigurationError,
    DuraLangError,
    LLMActivityError,
    MCPActivityError,
    StateSerializationError,
    ToolActivityError,
    WorkflowFailedError,
)
from duralang.proxy import _install_eager_task_patch
from duralang.registry import MCPSessionRegistry

# Install the eager_task_factory patch for Temporal + Python 3.12+ compatibility.
# LangGraph uses asyncio.eager_task_factory which bypasses loop.create_task() —
# this causes tasks to be garbage-collected on Temporal's workflow event loop.
_install_eager_task_patch()


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
    "dura_agent",
    "DuraConfig",
    "ActivityConfig",
    "DuraModel",
    "DuraTool",
    "DuraMCPSession",
    "DuraLangError",
    "ConfigurationError",
    "LLMActivityError",
    "ToolActivityError",
    "MCPActivityError",
    "WorkflowFailedError",
    "StateSerializationError",
]
