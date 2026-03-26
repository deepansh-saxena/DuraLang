"""LangForge exception hierarchy."""


class LangForgeError(Exception):
    """Base exception for all LangForge errors."""


class ConfigurationError(LangForgeError):
    """Checkpointer present, bad config, unknown provider."""


class CompilationError(LangForgeError):
    """GraphCompiler failed — lambda node, non-importable callable, etc."""


class NodeActivityError(LangForgeError):
    """Node function raised after max retries."""


class ToolActivityError(LangForgeError):
    """Tool call failed after max retries."""


class MCPActivityError(LangForgeError):
    """MCP call failed after max retries."""


class WorkflowFailedError(LangForgeError):
    """Entire workflow failed — wraps original cause."""


class StateSerializationError(LangForgeError):
    """State could not be serialized across an activity boundary."""


class DeterminismViolationError(LangForgeError):
    """Routing function detected with side effects or LLM calls."""
