"""DuraLang exception hierarchy."""


class DuraLangError(Exception):
    """Base exception for all DuraLang errors."""


class ConfigurationError(DuraLangError):
    """Lambda function, non-importable callable, unknown LLM provider."""


class LLMActivityError(DuraLangError):
    """LLM inference failed after max retries."""


class ToolActivityError(DuraLangError):
    """Tool not registered or failed after max retries."""


class MCPActivityError(DuraLangError):
    """MCP server not registered or call failed after max retries."""


class WorkflowFailedError(DuraLangError):
    """Unrecoverable workflow failure."""


class StateSerializationError(DuraLangError):
    """Argument or message could not be serialized."""
