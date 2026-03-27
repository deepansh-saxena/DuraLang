"""Data structures — activity payloads, results, and workflow types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkflowPayload:
    """Payload sent to start DuraLangWorkflow."""

    fn_path: str  # importable path: "my_module:my_agent"
    args: list  # JSON-serialized positional args
    kwargs: dict  # JSON-serialized keyword args
    config_dict: dict  # serialized DuraConfig


@dataclass
class WorkflowResult:
    """Result returned from DuraLangWorkflow."""

    return_value: Any  # JSON-serialized return value
    error: str | None = None


@dataclass
class LLMActivityPayload:
    """Payload sent to dura__llm activity."""

    messages: list[dict]  # serialized message history
    llm_identity: dict  # serialized LLMIdentity
    tool_schemas: list[dict]  # bound tool schemas
    invoke_kwargs: dict = field(default_factory=dict)


@dataclass
class LLMActivityResult:
    """Result returned from dura__llm activity."""

    ai_message: dict  # serialized AIMessage
    content: str  # response content (always normalized to str)


@dataclass
class ToolActivityPayload:
    """Payload sent to dura__tool activity."""

    tool_name: str
    tool_input: dict | str
    tool_call_id: str  # preserved from LLM tool_calls


@dataclass
class ToolActivityResult:
    """Result returned from dura__tool activity."""

    output: str
    tool_call_id: str
    error: str | None = None


@dataclass
class MCPActivityPayload:
    """Payload sent to dura__mcp activity."""

    server_name: str
    tool_name: str
    arguments: dict
    tool_call_id: str


@dataclass
class MCPActivityResult:
    """Result returned from dura__mcp activity."""

    content: list[dict]
    tool_call_id: str
    is_error: bool = False
