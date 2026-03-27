"""MessageSerializer + ArgSerializer — serialization for Temporal boundaries."""

from __future__ import annotations

from duralang.exceptions import StateSerializationError

MESSAGE_MAP: dict | None = None


def _get_message_map() -> dict:
    global MESSAGE_MAP
    if MESSAGE_MAP is None:
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )

        MESSAGE_MAP = {
            "HumanMessage": HumanMessage,
            "AIMessage": AIMessage,
            "ToolMessage": ToolMessage,
            "SystemMessage": SystemMessage,
        }
    return MESSAGE_MAP


class MessageSerializer:
    """Handles serialization/deserialization of LangChain messages."""

    @staticmethod
    def serialize(msg) -> dict:
        """Single LangChain message -> JSON-serializable dict."""
        return {
            "type": msg.__class__.__name__,
            "content": msg.content,
            "tool_calls": getattr(msg, "tool_calls", []) or [],
            "tool_call_id": getattr(msg, "tool_call_id", None),
            "name": getattr(msg, "name", None),
            "id": getattr(msg, "id", None),
        }

    @staticmethod
    def deserialize(d: dict):
        """JSON dict -> LangChain message object."""
        # Strip __dura_type__ tag if present
        d = {k: v for k, v in d.items() if k != "__dura_type__"}
        message_map = _get_message_map()
        cls = message_map.get(d["type"])
        if cls is None:
            raise StateSerializationError(f"Unknown message type: {d['type']}")

        kwargs: dict = {"content": d["content"]}
        if d.get("tool_calls"):
            kwargs["tool_calls"] = d["tool_calls"]
        if d.get("tool_call_id"):
            kwargs["tool_call_id"] = d["tool_call_id"]
        if d.get("name"):
            kwargs["name"] = d["name"]
        if d.get("id"):
            kwargs["id"] = d["id"]
        return cls(**kwargs)

    @staticmethod
    def serialize_many(messages: list) -> list[dict]:
        return [
            MessageSerializer.serialize(m) if not isinstance(m, dict) else m
            for m in messages
        ]

    @staticmethod
    def deserialize_many(dicts: list[dict]) -> list:
        return [MessageSerializer.deserialize(d) for d in dicts]


class ArgSerializer:
    """Serializes/deserializes function arguments for WorkflowPayload.

    Supported argument types:
    - Primitives: str, int, float, bool, None
    - Collections: list, dict, tuple
    - LangChain messages: HumanMessage, AIMessage, etc.
    """

    @staticmethod
    def serialize(args: tuple) -> list:
        return [ArgSerializer._serialize_item(a) for a in args]

    @staticmethod
    def serialize_kwargs(kwargs: dict) -> dict:
        return {k: ArgSerializer._serialize_item(v) for k, v in kwargs.items()}

    @staticmethod
    def deserialize(args: list, kwargs: dict) -> tuple[tuple, dict]:
        return (
            tuple(ArgSerializer._deserialize_item(a) for a in args),
            {k: ArgSerializer._deserialize_item(v) for k, v in kwargs.items()},
        )

    @staticmethod
    def serialize_result(result):
        return ArgSerializer._serialize_item(result)

    @staticmethod
    def deserialize_result(result):
        return ArgSerializer._deserialize_item(result)

    @staticmethod
    def _serialize_item(item):
        from langchain_core.messages import BaseMessage

        if isinstance(item, BaseMessage):
            return {"__dura_type__": "message", **MessageSerializer.serialize(item)}
        if isinstance(item, list):
            return [ArgSerializer._serialize_item(i) for i in item]
        if isinstance(item, tuple):
            return {"__dura_type__": "tuple", "items": [ArgSerializer._serialize_item(i) for i in item]}
        if isinstance(item, dict):
            return {k: ArgSerializer._serialize_item(v) for k, v in item.items()}
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        raise StateSerializationError(
            f"Cannot serialize argument of type {type(item).__name__}. "
            f"DuraLang supports: primitives, lists, dicts, LangChain messages."
        )

    @staticmethod
    def _deserialize_item(item):
        if isinstance(item, dict):
            dura_type = item.get("__dura_type__")
            if dura_type == "message":
                return MessageSerializer.deserialize(item)
            if dura_type == "tuple":
                return tuple(ArgSerializer._deserialize_item(i) for i in item["items"])
            return {k: ArgSerializer._deserialize_item(v) for k, v in item.items()}
        if isinstance(item, list):
            return [ArgSerializer._deserialize_item(i) for i in item]
        return item
