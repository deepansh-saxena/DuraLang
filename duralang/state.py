"""MessageSerializer + ArgSerializer — serialization for Temporal boundaries."""

from __future__ import annotations

from duralang.exceptions import StateSerializationError

MESSAGE_MAP: dict | None = None


def _get_message_map() -> dict:
    global MESSAGE_MAP
    if MESSAGE_MAP is None:
        from langchain_core.messages import (
            AIMessage,
            AIMessageChunk,
            ChatMessage,
            FunctionMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )

        MESSAGE_MAP = {
            "HumanMessage": HumanMessage,
            "AIMessage": AIMessage,
            "ToolMessage": ToolMessage,
            "SystemMessage": SystemMessage,
            "FunctionMessage": FunctionMessage,
            "ChatMessage": ChatMessage,
            "AIMessageChunk": AIMessage,  # Deserialize chunks as AIMessage
        }
    return MESSAGE_MAP


class MessageSerializer:
    """Handles serialization/deserialization of LangChain messages."""

    @staticmethod
    def serialize(msg) -> dict:
        """Single LangChain message -> JSON-serializable dict."""
        from langchain_core.messages import AIMessageChunk

        type_name = msg.__class__.__name__
        if isinstance(msg, AIMessageChunk):
            type_name = "AIMessage"  # Coalesce chunks for Temporal boundary

        return {
            "type": type_name,
            "content": msg.content,
            "tool_calls": getattr(msg, "tool_calls", []) or [],
            "tool_call_id": getattr(msg, "tool_call_id", None),
            "name": getattr(msg, "name", None),
            "id": getattr(msg, "id", None),
            "additional_kwargs": getattr(msg, "additional_kwargs", {}),
            "response_metadata": getattr(msg, "response_metadata", {}),
            "role": getattr(msg, "role", None),
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
        if d.get("additional_kwargs"):
            kwargs["additional_kwargs"] = d["additional_kwargs"]
        if d.get("response_metadata"):
            kwargs["response_metadata"] = d["response_metadata"]
        # ChatMessage requires a 'role' field
        if d["type"] == "ChatMessage":
            kwargs["role"] = d.get("role", "chat")
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


MAX_PAYLOAD_BYTES = 1_800_000  # 1.8MB — headroom for Temporal's ~2MB limit
MAX_DEPTH = 100


class ArgSerializer:
    """Serializes/deserializes function arguments for WorkflowPayload.

    Supported argument types:
    - Primitives: str, int, float, bool, None
    - Collections: list, dict, tuple, set, frozenset
    - LangChain messages: HumanMessage, AIMessage, etc.
    - Standard library: datetime, date, UUID, Enum, bytes
    """

    @staticmethod
    def serialize(args: tuple) -> list:
        return [ArgSerializer._serialize_item(a) for a in args]

    @staticmethod
    def serialize_kwargs(kwargs: dict) -> dict:
        return {k: ArgSerializer._serialize_item(v) for k, v in kwargs.items()}

    @staticmethod
    def validate_payload_size(serialized_args: list, serialized_kwargs: dict) -> None:
        """Validate total payload size before submitting to Temporal."""
        import json

        payload = {"args": serialized_args, "kwargs": serialized_kwargs}
        size = len(json.dumps(payload).encode("utf-8"))
        if size > MAX_PAYLOAD_BYTES:
            raise StateSerializationError(
                f"Serialized payload is {size:,} bytes, exceeding Temporal ~2MB limit. "
                f"Reduce message history or tool input size."
            )

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
    def _serialize_item(item, _depth=0):
        import datetime
        import enum
        import uuid

        if _depth > MAX_DEPTH:
            raise StateSerializationError(f"Serialization depth exceeds {MAX_DEPTH}.")

        from langchain_core.messages import BaseMessage

        if isinstance(item, BaseMessage):
            return {"__dura_type__": "message", **MessageSerializer.serialize(item)}
        if isinstance(item, list):
            return [ArgSerializer._serialize_item(i, _depth + 1) for i in item]
        if isinstance(item, tuple):
            return {"__dura_type__": "tuple", "items": [ArgSerializer._serialize_item(i, _depth + 1) for i in item]}
        if isinstance(item, dict):
            return {k: ArgSerializer._serialize_item(v, _depth + 1) for k, v in item.items()}
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        if isinstance(item, datetime.datetime):
            return {"__dura_type__": "datetime", "value": item.isoformat()}
        if isinstance(item, datetime.date):
            return {"__dura_type__": "date", "value": item.isoformat()}
        if isinstance(item, enum.Enum):
            return {
                "__dura_type__": "enum",
                "class": f"{type(item).__module__}.{type(item).__qualname__}",
                "value": item.value,
            }
        if isinstance(item, uuid.UUID):
            return {"__dura_type__": "uuid", "value": str(item)}
        if isinstance(item, (set, frozenset)):
            return {
                "__dura_type__": "set",
                "items": [ArgSerializer._serialize_item(i, _depth + 1) for i in item],
            }
        if isinstance(item, bytes):
            import base64

            return {"__dura_type__": "bytes", "value": base64.b64encode(item).decode("ascii")}
        raise StateSerializationError(
            f"Cannot serialize type {type(item).__name__} (value: {repr(item)[:100]}). "
            f"Supported: primitives, list, dict, tuple, datetime, Enum, UUID, bytes, set, BaseMessage."
        )

    @staticmethod
    def _deserialize_item(item, _depth=0):
        import datetime
        import uuid

        if _depth > MAX_DEPTH:
            raise StateSerializationError(f"Deserialization depth exceeds {MAX_DEPTH}.")

        if isinstance(item, dict):
            dura_type = item.get("__dura_type__")
            if dura_type == "message":
                return MessageSerializer.deserialize(item)
            if dura_type == "tuple":
                return tuple(ArgSerializer._deserialize_item(i, _depth + 1) for i in item["items"])
            if dura_type == "datetime":
                return datetime.datetime.fromisoformat(item["value"])
            if dura_type == "date":
                return datetime.date.fromisoformat(item["value"])
            if dura_type == "uuid":
                return uuid.UUID(item["value"])
            if dura_type == "set":
                return set(ArgSerializer._deserialize_item(i, _depth + 1) for i in item["items"])
            if dura_type == "bytes":
                import base64

                return base64.b64decode(item["value"])
            if dura_type == "enum":
                import importlib

                module_path, qualname = item["class"].rsplit(".", 1)
                mod = importlib.import_module(module_path)
                enum_cls = getattr(mod, qualname)
                return enum_cls(item["value"])
            return {k: ArgSerializer._deserialize_item(v, _depth + 1) for k, v in item.items()}
        if isinstance(item, list):
            return [ArgSerializer._deserialize_item(i, _depth + 1) for i in item]
        return item
