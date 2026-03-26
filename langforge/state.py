"""StateManager — serialization and deserialization of graph state across activity boundaries."""

from __future__ import annotations

from langforge.exceptions import StateSerializationError


class StateManager:
    """Handles serialization/deserialization of LangGraph state including LangChain messages."""

    # Message class name → constructor mapping (lazy loaded)
    _message_types: dict | None = None

    @classmethod
    def _get_message_types(cls) -> dict:
        if cls._message_types is None:
            from langchain_core.messages import (
                AIMessage,
                FunctionMessage,
                HumanMessage,
                SystemMessage,
                ToolMessage,
            )

            cls._message_types = {
                "HumanMessage": HumanMessage,
                "AIMessage": AIMessage,
                "ToolMessage": ToolMessage,
                "SystemMessage": SystemMessage,
                "FunctionMessage": FunctionMessage,
            }
        return cls._message_types

    @staticmethod
    def serialize(state: dict) -> dict:
        """Full state → JSON-serializable dict.

        Converts all LangChain message objects to dicts.
        Called before passing state into activity payloads.
        """
        try:
            result = {}
            for key, value in state.items():
                if isinstance(value, list):
                    result[key] = [StateManager._serialize_item(v) for v in value]
                else:
                    result[key] = StateManager._serialize_item(value)
            return result
        except Exception as e:
            raise StateSerializationError(f"Failed to serialize state: {e}") from e

    @staticmethod
    def serialize_delta(delta: dict) -> dict:
        """Same as serialize but for state deltas returned by node functions."""
        return StateManager.serialize(delta)

    @staticmethod
    def serialize_message(msg) -> dict:
        """Converts a single LangChain message object to a serializable dict."""
        return {
            "type": msg.__class__.__name__,
            "content": msg.content,
            "tool_calls": getattr(msg, "tool_calls", []),
            "tool_call_id": getattr(msg, "tool_call_id", None),
            "name": getattr(msg, "name", None),
            "id": getattr(msg, "id", None),
        }

    @staticmethod
    def deserialize(state_dict: dict, schema: dict | None = None) -> dict:
        """JSON dict → typed state with LangChain message objects.

        Used before passing state to routing functions and node functions.
        """
        result = {}
        for key, value in state_dict.items():
            if isinstance(value, list):
                result[key] = [StateManager._deserialize_item(v) for v in value]
            else:
                result[key] = value
        return result

    @staticmethod
    def deserialize_for_node(state_dict: dict) -> dict:
        """Deserialize state for use inside NodeActivity."""
        return StateManager.deserialize(state_dict)

    @staticmethod
    def _serialize_item(item):
        from langchain_core.messages import BaseMessage

        if isinstance(item, BaseMessage):
            return StateManager.serialize_message(item)
        return item

    @staticmethod
    def _deserialize_item(item):
        if not isinstance(item, dict) or "type" not in item:
            return item

        message_types = StateManager._get_message_types()
        cls = message_types.get(item["type"])
        if cls is None:
            return item

        kwargs: dict = {"content": item["content"]}
        if item.get("tool_calls"):
            kwargs["tool_calls"] = item["tool_calls"]
        if item.get("tool_call_id"):
            kwargs["tool_call_id"] = item["tool_call_id"]
        if item.get("name"):
            kwargs["name"] = item["name"]
        if item.get("id"):
            kwargs["id"] = item["id"]
        return cls(**kwargs)
