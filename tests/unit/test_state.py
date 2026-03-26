"""Tests for StateManager — serialization, deserialization, round-trip."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langforge.state import StateManager


class TestSerializeMessage:
    def test_human_message(self):
        msg = HumanMessage(content="Hello")
        result = StateManager.serialize_message(msg)
        assert result["type"] == "HumanMessage"
        assert result["content"] == "Hello"
        assert result["tool_calls"] == []

    def test_ai_message_with_tool_calls(self):
        msg = AIMessage(
            content="",
            tool_calls=[
                {"id": "tc_1", "name": "search", "args": {"query": "weather"}}
            ],
        )
        result = StateManager.serialize_message(msg)
        assert result["type"] == "AIMessage"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "tc_1"

    def test_tool_message(self):
        msg = ToolMessage(content="sunny", tool_call_id="tc_1")
        result = StateManager.serialize_message(msg)
        assert result["type"] == "ToolMessage"
        assert result["content"] == "sunny"
        assert result["tool_call_id"] == "tc_1"

    def test_system_message(self):
        msg = SystemMessage(content="You are helpful")
        result = StateManager.serialize_message(msg)
        assert result["type"] == "SystemMessage"


class TestSerialize:
    def test_serialize_state_with_messages(self):
        state = {
            "messages": [HumanMessage(content="hi"), AIMessage(content="hello")],
            "counter": 5,
        }
        result = StateManager.serialize(state)
        assert result["counter"] == 5
        assert len(result["messages"]) == 2
        assert result["messages"][0]["type"] == "HumanMessage"
        assert result["messages"][1]["type"] == "AIMessage"

    def test_serialize_empty_state(self):
        assert StateManager.serialize({}) == {}

    def test_serialize_non_message_list(self):
        state = {"tags": ["a", "b", "c"]}
        result = StateManager.serialize(state)
        assert result["tags"] == ["a", "b", "c"]


class TestDeserialize:
    def test_deserialize_messages(self):
        serialized = {
            "messages": [
                {"type": "HumanMessage", "content": "hi", "tool_calls": [], "tool_call_id": None, "name": None, "id": None},
                {"type": "AIMessage", "content": "hello", "tool_calls": [], "tool_call_id": None, "name": None, "id": None},
            ],
            "counter": 5,
        }
        result = StateManager.deserialize(serialized)
        assert isinstance(result["messages"][0], HumanMessage)
        assert isinstance(result["messages"][1], AIMessage)
        assert result["counter"] == 5

    def test_deserialize_tool_message(self):
        serialized = {
            "messages": [
                {"type": "ToolMessage", "content": "result", "tool_calls": [], "tool_call_id": "tc_1", "name": None, "id": None},
            ]
        }
        result = StateManager.deserialize(serialized)
        msg = result["messages"][0]
        assert isinstance(msg, ToolMessage)
        assert msg.tool_call_id == "tc_1"

    def test_deserialize_ai_with_tool_calls(self):
        serialized = {
            "messages": [
                {
                    "type": "AIMessage",
                    "content": "",
                    "tool_calls": [{"id": "tc_1", "name": "search", "args": {"q": "x"}}],
                    "tool_call_id": None,
                    "name": None,
                    "id": None,
                },
            ]
        }
        result = StateManager.deserialize(serialized)
        msg = result["messages"][0]
        assert isinstance(msg, AIMessage)
        assert len(msg.tool_calls) == 1

    def test_deserialize_unknown_type_passthrough(self):
        serialized = {
            "data": [{"type": "UnknownThing", "content": "x"}],
        }
        result = StateManager.deserialize(serialized)
        assert result["data"][0] == {"type": "UnknownThing", "content": "x"}


class TestRoundTrip:
    def test_human_ai_roundtrip(self):
        original = {
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(content="hello"),
            ],
            "counter": 42,
        }
        serialized = StateManager.serialize(original)
        deserialized = StateManager.deserialize(serialized)

        assert isinstance(deserialized["messages"][0], HumanMessage)
        assert deserialized["messages"][0].content == "hi"
        assert isinstance(deserialized["messages"][1], AIMessage)
        assert deserialized["messages"][1].content == "hello"
        assert deserialized["counter"] == 42

    def test_tool_call_roundtrip(self):
        original = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"id": "tc_1", "name": "calc", "args": {"x": 1}}],
                ),
                ToolMessage(content="result: 2", tool_call_id="tc_1"),
            ]
        }
        serialized = StateManager.serialize(original)
        deserialized = StateManager.deserialize(serialized)

        ai_msg = deserialized["messages"][0]
        assert isinstance(ai_msg, AIMessage)
        assert ai_msg.tool_calls[0]["id"] == "tc_1"

        tool_msg = deserialized["messages"][1]
        assert isinstance(tool_msg, ToolMessage)
        assert tool_msg.tool_call_id == "tc_1"
        assert tool_msg.content == "result: 2"
