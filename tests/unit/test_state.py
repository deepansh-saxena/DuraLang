"""Tests for MessageSerializer and ArgSerializer — serialization round-trips."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from duralang.exceptions import StateSerializationError
from duralang.state import ArgSerializer, MessageSerializer


class TestSerialize:
    def test_human_message(self):
        msg = HumanMessage(content="Hello")
        result = MessageSerializer.serialize(msg)
        assert result["type"] == "HumanMessage"
        assert result["content"] == "Hello"
        assert result["tool_calls"] == []

    def test_ai_message_with_tool_calls(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc_1", "name": "search", "args": {"query": "weather"}}],
        )
        result = MessageSerializer.serialize(msg)
        assert result["type"] == "AIMessage"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "tc_1"

    def test_tool_message(self):
        msg = ToolMessage(content="sunny", tool_call_id="tc_1")
        result = MessageSerializer.serialize(msg)
        assert result["type"] == "ToolMessage"
        assert result["content"] == "sunny"
        assert result["tool_call_id"] == "tc_1"

    def test_system_message(self):
        msg = SystemMessage(content="You are helpful")
        result = MessageSerializer.serialize(msg)
        assert result["type"] == "SystemMessage"


class TestDeserialize:
    def test_human_message(self):
        d = {"type": "HumanMessage", "content": "hi", "tool_calls": [], "tool_call_id": None, "name": None, "id": None}
        msg = MessageSerializer.deserialize(d)
        assert isinstance(msg, HumanMessage)
        assert msg.content == "hi"

    def test_ai_message_with_tool_calls(self):
        d = {
            "type": "AIMessage",
            "content": "",
            "tool_calls": [{"id": "tc_1", "name": "search", "args": {"q": "x"}}],
            "tool_call_id": None,
            "name": None,
            "id": None,
        }
        msg = MessageSerializer.deserialize(d)
        assert isinstance(msg, AIMessage)
        assert len(msg.tool_calls) == 1

    def test_tool_message(self):
        d = {"type": "ToolMessage", "content": "result", "tool_calls": [], "tool_call_id": "tc_1", "name": None, "id": None}
        msg = MessageSerializer.deserialize(d)
        assert isinstance(msg, ToolMessage)
        assert msg.tool_call_id == "tc_1"

    def test_unknown_type_raises(self):
        d = {"type": "UnknownMessage", "content": "x", "tool_calls": [], "tool_call_id": None, "name": None, "id": None}
        with pytest.raises(StateSerializationError, match="Unknown message type"):
            MessageSerializer.deserialize(d)

    def test_strips_dura_type_tag(self):
        d = {"__dura_type__": "message", "type": "HumanMessage", "content": "hi", "tool_calls": [], "tool_call_id": None, "name": None, "id": None}
        msg = MessageSerializer.deserialize(d)
        assert isinstance(msg, HumanMessage)
        assert msg.content == "hi"


class TestRoundTrip:
    def test_human_ai_roundtrip(self):
        original = [HumanMessage(content="hi"), AIMessage(content="hello")]
        serialized = MessageSerializer.serialize_many(original)
        deserialized = MessageSerializer.deserialize_many(serialized)
        assert isinstance(deserialized[0], HumanMessage)
        assert deserialized[0].content == "hi"
        assert isinstance(deserialized[1], AIMessage)
        assert deserialized[1].content == "hello"

    def test_tool_call_roundtrip(self):
        original = [
            AIMessage(
                content="",
                tool_calls=[{"id": "tc_1", "name": "calc", "args": {"x": 1}}],
            ),
            ToolMessage(content="result: 2", tool_call_id="tc_1"),
        ]
        serialized = MessageSerializer.serialize_many(original)
        deserialized = MessageSerializer.deserialize_many(serialized)

        ai_msg = deserialized[0]
        assert isinstance(ai_msg, AIMessage)
        assert ai_msg.tool_calls[0]["id"] == "tc_1"

        tool_msg = deserialized[1]
        assert isinstance(tool_msg, ToolMessage)
        assert tool_msg.tool_call_id == "tc_1"


class TestArgSerializer:
    def test_primitives(self):
        args = (42, "hello", True, None, 3.14)
        serialized = ArgSerializer.serialize(args)
        deserialized_args, deserialized_kwargs = ArgSerializer.deserialize(serialized, {})
        assert deserialized_args == args

    def test_list_of_messages(self):
        msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
        serialized = ArgSerializer.serialize((msgs,))
        deserialized_args, _ = ArgSerializer.deserialize(serialized, {})
        result = deserialized_args[0]
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[1], AIMessage)

    def test_kwargs(self):
        kwargs = {"name": "test", "count": 5}
        serialized = ArgSerializer.serialize_kwargs(kwargs)
        _, deserialized_kwargs = ArgSerializer.deserialize([], serialized)
        assert deserialized_kwargs == kwargs

    def test_result_roundtrip(self):
        msgs = [HumanMessage(content="hi"), AIMessage(content="answer")]
        serialized = ArgSerializer.serialize_result(msgs)
        deserialized = ArgSerializer.deserialize_result(serialized)
        assert isinstance(deserialized[0], HumanMessage)
        assert isinstance(deserialized[1], AIMessage)

    def test_unsupported_type_raises(self):
        with pytest.raises(StateSerializationError, match="Cannot serialize"):
            ArgSerializer.serialize((object(),))

    def test_nested_dict(self):
        data = {"key": {"nested": [1, 2, 3]}}
        serialized = ArgSerializer.serialize((data,))
        deserialized_args, _ = ArgSerializer.deserialize(serialized, {})
        assert deserialized_args[0] == data

    def test_tuple_roundtrip(self):
        data = (1, "two", 3)
        serialized = ArgSerializer.serialize_result(data)
        deserialized = ArgSerializer.deserialize_result(serialized)
        assert deserialized == data
