"""Edge-case tests for ArgSerializer and MessageSerializer."""

from __future__ import annotations

import datetime
import enum
import uuid

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    ChatMessage,
    FunctionMessage,
)

from duralang.exceptions import StateSerializationError
from duralang.state import ArgSerializer, MessageSerializer


# ---------------------------------------------------------------------------
# Helper enum for round-trip tests
# ---------------------------------------------------------------------------


class Color(enum.Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


# ---------------------------------------------------------------------------
# 1. datetime.datetime round-trip
# ---------------------------------------------------------------------------


class TestDatetimeRoundTrip:
    def test_naive_datetime(self):
        dt = datetime.datetime(2025, 6, 15, 10, 30, 45, 123456)
        serialized = ArgSerializer._serialize_item(dt)
        assert serialized["__dura_type__"] == "datetime"
        result = ArgSerializer._deserialize_item(serialized)
        assert result == dt

    def test_aware_datetime(self):
        dt = datetime.datetime(2025, 6, 15, 10, 30, 45, tzinfo=datetime.timezone.utc)
        serialized = ArgSerializer._serialize_item(dt)
        result = ArgSerializer._deserialize_item(serialized)
        assert result == dt

    def test_datetime_via_args_round_trip(self):
        dt = datetime.datetime(2025, 1, 1)
        args_ser = ArgSerializer.serialize((dt,))
        kwargs_ser = ArgSerializer.serialize_kwargs({"ts": dt})
        args, kwargs = ArgSerializer.deserialize(args_ser, kwargs_ser)
        assert args == (dt,)
        assert kwargs == {"ts": dt}


# ---------------------------------------------------------------------------
# 2. datetime.date round-trip
# ---------------------------------------------------------------------------


class TestDateRoundTrip:
    def test_date(self):
        d = datetime.date(2025, 12, 25)
        serialized = ArgSerializer._serialize_item(d)
        assert serialized["__dura_type__"] == "date"
        result = ArgSerializer._deserialize_item(serialized)
        assert result == d

    def test_date_via_result_round_trip(self):
        d = datetime.date(2000, 1, 1)
        ser = ArgSerializer.serialize_result(d)
        result = ArgSerializer.deserialize_result(ser)
        assert result == d


# ---------------------------------------------------------------------------
# 3. uuid.UUID round-trip
# ---------------------------------------------------------------------------


class TestUUIDRoundTrip:
    def test_uuid4(self):
        u = uuid.uuid4()
        serialized = ArgSerializer._serialize_item(u)
        assert serialized["__dura_type__"] == "uuid"
        result = ArgSerializer._deserialize_item(serialized)
        assert result == u
        assert isinstance(result, uuid.UUID)

    def test_uuid_deterministic(self):
        u = uuid.UUID("12345678-1234-5678-1234-567812345678")
        result = ArgSerializer._deserialize_item(ArgSerializer._serialize_item(u))
        assert result == u


# ---------------------------------------------------------------------------
# 4. bytes round-trip
# ---------------------------------------------------------------------------


class TestBytesRoundTrip:
    def test_bytes(self):
        b = b"\x00\x01\x02\xff"
        serialized = ArgSerializer._serialize_item(b)
        assert serialized["__dura_type__"] == "bytes"
        result = ArgSerializer._deserialize_item(serialized)
        assert result == b
        assert isinstance(result, bytes)

    def test_empty_bytes(self):
        b = b""
        result = ArgSerializer._deserialize_item(ArgSerializer._serialize_item(b))
        assert result == b


# ---------------------------------------------------------------------------
# 5. set and frozenset round-trip
# ---------------------------------------------------------------------------


class TestSetRoundTrip:
    def test_set(self):
        s = {1, 2, 3}
        serialized = ArgSerializer._serialize_item(s)
        assert serialized["__dura_type__"] == "set"
        result = ArgSerializer._deserialize_item(serialized)
        assert result == s
        assert isinstance(result, set)

    def test_frozenset_deserializes_as_set(self):
        fs = frozenset(["a", "b", "c"])
        serialized = ArgSerializer._serialize_item(fs)
        assert serialized["__dura_type__"] == "set"
        result = ArgSerializer._deserialize_item(serialized)
        assert result == {"a", "b", "c"}
        assert isinstance(result, set)

    def test_empty_set(self):
        s: set = set()
        result = ArgSerializer._deserialize_item(ArgSerializer._serialize_item(s))
        assert result == set()


# ---------------------------------------------------------------------------
# 6. Enum round-trip
# ---------------------------------------------------------------------------


class TestEnumRoundTrip:
    def test_color_enum(self):
        c = Color.GREEN
        serialized = ArgSerializer._serialize_item(c)
        assert serialized["__dura_type__"] == "enum"
        assert serialized["value"] == "green"
        result = ArgSerializer._deserialize_item(serialized)
        assert result is Color.GREEN

    def test_all_enum_values(self):
        for color in Color:
            result = ArgSerializer._deserialize_item(
                ArgSerializer._serialize_item(color)
            )
            assert result is color


# ---------------------------------------------------------------------------
# 7. Payload size validation
# ---------------------------------------------------------------------------


class TestPayloadSizeValidation:
    def test_oversized_payload_raises(self):
        # Create a payload larger than 1.8MB
        big_string = "x" * 2_000_000
        serialized_args = ArgSerializer.serialize((big_string,))
        serialized_kwargs = ArgSerializer.serialize_kwargs({})
        with pytest.raises(StateSerializationError, match=r"\d[\d,]+ bytes"):
            ArgSerializer.validate_payload_size(serialized_args, serialized_kwargs)

    def test_small_payload_passes(self):
        serialized_args = ArgSerializer.serialize(("hello",))
        serialized_kwargs = ArgSerializer.serialize_kwargs({})
        # Should not raise
        ArgSerializer.validate_payload_size(serialized_args, serialized_kwargs)


# ---------------------------------------------------------------------------
# 8. Recursion depth
# ---------------------------------------------------------------------------


class TestRecursionDepth:
    def test_deeply_nested_dict_raises(self):
        # Build a 200-level nested dict, well over MAX_DEPTH (100)
        nested: dict = {"leaf": True}
        for _ in range(200):
            nested = {"child": nested}

        with pytest.raises(StateSerializationError, match="depth"):
            ArgSerializer._serialize_item(nested)

    def test_within_depth_limit_succeeds(self):
        nested: dict = {"leaf": True}
        for _ in range(50):
            nested = {"child": nested}
        # Should not raise
        result = ArgSerializer._deserialize_item(
            ArgSerializer._serialize_item(nested)
        )
        # Walk down to the leaf
        current = result
        for _ in range(50):
            current = current["child"]
        assert current == {"leaf": True}


# ---------------------------------------------------------------------------
# 9. FunctionMessage round-trip
# ---------------------------------------------------------------------------


class TestFunctionMessageRoundTrip:
    def test_function_message(self):
        msg = FunctionMessage(name="get_weather", content="72F and sunny")
        serialized = MessageSerializer.serialize(msg)
        assert serialized["type"] == "FunctionMessage"
        assert serialized["name"] == "get_weather"
        result = MessageSerializer.deserialize(serialized)
        assert isinstance(result, FunctionMessage)
        assert result.content == "72F and sunny"
        assert result.name == "get_weather"


# ---------------------------------------------------------------------------
# 10. ChatMessage round-trip (requires role)
# ---------------------------------------------------------------------------


class TestChatMessageRoundTrip:
    def test_chat_message_preserves_role(self):
        msg = ChatMessage(role="assistant", content="Hello!")
        serialized = MessageSerializer.serialize(msg)
        assert serialized["type"] == "ChatMessage"
        assert serialized["role"] == "assistant"
        result = MessageSerializer.deserialize(serialized)
        assert isinstance(result, ChatMessage)
        assert result.content == "Hello!"
        assert result.role == "assistant"

    def test_chat_message_custom_role(self):
        msg = ChatMessage(role="moderator", content="Review this.")
        result = MessageSerializer.deserialize(MessageSerializer.serialize(msg))
        assert result.role == "moderator"


# ---------------------------------------------------------------------------
# 11. AIMessageChunk serializes as AIMessage type
# ---------------------------------------------------------------------------


class TestAIMessageChunkCoalescing:
    def test_chunk_serializes_as_ai_message_type(self):
        chunk = AIMessageChunk(content="partial response")
        serialized = MessageSerializer.serialize(chunk)
        assert serialized["type"] == "AIMessage"

    def test_chunk_deserializes_as_ai_message(self):
        chunk = AIMessageChunk(content="partial response")
        serialized = MessageSerializer.serialize(chunk)
        result = MessageSerializer.deserialize(serialized)
        assert isinstance(result, AIMessage)
        assert not isinstance(result, AIMessageChunk)
        assert result.content == "partial response"


# ---------------------------------------------------------------------------
# 12. response_metadata preserved through round-trip
# ---------------------------------------------------------------------------


class TestResponseMetadataPreserved:
    def test_response_metadata_round_trip(self):
        metadata = {
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
        msg = AIMessage(content="Done.", response_metadata=metadata)
        serialized = MessageSerializer.serialize(msg)
        assert serialized["response_metadata"] == metadata
        result = MessageSerializer.deserialize(serialized)
        assert result.response_metadata == metadata

    def test_empty_response_metadata(self):
        msg = AIMessage(content="Hi")
        serialized = MessageSerializer.serialize(msg)
        result = MessageSerializer.deserialize(serialized)
        assert result.response_metadata == {}
