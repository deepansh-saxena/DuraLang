"""Tests for forge__tool activity with mocked tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langforge.activities.tool import tool_activity
from langforge.exceptions import ToolActivityError
from langforge.graph_def import ToolActivityPayload
from langforge.registry import ToolRegistry


@pytest.fixture(autouse=True)
def clear_registry():
    ToolRegistry.clear()
    yield
    ToolRegistry.clear()


@pytest.fixture
def mock_heartbeat():
    with patch("langforge.activities.tool.activity") as mock_activity:
        mock_activity.heartbeat = MagicMock()
        yield mock_activity


class TestToolActivity:
    @pytest.mark.asyncio
    async def test_successful_tool_call(self, mock_heartbeat):
        mock_tool = MagicMock()
        mock_tool.name = "search"
        mock_tool.ainvoke = AsyncMock(return_value="Paris is the capital of France")
        ToolRegistry.register(mock_tool)

        payload = ToolActivityPayload(
            tool_name="search",
            tool_input={"query": "capital of france"},
            tool_call_id="tc_1",
        )

        result = await tool_activity(payload)

        assert result.output == "Paris is the capital of France"
        assert result.tool_call_id == "tc_1"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_tool_not_registered(self, mock_heartbeat):
        payload = ToolActivityPayload(
            tool_name="nonexistent",
            tool_input={},
            tool_call_id="tc_1",
        )

        with pytest.raises(ToolActivityError, match="not registered"):
            await tool_activity(payload)

    @pytest.mark.asyncio
    async def test_tool_value_error_non_retryable(self, mock_heartbeat):
        mock_tool = MagicMock()
        mock_tool.name = "bad_tool"
        mock_tool.ainvoke = AsyncMock(side_effect=ValueError("bad input"))
        ToolRegistry.register(mock_tool)

        payload = ToolActivityPayload(
            tool_name="bad_tool",
            tool_input={"bad": "data"},
            tool_call_id="tc_2",
        )

        result = await tool_activity(payload)
        assert result.error == "bad input"
        assert result.tool_call_id == "tc_2"

    @pytest.mark.asyncio
    async def test_tool_network_error_retryable(self, mock_heartbeat):
        mock_tool = MagicMock()
        mock_tool.name = "flaky_tool"
        mock_tool.ainvoke = AsyncMock(side_effect=ConnectionError("timeout"))
        ToolRegistry.register(mock_tool)

        payload = ToolActivityPayload(
            tool_name="flaky_tool",
            tool_input={},
            tool_call_id="tc_3",
        )

        with pytest.raises(ConnectionError):
            await tool_activity(payload)

    @pytest.mark.asyncio
    async def test_tool_call_id_preserved(self, mock_heartbeat):
        mock_tool = MagicMock()
        mock_tool.name = "echo"
        mock_tool.ainvoke = AsyncMock(return_value="echoed")
        ToolRegistry.register(mock_tool)

        payload = ToolActivityPayload(
            tool_name="echo",
            tool_input="hello",
            tool_call_id="unique_id_123",
        )

        result = await tool_activity(payload)
        assert result.tool_call_id == "unique_id_123"
