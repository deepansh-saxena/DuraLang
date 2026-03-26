"""Tests for forge__node activity with mocked LLM."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langforge.activities.node import node_activity
from langforge.graph_def import NodeActivityPayload
from langforge.state import StateManager


@pytest.fixture
def mock_heartbeat():
    with patch("langforge.activities.node.activity") as mock_activity:
        mock_activity.heartbeat = MagicMock()
        yield mock_activity


class TestNodeActivity:
    @pytest.mark.asyncio
    async def test_node_no_tool_calls(self, mock_heartbeat):
        """Node returns a simple state delta with no tool calls."""
        # Create a mock node function
        async def my_node(state, llm=None):
            return {"messages": [AIMessage(content="Hello back!")]}

        payload = NodeActivityPayload(
            node_name="my_node",
            callable_path="tests.unit.test_node_activity:_dummy_node",
            current_state=StateManager.serialize(
                {"messages": [HumanMessage(content="Hi")]}
            ),
            llm_config={"provider": "anthropic", "model": "claude-sonnet-4-6", "kwargs": {}},
            tool_schemas=[],
        )

        with patch("langforge.activities.node.importlib") as mock_importlib:
            mock_module = MagicMock()
            mock_module._dummy_node = my_node
            mock_importlib.import_module.return_value = mock_module

            with patch("langforge.activities.node.build_llm") as mock_build:
                mock_build.return_value = MagicMock()
                result = await node_activity(payload)

        assert result.is_final is True
        assert len(result.tool_calls) == 0
        assert "messages" in result.state_delta

    @pytest.mark.asyncio
    async def test_node_with_tool_calls(self, mock_heartbeat):
        """Node returns an AIMessage with tool_calls."""
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"id": "tc_1", "name": "search", "args": {"query": "weather"}},
                {"id": "tc_2", "name": "calc", "args": {"expr": "2+2"}},
            ],
        )

        async def my_node(state, llm=None):
            return {"messages": [ai_msg]}

        payload = NodeActivityPayload(
            node_name="agent",
            callable_path="test:my_node",
            current_state=StateManager.serialize(
                {"messages": [HumanMessage(content="What's the weather?")]}
            ),
            llm_config={"provider": "anthropic", "model": "claude-sonnet-4-6", "kwargs": {}},
            tool_schemas=[{"type": "function", "function": {"name": "search"}}],
        )

        with patch("langforge.activities.node.importlib") as mock_importlib:
            mock_module = MagicMock()
            mock_module.my_node = my_node
            mock_importlib.import_module.return_value = mock_module

            with patch("langforge.activities.node.build_llm") as mock_build:
                mock_llm = MagicMock()
                mock_llm.bind_tools.return_value = mock_llm
                mock_build.return_value = mock_llm
                result = await node_activity(payload)

        assert result.is_final is False
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].id == "tc_1"
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[1].id == "tc_2"
        assert result.tool_calls[1].name == "calc"


# Dummy function for import path testing
async def _dummy_node(state, llm=None):
    return {"messages": [AIMessage(content="test")]}
