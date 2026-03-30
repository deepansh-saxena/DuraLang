"""Tests for DuraModel, DuraTool, and MCP proxy."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from duralang.config import DuraConfig
from duralang.context import DuraContext
from duralang.dura_model import DuraModel
from duralang.dura_tool import DuraTool
from duralang.graph_def import LLMActivityResult, ToolActivityResult
from duralang.registry import ToolRegistry
from duralang.state import MessageSerializer


class TestDuraModel:
    @pytest.mark.asyncio
    async def test_passthrough_without_context(self):
        """When no DuraContext is set, DuraModel calls inner LLM."""
        from langchain_anthropic import ChatAnthropic
        from langchain_core.outputs import ChatGeneration, ChatResult

        inner = ChatAnthropic(model="claude-sonnet-4-6", api_key="test-key")
        expected_response = AIMessage(content="Hello!")

        model = DuraModel(inner_llm=inner)

        with patch.object(
            inner,
            "_agenerate",
            new=AsyncMock(
                return_value=ChatResult(generations=[ChatGeneration(message=expected_response)])
            ),
        ):
            result = await model.ainvoke([HumanMessage(content="hi")])

        assert isinstance(result, AIMessage)
        assert result.content == "Hello!"

    @pytest.mark.asyncio
    async def test_intercepts_with_context(self):
        """When DuraContext is set, DuraModel routes to execute_activity."""
        ai_msg_dict = MessageSerializer.serialize(AIMessage(content="Intercepted!"))
        mock_execute = AsyncMock(
            return_value=LLMActivityResult(ai_message=ai_msg_dict, content="Intercepted!")
        )

        ctx = DuraContext(
            workflow_id="test-wf",
            config=DuraConfig(),
            execute_activity=mock_execute,
            execute_child_agent=AsyncMock(),
        )

        from langchain_anthropic import ChatAnthropic

        inner = ChatAnthropic(model="claude-sonnet-4-6", api_key="test-key")
        model = DuraModel(inner_llm=inner)

        token = DuraContext.set(ctx)
        try:
            result = await model.ainvoke([HumanMessage(content="hi")])
        finally:
            DuraContext.reset(token)

        assert isinstance(result, AIMessage)
        assert result.content == "Intercepted!"
        mock_execute.assert_called_once()
        call_args = mock_execute.call_args
        assert call_args[0][0] == "dura__llm"

    def test_from_model_string_anthropic(self):
        model = DuraModel.from_model_string("claude-sonnet-4-6", api_key="test-key")
        assert isinstance(model, DuraModel)
        assert model.inner_llm.model == "claude-sonnet-4-6"


def _make_test_tool():
    """Create a real BaseTool for testing DuraTool."""
    from langchain_core.tools import tool

    @tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return f"sunny in {location}"

    return get_weather


class TestDuraTool:
    @pytest.mark.asyncio
    async def test_passthrough_without_context(self):
        """When no DuraContext is set, DuraTool calls inner tool."""
        inner_tool = _make_test_tool()
        dura_tool = DuraTool(inner_tool)
        result = await dura_tool._arun(location="NYC")

        assert result == "sunny in NYC"

    @pytest.mark.asyncio
    async def test_intercepts_with_context(self):
        """When DuraContext is set, DuraTool routes to execute_activity."""
        mock_execute = AsyncMock(
            return_value=ToolActivityResult(output="sunny", tool_call_id="tc_1")
        )
        ctx = DuraContext(
            workflow_id="test-wf",
            config=DuraConfig(),
            execute_activity=mock_execute,
            execute_child_agent=AsyncMock(),
        )

        inner_tool = _make_test_tool()
        dura_tool = DuraTool(inner_tool)

        token = DuraContext.set(ctx)
        try:
            result = await dura_tool._arun(location="NYC")
        finally:
            DuraContext.reset(token)

        assert result == "sunny"
        call_args = mock_execute.call_args
        assert call_args[0][0] == "dura__tool"

    @pytest.mark.asyncio
    async def test_returns_tool_message_with_tool_call_id(self):
        """When input has a tool_call_id, DuraTool returns ToolMessage."""
        mock_execute = AsyncMock(
            return_value=ToolActivityResult(output="sunny", tool_call_id="tc_1")
        )
        ctx = DuraContext(
            workflow_id="test-wf",
            config=DuraConfig(),
            execute_activity=mock_execute,
            execute_child_agent=AsyncMock(),
        )

        inner_tool = _make_test_tool()
        dura_tool = DuraTool(inner_tool)

        token = DuraContext.set(ctx)
        try:
            result = await dura_tool._arun(id="tc_1", location="NYC")
        finally:
            DuraContext.reset(token)

        assert isinstance(result, ToolMessage)
        assert result.content == "sunny"
        assert result.tool_call_id == "tc_1"
        assert result.name == "get_weather"
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_returns_error_tool_message(self):
        """Error results should return ToolMessage with status='error'."""
        mock_execute = AsyncMock(
            return_value=ToolActivityResult(output="", tool_call_id="tc_2", error="bad input")
        )
        ctx = DuraContext(
            workflow_id="test-wf",
            config=DuraConfig(),
            execute_activity=mock_execute,
            execute_child_agent=AsyncMock(),
        )

        inner_tool = _make_test_tool()
        dura_tool = DuraTool(inner_tool)

        token = DuraContext.set(ctx)
        try:
            result = await dura_tool._arun(id="tc_2", location="NYC")
        finally:
            DuraContext.reset(token)

        assert isinstance(result, ToolMessage)
        assert result.content == "bad input"
        assert result.status == "error"

    def test_registers_in_tool_registry(self):
        inner_tool = _make_test_tool()
        DuraTool(inner_tool)
        assert ToolRegistry.get("get_weather") is inner_tool
