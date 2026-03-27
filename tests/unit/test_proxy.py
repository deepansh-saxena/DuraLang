"""Tests for proxy interception — LLM, Tool, MCP proxies."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from duralang.config import DuraConfig, LLMIdentity
from duralang.context import DuraContext
from duralang.graph_def import LLMActivityResult, ToolActivityResult
from duralang.proxy import (
    DuraLLMProxy,
    DuraToolProxy,
    _extract_bound_tool_schemas,
    _safe_kwargs,
)
from duralang.registry import ToolRegistry
from duralang.state import MessageSerializer


@pytest.fixture(autouse=True)
def clear_registries():
    ToolRegistry.clear()
    yield
    ToolRegistry.clear()


class TestDuraLLMProxy:
    @pytest.mark.asyncio
    async def test_passthrough_without_context(self):
        """When no DuraContext is set, proxy calls original method."""
        original_response = AIMessage(content="Hello!")
        original_ainvoke = AsyncMock(return_value=original_response)
        mock_instance = MagicMock()
        mock_instance.__class__.__name__ = "ChatAnthropic"
        mock_instance.model = "claude-sonnet-4-6"
        mock_instance.temperature = None
        mock_instance.tools = None

        proxy_fn = DuraLLMProxy.make_ainvoke(original_ainvoke, mock_instance)
        result = await proxy_fn([HumanMessage(content="hi")])

        assert result == original_response
        original_ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_intercepts_with_context(self):
        """When DuraContext is set, proxy routes to execute_activity."""
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

        original_ainvoke = AsyncMock()
        mock_instance = MagicMock()
        mock_instance.__class__.__name__ = "ChatAnthropic"
        mock_instance.model = "claude-sonnet-4-6"
        mock_instance.temperature = None
        mock_instance.tools = None

        proxy_fn = DuraLLMProxy.make_ainvoke(original_ainvoke, mock_instance)

        token = DuraContext.set(ctx)
        try:
            result = await proxy_fn([HumanMessage(content="hi")])
        finally:
            DuraContext.reset(token)

        assert isinstance(result, AIMessage)
        assert result.content == "Intercepted!"
        original_ainvoke.assert_not_called()
        mock_execute.assert_called_once()
        call_args = mock_execute.call_args
        assert call_args[0][0] == "dura__llm"


class TestDuraToolProxy:
    @pytest.mark.asyncio
    async def test_passthrough_without_context(self):
        original_ainvoke = AsyncMock(return_value="sunny")
        mock_tool = MagicMock()
        mock_tool.name = "get_weather"

        proxy_fn = DuraToolProxy.make_ainvoke(original_ainvoke, mock_tool)
        result = await proxy_fn({"location": "NYC"})

        assert result == "sunny"
        original_ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_intercepts_with_context(self):
        mock_execute = AsyncMock(
            return_value=ToolActivityResult(output="sunny", tool_call_id="tc_1")
        )
        ctx = DuraContext(
            workflow_id="test-wf",
            config=DuraConfig(),
            execute_activity=mock_execute,
            execute_child_agent=AsyncMock(),
        )

        original_ainvoke = AsyncMock()
        mock_tool = MagicMock()
        mock_tool.name = "get_weather"

        proxy_fn = DuraToolProxy.make_ainvoke(original_ainvoke, mock_tool)

        token = DuraContext.set(ctx)
        try:
            result = await proxy_fn({"location": "NYC"})
        finally:
            DuraContext.reset(token)

        assert result == "sunny"
        original_ainvoke.assert_not_called()
        call_args = mock_execute.call_args
        assert call_args[0][0] == "dura__tool"

    @pytest.mark.asyncio
    async def test_returns_error_string(self):
        mock_execute = AsyncMock(
            return_value=ToolActivityResult(output="", tool_call_id="tc_1", error="bad input")
        )
        ctx = DuraContext(
            workflow_id="test-wf",
            config=DuraConfig(),
            execute_activity=mock_execute,
            execute_child_agent=AsyncMock(),
        )

        mock_tool = MagicMock()
        mock_tool.name = "calc"

        proxy_fn = DuraToolProxy.make_ainvoke(AsyncMock(), mock_tool)

        token = DuraContext.set(ctx)
        try:
            result = await proxy_fn({"x": 0})
        finally:
            DuraContext.reset(token)

        assert result == "bad input"


class TestExtractBoundToolSchemas:
    def test_extracts_from_tools_list(self):
        mock_tool = MagicMock()
        mock_tool.name = "search"
        mock_tool.args_schema = MagicMock()
        mock_tool.args_schema.model_json_schema.return_value = {"type": "object"}

        mock_llm = MagicMock()
        mock_llm.tools = [mock_tool]

        schemas = _extract_bound_tool_schemas(mock_llm)
        assert len(schemas) == 1
        assert schemas[0] == {"type": "object"}
        assert ToolRegistry.get("search") is mock_tool

    def test_handles_dict_schemas(self):
        mock_llm = MagicMock()
        mock_llm.tools = [{"type": "function", "function": {"name": "calc"}}]
        mock_llm._tools = None

        schemas = _extract_bound_tool_schemas(mock_llm)
        assert len(schemas) == 1

    def test_no_tools(self):
        mock_llm = MagicMock()
        mock_llm.tools = None
        mock_llm._tools = None

        schemas = _extract_bound_tool_schemas(mock_llm)
        assert schemas == []


class TestSafeKwargs:
    def test_filters_non_serializable(self):
        result = _safe_kwargs({
            "temperature": 0.7,
            "callback": lambda x: x,
            "max_tokens": 100,
        })
        assert result == {"temperature": 0.7, "max_tokens": 100}

    def test_keeps_none(self):
        result = _safe_kwargs({"key": None})
        assert result == {"key": None}
