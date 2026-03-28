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


    @pytest.mark.asyncio
    async def test_returns_tool_message_with_tool_call_id(self):
        """When input has a tool_call_id, proxy returns ToolMessage (for LangGraph ToolNode)."""
        from langchain_core.messages import ToolMessage

        mock_execute = AsyncMock(
            return_value=ToolActivityResult(output="sunny", tool_call_id="tc_1")
        )
        ctx = DuraContext(
            workflow_id="test-wf",
            config=DuraConfig(),
            execute_activity=mock_execute,
            execute_child_agent=AsyncMock(),
        )

        mock_tool = MagicMock()
        mock_tool.name = "get_weather"

        proxy_fn = DuraToolProxy.make_ainvoke(AsyncMock(), mock_tool)

        # Pass a ToolCall-style dict with "id"
        token = DuraContext.set(ctx)
        try:
            result = await proxy_fn({"location": "NYC", "id": "tc_1"})
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
        from langchain_core.messages import ToolMessage

        mock_execute = AsyncMock(
            return_value=ToolActivityResult(output="", tool_call_id="tc_2", error="bad input")
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
            result = await proxy_fn({"x": 0, "id": "tc_2"})
        finally:
            DuraContext.reset(token)

        assert isinstance(result, ToolMessage)
        assert result.content == "bad input"
        assert result.status == "error"


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

    def test_extracts_from_invoke_kwargs(self):
        """Tools from RunnableBinding kwargs (bind_tools) are captured."""
        mock_llm = MagicMock()
        mock_llm.tools = None
        mock_llm._tools = None

        tool_schema = {"name": "calc", "input_schema": {"type": "object"}, "description": "Calculate."}
        schemas = _extract_bound_tool_schemas(mock_llm, invoke_kwargs={"tools": [tool_schema]})
        assert len(schemas) == 1
        assert schemas[0] == tool_schema

    def test_invoke_kwargs_takes_precedence(self):
        """When tools are in both kwargs and instance, kwargs win."""
        mock_tool = MagicMock()
        mock_tool.name = "old_tool"
        mock_tool.args_schema = MagicMock()
        mock_tool.args_schema.model_json_schema.return_value = {"name": "old"}

        mock_llm = MagicMock()
        mock_llm.tools = [mock_tool]

        new_schema = {"name": "new_tool", "input_schema": {"type": "object"}}
        schemas = _extract_bound_tool_schemas(mock_llm, invoke_kwargs={"tools": [new_schema]})
        assert len(schemas) == 1
        assert schemas[0] == new_schema

    def test_invoke_kwargs_registers_base_tools(self):
        """BaseTool objects in kwargs are registered in ToolRegistry."""
        mock_tool = MagicMock()
        mock_tool.name = "kwarg_tool"
        mock_tool.args_schema = MagicMock()
        mock_tool.args_schema.model_json_schema.return_value = {"type": "object"}

        mock_llm = MagicMock()
        mock_llm.tools = None
        mock_llm._tools = None

        schemas = _extract_bound_tool_schemas(mock_llm, invoke_kwargs={"tools": [mock_tool]})
        assert len(schemas) == 1
        assert ToolRegistry.get("kwarg_tool") is mock_tool


class TestLLMProxyToolStripping:
    @pytest.mark.asyncio
    async def test_tools_stripped_from_invoke_kwargs(self):
        """Tools from bind_tools kwargs should go to tool_schemas, not invoke_kwargs."""
        ai_msg_dict = MessageSerializer.serialize(AIMessage(content="OK"))
        mock_execute = AsyncMock(
            return_value=LLMActivityResult(ai_message=ai_msg_dict, content="OK")
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

        tool_schema = {"name": "calc", "input_schema": {"type": "object"}}
        token = DuraContext.set(ctx)
        try:
            await proxy_fn(
                [HumanMessage(content="hi")],
                tools=[tool_schema],
                temperature=0.7,
            )
        finally:
            DuraContext.reset(token)

        payload = mock_execute.call_args[0][1]
        # tool_schemas should have the tool
        assert len(payload.tool_schemas) == 1
        assert payload.tool_schemas[0] == tool_schema
        # invoke_kwargs should NOT have tools
        assert "tools" not in payload.invoke_kwargs
        assert "tool_choice" not in payload.invoke_kwargs
        # Other kwargs preserved
        assert payload.invoke_kwargs.get("temperature") == 0.7


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
