"""Tests for dura_agent_tool — BaseTool generation from @dura functions."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.tools import BaseTool

from duralang.agent_tool import dura_agent_tool
from duralang.config import DuraConfig
from duralang.context import DuraContext
from duralang.decorator import dura
from duralang.exceptions import ConfigurationError


# ── Fixtures ─────────────────────────────────────────────────────────────────


@dura
async def search_agent(query: str) -> str:
    """Search agent — finds information on the web."""
    return f"results for {query}"


@dura
async def analysis_agent(data: str, question: str) -> str:
    """Analysis agent — runs calculations and identifies trends."""
    return f"analysis of {data}: {question}"


@dura
async def greeter(name: str, formal: bool) -> str:
    """Greet someone."""
    return f"Hello, {name}!"


@dura
async def flexible_agent(query: str, max_results: int, verbose: bool) -> str:
    """Agent with multiple typed parameters."""
    return query


@dura
async def with_defaults(query: str, limit: int = 10) -> str:
    """Agent with a default parameter."""
    return query


# ── dura_agent_tool returns a real BaseTool ──────────────────────────────────


class TestDuraAgentTool:
    def test_returns_base_tool(self):
        tool = dura_agent_tool(search_agent)
        assert isinstance(tool, BaseTool)

    def test_tool_name(self):
        tool = dura_agent_tool(search_agent)
        assert tool.name == "call_search_agent"

    def test_tool_description_from_docstring(self):
        tool = dura_agent_tool(search_agent)
        assert tool.description == "Search agent — finds information on the web."

    def test_custom_name(self):
        tool = dura_agent_tool(search_agent, name="find_stuff")
        assert tool.name == "find_stuff"

    def test_custom_description(self):
        tool = dura_agent_tool(search_agent, description="Custom desc.")
        assert tool.description == "Custom desc."

    def test_args_schema_single_param(self):
        tool = dura_agent_tool(search_agent)
        schema = tool.args_schema.model_json_schema()
        assert "query" in schema["properties"]
        assert schema["properties"]["query"]["type"] == "string"
        assert "query" in schema["required"]

    def test_args_schema_multi_param(self):
        tool = dura_agent_tool(analysis_agent)
        schema = tool.args_schema.model_json_schema()
        assert "data" in schema["properties"]
        assert "question" in schema["properties"]
        assert set(schema["required"]) == {"data", "question"}

    def test_args_schema_type_hints(self):
        tool = dura_agent_tool(flexible_agent)
        schema = tool.args_schema.model_json_schema()
        assert schema["properties"]["query"]["type"] == "string"
        assert schema["properties"]["max_results"]["type"] == "integer"
        assert schema["properties"]["verbose"]["type"] == "boolean"

    def test_args_schema_with_defaults(self):
        tool = dura_agent_tool(with_defaults)
        schema = tool.args_schema.model_json_schema()
        assert "query" in schema["required"]
        # 'limit' has a default so it should NOT be required
        assert "limit" not in schema.get("required", [])

    def test_marked_as_agent_tool(self):
        tool = dura_agent_tool(search_agent)
        assert getattr(tool, "__dura_agent_tool__", False) is True

    def test_rejects_non_dura_function(self):
        async def plain_fn(query: str) -> str:
            return query

        with pytest.raises(ConfigurationError, match="not a @dura function"):
            dura_agent_tool(plain_fn)

    def test_sync_run_raises(self):
        tool = dura_agent_tool(search_agent)
        with pytest.raises(NotImplementedError, match="async-only"):
            tool._run(query="test")

    @pytest.mark.asyncio
    async def test_arun_filters_unexpected_kwargs(self):
        """_arun should ignore kwargs not in the function signature."""
        tool = dura_agent_tool(search_agent)
        original_fn = search_agent.__wrapped__

        async def mock_child(fn, args, kwargs):
            return await original_fn(*args, **kwargs)

        ctx = DuraContext(
            workflow_id="test",
            config=DuraConfig(),
            execute_activity=AsyncMock(),
            execute_child_agent=mock_child,
        )
        token = DuraContext.set(ctx)
        try:
            result = await tool._arun(query="test", run_manager=object(), config={"key": "val"})
        finally:
            DuraContext.reset(token)
        assert result == "results for test"

    @pytest.mark.asyncio
    async def test_arun_passes_expected_kwargs(self):
        """_arun should pass through all expected parameters."""
        tool = dura_agent_tool(analysis_agent)
        original_fn = analysis_agent.__wrapped__

        async def mock_child(fn, args, kwargs):
            return await original_fn(*args, **kwargs)

        ctx = DuraContext(
            workflow_id="test",
            config=DuraConfig(),
            execute_activity=AsyncMock(),
            execute_child_agent=mock_child,
        )
        token = DuraContext.set(ctx)
        try:
            result = await tool._arun(data="nums", question="what trend?", extra_junk="ignore")
        finally:
            DuraContext.reset(token)
        assert result == "analysis of nums: what trend?"


# ── Mixing with regular tools ────────────────────────────────────────────────


class TestMixWithRegularTools:
    def test_can_coexist_in_list(self):
        from langchain_core.tools import tool as lc_tool

        @lc_tool
        def calculator(expression: str) -> str:
            """Evaluate a math expression."""
            return str(eval(expression))

        agent_t = dura_agent_tool(search_agent)
        all_tools = [agent_t, calculator]

        assert len(all_tools) == 2
        assert all(isinstance(t, BaseTool) for t in all_tools)

    def test_tools_by_name_dict(self):
        from langchain_core.tools import tool as lc_tool

        @lc_tool
        def calculator(expression: str) -> str:
            """Evaluate a math expression."""
            return str(eval(expression))

        agent_t = dura_agent_tool(search_agent)
        all_tools = [agent_t, calculator]
        by_name = {t.name: t for t in all_tools}

        assert "call_search_agent" in by_name
        assert "calculator" in by_name
        assert getattr(by_name["call_search_agent"], "__dura_agent_tool__", False)
        assert not getattr(by_name["calculator"], "__dura_agent_tool__", False)
