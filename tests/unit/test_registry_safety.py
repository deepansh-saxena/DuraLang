"""Thread-safe registry tests for ToolRegistry."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from duralang.registry import ToolRegistry


def _make_mock_tool(name: str, *, with_schema: bool = True) -> MagicMock:
    """Create a MagicMock that behaves like a BaseTool."""
    tool = MagicMock()
    tool.name = name
    if with_schema:
        tool.args_schema.model_json_schema.return_value = {"type": "object", "properties": {}}
    else:
        tool.args_schema = None
    return tool


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure every test starts and ends with a clean registry."""
    ToolRegistry.clear()
    yield
    ToolRegistry.clear()


class TestToolRegistryBasic:
    def test_register_and_get(self):
        tool = _make_mock_tool("search")
        ToolRegistry.register(tool)
        assert ToolRegistry.get("search") is tool

    def test_get_missing_returns_none(self):
        assert ToolRegistry.get("nonexistent") is None

    def test_register_same_name_warns(self):
        tool_a = _make_mock_tool("calculator")
        tool_b = _make_mock_tool("calculator")
        ToolRegistry.register(tool_a)

        with pytest.warns(UserWarning, match="Tool 'calculator' already registered"):
            ToolRegistry.register(tool_b)

        # The second instance should overwrite the first
        assert ToolRegistry.get("calculator") is tool_b

    def test_register_same_instance_no_warning(self):
        tool = _make_mock_tool("calculator")
        ToolRegistry.register(tool)

        # Re-registering the exact same instance should not warn
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ToolRegistry.register(tool)

        assert len(caught) == 0

    def test_get_schema_returns_cached_schema(self):
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        tool = _make_mock_tool("search")
        tool.args_schema.model_json_schema.return_value = schema

        ToolRegistry.register(tool)
        assert ToolRegistry.get_schema("search") == schema
        # model_json_schema should have been called exactly once (cached)
        tool.args_schema.model_json_schema.assert_called_once()

    def test_get_schema_missing_returns_none(self):
        assert ToolRegistry.get_schema("nonexistent") is None

    def test_get_schema_no_args_schema(self):
        tool = _make_mock_tool("simple", with_schema=False)
        ToolRegistry.register(tool)
        assert ToolRegistry.get_schema("simple") is None

    def test_clear_empties_registry_and_cache(self):
        tool = _make_mock_tool("search")
        ToolRegistry.register(tool)

        assert ToolRegistry.get("search") is not None
        assert ToolRegistry.get_schema("search") is not None

        ToolRegistry.clear()

        assert ToolRegistry.get("search") is None
        assert ToolRegistry.get_schema("search") is None


class TestToolRegistryThreadSafety:
    def test_concurrent_register_50_tools(self):
        """Register 50 tools from 50 threads. All must be present afterward."""
        tools = [_make_mock_tool(f"tool_{i}") for i in range(50)]
        errors: list[Exception] = []

        def register(t: MagicMock) -> None:
            try:
                ToolRegistry.register(t)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=register, args=(t,)) for t in tools]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent registration raised errors: {errors}"

        for i in range(50):
            registered = ToolRegistry.get(f"tool_{i}")
            assert registered is not None, f"tool_{i} was not registered"
            assert registered is tools[i]


# warnings import needed for the no-warning assertion
import warnings
