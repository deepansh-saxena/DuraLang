"""Shared test fixtures for DuraLang tests."""

from __future__ import annotations

import pytest

from duralang.registry import MCPSessionRegistry, ToolRegistry
from duralang.runner import DuraRunner


@pytest.fixture(autouse=True, scope="function")
def reset_duralang_globals():
    """Reset all DuraLang global state before and after each test."""
    ToolRegistry.clear()
    MCPSessionRegistry.clear()
    yield
    ToolRegistry.clear()
    MCPSessionRegistry.clear()
    DuraRunner.clear()
    # Reset MESSAGE_MAP so tests get fresh state
    import duralang.state as _state

    _state.MESSAGE_MAP = None
