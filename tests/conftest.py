"""Shared test fixtures for LangForge tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_state():
    """A simple serialized state dict for testing."""
    return {
        "messages": [
            {
                "type": "HumanMessage",
                "content": "Hello!",
                "tool_calls": [],
                "tool_call_id": None,
                "name": None,
                "id": None,
            }
        ],
        "counter": 0,
    }


@pytest.fixture
def sample_llm_config():
    """A basic LLM config dict for testing."""
    return {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "kwargs": {},
    }
