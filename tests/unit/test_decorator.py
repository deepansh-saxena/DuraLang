"""Tests for @dura decorator — wrapping behavior."""

from __future__ import annotations

import pytest

from duralang.config import DuraConfig
from duralang.decorator import dura


class TestDuraDecorator:
    def test_bare_decorator(self):
        @dura
        async def my_agent(messages):
            return messages

        assert hasattr(my_agent, "__dura__")
        assert my_agent.__dura__ is True
        assert isinstance(my_agent.__dura_config__, DuraConfig)

    def test_decorator_with_config(self):
        config = DuraConfig(temporal_host="custom:7233")

        @dura(config=config)
        async def my_agent(messages):
            return messages

        assert my_agent.__dura__ is True
        assert my_agent.__dura_config__.temporal_host == "custom:7233"

    def test_preserves_function_name(self):
        @dura
        async def research_agent(messages):
            """Does research."""
            return messages

        assert research_agent.__name__ == "research_agent"
        assert research_agent.__doc__ == "Does research."

    def test_default_config(self):
        @dura
        async def my_agent(messages):
            return messages

        config = my_agent.__dura_config__
        assert config.temporal_host == "localhost:7233"
        assert config.task_queue == "duralang"
        assert config.max_iterations == 50
