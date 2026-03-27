"""Tests for DuraContext — ContextVar injection and retrieval."""

from __future__ import annotations

import asyncio

import pytest

from duralang.config import DuraConfig
from duralang.context import DuraContext


class TestDuraContext:
    def test_default_is_none(self):
        assert DuraContext.get() is None

    def test_set_and_get(self):
        ctx = DuraContext(
            workflow_id="test-wf-1",
            config=DuraConfig(),
            execute_activity=lambda *a: None,
            execute_child_agent=lambda *a: None,
        )
        token = DuraContext.set(ctx)
        try:
            assert DuraContext.get() is ctx
            assert DuraContext.get().workflow_id == "test-wf-1"
        finally:
            DuraContext.reset(token)
        assert DuraContext.get() is None

    def test_reset_restores_none(self):
        ctx = DuraContext(
            workflow_id="test",
            config=DuraConfig(),
            execute_activity=lambda *a: None,
            execute_child_agent=lambda *a: None,
        )
        token = DuraContext.set(ctx)
        DuraContext.reset(token)
        assert DuraContext.get() is None

    @pytest.mark.asyncio
    async def test_propagates_through_asyncio(self):
        ctx = DuraContext(
            workflow_id="async-test",
            config=DuraConfig(),
            execute_activity=lambda *a: None,
            execute_child_agent=lambda *a: None,
        )
        token = DuraContext.set(ctx)

        async def check_context():
            return DuraContext.get()

        result = await check_context()
        DuraContext.reset(token)
        assert result is ctx
        assert result.workflow_id == "async-test"
