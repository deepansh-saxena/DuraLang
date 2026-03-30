"""Integration tests — @dura decorator against Temporal test server.

These tests spin up a local Temporal dev server, register workflows + activities,
and run @dura-decorated functions end-to-end. LLMs are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from duralang.config import DuraConfig, LLMIdentity
from duralang.context import DuraContext
from duralang.graph_def import (
    LLMActivityPayload,
    LLMActivityResult,
    ToolActivityPayload,
    ToolActivityResult,
    WorkflowPayload,
    WorkflowResult,
)
from duralang.registry import ToolRegistry
from duralang.runner import _DURA_REGISTRY, _serialize_config
from duralang.state import ArgSerializer, MessageSerializer
from duralang.workflow import DuraLangWorkflow


# ── Mock activities ───────────────────────────────────────────────────────────


@activity.defn(name="dura__llm")
async def mock_llm_activity(payload: LLMActivityPayload) -> LLMActivityResult:
    """Mock LLM that returns a simple response."""
    ai_msg = AIMessage(content="Hello from mock LLM!")
    return LLMActivityResult(
        ai_message=MessageSerializer.serialize(ai_msg),
        content="Hello from mock LLM!",
    )


@activity.defn(name="dura__tool")
async def mock_tool_activity(payload: ToolActivityPayload) -> ToolActivityResult:
    """Mock tool that returns a fixed result."""
    return ToolActivityResult(
        output=f"Tool {payload.tool_name} result",
        tool_call_id=payload.tool_call_id,
    )


@activity.defn(name="dura__mcp")
async def mock_mcp_activity(payload):
    pass


# ── Test function that will be used inside workflows ────────────────────────

# We define a simple agent function that can be resolved by path
async def simple_echo_agent(messages: list) -> list:
    """Simple agent that just returns messages unchanged.
    In real use, LLM proxy would intercept ainvoke calls."""
    return messages

simple_echo_agent.__dura__ = True


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_workflow_payload(
    fn_path: str,
    args: tuple = (),
    kwargs: dict = None,
    config: DuraConfig = None,
) -> WorkflowPayload:
    config = config or DuraConfig()
    return WorkflowPayload(
        fn_path=fn_path,
        args=ArgSerializer.serialize(args),
        kwargs=ArgSerializer.serialize_kwargs(kwargs or {}),
        config_dict=_serialize_config(config),
    )


@pytest.fixture(autouse=True)
def clean_registries():
    ToolRegistry.clear()
    # Register test functions so _resolve_callable accepts them
    test_fns = [
        f"{__name__}:simple_echo_agent",
        f"{__name__}:context_checker",
        f"{__name__}:failing_function",
        f"{__name__}:adder",
    ]
    for fn_path in test_fns:
        _DURA_REGISTRY.add(fn_path)
    yield
    ToolRegistry.clear()
    for fn_path in test_fns:
        _DURA_REGISTRY.discard(fn_path)


# ── Tests ────────────────────────────────────────────────────────────────────


class TestWorkflowExecution:
    """Test that DuraLangWorkflow can execute a function via Temporal."""

    @pytest.mark.asyncio
    async def test_simple_function_execution(self):
        """Workflow resolves function, deserializes args, calls it, returns result."""
        messages = [HumanMessage(content="Hello!")]
        payload = _make_workflow_payload(
            fn_path=f"{__name__}:simple_echo_agent",
            args=(messages,),
        )

        async with await WorkflowEnvironment.start_local() as env:
            async with Worker(
                env.client,
                task_queue="duralang",
                workflows=[DuraLangWorkflow],
                activities=[mock_llm_activity, mock_tool_activity, mock_mcp_activity],
            ):
                result = await env.client.execute_workflow(
                    DuraLangWorkflow.run,
                    payload,
                    id="test-simple-1",
                    task_queue="duralang",
                )

        assert isinstance(result, WorkflowResult)
        assert result.error is None
        # The return value is the serialized list of messages
        deserialized = ArgSerializer.deserialize_result(result.return_value)
        assert len(deserialized) == 1
        assert isinstance(deserialized[0], HumanMessage)
        assert deserialized[0].content == "Hello!"


class TestWorkflowWithDuraContext:
    """Test that DuraContext is set during function execution."""

    @pytest.mark.asyncio
    async def test_context_set_during_execution(self):
        """Function can read DuraContext inside the workflow."""

        async def context_checker() -> dict:
            ctx = DuraContext.get()
            if ctx is None:
                return {"has_context": False}
            return {
                "has_context": True,
                "workflow_id_prefix": ctx.workflow_id[:10],
            }

        payload = _make_workflow_payload(
            fn_path=f"{__name__}:context_checker",
        )

        async with await WorkflowEnvironment.start_local() as env:
            async with Worker(
                env.client,
                task_queue="duralang",
                workflows=[DuraLangWorkflow],
                activities=[mock_llm_activity, mock_tool_activity, mock_mcp_activity],
            ):
                result = await env.client.execute_workflow(
                    DuraLangWorkflow.run,
                    payload,
                    id="test-context-1",
                    task_queue="duralang",
                )

        assert result.error is None
        value = ArgSerializer.deserialize_result(result.return_value)
        assert value["has_context"] is True


# Must be module-level for workflow resolution — marked as @dura for _resolve_callable
async def context_checker() -> dict:
    ctx = DuraContext.get()
    if ctx is None:
        return {"has_context": False}
    return {
        "has_context": True,
        "workflow_id_prefix": ctx.workflow_id[:10],
    }

context_checker.__dura__ = True


class TestWorkflowErrorHandling:
    """Test that errors in user functions are captured."""

    @pytest.mark.asyncio
    async def test_function_error_captured(self):
        payload = _make_workflow_payload(
            fn_path=f"{__name__}:failing_function",
        )

        async with await WorkflowEnvironment.start_local() as env:
            async with Worker(
                env.client,
                task_queue="duralang",
                workflows=[DuraLangWorkflow],
                activities=[mock_llm_activity, mock_tool_activity, mock_mcp_activity],
            ):
                result = await env.client.execute_workflow(
                    DuraLangWorkflow.run,
                    payload,
                    id="test-error-1",
                    task_queue="duralang",
                )

        assert result.error is not None
        assert "intentional error" in result.error


async def failing_function():
    raise ValueError("intentional error")

failing_function.__dura__ = True


class TestArgSerializationEndToEnd:
    """Test that various argument types survive the Temporal boundary."""

    @pytest.mark.asyncio
    async def test_primitive_args(self):
        payload = _make_workflow_payload(
            fn_path=f"{__name__}:adder",
            args=(3, 4),
        )

        async with await WorkflowEnvironment.start_local() as env:
            async with Worker(
                env.client,
                task_queue="duralang",
                workflows=[DuraLangWorkflow],
                activities=[mock_llm_activity, mock_tool_activity, mock_mcp_activity],
            ):
                result = await env.client.execute_workflow(
                    DuraLangWorkflow.run,
                    payload,
                    id="test-args-1",
                    task_queue="duralang",
                )

        assert result.error is None
        assert ArgSerializer.deserialize_result(result.return_value) == 7


async def adder(a: int, b: int) -> int:
    return a + b

adder.__dura__ = True
