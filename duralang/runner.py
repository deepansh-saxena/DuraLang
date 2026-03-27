"""DuraRunner — Temporal client + worker lifecycle management.

Singleton per (temporal_host, task_queue) pair.
Started lazily on first @dura invocation.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from duralang.config import DuraConfig
from duralang.graph_def import WorkflowPayload, WorkflowResult
from duralang.state import ArgSerializer

_runners: dict[str, DuraRunner] = {}


class DuraRunner:
    """Manages the Temporal client and worker for a given DuraConfig."""

    def __init__(self, config: DuraConfig) -> None:
        self.config = config
        self._client = None
        self._worker = None
        self._worker_task = None

    @classmethod
    async def get_or_create(cls, config: DuraConfig) -> DuraRunner:
        key = f"{config.temporal_host}:{config.task_queue}"
        if key not in _runners:
            runner = cls(config)
            await runner._start()
            _runners[key] = runner
        return _runners[key]

    @classmethod
    def clear(cls) -> None:
        """Clear all runners. Used in tests."""
        _runners.clear()

    async def _start(self) -> None:
        from temporalio.client import Client
        from temporalio.worker import Worker

        from duralang.activities import llm_activity, mcp_activity, tool_activity
        from duralang.workflow import DuraLangWorkflow

        self._client = await Client.connect(
            self.config.temporal_host,
            namespace=self.config.temporal_namespace,
        )
        self._worker = Worker(
            self._client,
            task_queue=self.config.task_queue,
            workflows=[DuraLangWorkflow],
            activities=[llm_activity, tool_activity, mcp_activity],
        )
        self._worker_task = asyncio.create_task(self._worker.run())

    async def run(self, fn: Any, args: tuple, kwargs: dict) -> Any:
        """Execute fn as a top-level DuraLangWorkflow."""
        fn_path = _get_fn_path(fn)
        wf_id = f"duralang-{fn.__name__}-{uuid.uuid4().hex[:8]}"

        handle = await self._client.start_workflow(
            "DuraLangWorkflow",
            WorkflowPayload(
                fn_path=fn_path,
                args=ArgSerializer.serialize(args),
                kwargs=ArgSerializer.serialize_kwargs(kwargs),
                config_dict=_serialize_config(self.config),
            ),
            id=wf_id,
            task_queue=self.config.task_queue,
            result_type=WorkflowResult,
        )
        result: WorkflowResult = await handle.result()
        if result.error:
            from duralang.exceptions import WorkflowFailedError

            raise WorkflowFailedError(result.error)
        return ArgSerializer.deserialize_result(result.return_value)


def _get_fn_path(fn: Any) -> str:
    """Returns importable path: 'my_module:my_agent'."""
    from duralang.exceptions import ConfigurationError

    # Unwrap the @dura wrapper to get the original function
    original = getattr(fn, "__wrapped__", fn)

    if "<lambda>" in getattr(original, "__qualname__", ""):
        raise ConfigurationError("@dura cannot wrap lambda functions.")

    module = original.__module__
    qualname = original.__qualname__
    return f"{module}:{qualname}"


def _resolve_callable(fn_path: str) -> Any:
    """Resolves 'my_module:my_agent' -> the original unwrapped function.

    Important: getattr(module, name) returns the @dura wrapper. We must
    unwrap via __wrapped__ to get the original function body, otherwise
    the wrapper would see DuraContext is set and spawn infinite child workflows.
    """
    import importlib

    module_path, fn_name = fn_path.split(":", 1)
    module = importlib.import_module(module_path)
    fn = getattr(module, fn_name)
    # Unwrap the @dura decorator to get the original function
    return getattr(fn, "__wrapped__", fn)


def _serialize_config(config: DuraConfig) -> dict:
    """Serialize DuraConfig to a dict for workflow payload."""
    return {
        "temporal_host": config.temporal_host,
        "temporal_namespace": config.temporal_namespace,
        "task_queue": config.task_queue,
        "max_iterations": config.max_iterations,
        "child_workflow_timeout_seconds": config.child_workflow_timeout.total_seconds(),
    }
