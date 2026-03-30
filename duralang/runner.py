"""DuraRunner — Temporal client + worker lifecycle management.

Singleton per (temporal_host, task_queue) pair.
Started lazily on first @dura invocation.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from duralang.config import DuraConfig
from duralang.graph_def import WorkflowPayload, WorkflowResult
from duralang.state import ArgSerializer

logger = logging.getLogger(__name__)

_runners: dict[str, DuraRunner] = {}
_runners_lock = asyncio.Lock()

# Registry of @dura-decorated function paths. Prevents arbitrary code execution
# via crafted Temporal payloads (e.g. fn_path="os:system").
_DURA_REGISTRY: set[str] = set()


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
        async with _runners_lock:
            if key not in _runners:
                runner = cls(config)
                await runner._start()  # If this fails, runner NOT cached
                _runners[key] = runner  # Only cache after successful start
        return _runners[key]

    @classmethod
    def clear(cls) -> None:
        """Clear all runners. Used in tests."""
        _runners.clear()

    async def shutdown(self) -> None:
        """Gracefully shut down worker and client."""
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        key = f"{self.config.temporal_host}:{self.config.task_queue}"
        _runners.pop(key, None)

    @classmethod
    async def shutdown_all(cls) -> None:
        """Shut down all runners gracefully."""
        for runner in list(_runners.values()):
            await runner.shutdown()
        _runners.clear()

    async def _start(self) -> None:
        from temporalio.client import Client
        from temporalio.worker import Worker

        from duralang.activities import llm_activity, mcp_activity, tool_activity
        from duralang.workflow import DuraLangWorkflow

        tls_config = None
        if self.config.tls_client_cert:
            from pathlib import Path

            from temporalio.client import TLSConfig

            tls_config = TLSConfig(
                client_cert=Path(self.config.tls_client_cert).read_bytes(),
                client_private_key=Path(self.config.tls_client_key).read_bytes()
                if self.config.tls_client_key
                else b"",
            )

        self._client = await Client.connect(
            self.config.temporal_host,
            namespace=self.config.temporal_namespace,
            tls=tls_config,
        )
        self._worker = Worker(
            self._client,
            task_queue=self.config.task_queue,
            workflows=[DuraLangWorkflow],
            activities=[llm_activity, tool_activity, mcp_activity],
        )
        self._worker_task = asyncio.create_task(self._worker.run())
        self._worker_task.add_done_callback(self._on_worker_done)

    def _on_worker_done(self, task: asyncio.Task) -> None:
        """Handle worker task completion (crash recovery)."""
        if task.cancelled():
            logger.debug("DuraLang worker stopped")
        elif task.exception():
            logger.error(f"DuraLang worker crashed: {task.exception()}")
        # Remove from cache so next call re-creates
        key = f"{self.config.temporal_host}:{self.config.task_queue}"
        _runners.pop(key, None)

    async def run(
        self, fn: Any, args: tuple, kwargs: dict, workflow_id: str | None = None
    ) -> Any:
        """Execute fn as a top-level DuraLangWorkflow.

        Args:
            fn: The @dura-decorated function to execute.
            args: Positional arguments.
            kwargs: Keyword arguments.
            workflow_id: Optional fixed workflow ID. If provided and a workflow
                with this ID is already running, reconnects to it instead of
                starting a new one. Useful for crash recovery.
        """
        fn_path = _get_fn_path(fn)
        wf_id = workflow_id or f"duralang-{fn.__name__}-{uuid.uuid4().hex[:8]}"

        serialized_args = ArgSerializer.serialize(args)
        serialized_kwargs = ArgSerializer.serialize_kwargs(kwargs)
        ArgSerializer.validate_payload_size(serialized_args, serialized_kwargs)

        payload = WorkflowPayload(
            fn_path=fn_path,
            args=serialized_args,
            kwargs=serialized_kwargs,
            config_dict=_serialize_config(self.config),
        )

        try:
            handle = await self._client.start_workflow(
                "DuraLangWorkflow",
                payload,
                id=wf_id,
                task_queue=self.config.task_queue,
                result_type=WorkflowResult,
            )
        except Exception as e:
            # If workflow already exists (e.g. resuming after crash),
            # reconnect to the existing workflow handle.
            from temporalio.exceptions import WorkflowAlreadyStartedError

            if isinstance(e, WorkflowAlreadyStartedError):
                handle = self._client.get_workflow_handle(
                    wf_id, result_type=WorkflowResult
                )
            else:
                raise

        result: WorkflowResult = await handle.result()
        if result.error or result.error_type:
            from duralang.exceptions import WorkflowFailedError

            error_msg = f"[{result.error_type}] {result.error}" if result.error_type else result.error
            raise WorkflowFailedError(error_msg)
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

    # When run as `python script.py`, __module__ is "__main__".
    # Resolve to the real module path so child workflows can find
    # the function even if the worker restarts with a different entry point.
    if module == "__main__":
        import sys

        main_mod = sys.modules.get("__main__")
        if main_mod and hasattr(main_mod, "__spec__") and main_mod.__spec__:
            module = main_mod.__spec__.name
        elif main_mod and hasattr(main_mod, "__file__") and main_mod.__file__:
            import os

            # Convert file path to module path
            file_path = os.path.abspath(main_mod.__file__)
            for path in sorted(sys.path, key=len, reverse=True):
                if path and file_path.startswith(path):
                    rel = os.path.relpath(file_path, path)
                    module = rel.replace(os.sep, ".").removesuffix(".py")
                    break

    return f"{module}:{qualname}"


def _resolve_callable(fn_path: str) -> Any:
    """Resolves 'my_module:my_agent' -> the original unwrapped function.

    Validates that fn_path is in _DURA_REGISTRY to prevent arbitrary code
    execution via crafted Temporal payloads. Then unwraps __wrapped__ to get
    the original function body — without this, the wrapper would see
    DuraContext is set and spawn infinite child workflows.
    """
    import importlib

    from duralang.exceptions import ConfigurationError

    module_path, fn_name = fn_path.rsplit(":", 1)

    # When run as `python script.py`, __main__ and the resolved module path
    # (e.g. "examples.mcp_agent") are separate module objects with separate
    # globals. Prefer __main__ when it maps to the same module — this ensures
    # module-level state (e.g. MCP clients) set by the user is visible.
    import sys

    main_mod = sys.modules.get("__main__")
    main_spec_name = getattr(getattr(main_mod, "__spec__", None), "name", None)
    if main_mod and main_spec_name == module_path:
        module = main_mod
    else:
        # Import the module — this triggers @dura decorators which
        # register functions in _DURA_REGISTRY. Must happen before the check.
        module = importlib.import_module(module_path)

    if fn_path not in _DURA_REGISTRY:
        raise ConfigurationError(
            f"Function '{fn_path}' is not registered with @dura. "
            f"Only @dura-decorated functions can be resolved."
        )
    fn = getattr(module, fn_name)

    if not getattr(fn, "__dura__", False):
        raise ConfigurationError(f"Function '{fn_path}' is not @dura-decorated.")

    # Unwrap the @dura decorator to get the original function body
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
