"""DuraLangWorkflow — Temporal workflow that executes a @dura-decorated function.

The workflow:
1. Resolves the user's function from its importable path
2. Sets up DuraContext so proxy objects route calls to Temporal Activities
3. Calls the user's original function body
4. Returns the serialized result
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.workflow import ParentClosePolicy

with workflow.unsafe.imports_passed_through():
    from duralang.config import DuraConfig
    from duralang.context import DuraContext
    from duralang.graph_def import (
        LLMActivityPayload,
        LLMActivityResult,
        MCPActivityPayload,
        MCPActivityResult,
        ToolActivityPayload,
        ToolActivityResult,
        WorkflowPayload,
        WorkflowResult,
    )
    from duralang.runner import _resolve_callable, _get_fn_path, _serialize_config
    from duralang.state import ArgSerializer

    # Pre-import LangChain modules to avoid Temporal's 2-second deadlock
    # detector triggering during lazy imports inside the workflow.
    # dura_agent → create_agent → init_chat_model triggers a chain of imports
    # (langchain_anthropic → yaml → etc.) that can exceed 2 seconds.
    import langchain.agents  # noqa: F401

    import duralang.dura_agent  # noqa: F401
    import duralang.dura_model  # noqa: F401
    import duralang.dura_tool  # noqa: F401

    try:
        import langchain_anthropic  # noqa: F401
    except ImportError:
        pass
    try:
        import langchain_openai  # noqa: F401
    except ImportError:
        pass


def _sanitize_error(error_str: str) -> str:
    """Strip file paths and connection strings from error messages.

    Prevents leaking internal paths into Temporal event history.
    """
    import re

    # Windows paths (C:\..., D:\...)
    error_str = re.sub(r'[A-Za-z]:\\[^\s"\']+', '<path>', error_str)
    # Unix paths (any /path/to/something with depth)
    error_str = re.sub(r'/[^\s"\']*(?:/[^\s"\']+)+', '<path>', error_str)
    # Connection strings
    error_str = re.sub(
        r'(postgres|mysql|redis|amqp|mongodb)://[^\s"\']+',
        '<connection>',
        error_str,
    )
    return error_str


# sandboxed=False is required because LangChain's import chain (langchain →
# langchain_anthropic → yaml → etc.) exceeds Temporal's 2-second deadlock
# detector. The tradeoff: non-deterministic code in the user function body
# won't be caught by the sandbox. This is acceptable because user functions
# should only perform LLM/tool/MCP operations, all of which are routed
# through Temporal Activities (which ARE deterministic replay-safe).
@workflow.defn(name="DuraLangWorkflow", sandboxed=False)
class DuraLangWorkflow:
    """Temporal workflow that runs a user's @dura-decorated function."""

    @workflow.run
    async def run(self, payload: WorkflowPayload) -> WorkflowResult:
        """Execute the user's function with DuraContext set."""
        config = self._build_dura_config(payload.config_dict)

        try:
            # Resolve the user's function from its importable path
            fn = _resolve_callable(payload.fn_path)
            # Deserialize arguments
            args, kwargs = ArgSerializer.deserialize(payload.args, payload.kwargs)
        except Exception as e:
            import traceback as tb
            return WorkflowResult(
                return_value=None,
                error=_sanitize_error(str(e)),
                error_type=type(e).__qualname__,
                error_traceback=_sanitize_error(tb.format_exc()),
            )

        # Build the DuraContext closures
        async def _execute_activity(
            activity_name: str, activity_payload: Any, activity_config: Any
        ) -> Any:
            result_type_map = {
                "dura__llm": LLMActivityResult,
                "dura__tool": ToolActivityResult,
                "dura__mcp": MCPActivityResult,
            }
            return await workflow.execute_activity(
                activity_name,
                activity_payload,
                result_type=result_type_map.get(activity_name),
                start_to_close_timeout=activity_config.start_to_close_timeout,
                retry_policy=activity_config.retry_policy,
                heartbeat_timeout=activity_config.heartbeat_timeout,
            )

        _child_counter = 0

        async def _execute_child_agent(
            fn: Any, args: tuple, kwargs: dict
        ) -> Any:
            nonlocal _child_counter
            _child_counter += 1
            child_fn_path = _get_fn_path(fn)
            child_fn_name = child_fn_path.split(":")[-1]
            child_payload = WorkflowPayload(
                fn_path=child_fn_path,
                args=ArgSerializer.serialize(args),
                kwargs=ArgSerializer.serialize_kwargs(kwargs),
                config_dict=payload.config_dict,
            )
            # Use only the root workflow ID to prevent unbounded ID growth
            root_id = workflow.info().workflow_id.split("--child--")[0]
            child_id = (
                f"{root_id}"
                f"--child--{child_fn_name}"
                f"-{_child_counter}"
                f"--{workflow.info().run_id[:8]}"
            )
            result: WorkflowResult = await workflow.execute_child_workflow(
                DuraLangWorkflow.run,
                child_payload,
                id=child_id,
                task_queue=config.task_queue,
                execution_timeout=config.child_workflow_timeout,
                parent_close_policy=ParentClosePolicy.TERMINATE,
            )
            if result.error:
                from duralang.exceptions import WorkflowFailedError

                error_msg = f"[{result.error_type}] {result.error}" if result.error_type else result.error
                raise WorkflowFailedError(error_msg)
            return ArgSerializer.deserialize_result(result.return_value)

        ctx = DuraContext(
            workflow_id=workflow.info().workflow_id,
            config=config,
            execute_activity=_execute_activity,
            execute_child_agent=_execute_child_agent,
        )

        # Install context — all proxy objects will now route through Temporal
        token = DuraContext.set(ctx)
        try:
            return_value = await fn(*args, **kwargs)
        except Exception as e:
            import traceback as tb

            return WorkflowResult(
                return_value=None,
                error=_sanitize_error(str(e)),
                error_type=type(e).__qualname__,
                error_traceback=_sanitize_error(tb.format_exc()),
            )
        finally:
            DuraContext.reset(token)

        return WorkflowResult(
            return_value=ArgSerializer.serialize_result(return_value),
        )

    def _build_dura_config(self, config_dict: dict) -> DuraConfig:
        """Reconstruct DuraConfig from serialized dict."""
        config = DuraConfig()
        for key in ("temporal_host", "temporal_namespace", "task_queue", "max_iterations"):
            if key in config_dict:
                setattr(config, key, config_dict[key])
        if "child_workflow_timeout_seconds" in config_dict:
            config.child_workflow_timeout = timedelta(
                seconds=config_dict["child_workflow_timeout_seconds"]
            )
        return config
