"""DuraContext — ContextVar-based bridge between proxy objects and Temporal workflow."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Any, Callable

from duralang.config import DuraConfig

_dura_context: contextvars.ContextVar[DuraContext | None] = contextvars.ContextVar(
    "dura_context", default=None
)


@dataclass
class DuraContext:
    """Set in the ContextVar when user code executes inside a DuraLangWorkflow.

    Proxy objects read this to know how to route their calls.
    Returns None in activity threads (by design) — activities run on the
    Temporal worker's thread pool, not in the workflow event loop.
    """

    workflow_id: str
    config: DuraConfig
    execute_activity: Callable  # (activity_name, payload, activity_config) -> result
    execute_child_agent: Callable  # (fn, args, kwargs) -> result

    @classmethod
    def get(cls) -> DuraContext | None:
        return _dura_context.get()

    @classmethod
    def set(cls, ctx: DuraContext) -> contextvars.Token:
        return _dura_context.set(ctx)

    @classmethod
    def reset(cls, token: contextvars.Token) -> None:
        _dura_context.reset(token)
