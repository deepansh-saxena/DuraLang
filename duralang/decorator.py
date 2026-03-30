"""@dura decorator — the entire public API of DuraLang.

Usage:
    @dura
    async def my_agent(messages): ...

    @dura(config=DuraConfig(...))
    async def my_agent(messages): ...
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from duralang.config import DuraConfig
from duralang.context import DuraContext


def dura(_fn=None, *, config: DuraConfig | None = None):
    """Decorator that makes a LangChain agent function durable via Temporal.

    Supports both @dura and @dura(config=...) usage patterns.
    """

    def decorator(fn: Callable) -> Callable:
        _config = config or DuraConfig()
        fn.__dura__ = True
        fn.__dura_config__ = _config

        # Register in _DURA_REGISTRY so _resolve_callable can validate
        from duralang.runner import _DURA_REGISTRY, _get_fn_path
        fn_path = _get_fn_path(fn)
        _DURA_REGISTRY.add(fn_path)

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract DuraLang-specific kwargs before passing to the function
            workflow_id = kwargs.pop("_workflow_id", None)

            # Check if we are ALREADY inside a dura workflow context
            ctx = DuraContext.get()
            if ctx is not None:
                # Inside a parent dura workflow — this becomes a child workflow
                return await ctx.execute_child_agent(fn, args, kwargs)

            # Top-level call — start a new DuraLangWorkflow on Temporal
            from duralang.runner import DuraRunner

            runner = await DuraRunner.get_or_create(_config)
            return await runner.run(fn, args, kwargs, workflow_id=workflow_id)

        wrapper.__dura__ = True
        wrapper.__dura_config__ = _config
        return wrapper

    if _fn is not None:
        # Called as @dura (no parentheses)
        return decorator(_fn)
    # Called as @dura(...) (with parentheses)
    return decorator
