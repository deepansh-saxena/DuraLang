"""Convenience layer for exposing @dura functions as LangChain tools.

Wraps a @dura-decorated function in a real BaseTool subclass so it can be
used with ``create_agent`` alongside regular @tool functions.

Under the hood, calling an agent tool from within a @dura context triggers
a Temporal Child Workflow (not a dura__tool activity), giving the sub-agent
its own event history, timeouts, and retry boundaries.

Usage:
    from langchain.agents import create_agent
    from duralang import dura, dura_agent_tool

    @dura
    async def researcher(query: str) -> str:
        \"\"\"Research agent — gathers info via web search.\"\"\"
        ...

    @dura
    async def analyst(data: str, question: str) -> str:
        \"\"\"Analysis agent — runs calculations.\"\"\"
        ...

    # These are real BaseTool instances — mix with any @tool
    all_tools = [
        dura_agent_tool(researcher),
        dura_agent_tool(analyst),
        get_weather,     # regular @tool
        calculator,      # regular @tool
    ]

    @dura
    async def orchestrator(task: str) -> str:
        agent = create_agent(
            model="claude-sonnet-4-6",
            tools=all_tools,
        )
        # create_agent handles dispatch for both agent tools and regular tools
        result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})
        return result["messages"][-1].content
"""

import inspect
from typing import Any, get_type_hints

from langchain_core.tools import BaseTool
from pydantic import BaseModel, create_model

from duralang.exceptions import ConfigurationError

_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

_PYDANTIC_TYPE_MAP = {
    str: (str, ...),
    int: (int, ...),
    float: (float, ...),
    bool: (bool, ...),
}


def dura_agent_tool(
    fn, *, name: str | None = None, description: str | None = None
) -> BaseTool:
    """Create a BaseTool from a @dura-decorated async function.

    The returned tool is a real LangChain BaseTool that can be mixed with
    regular @tool functions in bind_tools() and dispatched with ainvoke().

    When called from within a @dura context, the tool calls the @dura
    function directly, which routes through Temporal as a Child Workflow.

    Args:
        fn: A @dura-decorated async function.
        name: Override tool name (default: "call_{fn.__name__}").
        description: Override description (default: function docstring).
    """
    if not getattr(fn, "__dura__", False):
        raise ConfigurationError(
            f"{fn.__name__} is not a @dura function. "
            f"Decorate it with @dura before passing to dura_agent_tool()."
        )

    tool_name = name or f"call_{fn.__name__}"
    tool_desc = description or fn.__doc__ or f"Call the {fn.__name__} agent."

    # Build Pydantic model from function signature for args_schema
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)
    fields = {}

    for param_name, param in sig.parameters.items():
        hint = hints.get(param_name, str)
        if param.default is inspect.Parameter.empty:
            fields[param_name] = (hint, ...)
        else:
            fields[param_name] = (hint, param.default)

    args_model = create_model(f"{tool_name}_args", **fields)

    # Capture fn in closure for the tool class
    _dura_fn = fn

    class DuraAgentBaseTool(BaseTool):
        """A BaseTool that dispatches to a @dura function as a Child Workflow."""

        name: str = tool_name
        description: str = tool_desc
        args_schema: type[BaseModel] = args_model

        # Mark this tool so the proxy knows NOT to route it through dura__tool
        # activity — it handles its own durable execution via child workflow.
        __dura_agent_tool__: bool = True

        async def _arun(self, **kwargs: Any) -> str:
            # Filter to only the parameters the @dura function expects.
            # LangChain/LangGraph may inject extra kwargs (run_manager, config, etc.)
            # that the user function doesn't accept.
            expected = set(args_model.model_fields.keys())
            filtered = {k: v for k, v in kwargs.items() if k in expected}
            result = await _dura_fn(**filtered)
            return str(result) if result is not None else ""

        def _run(self, **kwargs: Any) -> str:
            raise NotImplementedError(
                f"Agent tool '{self.name}' is async-only. Use ainvoke() instead."
            )

    tool_instance = DuraAgentBaseTool()

    # Mark instance so proxy can detect it
    object.__setattr__(tool_instance, "__dura_agent_tool__", True)

    return tool_instance
