"""dura_agent() — factory that wraps model and tools for Temporal durability."""

from __future__ import annotations

from typing import Any


def dura_agent(
    model: str | Any,
    tools: list | None = None,
    **kwargs,
):
    """Create a LangChain agent with Temporal durability.

    Wraps the model with DuraModel and each tool with DuraTool (or
    dura_agent_tool for @dura functions), then delegates to LangChain's
    create_agent().

    Args:
        model: Model name string (e.g., "claude-sonnet-4-6") or BaseChatModel instance.
        tools: List of tools — BaseTool instances, @tool functions, or @dura functions.
        **kwargs: Additional kwargs passed to create_agent() (e.g. system_prompt).

    Returns:
        A LangChain CompiledStateGraph (standard agent).
    """
    from langchain.agents import create_agent
    from langchain_core.tools import BaseTool

    from duralang.agent_tool import dura_agent_tool
    from duralang.dura_model import DuraModel
    from duralang.dura_tool import DuraTool

    # 1. Wrap model
    if isinstance(model, str):
        wrapped_model = DuraModel.from_model_string(model)
    elif isinstance(model, DuraModel):
        wrapped_model = model
    else:
        # Raw BaseChatModel instance — wrap it
        wrapped_model = DuraModel(inner_llm=model)

    # 2. Wrap tools
    wrapped_tools = []
    for tool in tools or []:
        if getattr(tool, "__dura__", False):
            # @dura-decorated function → Child Workflow tool
            wrapped_tools.append(dura_agent_tool(tool))
        elif isinstance(tool, DuraTool):
            # Already wrapped
            wrapped_tools.append(tool)
        elif isinstance(tool, BaseTool):
            # Plain BaseTool → wrap with DuraTool
            wrapped_tools.append(DuraTool(tool))
        elif callable(tool):
            # Plain function → convert to BaseTool first, then wrap
            from langchain_core.tools import tool as tool_decorator

            base_tool = tool_decorator(tool)
            wrapped_tools.append(DuraTool(base_tool))
        else:
            raise TypeError(f"Unsupported tool type: {type(tool).__name__}")

    # 3. Delegate to create_agent
    return create_agent(model=wrapped_model, tools=wrapped_tools, **kwargs)
