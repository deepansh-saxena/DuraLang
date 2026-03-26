"""forge__node activity — executes a LangGraph node (LLM call)."""

from __future__ import annotations

import importlib

from temporalio import activity

from langforge.config import LLMConfig
from langforge.graph_def import NodeActivityPayload, NodeActivityResult, ToolCallRequest
from langforge.llm_factory import build_llm
from langforge.state import StateManager


@activity.defn(name="forge__node")
async def node_activity(payload: NodeActivityPayload) -> NodeActivityResult:
    """Execute a single graph node.

    1. Resolve the node callable from its import path
    2. Build the LLM from config
    3. Bind tools to LLM if tool schemas provided
    4. Deserialize state and call the node function
    5. Extract tool calls from the result (if any)
    6. Return state delta + tool call requests
    """
    activity.heartbeat(f"starting node: {payload.node_name}")

    # 1. Resolve node callable
    module_path, fn_name = payload.callable_path.split(":")
    module = importlib.import_module(module_path)
    node_fn = getattr(module, fn_name)

    # 2. Build LLM from config
    llm_config = LLMConfig(**payload.llm_config) if isinstance(payload.llm_config, dict) else payload.llm_config
    llm = build_llm(llm_config)

    # 3. Bind tools to LLM using pre-computed schemas
    if payload.tool_schemas:
        llm = llm.bind_tools(payload.tool_schemas)

    # 4. Deserialize state
    typed_state = StateManager.deserialize_for_node(payload.current_state)

    # 5. Call the node function
    # Support both (state) and (state, llm=llm) signatures
    import inspect

    sig = inspect.signature(node_fn)
    params = list(sig.parameters.keys())

    if len(params) >= 2 or any(p == "llm" for p in params):
        result = await _call_node(node_fn, typed_state, llm=llm)
    else:
        # Inject LLM into state for nodes that expect it there
        typed_state["_llm"] = llm
        result = await _call_node(node_fn, typed_state)

    # 6. Extract tool calls from result
    tool_calls = []
    state_delta = {}

    if isinstance(result, dict):
        messages = result.get("messages", [])
        if not isinstance(messages, list):
            messages = [messages]
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls = [
                    ToolCallRequest(id=tc["id"], name=tc["name"], args=tc["args"])
                    for tc in msg.tool_calls
                ]
        state_delta = StateManager.serialize_delta(result)

    activity.heartbeat(f"node complete: {payload.node_name}, tool_calls={len(tool_calls)}")

    return NodeActivityResult(
        state_delta=state_delta,
        tool_calls=tool_calls,
        is_final=len(tool_calls) == 0,
    )


async def _call_node(node_fn, state, **kwargs):
    """Call a node function, handling both sync and async."""
    import asyncio

    if asyncio.iscoroutinefunction(node_fn):
        return await node_fn(state, **kwargs)
    else:
        return node_fn(state, **kwargs)
