"""LangForgeWorkflow — the core Temporal workflow that drives the ReAct loop.

This is the most critical file. The Workflow drives the entire execution:
- Schedules NodeActivity for LLM calls
- Detects tool calls in NodeActivityResult
- Schedules ToolActivity/MCPActivity in parallel via asyncio.gather
- Injects ToolMessages into state
- Evaluates edges and advances to the next node
- Handles human-in-the-loop via Temporal signals
"""

from __future__ import annotations

import asyncio
import importlib

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from langchain_core.messages import HumanMessage, ToolMessage
    from langgraph.graph.message import add_messages

    from langforge.config import ActivityConfig, ForgeConfig
    from langforge.graph_def import (
        ForgeGraphDefinition,
        ForgeNodeDef,
        MCPActivityPayload,
        MCPActivityResult,
        NodeActivityPayload,
        NodeActivityResult,
        ToolActivityPayload,
        ToolActivityResult,
        ToolCallRequest,
        WorkflowPayload,
        WorkflowResult,
    )
    from langforge.state import StateManager


@workflow.defn(name="LangForgeWorkflow")
class LangForgeWorkflow:
    """Temporal workflow that executes a LangGraph graph as Temporal activities."""

    def __init__(self) -> None:
        self._state: dict = {}
        self._waiting_for_human: bool = False
        self._human_input: str | None = None

    @workflow.signal
    async def human_input(self, value: str) -> None:
        """Signal handler for human-in-the-loop input."""
        self._human_input = value
        self._waiting_for_human = False

    @workflow.run
    async def run(self, payload: WorkflowPayload) -> WorkflowResult:
        """Execute the full graph as a sequence of activities."""
        graph_def = payload.graph_def
        forge_config = self._build_forge_config(payload.forge_config_dict)

        self._state = payload.initial_state
        current_node = graph_def.entry_point
        execution_order: list[str] = []
        total_tool_calls: int = 0

        while current_node != "__end__":
            execution_order.append(current_node)
            node_def = self._get_node_def(graph_def, current_node)
            node_config = forge_config.node_configs.get(
                current_node, forge_config.node_activity_config
            )

            # Build tool schemas for this node's available tools
            tool_schemas = self._build_tool_schemas(graph_def, current_node)

            # ── Step 1: Execute the node (LLM call) ──
            node_result: NodeActivityResult = await workflow.execute_activity(
                "forge__node",
                NodeActivityPayload(
                    node_name=current_node,
                    callable_path=node_def.callable_path,
                    current_state=self._state,
                    llm_config=payload.llm_config,
                    tool_schemas=tool_schemas,
                ),
                start_to_close_timeout=node_config.start_to_close_timeout,
                retry_policy=node_config.retry_policy,
                heartbeat_timeout=node_config.heartbeat_timeout,
            )

            # ── Step 2: Merge node's state delta ──
            self._state = self._apply_reducers(
                graph_def, self._state, node_result.state_delta
            )

            # ── Step 3: Execute tool calls if any ──
            if node_result.tool_calls:
                total_tool_calls += len(node_result.tool_calls)

                # Schedule all tool calls in parallel
                tool_tasks = []
                for tc in node_result.tool_calls:
                    activity_config = self._resolve_tool_config(
                        forge_config, tc.name
                    )
                    task = self._schedule_tool_or_mcp(tc, activity_config)
                    tool_tasks.append(task)

                tool_results = await asyncio.gather(*tool_tasks)

                # Inject ToolMessages into state
                self._state = self._inject_tool_messages(self._state, tool_results)

                # Loop back to the SAME node — agent sees tool results, reasons again
                continue

            # ── Step 4: No tool calls — evaluate edges, advance ──
            current_node = self._evaluate_next_node(
                graph_def, current_node, self._state
            )

            # ── Step 5: Human-in-the-loop pause ──
            if self._waiting_for_human:
                await workflow.wait_condition(lambda: not self._waiting_for_human)
                if self._human_input is not None:
                    self._state = self._inject_human_message(
                        self._state, self._human_input
                    )
                    self._human_input = None

        return WorkflowResult(
            final_state=self._state,
            node_execution_order=execution_order,
            total_tool_calls=total_tool_calls,
        )

    async def _schedule_tool_or_mcp(
        self,
        tc: ToolCallRequest,
        config: ActivityConfig,
    ):
        """Route a tool call to forge__tool or forge__mcp.

        MCP tools are prefixed: "mcp__{server_name}__{tool_name}"
        Regular tools: just the tool name
        """
        if tc.name.startswith("mcp__"):
            parts = tc.name.split("__", 2)
            server_name = parts[1]
            tool_name = parts[2]
            return await workflow.execute_activity(
                "forge__mcp",
                MCPActivityPayload(
                    server_name=server_name,
                    tool_name=tool_name,
                    arguments=tc.args,
                    tool_call_id=tc.id,
                ),
                start_to_close_timeout=config.start_to_close_timeout,
                retry_policy=config.retry_policy,
                heartbeat_timeout=config.heartbeat_timeout,
            )
        else:
            return await workflow.execute_activity(
                "forge__tool",
                ToolActivityPayload(
                    tool_name=tc.name,
                    tool_input=tc.args,
                    tool_call_id=tc.id,
                ),
                start_to_close_timeout=config.start_to_close_timeout,
                retry_policy=config.retry_policy,
                heartbeat_timeout=config.heartbeat_timeout,
            )

    def _apply_reducers(
        self, graph_def: ForgeGraphDefinition, state: dict, delta: dict
    ) -> dict:
        """Merge delta into state using reducer functions.

        For keys with no reducer: last-write-wins.
        For keys with reducer (e.g. add_messages): calls reducer(current, delta).
        MUST be deterministic.
        """
        new_state = dict(state)
        for key, value in delta.items():
            if key in graph_def.reducer_paths:
                module_path, fn_name = graph_def.reducer_paths[key].split(":")
                module = importlib.import_module(module_path)
                reducer_fn = getattr(module, fn_name)
                new_state[key] = reducer_fn(new_state.get(key, []), value)
            else:
                new_state[key] = value
        return new_state

    def _inject_tool_messages(self, state: dict, tool_results: list) -> dict:
        """Build ToolMessage objects from tool/MCP results and append to state.

        CRITICAL: tool_call_id must match the original ToolCallRequest.id.
        """
        tool_messages = []
        for result in tool_results:
            if isinstance(result, ToolActivityResult):
                content = result.error if result.error else result.output
            elif isinstance(result, MCPActivityResult):
                content = (
                    f"Error: {result.content}" if result.is_error else str(result.content)
                )
            else:
                content = str(result)

            tool_messages.append(
                ToolMessage(content=content, tool_call_id=result.tool_call_id)
            )

        serialized = [StateManager.serialize_message(m) for m in tool_messages]
        new_state = dict(state)
        current_msgs = new_state.get("messages", [])
        new_state["messages"] = add_messages(current_msgs, serialized)
        return new_state

    def _evaluate_next_node(
        self, graph_def: ForgeGraphDefinition, current_node: str, state: dict
    ) -> str:
        """Evaluate edges to determine the next node.

        Checks static edges first, then conditional edges.
        MUST be deterministic — no I/O, no LLM calls.
        """
        # Check static edges
        for edge in graph_def.edges:
            if edge.source == current_node:
                return edge.target

        # Check conditional edges
        for cond_edge in graph_def.conditional_edges:
            if cond_edge.source == current_node:
                module_path, fn_name = cond_edge.routing_callable_path.split(":")
                module = importlib.import_module(module_path)
                routing_fn = getattr(module, fn_name)
                # Deserialize state for routing fn
                typed_state = StateManager.deserialize(state, graph_def.state_schema)
                decision = routing_fn(typed_state)
                return cond_edge.path_map.get(decision, decision)

        return "__end__"

    def _get_node_def(
        self, graph_def: ForgeGraphDefinition, name: str
    ) -> ForgeNodeDef:
        for node in graph_def.nodes:
            if node.name == name:
                return node
        raise ValueError(f"Node '{name}' not found in graph definition")

    def _build_tool_schemas(
        self, graph_def: ForgeGraphDefinition, node_name: str
    ) -> list[dict]:
        """Return JSON schemas for tools available to this node."""
        return graph_def.node_tool_schemas.get(node_name, [])

    def _resolve_tool_config(
        self, forge_config: ForgeConfig, tool_name: str
    ) -> ActivityConfig:
        """Per-tool config override, falls back to MCP or tool default."""
        if tool_name.startswith("mcp__"):
            parts = tool_name.split("__", 2)
            server_name = parts[1]
            return forge_config.mcp_configs.get(
                server_name, forge_config.mcp_activity_config
            )
        return forge_config.tool_configs.get(
            tool_name, forge_config.tool_activity_config
        )

    def _inject_human_message(self, state: dict, content: str) -> dict:
        msg = StateManager.serialize_message(HumanMessage(content=content))
        new_state = dict(state)
        new_state["messages"] = add_messages(new_state.get("messages", []), [msg])
        return new_state

    def _build_forge_config(self, config_dict: dict) -> ForgeConfig:
        """Reconstruct ForgeConfig from serialized dict."""
        # Simple reconstruction — ActivityConfig fields are already basic types
        # For now, use defaults and override what's present
        config = ForgeConfig()
        for key in ("temporal_host", "temporal_namespace", "task_queue"):
            if key in config_dict:
                setattr(config, key, config_dict[key])
        return config
