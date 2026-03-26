"""Tests for LangForgeWorkflow — edge evaluation, reducer application, tool message injection."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langforge.graph_def import (
    ForgeConditionalEdgeDef,
    ForgeEdgeDef,
    ForgeGraphDefinition,
    ForgeNodeDef,
    ToolActivityResult,
    ToolCallRequest,
)
from langforge.state import StateManager
from langforge.workflow import LangForgeWorkflow


def _make_graph_def(
    nodes=None,
    edges=None,
    conditional_edges=None,
    entry_point="agent",
    reducer_paths=None,
) -> ForgeGraphDefinition:
    return ForgeGraphDefinition(
        graph_id="test",
        entry_point=entry_point,
        nodes=nodes or [ForgeNodeDef(name="agent", callable_path="test:agent")],
        edges=edges or [],
        conditional_edges=conditional_edges or [],
        state_schema={"messages": "list", "counter": "int"},
        reducer_paths=reducer_paths or {"messages": "langgraph.graph.message:add_messages"},
        node_tools={},
        node_mcp_servers={},
        node_tool_schemas={},
    )


class TestEvaluateNextNode:
    def test_static_edge(self):
        wf = LangForgeWorkflow()
        graph_def = _make_graph_def(
            nodes=[
                ForgeNodeDef(name="a", callable_path="test:a"),
                ForgeNodeDef(name="b", callable_path="test:b"),
            ],
            edges=[
                ForgeEdgeDef(source="a", target="b"),
                ForgeEdgeDef(source="b", target="__end__"),
            ],
        )

        assert wf._evaluate_next_node(graph_def, "a", {}) == "b"
        assert wf._evaluate_next_node(graph_def, "b", {}) == "__end__"

    def test_no_edges_returns_end(self):
        wf = LangForgeWorkflow()
        graph_def = _make_graph_def()
        assert wf._evaluate_next_node(graph_def, "agent", {}) == "__end__"


class TestApplyReducers:
    def test_last_write_wins_no_reducer(self):
        wf = LangForgeWorkflow()
        graph_def = _make_graph_def(reducer_paths={})
        state = {"counter": 5}
        delta = {"counter": 10}
        result = wf._apply_reducers(graph_def, state, delta)
        assert result["counter"] == 10

    def test_add_messages_reducer(self):
        wf = LangForgeWorkflow()
        graph_def = _make_graph_def()

        state = {
            "messages": [
                StateManager.serialize_message(HumanMessage(content="hi"))
            ]
        }
        delta = {
            "messages": [
                StateManager.serialize_message(AIMessage(content="hello"))
            ]
        }

        result = wf._apply_reducers(graph_def, state, delta)
        assert len(result["messages"]) == 2


class TestInjectToolMessages:
    def test_inject_tool_results(self):
        wf = LangForgeWorkflow()
        state = {
            "messages": [
                StateManager.serialize_message(HumanMessage(content="hi")),
                StateManager.serialize_message(
                    AIMessage(
                        content="",
                        tool_calls=[{"id": "tc_1", "name": "search", "args": {}}],
                    )
                ),
            ]
        }

        tool_results = [
            ToolActivityResult(output="result data", tool_call_id="tc_1"),
        ]

        new_state = wf._inject_tool_messages(state, tool_results)
        messages = new_state["messages"]
        assert len(messages) == 3
        # Last message should be the tool result
        last = messages[-1]
        if isinstance(last, dict):
            assert last.get("tool_call_id") == "tc_1" or last.get("type") == "ToolMessage"
        else:
            assert last.tool_call_id == "tc_1"


class TestBuildToolSchemas:
    def test_returns_schemas_for_node(self):
        wf = LangForgeWorkflow()
        graph_def = _make_graph_def()
        graph_def.node_tool_schemas["agent"] = [
            {"type": "function", "function": {"name": "search"}}
        ]
        schemas = wf._build_tool_schemas(graph_def, "agent")
        assert len(schemas) == 1

    def test_returns_empty_for_unknown_node(self):
        wf = LangForgeWorkflow()
        graph_def = _make_graph_def()
        schemas = wf._build_tool_schemas(graph_def, "nonexistent")
        assert schemas == []
