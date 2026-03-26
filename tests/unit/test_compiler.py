"""Tests for GraphCompiler — LangGraph → ForgeGraphDefinition."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

import pytest
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from langforge.compiler import GraphCompiler
from langforge.exceptions import CompilationError


# --- Test state and node definitions ---

class SimpleState(TypedDict):
    messages: Annotated[list, add_messages]
    counter: int


def node_a(state: SimpleState) -> dict:
    return {"counter": state["counter"] + 1}


def node_b(state: SimpleState) -> dict:
    return {"counter": state["counter"] + 10}


def route_by_counter(state: SimpleState) -> str:
    if state["counter"] > 5:
        return "end"
    return "continue"


class TestGraphCompiler:
    def test_compile_simple_linear_graph(self):
        builder = StateGraph(SimpleState)
        builder.add_node("a", node_a)
        builder.add_node("b", node_b)
        builder.set_entry_point("a")
        builder.add_edge("a", "b")
        builder.add_edge("b", END)
        graph = builder.compile()

        compiler = GraphCompiler()
        graph_def = compiler.compile(graph, "test_graph")

        assert graph_def.graph_id == "test_graph"
        assert graph_def.entry_point == "a"
        assert len(graph_def.nodes) == 2
        node_names = {n.name for n in graph_def.nodes}
        assert node_names == {"a", "b"}

    def test_compile_extracts_edges(self):
        builder = StateGraph(SimpleState)
        builder.add_node("a", node_a)
        builder.add_node("b", node_b)
        builder.set_entry_point("a")
        builder.add_edge("a", "b")
        builder.add_edge("b", END)
        graph = builder.compile()

        compiler = GraphCompiler()
        graph_def = compiler.compile(graph, "test_graph")

        edge_pairs = {(e.source, e.target) for e in graph_def.edges}
        assert ("a", "b") in edge_pairs
        assert ("b", "__end__") in edge_pairs

    def test_compile_conditional_edges(self):
        builder = StateGraph(SimpleState)
        builder.add_node("a", node_a)
        builder.add_node("b", node_b)
        builder.set_entry_point("a")
        builder.add_conditional_edges(
            "a",
            route_by_counter,
            {"continue": "b", "end": END},
        )
        builder.add_edge("b", END)
        graph = builder.compile()

        compiler = GraphCompiler()
        graph_def = compiler.compile(graph, "test_graph")

        assert len(graph_def.conditional_edges) >= 1
        cond = graph_def.conditional_edges[0]
        assert cond.source == "a"
        assert "continue" in cond.path_map
        assert "end" in cond.path_map

    def test_compile_extracts_reducers(self):
        builder = StateGraph(SimpleState)
        builder.add_node("a", node_a)
        builder.set_entry_point("a")
        builder.add_edge("a", END)
        graph = builder.compile()

        compiler = GraphCompiler()
        graph_def = compiler.compile(graph, "test_graph")

        assert "messages" in graph_def.reducer_paths
        assert "add_messages" in graph_def.reducer_paths["messages"]

    def test_compile_extracts_schema(self):
        builder = StateGraph(SimpleState)
        builder.add_node("a", node_a)
        builder.set_entry_point("a")
        builder.add_edge("a", END)
        graph = builder.compile()

        compiler = GraphCompiler()
        graph_def = compiler.compile(graph, "test_graph")

        assert "messages" in graph_def.state_schema
        assert "counter" in graph_def.state_schema

    def test_compile_callable_paths_importable(self):
        builder = StateGraph(SimpleState)
        builder.add_node("a", node_a)
        builder.set_entry_point("a")
        builder.add_edge("a", END)
        graph = builder.compile()

        compiler = GraphCompiler()
        graph_def = compiler.compile(graph, "test_graph")

        node_def = graph_def.nodes[0]
        assert ":" in node_def.callable_path
        module_path, fn_name = node_def.callable_path.split(":")
        assert fn_name == "node_a"

    def test_compile_rejects_lambda(self):
        builder = StateGraph(SimpleState)
        builder.add_node("bad", lambda state: {"counter": 0})
        builder.set_entry_point("bad")
        builder.add_edge("bad", END)
        graph = builder.compile()

        compiler = GraphCompiler()
        with pytest.raises(CompilationError, match="lambda"):
            compiler.compile(graph, "test_graph")

    def test_compile_empty_tool_maps(self):
        builder = StateGraph(SimpleState)
        builder.add_node("a", node_a)
        builder.set_entry_point("a")
        builder.add_edge("a", END)
        graph = builder.compile()

        compiler = GraphCompiler()
        graph_def = compiler.compile(graph, "test_graph")

        assert graph_def.node_tools == {}
        assert graph_def.node_mcp_servers == {}
