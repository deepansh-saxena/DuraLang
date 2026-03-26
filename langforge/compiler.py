"""GraphCompiler — compiles LangGraph StateGraph into ForgeGraphDefinition."""

from __future__ import annotations

import typing

from langforge.exceptions import CompilationError
from langforge.graph_def import (
    ForgeConditionalEdgeDef,
    ForgeEdgeDef,
    ForgeGraphDefinition,
    ForgeNodeDef,
)


class GraphCompiler:
    """Walks a compiled LangGraph StateGraph and produces a ForgeGraphDefinition."""

    def compile(self, graph, graph_id: str) -> ForgeGraphDefinition:
        """Compile a LangGraph CompiledStateGraph into a ForgeGraphDefinition.

        Args:
            graph: A CompiledStateGraph (result of StateGraph.compile()).
            graph_id: Unique identifier for this graph.

        Returns:
            ForgeGraphDefinition — fully serializable.
        """
        builder = graph.builder

        # --- Nodes ---
        nodes = []
        for name, runnable in builder.nodes.items():
            if name in ("__start__", "__end__"):
                continue
            callable_path = self._extract_callable_path(runnable, name)
            nodes.append(ForgeNodeDef(name=name, callable_path=callable_path))

        # --- Static edges ---
        edges = [
            ForgeEdgeDef(source=src, target=tgt)
            for src, tgt in builder.edges
        ]

        # --- Conditional edges ---
        conditional_edges = []
        for source_node, branch_dict in builder.branches.items():
            for branch_name, branch in branch_dict.items():
                routing_path = self._extract_callable_path(branch.condition, branch_name)
                conditional_edges.append(
                    ForgeConditionalEdgeDef(
                        source=source_node,
                        routing_callable_path=routing_path,
                        path_map=branch.ends or {},
                    )
                )

        # --- Entry point ---
        entry_point = self._resolve_entry_point(builder, edges)

        # --- State schema ---
        schema = self._extract_schema(builder.schema)

        # --- Reducers ---
        reducer_paths = self._extract_reducer_paths(builder.schema)

        return ForgeGraphDefinition(
            graph_id=graph_id,
            entry_point=entry_point,
            nodes=nodes,
            edges=edges,
            conditional_edges=conditional_edges,
            state_schema=schema,
            reducer_paths=reducer_paths,
            node_tools={},
            node_mcp_servers={},
        )

    def _resolve_entry_point(self, builder, edges: list[ForgeEdgeDef]) -> str:
        """Resolve the entry point node name.

        LangGraph stores the entry point as an edge from __start__ to the first node.
        """
        # Try builder.entry_point first (if it exists as a direct attribute)
        entry = getattr(builder, "_entry_point", None) or getattr(builder, "entry_point", None)
        if entry and entry not in ("__start__", "__end__"):
            return entry

        # Fall back to finding the edge from __start__
        for edge in edges:
            if edge.source == "__start__":
                return edge.target

        raise CompilationError("Could not determine graph entry point — no edge from __start__")

    def _extract_callable_path(self, obj, fallback_name: str) -> str:
        """Extract importable module path from a callable.

        Returns "module:qualname" format, e.g. "my_agent.nodes:agent_node".

        Raises CompilationError if:
        - obj is a lambda
        - obj is a closure with free variables
        - obj is not importable
        """
        # Unwrap RunnableLambda to get the underlying fn
        fn = getattr(obj, "func", obj)
        fn = getattr(fn, "__wrapped__", fn)

        # Some LangGraph runnables wrap further
        if hasattr(fn, "afunc"):
            fn = fn.afunc
        elif hasattr(fn, "func"):
            fn = fn.func

        if not callable(fn):
            raise CompilationError(f"Node '{fallback_name}' is not callable")

        qualname = getattr(fn, "__qualname__", "")
        if "<lambda>" in qualname:
            raise CompilationError(
                f"Node '{fallback_name}' is a lambda — use a named function"
            )
        if getattr(fn, "__closure__", None):
            raise CompilationError(
                f"Node '{fallback_name}' is a closure — use a top-level function"
            )

        module = getattr(fn, "__module__", None)
        if not module:
            raise CompilationError(
                f"Node '{fallback_name}' has no __module__ — cannot determine import path"
            )

        return f"{module}:{qualname}"

    def _extract_schema(self, schema_class) -> dict:
        """Returns field names → type strings from a TypedDict."""
        hints = {}
        if schema_class is None:
            return hints
        for k, v in getattr(schema_class, "__annotations__", {}).items():
            hints[k] = str(v)
        return hints

    def _extract_reducer_paths(self, schema_class) -> dict[str, str]:
        """Inspects Annotated fields for reducer functions.

        e.g. messages: Annotated[list, add_messages]
        Returns {"messages": "langgraph.graph.message:add_messages"}
        """
        if schema_class is None:
            return {}

        reducer_paths = {}
        try:
            hints = typing.get_type_hints(schema_class, include_extras=True)
        except Exception:
            return reducer_paths

        for key, hint in hints.items():
            if typing.get_origin(hint) is typing.Annotated:
                args = typing.get_args(hint)
                for meta in args[1:]:
                    if callable(meta) and hasattr(meta, "__module__"):
                        path = f"{meta.__module__}:{meta.__qualname__}"
                        reducer_paths[key] = path
        return reducer_paths
