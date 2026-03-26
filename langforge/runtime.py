"""ForgeRuntime — main entry point for LangForge."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from langforge.activities import mcp_activity, node_activity, tool_activity
from langforge.compiler import GraphCompiler
from langforge.config import ForgeConfig, LLMConfig
from langforge.exceptions import ConfigurationError
from langforge.graph_def import ForgeGraphDefinition, WorkflowPayload, WorkflowResult
from langforge.registry import MCPSessionRegistry, ToolRegistry
from langforge.state import StateManager
from langforge.workflow import LangForgeWorkflow

if TYPE_CHECKING:
    from temporalio.client import Client, WorkflowHandle
    from temporalio.worker import Worker


class ForgeRuntime:
    """Main runtime for LangForge — manages Temporal client, worker, and graph execution."""

    def __init__(self, config: ForgeConfig | None = None) -> None:
        self.config = config or ForgeConfig()
        self._client: Client | None = None
        self._worker: Worker | None = None
        self._graph_defs: dict[str, ForgeGraphDefinition] = {}

    async def __aenter__(self) -> ForgeRuntime:
        from temporalio.client import Client
        from temporalio.worker import Worker

        self._client = await Client.connect(
            self.config.temporal_host,
            namespace=self.config.temporal_namespace,
        )
        self._worker = Worker(
            self._client,
            task_queue=self.config.task_queue,
            workflows=[LangForgeWorkflow],
            activities=[node_activity, tool_activity, mcp_activity],
        )
        await self._worker.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._worker:
            await self._worker.__aexit__(*args)

    def register_graph(
        self,
        graph: Any,
        graph_id: str | None = None,
    ) -> str:
        """Compile and register a LangGraph CompiledStateGraph.

        Args:
            graph: A CompiledStateGraph (result of StateGraph.compile()).
            graph_id: Optional unique identifier. Auto-generated if not provided.

        Returns:
            The graph_id.

        Raises:
            ConfigurationError: If graph has a checkpointer set.
        """
        # Check no checkpointer
        if getattr(graph, "checkpointer", None) is not None:
            raise ConfigurationError(
                "LangForge replaces LangGraph's checkpointer. "
                "Compile your graph with checkpointer=None."
            )

        gid = graph_id or f"graph_{len(self._graph_defs)}"
        compiler = GraphCompiler()
        graph_def = compiler.compile(graph, gid)
        self._graph_defs[gid] = graph_def
        return gid

    def register_tools(
        self,
        tools: list,
        node_names: list[str] | None = None,
    ) -> None:
        """Register LangChain tools. Call before run().

        Args:
            tools: List of LangChain BaseTool instances.
            node_names: Optional list of node names these tools are available to.
                       If None, tools are available to all nodes.
        """
        for tool in tools:
            ToolRegistry.register(tool)

        # Update tool schemas in all graph defs
        tool_names = [t.name for t in tools]
        tool_schemas = ToolRegistry.get_schemas(tool_names)

        for graph_def in self._graph_defs.values():
            if node_names is None:
                # Make tools available to all nodes
                for node in graph_def.nodes:
                    existing = graph_def.node_tools.get(node.name, [])
                    graph_def.node_tools[node.name] = existing + tool_names
                    existing_schemas = graph_def.node_tool_schemas.get(node.name, [])
                    graph_def.node_tool_schemas[node.name] = existing_schemas + tool_schemas
            else:
                for name in node_names:
                    existing = graph_def.node_tools.get(name, [])
                    graph_def.node_tools[name] = existing + tool_names
                    existing_schemas = graph_def.node_tool_schemas.get(name, [])
                    graph_def.node_tool_schemas[name] = existing_schemas + tool_schemas

    async def register_mcp_session(
        self,
        session: Any,
        server_name: str,
        node_names: list[str] | None = None,
    ) -> None:
        """Register an MCP session and fetch its tool list.

        Args:
            session: An MCP ClientSession.
            server_name: Name for this MCP server.
            node_names: Optional list of node names these tools are available to.
        """
        MCPSessionRegistry.register(server_name, session)

        # Fetch tool list from MCP session
        tools_result = await session.list_tools()
        mcp_tool_schemas = []
        mcp_tool_names = []

        for tool in tools_result.tools:
            prefixed_name = f"mcp__{server_name}__{tool.name}"
            mcp_tool_names.append(prefixed_name)
            schema = {
                "type": "function",
                "function": {
                    "name": prefixed_name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema if hasattr(tool, "inputSchema") else {},
                },
            }
            mcp_tool_schemas.append(schema)

        MCPSessionRegistry.set_tool_schemas(server_name, mcp_tool_schemas)

        # Update graph defs
        for graph_def in self._graph_defs.values():
            if node_names is None:
                for node in graph_def.nodes:
                    existing = graph_def.node_mcp_servers.get(node.name, [])
                    graph_def.node_mcp_servers[node.name] = existing + [server_name]
                    existing_schemas = graph_def.node_tool_schemas.get(node.name, [])
                    graph_def.node_tool_schemas[node.name] = existing_schemas + mcp_tool_schemas
            else:
                for name in node_names:
                    existing = graph_def.node_mcp_servers.get(name, [])
                    graph_def.node_mcp_servers[name] = existing + [server_name]
                    existing_schemas = graph_def.node_tool_schemas.get(name, [])
                    graph_def.node_tool_schemas[name] = existing_schemas + mcp_tool_schemas

    async def run(
        self,
        graph_id: str,
        initial_state: dict,
        llm_config: LLMConfig,
        workflow_id: str | None = None,
    ) -> dict:
        """Run a graph synchronously — blocks until completion.

        Args:
            graph_id: ID returned from register_graph().
            initial_state: Initial graph state dict.
            llm_config: LLM configuration.
            workflow_id: Optional workflow ID. Auto-generated if not provided.

        Returns:
            Final deserialized graph state.
        """
        if self._client is None:
            raise ConfigurationError("ForgeRuntime not connected — use 'async with'")

        graph_def = self._graph_defs.get(graph_id)
        if graph_def is None:
            raise ConfigurationError(f"Graph '{graph_id}' not registered")

        wf_id = workflow_id or f"langforge-{graph_id}-{uuid.uuid4().hex[:8]}"

        serialized_state = StateManager.serialize(initial_state)

        handle = await self._client.start_workflow(
            LangForgeWorkflow.run,
            WorkflowPayload(
                graph_def=graph_def,
                initial_state=serialized_state,
                llm_config=asdict(llm_config),
                forge_config_dict=self._serialize_config(),
            ),
            id=wf_id,
            task_queue=self.config.task_queue,
        )
        result: WorkflowResult = await handle.result()

        if result.error:
            from langforge.exceptions import WorkflowFailedError

            raise WorkflowFailedError(result.error)

        return StateManager.deserialize(result.final_state, graph_def.state_schema)

    async def start(
        self,
        graph_id: str,
        initial_state: dict,
        llm_config: LLMConfig,
        workflow_id: str | None = None,
    ) -> str:
        """Start a graph execution asynchronously — returns workflow_id immediately."""
        if self._client is None:
            raise ConfigurationError("ForgeRuntime not connected — use 'async with'")

        graph_def = self._graph_defs.get(graph_id)
        if graph_def is None:
            raise ConfigurationError(f"Graph '{graph_id}' not registered")

        wf_id = workflow_id or f"langforge-{graph_id}-{uuid.uuid4().hex[:8]}"

        serialized_state = StateManager.serialize(initial_state)

        await self._client.start_workflow(
            LangForgeWorkflow.run,
            WorkflowPayload(
                graph_def=graph_def,
                initial_state=serialized_state,
                llm_config=asdict(llm_config),
                forge_config_dict=self._serialize_config(),
            ),
            id=wf_id,
            task_queue=self.config.task_queue,
        )
        return wf_id

    async def get_result(self, workflow_id: str) -> dict:
        """Get the result of a previously started workflow."""
        if self._client is None:
            raise ConfigurationError("ForgeRuntime not connected — use 'async with'")

        handle = self._client.get_workflow_handle(workflow_id)
        result: WorkflowResult = await handle.result()
        return result.final_state

    async def send_signal(self, workflow_id: str, signal: str, data: Any) -> None:
        """Send a signal to a running workflow (e.g. human input)."""
        if self._client is None:
            raise ConfigurationError("ForgeRuntime not connected — use 'async with'")

        handle = self._client.get_workflow_handle(workflow_id)
        await handle.signal(signal, data)

    def _serialize_config(self) -> dict:
        """Serialize ForgeConfig to a dict for workflow payload."""
        return {
            "temporal_host": self.config.temporal_host,
            "temporal_namespace": self.config.temporal_namespace,
            "task_queue": self.config.task_queue,
        }
