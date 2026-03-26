"""Graph definition dataclasses — fully serializable, produced by GraphCompiler."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ForgeNodeDef:
    """A single node in the graph definition."""

    name: str
    callable_path: str  # importable path: "my_agent.nodes:agent_node"


@dataclass
class ForgeEdgeDef:
    """A static edge between two nodes."""

    source: str
    target: str  # node name or "__end__"


@dataclass
class ForgeConditionalEdgeDef:
    """A conditional edge with routing function and path map."""

    source: str
    routing_callable_path: str  # importable path to routing function
    path_map: dict[str, str]  # return value → node name or "__end__"


@dataclass
class ForgeGraphDefinition:
    """Complete serializable graph definition produced by GraphCompiler."""

    graph_id: str
    entry_point: str
    nodes: list[ForgeNodeDef]
    edges: list[ForgeEdgeDef]
    conditional_edges: list[ForgeConditionalEdgeDef]
    state_schema: dict  # TypedDict field names → type strings
    reducer_paths: dict[str, str]  # state key → importable path to reducer fn

    # Tool names and MCP server names available to each node
    node_tools: dict[str, list[str]] = field(default_factory=dict)
    node_mcp_servers: dict[str, list[str]] = field(default_factory=dict)

    # Pre-computed tool schemas per node (populated by ForgeRuntime at register time)
    node_tool_schemas: dict[str, list[dict]] = field(default_factory=dict)


# --- Activity payloads and results ---


@dataclass
class ToolCallRequest:
    """A tool call requested by the LLM, extracted from NodeActivityResult."""

    id: str  # LangChain tool_call_id — MUST be preserved for ToolMessage injection
    name: str  # tool name — used to route to forge__tool or forge__mcp
    args: dict  # arguments to pass to the tool


@dataclass
class NodeActivityPayload:
    """Payload sent to forge__node activity."""

    node_name: str
    callable_path: str
    current_state: dict  # full serialized state
    llm_config: dict  # serialized LLMConfig
    tool_schemas: list[dict]  # JSON schemas for available tools


@dataclass
class NodeActivityResult:
    """Result returned from forge__node activity."""

    state_delta: dict  # state changes (excluding tool results)
    tool_calls: list[ToolCallRequest]  # tool calls the LLM wants to make
    is_final: bool  # True if no tool calls and node is done reasoning


@dataclass
class ToolActivityPayload:
    """Payload sent to forge__tool activity."""

    tool_name: str
    tool_input: dict | str
    tool_call_id: str  # preserved from ToolCallRequest.id


@dataclass
class ToolActivityResult:
    """Result returned from forge__tool activity."""

    output: str
    tool_call_id: str  # echoed back for ToolMessage construction
    error: str | None = None


@dataclass
class MCPActivityPayload:
    """Payload sent to forge__mcp activity."""

    server_name: str
    tool_name: str
    arguments: dict
    tool_call_id: str  # preserved from ToolCallRequest.id


@dataclass
class MCPActivityResult:
    """Result returned from forge__mcp activity."""

    content: list[dict]  # raw MCP result content blocks
    tool_call_id: str  # echoed back for ToolMessage construction
    is_error: bool = False


@dataclass
class WorkflowPayload:
    """Payload sent to start LangForgeWorkflow."""

    graph_def: ForgeGraphDefinition
    initial_state: dict  # JSON-serializable initial graph state
    llm_config: dict  # serialized LLMConfig
    forge_config_dict: dict  # serialized ForgeConfig


@dataclass
class WorkflowResult:
    """Result returned from LangForgeWorkflow."""

    final_state: dict
    node_execution_order: list[str]
    total_tool_calls: int
    error: str | None = None
