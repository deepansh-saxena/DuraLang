"""Configuration dataclasses for LangForge."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from temporalio.common import RetryPolicy


@dataclass
class LLMConfig:
    """LLM configuration — serializable across activity boundaries.

    API keys are NEVER stored here — read from environment variables at activity time.
    """

    provider: str  # "anthropic" | "openai" | "google" | "ollama"
    model: str  # e.g. "claude-sonnet-4-6"
    kwargs: dict = field(default_factory=dict)


@dataclass
class ActivityConfig:
    """Temporal activity configuration with retry policy."""

    start_to_close_timeout: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    schedule_to_close_timeout: timedelta | None = None
    heartbeat_timeout: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    retry_policy: RetryPolicy = field(
        default_factory=lambda: RetryPolicy(
            initial_interval=timedelta(seconds=1),
            backoff_coefficient=2.0,
            maximum_attempts=3,
            maximum_interval=timedelta(seconds=30),
            non_retryable_error_types=[
                "ValueError",
                "TypeError",
                "KeyError",
                "StateSerializationError",
                "ToolException",
            ],
        )
    )


@dataclass
class ForgeConfig:
    """Top-level LangForge configuration."""

    temporal_host: str = "localhost:7233"
    temporal_namespace: str = "default"
    task_queue: str = "langforge"

    node_activity_config: ActivityConfig = field(
        default_factory=lambda: ActivityConfig(
            start_to_close_timeout=timedelta(minutes=10),
            heartbeat_timeout=timedelta(seconds=60),
        )
    )
    tool_activity_config: ActivityConfig = field(
        default_factory=lambda: ActivityConfig(
            start_to_close_timeout=timedelta(minutes=2),
        )
    )
    mcp_activity_config: ActivityConfig = field(
        default_factory=lambda: ActivityConfig(
            start_to_close_timeout=timedelta(minutes=5),
            heartbeat_timeout=timedelta(seconds=30),
        )
    )

    # Per-node overrides keyed by node name
    node_configs: dict[str, ActivityConfig] = field(default_factory=dict)
    # Per-tool overrides keyed by tool name
    tool_configs: dict[str, ActivityConfig] = field(default_factory=dict)
    # Per-MCP-server overrides keyed by server name
    mcp_configs: dict[str, ActivityConfig] = field(default_factory=dict)
