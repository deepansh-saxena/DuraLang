"""DuraConfig, ActivityConfig, LLMIdentity dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from temporalio.common import RetryPolicy

from duralang.exceptions import ConfigurationError


@dataclass
class ActivityConfig:
    """Per-activity Temporal configuration."""

    start_to_close_timeout: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    heartbeat_timeout: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    retry_policy: RetryPolicy = field(
        default_factory=lambda: RetryPolicy(
            initial_interval=timedelta(seconds=1),
            backoff_coefficient=2.0,
            maximum_attempts=3,
            maximum_interval=timedelta(seconds=30),
            non_retryable_error_types=["ValueError", "TypeError", "KeyError"],
        )
    )


@dataclass
class DuraConfig:
    """Top-level DuraLang configuration."""

    temporal_host: str = "localhost:7233"
    temporal_namespace: str = "default"
    task_queue: str = "duralang"
    max_iterations: int = 50
    child_workflow_timeout: timedelta = field(
        default_factory=lambda: timedelta(hours=1)
    )

    llm_config: ActivityConfig = field(
        default_factory=lambda: ActivityConfig(
            start_to_close_timeout=timedelta(minutes=10),
            heartbeat_timeout=timedelta(minutes=5),  # LLM calls can take 30-120s; generous timeout
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=15),
                backoff_coefficient=2.0,
                maximum_attempts=10,
                maximum_interval=timedelta(minutes=2),
                non_retryable_error_types=["ValueError", "TypeError"],
            ),
        )
    )
    tool_config: ActivityConfig = field(
        default_factory=lambda: ActivityConfig(
            start_to_close_timeout=timedelta(minutes=2),
        )
    )
    mcp_config: ActivityConfig = field(
        default_factory=lambda: ActivityConfig(
            start_to_close_timeout=timedelta(minutes=5),
        )
    )


@dataclass
class LLMIdentity:
    """Serializable identifier for a BaseChatModel instance.

    Extracted from the instance at proxy interception time.
    Used by dura__llm Activity to reconstruct the LLM.
    """

    provider: str  # "anthropic" | "openai" | "google" | "ollama"
    model: str  # e.g. "claude-sonnet-4-6"
    kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_instance(cls, instance) -> LLMIdentity:
        """Inspects a BaseChatModel instance and extracts its identity."""
        instance_type = type(instance).__name__

        if instance_type == "ChatAnthropic":
            return cls(
                provider="anthropic",
                model=instance.model,
                kwargs={"temperature": getattr(instance, "temperature", None)},
            )
        elif instance_type == "ChatOpenAI":
            return cls(
                provider="openai",
                model=getattr(instance, "model_name", instance.model),
                kwargs={"temperature": getattr(instance, "temperature", None)},
            )
        elif instance_type == "ChatGoogleGenerativeAI":
            return cls(
                provider="google",
                model=instance.model,
                kwargs={"temperature": getattr(instance, "temperature", None)},
            )
        elif instance_type == "ChatOllama":
            return cls(
                provider="ollama",
                model=instance.model,
                kwargs={},
            )
        else:
            raise ConfigurationError(
                f"Cannot determine LLM provider from {instance_type}. "
                f"Supported: ChatAnthropic, ChatOpenAI, ChatGoogleGenerativeAI, ChatOllama."
            )
