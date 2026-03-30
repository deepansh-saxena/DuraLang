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
    tls_client_cert: str | None = None
    tls_client_key: str | None = None
    tls_root_ca: str | None = None
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


# Safe fields to capture from LLM instances — anything NOT in this set is dropped
_SAFE_LLM_KWARGS = {"temperature", "max_tokens", "top_p", "top_k", "timeout", "max_retries"}
# Fields that must NEVER be serialized into Temporal event history
_SENSITIVE_FIELDS = {
    "api_key", "api_secret", "access_token", "api_token",
    "openai_api_key", "anthropic_api_key",
}


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
        """Inspects a BaseChatModel instance and extracts its identity.

        Uses isinstance() checks (not string matching on type name) so that
        subclasses are correctly identified.
        """
        try:
            from langchain_anthropic import ChatAnthropic
            if isinstance(instance, ChatAnthropic):
                return cls._extract("anthropic", "model", instance)
        except ImportError:
            pass
        try:
            from langchain_openai import ChatOpenAI
            if isinstance(instance, ChatOpenAI):
                return cls._extract("openai", "model_name", instance)
        except ImportError:
            pass
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            if isinstance(instance, ChatGoogleGenerativeAI):
                return cls._extract("google", "model", instance)
        except ImportError:
            pass
        try:
            from langchain_ollama import ChatOllama
            if isinstance(instance, ChatOllama):
                return cls._extract("ollama", "model", instance)
        except ImportError:
            pass

        raise ConfigurationError(
            f"Cannot determine LLM provider from {type(instance).__name__}. "
            f"Supported: ChatAnthropic, ChatOpenAI, ChatGoogleGenerativeAI, ChatOllama."
        )

    @classmethod
    def _extract(cls, provider: str, model_attr: str, instance) -> LLMIdentity:
        kwargs = {}
        for key in _SAFE_LLM_KWARGS:
            val = getattr(instance, key, None)
            if val is not None:
                kwargs[key] = val
        # Defensive: strip any sensitive fields that slipped through
        for f in _SENSITIVE_FIELDS:
            kwargs.pop(f, None)
        return cls(provider=provider, model=getattr(instance, model_attr), kwargs=kwargs)
