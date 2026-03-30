"""DuraModel — context-aware LLM wrapper that routes through Temporal activities."""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from duralang.config import LLMIdentity
from duralang.context import DuraContext
from duralang.graph_def import LLMActivityPayload, LLMActivityResult
from duralang.registry import ToolRegistry
from duralang.state import MessageSerializer


class DuraModel(BaseChatModel):
    """A BaseChatModel that routes ainvoke() through Temporal when inside @dura."""

    inner_llm: BaseChatModel
    """The actual LLM instance to delegate to."""

    _cached_identity: LLMIdentity | None = None

    @property
    def _llm_type(self) -> str:
        return f"dura-{self.inner_llm._llm_type}"

    def bind_tools(self, tools, **kwargs):
        """Delegate tool formatting to inner LLM, then bind on self."""
        bound = self.inner_llm.bind_tools(tools, **kwargs)
        return self.bind(**bound.kwargs)

    def _get_identity(self) -> LLMIdentity:
        if self._cached_identity is None:
            self._cached_identity = LLMIdentity.from_instance(self.inner_llm)
        return self._cached_identity

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("DuraModel only supports async. Use ainvoke().")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        ctx = DuraContext.get()
        if ctx is None:
            # Outside dura context — passthrough to real LLM
            return await self.inner_llm._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )

        # Inside dura context — route through Temporal activity
        identity = self._get_identity()

        # Get tool schemas from bound tools (cached in registry)
        tool_schemas = []
        for tool in kwargs.get("tools", []):
            if hasattr(tool, "name"):
                cached = ToolRegistry.get_schema(tool.name)
                if cached:
                    tool_schemas.append(cached)
            elif isinstance(tool, dict):
                tool_schemas.append(tool)

        serialized_messages = [MessageSerializer.serialize(m) for m in messages]

        # Filter kwargs to only safe, serializable values
        safe_invoke_kwargs = {}
        for k, v in kwargs.items():
            if k in ("tools", "tool_choice"):
                continue  # Handled via tool_schemas
            if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                safe_invoke_kwargs[k] = v

        payload = LLMActivityPayload(
            messages=serialized_messages,
            llm_identity={
                "provider": identity.provider,
                "model": identity.model,
                "kwargs": identity.kwargs,
            },
            tool_schemas=tool_schemas,
            invoke_kwargs=safe_invoke_kwargs,
        )

        result: LLMActivityResult = await ctx.execute_activity(
            "dura__llm", payload, ctx.config.llm_config
        )

        ai_message = MessageSerializer.deserialize(result.ai_message)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    @classmethod
    def from_model_string(cls, model: str, **kwargs) -> DuraModel:
        """Create DuraModel from a model string like 'claude-sonnet-4-6'."""
        if "claude" in model or "anthropic" in model:
            from langchain_anthropic import ChatAnthropic

            inner = ChatAnthropic(model=model, **kwargs)
        elif "gpt" in model or "o1" in model or "o3" in model:
            from langchain_openai import ChatOpenAI

            inner = ChatOpenAI(model=model, **kwargs)
        elif "gemini" in model:
            from langchain_google_genai import ChatGoogleGenerativeAI

            inner = ChatGoogleGenerativeAI(model=model, **kwargs)
        else:
            from langchain_openai import ChatOpenAI

            inner = ChatOpenAI(model=model, **kwargs)  # Default to OpenAI-compatible
        return cls(inner_llm=inner)
