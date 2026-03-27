"""dura__llm activity — LLM inference call.

Reconstructs the LLM from LLMIdentity, rebinds tools, invokes, returns serialized AIMessage.
"""

from __future__ import annotations

from temporalio import activity

from duralang.config import LLMIdentity
from duralang.exceptions import ConfigurationError
from duralang.graph_def import LLMActivityPayload, LLMActivityResult
from duralang.state import MessageSerializer


def _normalize_content(content) -> str:
    """Convert LLM response content to a plain string."""
    if isinstance(content, list):
        content = " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return content or ""


def build_llm_from_identity(identity: LLMIdentity):
    """Instantiate a LangChain chat model from LLMIdentity."""
    kwargs = {k: v for k, v in identity.kwargs.items() if v is not None}

    if identity.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=identity.model, **kwargs)
    elif identity.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=identity.model, **kwargs)
    elif identity.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=identity.model, **kwargs)
    elif identity.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=identity.model, **kwargs)
    else:
        raise ConfigurationError(f"Unknown provider: '{identity.provider}'")


@activity.defn(name="dura__llm")
async def llm_activity(payload: LLMActivityPayload) -> LLMActivityResult:
    """Execute a single LLM inference call."""
    activity.heartbeat("llm: starting inference")

    # 1. Reconstruct LLM from identity
    identity = (
        LLMIdentity(**payload.llm_identity)
        if isinstance(payload.llm_identity, dict)
        else payload.llm_identity
    )
    llm = build_llm_from_identity(identity)

    # 2. Rebind tools if any were bound
    if payload.tool_schemas:
        llm = llm.bind_tools(payload.tool_schemas)

    # 3. Deserialize messages
    messages = MessageSerializer.deserialize_many(payload.messages)

    # 4. Invoke LLM
    response = await llm.ainvoke(messages, **payload.invoke_kwargs)

    activity.heartbeat("llm: inference complete")

    return LLMActivityResult(
        ai_message=MessageSerializer.serialize(response),
        content=_normalize_content(response.content),
    )
