"""LLM factory — instantiates LangChain chat models from config.

NEVER called in Workflow code — only inside NodeActivity.
API keys are read from environment variables.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from langforge.config import LLMConfig
from langforge.exceptions import ConfigurationError


def build_llm(config: LLMConfig) -> BaseChatModel:
    """Instantiate a LangChain chat model from LLMConfig."""
    if config.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=config.model, **config.kwargs)
    elif config.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=config.model, **config.kwargs)
    elif config.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=config.model, **config.kwargs)
    elif config.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=config.model, **config.kwargs)
    else:
        raise ConfigurationError(f"Unknown LLM provider: '{config.provider}'")
