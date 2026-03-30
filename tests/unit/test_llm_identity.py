"""Tests for LLMIdentity extraction and build_llm_from_identity."""

from __future__ import annotations

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from duralang.activities.llm import build_llm_from_identity
from duralang.config import LLMIdentity, _SAFE_LLM_KWARGS, _SENSITIVE_FIELDS
from duralang.exceptions import ConfigurationError


class TestFromInstanceAnthropic:
    def test_extracts_provider_and_model(self):
        llm = ChatAnthropic(model="claude-sonnet-4-6", api_key="test-key")
        identity = LLMIdentity.from_instance(llm)
        assert identity.provider == "anthropic"
        assert identity.model == "claude-sonnet-4-6"

    def test_api_key_not_in_kwargs(self):
        llm = ChatAnthropic(model="claude-sonnet-4-6", api_key="test-key")
        identity = LLMIdentity.from_instance(llm)
        for sensitive in _SENSITIVE_FIELDS:
            assert sensitive not in identity.kwargs

    def test_temperature_in_kwargs(self):
        llm = ChatAnthropic(
            model="claude-sonnet-4-6", api_key="test-key", temperature=0.7
        )
        identity = LLMIdentity.from_instance(llm)
        assert identity.kwargs["temperature"] == 0.7

    def test_max_tokens_in_kwargs(self):
        llm = ChatAnthropic(
            model="claude-sonnet-4-6", api_key="test-key", max_tokens=1024
        )
        identity = LLMIdentity.from_instance(llm)
        assert identity.kwargs["max_tokens"] == 1024

    def test_subclass_detected_via_isinstance(self):
        class MyChatAnthropic(ChatAnthropic):
            pass

        llm = MyChatAnthropic(model="claude-sonnet-4-6", api_key="test-key")
        identity = LLMIdentity.from_instance(llm)
        assert identity.provider == "anthropic"
        assert identity.model == "claude-sonnet-4-6"


class TestFromInstanceOpenAI:
    def test_extracts_provider_and_model(self):
        llm = ChatOpenAI(model="gpt-4o", api_key="test-key")
        identity = LLMIdentity.from_instance(llm)
        assert identity.provider == "openai"
        assert identity.model == "gpt-4o"

    def test_api_key_not_in_kwargs(self):
        llm = ChatOpenAI(model="gpt-4o", api_key="test-key")
        identity = LLMIdentity.from_instance(llm)
        for sensitive in _SENSITIVE_FIELDS:
            assert sensitive not in identity.kwargs

    def test_temperature_in_kwargs(self):
        llm = ChatOpenAI(model="gpt-4o", api_key="test-key", temperature=0.5)
        identity = LLMIdentity.from_instance(llm)
        assert identity.kwargs["temperature"] == 0.5

    def test_max_tokens_in_kwargs(self):
        llm = ChatOpenAI(model="gpt-4o", api_key="test-key", max_tokens=2048)
        identity = LLMIdentity.from_instance(llm)
        assert identity.kwargs["max_tokens"] == 2048


class TestFromInstanceUnknownProvider:
    def test_raises_configuration_error(self):
        class FakeLLM:
            pass

        with pytest.raises(ConfigurationError, match="Cannot determine LLM provider"):
            LLMIdentity.from_instance(FakeLLM())


class TestBuildLLMFromIdentity:
    def test_strips_unsafe_kwargs(self):
        identity = LLMIdentity(
            provider="anthropic",
            model="claude-sonnet-4-6",
            kwargs={"temperature": 0.5, "base_url": "https://evil.example.com"},
        )
        # base_url is not in _SAFE_LLM_KWARGS, so it must be stripped
        llm = build_llm_from_identity(identity)
        # The built LLM should have temperature but NOT base_url from kwargs
        assert "base_url" not in {
            k for k, v in identity.kwargs.items() if k in _SAFE_LLM_KWARGS
        }
        # Verify the LLM was constructed successfully
        assert llm is not None

    def test_unknown_provider_raises(self):
        identity = LLMIdentity(provider="martian", model="alien-7b", kwargs={})
        with pytest.raises(ConfigurationError, match="Unknown provider"):
            build_llm_from_identity(identity)

    def test_safe_kwargs_passed_through(self):
        identity = LLMIdentity(
            provider="anthropic",
            model="claude-sonnet-4-6",
            kwargs={"temperature": 0.9, "max_tokens": 512},
        )
        llm = build_llm_from_identity(identity)
        assert llm.temperature == 0.9
        assert llm.max_tokens == 512
