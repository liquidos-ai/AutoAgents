"""Trait-style Python pipeline layers over the Rust LLM provider pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

from ._core import (
    LLMProvider,
    Tool,
    _coerce_llm_provider,
    pipeline_cache_layer,
    pipeline_python_layer,
    pipeline_retry_layer,
)
from .execution import ExecutionLLM
from .types import ChatMessagePayload, JsonObject, LLMChatResponse


class LayerProtocol(Protocol):
    def build(self, next_provider: LLMProvider) -> LLMProvider: ...


class PipelineLayer:
    """Trait-style base class for Python-authored LLM pipeline layers.

    Override `chat(...)` and optionally `chat_with_tools(...)` /
    `chat_with_web_search(...)`. Other provider capabilities pass through to the
    wrapped Rust provider unchanged.
    """

    def build(self, next_provider: LLMProvider) -> LLMProvider:
        return pipeline_python_layer(_coerce_llm_provider(next_provider), self)

    async def chat(
        self,
        next_provider: ExecutionLLM,
        messages: str | ChatMessagePayload | Sequence[ChatMessagePayload],
        schema: str | JsonObject | None = None,
    ) -> LLMChatResponse:
        return await next_provider.chat(messages, schema)

    async def chat_with_tools(
        self,
        next_provider: ExecutionLLM,
        messages: str | ChatMessagePayload | Sequence[ChatMessagePayload],
        tools: Sequence[Tool],
        schema: str | JsonObject | None = None,
    ) -> LLMChatResponse:
        return await next_provider.chat_with_tools(messages, tools, schema)

    async def chat_with_web_search(
        self,
        next_provider: ExecutionLLM,
        input: str,
    ) -> LLMChatResponse:
        return await next_provider.chat_with_web_search(input)


@dataclass(slots=True)
class CacheLayer(PipelineLayer):
    ttl_seconds: Optional[int] = None

    def build(self, next_provider: LLMProvider) -> LLMProvider:
        return pipeline_cache_layer(_coerce_llm_provider(next_provider), self.ttl_seconds)


@dataclass(slots=True)
class RetryLayer(PipelineLayer):
    max_attempts: Optional[int] = None
    initial_backoff_ms: Optional[int] = None
    max_backoff_ms: Optional[int] = None
    jitter: Optional[bool] = None

    def build(self, next_provider: LLMProvider) -> LLMProvider:
        return pipeline_retry_layer(
            _coerce_llm_provider(next_provider),
            self.max_attempts,
            self.initial_backoff_ms,
            self.max_backoff_ms,
            self.jitter,
        )


class PipelineBuilder:
    """Python builder mirroring Rust's trait-based `PipelineBuilder`."""

    def __init__(self, provider: LLMProvider) -> None:
        self._base = _coerce_llm_provider(provider)
        self._layers: List[LayerProtocol] = []

    def add_layer(self, layer: LayerProtocol) -> "PipelineBuilder":
        self._layers.append(layer)
        return self

    def with_cache(self, ttl_seconds: Optional[int] = None) -> "PipelineBuilder":
        return self.add_layer(CacheLayer(ttl_seconds=ttl_seconds))

    def with_retry(
        self,
        max_attempts: Optional[int] = None,
        initial_backoff_ms: Optional[int] = None,
        max_backoff_ms: Optional[int] = None,
        jitter: Optional[bool] = None,
    ) -> "PipelineBuilder":
        return self.add_layer(
            RetryLayer(
                max_attempts=max_attempts,
                initial_backoff_ms=initial_backoff_ms,
                max_backoff_ms=max_backoff_ms,
                jitter=jitter,
            )
        )

    def build(self) -> LLMProvider:
        provider = self._base
        for layer in reversed(self._layers):
            provider = layer.build(provider)
        return provider


__all__ = ["PipelineBuilder", "PipelineLayer", "CacheLayer", "RetryLayer"]
