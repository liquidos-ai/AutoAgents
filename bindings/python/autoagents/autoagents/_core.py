"""Private bridge to the compiled AutoAgents extension."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable
from typing import TypeVar

from autoagents_py import (
    ActorAgentHandle,
    AgentBuilder,
    AgentHandle,
    Environment,
    EventStream,
    ExecutionJsonStream as _NativeExecutionJsonStream,
    ExecutionLLM as _NativeExecutionLLM,
    ExecutionMemory as _NativeExecutionMemory,
    ExecutionStringStream as _NativeExecutionStringStream,
    LLMBuilder,
    LLMProvider,
    MemoryProvider,
    RunStream,
    Runtime,
    Tool,
    Topic,
    memory_provider_from_impl,
    pipeline_cache_layer,
    pipeline_python_layer,
    pipeline_retry_layer,
    sliding_window_memory,
)

__all__ = [
    "AgentBuilder",
    "AgentHandle",
    "ActorAgentHandle",
    "Environment",
    "EventStream",
    "ExecutionLLM",
    "ExecutionMemory",
    "LLMBuilder",
    "LLMProvider",
    "MemoryProvider",
    "RunStream",
    "Runtime",
    "Tool",
    "Topic",
    "memory_provider_from_impl",
    "pipeline_cache_layer",
    "pipeline_python_layer",
    "pipeline_retry_layer",
    "sliding_window_memory",
]

_T = TypeVar("_T")


async def _await_native(awaitable: Awaitable[_T], *, timeout: float | None = None) -> _T:
    if timeout is None:
        return await awaitable
    return await asyncio.wait_for(awaitable, timeout=timeout)


async def _drive_native(awaitable: Awaitable[_T], *, timeout: float | None = None) -> _T:
    return await _await_native(awaitable, timeout=timeout)


async def _drive_coroutine(awaitable: Awaitable[_T]) -> _T:
    return await awaitable


class _NativeAsyncIterator(AsyncIterator[_T]):
    def __init__(self, inner: object) -> None:
        self._inner = inner

    def __aiter__(self) -> "_NativeAsyncIterator[_T]":
        return self

    async def __anext__(self) -> _T:
        return await _drive_native(self._inner.__anext__())  # type: ignore[attr-defined]


def _coerce_llm_provider(provider: object) -> LLMProvider:
    if isinstance(provider, LLMProvider):
        return provider

    raise TypeError(
        "expected an AutoAgents LLMProvider returned by LLMBuilder.build() "
        "or PipelineBuilder.build()"
    )
