"""Experimental Python-backed memory adapters."""

from __future__ import annotations

from typing import List, Optional, Protocol

from .._core import MemoryProvider as _CoreMemoryProvider
from .._core import memory_provider_from_impl
from ..types import ChatMessagePayload


class _SupportsMemory(Protocol):
    def memory(self, mem: _CoreMemoryProvider) -> object: ...


class PythonMemoryProvider:
    """Experimental base class for Python-backed memory providers."""

    def _to_core_provider(self) -> _CoreMemoryProvider:
        return memory_provider_from_impl(self)

    async def remember(self, message: ChatMessagePayload) -> None:
        raise NotImplementedError("PythonMemoryProvider.remember must be implemented")

    async def recall(
        self,
        query: str,
        limit: Optional[int] = None,
    ) -> List[ChatMessagePayload]:
        raise NotImplementedError("PythonMemoryProvider.recall must be implemented")

    async def clear(self) -> None:
        raise NotImplementedError("PythonMemoryProvider.clear must be implemented")

    def size(self) -> int:
        raise NotImplementedError("PythonMemoryProvider.size must be implemented")

    def apply(self, builder: _SupportsMemory) -> None:
        builder.memory(self._to_core_provider())


MemoryProvider = PythonMemoryProvider


def _require_callable_method(provider: object, method: str) -> None:
    attr = getattr(provider, method, None)
    if not callable(attr):
        raise TypeError(f"memory provider must define callable {method}()")


def coerce_memory_provider(provider: object) -> _CoreMemoryProvider:
    if isinstance(provider, _CoreMemoryProvider):
        return provider

    to_core_provider = getattr(provider, "_to_core_provider", None)
    if callable(to_core_provider):
        core_provider = to_core_provider()
        if isinstance(core_provider, _CoreMemoryProvider):
            return core_provider
        raise TypeError("memory provider _to_core_provider() must return a native MemoryProvider")

    if isinstance(provider, PythonMemoryProvider):
        return provider._to_core_provider()

    for method in ("remember", "recall", "clear", "size"):
        _require_callable_method(provider, method)

    return memory_provider_from_impl(provider)


def bind_memory_provider(provider: object) -> _CoreMemoryProvider:
    return coerce_memory_provider(provider)


def apply_memory_provider(builder: _SupportsMemory, provider: object) -> None:
    if isinstance(provider, PythonMemoryProvider):
        provider.apply(builder)
        return

    apply_fn = getattr(provider, "apply", None)
    if callable(apply_fn):
        apply_fn(builder)
        return

    builder.memory(coerce_memory_provider(provider))
