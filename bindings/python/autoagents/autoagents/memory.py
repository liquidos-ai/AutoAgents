"""Stable Rust-backed memory wiring for AutoAgents bindings."""

from __future__ import annotations

from typing import Protocol

from ._core import MemoryProvider as _CoreMemoryProvider
from .exceptions import AgentConfigError, ExperimentalFeatureError


class _SupportsMemory(Protocol):
    def memory(self, mem: _CoreMemoryProvider) -> object: ...


def coerce_memory_provider(provider: object) -> _CoreMemoryProvider:
    if isinstance(provider, _CoreMemoryProvider):
        return provider

    to_core_provider = getattr(provider, "_to_core_provider", None)
    if callable(to_core_provider):
        core_provider = to_core_provider()
        if isinstance(core_provider, _CoreMemoryProvider):
            return core_provider
        raise AgentConfigError(
            "stable memory providers must return a native Rust MemoryProvider"
        )

    raise ExperimentalFeatureError(
        "Python-backed memory adapters are experimental; "
        "use autoagents.experimental.bind_memory_provider(...) explicitly"
    )


def apply_memory_provider(builder: _SupportsMemory, provider: object) -> None:
    builder.memory(coerce_memory_provider(provider))
