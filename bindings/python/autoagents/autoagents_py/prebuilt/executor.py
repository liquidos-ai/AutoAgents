"""Prebuilt executor specs that mirror `autoagents_core::agent::prebuilt`."""

from __future__ import annotations

from .._executor_spec import _ExecutorSpec

_CODEACT_DEFAULT_SANDBOX_LIMITS = {
    "timeout_ms": 10_000,
    "memory_limit_bytes": 32 * 1024 * 1024,
    "max_source_bytes": 64 * 1024,
    "max_console_bytes": 32 * 1024,
    "max_tool_calls_per_execution": 32,
    "max_concurrent_tool_calls": 8,
}


class ReActAgent(_ExecutorSpec):
    _binding_kind = "react"

    def __init__(self, name: str, description: str) -> None:
        super().__init__(name, description)
        self._max_turns = 10

    def max_turns(self, turns: int) -> "ReActAgent":
        if turns <= 0:
            raise ValueError("max_turns must be > 0")
        self._max_turns = turns
        return self

    def __repr__(self) -> str:
        return f"ReActAgent(name={self._name!r}, max_turns={self._max_turns})"


class BasicAgent(_ExecutorSpec):
    _binding_kind = "basic"

    def __init__(self, name: str, description: str) -> None:
        super().__init__(name, description)
        self._max_turns = 1

    def __repr__(self) -> str:
        return f"BasicAgent(name={self._name!r})"


class CodeActAgent(_ExecutorSpec):
    _binding_kind = "codeact"

    def __init__(self, name: str, description: str) -> None:
        super().__init__(name, description)
        self._max_turns = 10
        self._sandbox_limits = dict(_CODEACT_DEFAULT_SANDBOX_LIMITS)

    def max_turns(self, turns: int) -> "CodeActAgent":
        if turns <= 0:
            raise ValueError("max_turns must be > 0")
        self._max_turns = turns
        return self

    def sandbox_limits(
        self,
        *,
        timeout_ms: int | None = None,
        memory_limit_bytes: int | None = None,
        max_source_bytes: int | None = None,
        max_console_bytes: int | None = None,
        max_tool_calls_per_execution: int | None = None,
        max_concurrent_tool_calls: int | None = None,
    ) -> "CodeActAgent":
        updates = {
            "timeout_ms": timeout_ms,
            "memory_limit_bytes": memory_limit_bytes,
            "max_source_bytes": max_source_bytes,
            "max_console_bytes": max_console_bytes,
            "max_tool_calls_per_execution": max_tool_calls_per_execution,
            "max_concurrent_tool_calls": max_concurrent_tool_calls,
        }
        for key, value in updates.items():
            if value is None:
                continue
            if value <= 0:
                raise ValueError(f"{key} must be > 0")
            self._sandbox_limits[key] = int(value)

        if (
            self._sandbox_limits["max_concurrent_tool_calls"]
            > self._sandbox_limits["max_tool_calls_per_execution"]
        ):
            raise ValueError(
                "max_concurrent_tool_calls must be <= max_tool_calls_per_execution"
            )
        return self

    def __repr__(self) -> str:
        return f"CodeActAgent(name={self._name!r}, max_turns={self._max_turns})"
