"""Prebuilt executor specs that mirror `autoagents_core::agent::prebuilt`."""

from __future__ import annotations

from .._executor_spec import _ExecutorSpec


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
