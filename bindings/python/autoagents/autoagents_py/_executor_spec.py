"""Internal executor spec primitives for the public Python API."""

from __future__ import annotations

import json
from typing import Sequence, TypeVar

from .tool import Tool

_ExecutorT = TypeVar("_ExecutorT", bound="_ExecutorSpec")
_HOOK_METHODS = (
    "on_agent_create",
    "on_run_start",
    "on_run_complete",
    "on_turn_start",
    "on_turn_complete",
    "on_tool_call",
    "on_tool_start",
    "on_tool_result",
    "on_tool_error",
    "on_agent_shutdown",
)


def validate_schema_json(schema_json: str) -> str:
    json.loads(schema_json)
    return schema_json


class _ExecutorSpec:
    _binding_kind: str

    def __init__(self, name: str, description: str) -> None:
        self._name = name
        self._description = description
        self._tools: list[Tool] = []
        self._output_schema_json: str | None = None
        self._hooks: object | None = None
        self._max_turns = 1

    def tools(self: _ExecutorT, tools: Sequence[Tool]) -> _ExecutorT:
        self._tools = list(tools)
        return self

    def output_schema(self: _ExecutorT, schema_json: str) -> _ExecutorT:
        self._output_schema_json = validate_schema_json(schema_json)
        return self

    def hooks(self: _ExecutorT, hooks: object) -> _ExecutorT:
        for method in _HOOK_METHODS:
            if hasattr(hooks, method) and not callable(getattr(hooks, method)):
                raise TypeError(f"hooks.{method} must be callable when provided")
        self._hooks = hooks
        return self
