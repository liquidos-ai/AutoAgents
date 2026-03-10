"""Python convenience callback classes for agent customization."""

from __future__ import annotations

from enum import Enum
from typing import AsyncIterator

from .types import ExecutorConfig, ExecutorContext, ExecutorOutput, ExecutorTask, JsonObject


class HookOutcome(str, Enum):
    """Hook outcome, mirroring Rust's `HookOutcome`."""

    CONTINUE = "continue"
    ABORT = "abort"


class AgentHooks:
    """Optional lifecycle callback base class.

    Subclassing this type is not required. Any object with matching optional
    hook methods can be passed to `.hooks(...)`.
    """

    async def on_agent_create(self) -> None:
        return None

    async def on_run_start(
        self,
        task: ExecutorTask,
        ctx: ExecutorContext,
    ) -> HookOutcome:
        return HookOutcome.CONTINUE

    async def on_run_complete(
        self,
        task: ExecutorTask,
        result: ExecutorOutput,
        ctx: ExecutorContext,
    ) -> None:
        return None

    async def on_turn_start(self, turn_index: int, ctx: ExecutorContext) -> None:
        return None

    async def on_turn_complete(self, turn_index: int, ctx: ExecutorContext) -> None:
        return None

    async def on_tool_call(
        self,
        tool_call: JsonObject,
        ctx: ExecutorContext,
    ) -> HookOutcome:
        return HookOutcome.CONTINUE

    async def on_tool_start(self, tool_call: JsonObject, ctx: ExecutorContext) -> None:
        return None

    async def on_tool_result(
        self,
        tool_call: JsonObject,
        result: JsonObject,
        ctx: ExecutorContext,
    ) -> None:
        return None

    async def on_tool_error(
        self,
        tool_call: JsonObject,
        err: JsonObject,
        ctx: ExecutorContext,
    ) -> None:
        return None

    async def on_agent_shutdown(self) -> None:
        return None


class AgentExecutor:
    """Experimental convenience base for custom Python executor callbacks.

    Subclassing this type is not required. `CustomExecutor(...)` accepts any
    object with callable `config()`, `execute(...)`, and `execute_stream(...)`
    methods. Import it through ``autoagents.experimental`` instead of the
    stable root package.
    """

    def config(self) -> ExecutorConfig:
        raise NotImplementedError("AgentExecutor.config must be implemented")

    async def execute(self, task: ExecutorTask, ctx: ExecutorContext) -> ExecutorOutput:
        raise NotImplementedError("AgentExecutor.execute must be implemented")

    def execute_stream(
        self,
        task: ExecutorTask,
        ctx: ExecutorContext,
    ) -> AsyncIterator[ExecutorOutput]:
        raise NotImplementedError("AgentExecutor.execute_stream must be implemented")
