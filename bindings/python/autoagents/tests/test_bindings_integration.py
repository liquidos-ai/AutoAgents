from __future__ import annotations

import asyncio
from typing import Any

import pytest

import autoagents as aa
from autoagents import experimental as experimental_api
from autoagents.exceptions import AgentRunError
from autoagents._core import AgentBuilder as CoreAgentBuilder


class _RecordingMemory:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def remember(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    async def recall(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if limit is None:
            return list(self.messages)
        return list(self.messages[-limit:])

    async def clear(self) -> None:
        self.messages.clear()

    def size(self) -> int:
        return len(self.messages)


class _RecordingHooks:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def on_agent_create(self) -> None:
        self.calls.append("create")

    async def on_run_start(self, task: aa.ExecutorTask, ctx: aa.ExecutorContext) -> aa.HookOutcome:
        self.calls.append(f"run_start:{task['prompt']}")
        return aa.HookOutcome.CONTINUE

    async def on_run_complete(
        self,
        task: aa.ExecutorTask,
        result: aa.ExecutorOutput,
        ctx: aa.ExecutorContext,
    ) -> None:
        self.calls.append(f"run_complete:{result['response']}")

    async def on_turn_start(self, turn_index: int, ctx: aa.ExecutorContext) -> None:
        self.calls.append(f"turn_start:{turn_index}")

    async def on_turn_complete(self, turn_index: int, ctx: aa.ExecutorContext) -> None:
        self.calls.append(f"turn_complete:{turn_index}")


class _MemoryExecutor:
    def config(self) -> aa.ExecutorConfig:
        return {"max_turns": 2}

    async def execute(self, task: aa.ExecutorTask, ctx: aa.ExecutorContext) -> aa.ExecutorOutput:
        assert ctx["memory"].is_configured() is True
        await ctx["memory"].remember({"role": "user", "content": task["prompt"]})
        recalled = await ctx["memory"].recall(task["prompt"], limit=10)
        size = await ctx["memory"].size()
        return {
            "response": f"{recalled[-1]['content']}:{size}",
            "done": True,
            "tool_calls": [],
        }

    async def execute_stream(
        self,
        task: aa.ExecutorTask,
        ctx: aa.ExecutorContext,
    ):
        yield {"response": "partial", "done": False, "tool_calls": []}
        yield {"response": task["prompt"], "done": True, "tool_calls": []}


class _ValidationExecutor:
    def config(self) -> aa.ExecutorConfig:
        return {"max_turns": 1}

    async def execute(self, task: aa.ExecutorTask, ctx: aa.ExecutorContext) -> aa.ExecutorOutput:
        errors: list[str] = []

        for callback in (
            lambda: ctx["llm"].chat(123),  # type: ignore[arg-type]
            lambda: ctx["llm"].chat("hello", object()),  # type: ignore[arg-type]
            lambda: ctx["llm"].chat_with_tools("hello", [object()]),  # type: ignore[list-item]
        ):
            try:
                await callback()
            except RuntimeError as exc:
                errors.append(str(exc))

        return {"response": " | ".join(errors), "done": True, "tool_calls": []}

    async def execute_stream(
        self,
        task: aa.ExecutorTask,
        ctx: aa.ExecutorContext,
    ):
        yield {"response": task["prompt"], "done": True, "tool_calls": []}


class _FailingHooks:
    async def on_run_start(self, task: aa.ExecutorTask, ctx: aa.ExecutorContext) -> aa.HookOutcome:
        raise RuntimeError("hook exploded")


@pytest.mark.asyncio
async def test_custom_executor_runs_against_real_binding(built_llm):
    hooks = _RecordingHooks()
    memory = _RecordingMemory()
    spec = experimental_api.CustomExecutor("echo", "Echo executor", _MemoryExecutor()).hooks(hooks)
    handle = (
        await experimental_api.ExperimentalAgentBuilder(spec)
        .llm(built_llm)
        .memory(memory)
        .build()
    )

    result = await handle.run(aa.Task(prompt="ping", system_prompt="system"))
    streamed = [chunk async for chunk in handle.run_stream("pong")]

    assert result["response"] == "ping:1"
    assert result["done"] is True
    assert result["tool_calls"] == []
    assert result["events"]
    assert all(not isinstance(event, dict) for event in result["events"])
    assert streamed == [
        {"response": "partial", "done": False, "tool_calls": []},
        {"response": "pong", "done": True, "tool_calls": []},
    ]
    assert hooks.calls[0] == "create"
    assert "run_start:ping" in hooks.calls
    assert "run_complete:ping:1" in hooks.calls
    assert "turn_start:1" in hooks.calls
    assert "turn_complete:1" in hooks.calls


@pytest.mark.asyncio
async def test_core_builder_native_awaitables_resume_without_wrapper_delays(built_llm):
    spec = experimental_api.CustomExecutor("echo", "Echo executor", _MemoryExecutor())
    builder = CoreAgentBuilder(spec)
    builder.llm(built_llm)
    experimental_api.apply_memory_provider(builder, _RecordingMemory())

    handle = await builder.build()
    result = await handle.run("hello")

    assert result["response"] == "hello:1"
    assert result["events"]


@pytest.mark.asyncio
async def test_agent_wrapper_builds_lazily_with_real_binding(built_llm):
    agent = experimental_api.Agent(
        experimental_api.CustomExecutor("echo", "Echo executor", _MemoryExecutor()),
        llm=built_llm,
        memory=aa.SlidingWindowMemory(4),
    )

    result = await agent.run("hello")
    assert result["response"] == "hello:1"

    await agent.reset()
    assert (await agent.run("again"))["response"] == "again:1"


@pytest.mark.asyncio
async def test_execution_llm_validation_errors_do_not_require_network(built_llm):
    handle = await (
        experimental_api.ExperimentalAgentBuilder(
            experimental_api.CustomExecutor("validate", "Validate inputs", _ValidationExecutor())
        )
        .llm(built_llm)
        .memory(_RecordingMemory())
        .build()
    )

    result = await handle.run("validate")
    assert "message must be a string or dict" in result["response"]
    assert "invalid schema:" in result["response"]
    assert "tools must contain Tool instances created by autoagents.tool" in result["response"]


@pytest.mark.asyncio
async def test_runtime_event_stream_builds_without_hanging():
    runtime = aa.Runtime()

    stream = await asyncio.wait_for(runtime.event_stream(), timeout=1.0)

    assert isinstance(stream, aa.EventStream)


@pytest.mark.asyncio
async def test_hook_failures_surface_as_agent_run_errors(built_llm):
    handle = await (
        aa.AgentBuilder(aa.BasicAgent("basic", "desc").hooks(_FailingHooks()))
        .llm(built_llm)
        .memory(aa.SlidingWindowMemory(4))
        .build()
    )

    with pytest.raises(aa.AgentRunError, match="hook on_run_start failed: hook exploded"):
        await handle.run("hello")


@pytest.mark.asyncio
async def test_agent_event_stream_requires_build_with_real_type():
    agent = experimental_api.Agent(
        experimental_api.CustomExecutor("echo", "Echo executor", _MemoryExecutor()),
        llm=aa.LLMBuilder("openai").api_key("test-key").model("gpt-4o-mini").build(),
        memory=_RecordingMemory(),
    )

    with pytest.raises(AgentRunError, match="must be built"):
        agent.event_stream()
