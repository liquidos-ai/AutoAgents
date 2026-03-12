from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

import autoagents_py as aa
from autoagents_py import experimental as experimental_api
from autoagents_py.agent import (
    ActorAgentHandle,
    Agent,
    AgentBuilder,
    AgentHandle,
    _infer_output_schema,
    _resolve_custom_executor_max_turns,
    _topic_name,
)
from autoagents_py.events import SendMessage
from autoagents_py.exceptions import (
    AgentBuildError,
    AgentRunError,
    AgentTimeoutError,
    ExperimentalFeatureError,
)
from autoagents_py.task import Task


class _AsyncSequence(AsyncIterator[dict[str, Any]]):
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._items = iter(items)

    def __aiter__(self) -> "_AsyncSequence":
        return self

    async def __anext__(self) -> dict[str, Any]:
        try:
            return next(self._items)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _FakeCoreHandle:
    def __init__(
        self,
        result: dict[str, Any] | None = None,
        error: Exception | None = None,
        stream_items: list[dict[str, Any]] | None = None,
    ) -> None:
        self.result = result or {"response": "ok", "tool_calls": [], "done": True, "events": []}
        self.error = error
        self.stream_items = stream_items or [{"response": "chunk", "done": False, "tool_calls": []}]
        self.run_payloads: list[Any] = []

    async def run(self, payload: Any) -> dict[str, Any]:
        self.run_payloads.append(payload)
        if self.error is not None:
            raise self.error
        return self.result

    def event_stream(self) -> _AsyncSequence:
        return _AsyncSequence(
            [
                {"kind": "send_message", "actor_id": "actor-1", "message": "hello"},
            ]
        )

    async def run_stream(self, payload: Any) -> _AsyncSequence:
        self.run_payloads.append(payload)
        if self.error is not None:
            raise self.error
        return _AsyncSequence(self.stream_items)


class _FakeActorCoreHandle:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send(self, prompt: str) -> None:
        self.messages.append(prompt)


def test_infer_output_schema_prefers_schema_helpers_and_skips_private_fields():
    class _ModelV2:
        @staticmethod
        def model_json_schema() -> dict[str, Any]:
            return {"type": "object", "properties": {"ok": {"type": "boolean"}}}

    class _ModelV1:
        @staticmethod
        def model_json_schema() -> dict[str, Any]:
            raise RuntimeError("boom")

        @staticmethod
        def schema() -> dict[str, Any]:
            return {"type": "object", "properties": {"value": {"type": "integer"}}}

    class _Annotated:
        visible: int
        _hidden: str

    assert _infer_output_schema(_ModelV2)["properties"]["ok"] == {"type": "boolean"}
    assert _infer_output_schema(_ModelV1)["properties"]["value"] == {"type": "integer"}
    assert _infer_output_schema(_Annotated) == {
        "type": "object",
        "properties": {"visible": {"type": "integer"}},
        "additionalProperties": False,
        "required": ["visible"],
    }


def test_topic_name_accepts_topic_objects():
    topic = aa.Topic("jobs")
    assert _topic_name("alerts") == "alerts"
    assert _topic_name(topic) == "jobs"


def test_resolve_custom_executor_max_turns_validates_configs():
    class _DictExecutor:
        def config(self) -> dict[str, int]:
            return {"max_turns": 3}

    class _ObjectConfig:
        max_turns = 5

    class _ObjectExecutor:
        def config(self) -> _ObjectConfig:
            return _ObjectConfig()

    class _MissingConfig:
        pass

    class _MissingMaxTurns:
        def config(self) -> dict[str, int]:
            return {}

    class _BadMaxTurns:
        def config(self) -> dict[str, int]:
            return {"max_turns": 0}

    assert _resolve_custom_executor_max_turns(_DictExecutor()) == 3
    assert _resolve_custom_executor_max_turns(_ObjectExecutor()) == 5

    with pytest.raises(TypeError, match="config"):
        _resolve_custom_executor_max_turns(_MissingConfig())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_turns"):
        _resolve_custom_executor_max_turns(_MissingMaxTurns())

    with pytest.raises(ValueError, match="must be > 0"):
        _resolve_custom_executor_max_turns(_BadMaxTurns())


@pytest.mark.asyncio
async def test_agent_handle_converts_results_and_streams_events():
    handle = AgentHandle(
        _FakeCoreHandle(
            result={
                "response": "ok",
                "tool_calls": [],
                "done": True,
                "events": [{"kind": "send_message", "actor_id": "actor-1", "message": "hello"}],
            },
            stream_items=[
                {"response": "partial", "done": False, "tool_calls": []},
                {"response": "done", "done": True, "tool_calls": []},
            ],
        )
    )

    result = await handle.run(Task(prompt="hello"))
    assert result["response"] == "ok"
    assert result["events"] == [SendMessage(actor_id="actor-1", message="hello")]

    response, events = await handle.run_with_events("ping")
    assert response == "ok"
    assert events == [SendMessage(actor_id="actor-1", message="hello")]

    event_stream = handle.event_stream()
    assert await anext(event_stream) == SendMessage(actor_id="actor-1", message="hello")

    streamed = [chunk async for chunk in handle.run_stream("stream")]
    assert streamed == [
        {"response": "partial", "done": False, "tool_calls": []},
        {"response": "done", "done": True, "tool_calls": []},
    ]


@pytest.mark.asyncio
async def test_agent_handle_maps_runtime_errors_and_timeouts():
    async def _slow_run(_: Any) -> dict[str, Any]:
        await asyncio.sleep(0.05)
        return {"response": "late", "tool_calls": [], "done": True, "events": []}

    class _SlowCoreHandle(_FakeCoreHandle):
        async def run(self, payload: Any) -> dict[str, Any]:
            return await _slow_run(payload)

    with pytest.raises(AgentTimeoutError, match="timed out"):
        await AgentHandle(_SlowCoreHandle()).run("slow", timeout=0.001)

    error_handle = AgentHandle(_FakeCoreHandle(error=RuntimeError("boom")))
    with pytest.raises(AgentRunError, match="boom"):
        await error_handle.run("fail")

    with pytest.raises(AgentRunError, match="boom"):
        await anext(error_handle.run_stream("fail"))


@pytest.mark.asyncio
async def test_actor_agent_handle_forwards_send():
    core = _FakeActorCoreHandle()
    handle = ActorAgentHandle(core)
    await handle.send("hello")
    assert core.messages == ["hello"]


@pytest.mark.asyncio
async def test_agent_builder_wraps_core_builder(monkeypatch):
    fake_core_handle = _FakeCoreHandle()
    fake_actor_handle = _FakeActorCoreHandle()

    class _FakeRustBuilder:
        def runtime(self, _runtime) -> None:
            return None

        def subscribe(self, _topic: str) -> None:
            return None

        async def build(self) -> _FakeCoreHandle:
            return fake_core_handle

        async def build_actor(self) -> _FakeActorCoreHandle:
            return fake_actor_handle

    builder = AgentBuilder(aa.BasicAgent("basic", "desc"))
    monkeypatch.setattr(builder, "_make_rust_builder", lambda: _FakeRustBuilder())

    handle = await builder.build()
    actor_handle = await builder.build_actor(aa.Runtime(), topics=["jobs"])

    assert isinstance(handle, AgentHandle)
    assert isinstance(actor_handle, ActorAgentHandle)


@pytest.mark.asyncio
async def test_agent_builder_maps_build_errors(monkeypatch):
    class _BadRustBuilder:
        def runtime(self, _runtime) -> None:
            return None

        def subscribe(self, _topic: str) -> None:
            return None

        async def build(self) -> None:
            raise RuntimeError("bad build")

        async def build_actor(self) -> None:
            raise RuntimeError("bad actor build")

    builder = AgentBuilder(aa.BasicAgent("basic", "desc"))
    monkeypatch.setattr(builder, "_make_rust_builder", lambda: _BadRustBuilder())

    with pytest.raises(AgentBuildError, match="bad build"):
        await builder.build()

    with pytest.raises(AgentBuildError, match="bad actor build"):
        await builder.build_actor(aa.Runtime())


def test_agent_builder_requires_llm_and_memory():
    builder = AgentBuilder(aa.BasicAgent("basic", "desc"))

    with pytest.raises(AgentBuildError, match="LLM provider is required"):
        builder._make_rust_builder()

    builder._llm = object()  # type: ignore[assignment]
    with pytest.raises(AgentBuildError, match="Memory provider is required"):
        builder._make_rust_builder()


@pytest.mark.asyncio
async def test_agent_convenience_wrapper_lazy_build_and_reset(monkeypatch):
    fake_handle = AgentHandle(_FakeCoreHandle())

    class _FakeBuilder:
        def __init__(self) -> None:
            self.build_calls = 0

        async def build(self) -> AgentHandle:
            self.build_calls += 1
            return fake_handle

    agent = Agent.__new__(Agent)
    agent._builder = _FakeBuilder()
    agent._handle = None

    with pytest.raises(AgentRunError, match="must be built"):
        agent.event_stream()

    result = await agent.run("hello")
    assert result["response"] == "ok"
    assert agent.event_stream().__class__.__name__ == "EventStream"

    await agent.reset()
    assert agent._builder.build_calls == 2


def test_custom_executor_max_turns_validation():
    class _Executor:
        def config(self) -> dict[str, int]:
            return {"max_turns": 2}

        async def execute(self, task, ctx):
            return {"response": "ok", "done": True, "tool_calls": []}

        async def execute_stream(self, task, ctx):
            yield {"response": "ok", "done": True, "tool_calls": []}

    executor = experimental_api.CustomExecutor("custom", "desc", _Executor())  # type: ignore[arg-type]
    assert executor.max_turns(3) is executor

    with pytest.raises(ValueError, match="max_turns must be > 0"):
        executor.max_turns(0)


def test_stable_agent_builder_rejects_experimental_custom_executor():
    class _Executor:
        def config(self) -> dict[str, int]:
            return {"max_turns": 1}

        async def execute(self, task, ctx):
            return {"response": "ok", "done": True, "tool_calls": []}

        async def execute_stream(self, task, ctx):
            yield {"response": "ok", "done": True, "tool_calls": []}

    with pytest.raises(ExperimentalFeatureError, match="CustomExecutor is experimental"):
        AgentBuilder(experimental_api.CustomExecutor("custom", "desc", _Executor()))
