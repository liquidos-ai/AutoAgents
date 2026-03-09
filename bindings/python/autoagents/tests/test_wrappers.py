from __future__ import annotations

import json
from typing import Any

import pytest

import autoagents as aa
import autoagents.prebuilt as prebuilt_module
import autoagents.prebuilt.memory as prebuilt_memory_module
from autoagents import _core as core_module
from autoagents import experimental as experimental_api
from autoagents import runtime as runtime_module
from autoagents.prebuilt import BasicAgent, ReActAgent, SlidingWindowMemory
from autoagents.task import ImageMime, Task, TaskImage
from autoagents.traits import AgentHooks, HookOutcome


class _FakeCoreTopic:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"Topic('{self.name}')"


class _FakeCoreEventStream:
    async def __anext__(self) -> Any:
        raise StopAsyncIteration


class _FakeCoreRuntime:
    def __init__(self) -> None:
        self.published: list[tuple[Any, str]] = []
        self.inner_stream = _FakeCoreEventStream()

    async def publish(self, topic: Any, task: str) -> None:
        self.published.append((topic, task))

    async def event_stream(self) -> _FakeCoreEventStream:
        return self.inner_stream


class _FakeCoreEnvironment:
    def __init__(self) -> None:
        self.registered: list[Any] = []
        self.ran = False
        self.inner_stream = _FakeCoreEventStream()

    def register_runtime(self, runtime: Any) -> None:
        self.registered.append(runtime)

    def run(self) -> None:
        self.ran = True

    def event_stream(self) -> _FakeCoreEventStream:
        return self.inner_stream


class _DemoMemory(experimental_api.MemoryProvider):
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def _to_core_provider(self) -> dict[str, Any]:
        return {"kind": "demo"}

    async def remember(self, message: dict[str, Any]) -> None:
        self.calls.append(("remember", message))

    async def recall(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append(("recall", (query, limit)))
        return []

    async def clear(self) -> None:
        self.calls.append(("clear", None))

    def size(self) -> int:
        return len(self.calls)


def test_public_api_exports_key_symbols():
    exported = set(aa.__all__)
    assert {"Agent", "AgentBuilder", "LLMBuilder", "Tool", "Topic"}.issubset(exported)
    assert "ExecutorProtocol" not in exported
    assert "HooksProtocol" not in exported
    assert "CustomExecutor" not in exported
    assert "MemoryProvider" not in exported
    assert "AgentExecutor" not in exported
    assert "experimental" in exported
    assert {"BasicAgent", "ReActAgent", "SlidingWindowMemory"} == set(prebuilt_module.__all__)


def test_core_coerce_llm_provider_accepts_built_provider(built_llm):
    assert core_module._coerce_llm_provider(built_llm) is built_llm


def test_core_coerce_llm_provider_rejects_unknown_object():
    with pytest.raises(TypeError, match="expected an AutoAgents LLMProvider"):
        core_module._coerce_llm_provider(object())


def test_task_payloads_cover_optional_fields():
    image = TaskImage(mime=ImageMime.PNG, data=b"png-bytes")
    assert image.to_payload() == {"mime": "png", "data": b"png-bytes"}

    custom_image = TaskImage(mime="jpeg", data=b"jpeg-bytes")
    task = Task(prompt="describe", image=custom_image, system_prompt="system")
    assert task.to_payload() == {
        "prompt": "describe",
        "system_prompt": "system",
        "image": {"mime": "jpeg", "data": b"jpeg-bytes"},
    }


@pytest.mark.asyncio
async def test_runtime_and_environment_wrap_core_types(monkeypatch):
    monkeypatch.setattr(runtime_module, "_CoreTopic", _FakeCoreTopic)
    monkeypatch.setattr(runtime_module, "_CoreRuntime", _FakeCoreRuntime)
    monkeypatch.setattr(runtime_module, "_CoreEnvironment", _FakeCoreEnvironment)

    runtime = runtime_module.Runtime()
    topic = runtime_module.Topic("jobs")
    assert topic.name == "jobs"
    assert repr(topic) == "Topic('jobs')"

    await runtime.publish(topic, "hello")
    stream = await runtime.event_stream()
    assert stream.__class__.__name__ == "EventStream"

    environment = runtime_module.Environment()
    environment.register_runtime(runtime)
    environment.run()
    env_stream = environment.event_stream()
    assert env_stream.__class__.__name__ == "EventStream"


@pytest.mark.asyncio
async def test_memory_provider_base_methods_raise():
    provider = experimental_api.MemoryProvider()

    with pytest.raises(NotImplementedError, match="remember"):
        await provider.remember({"role": "user", "content": "hi"})

    with pytest.raises(NotImplementedError, match="recall"):
        await provider.recall("query")

    with pytest.raises(NotImplementedError, match="clear"):
        await provider.clear()

    with pytest.raises(NotImplementedError, match="size"):
        provider.size()


def test_memory_provider_apply_uses_core_provider():
    provider = _DemoMemory()
    applied: list[Any] = []

    class _FakeBuilder:
        def memory(self, mem: Any) -> None:
            applied.append(mem)

    provider.apply(_FakeBuilder())
    assert applied == [{"kind": "demo"}]


def test_sliding_window_memory_uses_binding_constructor(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(prebuilt_memory_module, "sliding_window_memory", lambda size: (sentinel, size))

    assert SlidingWindowMemory(7)._to_core_provider() == (sentinel, 7)


@pytest.mark.asyncio
async def test_agent_hooks_default_methods_return_continue():
    hooks = AgentHooks()
    ctx = {
        "id": "ctx",
        "name": "agent",
        "description": "desc",
        "stream": False,
        "messages": [],
        "llm": object(),
        "memory": object(),
    }
    task = {"prompt": "hello"}
    result = {"response": "ok", "done": True, "tool_calls": []}

    assert await hooks.on_run_start(task, ctx) is HookOutcome.CONTINUE
    assert await hooks.on_tool_call({"name": "search"}, ctx) is HookOutcome.CONTINUE
    assert await hooks.on_agent_create() is None
    assert await hooks.on_run_complete(task, result, ctx) is None
    assert await hooks.on_turn_start(1, ctx) is None
    assert await hooks.on_turn_complete(1, ctx) is None
    assert await hooks.on_tool_start({"name": "search"}, ctx) is None
    assert await hooks.on_tool_result({"name": "search"}, {"ok": True}, ctx) is None
    assert await hooks.on_tool_error({"name": "search"}, {"error": "bad"}, ctx) is None
    assert await hooks.on_agent_shutdown() is None


@pytest.mark.asyncio
async def test_agent_executor_base_methods_raise_not_implemented():
    executor = experimental_api.AgentExecutor()
    ctx = {
        "id": "ctx",
        "name": "agent",
        "description": "desc",
        "stream": False,
        "messages": [],
        "llm": object(),
        "memory": object(),
    }

    with pytest.raises(NotImplementedError, match="config"):
        executor.config()

    with pytest.raises(NotImplementedError, match="execute"):
        await executor.execute({"prompt": "hello"}, ctx)

    with pytest.raises(NotImplementedError, match="execute_stream"):
        executor.execute_stream({"prompt": "hello"}, ctx)


def test_executor_specs_validate_schema_and_mutate_in_place():
    agent = BasicAgent("basic", "desc")
    assert agent.tools([]) is agent
    assert agent.hooks(object()) is agent
    assert agent.output_schema(json.dumps({"type": "object"})) is agent
    assert repr(agent) == "BasicAgent(name='basic')"

    react = ReActAgent("react", "desc")
    assert react.max_turns(3) is react
    assert repr(react) == "ReActAgent(name='react', max_turns=3)"

    with pytest.raises(ValueError, match="max_turns must be > 0"):
        react.max_turns(0)

    with pytest.raises(json.JSONDecodeError):
        agent.output_schema("{")
