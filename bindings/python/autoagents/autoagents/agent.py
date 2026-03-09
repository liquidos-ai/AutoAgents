"""Public Python agent API built on top of the private Rust bridge."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import AsyncIterator, Callable, List, Optional, Tuple, Type, Union, cast, get_type_hints

from ._core import AgentBuilder as _CoreAgentBuilder
from ._core import ActorAgentHandle as _CoreActorAgentHandle
from ._core import AgentHandle as _CoreAgentHandle
from ._core import LLMProvider
from ._core import _NativeAsyncIterator, _drive_coroutine, _drive_native
from ._core import _coerce_llm_provider
from ._executor_spec import _ExecutorSpec
from .events import EventStream, ProtocolEvent, events_from_payloads
from .exceptions import (
    AgentBuildError,
    AgentConfigError,
    AgentRunError,
    AgentTimeoutError,
    ExperimentalFeatureError,
)
from .memory import apply_memory_provider
from .runtime import Runtime, Topic
from .task import Task
from .tool import _dataclass_json_schema, _json_schema_for_type
from .types import AgentRunResult, ExecutorConfig, ExecutorOutput, JsonObject, TaskPayload


def _infer_output_schema(model: Type[object]) -> JsonObject:
    """Build a JSON Schema from dataclass/pydantic/annotated class types."""
    model_json_schema = cast(
        Optional[Callable[[], JsonObject]],
        getattr(model, "model_json_schema", None),
    )
    if callable(model_json_schema):
        try:
            return model_json_schema()
        except Exception:
            pass

    schema_fn = cast(Optional[Callable[[], JsonObject]], getattr(model, "schema", None))
    if callable(schema_fn):
        try:
            return schema_fn()
        except Exception:
            pass

    if dataclasses.is_dataclass(model) and isinstance(model, type):
        return _dataclass_json_schema(model)

    try:
        hints = get_type_hints(model)
    except Exception:
        hints = {}
    props: JsonObject = {}
    required: List[str] = []
    for name, hinted_type in hints.items():
        if name.startswith("_"):
            continue
        props[name] = _json_schema_for_type(hinted_type)
        required.append(name)

    schema: JsonObject = {
        "type": "object",
        "properties": props,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def _to_structured_output_format(model: Type[object]) -> JsonObject:
    model_name = getattr(model, "__name__", "StructuredOutput")
    return {
        "name": model_name,
        "description": f"Structured output for {model_name}",
        "schema": _infer_output_schema(model),
        "strict": True,
    }


def _require_callable_attr(obj: object, attr: str, owner_name: str) -> None:
    if not callable(getattr(obj, attr, None)):
        raise TypeError(f"{owner_name} must implement callable {attr}()")


def _resolve_custom_executor_max_turns(executor: object) -> int:
    _require_callable_attr(executor, "config", "Custom executors")
    config_fn = getattr(executor, "config", None)
    try:
        config = cast(ExecutorConfig, config_fn())
    except NotImplementedError as exc:
        raise TypeError("Custom executors must implement config()") from exc
    if isinstance(config, dict):
        if "max_turns" not in config:
            raise ValueError("Custom executor config() must include max_turns")
        max_turns = int(config["max_turns"])
    else:
        if not hasattr(config, "max_turns"):
            raise ValueError("Custom executor config() must include max_turns")
        max_turns = int(getattr(config, "max_turns"))
    if max_turns <= 0:
        raise ValueError("Custom executor config.max_turns must be > 0")
    return max_turns


class CustomExecutor(_ExecutorSpec):
    """Experimental Python-backed executor specification."""

    _binding_kind = "custom"

    def __init__(self, name: str, description: str, executor: object) -> None:
        super().__init__(name, description)
        _require_callable_attr(executor, "execute", "Custom executors")
        _require_callable_attr(executor, "execute_stream", "Custom executors")
        self._executor_impl = executor
        self._max_turns = _resolve_custom_executor_max_turns(executor)

    def max_turns(self, turns: int) -> "CustomExecutor":
        if turns <= 0:
            raise ValueError("max_turns must be > 0")
        self._max_turns = turns
        return self

    def __repr__(self) -> str:
        return f"CustomExecutor(name={self._name!r})"


ExecutorLike = Union[_ExecutorSpec, CustomExecutor]
TopicLike = Union[str, Topic]


def _topic_name(topic: TopicLike) -> str:
    return topic if isinstance(topic, str) else topic.name


def _binding_kind(agent: object) -> str | None:
    kind = getattr(agent, "_binding_kind", None)
    return kind if isinstance(kind, str) else None


def _validate_stable_agent(agent: object) -> None:
    if _binding_kind(agent) == "custom":
        raise ExperimentalFeatureError(
            "CustomExecutor is experimental; import it from autoagents.experimental"
        )


class AgentHandle:
    """Built agent handle with run and stream methods."""

    def __init__(self, handle: _CoreAgentHandle) -> None:
        self._handle = handle

    async def run(
        self,
        task: Union[str, Task],
        *,
        timeout: Optional[float] = None,
    ) -> AgentRunResult:
        async def _run() -> AgentRunResult:
            payload: Union[str, TaskPayload] = (
                task.to_payload() if isinstance(task, Task) else task
            )

            try:
                coro = self._handle.run(payload)
                result = (
                    cast(AgentRunResult, await _drive_native(coro))
                    if timeout is None
                    else cast(AgentRunResult, await _drive_native(coro, timeout=timeout))
                )
            except asyncio.TimeoutError as exc:
                raise AgentTimeoutError(f"Agent run timed out after {timeout}s") from exc
            except Exception as exc:
                raise AgentRunError(str(exc)) from exc

            result["events"] = events_from_payloads(result.get("events", []))
            return result

        return await _drive_coroutine(_run())

    def event_stream(self) -> EventStream:
        return EventStream(self._handle.event_stream())

    async def events(self) -> AsyncIterator[ProtocolEvent]:
        async for event in self.event_stream():
            yield event

    async def run_with_events(
        self,
        task: Union[str, Task],
        *,
        timeout: Optional[float] = None,
    ) -> Tuple[Optional[str], List[ProtocolEvent]]:
        result = await self.run(task, timeout=timeout)
        return result.get("response"), cast(List[ProtocolEvent], result.get("events", []))

    async def run_stream(self, task: Union[str, Task]) -> AsyncIterator[ExecutorOutput]:
        payload: Union[str, TaskPayload] = (
            task.to_payload() if isinstance(task, Task) else task
        )
        try:
            output_stream = await _drive_native(self._handle.run_stream(payload))
        except Exception as exc:
            raise AgentRunError(str(exc)) from exc

        async for chunk in _NativeAsyncIterator[ExecutorOutput](output_stream):
            yield chunk


class ActorAgentHandle:
    """Built actor agent handle with direct mailbox send support."""

    def __init__(self, handle: _CoreActorAgentHandle) -> None:
        self._handle = handle

    async def send(self, prompt: str) -> None:
        await _drive_coroutine(_drive_native(self._handle.send(prompt)))


class _BaseAgentBuilder:
    """Python builder that mirrors Rust ``AgentBuilder::new(executor)``."""

    def __init__(self, agent: object, *, allow_experimental: bool) -> None:
        if not allow_experimental:
            _validate_stable_agent(agent)
        if not isinstance(agent, _ExecutorSpec):
            raise AgentConfigError("agent must be an AutoAgents executor specification")
        self._agent = agent
        self._allow_experimental = allow_experimental
        self._llm: Optional[LLMProvider] = None
        self._memory: Optional[object] = None
        self._output_schema: Optional[str] = None

    def llm(self, provider: LLMProvider) -> "AgentBuilder":
        self._llm = _coerce_llm_provider(provider)
        return self

    def memory(self, provider: object) -> "AgentBuilder":
        self._memory = provider
        return self

    def output(self, model: Type[object]) -> "AgentBuilder":
        schema = _to_structured_output_format(model)
        self._output_schema = json.dumps(schema)
        return self

    def _make_rust_builder(self) -> _CoreAgentBuilder:
        if self._llm is None:
            raise AgentBuildError("LLM provider is required")
        if self._memory is None:
            raise AgentBuildError("Memory provider is required")

        builder = _CoreAgentBuilder(self._agent)
        builder.llm(self._llm)
        if self._output_schema is not None:
            self._agent._output_schema_json = self._output_schema
        if self._allow_experimental:
            from .experimental.memory import apply_memory_provider as apply_experimental_memory

            apply_experimental_memory(builder, self._memory)
        else:
            apply_memory_provider(builder, self._memory)
        return builder

    async def build(self) -> AgentHandle:
        async def _build() -> AgentHandle:
            try:
                handle = await _drive_native(self._make_rust_builder().build())
            except AgentBuildError:
                raise
            except Exception as exc:
                raise AgentBuildError(str(exc)) from exc
            return AgentHandle(handle)

        return await _drive_coroutine(_build())

    async def build_actor(
        self,
        runtime: Runtime,
        *,
        topics: Optional[List[TopicLike]] = None,
    ) -> ActorAgentHandle:
        async def _build_actor() -> ActorAgentHandle:
            try:
                builder = self._make_rust_builder()
                builder.runtime(runtime._inner)
                for topic in topics or []:
                    builder.subscribe(_topic_name(topic))
                return ActorAgentHandle(await _drive_native(builder.build_actor()))
            except AgentBuildError:
                raise
            except Exception as exc:
                raise AgentBuildError(str(exc)) from exc

        return await _drive_coroutine(_build_actor())


class AgentBuilder(_BaseAgentBuilder):
    """Stable builder for Rust-backed agents and Rust-backed memory."""

    def __init__(self, agent: _ExecutorSpec) -> None:
        super().__init__(agent, allow_experimental=False)


class _BaseAgent:
    """Convenience wrapper around ``AgentBuilder`` with lazy build semantics."""

    def __init__(
        self,
        agent: object,
        *,
        llm: LLMProvider,
        memory: Optional[object] = None,
        output: Optional[Type[object]] = None,
        allow_experimental: bool,
    ) -> None:
        builder = _BaseAgentBuilder(agent, allow_experimental=allow_experimental).llm(llm)
        if memory is not None:
            builder.memory(memory)
        if output is not None:
            builder.output(output)

        self._builder = builder
        self._handle: Optional[AgentHandle] = None

    async def build(self) -> "Agent":
        self._handle = await self._builder.build()
        return self

    async def reset(self) -> "Agent":
        self._handle = await self._builder.build()
        return self

    async def _ensure_built(self) -> AgentHandle:
        if self._handle is None:
            await self.build()
        assert self._handle is not None
        return self._handle

    async def run(
        self,
        task: Union[str, Task],
        *,
        timeout: Optional[float] = None,
    ) -> AgentRunResult:
        handle = await self._ensure_built()
        return await handle.run(task, timeout=timeout)

    def event_stream(self) -> EventStream:
        if self._handle is None:
            raise AgentRunError("Agent must be built before requesting an event stream")
        return self._handle.event_stream()

    async def events(self) -> AsyncIterator[ProtocolEvent]:
        handle = await self._ensure_built()
        async for event in handle.events():
            yield event

    async def run_with_events(
        self,
        task: Union[str, Task],
        *,
        timeout: Optional[float] = None,
    ) -> Tuple[Optional[str], List[ProtocolEvent]]:
        handle = await self._ensure_built()
        return await handle.run_with_events(task, timeout=timeout)

    async def run_stream(self, task: Union[str, Task]) -> AsyncIterator[ExecutorOutput]:
        handle = await self._ensure_built()
        async for chunk in handle.run_stream(task):
            yield chunk


class Agent(_BaseAgent):
    """Stable convenience wrapper for Rust-backed agents and Rust-backed memory."""

    def __init__(
        self,
        agent: _ExecutorSpec,
        *,
        llm: LLMProvider,
        memory: Optional[object] = None,
        output: Optional[Type[object]] = None,
    ) -> None:
        super().__init__(
            agent,
            llm=llm,
            memory=memory,
            output=output,
            allow_experimental=False,
        )
