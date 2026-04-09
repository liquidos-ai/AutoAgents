"""Typed protocol events exposed by the public Python API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable, Mapping, Union

from ._core import _drive_native
from ._core import EventStream as _CoreEventStream
from .types import JsonObject, JsonValue


@dataclass(slots=True, frozen=True)
class NewTask:
    actor_id: str
    prompt: str
    system_prompt: str | None


@dataclass(slots=True, frozen=True)
class TaskStarted:
    sub_id: str
    actor_id: str
    actor_name: str
    task_description: str


@dataclass(slots=True, frozen=True)
class TaskComplete:
    sub_id: str
    actor_id: str
    actor_name: str
    result: str


@dataclass(slots=True, frozen=True)
class TaskError:
    sub_id: str
    actor_id: str
    error: str


@dataclass(slots=True, frozen=True)
class SendMessage:
    actor_id: str
    message: str


@dataclass(slots=True, frozen=True)
class ToolCallRequested:
    sub_id: str
    actor_id: str
    id: str
    tool_name: str
    arguments: str


@dataclass(slots=True, frozen=True)
class ToolCallCompleted:
    sub_id: str
    actor_id: str
    id: str
    tool_name: str
    result: JsonValue


@dataclass(slots=True, frozen=True)
class ToolCallFailed:
    sub_id: str
    actor_id: str
    id: str
    tool_name: str
    error: str


@dataclass(slots=True, frozen=True)
class TurnStarted:
    sub_id: str
    actor_id: str
    turn_number: int
    max_turns: int


@dataclass(slots=True, frozen=True)
class TurnCompleted:
    sub_id: str
    actor_id: str
    turn_number: int
    final_turn: bool


@dataclass(slots=True, frozen=True)
class CodeExecutionStarted:
    sub_id: str
    actor_id: str
    execution_id: str
    language: str
    source: str


@dataclass(slots=True, frozen=True)
class CodeExecutionConsole:
    sub_id: str
    actor_id: str
    execution_id: str
    message: str


@dataclass(slots=True, frozen=True)
class CodeExecutionCompleted:
    sub_id: str
    actor_id: str
    execution_id: str
    result: JsonValue
    duration_ms: int


@dataclass(slots=True, frozen=True)
class CodeExecutionFailed:
    sub_id: str
    actor_id: str
    execution_id: str
    error: str
    duration_ms: int


@dataclass(slots=True, frozen=True)
class StreamChunk:
    sub_id: str
    chunk: JsonObject


@dataclass(slots=True, frozen=True)
class StreamToolCall:
    sub_id: str
    tool_call: JsonValue


@dataclass(slots=True, frozen=True)
class StreamComplete:
    sub_id: str


ProtocolEvent = Union[
    NewTask,
    TaskStarted,
    TaskComplete,
    TaskError,
    SendMessage,
    ToolCallRequested,
    ToolCallCompleted,
    ToolCallFailed,
    TurnStarted,
    TurnCompleted,
    CodeExecutionStarted,
    CodeExecutionConsole,
    CodeExecutionCompleted,
    CodeExecutionFailed,
    StreamChunk,
    StreamToolCall,
    StreamComplete,
]

_EVENT_TYPES = {
    "new_task": NewTask,
    "task_started": TaskStarted,
    "task_complete": TaskComplete,
    "task_error": TaskError,
    "send_message": SendMessage,
    "tool_call_requested": ToolCallRequested,
    "tool_call_completed": ToolCallCompleted,
    "tool_call_failed": ToolCallFailed,
    "turn_started": TurnStarted,
    "turn_completed": TurnCompleted,
    "code_execution_started": CodeExecutionStarted,
    "code_execution_console": CodeExecutionConsole,
    "code_execution_completed": CodeExecutionCompleted,
    "code_execution_failed": CodeExecutionFailed,
    "stream_chunk": StreamChunk,
    "stream_tool_call": StreamToolCall,
    "stream_complete": StreamComplete,
}


def event_from_payload(payload: Any) -> ProtocolEvent:
    if isinstance(payload, tuple(_EVENT_TYPES.values())):
        return payload
    if not isinstance(payload, Mapping):
        raise TypeError(f"event payload must be a mapping, got {type(payload).__name__}")
    kind = payload.get("kind")
    if not isinstance(kind, str):
        raise TypeError("event payload missing string field 'kind'")
    event_type = _EVENT_TYPES.get(kind)
    if event_type is None:
        raise ValueError(f"unknown event kind: {kind}")
    values = {key: value for key, value in payload.items() if key != "kind"}
    return event_type(**values)


def events_from_payloads(payloads: Iterable[Any]) -> list[ProtocolEvent]:
    return [event_from_payload(payload) for payload in payloads]


class EventStream(AsyncIterator[ProtocolEvent]):
    """Typed async iterator over protocol events."""

    def __init__(self, inner: _CoreEventStream) -> None:
        self._inner = inner

    def __aiter__(self) -> "EventStream":
        return self

    async def __anext__(self) -> ProtocolEvent:
        payload = await _drive_native(self._inner.__anext__())
        return event_from_payload(payload)
