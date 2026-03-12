from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from autoagents_py.events import (
    EventStream,
    NewTask,
    SendMessage,
    StreamChunk,
    StreamComplete,
    StreamToolCall,
    TaskComplete,
    TaskError,
    TaskStarted,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallRequested,
    TurnCompleted,
    TurnStarted,
    event_from_payload,
    events_from_payloads,
)


class _FakeCoreEventStream:
    def __init__(self, payloads: list[Any]) -> None:
        self._items: Iterator[Any] = iter(payloads)

    async def __anext__(self) -> Any:
        try:
            return next(self._items)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def test_event_from_payload_covers_all_known_event_kinds():
    payloads = [
        (
            {"kind": "new_task", "actor_id": "actor-1", "prompt": "hi", "system_prompt": None},
            NewTask(actor_id="actor-1", prompt="hi", system_prompt=None),
        ),
        (
            {
                "kind": "task_started",
                "sub_id": "sub-1",
                "actor_id": "actor-1",
                "actor_name": "planner",
                "task_description": "plan",
            },
            TaskStarted(
                sub_id="sub-1",
                actor_id="actor-1",
                actor_name="planner",
                task_description="plan",
            ),
        ),
        (
            {
                "kind": "task_complete",
                "sub_id": "sub-1",
                "actor_id": "actor-1",
                "actor_name": "planner",
                "result": "done",
            },
            TaskComplete(
                sub_id="sub-1",
                actor_id="actor-1",
                actor_name="planner",
                result="done",
            ),
        ),
        (
            {"kind": "task_error", "sub_id": "sub-1", "actor_id": "actor-1", "error": "boom"},
            TaskError(sub_id="sub-1", actor_id="actor-1", error="boom"),
        ),
        (
            {"kind": "send_message", "actor_id": "actor-1", "message": "hello"},
            SendMessage(actor_id="actor-1", message="hello"),
        ),
        (
            {
                "kind": "tool_call_requested",
                "sub_id": "sub-1",
                "actor_id": "actor-1",
                "id": "call-1",
                "tool_name": "search",
                "arguments": "{}",
            },
            ToolCallRequested(
                sub_id="sub-1",
                actor_id="actor-1",
                id="call-1",
                tool_name="search",
                arguments="{}",
            ),
        ),
        (
            {
                "kind": "tool_call_completed",
                "sub_id": "sub-1",
                "actor_id": "actor-1",
                "id": "call-1",
                "tool_name": "search",
                "result": {"ok": True},
            },
            ToolCallCompleted(
                sub_id="sub-1",
                actor_id="actor-1",
                id="call-1",
                tool_name="search",
                result={"ok": True},
            ),
        ),
        (
            {
                "kind": "tool_call_failed",
                "sub_id": "sub-1",
                "actor_id": "actor-1",
                "id": "call-1",
                "tool_name": "search",
                "error": "bad input",
            },
            ToolCallFailed(
                sub_id="sub-1",
                actor_id="actor-1",
                id="call-1",
                tool_name="search",
                error="bad input",
            ),
        ),
        (
            {
                "kind": "turn_started",
                "sub_id": "sub-1",
                "actor_id": "actor-1",
                "turn_number": 1,
                "max_turns": 2,
            },
            TurnStarted(sub_id="sub-1", actor_id="actor-1", turn_number=1, max_turns=2),
        ),
        (
            {
                "kind": "turn_completed",
                "sub_id": "sub-1",
                "actor_id": "actor-1",
                "turn_number": 1,
                "final_turn": False,
            },
            TurnCompleted(sub_id="sub-1", actor_id="actor-1", turn_number=1, final_turn=False),
        ),
        (
            {"kind": "stream_chunk", "sub_id": "sub-1", "chunk": {"text": "a"}},
            StreamChunk(sub_id="sub-1", chunk={"text": "a"}),
        ),
        (
            {"kind": "stream_tool_call", "sub_id": "sub-1", "tool_call": {"id": "call-1"}},
            StreamToolCall(sub_id="sub-1", tool_call={"id": "call-1"}),
        ),
        (
            {"kind": "stream_complete", "sub_id": "sub-1"},
            StreamComplete(sub_id="sub-1"),
        ),
    ]

    for payload, expected in payloads:
        assert event_from_payload(payload) == expected


def test_event_from_payload_returns_existing_protocol_event():
    event = NewTask(actor_id="actor-1", prompt="prompt", system_prompt="system")
    assert event_from_payload(event) is event


def test_event_from_payload_rejects_invalid_payloads():
    with pytest.raises(TypeError, match="mapping"):
        event_from_payload(1)

    with pytest.raises(TypeError, match="missing string field 'kind'"):
        event_from_payload({"actor_id": "actor-1"})

    with pytest.raises(ValueError, match="unknown event kind"):
        event_from_payload({"kind": "mystery"})


def test_events_from_payloads_preserves_order():
    events = events_from_payloads(
        [
            {"kind": "send_message", "actor_id": "actor-1", "message": "hello"},
            {"kind": "stream_complete", "sub_id": "sub-1"},
        ]
    )
    assert events == [
        SendMessage(actor_id="actor-1", message="hello"),
        StreamComplete(sub_id="sub-1"),
    ]


@pytest.mark.asyncio
async def test_event_stream_wraps_core_iterator():
    stream = EventStream(
        _FakeCoreEventStream(
            [
                {"kind": "new_task", "actor_id": "actor-1", "prompt": "hi", "system_prompt": None},
            ]
        )
    )

    assert stream.__aiter__() is stream
    assert await stream.__anext__() == NewTask(
        actor_id="actor-1",
        prompt="hi",
        system_prompt=None,
    )

    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()
