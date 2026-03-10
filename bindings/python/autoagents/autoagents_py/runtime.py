"""Python wrappers for multi-agent runtime types."""

from __future__ import annotations

from ._core import _drive_native
from ._core import Environment as _CoreEnvironment
from ._core import Runtime as _CoreRuntime
from ._core import Topic as _CoreTopic
from .events import EventStream


class Topic:
    def __init__(self, name: str) -> None:
        self._inner = _CoreTopic(name)

    @property
    def name(self) -> str:
        return self._inner.name

    def __repr__(self) -> str:
        return repr(self._inner)


class Runtime:
    def __init__(self) -> None:
        self._inner = _CoreRuntime()

    async def publish(self, topic: Topic, task: str) -> None:
        await _drive_native(self._inner.publish(topic._inner, task))

    async def event_stream(self) -> EventStream:
        return EventStream(await _drive_native(self._inner.event_stream()))


class Environment:
    def __init__(self) -> None:
        self._inner = _CoreEnvironment()

    def register_runtime(self, runtime: Runtime) -> None:
        self._inner.register_runtime(runtime._inner)

    def run(self) -> None:
        self._inner.run()

    def event_stream(self) -> EventStream:
        return EventStream(self._inner.event_stream())


__all__ = ["Runtime", "Environment", "Topic"]
