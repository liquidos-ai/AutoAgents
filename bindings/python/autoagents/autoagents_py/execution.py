"""Python wrappers for native execution-context helpers."""

from __future__ import annotations

from typing import Any, AsyncIterator

from ._core import (
    _NativeAsyncIterator,
    _NativeExecutionJsonStream,
    _NativeExecutionLLM,
    _NativeExecutionMemory,
    _NativeExecutionStringStream,
    _drive_native,
)


class _ExecutionStringStream(AsyncIterator[str]):
    def __init__(self, inner: _NativeExecutionStringStream) -> None:
        self._inner = _NativeAsyncIterator[str](inner)

    def __aiter__(self) -> "_ExecutionStringStream":
        return self

    async def __anext__(self) -> str:
        return await self._inner.__anext__()


class _ExecutionJsonStream(AsyncIterator[dict[str, Any]]):
    def __init__(self, inner: _NativeExecutionJsonStream) -> None:
        self._inner = _NativeAsyncIterator[dict[str, Any]](inner)

    def __aiter__(self) -> "_ExecutionJsonStream":
        return self

    async def __anext__(self) -> dict[str, Any]:
        return await self._inner.__anext__()


class ExecutionLLM:
    def __init__(self, inner: _NativeExecutionLLM) -> None:
        self._inner = inner

    def __repr__(self) -> str:
        return repr(self._inner)

    async def chat(self, messages: Any, schema: Any = None) -> dict[str, Any]:
        return await _drive_native(self._inner.chat(messages, schema))

    async def chat_with_struct(self, messages: Any, schema: Any) -> dict[str, Any]:
        return await _drive_native(self._inner.chat_with_struct(messages, schema))

    async def chat_with_tools(
        self,
        messages: Any,
        tools: Any,
        schema: Any = None,
    ) -> dict[str, Any]:
        return await _drive_native(self._inner.chat_with_tools(messages, tools, schema))

    async def chat_with_tools_struct(
        self,
        messages: Any,
        tools: Any,
        schema: Any,
    ) -> dict[str, Any]:
        return await _drive_native(self._inner.chat_with_tools_struct(messages, tools, schema))

    async def chat_with_web_search(self, input: str) -> dict[str, Any]:
        return await _drive_native(self._inner.chat_with_web_search(input))

    async def chat_stream(self, messages: Any, schema: Any = None) -> AsyncIterator[str]:
        stream = await _drive_native(self._inner.chat_stream(messages, schema))
        return _ExecutionStringStream(stream)

    async def chat_stream_struct(
        self,
        messages: Any,
        tools: Any = None,
        schema: Any = None,
    ) -> AsyncIterator[dict[str, Any]]:
        stream = await _drive_native(self._inner.chat_stream_struct(messages, tools, schema))
        return _ExecutionJsonStream(stream)

    async def chat_stream_with_tools(
        self,
        messages: Any,
        tools: Any,
        schema: Any = None,
    ) -> AsyncIterator[dict[str, Any]]:
        stream = await _drive_native(self._inner.chat_stream_with_tools(messages, tools, schema))
        return _ExecutionJsonStream(stream)


class ExecutionMemory:
    def __init__(self, inner: _NativeExecutionMemory) -> None:
        self._inner = inner

    def __repr__(self) -> str:
        return repr(self._inner)

    def is_configured(self) -> bool:
        return self._inner.is_configured()

    async def recall(self, query: str, limit: int | None = None) -> list[dict[str, Any]]:
        return await _drive_native(self._inner.recall(query, limit))

    async def remember(self, message: Any) -> None:
        await _drive_native(self._inner.remember(message))

    async def clear(self) -> None:
        await _drive_native(self._inner.clear())

    async def size(self) -> int:
        return await _drive_native(self._inner.size())


__all__ = ["ExecutionLLM", "ExecutionMemory"]
