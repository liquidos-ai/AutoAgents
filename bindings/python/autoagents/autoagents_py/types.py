"""Shared typing contracts for AutoAgents Python bindings."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    TypedDict,
    Union,
    runtime_checkable,
)

if TYPE_CHECKING:
    from .events import ProtocolEvent

JsonPrimitive = Union[str, int, float, bool, None]
JsonValue = Union[JsonPrimitive, Dict[str, "JsonValue"], Sequence["JsonValue"]]
JsonObject = Dict[str, JsonValue]
HookOutcomeLike = str


class TaskImagePayload(TypedDict):
    mime: str
    data: bytes


class TaskPayload(TypedDict, total=False):
    prompt: str
    system_prompt: str
    image: TaskImagePayload


class ExecutorTask(TypedDict, total=False):
    prompt: str
    system_prompt: Optional[str]
    submission_id: str
    completed: bool
    result_json: Optional[str]


class ChatMessagePayload(TypedDict, total=False):
    role: str
    message_type: str
    type: str
    content: str
    text: str


class LLMChatResponse(TypedDict, total=False):
    text: Optional[str]
    tool_calls: Optional[List[JsonObject]]
    thinking: Optional[str]
    usage: Optional[JsonObject]


@runtime_checkable
class ExecutionLLMProtocol(Protocol):
    async def chat(
        self,
        messages: Union[str, ChatMessagePayload, Sequence[ChatMessagePayload]],
        schema: Optional[Union[str, JsonObject]] = None,
    ) -> LLMChatResponse: ...
    async def chat_with_struct(
        self,
        messages: Union[str, ChatMessagePayload, Sequence[ChatMessagePayload]],
        schema: Union[str, JsonObject],
    ) -> LLMChatResponse: ...
    async def chat_with_tools(
        self,
        messages: Union[str, ChatMessagePayload, Sequence[ChatMessagePayload]],
        tools: Sequence[object],
        schema: Optional[Union[str, JsonObject]] = None,
    ) -> LLMChatResponse: ...
    async def chat_with_tools_struct(
        self,
        messages: Union[str, ChatMessagePayload, Sequence[ChatMessagePayload]],
        tools: Sequence[object],
        schema: Union[str, JsonObject],
    ) -> LLMChatResponse: ...
    async def chat_with_web_search(self, input: str) -> LLMChatResponse: ...
    async def chat_stream(
        self,
        messages: Union[str, ChatMessagePayload, Sequence[ChatMessagePayload]],
        schema: Optional[Union[str, JsonObject]] = None,
    ) -> AsyncIterator[str]: ...
    async def chat_stream_struct(
        self,
        messages: Union[str, ChatMessagePayload, Sequence[ChatMessagePayload]],
        tools: Optional[Sequence[object]] = None,
        schema: Optional[Union[str, JsonObject]] = None,
    ) -> AsyncIterator[JsonObject]: ...
    async def chat_stream_with_tools(
        self,
        messages: Union[str, ChatMessagePayload, Sequence[ChatMessagePayload]],
        tools: Sequence[object],
        schema: Optional[Union[str, JsonObject]] = None,
    ) -> AsyncIterator[JsonObject]: ...


@runtime_checkable
class ExecutionMemoryProtocol(Protocol):
    def is_configured(self) -> bool: ...

    async def recall(
        self,
        query: str,
        limit: Optional[int] = None,
    ) -> List[ChatMessagePayload]: ...

    async def remember(self, message: Union[str, ChatMessagePayload]) -> None: ...
    async def clear(self) -> None: ...
    async def size(self) -> int: ...


class ExecutorContext(TypedDict):
    id: str
    name: str
    description: str
    stream: bool
    messages: List[ChatMessagePayload]
    llm: ExecutionLLMProtocol
    memory: ExecutionMemoryProtocol


class ExecutorConfig(TypedDict, total=False):
    max_turns: int


class CodeActExecutionRecord(TypedDict):
    execution_id: str
    source: str
    console: List[str]
    tool_calls: List[JsonObject]
    result: Optional[JsonValue]
    success: bool
    error: Optional[str]
    duration_ms: int


class ExecutorOutput(TypedDict):
    response: str
    done: bool
    tool_calls: List[JsonObject]
    executions: List[CodeActExecutionRecord]


class AgentRunResult(TypedDict):
    response: str
    tool_calls: List[JsonObject]
    executions: List[CodeActExecutionRecord]
    done: bool
    events: List["ProtocolEvent"]


@runtime_checkable
class HooksProtocol(Protocol):
    async def on_agent_create(self) -> None: ...
    async def on_run_start(
        self,
        task: ExecutorTask,
        ctx: ExecutorContext,
    ) -> HookOutcomeLike: ...
    async def on_run_complete(
        self,
        task: ExecutorTask,
        result: ExecutorOutput,
        ctx: ExecutorContext,
    ) -> None: ...
    async def on_turn_start(self, turn_index: int, ctx: ExecutorContext) -> None: ...
    async def on_turn_complete(self, turn_index: int, ctx: ExecutorContext) -> None: ...
    async def on_tool_call(
        self,
        tool_call: JsonObject,
        ctx: ExecutorContext,
    ) -> HookOutcomeLike: ...
    async def on_tool_start(self, tool_call: JsonObject, ctx: ExecutorContext) -> None: ...
    async def on_tool_result(
        self,
        tool_call: JsonObject,
        result: JsonObject,
        ctx: ExecutorContext,
    ) -> None: ...
    async def on_tool_error(
        self,
        tool_call: JsonObject,
        err: JsonObject,
        ctx: ExecutorContext,
    ) -> None: ...
    async def on_agent_shutdown(self) -> None: ...


@runtime_checkable
class ExecutorProtocol(Protocol):
    def config(self) -> ExecutorConfig: ...
    async def execute(
        self, task: ExecutorTask, ctx: ExecutorContext
    ) -> ExecutorOutput: ...
    def execute_stream(
        self,
        task: ExecutorTask,
        ctx: ExecutorContext,
    ) -> AsyncIterator[ExecutorOutput]: ...
