"""Unit tests for the autoagents Python bindings (no real API calls)."""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from typing import Dict, List, Literal, Optional, Union

import pytest
import autoagents_py as aa
import autoagents_py.memory as memory_module
from autoagents_py import experimental as experimental_api
from autoagents_py.prebuilt import BasicAgent, SlidingWindowMemory
from autoagents_py.agent import AgentBuilder, _infer_output_schema
from autoagents_py.exceptions import (
    AgentBuildError,
    ExperimentalFeatureError,
    AgentRunError,
    AgentTimeoutError,
    AutoAgentsError,
)
from autoagents_py.tool import _infer_schema, _json_schema_for_type, _wrap_callable, tool
from autoagents_py.types import AgentRunResult


class _Status(Enum):
    OK = "ok"
    ERROR = "error"


class TestJsonSchemaForType:
    def test_primitives(self):
        assert _json_schema_for_type(int) == {"type": "integer"}
        assert _json_schema_for_type(float) == {"type": "number"}
        assert _json_schema_for_type(str) == {"type": "string"}
        assert _json_schema_for_type(bool) == {"type": "boolean"}

    def test_none_type(self):
        assert _json_schema_for_type(type(None)) == {"type": "null"}

    def test_optional(self):
        schema = _json_schema_for_type(Optional[str])
        assert schema == {"anyOf": [{"type": "string"}, {"type": "null"}]}

    def test_optional_int(self):
        schema = _json_schema_for_type(Optional[int])
        assert schema == {"anyOf": [{"type": "integer"}, {"type": "null"}]}

    def test_list_typed(self):
        schema = _json_schema_for_type(List[str])
        assert schema == {"type": "array", "items": {"type": "string"}}

    def test_list_nested(self):
        schema = _json_schema_for_type(List[int])
        assert schema["items"] == {"type": "integer"}

    def test_list_untyped(self):
        schema = _json_schema_for_type(list)
        assert schema == {"type": "array"}

    def test_dict_typed(self):
        schema = _json_schema_for_type(Dict[str, int])
        assert schema == {"type": "object", "additionalProperties": {"type": "integer"}}

    def test_dict_untyped(self):
        schema = _json_schema_for_type(dict)
        assert schema == {"type": "object"}

    def test_literal(self):
        schema = _json_schema_for_type(Literal["fast", "slow"])
        assert schema == {"enum": ["fast", "slow"]}

    def test_literal_mixed(self):
        schema = _json_schema_for_type(Literal[1, 2, 3])
        assert schema == {"enum": [1, 2, 3]}

    def test_union(self):
        schema = _json_schema_for_type(Union[str, int])
        assert schema == {"anyOf": [{"type": "string"}, {"type": "integer"}]}

    def test_enum(self):
        schema = _json_schema_for_type(_Status)
        assert schema == {"enum": ["ok", "error"]}

    def test_dataclass(self):
        @dataclasses.dataclass
        class Point:
            x: float
            y: float
            label: Optional[str] = None

        schema = _json_schema_for_type(Point)
        assert schema["type"] == "object"
        assert schema["properties"]["x"] == {"type": "number"}
        assert schema["properties"]["y"] == {"type": "number"}
        assert schema["properties"]["label"] == {
            "anyOf": [{"type": "string"}, {"type": "null"}]
        }
        assert "x" in schema["required"]
        assert "y" in schema["required"]
        assert "label" not in schema["required"]

    def test_list_of_optional(self):
        schema = _json_schema_for_type(List[Optional[str]])
        assert schema["items"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}


class TestInferSchema:
    def test_basic(self):
        def add(a: float, b: float) -> float:
            return a + b

        schema = _infer_schema(add)
        assert schema["type"] == "object"
        assert schema["properties"]["a"] == {"type": "number"}
        assert schema["properties"]["b"] == {"type": "number"}
        assert "a" in schema["required"]
        assert "b" in schema["required"]

    def test_optional_param(self):
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}"

        schema = _infer_schema(greet)
        assert "name" in schema["required"]
        assert "greeting" not in schema.get("required", [])

    def test_no_hints(self):
        def mystery(x, y):
            return x + y

        schema = _infer_schema(mystery)
        assert schema["properties"]["x"]["type"] == "string"
        assert schema["properties"]["y"]["type"] == "string"

    def test_optional_param_schema(self):
        def fn(q: str, limit: Optional[int] = None) -> list:
            return []

        schema = _infer_schema(fn)
        assert schema["properties"]["limit"] == {
            "anyOf": [{"type": "integer"}, {"type": "null"}]
        }

    def test_list_param(self):
        def fn(tags: List[str]) -> None:
            pass

        schema = _infer_schema(fn)
        assert schema["properties"]["tags"] == {
            "type": "array",
            "items": {"type": "string"},
        }

    def test_literal_param(self):
        def fn(mode: Literal["fast", "slow"]) -> None:
            pass

        schema = _infer_schema(fn)
        assert schema["properties"]["mode"] == {"enum": ["fast", "slow"]}

    def test_enum_param(self):
        def fn(status: _Status) -> None:
            pass

        schema = _infer_schema(fn)
        assert schema["properties"]["status"] == {"enum": ["ok", "error"]}

    def test_return_excluded(self):
        def fn(x: int) -> str:
            return str(x)

        schema = _infer_schema(fn)
        assert "return" not in schema["properties"]


class TestWrapCallable:
    def test_sync(self):
        def add(a: float, b: float) -> float:
            return a + b

        wrapper = _wrap_callable(add)
        assert wrapper({"a": 3.0, "b": 4.0}) == 7.0

    @pytest.mark.asyncio
    async def test_async(self):
        async def fetch(url: str) -> dict:
            return {"url": url, "status": 200}

        wrapper = _wrap_callable(fetch)
        result = await wrapper({"url": "https://example.com"})
        assert result["status"] == 200

    def test_extra_keys_raise(self):
        def fn(a: int) -> int:
            return a

        wrapper = _wrap_callable(fn)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            wrapper({"a": 5, "ignored": "x"})


class TestToolDecorator:
    def test_creates_tool(self):
        @tool(description="Multiply")
        def multiply(a: float, b: float) -> float:
            return a * b

        assert isinstance(multiply, aa.Tool)

    def test_name_from_function(self):
        @tool(description="No-op")
        def noop() -> None:
            pass

        assert repr(noop) == "Tool(name='noop')"

    def test_custom_name(self):
        @tool(description="Add", name="addition")
        def add(a: int, b: int) -> int:
            return a + b

        assert repr(add) == "Tool(name='addition')"

    def test_docstring_fallback(self):
        @tool()
        def square(n: float) -> float:
            """Square a number."""
            return n * n

        assert isinstance(square, aa.Tool)

    def test_schema_has_correct_types(self):
        @tool(description="Search")
        def search(query: str, limit: Optional[int] = 10) -> List[str]:
            return []

        assert isinstance(search, aa.Tool)

    def test_literal_in_tool(self):
        @tool(description="Mode select")
        def run(mode: Literal["fast", "slow"]) -> str:
            return mode

        assert isinstance(run, aa.Tool)

    def test_varargs_are_rejected(self):
        with pytest.raises(TypeError, match="named parameters"):
            @tool(description="Bad")
            def bad(*args: int) -> int:
                return sum(args)


class TestOutputSchema:
    def test_infer_output_schema_dataclass(self):
        @dataclasses.dataclass
        class Result:
            answer: str
            confidence: float

        schema = _infer_output_schema(Result)
        assert schema["type"] == "object"
        assert schema["properties"]["answer"] == {"type": "string"}
        assert schema["properties"]["confidence"] == {"type": "number"}

    def test_output_model_registers_schema_on_builder(self):
        @dataclasses.dataclass
        class Out:
            value: int

        builder = AgentBuilder(BasicAgent("x", "y")).output(Out)
        schema_text = builder._output_schema
        assert isinstance(schema_text, str)
        schema = json.loads(schema_text)
        assert schema["name"] == "Out"
        assert schema["strict"] is True

    def test_agent_run_result_has_no_python_structured_output_field(self):
        annotations = AgentRunResult.__annotations__
        assert "structured_output" not in annotations


class TestLLMBuilder:
    def test_unknown_backend(self):
        with pytest.raises(Exception, match="Unknown LLM backend"):
            aa.LLMBuilder("nonexistent_backend_xyz").build()

    def test_missing_api_key_openai(self):
        with pytest.raises(Exception):
            aa.LLMBuilder("openai").model("gpt-4o").build()


class TestPyTool:
    def test_invalid_schema(self):
        with pytest.raises(Exception):
            aa.Tool("bad", "desc", "NOT VALID JSON", lambda args: None)

    def test_valid_construction(self):
        schema = json.dumps(
            {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            }
        )
        t = aa.Tool("my_tool", "Does something", schema, lambda args: {"result": args["x"] * 2})
        assert repr(t) == "Tool(name='my_tool')"


class TestExceptions:
    def test_hierarchy(self):
        assert issubclass(AgentBuildError, AutoAgentsError)
        assert issubclass(AgentRunError, AutoAgentsError)
        assert issubclass(AgentTimeoutError, AgentRunError)

    def test_build_error_is_runtime_error(self):
        assert issubclass(AgentBuildError, RuntimeError)


class TestLLMProviderCoercion:
    def test_unknown_provider_is_rejected(self):
        with pytest.raises(TypeError, match="expected an AutoAgents LLMProvider"):
            aa._core._coerce_llm_provider(object())

    def test_agent_builder_coerces_provider_before_build(self, monkeypatch):
        sentinel_provider = object()

        class FakeCoreBuilder:
            def __init__(self, agent):
                self.agent = agent
                self.llm_provider = None
                self.memory_provider = None

            def llm(self, provider):
                self.llm_provider = provider
                return self

            def memory(self, provider):
                self.memory_provider = provider
                return self

        monkeypatch.setattr("autoagents_py.agent._CoreAgentBuilder", FakeCoreBuilder)
        monkeypatch.setattr(
            "autoagents_py.agent._coerce_llm_provider",
            lambda provider: sentinel_provider,
        )

        builder = AgentBuilder(BasicAgent("x", "y")).llm(object()).memory(
            SlidingWindowMemory()
        )
        rust_builder = builder._make_rust_builder()

        assert builder._llm is sentinel_provider
        assert rust_builder.llm_provider is sentinel_provider


class TestAgentBuilderExecutorProtocol:
    def test_stable_builder_rejects_custom_executor_spec(self):
        class CustomMathExecutor:
            def config(self):
                return {"max_turns": 2}

            async def execute(self, task, ctx):
                return {"response": str(task.get("prompt", "")), "done": True, "tool_calls": []}

            async def execute_stream(self, task, ctx):
                yield {"response": str(task.get("prompt", "")), "done": True, "tool_calls": []}

        with pytest.raises(ExperimentalFeatureError, match="CustomExecutor is experimental"):
            aa.AgentBuilder(
                experimental_api.CustomExecutor(
                    "wrapped",
                    "Wrapped executor",
                    CustomMathExecutor(),
                )
            )

    def test_custom_executor_requires_config(self):
        class MissingConfigExecutor:
            async def execute(self, task, ctx):
                return {"response": "ok", "done": True, "tool_calls": []}

            async def execute_stream(self, task, ctx):
                yield {"response": "ok", "done": True, "tool_calls": []}

        with pytest.raises(TypeError, match="config"):
            experimental_api.CustomExecutor(
                "wrapped",
                "Wrapped executor",
                MissingConfigExecutor(),
            )


class TestMemoryProvider:
    def test_stable_memory_apply_rejects_python_backed_memory(self):
        class DemoMemory:
            async def remember(self, message):
                return None

            async def recall(self, query, limit=None):
                return []

            async def clear(self):
                return None

            def size(self):
                return 0

        class FakeBuilder:
            def memory(self, mem):
                return self

        with pytest.raises(ExperimentalFeatureError, match="Python-backed memory adapters"):
            memory_module.apply_memory_provider(FakeBuilder(), DemoMemory())

    def test_experimental_memory_apply_uses_bridge(self, monkeypatch):
        calls = []

        class DemoMemory:
            async def remember(self, message):
                return None

            async def recall(self, query, limit=None):
                return []

            async def clear(self):
                return None

            def size(self):
                return 0

        class FakeBuilder:
            def memory(self, mem):
                calls.append(mem)
                return self

        sentinel = object()
        monkeypatch.setattr(
            "autoagents_py.experimental.memory.memory_provider_from_impl",
            lambda mem: sentinel,
        )

        experimental_api.apply_memory_provider(FakeBuilder(), DemoMemory())
        assert calls == [sentinel]

    def test_stable_agent_builder_rejects_plain_memory_objects(self, monkeypatch):
        calls = []

        class DemoMemory:
            async def remember(self, message):
                return None

            async def recall(self, query, limit=None):
                return []

            async def clear(self):
                return None

            def size(self):
                return 0

        class FakeCoreBuilder:
            def __init__(self, agent):
                self.agent = agent

            def llm(self, provider):
                return self

            def memory(self, mem):
                calls.append(mem)
                return self

        sentinel = object()
        monkeypatch.setattr("autoagents_py.agent._CoreAgentBuilder", FakeCoreBuilder)
        monkeypatch.setattr("autoagents_py.agent._coerce_llm_provider", lambda provider: provider)
        monkeypatch.setattr(
            "autoagents_py.experimental.memory.memory_provider_from_impl",
            lambda mem: sentinel,
        )

        builder = AgentBuilder(BasicAgent("x", "y")).llm(object()).memory(DemoMemory())
        with pytest.raises(ExperimentalFeatureError, match="Python-backed memory adapters"):
            builder._make_rust_builder()
        assert calls == []

    def test_experimental_agent_builder_accepts_plain_memory_objects(self, monkeypatch):
        calls = []

        class DemoMemory:
            async def remember(self, message):
                return None

            async def recall(self, query, limit=None):
                return []

            async def clear(self):
                return None

            def size(self):
                return 0

        class FakeCoreBuilder:
            def __init__(self, agent):
                self.agent = agent

            def llm(self, provider):
                return self

            def memory(self, mem):
                calls.append(mem)
                return self

        sentinel = object()
        monkeypatch.setattr("autoagents_py.agent._CoreAgentBuilder", FakeCoreBuilder)
        monkeypatch.setattr("autoagents_py.agent._coerce_llm_provider", lambda provider: provider)
        monkeypatch.setattr(
            "autoagents_py.experimental.memory.memory_provider_from_impl",
            lambda mem: sentinel,
        )

        experimental_api.ExperimentalAgentBuilder(BasicAgent("x", "y")).llm(
            object()
        ).memory(DemoMemory())._make_rust_builder()
        assert calls == [sentinel]

    def test_sliding_window_memory_requires_positive_window(self):
        with pytest.raises(ValueError, match="window_size must be > 0"):
            SlidingWindowMemory(window_size=0)


class TestPipelineBuilder:
    def test_add_layer_applies_in_insertion_order(self, monkeypatch):
        calls = []

        class TagLayer:
            def __init__(self, name):
                self.name = name

            def build(self, next_provider):
                calls.append((self.name, next_provider))
                return f"{self.name}({next_provider})"

        provider = "base"
        monkeypatch.setattr("autoagents_py.pipeline._coerce_llm_provider", lambda provider: provider)
        built = (
            aa.PipelineBuilder(provider)
            .add_layer(TagLayer("outer"))
            .add_layer(TagLayer("inner"))
            .build()
        )

        assert built == "outer(inner(base))"
        assert calls == [("inner", "base"), ("outer", "inner(base)")]

    def test_custom_pipeline_layer_uses_python_bridge(self, monkeypatch):
        sentinel = object()
        calls = []

        class DemoLayer(aa.PipelineLayer):
            pass

        monkeypatch.setattr(
            "autoagents_py.pipeline.pipeline_python_layer",
            lambda provider, layer: calls.append((provider, layer)) or sentinel,
        )
        monkeypatch.setattr(
            "autoagents_py.pipeline._coerce_llm_provider",
            lambda provider: provider,
        )

        built = DemoLayer().build("base")
        assert built is sentinel
        assert calls and calls[0][0] == "base"


class TestTopic:
    def test_creation(self):
        topic = aa.Topic("my-tasks")
        assert topic.name == "my-tasks"
        assert repr(topic) == "Topic('my-tasks')"
