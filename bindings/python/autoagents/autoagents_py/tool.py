"""@tool decorator: auto-infers JSON schema from Python type hints."""

from __future__ import annotations

import dataclasses
import inspect
import json
from enum import Enum
from typing import (
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from ._core import Tool

from .types import JsonObject

try:
    from types import UnionType as _UnionType  # Python 3.10+
except ImportError:  # pragma: no cover - Python 3.9
    _UnionType = None

# ── Primitive mapping ─────────────────────────────────────────────────────────

_PRIMITIVES: Dict[type, str] = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    bytes: "string",
}


# ── Core type → JSON Schema conversion ───────────────────────────────────────


def _json_schema_for_type(py_type: object) -> JsonObject:
    """Recursively convert a Python type annotation to a JSON Schema dict.

    Handles: primitives, Optional[T], Union[A, B], List[T], Dict[K, V],
    Literal[...], Enum subclasses, dataclasses, Pydantic v1/v2 models.
    """
    if py_type is type(None):
        return {"type": "null"}

    if py_type in _PRIMITIVES:
        return {"type": _PRIMITIVES[py_type]}

    if py_type is list:
        return {"type": "array"}

    if py_type is dict:
        return {"type": "object"}

    origin = get_origin(py_type)
    args = get_args(py_type)

    # Union[T, ...] / Optional[T]
    if _is_union(origin):
        non_none = [a for a in args if a is not type(None)]
        has_none = type(None) in args
        if len(non_none) == 1 and has_none:
            # Optional[T] → {anyOf: [inner, null]}
            return {"anyOf": [_json_schema_for_type(non_none[0]), {"type": "null"}]}
        return {"anyOf": [_json_schema_for_type(a) for a in args]}

    # List[T]
    if origin is list:
        if args:
            return {"type": "array", "items": _json_schema_for_type(args[0])}
        return {"type": "array"}

    # Dict[K, V]
    if origin is dict:
        if len(args) >= 2:
            return {
                "type": "object",
                "additionalProperties": _json_schema_for_type(args[1]),
            }
        return {"type": "object"}

    # Literal["a", "b", ...]
    if origin is Literal:
        return {"enum": list(args)}

    # Pydantic v2
    model_json_schema = cast(
        Optional[Callable[[], JsonObject]],
        getattr(py_type, "model_json_schema", None),
    )
    if callable(model_json_schema):
        try:
            return model_json_schema()
        except Exception:
            pass

    # Pydantic v1
    schema_fn = cast(
        Optional[Callable[[], JsonObject]],
        getattr(py_type, "schema", None),
    )
    if callable(schema_fn):
        try:
            return schema_fn()
        except Exception:
            pass

    # dataclass
    if dataclasses.is_dataclass(py_type) and isinstance(py_type, type):
        return _dataclass_json_schema(py_type)

    # Enum subclass
    if isinstance(py_type, type) and issubclass(py_type, Enum):
        return {"enum": [e.value for e in py_type]}

    return {"type": "string"}


def _is_union(origin: object) -> bool:
    return origin is Union or (_UnionType is not None and origin is _UnionType)


def _dataclass_json_schema(model: type) -> JsonObject:
    """Build a JSON Schema object from a dataclass type."""
    try:
        hints = get_type_hints(model)
    except Exception:
        hints = {}

    props: JsonObject = {}
    required: List[str] = []

    for field in dataclasses.fields(model):  # type: ignore[arg-type]
        props[field.name] = _json_schema_for_type(hints.get(field.name, str))
        if (
            field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING  # type: ignore[misc]
        ):
            required.append(field.name)

    schema: JsonObject = {
        "type": "object",
        "properties": props,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


# ── _infer_schema  ─────────────────────────────────────────────────────────────


def _infer_schema(fn: Callable[..., object]) -> JsonObject:
    """Infer a JSON Schema object from a function's type hints.

    Supports Optional, Union, List, Dict, Literal, Enum, dataclass,
    and Pydantic models. Parameters without annotations default to string.
    """
    try:
        hints = get_type_hints(fn)
    except Exception:
        # Fallback for locally-scoped types that get_type_hints can't resolve.
        hints = dict(getattr(fn, "__annotations__", {}))
    hints.pop("return", None)

    sig = inspect.signature(fn)
    props: JsonObject = {}
    required: List[str] = []

    for name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(
                "AutoAgents tools must use named parameters without *args, **kwargs, "
                "or positional-only arguments"
            )
        py_type = hints.get(name, str)
        props[name] = _json_schema_for_type(py_type)
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: JsonObject = {
        "type": "object",
        "properties": props,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


# ── Callable wrapper ──────────────────────────────────────────────────────────

ToolSyncCallable = Callable[[JsonObject], object]
ToolAsyncCallable = Callable[[JsonObject], Awaitable[object]]
WrappedToolCallable = Union[ToolSyncCallable, ToolAsyncCallable]


def _wrap_callable(fn: Callable[..., object]) -> WrappedToolCallable:
    """Adapt ``fn(**kwargs)`` to the ``fn(args_dict)`` calling convention.

    The Rust `ToolRuntime::execute` passes a single Python dict. This wrapper
    unpacks it into the natural Python signature.
    """
    sig = inspect.signature(fn)
    is_async = inspect.iscoroutinefunction(fn)

    if is_async:
        async_fn = cast(Callable[..., Awaitable[object]], fn)

        async def async_wrapper(args: JsonObject) -> object:
            bound = sig.bind(**args)
            bound.apply_defaults()
            return await async_fn(*bound.args, **bound.kwargs)

        async_wrapper.__name__ = fn.__name__
        return async_wrapper
    else:

        def sync_wrapper(args: JsonObject) -> object:
            bound = sig.bind(**args)
            bound.apply_defaults()
            return fn(*bound.args, **bound.kwargs)

        sync_wrapper.__name__ = fn.__name__
        return sync_wrapper


# ── @tool decorator ───────────────────────────────────────────────────────────


def tool(
    description: str = "",
    name: Optional[str] = None,
    schema: Optional[JsonObject] = None,
) -> "Callable[[Callable[..., object]], Tool]":
    """Decorator that converts a plain Python function into a ``Tool``.

    Args:
        description: Human-readable description passed to the LLM. Falls back
            to the function's docstring if empty.
        name: Tool name visible to the LLM. Defaults to ``fn.__name__``.
        schema: Override JSON Schema dict. Auto-inferred from type hints when
            omitted. Supports Optional, Union, List, Dict, Literal, Enum,
            dataclass, and Pydantic models.

    Examples::

        @tool(description="Add two numbers")
        def add(a: float, b: float) -> float:
            return a + b

        @tool(description="Search within a category")
        def search(query: str, category: Optional[str] = None) -> List[str]:
            ...

        @tool(description="Look up status")
        async def status(mode: Literal["fast", "slow"]) -> dict:
            ...
    """

    def decorator(fn: Callable[..., object]) -> "Tool":
        _name = name or fn.__name__
        _description = description or (fn.__doc__ or "").strip()
        _schema = schema or _infer_schema(fn)
        _callable = _wrap_callable(fn)
        return Tool(_name, _description, json.dumps(_schema), _callable)

    return decorator
