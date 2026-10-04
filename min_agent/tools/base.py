"""Tool declaration: the ``@tool`` decorator and :class:`ToolSpec`.

A tool is *data* (name + description + JSON Schema) plus *behaviour* (a
function). Keeping them in one object is what lets the registry hand the
schema list to the model and dispatch the call without any if/elif chain in the
loop.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

# JSON Schema dialect we emit. Kept deliberately small: the subset every tool
# actually needs. Anything fancier (oneOf, $ref, ...) is a schema the model will
# get wrong, not one it needs.
_OBJECT = "object"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., Any]
    tags: tuple[str, ...] = ()

    def to_api(self) -> dict[str, Any]:
        """Render the Anthropic Messages ``tools`` entry for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def required_args(self) -> list[str]:
        return list(self.input_schema.get("required", []))


@dataclass
class ToolCall:
    """One requested invocation, already parsed out of the model output."""

    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)

    def signature(self) -> str:
        """Stable identity for a call, used to detect looping."""
        import json

        return f"{self.name}:{json.dumps(self.args, sort_keys=True, ensure_ascii=False, default=str)}"


def _schema_from_signature(fn: Callable[..., Any]) -> dict[str, Any]:
    """Derive a JSON Schema from the function signature.

    Python type hints map onto JSON Schema types. ``Annotated[str, "the city"]``
    supplies the per-argument description, which keeps schema and docstring
    from drifting apart.
    """
    props: dict[str, Any] = {}
    required: list[str] = []
    sig = inspect.signature(fn)
    try:
        hints = _resolved_hints(fn)
    except Exception:  # pragma: no cover - exotic annotations
        hints = {}

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls") or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        # `from __future__ import annotations` makes param.annotation a *string*,
        # so the resolved hint (which is a real object again) is the only place
        # Annotated[...] metadata can be read from.
        annotation = hints.get(pname, param.annotation)
        entry: dict[str, Any] = _schema_for(annotation)
        desc = _annotated_description(annotation)
        if desc:
            entry["description"] = desc
        if param.default is not inspect.Parameter.empty:
            entry["default"] = param.default
        else:
            required.append(pname)
        props[pname] = entry

    schema: dict[str, Any] = {"type": _OBJECT, "properties": props}
    if required:
        schema["required"] = required
    return schema


def _resolved_hints(fn: Callable[..., Any]) -> dict[str, Any]:
    import typing

    return typing.get_type_hints(fn, include_extras=True)


def _annotated_description(annotation: Any) -> str:
    """Pull the description out of ``Annotated[T, "..."]``."""
    if typing_is_annotated(annotation):
        for meta in getattr(annotation, "__metadata__", ()):
            if isinstance(meta, str):
                return meta
    return ""


def typing_is_annotated(annotation: Any) -> bool:
    return hasattr(annotation, "__metadata__") and hasattr(annotation, "__origin__")


def _schema_for(python_type: Any) -> dict[str, Any]:
    """Map one Python annotation onto a JSON Schema map.

    Handles containers (``list[str]`` -> ``{"type":"array","items":...}``)
    because the old code emitted the string ``"array[string]"``, which the
    json-schema validator refuses to compile.  ``dict[str, X]`` also carries
    ``additionalProperties``.  ``Annotated`` and ``Optional`` are unwrapped.
    """
    if typing_is_annotated(python_type):
        args = [a for a in getattr(python_type, "__args__", ()) if a is not type(None)]
        return _schema_for(args[0]) if args else {"type": "string"}

    origin = getattr(python_type, "__origin__", None)
    if origin is not None:
        args = [a for a in getattr(python_type, "__args__", ()) if a is not type(None)]
        if origin in (list, tuple, set):
            item = _schema_for(args[0]) if args and args[0] is not Any else {"type": "string"}
            return {"type": "array", "items": item}
        if origin is dict:
            value = args[1] if len(args) > 1 and args[1] is not Any else None
            if value is not None:
                return {"type": "object", "additionalProperties": _schema_for(value)}
            return {"type": "object"}
        if args:
            return _schema_for(args[0])
        return {"type": "string"}

    if python_type is bool:
        return {"type": "boolean"}
    if python_type is int:
        return {"type": "integer"}
    if python_type is float:
        return {"type": "number"}
    if python_type is str:
        return {"type": "string"}
    if python_type in (list, tuple, set):
        return {"type": "array"}
    if python_type is dict:
        return {"type": "object"}
    if python_type is type(None):
        return {"type": "null"}
    return {"type": "string"}


def tool(
    name: str | None = None,
    description: str | None = None,
    *,
    schema: dict[str, Any] | None = None,
    tags: tuple[str, ...] = (),
) -> Callable[[Callable[..., Any]], ToolSpec]:
    """Turn a function into a :class:`ToolSpec`.

    Usage::

        from typing import Annotated

        @tool(tags=("math",))
        def calculator(expression: Annotated[str, "the arithmetic to evaluate"]) -> str:
            ...

    The description is taken from the docstring when not given, because the
    description is the only thing the model uses to decide *whether* to call the
    tool -- it is prompt, not documentation.
    """

    def decorate(fn: Callable[..., Any]) -> ToolSpec:
        spec = ToolSpec(
            name=name if isinstance(name, str) else fn.__name__,
            description=(description or inspect.getdoc(fn) or fn.__name__).strip(),
            input_schema=schema if schema is not None else _schema_from_signature(fn),
            fn=fn,
            tags=tags,
        )
        return spec

    # Support both `@tool` (bare) and `@tool(name="...")`.
    if callable(name) and description is None:
        return decorate(name)

    return decorate
