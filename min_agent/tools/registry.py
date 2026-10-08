"""The tool registry: registration, schema exposure, validation, dispatch.

Two invariants this module is responsible for:

1. **The schema list is the single source of truth.**  Whatever is registered
   here is exactly what the model is shown, in the same order, every turn.  No
   separate hand-maintained ``TOOLS`` list can drift.
2. **A tool never crashes the loop.**  Missing tool, bad arguments, raised
   exception, timeout -- all of them become a structured payload that the loop
   feeds back to the model so it can correct itself.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Iterable

from ..errors import ToolError, ToolInputError, ToolNotFoundError, ToolTimeoutError
from .base import ToolSpec

try:
    import jsonschema
except ImportError:  # pragma: no cover - jsonschema is a hard dependency
    jsonschema = None  # type: ignore[assignment]


def _describe_validation_error(exc: Exception) -> str:
    """Turn a jsonschema error into something a model can act on."""
    path = getattr(exc, "absolute_path", None)
    where = ".".join(str(p) for p in path) if path else "(root)"
    return f"argument {where!r}: {getattr(exc, 'message', str(exc))}"


class ToolRegistry:
    """An ordered, name-addressed collection of tools."""

    def __init__(self, specs: Iterable[ToolSpec] = ()):
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            self.register(spec)

    # -- registration -------------------------------------------------------
    def register(self, spec: ToolSpec, *, replace: bool = False) -> ToolSpec:
        if spec.name in self._specs and not replace:
            raise ValueError(f"tool already registered: {spec.name!r}")
        self._specs[spec.name] = spec
        return spec

    def register_fn(self, name: str | None = None, description: str | None = None, **kw):
        """``@registry.register_fn()`` -- same as ``@tool(...)`` but self-registering."""
        from .base import tool as _tool

        def decorate(fn: Callable[..., Any]) -> ToolSpec:
            return self.register(_tool(name=name, description=description, **kw)(fn), replace=True)

        return decorate

    def unregister(self, name: str) -> None:
        self._specs.pop(name, None)

    # -- lookup -------------------------------------------------------------
    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self):
        return iter(self._specs.values())

    def names(self) -> list[str]:
        return list(self._specs)

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise ToolNotFoundError(name, self._specs) from None

    # -- schema exposure ----------------------------------------------------
    def to_api(self) -> list[dict[str, Any]]:
        """The ``tools`` array for the Anthropic Messages API."""
        return [s.to_api() for s in self._specs.values()]

    def catalog(self) -> str:
        """One-line-per-tool digest, used for logging and for the memory prompt."""
        return "\n".join(f"- {s.name}: {s.description}" for s in self._specs.values())

    # -- validation + dispatch ---------------------------------------------
    def validate(self, name: str, args: dict[str, Any]) -> None:
        """Raise :class:`ToolInputError` if ``args`` do not fit the tool's schema."""
        spec = self.get(name)
        if jsonschema is None:  # pragma: no cover
            return
        try:
            jsonschema.validate(args, spec.input_schema)
        except jsonschema.SchemaError as exc:
            # Our schema is broken - a bug in the tool author's annotations,
            # not a mistake by the model.  Raise it loudly instead of telling
            # the model its (possibly correct) args are invalid.
            raise ValueError(f"invalid JSON schema for {name!r}: {exc}") from exc
        except jsonschema.ValidationError as exc:
            expected = spec.required_args()
            if expected:
                hint = f"Required argument(s): {', '.join(expected)}."
            else:
                hint = "This tool takes no arguments."
            raise ToolInputError(
                f"Invalid arguments for {name!r}: {_describe_validation_error(exc)}", hint=hint
            ) from exc

    def call(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
        max_result_chars: int | None = None,
    ) -> dict[str, Any]:
        """Validate, execute, and normalise into ``{"ok", "result"|"error", ...}``.

        Never raises for tool-level problems: the caller is the agent loop, and
        a raised exception there would abort a turn over one bad tool call.
        """
        args = dict(args or {})
        started = time.perf_counter()
        result: dict[str, Any] = {"ok": True, "tool": name, "args": args}

        try:
            self.validate(name, args)
        except ValueError as exc:
            # Broken schema = our bug, not the model's args.  Still returned as
            # data so a mapping session keeps running, but marked as internal so
            # nobody mistakes it for a validation complaint.
            return {**result, "ok": False, "error": f"internal error: {exc}", "latency_ms": _ms(started)}
        except ToolError as exc:
            return {**result, "ok": False, "error": exc.to_model(), "latency_ms": _ms(started)}

        spec = self.get(name)
        try:
            value = self._invoke(spec.fn, args, timeout)
        except ToolError as exc:
            return {**result, "ok": False, "error": exc.to_model(), "latency_ms": _ms(started)}
        except Exception as exc:  # noqa: BLE001 - deliberate catch-all
            return {
                **result,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": _ms(started),
            }

        text = value if isinstance(value, str) else _to_text(value)
        truncated = False
        if max_result_chars is not None and len(text) > max_result_chars:
            text = text[:max_result_chars] + f"\n... [truncated, {len(text)} chars total]"
            truncated = True
        return {
            **result,
            "result": text,
            "truncated": truncated,
            "latency_ms": _ms(started),
        }

    @staticmethod
    def _invoke(fn: Callable[..., Any], args: dict[str, Any], timeout: float | None) -> Any:
        if timeout is None:
            return fn(**args)
        # A Python thread cannot be killed, so this budget bounds *our waiting*,
        # not the tool's execution.  `shutdown(wait=False, cancel_futures=True)`
        # is load-bearing: raising from inside a `with ThreadPoolExecutor(...)`
        # block makes `__exit__` run `shutdown(wait=True)`, which joins the
        # runaway worker -- so a hung tool used to block the agent loop for
        # exactly as long as it cared to.  The orphan keeps running until it
        # returns on its own, and a hard kill would need a process pool.
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout

        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(_call_without_context_exit, fn, args)
            try:
                return future.result(timeout=timeout)
            except FTimeout as exc:
                future.cancel()
                raise ToolTimeoutError(
                    f"{fn.__name__} exceeded its {timeout:g}s budget"
                ) from exc
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    # -- helpers for prompts / tests ---------------------------------------
    def catalog_names(self) -> list[str]:
        return [s.name for s in self._specs.values()]


def _call_without_context_exit(fn: Callable[..., Any], args: dict[str, Any]) -> Any:
    return fn(**args)


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _to_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return str(value)
