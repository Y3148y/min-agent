"""Tool package: the assembled default tool set.

``build_registry(ctx)`` is the single place where the tool pool is constructed
for a session.  Tools that carry per-session state (``todo``) or long-term
memory (``remember``) get their backend here, from the session's own directory
-- which is what keeps two windows independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from ..config import Config
from . import search  # noqa: F401  -- the submodule, not its ToolSpec
from .base import ToolCall, ToolSpec, tool
from .calculator import calculator
from .read_docs import build_read_docs_tool
from .registry import ToolRegistry
from .todo import TodoStore, build_todo_tool, make_todo_tool
from .weather import make_weather_tool, weather

__all__ = [
    "ToolCall",
    "ToolSpec",
    "tool",
    "ToolRegistry",
    "ToolContext",
    "TodoStore",
    "build_registry",
    "build_read_docs_tool",
    "build_todo_tool",
    "build_remember_tool",
    "calculator",
    "search",
    "weather",
    "make_weather_tool",
]


@dataclass
class ToolContext:
    """Everything a tool may need from its host session."""

    session_id: str
    session_dir: Path
    config: Config

    @property
    def todo_path(self) -> Path:
        return self.session_dir / "todo.json"


def build_remember_tool(
    memory: Any,
    source_session: str,
) -> ToolSpec:
    """The ``remember`` tool: a door from the model into long-term memory.

    This is one of the two store triggers (the other is the end-of-session
    extraction sweep).  The model decides when a statement is worth keeping;
    the MemoryStore's own gating decides whether it *gets* kept.
    """

    @tool(
        tags=("memory",),
        description=(
            "Save a stable fact or preference the user just told you into long-term memory "
            "shared across all of this user's windows, e.g. \"我住在北京\" or \"讨厌被叫英文名\". "
            "Call it when the user states something that will still matter next week. Do not save "
            "one-off tasks or answers -- the short transcript already covers those."
        ),
    )
    def remember(
        fact: Annotated[str, "the durable fact, exactly as the user stated it"],
    ) -> str:
        result = memory.remember(fact, source_session=source_session)
        return result if result else "(rejected: not a durable fact, or already known)"

    return remember


def build_registry(ctx: ToolContext, memory: Any = None) -> tuple[ToolRegistry, TodoStore]:
    """Assemble the tool pool for one session.

    Returns the registry plus the ``TodoStore`` so the loop can inject a live
    digest of the list into the system prompt without going through a tool call.
    """
    ctx.session_dir.mkdir(parents=True, exist_ok=True)
    todo_spec, todo_store = make_todo_tool(ctx.todo_path)

    specs: list[ToolSpec] = [
        calculator,
        make_weather_tool(ctx.config.weather_backend),  # mock (default) or wttr.in
        search.search,
        build_read_docs_tool(ctx.config.docs_dir),
        todo_spec,
    ]
    if memory is not None:
        specs.append(build_remember_tool(memory, ctx.session_id))

    registry = ToolRegistry(specs)
    return registry, todo_store