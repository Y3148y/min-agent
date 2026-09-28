"""Tool package: the assembled default tool set.

``build_registry(ctx)`` is the single place where the tool pool is constructed
for a session.  Tools that carry per-session state (``todo``) get their state
here, from the session's own directory -- which is what keeps two windows
independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from .base import ToolCall, ToolSpec, tool
from .calculator import calculator
from .read_docs import build_read_docs_tool
from .registry import ToolRegistry
from .search import search
from .todo import TodoStore, build_todo_tool, make_todo_tool
from .weather import weather

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
    "calculator",
    "search",
    "weather",
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


def build_registry(ctx: ToolContext) -> tuple[ToolRegistry, TodoStore]:
    """Assemble the tool pool for one session.

    Returns the registry plus the ``TodoStore`` so the loop can inject a live
    digest of the list into the system prompt without going through a tool call.
    """
    ctx.session_dir.mkdir(parents=True, exist_ok=True)
    todo_spec, todo_store = make_todo_tool(ctx.todo_path)

    registry = ToolRegistry(
        [
            calculator,
            weather,
            search,
            build_read_docs_tool(ctx.config.docs_dir),
            todo_spec,
        ]
    )
    return registry, todo_store
