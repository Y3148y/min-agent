"""``todo`` -- the only stateful tool, and the reason session isolation is visible.

State lives in a JSON file inside the *session's own directory*, so window 1 and
window 2 can both hold a todo list without seeing each other's.  That is not
incidental: the brief's "两个窗口互不影响" is only demonstrable if at least one
tool carries per-session state.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Any, Callable

from ..errors import ToolError
from ..paths import atomic_write_text
from .. import textutil
from .base import ToolSpec, tool

_STATUS = ("pending", "done")
_ACTIONS = ("add", "list", "done", "remove", "clear")


@dataclass
class TodoItem:
    id: int
    text: str
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    done_at: float | None = None


class TodoStore:
    """Per-session todo list, persisted as JSON."""

    def __init__(self, path: Path):
        self.path = path
        self.items: list[TodoItem] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupt todo file must not take the session down: the todo list
            # is a convenience, the conversation is the product.
            self.items = []
            return
        self.items = []
        for row in raw:
            if not isinstance(row, dict) or "id" not in row or "text" not in row:
                continue  # a row from another schema -- skip, don't crash
            try:
                self.items.append(TodoItem(**row))
            except TypeError:
                continue  # a future/partial row -- keep the rest of the list

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self.path,
            json.dumps([asdict(i) for i in self.items], ensure_ascii=False, indent=2),
        )

    def add(self, text: str) -> TodoItem:
        """Add a todo, idempotently on the normalised text.

        The model calls tools it decided to call, and it re-calls them: a turn
        that hit max_tokens mid-answer, a tool that returned is_error, a repeat
        the loop guard let through.  `add` used to mint `max(id)+1` every time,
        so every retry silently duplicated the entry.  The old one stays and is
        returned instead, which makes a retry harmless and keeps the id stable
        for a `done` that follows it.

        Normalising (whitespace collapsed, casefolded) rather than comparing
        raw means "买 菜" and "买菜" are one item.  A user who genuinely wants
        the same text twice can disambiguate in the text itself.
        """
        key = _normalise(text)
        existing = next((i for i in self.items if _normalise(i.text) == key), None)
        if existing is not None:
            return existing
        nid = max((i.id for i in self.items), default=0) + 1
        item = TodoItem(id=nid, text=text)
        self.items.append(item)
        self.save()
        return item

    def list_items(self, status: str | None = None) -> list[TodoItem]:
        return [i for i in self.items if status is None or i.status == status]

    def complete(self, todo_id: int | None = None, text_contains: str = "") -> TodoItem:
        candidates = [i for i in self.items if i.status == "pending"]
        if todo_id is not None:
            match = next((i for i in candidates if i.id == todo_id), None)
            if match is None:
                known = ", ".join(str(i.id) for i in self.items) or "(none)"
                raise ToolError(
                    f"No pending todo with id {todo_id}",
                    hint=f"Existing ids: {known}.",
                )
        elif text_contains:
            needle = text_contains.strip().lower()
            match = next((i for i in candidates if needle in i.text.lower()), None)
            if match is None:
                raise ToolError(
                    f"No pending todo matching {text_contains!r}",
                    hint="Call todo(action='list') to see what is open.",
                )
        else:
            raise ToolError("Complete which item?", hint="Pass id= or text_contains=")
        match.status = "done"
        match.done_at = time.time()
        self.save()
        return match

    def remove(self, todo_id: int) -> TodoItem:
        match = next((i for i in self.items if i.id == todo_id), None)
        if match is None:
            raise ToolError(f"No todo with id {todo_id}")
        self.items.remove(match)
        self.save()
        return match

    def clear(self) -> int:
        n = len(self.items)
        self.items = []
        self.save()
        return n

    def digest(self) -> str:
        """One line per item, for injection into the system prompt."""
        if not self.items:
            return "(empty)"
        return "\n".join(f"[{i.id}] {'x' if i.status == 'done' else ' '} {i.text}" for i in self.items)


def build_todo_tool(store: TodoStore) -> ToolSpec:
    """Create the ``todo`` tool bound to ``store``.

    Closure over the store rather than a module global: two windows in the same
    process would otherwise share one list.
    """

    @tool(
        tags=("state",),
        description=(
            "Manage the session's to-do list. action='add' creates an item; 'list' shows the "
            "list; 'done' completes one (by id or by keyword); 'remove' deletes one; 'clear' "
            "empties the list. Call this whenever the user asks to note something down, track "
            "work, or check what is still outstanding."
        ),
    )
    def todo(
        action: Annotated[str, f"one of: {', '.join(_ACTIONS)}"],
        item: Annotated[str, "the text of the new item (required for action='add')"] = "",
        id: Annotated[int, "id of an existing item (for action='done' or 'remove')"] = 0,
        text_contains: Annotated[str, "substring to match an item by (alternative to id)"] = "",
    ) -> str:
        action = action.strip().lower()
        if action not in _ACTIONS:
            raise ToolError(
                f"Unknown action {action!r}",
                hint=f"Use one of: {', '.join(_ACTIONS)}.",
            )
        if action == "add":
            text = item.strip()
            if not text:
                raise ToolError("action='add' needs item=<text>")
            created = store.add(text)
            return f"Added todo #{created.id}: {created.text}\n\nOpen list:\n{store.digest()}"
        if action == "list":
            status_filter = "done" if text_contains.strip().lower() == "done" else None
            rows: list[dict[str, Any]] = [asdict(i) for i in store.list_items(status_filter)]
            pending = len(store.list_items("pending"))
            return json.dumps(
                {"pending": pending, "total": len(store.items), "items": rows},
                ensure_ascii=False,
                indent=2,
            )
        if action == "done":
            done = store.complete(id or None, text_contains)
            return f"Completed #{done.id}: {done.text}\n\nOpen list:\n{store.digest()}"
        if action == "remove":
            gone = store.remove(id)
            return f"Removed #{gone.id}: {gone.text}"
        return f"Cleared {store.clear()} item(s)."

    return todo


def make_todo_tool(path: Path) -> tuple[ToolSpec, TodoStore]:
    store = TodoStore(path)
    return build_todo_tool(store), store


def _normalise(text: str) -> str:
    """Same key as textutil.normalise, which MemoryStore uses for its dedup, so
    the two agree on what counts as "the same text"."""
    return textutil.normalise(text)
