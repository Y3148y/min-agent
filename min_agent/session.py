"""One session: its messages and its persistence.

A session is a directory::

    .sessions/<user>/<id>/
        meta.json          # id, user, timestamps, last summary
        transcript.jsonl   # append-only message history (plain dicts)
        todo.json          # this window's todo list
        summary.md         # what the compaction step most recently derived

Appending is the only write path for the transcript *while a turn is running*:
it opens the file in "a" mode and writes one line, so a half-written line is
the only thing a crash can produce, and ``_load_transcript`` drops a trailing
partial line.  Three paths rewrite the whole file, and all three go through
:func:`atomic_write_text` -- compaction, turn repair and rollback, each of which
used to truncate first and would have lost the session outright if interrupted.
Everything here is plain ``dict``-shaped so a session can be reloaded without
any SDK imports.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .parser import ParsedTurn
from .paths import atomic_write_text


@dataclass
class SessionMeta:
    id: str
    user: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    turn_count: int = 0
    tool_calls: int = 0
    summary: str = ""


class Session:
    """In-memory state + durable store for one window."""

    def __init__(self, meta: SessionMeta, node_dir: Path):
        self.meta = meta
        self.dir = node_dir
        self.messages: list[dict[str, Any]] = []
        self._load_transcript()

    # -- construction -------------------------------------------------------
    @classmethod
    def create(cls, session_id: str, user: str, node_dir: Path) -> "Session":
        # Sanitise: the id later becomes a directory name, so it must not be a
        # path escape or contain Windows-reserved characters.
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id).strip() or "default"
        node_dir.mkdir(parents=True, exist_ok=True)
        return cls(SessionMeta(id=safe, user=user), node_dir)

    @classmethod
    def load(cls, node_dir: Path) -> "Session":
        meta = _read_meta(node_dir)
        return cls(meta, node_dir)

    # -- persistence -------------------------------------------------------
    def _load_transcript(self) -> None:
        path = self.dir / "transcript.jsonl"
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.messages.append(_json_message(json.loads(line)))
        self.messages = _drop_hanging_tool_refs(self.messages)

    def _save_meta(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self.dir / "meta.json",
            json.dumps(asdict(self.meta), ensure_ascii=False, indent=2),
        )

    def append(self, role: str, content: Any) -> None:
        message = {"role": role, "content": content}
        self.messages.append(message)
        self._append_jsonl(message)
        if role == "user":
            self.meta.turn_count += 1
        self.meta.updated_at = time.time()
        self._save_meta()

    def append_tool_results(self, results: list[dict[str, Any]]) -> None:
        """Append the tool_result turn that answers the prior assistant turn."""
        message = {"role": "user", "content": results}
        self.messages.append(message)
        self._append_jsonl(message)
        self.meta.tool_calls += len(results)
        self.meta.updated_at = time.time()
        self._save_meta()

    def rollback_turn(self, assistant_turn: dict[str, Any]) -> None:
        """Undo the last two messages (assistant + tool results) after a fatal LLM error.

        Keeps the transcript a valid alternating user/assistant sequence even
        when a user turn dies midway -- otherwise the next retry would send
        [user, assistant, user] which some endpoints reject.
        """
        if len(self.messages) >= 2:
            self.messages = self.messages[:-2]
        # Rebuild the file from scratch; transcript files are small by design.
        self._rewrite_jsonl()

    def set_summary(self, summary: str) -> None:
        self.meta.summary = summary
        summary_path = self.dir / "summary.md"
        summary_path.write_text(
            f"# Session {self.meta.id} -- summary\n\n{summary}\n", encoding="utf-8"
        )
        self._save_meta()

    def remember(self, text: str) -> None:
        """Short-term marker used by the concise-memory test: session-level."""
        self.meta.summary = f"{self.meta.summary}\n- RECALLED: {text}".strip()
        self._save_meta()

    def _append_jsonl(self, message: dict[str, Any]) -> None:
        with (self.dir / "transcript.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(message, ensure_ascii=False) + "\n")

    def _rewrite_jsonl(self) -> None:
        atomic_write_text(
            self.dir / "transcript.jsonl",
            "".join(
                json.dumps(message, ensure_ascii=False) + "\n"
                for message in self.messages
            ),
        )

    # -- views --------------------------------------------------------------
    def user_messages(self) -> list[dict[str, Any]]:
        return [m for m in self.messages if m["role"] == "user"]

    def __len__(self) -> int:
        return len(self.messages)


def _read_meta(node_dir: Path) -> SessionMeta:
    path = node_dir / "meta.json"
    if not path.exists():
        return SessionMeta(id=node_dir.name, user="default")
    return SessionMeta(**json.loads(path.read_text(encoding="utf-8")))


def _json_message(message: dict[str, Any]) -> dict[str, Any]:
    """Round-trip safety: content may be a str or a list of block dicts."""
    content = message.get("content")
    if isinstance(content, list):
        message["content"] = [
            {k: v for k, v in block.items()}
            for block in content
            if isinstance(block, dict)
        ]
    return message


def _drop_hanging_tool_refs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Repair transcript artifacts after an interrupted process.

    A `tool_result` pointing at a `tool_use` whose assistant turn was lost
    (crash between the two appends) would make the next request invalid, so we
    drop both halves.  We also drop a trailing assistant turn that was never
    answered (the loop's own rollback, persisted mid-way).
    """
    seen_use_ids: set[str] = set()
    out: list[dict[str, Any]] = []
    for i, message in enumerate(messages):
        content = message.get("content")
        if message["role"] == "assistant" and isinstance(content, list):
            seen_use_ids.update(b.get("id", "") for b in content if b.get("type") == "tool_use")
            out.append(message)
            continue
        if message["role"] == "user" and isinstance(content, list):
            blocks = [b for b in content if b.get("type") != "tool_result" or b.get("tool_use_id") in seen_use_ids]
            if not blocks:
                continue  # nothing left but dangling results -> drop the turn
            message["content"] = blocks
        out.append(message)
    if out and out[-1]["role"] == "assistant" and isinstance(out[-1].get("content"), list):
        if any(b.get("type") == "tool_use" for b in out[-1]["content"]):
            out = out[:-1]
    return out