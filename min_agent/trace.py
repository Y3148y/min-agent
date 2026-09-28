"""Execution trace: one append-only JSONL file per session, plus console output.

Every interesting thing the runtime does emits a :class:`TraceEvent`. The file
is the durable record (post-mortem debugging, and the tests use it to assert on
*what was actually sent to the model*), the console renderer is the live view.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO

# --------------------------------------------------------------------------- #
# ANSI helpers (disabled when not a TTY or NO_COLOR is set)
# --------------------------------------------------------------------------- #
_NO_COLOR = bool(os.getenv("NO_COLOR")) or not sys.stdout.isatty()


def _c(code: str) -> str:
    return "" if _NO_COLOR else code


DIM = _c("\033[2m")
BOLD = _c("\033[1m")
RED = _c("\033[31m")
GREEN = _c("\033[32m")
YELLOW = _c("\033[33m")
BLUE = _c("\033[34m")
CYAN = _c("\033[36m")
RESET = _c("\033[0m")


@dataclass
class TraceEvent:
    ts: float
    seq: int
    kind: str
    session: str
    data: dict[str, Any] = field(default_factory=dict)


class Tracer:
    """Collects events for one session.

    :param console: where the human-readable rendering goes (``None`` = quiet).
    :param file: append-only JSONL destination (``None`` = no file).
    """

    def __init__(
        self,
        session_id: str,
        *,
        traces_root: Path | None = None,
        prefix: str = "",
        console: bool = True,
        stream: TextIO | None = None,
    ):
        self.session_id = session_id
        self.events: list[TraceEvent] = []
        self.console = console
        self._stream = stream or sys.stdout
        self._lock = threading.Lock()
        self._seq = 0
        self._fh: TextIO | None = None
        self.path: Path | None = None

        if traces_root is not None:
            traces_root.mkdir(parents=True, exist_ok=True)
            self.path = traces_root / f"{prefix}{session_id}.jsonl"
            self._fh = self.path.open("a", encoding="utf-8")

    # -- emit ---------------------------------------------------------------
    def emit(self, kind: str, **data: Any) -> TraceEvent:
        with self._lock:
            self._seq += 1
            ev = TraceEvent(
                ts=time.time(), seq=self._seq, kind=kind, session=self.session_id, data=data
            )
            self.events.append(ev)
            if self._fh is not None:
                self._fh.write(json.dumps(asdict(ev), ensure_ascii=False, default=str) + "\n")
                self._fh.flush()
            if self.console:
                self._render(ev)
        return ev

    # -- console renderers --------------------------------------------------
    def _render(self, ev: TraceEvent) -> None:
        d = ev.data
        s = self._stream
        if ev.kind == "user":
            s.write(f"{CYAN}you{RESET} {d.get('text', '')}\n")
        elif ev.kind == "llm_request":
            s.write(
                f"{DIM}-> llm  turn={d.get('turn')} msgs={d.get('messages')} "
                f"~{d.get('est_tokens')}tok tools={len(d.get('tools', []))}{RESET}\n"
            )
        elif ev.kind == "llm_response":
            s.write(
                f"{DIM}<- llm  {d.get('latency_ms')}ms in={d.get('input_tokens')} "
                f"out={d.get('output_tokens')} stop={d.get('stop_reason')} "
                f"blocks={[b for b in d.get('blocks', [])]}{RESET}\n"
            )
        elif ev.kind == "reasoning":
            text = (d.get("text") or "").replace("\n", " ")
            s.write(f"{DIM}   think: {text[:160]}{RESET}\n")
        elif ev.kind == "tool_call":
            icon = f"{GREEN}ok{RESET}" if d.get("ok") else f"{RED}err{RESET}"
            s.write(
                f"{YELLOW}[tool]{RESET} {BOLD}{d.get('name')}{RESET} "
                f"{DIM}{_short(d.get('args'))}{RESET} {icon} {d.get('latency_ms')}ms\n"
            )
        elif ev.kind == "tool_result":
            preview = (d.get("preview") or "").replace("\n", " ")
            s.write(f"{DIM}   {preview[:200]}{RESET}\n")
        elif ev.kind == "memory_recall":
            hits = d.get("hits") or []
            if hits:
                s.write(f"{BLUE}[mem]{RESET} recalled {len(hits)}: " + ", ".join(hits) + "\n")
        elif ev.kind == "memory_store":
            s.write(f"{BLUE}[mem]{RESET} stored: {d.get('text')}{RESET}\n")
        elif ev.kind == "compact":
            s.write(
                f"{YELLOW}[ctx]{RESET} compacted: {d.get('before')} -> {d.get('after')} "
                f"est tokens (~{d.get('est_tokens')})\n"
            )
        elif ev.kind == "error":
            s.write(f"{RED}[err] {RESET}{d.get('message')}\n")
        elif ev.kind == "warning":
            s.write(f"{YELLOW}[warn] {RESET}{d.get('message')}\n")
        elif ev.kind == "final":
            s.write(f"\n{BOLD}agent{RESET} {d.get('text')}\n\n")
        s.flush()

    # -- lifecycle ----------------------------------------------------------
    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # -- queries used by tests --------------------------------------------
    def of_kind(self, *kinds: str) -> list[TraceEvent]:
        wanted = set(kinds)
        return [e for e in self.events if e.kind in wanted]

    def last(self, kind: str) -> TraceEvent | None:
        found = self.of_kind(kind)
        return found[-1] if found else None


def _short(value: Any, limit: int = 120) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # pragma: no cover
        text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def read_trace(path: Path) -> list[dict[str, Any]]:
    """Load a trace file written by :class:`Tracer`."""
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events
