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
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO

from .paths import ensure_within, safe_segment

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


# How many events to keep in RAM for `of_kind`/`last`.  Generous enough that a
# normal turn's worth is always resident, small enough that a multi-hour REPL
# cannot grow without bound.
_DEFAULT_EVENTS_KEEP = 2000


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
        events_keep: int = _DEFAULT_EVENTS_KEEP,
    ):
        self.session_id = session_id
        # Bounded: a long REPL used to grow this list for the lifetime of the
        # process, one full copy of every event (including complete tool output)
        # held in RAM.  The JSONL file on disk is the real record and is
        # unbounded on purpose; this is only the in-memory query window.
        self.events: deque[TraceEvent] = deque(maxlen=events_keep)
        self.console = console
        self._stream = stream or sys.stdout
        self._lock = threading.Lock()
        self._seq = 0
        self._fh: TextIO | None = None
        self.path: Path | None = None
        self._live_agent = False

        if traces_root is not None:
            traces_root.mkdir(parents=True, exist_ok=True)
            # Both halves of the file name are user-controlled, and this used to
            # interpolate them raw: `--user ../..` produced the path
            # `.traces/../../.main.jsonl` and happily wrote outside the
            # workspace, while the session store next to it sanitised all along.
            # Strip the separator off ``prefix``, sanitise the two halves on
            # their own, then put the dot back so the on-disk name stays
            # "<user>.<session>.jsonl" and existing traces keep resolving.
            stem = safe_segment(prefix.rstrip(".")) if prefix else ""
            name = f"{stem}.{safe_segment(session_id)}" if stem else safe_segment(session_id)
            self.path = ensure_within(traces_root, traces_root / f"{name}.jsonl")
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

    # -- live streaming of the final line ----------------------------------
    def stream(self, kind: str, delta: str) -> None:
        """Stream text onto the console in pure-append mode.

        The answer opens once with an ``agent`` label, then every delta is
        written verbatim (no ``\\r`` rewrites, no repeated prefixes), so what
        reaches the console equals the final text exactly -- on every terminal,
        including legacy cmd windows where ``\\r`` does not reset the cursor.
        """
        if not self.console or kind != "text" or not delta:
            return
        s = self._stream
        if not self._live_agent:
            s.write(f"\n{BOLD}agent{RESET} ")
            self._live_agent = True
        s.write(delta)
        s.flush()

    # -- console renderers --------------------------------------------------
    def _render(self, ev: TraceEvent) -> None:
        d = ev.data
        s = self._stream
        if ev.kind == "user":
            s.write(f"{CYAN}you{RESET} {d.get('text', '')}\n")
        elif ev.kind == "llm_request":
            tools = d.get("tools", 0)
            tools = len(tools) if isinstance(tools, (list, tuple)) else tools
            s.write(
                f"{DIM}-> llm  turn={d.get('turn')} msgs={d.get('messages')} "
                f"~{d.get('est_tokens')}tok tools={tools}{RESET}\n"
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
            text = d.get("text") or ""
            if self._live_agent:
                # The answer was streamed verbatim; just close the line with a
                # blank separator -- no re-print of the content.
                self._stream.write("\n\n")
                self._live_agent = False
            else:
                self._stream.write(f"\n{BOLD}agent{RESET} {text}\n\n")
            self._stream.flush()
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
        # Reverse scan, so a bounded window still answers "most recent" in O(1)
        # amortised rather than materialising the whole filter.
        for event in reversed(self.events):
            if event.kind == kind:
                return event
        return None


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
