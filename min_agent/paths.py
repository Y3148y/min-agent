"""Turning user-supplied names into paths, safely and in exactly one place.

``--user`` and ``--session`` come straight off the command line, and two
components turn them into file names: :class:`~min_agent.store.SessionStore` and
:class:`~min_agent.trace.Tracer`.  The store sanitised; the tracer interpolated
the raw values, so ``--user ../..`` walked the trace file out of ``.traces/``
while the session directory next to it was being filtered all along.

Keeping the rule here means a new writer cannot quietly reintroduce the gap.
"""

from __future__ import annotations

from pathlib import Path


def safe_segment(segment: str) -> str:
    """Reduce ``segment`` to a single harmless path component.

    Anything outside ``[alnum-_]`` becomes ``_``, so separators, ``..`` and
    Windows drive/UNC forms all collapse; an empty result becomes ``default``.

    ``str.isalnum`` is Unicode-aware, so CJK window names survive intact --
    that is deliberate, not an oversight.
    """
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in segment)
    return cleaned or "default"


def ensure_within(root: Path, path: Path) -> Path:
    """Return ``path`` unchanged, or raise if it resolves outside ``root``.

    Second line of defence.  Sanitising the segments is the real guard; this
    catches anything that still manages to climb out, and it fails loudly
    rather than silently writing somewhere else.
    """
    root = root.resolve()
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"refusing to write outside {root}: {path}")
    return path
