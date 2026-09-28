"""``read_docs`` -- retrieval over the repository's own ``docs/`` folder.

This is the tool that makes "带着工具的追问" honest: a follow-up like "那第二条
是什么意思?" only works if the earlier tool output is still reachable, and a
follow-up that needs *more* than was in the snippet has to go back to the source.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Callable

from ..errors import ToolError
from .base import ToolSpec, tool

_MAX_READ_CHARS = 6000
_TOKEN = re.compile(r"[a-z0-9]+|[一-鿿]")


def _scan(docs_dir: Path) -> list[Path]:
    if not docs_dir.exists():
        return []
    return sorted(
        p for p in docs_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt"}
    )


def _rank(files: list[Path], query: str) -> list[tuple[int, Path, list[str]]]:
    q = {t.lower() for t in _TOKEN.findall(query) if len(t.strip()) > 1}
    if not q:
        return []
    hits: list[tuple[int, Path, list[str]]] = []
    for path in files:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        matched = [
            (i, line)
            for i, line in enumerate(lines)
            if q & {t.lower() for t in _TOKEN.findall(line)}
        ]
        if matched:
            hits.append((len(matched), path, [f"{i}: {line}" for i, line in matched[:6]]))
    hits.sort(key=lambda t: -t[0])
    return hits


def _resolve(docs_dir: Path, name: str) -> Path:
    """Map a user/model-supplied name onto a real file inside docs/."""
    needle = name.strip().lstrip("./")
    candidates = [c for c in _scan(docs_dir) if c.name == needle or str(c.relative_to(docs_dir)) == needle]
    if not candidates:
        candidates = [c for c in _scan(docs_dir) if needle.lower() in c.name.lower()]
    if not candidates:
        available = ", ".join(c.name for c in _scan(docs_dir)) or "(docs/ is empty)"
        raise ToolError(f"No document named {name!r}", hint=f"Available: {available}")
    return candidates[0]


def build_read_docs_tool(docs_dir: Path) -> ToolSpec:
    """Bind the tool to a concrete docs directory."""

    @tool(
        tags=("knowledge",),
        description=(
            "Read the project's own documentation in docs/. action='list' lists the files, "
            "'search' finds lines matching a query, 'read' returns a whole file (optionally one "
            "section). Use it to check details the user asks about that are not in the chat, and "
            "to go back to the source when a search snippet is not enough."
        ),
    )
    def read_docs(
        action: Annotated[str, "one of: list, search, read"] = "list",
        query: Annotated[str, "keywords, for action='search'"] = "",
        name: Annotated[str, "file name, for action='read'"] = "",
        section: Annotated[str, "section heading to start from, for action='read'"] = "",
    ) -> str:
        action = action.strip().lower()
        files = _scan(docs_dir)

        if action == "list":
            if not files:
                return "docs/ is empty."
            return "\n".join(f"- {p.relative_to(docs_dir)} ({p.stat().st_size} bytes)" for p in files)

        if action == "search":
            if not query.strip():
                raise ToolError("action='search' needs query=<keywords>")
            hits = _rank(files, query)
            if not hits:
                return f"No lines in docs/ match {query!r}."
            out = [f"{len(hits)} file(s) matched {query!r}:"]
            for _, path, lines in hits:
                out.append(f"\n## {path.relative_to(docs_dir)}")
                out.extend(f"  {ln}" for ln in lines)
            return "\n".join(out)

        if action == "read":
            if not name.strip():
                raise ToolError("action='read' needs name=<file>")
            path = _resolve(docs_dir, name)
            text = path.read_text(encoding="utf-8")
            if section.strip():
                heading = re.compile(
                    r"^#{1,6}\s*" + re.escape(section.strip()) + r"\s*$", re.MULTILINE | re.IGNORECASE
                )
                m = heading.search(text)
                if not m:
                    headings = re.findall(r"^#{1,6}\s*(.+)$", text, re.MULTILINE)
                    raise ToolError(
                        f"No section {section!r} in {path.name}",
                        hint=f"Sections: {', '.join(headings[:20]) or '(none)'}",
                    )
                nxt = re.search(r"^#{1,6}\s", text[m.end():], re.MULTILINE)
                end = m.end() + nxt.start() if nxt else len(text)
                text = text[m.end() : end].strip()
            if len(text) > _MAX_READ_CHARS:
                text = text[:_MAX_READ_CHARS] + f"\n... [truncated at {_MAX_READ_CHARS} chars]"
            return f"# {path.relative_to(docs_dir)}\n\n{text}"

        raise ToolError(
            f"Unknown action {action!r}", hint="Use one of: list, search, read"
        )

    return read_docs
