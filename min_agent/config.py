"""Central configuration.

Every tunable the agent has is resolved exactly once, here, so that the rest
of the runtime can be constructed with plain values and tests can override a
single ``Config`` object instead of patching module globals.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # pragma: no cover - trivial import shim
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_a, **_kw):  # type: ignore[misc]
        return False


REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_str(key: str, default: str) -> str:
    raw = os.getenv(key)
    return default if raw is None or raw.strip() == "" else raw.strip()


@dataclass
class Config:
    """Runtime configuration for one agent process."""

    # --- LLM ---
    model: str = "qwen3.7-flash-2026-07-15"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 4096
    request_timeout: float = 120.0
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 8.0

    # --- loop guards ---
    max_turns: int = 12
    max_repeat_call: int = 3

    # --- tools ---
    tool_timeout: float = 10.0
    max_tool_result_chars: int = 2000

    # --- context management ---
    context_budget: int = 24_000
    keep_recent_messages: int = 6
    summary_max_chars: int = 800

    # --- memory ---
    memory_top_k: int = 5
    memory_ttl_days: int = 90

    # --- storage roots (per-process; each window points at the same repo) ---
    workspace: Path = field(default_factory=lambda: REPO_ROOT)
    sessions_dirname: str = ".sessions"
    traces_dirname: str = ".traces"
    memory_dirname: str = ".memory"
    docs_dir: Path = DOCS_DIR

    @property
    def sessions_root(self) -> Path:
        return self.workspace / self.sessions_dirname

    @property
    def traces_root(self) -> Path:
        return self.workspace / self.traces_dirname

    @property
    def memory_root(self) -> Path:
        return self.workspace / self.memory_dirname

    def missing_llm_config(self) -> list[str]:
        """Names of the settings that must be filled in before we can start."""
        problems: list[str] = []
        if not self.api_key:
            problems.append("ANTHROPIC_API_KEY")
        if not self.model:
            problems.append("MODEL_ID")
        return problems


def load_config(**overrides) -> Config:
    """Build a :class:`Config` from ``.env`` + the environment.

    ``.env`` is looked up next to the repo root, not next to the caller's cwd,
    so ``python -m min_agent`` behaves the same from any directory.
    """
    load_dotenv(REPO_ROOT / ".env", override=True)

    defaults = Config(
        model=_env_str("MODEL_ID", "qwen3.7-flash-2026-07-15"),
        api_key=_env_str("ANTHROPIC_API_KEY", ""),
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        max_turns=_env_int("MAX_TURNS", 12),
        max_repeat_call=_env_int("MAX_REPEAT_CALL", 3),
        tool_timeout=float(_env_int("TOOL_TIMEOUT", 10)),
        max_tool_result_chars=_env_int("MAX_TOOL_RESULT_CHARS", 2000),
        context_budget=_env_int("CONTEXT_BUDGET", 24_000),
        keep_recent_messages=_env_int("KEEP_RECENT_MESSAGES", 6),
        summary_max_chars=_env_int("SUMMARY_MAX_CHARS", 800),
        memory_top_k=_env_int("MEMORY_TOP_K", 5),
        memory_ttl_days=_env_int("MEMORY_TTL_DAYS", 90),
    )
    for key, value in overrides.items():
        if not hasattr(defaults, key):
            raise AttributeError(f"unknown config field: {key}")
        setattr(defaults, key, value)
    return defaults
