"""Central configuration.

Every tunable the agent has is resolved exactly once, here, so that the rest
of the runtime can be constructed with plain values and tests can override a
single ``Config`` object instead of patching module globals.
"""

from __future__ import annotations

import os
import warnings
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
        _warn_bad_env(key, raw, default)
        return default


def _env_float(key: str, default: float) -> float:
    """Like :func:`_env_int` but keeps the fraction.

    TOOL_TIMEOUT went through `_env_int`, so `TOOL_TIMEOUT=2.5` raised
    ValueError and silently became 10 -- a sub-second tool budget could not be
    expressed at all, and nothing said so.
    """
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        _warn_bad_env(key, raw, default)
        return default


def _warn_bad_env(key: str, raw: str, default) -> None:
    warnings.warn(
        f"{key}={raw!r} is not a valid number; falling back to {default!r}",
        RuntimeWarning,
        stacklevel=3,
    )


def _env_str(key: str, default: str) -> str:
    raw = os.getenv(key)
    return default if raw is None or raw.strip() == "" else raw.strip()


@dataclass
class Config:
    """Runtime configuration for one agent process."""

    # --- LLM ---
    model: str = "qwen3.8-flash"
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
    # "mock"    deterministic fake forecast (default, fully offline)
    # "wttr.in" real forecast from https://wttr.in (no API key), mock fallback if unreachable
    weather_backend: str = "mock"

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
        problems.extend(self.invalid_limits())
        return problems

    def invalid_limits(self) -> list[str]:
        """Fields that are set but would misbehave, as ``FIELD=why`` strings.

        The numbers all have a zero or a negative, and every one of them was
        accepted silently before.  MAX_TURNS=0 makes range(1, 1) empty, so the
        for/else in run_turn wraps up on the spot and the agent answers
        nothing; MAX_TURNS=-5 behaves the same.  MAX_TOOL_RESULT_CHARS=-1
        truncates to text[:-1] on every result, quietly corrupting tool output.
        TOOL_TIMEOUT=0 makes every tool time out immediately.  Catching them at
        load time turns a mysterious wrong answer into a named setting.
        """
        problems: list[str] = []
        for name in (
            "max_turns",
            "max_repeat_call",
            "max_tool_result_chars",
            "context_budget",
            "keep_recent_messages",
            "summary_max_chars",
            "memory_top_k",
            "memory_ttl_days",
        ):
            if getattr(self, name) <= 0:
                problems.append(f"{name.upper()}=must be > 0 (got {getattr(self, name)})")
        for name in ("tool_timeout", "request_timeout"):
            value = getattr(self, name)
            if value <= 0:
                problems.append(f"{name.upper()}=must be > 0 (got {value})")
        return problems


def load_config(**overrides) -> Config:
    """Build a :class:`Config` from ``.env`` + the environment.

    ``.env`` is looked up next to the repo root, not next to the caller's cwd,
    so ``python -m min_agent`` behaves the same from any directory.
    """
    load_dotenv(REPO_ROOT / ".env", override=True)

    defaults = Config(
        model=_env_str("MODEL_ID", "qwen3.8-flash"),
        api_key=_env_str("ANTHROPIC_API_KEY", ""),
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        max_turns=_env_int("MAX_TURNS", 12),
        max_repeat_call=_env_int("MAX_REPEAT_CALL", 3),
        tool_timeout=_env_float("TOOL_TIMEOUT", 10.0),
        max_tool_result_chars=_env_int("MAX_TOOL_RESULT_CHARS", 2000),
        request_timeout=_env_float("REQUEST_TIMEOUT", 120.0),
        weather_backend=_env_str("WEATHER_BACKEND", "mock"),
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
