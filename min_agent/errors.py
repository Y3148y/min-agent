"""Exception hierarchy and the retry policy.

The rule the runtime follows: **a tool failure is data, an LLM failure is an
exception.**  A broken tool must be reported back to the model as a
``tool_result`` so the model can correct itself; a broken transport must never
be silently swallowed, because that would leave the user staring at a spinner.
"""

from __future__ import annotations

import random
import time
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Tool errors
# --------------------------------------------------------------------------- #
class AgentError(Exception):
    """Base class for every error this project raises on purpose."""


class ToolError(AgentError):
    """A tool failed. Carries a message meant to be shown to the model."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint

    def to_model(self) -> str:
        return f"{self}\n{self.hint}" if self.hint else str(self)


class ToolNotFoundError(ToolError):
    """The model asked for a tool that is not registered."""

    def __init__(self, name: str, known: Iterable[str] = ()):
        names = ", ".join(sorted(known)) or "(none)"
        super().__init__(
            f"Unknown tool: {name!r}",
            hint=f"Available tools: {names}. Call one of those instead.",
        )
        self.name = name


class ToolInputError(ToolError):
    """The arguments did not satisfy the tool's JSON Schema."""


class ToolTimeoutError(ToolError):
    """The tool exceeded its wall-clock budget."""


# --------------------------------------------------------------------------- #
# LLM / transport errors
# --------------------------------------------------------------------------- #
class LLMError(AgentError):
    """A request to the model failed in a way we could not paper over."""


class LLMTransportError(LLMError):
    """Network blip, timeout, 5xx: worth retrying."""


class LLMRateLimitError(LLMTransportError):
    """429 / overloaded: worth retrying."""


class LLMContextOverflow(LLMError):
    """The prompt no longer fits the model's context window.

    Not retryable as-is -- the caller must compact and then retry once.
    """


class LLMBadRequestError(LLMError):
    """The provider rejected the request shape. Retrying will not help."""


class AgentAborted(AgentError):
    """The loop was stopped on purpose (guard rail, Ctrl-C, token limit)."""


# --------------------------------------------------------------------------- #
# Retry
# --------------------------------------------------------------------------- #
_RETRYABLE: tuple[type[Exception], ...] = (
    LLMTransportError,
    LLMRateLimitError,
    TimeoutError,
    ConnectionError,
)


def classify(exc: Exception) -> LLMError:
    """Map a raw provider/transport exception onto our hierarchy.

    Kept separate from :class:`LLMClient` so it can be unit-tested against
    plain ``Exception`` objects without any SDK involvement.
    """
    if isinstance(exc, LLMError):
        return exc

    name = type(exc).__name__.lower()
    text = str(exc).lower()

    if "context" in text and ("long" in text or "too long" in text or "exceed" in text):
        return LLMContextOverflow(str(exc))
    if "rate" in text or "429" in text or "overload" in text or "529" in text:
        return LLMRateLimitError(str(exc))
    if isinstance(exc, (TimeoutError, ConnectionError)) or "timeout" in name or "connect" in name:
        return LLMTransportError(str(exc))
    if "status 5" in text or "internal server" in text or "service unavailable" in text:
        return LLMTransportError(str(exc))
    if "status 4" in text or "invalid" in text or "bad request" in text:
        return LLMBadRequestError(str(exc))
    return LLMTransportError(str(exc))


def backoff_delays(attempts: int, base: float = 1.0, cap: float = 8.0) -> list[float]:
    """Exponential backoff with full jitter, returned as a list.

    Full jitter (``random.uniform(0, d)``) rather than fixed backoff: when a
    provider starts recovering, every client that retried on the same cadence
    would hit it on the same cadence again.
    """
    delays: list[float] = []
    for i in range(max(attempts, 0)):
        ceiling = min(cap, base * (2**i))
        delays.append(random.uniform(0.0, ceiling))
    return delays


def call_with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base: float = 1.0,
    cap: float = 8.0,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn`` with retries. ``attempts`` counts the total tries."""
    delays = backoff_delays(attempts - 1, base=base, cap=cap)
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except LLMContextOverflow:
            raise  # compacting is the only fix
        except LLMBadRequestError:
            raise  # deterministic rejection
        except _RETRYABLE as exc:  # type: ignore[misc]
            last = exc
            if i == attempts - 1:
                break
            delay = delays[i] if i < len(delays) else cap
            if on_retry:
                on_retry(i + 1, exc, delay)
            sleep(delay)
        except Exception as exc:  # raw provider/transport error -> classify first
            mapped = classify(exc)
            if not isinstance(mapped, (LLMTransportError, LLMRateLimitError)):
                raise mapped from exc  # deterministic: context overflow / bad request
            last = mapped
            if i == attempts - 1:
                break
            delay = delays[i] if i < len(delays) else cap
            if on_retry:
                on_retry(i + 1, mapped, delay)
            sleep(delay)
    assert last is not None
    raise last
