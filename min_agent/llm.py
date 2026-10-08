"""The LLM client: one ``complete(...)`` surface, two backends.

* :class:`AnthropicLLM` -- the real thing, against ``ANTHROPIC_BASE_URL``
  (either api.anthropic.com or any Anthropic-compatible gateway).
* :class:`ScriptedLLM` lives in the tests -- it replays scripted content so the
  whole loop can run with zero network.

The response is normalised into plain dicts immediately, so the parser, the
transcript serializer and the fake client all speak the same shape.  Nothing in
the loop ever touches SDK objects.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import Config
from .errors import call_with_retry, classify


# --------------------------------------------------------------------------- #
# Shared response shape
# --------------------------------------------------------------------------- #
@dataclass
class LLMResponse:
    content: list[dict[str, Any]]
    stop_reason: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = field(default=None, repr=False)

    @property
    def has_tool_use(self) -> bool:
        return any(b.get("type") == "tool_use" for b in self.content)

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content if b.get("type") == "text")


@dataclass
class LLMRequest:
    messages: list[dict[str, Any]]
    system: str = ""
    tools: list[dict[str, Any]] = field(default_factory=list)
    max_tokens: int = 4096


@dataclass
class StreamEvent:
    """One live chunk of a streaming response.

    ``kind`` is ``"text"``/``"thinking"`` for incremental deltas and ``"done"``
    for the final complete message.  The loop renders the deltas onto the
    console as they arrive and assembles the ``LLMResponse`` from the ``done``
    event, so downstream code sees exactly what the blocking path produced.
    """

    kind: str
    delta: str = ""
    response: "LLMResponse | None" = None


class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...


# --------------------------------------------------------------------------- #
# Real backend
# --------------------------------------------------------------------------- #
class AnthropicLLM:
    """Anthropic Messages protocol, wherever ``base_url`` points it."""

    def __init__(self, config: Config):
        from anthropic import Anthropic

        self.config = config
        # request_timeout used to be a config field that nothing read, so the
        # SDK's own default (10 minutes) applied and one slow gateway could hold
        # a turn open that long.  The per-call overrides in _request_kwargs
        # exist because the summariser needs its own, smaller budget.
        self.client = Anthropic(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.request_timeout,
        )
        self._last_latency_ms = 0

    @property
    def last_latency_ms(self) -> int:
        return self._last_latency_ms

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        try:
            # NB: attention on temperature.  The anthropic SDK >= 1.x removed
            # `temperature` from the Messages.create surface entirely, and some
            # Anthropic-compatible gateways reject it as unknown -- so we do
            # NOT forward it.
            response = call_with_retry(
                lambda: self.client.messages.create(
                    model=self.config.model,
                    max_tokens=request.max_tokens,
                    system=request.system or None,
                    messages=request.messages,
                    tools=request.tools or None,
                ),
                attempts=self.config.max_retries,
                base=self.config.retry_base_delay,
                cap=self.config.retry_max_delay,
            )
        finally:
            self._last_latency_ms = int((time.perf_counter() - started) * 1000)

        blocks = [_block_to_dict(b) for b in response.content]
        usage = getattr(response, "usage", None)
        return LLMResponse(
            content=blocks,
            stop_reason=getattr(response, "stop_reason", "") or "",
            model=getattr(response, "model", self.config.model),
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            raw=response,
        )

    def stream(self, request: LLMRequest) -> Iterator[StreamEvent]:
        """Like :meth:`complete`, but yields incremental ``text``/``thinking``
        deltas as they arrive and finishes with a ``done`` event carrying the
        assembled :class:`LLMResponse`.

        Retry policy lives in the connect phase only: ``messages.stream``
        returns a manager without any network I/O, so we retry ``__enter__``
        (which opens the connection) and let a mid-stream failure abort the
        turn instead of re-winding already-rendered lines.
        """
        started = time.perf_counter()
        manager = call_with_retry(
            lambda: self.client.messages.stream(
                model=self.config.model,
                max_tokens=request.max_tokens,
                system=request.system or None,
                messages=request.messages,
                tools=request.tools or None,
            ),
            attempts=self.config.max_retries,
            base=self.config.retry_base_delay,
            cap=self.config.retry_max_delay,
        )
        stream = call_with_retry(
            lambda: manager.__enter__(),
            attempts=self.config.max_retries,
            base=self.config.retry_base_delay,
            cap=self.config.retry_max_delay,
        )
        try:
            try:
                for ev in stream:
                    # Dispatch on the event's own `type` field instead of importing
                    # the (private, version-fragile) event classes: parsed events
                    # carry ``type == "text"/"thinking"`` while raw deltas use
                    # ``*_delta`` / ``content_block_*`` -- no collisions.
                    ev_type = getattr(ev, "type", "")
                    if ev_type == "text":
                        yield StreamEvent("text", delta=getattr(ev, "text", ""))
                    elif ev_type == "thinking":
                        yield StreamEvent("thinking", delta=getattr(ev, "thinking", ""))
                final = stream.get_final_message()
            except Exception as exc:
                # A mid-stream failure must be classified, not leaked as a raw
                # SDK exception: run_turn only catches LLMError, so an unclassified
                # raise would crash the REPL with a traceback instead of the
                # friendly "模型请求失败" message.
                raise classify(exc) from exc
        finally:
            try:
                manager.__exit__(None, None, None)
            finally:
                self._last_latency_ms = int((time.perf_counter() - started) * 1000)

        blocks = [_block_to_dict(b) for b in final.content]
        usage = getattr(final, "usage", None)
        yield StreamEvent(
            "done",
            response=LLMResponse(
                content=blocks,
                stop_reason=getattr(final, "stop_reason", "") or "",
                model=getattr(final, "model", self.config.model),
                input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
                output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
                raw=final,
            ),
        )


def _block_to_dict(block: Any) -> dict[str, Any]:
    """Normalise one SDK content block into a plain dict."""
    kind = getattr(block, "type", None)
    base = {"type": kind}
    if kind == "text":
        base["text"] = getattr(block, "text", "")
    elif kind == "thinking":
        base["thinking"] = getattr(block, "thinking", "")
        base["signature"] = getattr(block, "signature", None)
    elif kind == "tool_use":
        base["id"] = getattr(block, "id", "")
        base["name"] = getattr(block, "name", "")
        base["input"] = dict(getattr(block, "input", {}))
    else:
        for attr in ("text", "id", "name", "input", "thinking", "signature", "json"):
            value = getattr(block, attr, None)
            if value is not None:
                base[attr] = value
    return base


# --------------------------------------------------------------------------- #
# Token estimation (provider-independent, no tokenizer dependency)
# --------------------------------------------------------------------------- #
_CJK_RE = re.compile(r"[\u3000-\u9fff\uff00-\uffef]")


def estimate_tokens(text: str) -> int:
    """Approximate token count.

    Deliberately heuristic (no tiktoken dependency): Han/CJK chars count close
    to one token each on modern tokenizers, Latin word characters average 3.5
    per token, and whitespace is nearly free.  We over-estimate slightly so the
    compaction trigger errs on the safe side.
    """
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    rest = len(text) - cjk
    return cjk + max(0, int(rest / 3.5)) + 1


def estimate_message_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
    """Sum of message payload tokens plus a small per-message overhead."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind == "tool_use":
                    total += estimate_tokens(block.get("name", "") + _json(block.get("input", {})))
                elif kind == "tool_result":
                    total += estimate_tokens(_json(block.get("content", "")))
                else:
                    total += estimate_tokens(_json(block))
        elif content is not None:
            total += estimate_tokens(_json(content))
        total += 4  # protocol overhead per message
    if tools:
        total += estimate_tokens(_json(tools)) // 2
    return total


def _json(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # pragma: no cover - non-serialisable
        return str(value)