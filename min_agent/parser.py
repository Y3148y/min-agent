"""Parsing the model output.

Every "decide" step of the loop goes through here.  The raw output is a list of
content blocks in Anthropic's shape::

    ["thinking" block] -> agent's private reasoning   (trace, not transcript)
    ["text" block]     -> visible prose (often the final answer)
    ["tool_use" block] -> {id, name, input}           -> step 3
    [anything else]    -> tolerated, logged

Two robustness paths matter:

* ``stop_reason == "max_tokens"`` means the model ran out of budget -- the
  output is *not* a trustworthy final answer, and the loop must not hand a
  half-written answer to the user as if it were complete.
* some gateways / models answer with *text* instead of native ``tool_use``
  blocks even when tools were offered; the text fallback then extracts a
  ``<tool_call>`` / ```json``` payload so the loop does not depend on one SDK's
  behaviour.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .tools.base import ToolCall


@dataclass
class ParsedTurn:
    reasoning: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    truncated: bool = False
    ignored: list[str] = field(default_factory=list)  # block types we skip

    @property
    def should_act(self) -> bool:
        """Step 2 decision: call tools, or answer the user."""
        return bool(self.tool_calls)

    @property
    def final_answer(self) -> str:
        """Step 4 output when the model chose not to call more tools."""
        return "\n".join(t.strip() for t in self.texts).strip()

    def has_substantive_answer(self) -> bool:
        return bool(self.final_answer and len(self.final_answer) >= 2)


_TOOL_CALL_FENCED = re.compile(
    r"<tool_call>\s*(\{.*?\}|\[.*?\])\s*</tool_call>", re.DOTALL
)
_JSON_FENCED = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def parse_response(
    content: list[Any],
    *,
    stop_reason: str = "",
    text_fallback: bool = True,
) -> ParsedTurn:
    """Split content blocks into reasoning / text / tool calls."""
    blocks: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, dict):
            blocks.append(block)
        else:
            blocks.append(_any_to_dict(block))

    parsed = ParsedTurn(truncated=stop_reason == "max_tokens")

    for block in blocks:
        kind = block.get("type")
        if not isinstance(kind, str):
            continue
        if kind == "thinking":
            text = block.get("thinking") or block.get("text") or ""
            if text:
                parsed.reasoning.append(str(text))
        elif kind == "text":
            text = str(block.get("text") or "")
            if text.strip():
                parsed.texts.append(text)
        elif kind == "tool_use":
            parsed.tool_calls.append(
                ToolCall(
                    id=str(block.get("id") or ""),
                    name=str(block.get("name") or ""),
                    args=dict(block.get("input") or {}),
                )
            )
        else:
            parsed.ignored.append(kind)

    # Native tool_use wins; if the endpoint answered with text only, try to
    # recover a tool call from the prose.
    if not parsed.tool_calls and text_fallback:
        for text in parsed.texts:
            recovered = _extract_tool_call(text)
            if recovered is not None and not any(
                c.signature() == recovered.signature() for c in parsed.tool_calls
            ):
                parsed.tool_calls.append(recovered)

    return parsed


def _extract_tool_call(text: str) -> ToolCall | None:
    """Recover ``{name:..., arguments:{...}}`` or ``{tool,..., args}`` forms."""
    candidates = []
    for pattern in (_TOOL_CALL_FENCED, _JSON_FENCED):
        matches = list(pattern.finditer(text))
        if matches:
            candidates.extend(m.group(1) for m in matches)
            break  # prefer the explicit <tool_call> envelope
    if not candidates:
        return None

    for payload in candidates:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue

        name = data.get("name") or data.get("tool") or data.get("function")
        params = (
            data.get("input")
            or data.get("arguments")
            or data.get("args")
            or data.get("parameters")
        )
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {}
        if isinstance(name, str) and isinstance(params, dict):
            return ToolCall(id="text-fallback", name=name, args=dict(params))
    return None


def parse_single_tool_json(text: str) -> ToolCall | None:
    """Parse one complete ``tool_call`` JSON object (used by a few tests)."""
    return _extract_tool_call(f"<tool_call>{text}</tool_call>")


def _any_to_dict(block: Any) -> dict[str, Any]:
    base: dict[str, Any] = {}
    for attr in ("type", "text", "thinking", "signature", "id", "name", "input", "json"):
        value = getattr(block, attr, None)
        if value is not None:
            base[attr] = value
    return base