"""A scripted LLM with the exact same surface as ``AnthropicLLM``.

The whole agent loop is exercised here with zero network.  A script is a list
of *scenes*; each scene is either a dict (content blocks + stop_reason), an
``LLMResponse``, or a callable ``(request) -> scene`` for assertions that need
to depend on the incoming request (e.g. "the second message must contain the
first tool result").
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, Sequence

from min_agent.llm import LLMRequest, LLMResponse


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def think_block(text: str) -> dict:
    return {"type": "thinking", "thinking": text}


def tool_block(tool_use_id: str, name: str, **args) -> dict:
    return {"type": "tool_use", "id": tool_use_id, "name": name, "input": args}


def result_block(tool_use_id: str) -> dict:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": "ok"}


class ScriptedLLM:
    """Deterministically replay ``script`` against every ``complete`` call."""

    kind = "scripted"

    def __init__(
        self,
        script: Sequence[Any] = (),
        *,
        default: str = "（脚本已耗尽）已根据前文回答完毕。",
        fail_on_empty: bool = True,
        record: bool = True,
    ):
        self.script: list[Any] = list(script)
        self.default = default
        self.fail_on_empty = fail_on_empty
        self.record = record
        self.requests: list[LLMRequest] = []
        self.responses: list[LLMResponse] = []
        self.last_latency_ms = 3

    def complete(self, request: LLMRequest) -> LLMResponse:
        if self.record:
            self.requests.append(request)
        if self.script:
            scene = self.script.pop(0)
            if isinstance(scene, Exception):
                raise scene
        elif self.fail_on_empty:
            raise AssertionError(
                "ScriptedLLM ran out of scripted responses; feed a longer `script`."
            )
        else:
            scene = {"content": [text_block(self.default)], "stop_reason": "end_turn"}

        if callable(scene):
            scene = scene(request)  # may return another scene
        if isinstance(scene, LLMResponse):
            response = scene
        else:
            response = LLMResponse(
                content=list(scene.get("content", [])),
                stop_reason=scene.get("stop_reason", "end_turn"),
                model="scripted",
                input_tokens=scene.get("input_tokens", 0),
                output_tokens=scene.get("output_tokens", 0),
            )
        self.responses.append(response)
        return response

    # -- introspection ------------------------------------------------------
    def last_request(self, n: int = 1) -> LLMRequest:
        return self.requests[-n]

    def transcript_text(self) -> str:
        """Everything the model has seen so far, as flat text (for asserts)."""
        from min_agent.llm import _json

        parts = []
        for req in self.requests:
            for msg in req.messages:
                parts.append(_json(msg.get("content")))
        return "\n".join(parts)


class CapturingLLM(Protocol):
    kind: str
    requests: list[LLMRequest]


def menu_default(**kw: Any) -> Callable:
    return lambda req: kw