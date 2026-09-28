"""The agent loop -- the four steps, made explicit.

    Step 1  receive the user input                -> append to the session
    Step 2  decide answer-vs-act                  -> LLM call + output parser
    Step 3  act, if that is what the model chose  -> run the tools, feed back
    Step 4  continue or finish                    -> repeat while tool calls
                                                      arrive; else answer

Everything else the brief asks for hangs off this core: a registry the model
chooses from (step 2), concurrent tool execution + trace (step 3), guard rails
(max turns, repeat-call detection), context compaction, memory recall before
step 2, and session persistence around step 1.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .config import Config
from .context import LLMCompactor, build_system_prompt, compact_messages
from .errors import AgentAborted, LLMContextOverflow, LLMError
from .llm import LLMRequest, estimate_message_tokens, estimate_tokens
from .memory import MemoryStore, extract_facts
from .parser import parse_response
from .session import Session
from .store import SessionStore
from .tools import ToolContext, build_registry
from .trace import Tracer


@dataclass
class TurnResult:
    text: str = ""
    turns: int = 0
    tool_calls: int = 0
    compressed: bool = False
    guard_aborted: bool = False
    turn_budget_exhausted: bool = False
    error: str = ""


@dataclass
class _Compiled:
    system: str
    tools: list[dict[str, Any]]


class Agent:
    """One window of one user. All mutable state lives in the session object."""

    def __init__(
        self,
        *,
        config: Config,
        user: str,
        session_id: str,
        session: Session | None = None,
        llm: Any = None,
        trace: Tracer | None = None,
    ):
        self.config = config
        self.user = user
        self.session_id = session_id

        if session is not None:
            self.session = session
        else:
            self.session = SessionStore(config.sessions_root).open(user, session_id, create=True)

        if llm is not None:
            self.llm = llm
        else:
            from .llm import AnthropicLLM

            missing = config.missing_llm_config()
            if missing:
                raise RuntimeError(
                    "LLM not configured: missing "
                    + ", ".join(missing)
                    + " (copy .env.example to .env)"
                )
            self.llm = AnthropicLLM(config)

        self.trace = trace or Tracer(session_id, traces_root=config.traces_root)
        self.memory = MemoryStore(config.memory_root / self._safe_user() / "facts.json")

        self.tool_ctx = ToolContext(
            session_id=session_id,
            session_dir=self.session.dir,
            config=config,
        )
        self.registry, self.todo_store = build_registry(self.tool_ctx, memory=self.memory)
        self.compactor = LLMCompactor(self.llm)

    def _safe_user(self) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in self.user) or "default"

    # ------------------------------------------------------------------ #
    # Step 1 + the loop
    # ------------------------------------------------------------------ #
    def run_turn(self, user_input: str) -> TurnResult:
        result = TurnResult()
        user_input = user_input.strip()
        if not user_input:
            return result

        self.trace.emit("user", text=user_input)
        self.session.append("user", user_input)

        # Memory recall runs here, before the model is ever asked -- the 
        # current user input is the query.
        memory_items = self.memory.recall(user_input, top_k=self.config.memory_top_k)
        if memory_items:
            self.trace.emit("memory_recall", hits=[m.text for m in memory_items])

        recent_calls: list[str] = []
        try:
            for turn in range(1, self.config.max_turns + 1):
                if recent_calls and recent_calls.count(recent_calls[-1]) >= self.config.max_repeat_call:
                    self.trace.emit("warning", message="tool calls looping on identical args")
                    result.guard_aborted = True
                    result.text = self._wrap_up(recent_calls)
                    break

                parsed = self._ask(memory_items, allow_tools=True)

                if not parsed.should_act:
                    if parsed.truncated:
                        # Output budget ran out mid-answer: never hand a
                        # fragment to the user -- ask for its conclusion.
                        self.session.append("user", "（继续。把答案说完，不要调用工具。）")
                        self.trace.emit("warning", message="max_tokens reached; continuing")
                        continue
                    if not parsed.has_substantive_answer():
                        self.session.append("user", "（没有收到可用回复，请直接给出答案。）")
                        self.trace.emit("warning", message="empty reply; nudging")
                        continue
                    result.text = parsed.final_answer
                    self.trace.emit("final", text=result.text)
                    break

                calls = parsed.tool_calls
                result.tool_calls += self._execute_tools(calls, recent_calls)
            else:
                # max_turns exhausted: force a tool-less conclusion
                result.turn_budget_exhausted = True
                result.text = self._wrap_up(recent_calls)

        except LLMContextOverflow:
            self._repair_last_turn("user")
            result.text = "上下文超过模型上限且压缩失败。请换个更简短的问法，或开启新窗口。"
            result.error = "context overflow"
        except LLMError as exc:
            self._repair_last_turn("user")
            result.text = f"模型请求失败，请稍后重试。原因：{exc}"
            result.error = str(exc)
        except AgentAborted as exc:
            self._repair_last_turn("user")
            result.text = str(exc)
        result.turns = len(recent_calls)
        return result

    # ------------------------------------------------------------------ #
    # Step 2: one LLM decision
    # ------------------------------------------------------------------ #
    def _ask(self, memory_items, *, allow_tools: bool, continuation: bool = False):
        compiled = self._compile(memory_items, allow_tools=allow_tools)
        self._maybe_compact(compiled)

        request = LLMRequest(
            messages=list(self.session.messages),
            system=compiled.system,
            tools=compiled.tools if allow_tools else [],
            max_tokens=self.config.max_tokens,
        )
        self.trace.emit(
            "llm_request",
            continuation=continuation,
            messages=len(request.messages),
            est_tokens=estimate_message_tokens(request.messages)
            + estimate_tokens(compiled.system),
            tools=len(request.tools),
        )

        for attempt in (1, 2):
            try:
                response = self.llm.complete(request)
                break
            except LLMContextOverflow:
                if attempt == 2:
                    raise
                # The prompt grew between request and dispatch; compact and retry.
                self._maybe_compact(compiled, force=True)
        else:  # pragma: no cover - unreachable, for type checkers
            raise LLMContextOverflow("unreachable")

        self.trace.emit(
            "llm_response",
            latency_ms=getattr(self.llm, "last_latency_ms", 0),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            stop_reason=response.stop_reason,
            blocks=[b.get("type") for b in response.content[:8]],
        )
        parsed = parse_response(response.content, stop_reason=response.stop_reason)

        # Persist the assistant turn; thinking blocks are kept out of the
        # transcript (verified safe against the upstream API).
        self.session.append("assistant", _strip_thinking(response.content))
        for reasoning in parsed.reasoning:
            self.trace.emit("reasoning", text=reasoning)
        return parsed

    # ------------------------------------------------------------------ #
    # Step 3: run tools, feed results back
    # ------------------------------------------------------------------ #
    def _execute_tools(self, calls, recent_calls: list[str]) -> int:
        def one(call) -> dict[str, Any]:
            recent_calls.append(call.signature())
            outcome = self.registry.call(
                call.name,
                call.args,
                timeout=self.config.tool_timeout,
                max_result_chars=self.config.max_tool_result_chars,
            )
            self.trace.emit(
                "tool_call",
                name=call.name,
                args=call.args,
                ok=outcome["ok"],
                latency_ms=outcome["latency_ms"],
            )
            content: Any
            if outcome["ok"]:
                content = outcome["result"]
                self.trace.emit("tool_result", preview=outcome["result"])
                return {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": content,
                }
            content = f"Error: {outcome['error']}"
            self.trace.emit("error", message=f"{call.name}: {outcome['error']}")
            return {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": content,
                "is_error": True,
            }

        with ThreadPoolExecutor(max_workers=min(4, len(calls))) as pool:
            blocks = list(pool.map(one, calls))
        if blocks:
            self.session.append_tool_results(blocks)
        return len(calls)

    # ------------------------------------------------------------------ #
    # guards
    # ------------------------------------------------------------------ #
    def _wrap_up(self, recent_calls: list[str]) -> str:
        self.trace.emit(
            "warning",
            message=f"tool budget exhausted ({self.config.max_turns} turns)",
        )
        self.session.append(
            "user",
            "（工具调用次数已达上限。基于已获取的结果直接给出最终答案，不要调用工具。）",
        )
        try:
            parsed = self._ask([], allow_tools=False, continuation=True)
            return parsed.final_answer or "已达到工具调用上限，但没有可用的结论。"
        except LLMError as exc:  # noqa: BLE001
            return f"已达到工具调用上限，且收尾回答生成失败（{exc}）。"

    # ------------------------------------------------------------------ #
    # repairs
    # ------------------------------------------------------------------ #
    def _repair_last_turn(self, role: str) -> None:
        """Keep the transcript a clean user/assistant alternation after a fatal error.

        - overflow / transport failure right after a *user* turn: drop that
          unanswered user message so a retry does not send [user, user,...].
        - failure after we already persisted an assistant turn plus dangling
          tool_results: drop both (a tool_result without its tool_use is
          rejected by the provider).
        """
        messages = self.session.messages
        if not messages:
            return
        if messages[-1]["role"] == role:
            messages.pop()
            self.session._rewrite_jsonl()
        # tool_results whose assistant turn died mid-appending
        if len(messages) >= 2:
            head, tail = messages[-2], messages[-1]
            if (
                head["role"] == "assistant"
                and tail["role"] == "user"
                and _is_tool_results_message(tail)
            ):
                self.session.messages = messages[:-2]
                self.session._rewrite_jsonl()

    # ------------------------------------------------------------------ #
    # context management
    # ------------------------------------------------------------------ #
    def _compile(self, memory_items, *, allow_tools: bool) -> _Compiled:
        system = build_system_prompt(
            user=self.user,
            session_id=self.session_id,
            memory_notes=self.memory.notes(memory_items) if memory_items else "",
            todo_digest=self.todo_store.digest(),
            summary=self.session.meta.summary,
            max_tool_output=str(self.config.max_tool_result_chars),
        )
        return _Compiled(system=system, tools=self.registry.to_api() if allow_tools else [])

    def _maybe_compact(self, compiled: _Compiled, *, force: bool = False) -> bool:
        budget = self.config.context_budget
        system_tokens = estimate_tokens(compiled.system)
        current = system_tokens + estimate_message_tokens(self.session.messages, compiled.tools)
        if not (force or current > budget):
            return False
        compact_messages(
            self.session,
            budget=budget,
            keep_recent=self.config.keep_recent_messages,
            summary_max=self.config.summary_max_chars,
            compactor=self.compactor,
            system_overhead=system_tokens,
            trace=self.trace,
        )
        return True

    # ------------------------------------------------------------------ #
    # window close: long-term memory sweep
    # ------------------------------------------------------------------ #
    def patrol(self) -> int:
        """Extract durable facts from this session into long-term memory."""
        if not self.session.messages:
            return 0
        stored = 0
        for text in extract_facts(self.session.messages, self.llm):
            if self.memory.remember(text, source_session=self.session_id):
                stored += 1
        if stored:
            self.trace.emit("memory_store", text=f"{stored} fact(s) at session close")
        return stored

    def close(self) -> None:
        self.trace.close()


def _strip_thinking(content: list[dict]) -> list[dict]:
    """Thinking blocks are reasoning gold but cheap noise in the transcript."""
    return [b for b in content if b.get("type") != "thinking"]


def _is_tool_results_message(message: dict) -> bool:
    content = message.get("content")
    return (
        message.get("role") == "user"
        and isinstance(content, list)
        and bool(content)
        and all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    )