"""Command-line entry point: ``min-agent``.

One user, many windows. A window is session with its own transcript, todo
list and trace; two processes may talk in two *different* windows at once, and
the store hands out one writer per window (see
:class:`~min_agent.store.SessionStore`) so a second process on the same window
is refused instead of quietly clobbering the first one's turns.  ``patrol()``
runs on close so the long-term memory sweep happens exactly when a window ends.
"""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .paths import LockBusy
from .store import SessionStore


def _print(*parts, **kw):
    print(*parts, flush=True, **kw)


def build_parser() -> argparse.ArgumentParser:
    # --user / --session belong to both the main command and every
    # subcommand, so argparse will accept them *before or after* the
    # subcommand word (e.g. `min-agent --user alice trace --session w`).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--user", default="local", help="which user's windows to use")
    common.add_argument(
        "--session",
        default="main",
        help="window (session) id; each is an independent conversation",
    )

    parser = argparse.ArgumentParser(
        prog="min-agent",
        parents=[common],
        description="A minimal agent: ReAct loop, tools, sessions, memory, traces.",
    )
    parser.add_argument("--demo", action="store_true", help="run the built-in demo script")
    parser.add_argument("--env", default=".env", help="path to the .env file")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("sessions", parents=[common], help="list this user's windows")
    sub.add_parser("trace", parents=[common], help="Pretty-print a window's JSONL trace")
    sub.add_parser("tools", parents=[common], help="list the tools available to the model")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config()
    store = SessionStore(config.sessions_root)

    if args.command == "sessions":
        return _cmd_sessions(store, args)
    if args.command == "trace":
        return _cmd_trace(config, args)
    if args.command == "tools":
        return _cmd_tools(config, args)

    from .loop import Agent
    from .trace import Tracer

    try:
        opened = store.open(args.user, args.session, create=True)
    except LockBusy as exc:
        # Expected, and the user's mistake to fix, so: one line, exit 2.  A
        # traceback here would read like the tool is broken.
        print(f"[min-agent] {exc}", file=sys.stderr)
        return 2
    trace = Tracer(
        args.session,
        traces_root=config.traces_root,
        prefix=f"{args.user}.",
        console=not args.demo,
    )
    agent = Agent(config=config, user=args.user, session_id=args.session, session=opened, trace=trace)
    _print(f"[min-agent] window {args.session!r} for user {args.user!r} is open.")
    _print(f"[min-agent] tools: {', '.join(sorted(agent.registry.names()))}")
    try:
        if args.demo:
            _demo(agent)
        else:
            _repl(agent)
    finally:
        stored = agent.patrol()
        if stored:
            _print(f"[min-agent] stored {stored} fact(s) into long-term memory on close.")
        agent.close()
    return 0


# --------------------------------------------------------------------------- #
# REPL / demo
# --------------------------------------------------------------------------- #
_META = {
    "/quit": "leave the window (facts are extracted into long-term memory)",
    "/help": "this list",
    "/tools": "list the tools the model can call",
    "/session": "show this window's state (messages, summary, todo)",
}


def _repl(agent) -> None:
    while True:
        try:
            line = input("you> ")
        except (EOFError, KeyboardInterrupt):
            _print()
            _print("[min-agent] bye.")
            return
        line = line.strip()
        if not line:
            continue
        if line in ("/quit", "/exit", "/bye"):
            _print("[min-agent] bye.")
            return
        if line == "/help":
            for k, v in _META.items():
                _print(f"  {k:<10} {v}")
            continue
        if line == "/tools":
            for name in sorted(agent.registry.names()):
                spec = agent.registry.get(name)
                _print(f"  {name:<12} {spec.description}")
            continue
        if line == "/session":
            _print(f"  session: {agent.session.meta.id!r}  turns={agent.session.meta.turn_count}")
            _print(f"  tool calls so far: {agent.session.meta.tool_calls}")
            _print(f"  summary: {agent.session.meta.summary or '(none yet)'}")
            continue
        result = agent.run_turn(line)
        if result.error:
            _print(f"[min-agent] warning: {result.error}")


def _demo(agent) -> None:
    """A short canned conversation that exercises the tool pool."""
    prompts = [
        "你好，请用一句话介绍你自己",
        "计算 128*4 等于多少",
        "在待办里记下：周五前完成周报",
        "查一下明天上海会不会下雨",
    ]
    for prompt in prompts:
        _print(f"you> {prompt}")
        result = agent.run_turn(prompt)
        _print("ai>  ", result.text or "(no reply)")


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #
def _cmd_sessions(store, args) -> int:
    rows = store.list_sessions(args.user)
    if not rows:
        _print(f"[min-agent] no windows yet for user {args.user!r}.")
        return 0
    _print(f"windows for user {args.user!r} (newest first):")
    for row in rows:
        _print(f"  {row['id']:<20} turns={row.get('turn_count', 0):<5} summary={row.get('summary') or ''}")
    return 0


def _cmd_trace(config, args) -> int:
    from .paths import safe_segment
    from .trace import read_trace

    # The tracer sanitises when it *writes*; the reader must apply the same
    # rule or a raw id (../.., windows device name) simply fails to find the
    # file it just created.
    path = config.traces_root / f"{safe_segment(args.user)}.{safe_segment(args.session)}.jsonl"
    if not path.exists():
        _print(f"[min-agent] no trace at {path.name} for user {args.user!r} / session {args.session!r}.")
        return 1
    for event in read_trace(path):
        _print(_format_event(event))
    return 0


def _format_event(event: dict) -> str:
    kind = event.get("kind", "?")
    parts = [f"[{kind:<12}]"]
    for key in ("name", "tool", "reason", "status", "text", "result", "summary"):
        value = event.get(key)
        if value is None:
            continue
        parts.append(f"{key}={str(value)[:160]}")
        break  # show the first meaningful payload only
    return "  ".join(parts)


def _cmd_tools(config, args) -> int:
    from .memory import MemoryStore
    from .tools import ToolContext, build_registry

    session_dir = config.sessions_root / "cli" / "tools-preview"
    session_dir.mkdir(parents=True, exist_ok=True)
    ctx = ToolContext(session_id="cli", session_dir=session_dir, config=config)
    # `memory` is what makes `remember` appear; without it this command listed
    # five tools while the running agent had six.
    memory = MemoryStore(config.memory_root / "cli" / "facts.json")
    registry, _ = build_registry(ctx, memory=memory)
    for name in sorted(registry.names()):
        spec = registry.get(name)
        schema = spec.to_api().get("input_schema", {})
        props = ", ".join(schema.get("properties", {}).keys()) or "(no params)"
        _print(f"{name:<12} {spec.description}")
        _print(f"            params: {props}")
    return 0


if __name__ == "__main__":
    sys.exit(main())