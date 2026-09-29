"""CLI-level wiring: a second ``min-agent`` on a live window must fail cleanly.

The unit tests prove the lock refuses a second holder; this proves ``main()``
turns that refusal into a one-line error and exit code 2 instead of a stack
trace.  Two real processes, because per-process exclusion is the point.

Both processes use the real REPO_ROOT workspace (the CLI does not take a
workspace override), under a unique user id, and the test removes the window
directory afterwards.  ``.sessions/`` is gitignored, so the only residue is
removed by the test itself.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid

import pytest

from min_agent.config import Config
from min_agent.store import SessionStore

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(sys.platform == "win32" and os.getenv("CI") == "true", reason="windows CI console encoding")
def test_a_second_cli_on_a_live_window_exits_2():
    scope = f"cli-lock-{uuid.uuid4().hex[:6]}"
    store = SessionStore(Config().sessions_root)
    lock_path = store.lock_path(scope, "main")
    node = store.node_dir(scope, "main")
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time;"
            f"sys.path.insert(0, {_REPO!r});"
            "from pathlib import Path;"
            "from min_agent.paths import FileLock;"
            f"lk = FileLock(Path({str(lock_path)!r}), label='x').acquire();"
            "print('held', flush=True);"
            "time.sleep(30)",
        ],
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        assert holder.stdout.readline().strip() == "held", "holder never acquired"

        run = subprocess.run(
            [sys.executable, "-m", "min_agent", "--user", scope, "--session", "main", "--demo"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        assert run.returncode == 2, run.stdout + run.stderr
        assert "already has this window open" in run.stderr, run.stderr
        assert "Traceback" not in run.stderr, run.stderr
    finally:
        holder.kill()
        holder.wait(timeout=10)
        assert holder.stdout is not None
        holder.stdout.close()
        import shutil

        # node is <user>/main; the unique <user> scope dir is created by the
        # holder's lock and left empty -- remove the whole scope.
        targets = [node.parent, node]
        deadline = time.monotonic() + 10.0
        while targets and time.monotonic() < deadline:
            for t in list(targets):
                if t.exists():
                    try:
                        shutil.rmtree(t)
                    except OSError:
                        time.sleep(0.2)
                        continue
                targets.remove(t)
        if any(t.exists() for t in [node.parent, node]):
            raise RuntimeError(f"could not clean up {node.parent}")