"""Every third-party package we import must be declared in pyproject.toml."""

from __future__ import annotations

import ast
import pathlib
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

THIRD_PARTY = {
    "anthropic": "anthropic",
    "dotenv": "python-dotenv",
    "jsonschema": "jsonschema",
    "httpx": "httpx",
}


def _declared() -> set[str]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return {d.split(">=")[0].split("==")[0].strip() for d in data["project"]["dependencies"]}


def _imported() -> set[str]:
    names: set[str] = set()
    for path in (REPO_ROOT / "min_agent").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    names.add(node.module.split(".")[0])
    return names


def test_third_party_imports_are_declared():
    declared = _declared()
    imported = _imported()
    undeclared = {
        THIRD_PARTY[n] for n in imported if n in THIRD_PARTY and THIRD_PARTY[n] not in declared
    }
    assert not undeclared, f"imported but not declared in pyproject.toml: {sorted(undeclared)}"