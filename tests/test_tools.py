"""Tests for the tool registry, schema validation and the five built-in tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import pytest

from min_agent.errors import ToolError, ToolInputError, ToolNotFoundError
from min_agent.tools import TodoStore, build_todo_tool, build_registry
from min_agent.tools.base import ToolCall, tool
from min_agent.tools.registry import ToolRegistry

# --------------------------------------------------------------------------- #
# Registry mechanics
# --------------------------------------------------------------------------- #


def test_decorator_derives_schema_and_description():
    @tool
    def ping(echo: str) -> str:
        """Echo exactly what is given."""
        return echo

    assert ping.name == "ping"
    assert ping.description == "Echo exactly what is given."
    api = ping.to_api()
    assert api["name"] == "ping"
    assert api["input_schema"]["properties"]["echo"]["type"] == "string"
    assert api["input_schema"]["required"] == ["echo"]


def test_decorator_extracts_annotated_descriptions():
    @tool
    def greet(
        name: Annotated[str, "who to greet"],
        times: Annotated[int, "how many times"] = 1,
    ) -> str:
        """Greet someone."""
        return name * times

    schema = greet.input_schema
    assert schema["properties"]["name"]["description"] == "who to greet"
    assert schema["properties"]["times"]["type"] == "integer"
    assert "times" not in schema.get("required", [])  # has a default -> optional
    assert schema["required"] == ["name"]


def test_container_types_map_to_valid_json_schema():
    """Regression: list[str] mapped to the literal string "array[string]",
    which is not a valid JSON Schema type and made jsonschema.validate raise
    SchemaError (mis-reported as the model's fault).  Containers must emit real
    array/object maps."""
    @tool
    def organize(
        names: list[str],
        index: dict[str, list[str]],
        note: Optional[str] = None,
    ) -> str:
        """Organize things."""
        return str(len(names))

    schema = organize.input_schema
    assert schema["properties"]["names"] == {"type": "array", "items": {"type": "string"}}
    assert schema["properties"]["index"] == {
        "type": "object",
        "additionalProperties": {"type": "array", "items": {"type": "string"}},
    }
    assert schema["properties"]["note"]["type"] == "string"  # Optional unwraps
    import jsonschema

    jsonschema.validate(
        {"names": ["a"], "index": {"x": ["b"]}}, schema
    )  # must not raise SchemaError


def test_broken_schema_surfaces_as_value_error_not_input_error():
    """A schema bug is ours, not the model's: it must raise loudly instead of
    being reported to the model as invalid arguments."""
    from min_agent.tools.base import ToolSpec

    from min_agent.tools.calculator import calculator

    broken = ToolSpec(
        name="broken",
        description="x",
        input_schema={"type": "array[string]"},  # old _json_type output
        fn=lambda: "x",
    )
    reg = ToolRegistry([broken, calculator])
    with pytest.raises(ValueError):
        reg.validate("broken", {})
    res = reg.call("broken", {})  # the loop-facing surface still returns data
    assert res["ok"] is False
    assert "internal error" in res["error"]
    assert "Invalid arguments" not in res["error"]


def test_duplicate_registration_rejected(tmp_path):
    from min_agent.tools.calculator import calculator

    reg = ToolRegistry([calculator])
    with pytest.raises(ValueError):
        reg.register(calculator)


def test_unknown_tool_raises_tool_not_found():
    from min_agent.tools.calculator import calculator

    reg = ToolRegistry([calculator])
    with pytest.raises(ToolNotFoundError):
        reg.get("missing")
    # Through the dispatching surface a missing tool is *data*, not an
    # exception: the loop must feed it back to the model.
    res = reg.call("missing", {})
    assert res["ok"] is False
    assert "Unknown tool" in res["error"]


def test_schema_violation_becomes_struct_error():
    from min_agent.tools.calculator import calculator

    reg = ToolRegistry([calculator])
    res = reg.call("calculator", {})  # expression is required
    assert res["ok"] is False
    assert "expression" in res["error"]
    assert res["tool"] == "calculator"
    assert "latency_ms" in res


def test_tool_exception_becomes_struct_error():
    from min_agent.tools.calculator import calculator

    reg = ToolRegistry([calculator])
    res = reg.call("calculator", {"expression": "1 / 0"})
    assert res["ok"] is False
    assert "Division by zero" in res["error"]


def test_to_api_matches_callable_names():
    from min_agent.tools.calculator import calculator
    from min_agent.tools.search import search
    from min_agent.tools.weather import weather

    reg = ToolRegistry([calculator, weather, search])
    api_names = {t["name"] for t in reg.to_api()}
    assert api_names == {"calculator", "weather", "search"}
    assert reg.names() == ["calculator", "weather", "search"]


def test_result_truncation_limit():
    from min_agent.tools.calculator import calculator

    reg = ToolRegistry([calculator])
    res = reg.call("calculator", {"expression": "9" * 3}, max_result_chars=8)
    assert res["ok"] is True
    assert res["truncated"] is True
    assert len(res["result"]) <= 60  # 8 chars + ellipsis suffix


def test_tool_call_signature_is_stable():
    a = ToolCall(id="1", name="calc", args={"x": 2})
    b = ToolCall(id="2", name="calc", args={"x": 2})
    c = ToolCall(id="3", name="calc", args={"x": 3})
    assert a.signature() == b.signature()
    assert a.signature() != c.signature()


def test_timeout_bounds_wait_not_just_the_raised_error():
    """A hung tool must stop blocking the loop at the budget, not at its own pace.

    Regression: `_invoke` used to `raise` from inside the `with
    ThreadPoolExecutor(...)` block, so `__exit__` ran `shutdown(wait=True)` and
    joined the runaway worker.  The timeout only decided *when the error was
    raised*; the caller still waited for the tool to finish.
    """
    import time

    @tool
    def hang(seconds: Annotated[int, "how long to sleep"] = 5) -> str:
        """Sleep for a while and then give up."""
        time.sleep(seconds)
        return "eventually"

    reg = ToolRegistry([hang])
    started = time.perf_counter()
    res = reg.call("hang", {"seconds": 5}, timeout=0.1)
    elapsed = time.perf_counter() - started

    assert res["ok"] is False
    assert "exceeded" in res["error"]
    assert elapsed < 1.0, f"call() waited {elapsed:.2f}s on a 0.1s budget"


def test_timeout_leaves_a_fast_tool_untouched():
    import time

    @tool
    def quick() -> str:
        """Return immediately."""
        time.sleep(0.01)
        return "fast"
    res = ToolRegistry([quick]).call("quick", {}, timeout=2.0)
    assert res["ok"] is True
    assert res["result"] == "fast"


# --------------------------------------------------------------------------- #
# calculator
# --------------------------------------------------------------------------- #


class TestCalculator:
    def test_basic_arithmetic(self, reg):
        assert reg.call("calculator", {"expression": "(128*37)/4 + 19"})["result"] == "(128*37)/4 + 19 = 1203"

    def test_functions(self, reg):
        assert reg.call("calculator", {"expression": "sqrt(144)"})["result"].endswith("= 12")
        assert reg.call("calculator", {"expression": "min(3,7,1)"})["result"].endswith("= 1")

    def test_constants(self, reg):
        assert reg.call("calculator", {"expression": "pi * 2"})["result"].endswith("= 6.283185307")

    def test_thousands_separator(self, reg):
        res = reg.call("calculator", {"expression": "1,000 + 5"})
        assert res["ok"] is True
        assert res["result"] == "1000 + 5 = 1005"

    def test_division_by_zero_is_error(self, reg):
        res = reg.call("calculator", {"expression": "1/0"})
        assert res["ok"] is False
        assert "Division by zero" in res["error"]

    @pytest.mark.parametrize(
        "expr",
        [
            "__import__('os')",
            "open('x')",
            "2 + 2 apples",
            "9**9**9",
            "{'a': 1}",
            "[1, 2, 3]",
            "(lambda: 1)()",
        ],
    )
    def test_unsafe_or_ill_formed_inputs_rejected(self, reg, expr):
        res = reg.call("calculator", {"expression": expr})
        assert res["ok"] is False


# --------------------------------------------------------------------------- #
# weather (deterministic mock)
# --------------------------------------------------------------------------- #


class TestWeather:
    def test_returns_numbers_for_city(self, reg):
        res = reg.call("weather", {"city": "Shanghai"})
        assert res["ok"] is True
        assert "Shanghai" in res["result"]
        assert "C" in res["result"]

    def test_deterministic_for_same_city_and_date(self, reg):
        a = reg.call("weather", {"city": "Beijing", "date": "2026-01-15"})["result"]
        b = reg.call("weather", {"city": "Beijing", "date": "2026-01-15"})["result"]
        assert a == b

    def test_differs_by_city(self, reg):
        a = reg.call("weather", {"city": "Beijing", "date": "2026-01-15"})["result"]
        b = reg.call("weather", {"city": "Shanghai", "date": "2026-01-15"})["result"]
        assert a != b

    def test_bad_date_rejected(self, reg):
        assert reg.call("weather", {"city": "x", "date": "14/05/2026"})["ok"] is False


class TestWeatherBackends:
    def test_default_is_mock(self, tmp_path):
        from min_agent.config import Config
        from min_agent.tools import ToolContext

        cfg = Config(workspace=tmp_path)
        spec = __import__("min_agent.tools", fromlist=["make_weather_tool"]).make_weather_tool(
            cfg.weather_backend
        )
        res = ToolRegistry([spec]).call("weather", {"city": "Shanghai"})
        assert res["ok"] is True
        assert "mock-forecast" in res["result"]

    def test_wttr_in_parses_response(self, monkeypatch, tmp_path):
        import importlib

        wmod = importlib.import_module("min_agent.tools.weather")

        fake = {
            "weather": [
                {
                    "date": "2026-06-15",
                    "maxtempC": "31",
                    "mintempC": "24",
                    "maxtempF": "88",
                    "mintempF": "75",
                    "avghumidity": 78,
                    "maxwindspeedKmph": "19",
                    "hourly": [{"weatherDesc": [{"value": "多云"}]}],
                }
            ],
            "current_condition": [{"humidity": "80", "windspeedKmph": "12", "weatherDesc": [{"value": "多云"}]}],
        }

        class _FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return fake

        def _fake_get(url, timeout=None, follow_redirects=None):
            assert "wttr.in" in url
            return _FakeResponse()

        monkeypatch.setattr("httpx.get", _fake_get)
        row = wmod.fetch_wttr_in("Shanghai", __import__("datetime").date(2026, 6, 15))
        assert row["temp_high_c"] == 31.0
        assert row["condition"] == "多云"
        assert row["source"] == "wttr.in"

    def test_wttr_in_falls_back_when_unreachable(self, monkeypatch, tmp_path):
        from min_agent.config import Config

        def _boom(url, timeout=None, follow_redirects=None):
            raise ConnectionError("simulated offline")

        monkeypatch.setattr("httpx.get", _boom)
        spec = __import__("min_agent.tools", fromlist=["make_weather_tool"]).make_weather_tool(
            "wttr.in", timeout=0.5
        )
        res = ToolRegistry([spec]).call("weather", {"city": "Shanghai"})
        assert res["ok"] is True
        assert "fallback" in res["result"]
        assert "mock-forecast" in res["result"]


# --------------------------------------------------------------------------- #
# search (mock but ranked)
# --------------------------------------------------------------------------- #


class TestSearch:
    def test_returns_ranked_results(self, reg):
        res = reg.call("search", {"query": "agent memory context", "limit": 3})
        assert res["ok"] is True
        assert "arxiv.org" in res["result"]
        assert "1." in res["result"]

    def test_no_results(self, reg):
        res = reg.call("search", {"query": "zzzz nonexistent qqqqq"})
        assert res["ok"] is True
        assert "No results" in res["result"]

    def test_empty_query_rejected(self, reg):
        assert reg.call("search", {"query": "   "})["ok"] is False

    def test_limit_is_clamped(self, reg):
        res = reg.call("search", {"query": "agent", "limit": 99})
        assert res["ok"] is True

    def test_package_search_name_is_the_module_not_the_tool(self):
        """Regression: tools/__init__.py used to re-export the search ToolSpec under
        the name ``search``, shadowing the submodule, so ``from min_agent.tools
        import search`` returned a ToolSpec and you could not reach helpers like
        ``search_corpus`` through it."""
        from min_agent.tools import search

        assert hasattr(search, "search_corpus")
        assert hasattr(search, "search")

    def test_results_are_ranked_real_bm25(self):
        """Regression: the old scorer was overlap/sqrt(len), not BM25, so a
        short doc with one rare term could outrank a long doc that actually
        matches more.  BM25 must prefer docs whose terms recur and appear
        rarely elsewhere."""
        from min_agent.tools.search import search_corpus

        first = search_corpus("planning module memory module")
        assert first[0]["title"] == "A Survey on Large Language Model based Autonomous Agents"

    def test_snippet_is_centered_on_the_hit(self):
        """Regression: the snippet used a *token* index as a *character* index
        ("1 token == 1 char for our corpus"), so it started far before the hit.
        The hit's text must fall inside the snippet window."""
        from min_agent.tools.search import search_corpus

        result = search_corpus("reflexion")[0]
        assert "Reflexion" in result["snippet"]

    def test_snippet_start_and_end_ellipsis_flags_truncation(self):
        from min_agent.tools.search import _snippet

        text = ("word " * 200) + "needle" + (" tail " * 200)
        snippet = _snippet(text, {"needle"})
        assert snippet.startswith("...")
        assert snippet.endswith("...")
        assert "needle" in snippet


# --------------------------------------------------------------------------- #
# todo (per-session state)
# --------------------------------------------------------------------------- #


class TestTodo:
    def test_add_and_list(self, tmp_path):
        spec, store = _todo(tmp_path)
        assert reg_result(spec, "add", item="查天气")["ok"]
        assert reg_result(spec, "add", item="写周报")["ok"]
        listing = reg_result(spec, "list")
        assert listing["ok"] and json.loads(listing["result"])["pending"] == 2
        assert store.path.exists()

    def test_persists_across_reload(self, tmp_path):
        spec, _ = _todo(tmp_path)
        reg_result(spec, "add", item="买牛奶")
        _, reloaded = _todo(tmp_path)  # new store over the same file
        assert [i.text for i in reloaded.items] == ["买牛奶"]

    def test_complete_by_id_and_keyword(self, tmp_path):
        spec, _ = _todo(tmp_path)
        reg_result(spec, "add", item="完成注册")
        res = reg_result(spec, "done", id=1)
        assert res["ok"] and "Completed" in res["result"]
        # id 1 is done; completing again by keyword fails
        assert reg_result(spec, "done", text_contains="注册")["ok"] is False

    def test_invalid_action(self, tmp_path):
        spec, _ = _todo(tmp_path)
        assert reg_result(spec, "explode")["ok"] is False

    def test_todo_isolation_between_sessions(self, tmp_path):
        """Two windows must not see each other's todos -- this is the isolation
        test the brief cares about, at the state layer."""
        spec_a, _ = _todo(tmp_path / "a")
        spec_b, _ = _todo(tmp_path / "b")
        reg_result(spec_a, "add", item="窗口1的待办")
        listing_b = reg_result(spec_b, "list")
        assert json.loads(listing_b["result"])["pending"] == 0


# --------------------------------------------------------------------------- #
# full registry (session-scoped)
# --------------------------------------------------------------------------- #


def test_build_registry_has_all_default_tools(tmp_path):
    from min_agent.config import Config

    cfg = Config(workspace=tmp_path, docs_dir=tmp_path / "docs")
    reg, store = build_registry(
        __import__("min_agent.tools", fromlist=["ToolContext"]).ToolContext(
            session_id="w1", session_dir=tmp_path / "w1", config=cfg
        )
    )
    names = set(reg.names())
    assert {"calculator", "search", "weather", "todo", "read_docs"} <= names
    assert isinstance(store, TodoStore)


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #


def reg_result(spec, action, **kw):
    reg = ToolRegistry([spec])
    return reg.call("todo", {"action": action, **kw})


def _todo(path):
    from min_agent.tools.todo import make_todo_tool

    if Path(path).suffix == "":  # a directory -> append the file name
        path = path / "todo.json"
    return make_todo_tool(path)


@pytest.fixture()
def reg():
    from min_agent.tools.calculator import calculator
    from min_agent.tools.search import search
    from min_agent.tools.weather import weather

    return ToolRegistry([calculator, search, weather])


# --------------------------------------------------------------------------- #
# schema: Union and Literal
# --------------------------------------------------------------------------- #


def test_union_type_maps_to_any_of():
    """PEP 604 (X | Y) and typing.Union both used to collapse to {"type":"string"},
    silently lying to the model about what the tool accepts."""
    from typing import Union

    from min_agent.tools.base import _schema_for

    assert _schema_for(Union[str, int]) == {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert _schema_for(int | str) == {"anyOf": [{"type": "integer"}, {"type": "string"}]}


def test_optional_unwraps_to_inner_type():
    from min_agent.tools.base import _schema_for

    assert _schema_for(Optional[str]) == {"type": "string"}


def test_literal_maps_to_enum():
    from typing import Literal

    from min_agent.tools.base import _schema_for

    assert _schema_for(Literal["a", "b"]) == {"enum": ["a", "b"]}


def test_validation_hint_lists_field_names_not_the_whole_schema():
    """The hint used to dump the entire input_schema dict (Python repr, single
    quotes) into the model's error -- unreadable and unactionable."""
    from min_agent.tools.calculator import calculator

    reg = ToolRegistry([calculator])
    res = reg.call("calculator", {})
    assert res["ok"] is False
    assert "expression" in res["error"]
    assert "Required argument" in res["error"]
    assert "input_schema" not in res["error"]