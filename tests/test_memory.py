"""Semantic memory: gating, dedup, recall timing, TTL, extraction."""

from __future__ import annotations

import json
import time

from min_agent import memory as memory_module
from min_agent.memory import MemoryStore, extract_facts

from fake_llm import ScriptedLLM, text_block


def _store(tmp_path, *items, tags=()):
    store = MemoryStore(tmp_path / "facts.json")
    for text in items:
        store.remember(text, source_session="w1", tags=tags)
    return store


# --------------------------------------------------------------------------- #
# storage gating
# --------------------------------------------------------------------------- #


def test_gate_rejects_transient_statements(tmp_path):
    store = MemoryStore(tmp_path / "f.json")
    store.remember("用户住在北京市", "w1")  # durable -> stored
    assert len(store) == 1
    for bad in ("好的", "帮我查一下天气", "明天8点开会", "你吃饭了吗", "BA", "我什么时候来?"):
        store.remember(bad, "w1")
    assert len(store) == 1


def test_every_documented_example_passes_the_gate(tmp_path):
    """The gate rejects transient utterances, but the example facts shipped in
    the extraction prompt, the remember tool description and the README must
    all be storable -- otherwise we teach the model facts our own gate refuses.
    """
    store = MemoryStore(tmp_path / "f.json")
    for fact in memory_module._DOCUMENTED_FACTS:
        reason = store._gate(fact)
        assert reason == "", f"documented example rejected: {fact!r} -> {reason}"
        store.remember(fact, "w1")
    assert len(store) == len(memory_module._DOCUMENTED_FACTS)


def test_numeric_facts_are_stored_not_mistaken_for_tool_output(tmp_path):
    """A real number in a fact is not evidence of tool output (the end-of-session
    extraction reads user turns only, so provenance is clean by construction).
    This used to be rejected outright."""
    store = MemoryStore(tmp_path / "f.json")
    store.remember("我月薪 20.5k", "w1")
    assert store.all()[0].text == "我月薪 20.5k"


def test_dedup_by_normalised_text(tmp_path):
    store = MemoryStore(tmp_path / "f.json")
    store.remember("我喜欢 简洁 的回答", "w1")
    status = store.remember("我喜欢简洁的回答", "w1")
    assert status.startswith("already")
    assert len(store) == 1
    assert store.all()[0].hits >= 1


def test_remember_round_trips_tags(tmp_path):
    store = MemoryStore(tmp_path / "f.json")
    store.remember("用户喜欢BA风格", "w1", tags=["preference"])
    assert store.all()[0].tags == ["preference"]


def test_persistence_across_reload(tmp_path):
    path = tmp_path / "facts.json"
    MemoryStore(path).remember("我是后端实习生", "w1")
    reloaded = MemoryStore(path)
    assert [i.text for i in reloaded.all()] == ["我是后端实习生"]


# --------------------------------------------------------------------------- #
# recall: keyword scoring at the start of a user turn
# --------------------------------------------------------------------------- #


def test_recall_ranks_by_term_overlap(tmp_path):
    store = _store(tmp_path, "用户喜欢BA风格", "明天有雨比较担心通勤", "用户住在北京")
    hits = store.recall("用户住在哪？北京吧", top_k=2)
    assert hits[0].text == "用户住在北京"


def test_recall_respects_top_k_and_skip_outside_query(tmp_path):
    store = _store(tmp_path, "用户喜欢BA风格", "用户住在北京")
    hits = store.recall("说说北京", top_k=1)
    assert len(hits) == 1
    assert hits[0].text == "用户住在北京"


def test_recall_honours_ttl(tmp_path):
    store = _store(tmp_path, "用户住在北京")
    store.all()[0].created_at = time.time() - 3 * 86400  # 3 days ago
    assert store.recall("用户住哪", ttl_days=1) == []
    assert len(store.recall("北京", ttl_days=30)) == 1


def test_recall_increments_hit_counter(tmp_path):
    store = _store(tmp_path, "用户住在北京")
    store.recall("北京在哪里")
    assert store.all()[0].hits == 1


# --------------------------------------------------------------------------- #
# the write path: a read must not write, and a write must not clobber
# --------------------------------------------------------------------------- #


def _count_writes(monkeypatch) -> list:
    """Record every atomic write the store performs."""
    calls: list = []
    real = memory_module.atomic_write_text

    def spy(path, text):
        calls.append(path)
        return real(path, text)

    monkeypatch.setattr(memory_module, "atomic_write_text", spy)
    return calls


def test_recall_does_not_touch_disk(monkeypatch, tmp_path):
    """recall() runs at the start of every user turn.

    It used to call save() to bump the hit counter, i.e. rewrite the whole
    facts.json on the critical path of every turn for a number nothing reads
    back.  Read path must stay read-only.
    """
    store = _store(tmp_path, "用户住在北京", "用户喜欢简洁的回答")
    calls = _count_writes(monkeypatch)  # patch after the setup writes
    store.recall("北京", top_k=2)
    assert calls == [], f"recall() wrote to disk: {calls}"


def test_hit_counters_persist_at_flush(tmp_path):
    """Deferring the write must not defer the counter past the window edge."""
    store = _store(tmp_path, "用户住在北京")
    store.recall("北京")
    assert store._dirty, "recall() should have marked the store dirty"
    store.flush()
    assert MemoryStore(tmp_path / "facts.json").all()[0].hits == 1


def test_flush_is_a_no_op_when_nothing_changed(tmp_path):
    store = _store(tmp_path, "用户住在北京")
    before = (tmp_path / "facts.json").read_bytes()
    store.flush()
    assert (tmp_path / "facts.json").read_bytes() == before


def test_two_windows_do_not_erase_each_others_facts(tmp_path):
    """Each window builds its own MemoryStore at startup, so two windows hold
    two divergent in-memory lists of the same file.  Window A loads first, then
    B stores a fact, then A stores one too -- A's list is now stale, and the
    old save() wrote A's list verbatim, silently deleting B's fact.
    """
    path = tmp_path / "facts.json"
    a = MemoryStore(path)
    b = MemoryStore(path)  # second window, also loaded before either write
    a.remember("用户住在北京", "w1")
    b.remember("用户喜欢跑步", "w2")
    a.remember("用户常驻厦门", "w1")  # a is the stale writer now

    texts = {i.text for i in MemoryStore(path).all()}
    assert texts == {"用户住在北京", "用户喜欢跑步", "用户常驻厦门"}
    assert [i.text for i in b.all()] == ["用户住在北京", "用户喜欢跑步"]


def test_corrupt_facts_file_degrades_to_empty(tmp_path):
    """A truncated write must not brick the window; memory is an optimisation."""
    path = tmp_path / "facts.json"
    path.write_text("{not json at all", encoding="utf-8")
    store = MemoryStore(path)
    assert store.all() == []
    store.remember("用户住在北京", "w1")
    assert [i.text for i in MemoryStore(path).all()] == ["用户住在北京"]


def test_row_with_unknown_field_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "facts.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "a",
                    "text": "旧事实",
                    "tags": [],
                    "source_session": "w0",
                    "created_at": 0.0,
                    "hits": 0,
                    "embedding": [0.1],  # a field this version knows nothing about
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert MemoryStore(path).all() == []
    # the stale file is still usable: a new fact can be stored over it
    MemoryStore(path).remember("用户常驻厦门喜欢跑步", "w1")
    assert len(MemoryStore(path).all()) == 1


# --------------------------------------------------------------------------- #
# end-of-session extraction (store timing: window close)
# --------------------------------------------------------------------------- #


def test_extract_facts_parses_llm_json(tmp_path):
    llm = ScriptedLLM(
        [{"content": [text_block('["我住在北京", "喜欢简洁的答案"]')], "stop_reason": "end_turn"}]
    )
    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "我住在北京，喜欢简洁的答案。"},
        {"role": "assistant", "content": "好的，记住了。"},
    ]
    assert extract_facts(messages, llm, max_facts=8) == ["我住在北京", "喜欢简洁的答案"]


def test_extract_facts_tolerates_garbage(tmp_path):
    llm = ScriptedLLM(
        [{"content": [text_block("抱歉，我不能总结。")], "stop_reason": "end_turn"}]
    )
    assert extract_facts([{"role": "user", "content": "你好"}], llm) == []


def test_extract_facts_skips_empty_session(tmp_path):
    llm = ScriptedLLM([{"content": [text_block("[]")], "stop_reason": "end_turn"}])
    assert extract_facts([], llm) == []