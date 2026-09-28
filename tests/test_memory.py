"""Semantic memory: gating, dedup, recall timing, TTL, extraction."""

from __future__ import annotations

import time

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
    for bad in ("好的", "帮我查一下天气", "明天8点开会", "结果是 37.5 度", "你吃饭了吗"):
        store.remember(bad, "w1")
    assert len(store) == 1


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