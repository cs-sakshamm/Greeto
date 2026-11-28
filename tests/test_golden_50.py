"""Fast regression gate for the 50-question spoken golden suite.

Retrieval and grounding run deterministically with the hashed embedder, so
this test needs no PyTorch, Qwen GGUF, Piper voice, microphone or network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.brain.knowledge_base import KnowledgeBase
from app.brain.query_normalizer import normalize_query
from eval.golden_50_eval import (
    build_knowledge,
    load_queries,
    score_retrieval_case,
    summarise,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def golden_kb(tmp_path_factory) -> KnowledgeBase:
    index_dir = tmp_path_factory.mktemp("golden-50-index")
    return build_knowledge("hashing", str(index_dir))


def test_golden_dataset_has_50_balanced_questions(golden_kb):
    payload = load_queries()
    cases = payload["queries"]

    assert len(cases) == 50
    assert sum(1 for case in cases if case["type"] == "factual") == 35
    assert sum(1 for case in cases if case["type"] == "ambiguous") == 5
    assert sum(1 for case in cases if case["type"] == "adversarial") == 10

    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))

    known_qa = {
        chunk.qa_id for chunk in golden_kb.chunks if chunk.qa_id
    }
    for case in cases:
        for qa_id in case.get("expected_qa_ids") or []:
            assert qa_id in known_qa, "%s references missing %s" % (case["id"], qa_id)


def test_golden_50_recall_and_grounding_gates(golden_kb):
    payload = load_queries()
    rows = [score_retrieval_case(golden_kb, case) for case in payload["queries"]]
    summary = summarise(rows, payload, "pytest")

    assert summary["top3_recall"] >= 0.96, summary["failures"]
    assert summary["factual_accuracy"] >= 0.95, summary["failures"]
    assert summary["adversarial_accuracy"] >= 0.90, summary["failures"]
    assert summary["latency"]["kb_ms"]["p95"] < 20.0, summary["latency"]


def test_voice_corpus_metadata_is_complete():
    path = ROOT / "knowledge_base" / "greeto_voice_kb.jsonl"
    assert path.exists(), "run scripts/enrich_kb_for_voice.py first"

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 113
    for row in rows:
        assert row["kb_id"].startswith("KB-")
        assert row["category"]
        assert row["effective_session"] == "2026-27"
        assert row["volatility"] in {"annual", "static"}
        assert 3 <= len(row["search_aliases"]) <= 5
        assert row["spoken_word_count"] <= 60
        assert row["spoken_answer"].endswith((".", "!", "?"))


def test_query_normalizer_expands_branch_and_asr_aliases():
    assert "computer science and engineering" in normalize_query("cs seats")
    assert "artificial intelligence and machine learning" in normalize_query("aiml")
    assert "hostel" in normalize_query("hostile rules")
    assert "its engineering college" in normalize_query("i t s address")
    assert "btech lateral entry second year" in normalize_query("lateral entry")
    assert "transfer certificate" in normalize_query("tc required")

    # The pronoun must not be mistaken for Mechanical Engineering.
    assert "mechanical" not in normalize_query("tell me about the hostel")
