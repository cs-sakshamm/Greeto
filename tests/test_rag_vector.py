"""Vector + hybrid retrieval checks for the Greeto knowledge base."""

from __future__ import annotations

import time

import pytest

from app.brain.knowledge_base import KnowledgeBase
from app.brain.vector_store import (
    build_embedder,
    load_chunks,
    load_chunks_from_text,
    tables_to_prose,
)


RAG_CONFIG = {
    "rag": {
        "enabled": True,
        "sources": ["knowledge_base"],
        "embedding": {"backend": "hashing", "hashing_dim": 384},
        "retrieval": {
            "top_k": 3,
            "dense_top_k": 8,
            "max_context_chars": 1100,
            "min_confidence": 0.30,
        },
    }
}


@pytest.fixture(scope="module")
def knowledge(tmp_path_factory):
    index_dir = tmp_path_factory.mktemp("rag-index")
    return KnowledgeBase(
        config=RAG_CONFIG,
        index_dir=index_dir,
        embedder=build_embedder("hashing"),
    )


def _context_for(knowledge: KnowledgeBase, question: str) -> str:
    result = knowledge.search_with_confidence(question)
    assert result.context, "no retrieval context for %r" % question
    assert result.confidence > 0.3, result.confidence
    return result.context


def test_helpline_number_is_retrieved_cleanly(knowledge):
    context = _context_for(knowledge, "What is the admission helpline number?")
    assert "1800-180-0840" in context


def test_mba_fee_amount_is_retrieved(knowledge):
    context = _context_for(knowledge, "What is the total fee for the MBA program?")
    assert "2,56,500" in context


def test_branch_discount_is_retrieved(knowledge):
    context = _context_for(knowledge, "Is there a branch wise fee discount?")
    assert "20,000" in context


def test_refund_sop_deadline_is_retrieved(knowledge):
    context = _context_for(knowledge, "How long does the refund process take?")
    assert "14 days" in context


def test_bank_ifsc_identifier_is_retrieved(knowledge):
    context = _context_for(knowledge, "What is the Canara Bank IFSC code?")
    assert "CNRB0018556" in context


def test_atomic_qa_records_do_not_leak_the_next_section_title(knowledge):
    context = _context_for(knowledge, "How can I submit an online admission inquiry?")
    assert "Admission Document Checklist and Verification Rules" not in context


def test_citations_are_stripped_at_ingestion(tmp_path):
    document = tmp_path / "cited.md"
    document.write_text(
        "The helpline is 1800-180-0840.\ue200cite\ue202turn0search3\ue201",
        encoding="utf-8",
    )

    chunks = load_chunks(tmp_path)
    text = "\n".join(chunk.text for chunk in chunks)

    assert "1800-180-0840" in text
    assert "\ue200" not in text
    assert "turn0search3" not in text


def test_html_and_seo_spam_are_removed():
    chunks = load_chunks_from_text(
        "<html><title>PGSLOT situs slot gacor</title>"
        "<body><p>The campus PIN code is 201310.</p></body></html>",
        source="spam.md",
    )

    text = "\n".join(chunk.text for chunk in chunks).casefold()

    assert "201310" in text
    assert "pgslot" not in text
    assert "<p>" not in text
    assert "<html>" not in text


def test_pipe_tables_become_spoken_prose():
    prose = tables_to_prose(
        "| Programme | Fee |\n"
        "| --- | --- |\n"
        "| MBA | 2,56,500 rupees |"
    )

    assert "Programme is MBA" in prose
    assert "Fee is 2,56,500 rupees" in prose


def test_retrieval_p95_stays_inside_the_cpu_budget(knowledge):
    questions = [
        "What is the admission helpline number?",
        "MBA total fee",
        "branch wise fee discount",
        "refund process 14 days",
        "hostel rules",
        "highest placement package",
        "defence scholarship",
        "campus PIN code",
    ] * 5

    # Warm the dense path once; the first call may allocate numpy buffers.
    knowledge.search_with_confidence(questions[0])

    timings = []
    for question in questions:
        started = time.perf_counter()
        knowledge.search_with_confidence(question)
        timings.append((time.perf_counter() - started) * 1000.0)

    timings.sort()
    p95 = timings[int(len(timings) * 0.95)]

    assert p95 < 25.0, "retrieval p95 was %.2f ms" % p95
