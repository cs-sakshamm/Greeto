"""Grounding, streaming and stage-timing checks.

These tests do not load llama.cpp or play audio. They use a chunked fake
LLM and a recording TTS double, so they assert the orchestration contract:

* streamed sentences reach TTS before generation finishes;
* stage timestamps are recorded;
* invented numbers are replaced with the grounded helpline answer;
* a factual question with no retrieved evidence never reaches the LLM.
"""

from __future__ import annotations

import logging
import threading
import time

import pytest

import app.brain.conversation_manager as conversation_module
from app.brain.conversation_manager import ConversationManager
from app.brain.knowledge_base import RetrievalResult
from app.brain.parser import ResponseParser
from app.brain.prompt_manager import PromptManager
from app.brain.streaming import ReplyStreamExtractor, SentenceChunker
from app.core.event_bus import EventBus


class DummyContainer:
    def __init__(self):
        self.event_bus = EventBus()
        self.logger = logging.getLogger("grounding-tests")
        self.config = None


class RecordingTTS:
    def __init__(self):
        self.spoken = []
        self._lock = threading.Lock()

    def is_speaking(self) -> bool:
        return False

    def cancel(self):
        return None

    def speak(self, text, turn_id=None):
        with self._lock:
            self.spoken.append((text, time.perf_counter(), turn_id))
        return True


class GroundedKB:
    confidence_gate = True

    def __init__(self, context: str = "", confidence: float = 0.9):
        self.context = context
        self.confidence = confidence

    def search_with_confidence(self, query: str) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            context=self.context,
            confidence=self.confidence,
            matches=[],
        )

    def search(self, query: str) -> str:
        return self.context


class RefusingLLM:
    """Return the sanctioned refusal even though the context has the answer."""

    def stream_generate(self, prompt):
        yield (
            '{"reply": "I couldn\'t find that information in the college '
            'database. Please contact the admission office at '
            '1800-180-0840.", "emotion": "empathetic", "action": "answer"}'
        )


class ChunkedLLM:
    """JSON reply split across tokens with a deliberate pause per chunk."""

    PARTS = (
        '{"reply": "The campus is in Greater Noida.',
        " The admission helpline is 1800-180-0840.",
        ' Please contact the office for details.", '
        '"emotion": "neutral", "action": "answer"}',
    )

    def __init__(self, delay: float = 0.12):
        self.delay = delay

    def stream_generate(self, prompt):
        for part in self.PARTS:
            time.sleep(self.delay)
            yield part


def build_manager(monkeypatch, tts, llm, knowledge):
    monkeypatch.setattr(
        conversation_module,
        "KnowledgeBase",
        lambda config=None: knowledge,
    )
    return ConversationManager(DummyContainer(), tts_service=tts, llm=llm)


def test_streamed_sentences_reach_tts_before_generation_finishes(monkeypatch):
    tts = RecordingTTS()
    manager = build_manager(
        monkeypatch,
        tts,
        ChunkedLLM(delay=0.12),
        GroundedKB(
            context="The campus is in Greater Noida. "
            "The admission helpline is 1800-180-0840.",
            confidence=0.92,
        ),
    )

    responses = []
    manager.event_bus.subscribe(
        "llm_response_ready",
        lambda **payload: responses.append(payload["response"]),
    )

    manager.process_message("Where is the campus and what is the helpline?")

    turn_id = manager._current_turn_id
    timing = manager._timings[turn_id]

    assert len(tts.spoken) >= 2, tts.spoken
    assert "Greater Noida" in tts.spoken[0][0]
    assert tts.spoken[0][1] < timing["llm_done"], (
        "first sentence was not spoken until generation finished"
    )
    assert timing["llm_first_token"] <= timing["llm_done"]
    assert timing["speech_end"] <= timing["stt_done"] <= timing["kb_done"]
    assert len(responses) == 1
    assert "1800-180-0840" in responses[0].reply


def test_confident_evidence_replaces_a_model_refusal(monkeypatch):
    tts = RecordingTTS()
    manager = build_manager(
        monkeypatch,
        tts,
        RefusingLLM(),
        GroundedKB(
            context=(
                "Q: What is the campus PIN code?\n"
                "A: The campus is at Plot 46, Knowledge Park-III, "
                "Greater Noida, PIN code 201310."
            ),
            confidence=0.80,
        ),
    )

    manager.process_message("What is the campus PIN code?")

    assert tts.spoken
    spoken_texts = [text for text, _, _ in tts.spoken]
    assert any("201310" in text for text in spoken_texts)
    assert not any("couldn't find" in text.casefold() for text in spoken_texts)


def test_low_confidence_factual_question_never_reaches_the_llm(monkeypatch):
    class ExplodingLLM:
        def stream_generate(self, prompt):
            raise AssertionError("the LLM must not be called without evidence")

    tts = RecordingTTS()
    manager = build_manager(
        monkeypatch,
        tts,
        ExplodingLLM(),
        GroundedKB(context="", confidence=0.0),
    )

    manager.process_message("What is the fee for the MBBS programme?")

    assert len(tts.spoken) == 1
    spoken, _, _ = tts.spoken[0]
    assert "couldn't find" in spoken.lower()
    assert "1800-180-0840" in spoken


@pytest.mark.parametrize(
    "reply, context",
    [
        ("The B.Tech CSE fee is ₹1,00,000 per year.", "Fees depend on the programme."),
        ("The placement rate is 97 percent.", "The placement rate is published annually."),
        ("The next session begins in 2027.", "The academic calendar is published annually."),
        ("Call the office at 9999999999.", "Call the admission office for details."),
    ],
)
def test_invented_numbers_are_replaced(reply, context):
    grounded = ConversationManager._ground_reply(reply, context)

    assert grounded == ConversationManager.GROUNDED_FALLBACK
    assert "1800-180-0840" in grounded


def test_numbers_present_in_context_are_kept():
    reply = "The MBA total fee is 2,56,500 rupees, payable in four installments."
    context = "The total fee for the MBA program is 2,56,500 rupees."

    assert ConversationManager._ground_reply(reply, context) == reply


def test_sanctioned_helpline_answer_is_always_grounded():
    fallback = ConversationManager.GROUNDED_FALLBACK

    assert ConversationManager._ground_reply(fallback, "") == fallback


def test_parser_strips_your_answer_prefix():
    parser = ResponseParser()

    assert parser.parse("Your answer: The campus PIN code is 201310.").reply == (
        "The campus PIN code is 201310."
    )
    assert parser.parse(
        '{"reply": "Your answer: The campus PIN code is 201310.", '
        '"emotion": "neutral", "action": "answer"}'
    ).reply == "The campus PIN code is 201310."


def test_history_is_omitted_for_standalone_factual_questions():
    prompt_manager = PromptManager(style="qwen")
    history = [
        {"role": "user", "content": "What is the MBA fee?"},
        {"role": "assistant", "content": "The MBA fee is 2,56,500 rupees."},
    ]

    standalone = prompt_manager.build_prompt(
        "What documents are required for B.Tech admission?",
        history,
        "Q: What documents are required?\nA: Bring mark sheets.",
    )
    followup = prompt_manager.build_prompt(
        "What about the MBA fee?",
        history,
        "Q: What is the MBA fee?\nA: 2,56,500 rupees.",
    )

    assert "The MBA fee is 2,56,500 rupees." not in standalone
    assert "The MBA fee is 2,56,500 rupees." in followup


def test_bare_phone_number_becomes_a_spoken_sentence():
    spoken = ConversationManager._prepare_spoken_answer("1800-180-0840")

    assert spoken == "The admission helpline number is 1800-180-0840."


def test_reply_extractor_decodes_escaped_json_tokens():
    extractor = ReplyStreamExtractor()
    tokens = [
        '{"reply": "The fee is 2,56,500 rupees.',
        ' It is payable in four installments.\\nContact 1800-180-0840.',
        '", "emotion": "neutral", "action": "answer"}',
    ]

    decoded = "".join(extractor.feed(token) for token in tokens)

    assert extractor.reply_started
    assert "2,56,500" in decoded
    assert "installments." in decoded
    assert "\\n" not in decoded
    assert "1800-180-0840" in decoded


def test_reply_extractor_passes_plain_prose_through():
    extractor = ReplyStreamExtractor()

    first = extractor.feed("The campus is in Greater Noida.")
    second = extractor.feed(" It has a central library.")

    assert "Greater Noida" in first
    assert "central library" in second


def test_sentence_chunker_respects_abbreviations_and_closes_the_tail():
    chunker = SentenceChunker(min_chars=8, max_chars=240)

    sentences = chunker.feed(
        "The fee is Rs. 2,56,500 per year. It is payable yearly."
    )
    tail = chunker.flush()

    assert sentences == ["The fee is Rs. 2,56,500 per year.", "It is payable yearly."]
    assert tail == ""


def test_sentence_chunker_does_not_stall_on_unpunctuated_output():
    chunker = SentenceChunker(min_chars=8, max_chars=40)

    sentences = chunker.feed("word " * 30)
    tail = chunker.flush()

    assert sentences
    assert all(len(sentence) <= 40 for sentence in sentences)
    assert len(tail) <= 40
