"""Regression tests for the second round of live-kiosk fixes.

Covers three failures seen on the real kiosk:

* streamed and final answers leaking prompt scaffolding such as
  ``Your answer:`` into TTS, captions, logs or history;
* general conversation being answered with the strict college prompt, so
  small talk became a database refusal;
* Saksham never being greeted because a single low-scoring live match was
  neither diagnosed nor confirmed over consecutive frames.
"""

from __future__ import annotations

import logging

import pytest

import app.brain.conversation_manager as conversation_module
from app.brain.conversation_manager import ConversationManager
from app.brain.knowledge_base import RetrievalResult
from app.brain.llm import LocalLLM
from app.brain.parser import ResponseParser
from app.brain.prompt_manager import PromptManager
from app.brain.streaming import (
    ReplyPrefixFilter,
    ReplyStreamExtractor,
    SentenceChunker,
    strip_reply_prefix,
)
from app.core.event_bus import EventBus
from app.gui.kiosk_presenter import KioskPresenter
from app.vision.enrollment import EnrollmentError
from app.vision.face_recognizer import FaceRecognizer
from app.vision.recognition import RecognitionConfig
from app.vision.vision_service import VisionService
from scripts.enroll_face import self_match_scores, verify_self_match

from tests.test_recognition import (
    FakeCamera,
    FakeContainer,
    FakeDatabase,
    FakeModel,
    FakeVisitorDatabase,
    face,
    unit,
)


# ----------------------------------------------------------------------
# Fix 1: no scaffolding prefixes in streaming or final replies
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw",
    [
        "Your answer: The campus PIN code is 201310.",
        "Answer: The campus PIN code is 201310.",
        "ANSWER: The campus PIN code is 201310.",
        "Assistant: The campus PIN code is 201310.",
        "Greeto: The campus PIN code is 201310.",
        "Reply: The campus PIN code is 201310.",
        "Response: The campus PIN code is 201310.",
        "A: The campus PIN code is 201310.",
        "Greeto: Answer: The campus PIN code is 201310.",
    ],
)
def test_strip_reply_prefix_removes_every_scaffolding_label(raw):
    assert strip_reply_prefix(raw) == "The campus PIN code is 201310."


def test_strip_reply_prefix_keeps_a_real_sentence_starting_with_a():
    text = "A good place to study on campus is the central library."

    assert strip_reply_prefix(text) == text


@pytest.mark.parametrize(
    "raw",
    [
        "Your answer: The campus PIN code is 201310.",
        "Assistant: The campus PIN code is 201310.",
        "A: The campus PIN code is 201310.",
    ],
)
def test_parser_never_returns_a_scaffolding_prefix(raw):
    reply = ResponseParser().parse(raw).reply

    assert reply == "The campus PIN code is 201310."

    json_reply = ResponseParser().parse(
        '{"reply": "%s", "emotion": "neutral", "action": "answer"}' % raw
    ).reply

    assert json_reply == "The campus PIN code is 201310."


def test_streamed_tokens_never_emit_your_answer_before_tts():
    extractor = ReplyStreamExtractor()
    chunker = SentenceChunker(min_chars=8, max_chars=240)

    tokens = [
        '{"reply": "Your',
        " answer: The campus",
        " PIN code is 201310.",
        ' The office is open on weekdays.',
        '", "emotion": "neutral", "action": "answer"}',
    ]

    decoded = ""
    sentences = []
    for token in tokens:
        for chunk in [extractor.feed(token)]:
            decoded += chunk
            sentences.extend(chunker.feed(chunk))
    sentences.extend([chunker.flush()])
    sentences = [sentence for sentence in sentences if sentence]

    assert sentences
    assert "Your answer" not in decoded
    assert sentences[0] == "The campus PIN code is 201310."
    assert all("Your answer" not in sentence for sentence in sentences)


def test_plain_stream_starting_with_a_is_not_swallowed():
    extractor = ReplyStreamExtractor()

    first = extractor.feed("A good place to study")
    second = extractor.feed(" is the central library.")

    assert first.startswith("A good")
    assert "central library" in second


def test_prefix_filter_settles_immediately_for_ordinary_text():
    prefix_filter = ReplyPrefixFilter()

    assert prefix_filter.feed("The campus") == "The campus"
    assert prefix_filter.feed(" is in Noida.") == " is in Noida."


def test_prefix_filter_holds_back_partial_label_until_flush():
    prefix_filter = ReplyPrefixFilter()

    assert prefix_filter.feed("Your") == ""
    assert prefix_filter.flush() == "Your"


# ----------------------------------------------------------------------
# Fix 2: general conversational lane
# ----------------------------------------------------------------------

def test_general_prompt_has_no_knowledge_block_or_denial():
    prompt = PromptManager(style="qwen").build_prompt(
        "What is the capital of France?",
        history=[],
        context="Q: What is the MBA fee?\nA: 2,56,500 rupees.",
        lane="general",
    )

    assert "College knowledge" not in prompt
    assert "No relevant college knowledge found" not in prompt
    assert "Retrieved Knowledge" not in prompt
    assert "What is the capital of France?" in prompt
    assert "1800-180-0840" in prompt
    assert '"reply": "your answer"' not in prompt


def test_general_prompt_includes_recent_conversation():
    prompt = PromptManager(style="qwen").build_prompt(
        "And what about the weather there?",
        history=[
            {"role": "user", "content": "I am planning to visit Paris."},
            {"role": "assistant", "content": "Paris is lovely in spring."},
        ],
        lane="general",
    )

    assert "Conversation so far" in prompt
    assert "Paris is lovely in spring." in prompt


def test_college_prompt_still_uses_retrieved_knowledge():
    prompt = PromptManager(style="qwen").build_prompt(
        "What is the MBA fee?",
        history=[],
        context="Q: What is the MBA fee?\nA: The total fee is 2,56,500 rupees.",
        lane="college",
    )

    assert "College knowledge" in prompt
    assert "2,56,500" in prompt


def test_system_prompts_do_not_prime_the_your_answer_label():
    college = PromptManager.system_prompt()
    general = PromptManager.general_system_prompt()

    assert "your answer" not in college.casefold()
    assert "your answer" not in general.casefold()
    assert "I couldn't find that information" in college
    assert "1800-180-0840" in general


def test_general_lane_uses_softer_generation_settings():
    llm = LocalLLM.__new__(LocalLLM)
    llm.max_tokens = 96
    llm.temperature = 0.2
    llm.top_p = 0.8
    llm.repeat_penalty = 1.05
    llm.prompt_style = "qwen"
    llm.general_sampling = {
        "max_tokens": 140,
        "temperature": 0.45,
        "top_p": 0.9,
        "repeat_penalty": 1.12,
    }

    strict = llm._sampling_options(lane="college")
    general = llm._sampling_options(lane="general")

    assert strict["temperature"] == 0.2
    assert strict["max_tokens"] == 96
    assert general["temperature"] == 0.45
    assert general["max_tokens"] == 140
    assert general["repeat_penalty"] == 1.12


class RecordingContainer:
    def __init__(self):
        self.event_bus = EventBus()
        self.logger = logging.getLogger("round2-tests")
        self.config = None


class RecordingTTS:
    def __init__(self):
        self.spoken = []

    def is_speaking(self) -> bool:
        return False

    def cancel(self):
        return None

    def speak(self, text, turn_id=None):
        self.spoken.append(text)
        return True


class RecordingLLM:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []
        self.lanes = []

    def stream_generate(self, prompt, cancel_event=None, lane="college"):
        self.prompts.append(prompt)
        self.lanes.append(lane)
        yield '{"reply": "%s", "emotion": "happy", "action": "answer"}' % (
            self.reply
        )


class StubKB:
    confidence_gate = True

    def __init__(self, context="", confidence=0.0):
        self.context = context
        self.confidence = confidence

    def search_with_confidence(self, query):
        return RetrievalResult(
            query=query,
            context=self.context,
            confidence=self.confidence,
            matches=[],
        )

    def search(self, query):
        return self.context


def build_manager(monkeypatch, tts, llm, knowledge):
    monkeypatch.setattr(
        conversation_module,
        "KnowledgeBase",
        lambda config=None: knowledge,
    )
    return ConversationManager(
        RecordingContainer(), tts_service=tts, llm=llm
    )


def test_general_question_uses_general_lane_without_knowledge(monkeypatch):
    tts = RecordingTTS()
    llm = RecordingLLM(
        "Paris is the capital of France. It is famous for its art and history."
    )
    manager = build_manager(monkeypatch, tts, llm, StubKB())

    manager.process_message("What is the capital of France?")

    assert llm.lanes == ["general"]
    assert "College knowledge" not in llm.prompts[0]
    assert "No relevant college knowledge found" not in llm.prompts[0]
    assert any("Paris" in text for text in tts.spoken)


def test_college_question_still_uses_college_lane(monkeypatch):
    tts = RecordingTTS()
    llm = RecordingLLM("The B.Tech fee is 1,25,000 rupees per year.")
    context = "Q: What is the B.Tech fee?\nA: 1,25,000 rupees per year."
    manager = build_manager(
        monkeypatch, tts, llm, StubKB(context=context, confidence=0.9)
    )

    manager.process_message("What is the B.Tech fee?")

    assert llm.lanes == ["college"]
    assert "College knowledge" in llm.prompts[0]
    assert "1,25,000" in llm.prompts[0]


# ----------------------------------------------------------------------
# Fix 3: recognition margin, consecutive-frame confirmation and greeting
# ----------------------------------------------------------------------

class ConfiguredConfig:
    """Config double that returns the recognition mapping under test."""

    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def get_value(self, file_name, *keys, default=None):
        if file_name != "vision":
            return default
        if keys == ("recognition",):
            return self.mapping
        if len(keys) == 2 and keys[0] == "recognition":
            return self.mapping.get(keys[1], default)
        return default


def make_vision(recognition=None, event_bus=None):
    container = FakeContainer()
    container.config = ConfiguredConfig(recognition)
    if event_bus is not None:
        container.event_bus = event_bus

    vision = VisionService(
        container,
        camera=FakeCamera(),
        embedding_model=FakeModel(),
        database=FakeDatabase({"Saksham": [unit(0)]}),
        visitor_database=FakeVisitorDatabase(),
    )

    return vision, container.event_bus


def detected_events(bus):
    return [payload for event, payload in bus.events if event == "visitor.detected"]


def test_recognition_margin_reports_a_near_tie_as_unknown():
    query = unit(0)
    almost = unit(0) * 0.999 + unit(1) * 0.01

    recognizer = FaceRecognizer(
        FakeDatabase({"Alice": [query], "Bob": [almost]}),
        config=RecognitionConfig(threshold=0.45, decision_margin=0.05),
    )

    result = recognizer.recognize_result(face(query))

    assert result.recognized is False
    assert result.status == "unknown"
    assert result.candidate in {"Alice", "Bob"}
    assert "margin" in result.reason


def test_recognition_accepts_a_clear_winner_with_the_margin():
    recognizer = FaceRecognizer(
        FakeDatabase({"Alice": [unit(0)], "Bob": [unit(1)]}),
        config=RecognitionConfig(threshold=0.45, decision_margin=0.05),
    )

    result = recognizer.recognize_result(face(unit(0)))

    assert result.recognized is True
    assert result.name == "Alice"


def test_vision_requires_consecutive_frames_before_confirming():
    vision, bus = make_vision(
        {"threshold": 0.45, "decision_margin": 0.05, "confirmation_frames": 2}
    )

    first_faces = vision.recognizer.recognize_all([face(unit(0))])
    vision.tracker.update(first_faces)
    vision._publish_events(first_faces)

    assert detected_events(bus) == []
    assert vision.current_identity()["present"] is True
    assert vision.current_identity()["recognized"] is False

    second_faces = vision.recognizer.recognize_all([face(unit(0))])
    vision.tracker.update(second_faces)
    vision._publish_events(second_faces)

    events = detected_events(bus)
    assert len(events) == 1
    assert events[0]["name"] == "Saksham"
    assert events[0]["greet"] is True
    assert vision.current_identity()["recognized"] is True

    third_faces = vision.recognizer.recognize_all([face(unit(0))])
    vision.tracker.update(third_faces)
    vision._publish_events(third_faces)

    assert len(detected_events(bus)) == 1


def test_live_greeting_reaches_the_presenter_once(app):
    bus = EventBus()
    vision, bus = make_vision(
        {"threshold": 0.45, "decision_margin": 0.05, "confirmation_frames": 2},
        event_bus=bus,
    )

    presenter = KioskPresenter(event_bus=bus)

    requests = []
    captions = []

    bus.subscribe(
        "greeting.speak_requested",
        lambda **payload: requests.append(payload),
    )
    presenter.caption.connect(
        lambda text, duration: captions.append(text)
    )

    def cycle():
        faces = vision.recognizer.recognize_all([face(unit(0))])
        vision.tracker.update(faces)
        vision._publish_events(faces)
        app.processEvents()

    cycle()
    assert requests == []

    cycle()
    app.processEvents()

    assert len(requests) == 1
    assert requests[0]["name"] == "Saksham"
    assert "Hello, Saksham" in requests[0]["text"]
    assert any("Hello, Saksham" in caption for caption in captions)

    sitting_id = requests[0]["sitting_id"]
    assert sitting_id is not None

    cycle()
    app.processEvents()

    assert len(requests) == 1
    assert presenter.visitor_present is True

    presenter.shutdown()


# ----------------------------------------------------------------------
# Live enrolment helper
# ----------------------------------------------------------------------

def test_enrolment_self_match_accepts_consistent_views():
    weakest, mean = self_match_scores([unit(0), unit(0) * 4.0, unit(0)])

    assert weakest == pytest.approx(1.0, abs=1e-5)
    assert mean == pytest.approx(1.0, abs=1e-5)


def test_enrolment_self_match_rejects_inconsistent_views():
    with pytest.raises(EnrollmentError):

        verify_self_match([unit(0), unit(1)])


def test_enrolment_self_match_requires_two_views():
    with pytest.raises(EnrollmentError):

        self_match_scores([unit(0)])
