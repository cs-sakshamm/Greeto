"""Regression tests for the four live-testing fixes (2026-11-28).

Covers:

1. Real-time barge-in: immediate audio purge, turn fencing, stale captions.
2. Presence greeting and live identity questions.
3. Short/low-RMS STT handling, prompting, and phonetic repair.
4. Hybrid routing (Option A): persona, strict college KB, general LLM.

No camera, microphone, Whisper model, Piper voice or LLM is opened.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.brain.conversation_manager as conversation_module
from app.brain.conversation_manager import ConversationManager
from app.core.event_bus import EventBus
from app.gui.kiosk_presenter import KioskPresenter
from app.speech import speech_service as speech_module
from app.speech.speech_service import SpeechService
from app.speech.transcript_normalizer import normalize_transcript
from app.speech.tts_service import TTSService
from app.vision.models import Face
from app.vision.vision_service import VisionService

from tests.test_barge_in import (
    DictConfig,
    NoopAudioCapture,
    SilentSTT,
    VoiceVAD,
    _speech_config,
)
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
# Stubs
# ----------------------------------------------------------------------


class Knowledge:
    """Retriever stub with an explicit context/confidence."""

    confidence_gate = True
    rag = {"retrieval": {"answer_confidence_floor": 0.55}}

    def __init__(self, config=None):
        self.context = ""
        self.confidence = 0.0

    def search_with_confidence(self, text):
        return types.SimpleNamespace(
            context=self.context,
            confidence=self.confidence,
            has_evidence=bool(self.context.strip()),
        )


class FixedLLM:
    prompt_style = "qwen"
    disable_thinking = False

    def __init__(self, reply="Okay."):
        self.reply = reply
        self.prompts = []

    def stream_generate(self, prompt, cancel_event=None):
        self.prompts.append(prompt)
        yield json.dumps(
            {"reply": self.reply, "emotion": "neutral", "action": "answer"}
        )


class RecordingTTS:
    def __init__(self):
        self.spoken = []
        self.stop_calls = 0

    def is_speaking(self):
        return False

    def speak(self, text, turn_id=None):
        self.spoken.append((turn_id, text))
        return True

    def stop(self):
        self.stop_calls += 1
        return True

    def cancel(self):
        return self.stop()


def config(streaming=False):
    return DictConfig(
        {
            "llm": {
                "streaming": {
                    "sentence_tts": streaming,
                    "min_sentence_chars": 8,
                    "max_sentence_chars": 240,
                }
            }
        }
    )


def make_manager(monkeypatch, llm, tts=None, vision=None):
    monkeypatch.setattr(conversation_module, "KnowledgeBase", Knowledge)
    container = types.SimpleNamespace(
        logger=logging.getLogger("live-fixes"),
        event_bus=EventBus(),
        config=config(),
    )
    manager = ConversationManager(
        container,
        tts_service=tts or RecordingTTS(),
        llm=llm,
        vision_service=vision,
    )
    return manager, container.event_bus, manager.knowledge


class StubVision:
    def __init__(self, identity):
        self.identity = identity

    def current_identity(self):
        return dict(self.identity)


def identity(present=True, recognized=False, name=None):
    return {
        "present": present,
        "recognized": recognized,
        "name": name,
        "present_sitting": None,
        "sitting_id": 1 if present else None,
        "confidence": 0.9 if recognized else None,
    }


# ----------------------------------------------------------------------
# Fix 1: barge-in
# ----------------------------------------------------------------------


class Terminatable:
    def __init__(self):
        self.terminated = False

    def terminate(self):
        self.terminated = True


def test_tts_stop_purges_player_and_synthesis_and_logs_stop_ms(tmp_path):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"placeholder")
    bus = EventBus()
    container = types.SimpleNamespace(logger=logging.getLogger("tts"), event_bus=bus)
    service = TTSService(container, model_path=model, engine="subprocess")

    player = Terminatable()
    synth = Terminatable()
    service._active_player = player
    service._active_synth_process = synth
    service._active_turn_id = 7

    stopped = []
    bus.subscribe("tts.stopped", lambda **payload: stopped.append(payload))

    assert service.stop() is True

    assert player.terminated, "the active audio player must be terminated"
    assert synth.terminated, "in-flight synthesis must be purged"
    assert stopped and stopped[0]["interrupted"] is True
    assert stopped[0]["turn_id"] == 7
    assert stopped[0]["stop_ms"] >= 0.0


def test_low_rms_barge_in_is_caught_with_input_gain(monkeypatch, ):
    bus = EventBus()
    monkeypatch.setattr(speech_module, "SpeechToText", SilentSTT)
    monkeypatch.setattr(speech_module, "AudioCapture", NoopAudioCapture)

    cfg = _speech_config()
    cfg.data["speech"]["audio"]["input_gain"] = 2.0

    container = types.SimpleNamespace(
        logger=logging.getLogger("gain"), event_bus=bus, config=cfg
    )
    service = SpeechService(container)
    service.vad = VoiceVAD()
    service.stt = None

    fired = []
    bus.subscribe("speech.barge_in", lambda **payload: fired.append(payload))

    bus.publish("visitor.detected", name="Saksham", sitting_id=1, greet=True)
    bus.publish("tts.started", turn_id=1, text="Answer")

    # RMS ~0.0043: the real low-SNR capture from the live test.
    for _ in range(5):
        service._handle_chunk(np.full(1024, 0.0043, dtype=np.float32))

    assert len(fired) == 1, "quiet sustained speech must interrupt playback"
    assert fired[0]["rms"] >= 0.0043 * 1.9, "gain should be applied before the gate"


def test_low_rms_barge_in_fails_without_gain(monkeypatch):
    """Documents the cause of the original failure, not a behaviour we want."""

    bus = EventBus()
    monkeypatch.setattr(speech_module, "SpeechToText", SilentSTT)
    monkeypatch.setattr(speech_module, "AudioCapture", NoopAudioCapture)

    container = types.SimpleNamespace(
        logger=logging.getLogger("no-gain"), event_bus=bus, config=_speech_config()
    )
    service = SpeechService(container)
    service.vad = VoiceVAD()
    service.stt = None

    fired = []
    bus.subscribe("speech.barge_in", lambda **payload: fired.append(payload))

    service.state = SpeechService.STATE_LISTENING

    bus.publish("visitor.detected", name="Saksham", sitting_id=1, greet=True)
    bus.publish("tts.started", turn_id=1, text="Answer")

    for _ in range(5):
        service._handle_chunk(np.full(1024, 0.0043, dtype=np.float32))

    assert fired == []
    assert service.state == SpeechService.STATE_LISTENING


def test_presenter_discards_late_output_from_interrupted_turn(app):
    bus = EventBus()
    presenter = KioskPresenter(event_bus=bus)

    captions = []
    presenter.caption.connect(lambda text, duration: captions.append(text))

    bus.publish("conversation.processing", text="fee", turn_id=5)
    app.processEvents()
    bus.publish(
        "speech.speaking_started", text="The fee is...", turn_id=5, utterance=1
    )
    app.processEvents()
    assert presenter.current_state() == "speaking"

    bus.publish("speech.barge_in", t_barge_in=time.perf_counter())
    app.processEvents()
    assert presenter.current_state() == "listening"
    assert presenter.busy is False

    captions.clear()

    # The old turn keeps trickling out after the barge-in.
    bus.publish(
        "speech.speaking_started", text="Stale sentence.", turn_id=5, utterance=2
    )
    bus.publish(
        "llm_response_ready",
        response=types.SimpleNamespace(reply="Stale sentence.", emotion="neutral"),
        turn_id=5,
    )
    app.processEvents()

    assert captions == [], "stale captions must not reappear after barge-in"
    assert presenter.current_state() == "listening"

    # The new turn is still allowed through.
    bus.publish("conversation.processing", text="new", turn_id=6)
    bus.publish("speech.speaking_started", text="Fresh answer.", turn_id=6, utterance=3)
    app.processEvents()
    assert "Fresh answer." in captions

    presenter.shutdown()


# ----------------------------------------------------------------------
# Fix 2: greeting and live identity
# ----------------------------------------------------------------------


def test_vision_exposes_live_identity_snapshot():
    vision = VisionService(
        FakeContainer(),
        camera=FakeCamera(),
        embedding_model=FakeModel(),
        database=FakeDatabase({"Saksham": [unit(0)]}),
        visitor_database=FakeVisitorDatabase(),
    )

    assert vision.current_identity()["present"] is False

    faces = vision.recognizer.recognize_all([face(unit(0))])
    vision.tracker.update(faces)
    vision._publish_events(faces)

    snapshot = vision.current_identity()
    assert snapshot["present"] is True
    assert snapshot["recognized"] is True
    assert snapshot["name"] == "Saksham"

    # An unrecognized face is present but not identified.
    stranger = vision.recognizer.recognize_all([face(unit(9))])
    vision.tracker.update(stranger)
    vision._publish_events(stranger)

    snapshot = vision.current_identity()
    assert snapshot["present"] is True
    assert snapshot["recognized"] is False
    assert snapshot["name"] is None


def test_greeting_is_deferred_not_dropped_during_an_active_turn(app):
    bus = EventBus()
    presenter = KioskPresenter(event_bus=bus)
    presenter.SUCCESS_HOLD_MS = 0

    greeted = []
    bus.subscribe("greeting.speak_requested", lambda **payload: greeted.append(payload))

    bus.publish("conversation.processing", text="fee", turn_id=1)
    app.processEvents()

    bus.publish(
        "visitor.detected",
        name="Saksham",
        face=types.SimpleNamespace(visitor_id=3),
        sitting_id=9,
        greet=True,
    )
    app.processEvents()

    assert greeted == [], "an active turn must not be interrupted by a greeting"

    bus.publish("speech.speaking_started", text="Here is the fee.", turn_id=1, utterance=1)
    bus.publish("speech.speaking_finished", turn_id=1, utterance=1)
    app.processEvents()
    time.sleep(0.01)
    app.processEvents()

    assert len(greeted) == 1
    assert greeted[0]["name"] == "Saksham"
    assert "Hello, Saksham" in greeted[0]["text"]
    assert ("sitting", 9) in presenter._greeted

    presenter.shutdown()


@pytest.mark.parametrize(
    "vision_state, expected",
    [
        (identity(name="Saksham", recognized=True), "Yes, I see you, Saksham."),
        (identity(present=True), "no enrolled face match"),
        (identity(present=False), "don't see anyone"),
    ],
)
def test_identity_questions_use_live_vision(monkeypatch, vision_state, expected):
    llm = FixedLLM("This must not be used.")
    tts = RecordingTTS()
    manager, bus, _ = make_manager(monkeypatch, llm, tts, StubVision(vision_state))

    manager.process_message("Do you recognize me?")

    assert len(tts.spoken) == 1
    assert expected in tts.spoken[0][1]
    assert "1800-180-0840" not in tts.spoken[0][1]
    assert llm.prompts == [], "identity questions must not reach the LLM"
    assert manager.history[-1]["role"] == "assistant"


def test_identity_event_fallback_without_vision_service(monkeypatch):
    manager, bus, _ = make_manager(monkeypatch, FixedLLM("unused"))

    bus.publish(
        "visitor.detected",
        name="Saksham",
        face=types.SimpleNamespace(visitor_id=2),
        sitting_id=4,
        greet=True,
    )

    manager.process_message("Who am I?")

    assert manager.tts.spoken[-1][1] == "Yes, I see you, Saksham."


# ----------------------------------------------------------------------
# Fix 3: STT accuracy
# ----------------------------------------------------------------------


def test_quiet_utterance_is_normalized_before_whisper(monkeypatch):
    bus = EventBus()
    monkeypatch.setattr(speech_module, "SpeechToText", SilentSTT)
    monkeypatch.setattr(speech_module, "AudioCapture", NoopAudioCapture)
    container = types.SimpleNamespace(
        logger=logging.getLogger("norm"), event_bus=bus, config=_speech_config()
    )
    service = SpeechService(container)

    quiet = np.full(16000, 0.0043, dtype=np.float32)
    normalized, gain = service._normalize_for_stt(quiet)

    assert gain > 10.0
    assert float(np.max(np.abs(normalized))) > 0.1

    silent = np.zeros(16000, dtype=np.float32)
    _, silent_gain = service._normalize_for_stt(silent)
    assert silent_gain == 1.0, "silence must never be amplified into speech"


def test_stt_model_switch_falls_back_to_model_path(monkeypatch):
    bus = EventBus()
    monkeypatch.setattr(speech_module, "SpeechToText", SilentSTT)
    monkeypatch.setattr(speech_module, "AudioCapture", NoopAudioCapture)

    cfg = _speech_config()
    cfg.data["speech"]["stt"]["model"] = "nonexistent-model"
    container = types.SimpleNamespace(
        logger=logging.getLogger("model"), event_bus=bus, config=cfg
    )
    service = SpeechService(container)

    assert service._resolve_stt_model_path() == "models/whisper/ggml-tiny.en.bin"


def test_phonetic_alias_repair_is_phrase_level_and_idempotent():
    assert normalize_transcript("Who love you?")[0] == "Who are you?"
    assert normalize_transcript("who loves you")[0] == "who are you"
    assert normalize_transcript("Who are you?")[1] == []
    assert normalize_transcript("how are you")[0] == "how are you"
    assert normalize_transcript("do you recognize me")[1] == []


def test_conversation_manager_repairs_misheard_persona_question(monkeypatch):
    manager, bus, _ = make_manager(monkeypatch, FixedLLM("unused"))

    manager.handle_speech("who love you")

    deadline = time.perf_counter() + 2.0
    while not manager.tts.spoken and time.perf_counter() < deadline:
        time.sleep(0.01)

    assert manager.tts.spoken
    assert manager.tts.spoken[0][1] == ConversationManager.PERSONA_REPLY


# ----------------------------------------------------------------------
# Fix 4: hybrid routing (Option A)
# ----------------------------------------------------------------------


def test_persona_question_is_answered_without_kb_or_llm(monkeypatch):
    llm = FixedLLM("unused")
    tts = RecordingTTS()
    manager, bus, kb = make_manager(monkeypatch, llm, tts)
    kb.context = ""
    kb.confidence = 0.0

    manager.process_message("Who are you?")

    spoken = tts.spoken[0][1]
    assert spoken == ConversationManager.PERSONA_REPLY
    assert "Greeto" in spoken
    assert "I.T.S Engineering College" in spoken
    assert "couldn't find that information" not in spoken
    assert llm.prompts == []


def test_college_fact_stays_grounded_in_kb(monkeypatch):
    llm = FixedLLM("The total MBA fee is 2,56,500 rupees, payable in four installments.")
    tts = RecordingTTS()
    manager, bus, kb = make_manager(monkeypatch, llm, tts)
    kb.context = (
        "Q: What is the total fee for the MBA program?\n"
        "A: The total fee for the two-year MBA program is 2,56,500 rupees, "
        "payable in four installments."
    )
    kb.confidence = 0.92

    manager.process_message("What is the total MBA fee?")

    spoken = tts.spoken[0][1]
    assert "2,56,500" in spoken
    assert "1800-180-0840" not in spoken
    assert llm.prompts, "a college question with evidence goes through the LLM"


def test_college_fact_without_evidence_uses_helpline(monkeypatch):
    llm = FixedLLM("Invented answer.")
    tts = RecordingTTS()
    manager, bus, kb = make_manager(monkeypatch, llm, tts)
    kb.context = ""
    kb.confidence = 0.0

    manager.process_message("What is the hostel fee for first year?")

    spoken = tts.spoken[0][1]
    assert "1800-180-0840" in spoken
    assert llm.prompts == []


def test_general_question_is_answered_naturally_without_helpline(monkeypatch):
    llm = FixedLLM(
        "The capital of France is Paris. The city has been a capital since 1900."
    )
    tts = RecordingTTS()
    manager, bus, kb = make_manager(monkeypatch, llm, tts)
    kb.context = ""
    kb.confidence = 0.0

    manager.process_message("What is the capital of France?")

    spoken = tts.spoken[0][1]
    assert "Paris" in spoken
    assert "1800-180-0840" not in spoken
    assert "couldn't find" not in spoken.lower()
    assert llm.prompts, "general questions are answered by the LLM"
