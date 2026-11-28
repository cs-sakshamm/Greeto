"""Voice barge-in regression tests.

No camera, microphone, Whisper model, Piper voice or LLM is opened. The
tests drive the real SpeechService, TTSService, ConversationManager and
KioskPresenter event paths with deterministic stubs.
"""

from __future__ import annotations

import logging
import sys
import threading
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
from app.speech.tts_service import TTSService


class DictConfig:
    """Minimal ConfigManager.get_value() walker."""

    def __init__(self, data):
        self.data = data

    def get_value(self, section, *keys, default=None):
        node = self.data
        for key in (section,) + keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return default if node is None else node


class VoiceVAD:
    energy_threshold = 0.0035

    @staticmethod
    def calculate_energy(audio):
        if audio is None or len(audio) == 0:
            return 0.0
        arr = np.asarray(audio, dtype=np.float32)
        return float(np.sqrt(np.mean(np.square(arr))))

    @staticmethod
    def is_speech(audio):
        return True


class SilentSTT:
    def __init__(self, **kwargs):
        pass


class NoopAudioCapture:
    def __init__(self, **kwargs):
        pass


def _speech_config(**barge_overrides):
    barge = {
        "enabled": True,
        "require_visitor_in_frame": True,
        "min_speech_ms": 200,
        "playback_vad_threshold_multiplier": 1.8,
        "cooldown_ms": 300,
    }
    barge.update(barge_overrides)
    return DictConfig(
        {
            "speech": {
                "audio": {
                    "sample_rate": 16000,
                    "chunk_size": 1024,
                    "device": None,
                    "queue_max_chunks": 100,
                    "min_recording_seconds": 0.3,
                    "max_recording_seconds": 15.0,
                },
                "vad": {
                    "aggressiveness": 2,
                    "frame_ms": 30,
                    "energy_threshold": 0.0035,
                    "silence_ms": 600,
                    "min_speech_ms": 120,
                    "pre_speech_ms": 384,
                },
                "stt": {
                    "model_path": "models/whisper/ggml-tiny.en.bin",
                    "n_threads": 1,
                    "language": "en",
                    "min_text_length": 3,
                    "max_text_length": 500,
                },
                "pipeline": {
                    "mute_during_speech": True,
                    "speech_cooldown_seconds": 0.0,
                    "debug_save_audio": False,
                },
                "barge_in": barge,
            }
        }
    )


@pytest.fixture()
def bus():
    return EventBus()


def _make_speech_service(monkeypatch, bus, config=None):
    monkeypatch.setattr(speech_module, "SpeechToText", SilentSTT)
    monkeypatch.setattr(speech_module, "AudioCapture", NoopAudioCapture)
    container = types.SimpleNamespace(
        logger=logging.getLogger("barge-in-test"),
        event_bus=bus,
        config=config or _speech_config(),
    )
    service = SpeechService(container)
    service.vad = VoiceVAD()
    service.stt = None
    return service


def _chunk(amplitude=0.1, size=1024):
    return np.full(size, amplitude, dtype=np.float32)


def _record(bus, name, events):
    def handler(**payload):
        events.append((name, payload))

    bus.subscribe(name, handler)


def test_visitor_speech_interrupts_tts_and_preserves_onset(monkeypatch, bus):
    service = _make_speech_service(monkeypatch, bus)

    events = []
    _record(bus, "speech.barge_in", events)

    bus.publish("visitor.detected", name="Asha", sitting_id=1, greet=True)

    # Piper is playing when the visitor starts speaking.
    bus.publish("tts.started", turn_id=1, text="Answer")
    assert service._tts_speaking is True

    first_chunk = _chunk(0.12)
    chunks = [first_chunk] + [_chunk(0.12) for _ in range(4)]
    for chunk in chunks:
        service._handle_chunk(chunk)

    assert len(events) == 1, "sustained speech should fire exactly one barge-in"
    payload = events[0][1]
    assert payload["min_speech_ms"] == 200
    assert service.state == SpeechService.STATE_RECORDING

    # Every chunk from speech onset onward is kept, including the first.
    assert len(service.audio_buffer) >= 1
    assert service.audio_buffer[0].shape == first_chunk.shape
    assert service.total_samples_recorded >= len(chunks) * first_chunk.size
    assert payload["t_speech_onset"] <= payload["t_barge_in"]


def test_no_barge_in_without_visitor_in_frame(monkeypatch, bus):
    service = _make_speech_service(monkeypatch, bus)
    events = []
    _record(bus, "speech.barge_in", events)

    service._tts_speaking = True
    service.state = SpeechService.STATE_LISTENING
    for _ in range(6):
        service._handle_chunk(_chunk(0.3))

    assert events == []
    assert service.state == SpeechService.STATE_LISTENING


def test_brief_noise_or_quiet_audio_does_not_self_interrupt(monkeypatch, bus):
    service = _make_speech_service(monkeypatch, bus)
    events = []
    _record(bus, "speech.barge_in", events)

    bus.publish("visitor.detected", name="Asha", sitting_id=1, greet=True)
    service._tts_speaking = True

    # Below the elevated playback threshold (0.0035 * 1.8 = 0.0063).
    for _ in range(6):
        service._handle_chunk(_chunk(0.001))

    # Loud but shorter than min_speech_ms (200 ms), followed by silence.
    service._handle_chunk(_chunk(0.2))
    for _ in range(6):
        service._handle_chunk(_chunk(0.0))

    assert events == []
    assert service.audio_buffer == []


def test_tts_stop_is_fast_and_idempotent(tmp_path):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"placeholder")
    bus = EventBus()
    container = types.SimpleNamespace(
        logger=logging.getLogger("tts-stop-test"),
        event_bus=bus,
        config=None,
    )
    service = TTSService(container, model_path=model, engine="subprocess")
    stopped = []
    _record(bus, "tts.stopped", stopped)

    service._is_speaking = True
    service._cancelled = False

    started = time.perf_counter()
    service.stop()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    assert elapsed_ms < 50.0
    assert service._cancelled is True
    assert stopped and stopped[0][1]["interrupted"] is True
    assert service.stop() is True  # idempotent


class FakeKB:
    confidence_gate = True
    rag = {"retrieval": {"answer_confidence_floor": 0.55}}

    def __init__(self, config=None):
        pass

    def search_with_confidence(self, text):
        return types.SimpleNamespace(
            context=(
                "Q: What is the total fee for the MBA program?\n"
                "A: The total fee for the two-year MBA program is "
                "2,56,500 rupees, payable in four installments."
            ),
            confidence=0.92,
            has_evidence=True,
        )


class FakeLLM:
    prompt_style = "qwen"
    disable_thinking = False

    def __init__(self):
        self.consumed = 0
        self.delay = 0.01
        self._release = threading.Event()
        self.text = (
            '{"reply": "The MBA total fee is 2,56,500 rupees in four '
            'installments.", "emotion": "neutral", "action": "answer"}'
        )

    def stream_generate(self, prompt):
        for character in self.text:
            if self._release.is_set():
                return
            self.consumed += 1
            if self.delay:
                time.sleep(self.delay)
            yield character


class FakeTTS:
    def __init__(self):
        self.spoken = []
        self.stop_calls = 0
        self._stop = threading.Event()

    def speak(self, text, turn_id=None):
        self.spoken.append((turn_id, text))
        self._stop.wait(0.02)
        return True

    def stop(self):
        self.stop_calls += 1
        self._stop.set()
        return True

    def is_speaking(self):
        return False


def test_conversation_manager_aborts_turn_and_answers_new_turn(monkeypatch, bus):
    monkeypatch.setattr(conversation_module, "KnowledgeBase", FakeKB)

    container = types.SimpleNamespace(
        logger=logging.getLogger("barge-in-manager"),
        event_bus=bus,
        config=DictConfig(
            {
                "llm": {
                    "streaming": {
                        "sentence_tts": True,
                        "min_sentence_chars": 8,
                        "max_sentence_chars": 240,
                    }
                }
            }
        ),
    )
    llm = FakeLLM()
    tts = FakeTTS()
    manager = ConversationManager(container, tts_service=tts, llm=llm)
    manager.knowledge = FakeKB()

    handled = []
    responses = []
    _record(bus, "barge_in.handled", handled)
    _record(bus, "llm_response_ready", responses)

    manager._current_turn_id = 1
    worker = threading.Thread(
        target=manager.process_message,
        args=("Tell me about the MBA fee", 1, time.perf_counter(), time.perf_counter()),
        daemon=True,
    )
    worker.start()

    deadline = time.perf_counter() + 2.0
    while llm.consumed < 8 and time.perf_counter() < deadline:
        time.sleep(0.005)

    assert llm.consumed >= 8, "fake LLM should be streaming before barge-in"
    consumed_at_barge_in = llm.consumed

    bus.publish("speech.barge_in", t_barge_in=time.perf_counter(), visitor_in_frame=True)
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert manager._current_turn_id == 2
    assert tts.stop_calls >= 1
    assert handled and handled[0][1]["turn_id"] == 1
    assert manager._timings[1]["barge_in_to_audio_stop_ms"] < 50.0

    # The interrupted turn's assistant fragment is not left in history.
    assert not any(
        message.get("role") == "assistant" and message.get("turn_id") == 1
        for message in manager.history
    )

    # The new utterance is answered normally on the fresh turn.
    llm.delay = 0.0
    manager.process_message("What is the MBA total fee?", turn_id=2)

    assert any(payload.get("turn_id") == 2 for _, payload in responses)
    assert any(turn_id == 2 for turn_id, _ in tts.spoken)

    # Old tokens stopped early rather than streaming the whole reply.
    assert consumed_at_barge_in < len(llm.text)


def test_kiosk_barge_in_clears_caption_without_regreeting(app, bus):
    presenter = KioskPresenter(event_bus=bus)
    greeted = []
    _record(bus, "greeting.speak_requested", greeted)

    face = types.SimpleNamespace(visitor_id=11)
    bus.publish(
        "visitor.detected",
        name="Asha",
        face=face,
        sitting_id=7,
        greet=True,
    )
    for _ in range(20):
        app.processEvents()
        time.sleep(0.002)

    assert len(greeted) == 1
    assert ("sitting", 7) in presenter._greeted

    bus.publish("conversation.processing", text="fee", turn_id=1)
    bus.publish("speech.speaking_started", text="The fee is...", turn_id=1, utterance=1)
    for _ in range(20):
        app.processEvents()
        time.sleep(0.002)

    bus.publish("speech.barge_in", t_barge_in=time.perf_counter())
    for _ in range(20):
        app.processEvents()
        time.sleep(0.002)

    assert presenter.current_state() == "listening"
    assert presenter.busy is False
    assert ("sitting", 7) in presenter._greeted

    # Even a duplicate detection from the same sitting must not greet again.
    bus.publish(
        "visitor.detected",
        name="Asha",
        face=face,
        sitting_id=7,
        greet=True,
    )
    for _ in range(20):
        app.processEvents()
        time.sleep(0.002)

    assert len(greeted) == 1
    presenter.shutdown()
