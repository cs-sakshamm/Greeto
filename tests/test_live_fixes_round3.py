"""Regression tests for the third round of live-kiosk fixes.

Covers the three failures seen while running the kiosk:

* a visitor who stepped away and came back was not greeted again, and the
  recognition diagnostic line printed on every camera frame;
* speaking over Greeto never interrupted playback on Windows, because
  ``winsound.PlaySound`` blocks and the barge-in gate was unreachable at
  laptop-mic levels;
* the model occasionally answered with the prompt placeholder
  ``short spoken answer``.

No camera, microphone, InsightFace model, Piper voice or LLM is opened.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import types
import wave

import numpy as np
import pytest

import app.brain.conversation_manager as conversation_module
from app.brain.conversation_manager import ConversationManager
from app.brain.parser import ResponseParser
from app.brain.prompt_manager import PromptManager
from app.brain.streaming import (
    ReplyPrefixFilter,
    ReplyStreamExtractor,
    SentenceChunker,
    strip_placeholders,
    strip_reply_prefix,
)
from app.core.event_bus import EventBus
from app.gui.kiosk_presenter import KioskPresenter
from app.speech import speech_service as speech_module
from app.speech.speech_service import SpeechService
from app.speech.tts_service import TTSService
from app.vision.vision_service import VisionService

from tests.test_barge_in import DictConfig, FakeKB, FakeLLM, FakeTTS
from tests.test_recognition import (
    FakeBus,
    FakeCamera,
    FakeDatabase,
    FakeModel,
    FakeVisitorDatabase,
    face,
    unit,
)


# ----------------------------------------------------------------------
# Fix 1: absence ends the sitting (so a return is greeted again) and the
# recognition log is rate limited
# ----------------------------------------------------------------------

class VisionConfig:
    """Config double that serves vision.recognition and vision.greeting."""

    def __init__(
        self,
        absence: float = 3.0,
        diagnostic_interval: float = 2.0,
        confirmation_frames: int = 2,
    ) -> None:
        self.recognition = {
            "threshold": 0.45,
            "decision_margin": 0.05,
            "confirmation_frames": confirmation_frames,
            "diagnostic_interval_s": diagnostic_interval,
        }
        self.greeting = {
            "enabled": True,
            "absence_timeout_s": absence,
            "min_confidence": None,
        }

    def get_value(self, file_name, *keys, default=None):
        if file_name != "vision":
            return default

        node = {"recognition": self.recognition, "greeting": self.greeting}

        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]

        return default if node is None else node


@pytest.fixture()
def bus():
    return EventBus()


def make_vision(config=None, event_bus=None, logger_name="vision-round3"):
    container = types.SimpleNamespace(
        logger=logging.getLogger(logger_name),
        config=config or VisionConfig(),
        event_bus=event_bus if event_bus is not None else FakeBus(),
    )

    vision = VisionService(
        container,
        camera=FakeCamera(),
        embedding_model=FakeModel(),
        database=FakeDatabase({"Saksham": [unit(0)]}),
        visitor_database=FakeVisitorDatabase(),
    )

    return vision, container.event_bus


def detect(vision, now, x=100):
    """Run one recognition cycle and publish its events."""

    faces = vision.recognizer.recognize_all([face(unit(0))])
    vision.tracker.update(faces)

    if x != 100:
        for detected in faces:
            detected.x = x

    vision._publish_events(faces, now=now)

    return faces


def detected_events(bus):
    return [payload for event, payload in bus.events if event == "visitor.detected"]


def test_absence_ends_the_sitting_so_a_return_is_greeted_again():
    vision, bus = make_vision()

    start = 1000.0

    detect(vision, start)
    detect(vision, start + 0.4)

    events = detected_events(bus)
    assert len(events) == 1
    assert events[0]["greet"] is True
    first_sitting = events[0]["sitting_id"]

    # Staying in frame never re-greets.
    for offset in (0.8, 1.2, 1.6):
        detect(vision, start + offset)

    assert len(detected_events(bus)) == 1

    # Away longer than absence_timeout_s: the sitting ends and the kiosk
    # is told nobody is in frame.
    gone_at = start + 2.0
    absence = vision.greeting_absence_s
    assert absence == pytest.approx(3.0)

    vision._publish_events([], now=gone_at + absence + 0.1)

    assert any(event == "visitor.left" for event, _ in bus.events)
    assert vision.current_identity()["present"] is False
    assert vision.current_identity()["recognized"] is False

    # Returning starts a fresh sitting, which must greet again.
    back = gone_at + absence + 1.0
    detect(vision, back, x=420)
    detect(vision, back + 0.4, x=420)

    events = detected_events(bus)
    assert len(events) == 2
    assert events[1]["greet"] is True
    assert events[1]["sitting_id"] != first_sitting

    # And still only once for that new sitting.
    detect(vision, back + 0.8, x=420)
    assert len(detected_events(bus)) == 2


def test_vision_left_is_published_once_per_absence():
    vision, bus = make_vision()

    detect(vision, 500.0)
    detect(vision, 500.4)

    absence = vision.greeting_absence_s

    vision._publish_events([], now=500.9)
    assert not any(event == "visitor.left" for event, _ in bus.events)

    vision._publish_events([], now=500.9 + absence + 0.1)
    left = [event for event, _ in bus.events if event == "visitor.left"]
    assert len(left) == 1

    vision._publish_events([], now=500.9 + absence + 5.0)
    left = [event for event, _ in bus.events if event == "visitor.left"]
    assert len(left) == 1


def test_recognition_diagnostics_are_rate_limited_not_per_frame(caplog):
    logger_name = "vision-log-throttle"
    vision, _ = make_vision(logger_name=logger_name)

    caplog.set_level(logging.INFO, logger=logger_name)

    base = 900.0

    for index in range(20):
        detect(vision, base + index * 0.01)

    lines = [
        record.getMessage()
        for record in caplog.records
        if record.name == logger_name
        and record.getMessage().startswith("Vision match:")
    ]

    assert len(lines) == 1, "recognition logged per frame: %r" % (lines,)

    # A state change inside the hard floor still cannot add a second line.
    detect(vision, base + 0.3)

    lines = [
        record.getMessage()
        for record in caplog.records
        if record.name == logger_name
        and record.getMessage().startswith("Vision match:")
    ]

    assert len(lines) == 1

    # After the steady-state interval exactly one heartbeat is due.
    detect(vision, base + 0.3 + 2.1)

    lines = [
        record.getMessage()
        for record in caplog.records
        if record.name == logger_name
        and record.getMessage().startswith("Vision match:")
    ]

    assert len(lines) == 2


def test_presenter_greets_again_for_a_new_sitting_but_not_within_one(app):
    bus = EventBus()
    presenter = KioskPresenter(event_bus=bus)

    requests = []
    bus.subscribe("greeting.speak_requested", lambda **payload: requests.append(payload))

    def publish(sitting_id):
        bus.publish(
            "visitor.detected",
            name="Saksham",
            face=types.SimpleNamespace(visitor_id=1),
            sitting_id=sitting_id,
            greet=True,
        )
        app.processEvents()

    publish(1)
    assert len(requests) == 1
    assert "Hello, Saksham" in requests[0]["text"]

    publish(1)
    assert len(requests) == 1

    # Leaving the frame ends the sitting; returning is a new greeting.
    bus.publish("visitor.left")
    app.processEvents()

    publish(2)
    assert len(requests) == 2
    assert "Hello, Saksham" in requests[1]["text"]

    publish(2)
    assert len(requests) == 2

    presenter.shutdown()


# ----------------------------------------------------------------------
# Fix 2: live barge-in
# ----------------------------------------------------------------------

def _barge_speech_config(**overrides):
    barge = {
        "enabled": True,
        "require_visitor_in_frame": True,
        "min_speech_ms": 160,
        "playback_vad_threshold_multiplier": 1.3,
        "cooldown_ms": 300,
    }
    barge.update(overrides)

    return DictConfig(
        {
            "speech": {
                "audio": {
                    "sample_rate": 16000,
                    "chunk_size": 1024,
                    "device": None,
                    "queue_max_chunks": 100,
                    "input_gain": 1.0,
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


class LoudVAD:
    """VAD double for a normal-speaking visitor at laptop-mic level."""

    energy_threshold = 0.0035

    @staticmethod
    def calculate_energy(audio):
        if audio is None or len(audio) == 0:
            return 0.0
        arr = np.asarray(audio, dtype=np.float32)
        return float(np.sqrt(np.mean(np.square(arr))))

    @staticmethod
    def is_speech(audio):
        return LoudVAD.calculate_energy(audio) >= 0.0035


class SilentSTT:
    def __init__(self, **kwargs):
        pass


class NoopAudioCapture:
    def __init__(self, **kwargs):
        pass


def make_speech_service(monkeypatch, bus, config=None):
    monkeypatch.setattr(speech_module, "SpeechToText", SilentSTT)
    monkeypatch.setattr(speech_module, "AudioCapture", NoopAudioCapture)

    container = types.SimpleNamespace(
        logger=logging.getLogger("barge-in-round3"),
        event_bus=bus,
        config=config or _barge_speech_config(),
    )

    service = SpeechService(container)
    service.vad = LoudVAD()
    service.stt = None

    return service


def test_realistic_laptop_mic_speech_fires_barge_in(monkeypatch, bus):
    service = make_speech_service(monkeypatch, bus)

    events = []
    bus.subscribe("speech.barge_in", lambda **payload: events.append(payload))

    bus.publish("visitor.detected", name="Saksham", sitting_id=1, greet=True)
    bus.publish("tts.started", turn_id=1, text="Answer")

    # RMS 0.008 is what a visitor actually reaches on the kiosk mic; the
    # old 1.8x gate (0.0063) plus the old mic handling never fired.
    for _ in range(4):
        service._handle_chunk(np.full(1024, 0.008, dtype=np.float32))

    assert len(events) == 1
    assert events[0]["rms"] == pytest.approx(0.008, abs=1e-6)
    assert service.state == SpeechService.STATE_RECORDING
    assert service.audio_buffer


def test_quiet_room_noise_does_not_self_interrupt(monkeypatch, bus):
    service = make_speech_service(monkeypatch, bus)

    events = []
    bus.subscribe("speech.barge_in", lambda **payload: events.append(payload))

    bus.publish("visitor.detected", name="Saksham", sitting_id=1, greet=True)
    service._tts_speaking = True
    service.state = SpeechService.STATE_LISTENING

    # Below the adaptive gate (VAD 0.0035 * 1.3 = 0.00455).
    for _ in range(10):
        service._handle_chunk(np.full(1024, 0.001, dtype=np.float32))

    assert events == []
    assert service.state == SpeechService.STATE_LISTENING


class FakeOutputStream:
    """Output stream whose write() blocks until aborted, like real audio."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.writes = 0
        self.aborted = False
        self.closed = False
        self._release = threading.Event()

    def start(self):
        return None

    def write(self, data):
        self.writes += 1
        self._release.wait(2.0)

    def abort(self):
        self.aborted = True
        self._release.set()

    def stop(self):
        self._release.set()

    def close(self):
        self.closed = True


def write_silent_wav(path, seconds=1.0):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * int(16000 * seconds))


def test_tts_stop_aborts_active_playback_quickly(monkeypatch, tmp_path):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"placeholder")

    container = types.SimpleNamespace(
        logger=logging.getLogger("tts-round3"),
        event_bus=EventBus(),
        config=None,
    )

    service = TTSService(container, model_path=model, engine="subprocess")

    stream = FakeOutputStream()
    fake_sounddevice = types.SimpleNamespace(OutputStream=lambda **kwargs: stream)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    wav_path = tmp_path / "speech.wav"
    write_silent_wav(wav_path)

    service._cancelled = False
    service._stop_event.clear()

    played = []

    worker = threading.Thread(
        target=lambda: played.append(service._play_audio(str(wav_path))),
        daemon=True,
    )
    worker.start()

    deadline = time.perf_counter() + 2.0
    while stream.writes == 0 and time.perf_counter() < deadline:
        time.sleep(0.005)

    assert stream.writes >= 1, "playback never started"

    started = time.perf_counter()
    service.stop()
    worker.join(timeout=1.0)

    assert not worker.is_alive(), "stop() did not unblock playback"
    assert stream.aborted is True
    assert service._stop_event.is_set()
    assert played == [True]
    assert (time.perf_counter() - started) < 0.5


def test_barge_in_stops_playback_and_answers_the_interrupting_utterance(
    monkeypatch, bus
):
    monkeypatch.setattr(conversation_module, "KnowledgeBase", FakeKB)

    speech = make_speech_service(monkeypatch, bus)

    tts = FakeTTS()
    llm = FakeLLM()
    llm.delay = 0.0

    container = types.SimpleNamespace(
        logger=logging.getLogger("barge-in-manager-round3"),
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

    manager = ConversationManager(container, tts_service=tts, llm=llm)
    manager.knowledge = FakeKB()

    responses = []
    bus.subscribe(
        "llm_response_ready",
        lambda **payload: responses.append(payload),
    )

    bus.publish("visitor.detected", name="Saksham", sitting_id=1, greet=True)
    bus.publish("tts.started", turn_id=1, text="The fee is...")

    for _ in range(4):
        speech._handle_chunk(np.full(1024, 0.008, dtype=np.float32))

    assert tts.stop_calls >= 1, "barge-in did not stop TTS playback"

    # The interrupting utterance is transcribed and answered on a new turn.
    bus.publish(
        "speech_recognized",
        text="What is the MBA total fee?",
        barge_in=True,
        t_speech_end=time.perf_counter(),
        t_stt_done=time.perf_counter(),
    )

    deadline = time.perf_counter() + 3.0
    while not responses and time.perf_counter() < deadline:
        time.sleep(0.01)

    assert responses, "the interrupting utterance was never answered"
    assert any(turn_id >= 2 for turn_id, _ in tts.spoken)


# ----------------------------------------------------------------------
# Fix 3: no placeholder echoes
# ----------------------------------------------------------------------

def test_prompts_do_not_contain_a_copyable_placeholder():
    for prompt in (
        PromptManager.system_prompt(),
        PromptManager.general_system_prompt(),
        PromptManager().build_prompt("What is the fee?", [], context="Q: fee"),
        PromptManager().build_prompt("Hello", [], lane="general"),
    ):
        lowered = prompt.casefold()

        assert "short spoken answer" not in lowered
        assert "<short" not in lowered
        assert "<reply>" not in lowered


@pytest.mark.parametrize(
    "raw",
    [
        "short spoken answer",
        "<short spoken answer>",
        "Your answer: short spoken answer",
        "<reply>",
    ],
)
def test_placeholder_only_replies_never_reach_the_speaker(raw):
    assert strip_placeholders(strip_reply_prefix(raw)) == ""

    reply = ResponseParser().parse(
        '{"reply": "%s", "emotion": "neutral", "action": "answer"}' % raw
    ).reply

    assert reply
    assert "short spoken answer" not in reply.casefold()
    assert "<reply>" not in reply.casefold()


def test_placeholder_echo_is_not_streamed_to_tts():
    extractor = ReplyStreamExtractor()
    chunker = SentenceChunker(min_chars=8, max_chars=240)

    tokens = [
        '{"reply": "<short',
        " spoken answer>",
        '", "emotion": "neutral", "action": "answer"}',
    ]

    decoded = ""
    sentences = []

    for token in tokens:
        chunk = extractor.feed(token)
        decoded += chunk
        sentences.extend(chunker.feed(chunk))

    sentences.append(chunker.flush())
    sentences = [sentence for sentence in sentences if sentence]

    assert "short spoken answer" not in decoded.casefold()
    assert sentences == []


def test_placeholder_is_stripped_from_a_real_sentence():
    assert strip_placeholders("The fee is <short spoken answer> 1,25,000.") == (
        "The fee is  1,25,000."
    )

    assert "short spoken answer" not in ReplyPrefixFilter().feed("short spoken answer")


def test_spoken_answer_preparation_drops_a_placeholder():
    assert ConversationManager._prepare_spoken_answer("short spoken answer") == ""
