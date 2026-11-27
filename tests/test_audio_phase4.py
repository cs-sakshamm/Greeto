"""
Phase 4 Speech Recognition and Audio Pipeline Automated Tests.

Validates:
- AudioCapture initialization, device resolution, resampling, queue overflow protection.
- VoiceActivityDetector with WebRTC VAD and energy fallback on silence, speech, and noise.
- SpeechToText offline Whisper model loading, transcription, hallucination filtering.
- SpeechService lifecycle states (IDLE -> LISTENING -> RECORDING -> PROCESSING),
  pre-speech buffer preservation, end-of-speech silence segmentation,
  acoustic feedback suppression during TTS speaking, and ConversationManager integration.

Run:
    python tests/test_audio_phase4.py
"""

from __future__ import annotations

import logging
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from app.speech.audio_capture import AudioCapture, resample_audio
from app.speech.vad import VoiceActivityDetector
from app.speech.stt import SpeechToText
from app.speech.speech_service import SpeechService
from app.core.event_bus import EventBus

FAILURES = []


def check(name, fn):
    """Execute a single check, measure elapsed time, record failure if any."""
    started = time.perf_counter()
    try:
        fn()
        elapsed = time.perf_counter() - started
        print("[PASS] %-64s (%.2fs)" % (name, elapsed))
    except Exception as error:  # noqa: BLE001
        print("[FAIL] %-64s -> %s: %s" % (name, type(error).__name__, error))
        FAILURES.append(name)


class FakeConfig:
    def __init__(self, data=None):
        self.data = data or {}

    def get_value(self, *keys, default=None):
        curr = self.data
        for k in keys:
            if isinstance(curr, dict) and k in curr:
                curr = curr[k]
            else:
                return default
        return curr if curr is not None else default


class FakeContainer:
    def __init__(self, config_data=None):
        self.event_bus = EventBus()
        self.config = FakeConfig(config_data or {})
        self.logger = logging.getLogger("test_audio_phase4")
        self.logger.addHandler(logging.NullHandler())
        self.logger.propagate = False


def generate_tone(freq=440.0, duration=0.5, sr=16000, amp=0.5):
    """Generate a pure sine tone."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def generate_noise(duration=0.5, sr=16000, amp=0.005):
    """Generate low-amplitude white noise."""
    return (np.random.randn(int(sr * duration)) * amp).astype(np.float32)


# ======================================================================
# 1. Microphone Capture & Device Management Tests (Milestone 4.1)
# ======================================================================

def test_capture_initialization():
    cap = AudioCapture(sample_rate=16000, chunk_size=1024, channels=1)
    assert cap.sample_rate == 16000
    assert cap.chunk_size == 1024
    assert cap.channels == 1
    assert not cap.running
    assert cap.audio_queue.empty()


def test_device_enumeration():
    devices = AudioCapture.list_input_devices()
    assert isinstance(devices, list)
    for dev in devices:
        assert "index" in dev
        assert "name" in dev
        assert dev["channels"] > 0


def test_device_resolution_invalid_type_or_missing():
    # String for non-existent device
    try:
        AudioCapture.resolve_device("TotallyFakeMicrophone_9999XYZ")
        assert False, "Should have raised ValueError for non-existent device"
    except ValueError as e:
        assert "No input audio device matches" in str(e) or "Invalid" in str(e)


def test_capture_bounded_queue_drop_oldest():
    cap = AudioCapture(queue_max_chunks=3)
    chunk1 = np.array([1.0], dtype=np.float32)
    chunk2 = np.array([2.0], dtype=np.float32)
    chunk3 = np.array([3.0], dtype=np.float32)
    chunk4 = np.array([4.0], dtype=np.float32)

    cap._push_chunk(chunk1)
    cap._push_chunk(chunk2)
    cap._push_chunk(chunk3)
    assert cap.audio_queue.qsize() == 3

    # Pushing 4th chunk drops oldest (chunk1)
    cap._push_chunk(chunk4)
    assert cap.audio_queue.qsize() == 3

    items = [cap.get_audio(timeout=0.1)[0] for _ in range(3)]
    assert items == [2.0, 3.0, 4.0]


def test_capture_resampling():
    orig_sr = 48000
    target_sr = 16000
    tone_48k = generate_tone(freq=440.0, duration=0.1, sr=orig_sr, amp=0.8)
    resampled_16k = resample_audio(tone_48k, orig_sr, target_sr)

    expected_len = int(round(len(tone_48k) * target_sr / orig_sr))
    assert abs(len(resampled_16k) - expected_len) <= 1
    assert resampled_16k.dtype == np.float32


def test_capture_stop_cleanup():
    cap = AudioCapture()
    cap._push_chunk(np.zeros(100, dtype=np.float32))
    assert not cap.audio_queue.empty()
    cap.stop()
    assert cap.audio_queue.empty()
    assert not cap.running


def test_capture_error_callback():
    errors = []
    cap = AudioCapture(device="InvalidDevice_XYZ", on_error=lambda msg: errors.append(msg))
    # Directly invoke capture loop to test error capture without hanging
    cap._capture_loop()
    assert len(errors) == 1
    assert "InvalidDevice_XYZ" in errors[0]


# ======================================================================
# 2. Voice Activity & Silence Detection Tests (Milestone 4.2)
# ======================================================================

def test_vad_empty_and_none():
    vad = VoiceActivityDetector()
    assert vad.is_speech(None) is False
    assert vad.is_speech(np.array([], dtype=np.float32)) is False
    assert vad.calculate_energy(None) == 0.0


def test_vad_silence_below_energy_threshold():
    vad = VoiceActivityDetector(energy_threshold=0.02)
    silence = np.zeros(1600, dtype=np.float32)
    assert vad.is_speech(silence) is False
    assert vad.calculate_energy(silence) == 0.0


def test_vad_quiet_background_noise():
    vad = VoiceActivityDetector(energy_threshold=0.03)
    noise = generate_noise(duration=0.2, amp=0.005)
    assert vad.is_speech(noise) is False


def test_vad_speech_detection():
    vad = VoiceActivityDetector(energy_threshold=0.01)
    tone = generate_tone(freq=300.0, duration=0.2, amp=0.5)
    # Energy is high; webrtcvad or energy fallback evaluates it as speech
    assert vad.is_speech(tone) is True
    assert vad.calculate_energy(tone) > 0.1


def test_vad_analyze_chunk():
    vad = VoiceActivityDetector(energy_threshold=0.01)
    tone = generate_tone(freq=300.0, duration=0.09, amp=0.5)  # 90ms = three 30ms frames
    is_speech, energy, speech_frames, total_frames = vad.analyze_chunk(tone)
    assert is_speech is True
    assert energy > 0.05
    assert total_frames >= 1


# ======================================================================
# 3. Whisper STT Reliability & Text Normalization Tests (Milestone 4.3)
# ======================================================================

def test_stt_missing_model_raises():
    try:
        SpeechToText(model_path="models/whisper/does_not_exist.bin")
        assert False, "Should raise FileNotFoundError"
    except FileNotFoundError as err:
        assert "not found" in str(err)


def test_stt_clean_text_normalization():
    # Use dummy instance with mocked model
    with patch("pywhispercpp.model.Model"):
        # Create temp file so model_path exists
        with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
            stt = SpeechToText(model_path=tmp.name, min_text_length=3)

            # Known hallucinations
            assert stt.clean_text("[BLANK_AUDIO]") == ""
            assert stt.clean_text("[silence]") == ""
            assert stt.clean_text("(music)") == ""
            assert stt.clean_text("Thank you.") == ""
            assert stt.clean_text("Thanks for watching!") == ""

            # Repeated loops
            assert stt.clean_text("you you you you") == ""

            # Punctuation only
            assert stt.clean_text("... ???") == ""

            # Short text under min_text_length
            assert stt.clean_text("ok") == ""

            # Valid text with bracketed tokens stripped
            valid = stt.clean_text("[applause] Where is the CSE department? [cheering]")
            assert valid == "Where is the CSE department?"


def test_stt_live_model_transcription():
    model_path = Path("models/whisper/ggml-tiny.en.bin")
    if not model_path.exists():
        print("       (Skipping live model check: ggml-tiny.en.bin not found)")
        return

    stt = SpeechToText(model_path=str(model_path), n_threads=4)
    assert stt.is_available()

    # Pure silence -> [BLANK_AUDIO] which clean_text safely filters to empty string
    silence = np.zeros(16000, dtype=np.float32)
    result = stt.transcribe(silence)
    assert result == "", f"Expected empty text for silence, got '{result}'"


# ======================================================================
# 4. SpeechService Pipeline & Lifecycle Tests (Milestone 4.4 & 4.5)
# ======================================================================

def test_speech_service_lifecycle_states():
    container = FakeContainer({
        "speech": {
            "audio": {"min_recording_seconds": 0.1, "max_recording_seconds": 2.0},
            "vad": {"min_speech_ms": 60, "silence_ms": 120, "pre_speech_ms": 120},
        }
    })
    service = SpeechService(container)
    assert service.state == SpeechService.STATE_IDLE

    # Mock audio_capture to avoid opening physical mic
    service.audio_capture = MagicMock()
    service.start()
    assert service.state == SpeechService.STATE_LISTENING
    assert service.running is True

    service.stop()
    assert service.state == SpeechService.STATE_IDLE
    assert service.running is False


def test_speech_service_recording_events_and_pre_speech():
    container = FakeContainer({
        "speech": {
            "audio": {"chunk_size": 1024, "sample_rate": 16000, "min_recording_seconds": 0.1},
            "vad": {"min_speech_ms": 60, "silence_ms": 120, "pre_speech_ms": 300},
        }
    })
    service = SpeechService(container)
    events = []
    for ev in (
        "speech.listening_started",
        "speech.recording_started",
        "speech.recording_finished",
        "speech.transcription_started",
        "speech.transcription_completed",
        "speech.no_speech_detected",
    ):
        container.event_bus.subscribe(ev, lambda ev_name=ev, **kw: events.append((ev_name, kw)))

    service.audio_capture = MagicMock()
    service._set_state(SpeechService.STATE_LISTENING)

    # Mock VAD
    service.vad = MagicMock()
    chunk = np.zeros(1024, dtype=np.float32)

    # 1. Provide silent chunks (accumulating in pre_speech_buffer)
    service.vad.is_speech.return_value = False
    service._handle_chunk(chunk)
    service._handle_chunk(chunk)
    assert len(service.pre_speech_buffer) == 2
    assert service.state == SpeechService.STATE_LISTENING

    # 2. Provide speech chunks satisfying min_speech_streak
    service.vad.is_speech.return_value = True
    service._handle_chunk(chunk)
    assert service.state == SpeechService.STATE_RECORDING
    recording_starts = [e for e in events if e[0] == "speech.recording_started"]
    assert len(recording_starts) == 1, "speech.recording_started should be emitted exactly once"

    # Audio buffer should contain pre-speech buffer + first speech chunk
    assert len(service.audio_buffer) >= 3

    # 3. Continuous speech chunks should not re-emit recording_started
    service._handle_chunk(chunk)
    recording_starts = [e for e in events if e[0] == "speech.recording_started"]
    assert len(recording_starts) == 1

    # 4. Provide silence to trigger end-of-speech segmentation
    service.vad.is_speech.return_value = False
    service._handle_chunk(chunk)
    service._handle_chunk(chunk)
    assert service.state == SpeechService.STATE_PROCESSING

    recording_finishes = [e for e in events if e[0] == "speech.recording_finished"]
    assert len(recording_finishes) == 1
    assert "duration_seconds" in recording_finishes[0][1]


def test_speech_service_acoustic_feedback_suppression():
    container = FakeContainer({"speech": {"pipeline": {"mute_during_speech": True}}})
    service = SpeechService(container)
    service.audio_capture = MagicMock()
    service._set_state(SpeechService.STATE_LISTENING)

    # Simulate TTS starting
    container.event_bus.publish("speech.speaking_started", text="Hello visitor", utterance=1)
    assert service._tts_speaking is True

    # Incoming loud speech while TTS is speaking must be ignored
    service.vad = MagicMock()
    service.vad.is_speech.return_value = True
    chunk = generate_tone(440, 0.1)
    service._handle_chunk(chunk)
    assert service.state == SpeechService.STATE_LISTENING
    assert len(service.audio_buffer) == 0

    # TTS finishes
    container.event_bus.publish("speech.speaking_finished", utterance=1)
    assert service._tts_speaking is False
    assert service._tts_cooldown_until > time.monotonic()


def test_speech_service_conversation_manager_dispatch():
    container = FakeContainer()
    service = SpeechService(container)

    received_speech = []
    container.event_bus.subscribe("speech_recognized", lambda text, **kw: received_speech.append(text))

    # Mock STT to return a valid question
    mock_stt = MagicMock()
    mock_stt.is_available.return_value = True
    mock_stt.transcribe.return_value = "What is the fee for B.Tech CSE?"
    service.stt = mock_stt

    # Process valid audio
    audio = np.ones(16000, dtype=np.float32)
    service._process_recorded_audio(audio)

    assert received_speech == ["What is the fee for B.Tech CSE?"]


def test_speech_service_rejects_empty_transcription():
    container = FakeContainer()
    service = SpeechService(container)

    received_speech = []
    no_speech_events = []
    container.event_bus.subscribe("speech_recognized", lambda text, **kw: received_speech.append(text))
    container.event_bus.subscribe("speech.no_speech_detected", lambda **kw: no_speech_events.append(kw))

    # Mock STT to return empty
    mock_stt = MagicMock()
    mock_stt.is_available.return_value = True
    mock_stt.transcribe.return_value = ""
    service.stt = mock_stt

    audio = np.ones(16000, dtype=np.float32)
    service._process_recorded_audio(audio)

    assert len(received_speech) == 0, "Empty transcription must not reach ConversationManager"
    assert len(no_speech_events) == 1


# ======================================================================
# Main Execution
# ======================================================================

def main():
    print("Greeto Phase 4 Speech Pipeline Automated Tests")
    print("=" * 66)

    # 1. Microphone capture tests
    check("M1. AudioCapture initialization & defaults", test_capture_initialization)
    check("M2. AudioCapture device enumeration", test_device_enumeration)
    check("M3. AudioCapture invalid device resolution", test_device_resolution_invalid_type_or_missing)
    check("M4. AudioCapture bounded queue drop-oldest", test_capture_bounded_queue_drop_oldest)
    check("M5. AudioCapture resampling (48kHz -> 16kHz)", test_capture_resampling)
    check("M6. AudioCapture stop & queue draining", test_capture_stop_cleanup)
    check("M7. AudioCapture error callback propagation", test_capture_error_callback)

    # 2. VAD tests
    check("V1. VAD empty and None input safety", test_vad_empty_and_none)
    check("V2. VAD silence below energy threshold rejected", test_vad_silence_below_energy_threshold)
    check("V3. VAD ambient room noise rejected", test_vad_quiet_background_noise)
    check("V4. VAD human speech tone detection", test_vad_speech_detection)
    check("V5. VAD detailed chunk frame analysis", test_vad_analyze_chunk)

    # 3. Whisper STT tests
    check("W1. SpeechToText missing model file error handling", test_stt_missing_model_raises)
    check("W2. SpeechToText text normalization & hallucination rejection", test_stt_clean_text_normalization)
    check("W3. SpeechToText live offline Whisper transcription", test_stt_live_model_transcription)

    # 4. SpeechService pipeline tests
    check("S1. SpeechService lifecycle states (IDLE/LISTENING/STOP)", test_speech_service_lifecycle_states)
    check("S2. SpeechService VAD segmentation & pre-speech buffer", test_speech_service_recording_events_and_pre_speech)
    check("S3. SpeechService acoustic feedback suppression during TTS", test_speech_service_acoustic_feedback_suppression)
    check("S4. SpeechService dispatches valid speech to ConversationManager", test_speech_service_conversation_manager_dispatch)
    check("S5. SpeechService rejects empty/hallucinated transcription", test_speech_service_rejects_empty_transcription)

    print("-" * 66)
    if FAILURES:
        print("RESULT: %d FAILED -> %s" % (len(FAILURES), FAILURES))
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
