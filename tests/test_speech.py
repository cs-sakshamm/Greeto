"""
Phase 3 Milestone 4 speech-output checks.

These use a temporary model placeholder, a fake Piper process and a fake
audio player. No model is loaded, nothing is played and no microphone is
opened.

    python tests/test_speech.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:

    sys.path.insert(0, str(ROOT))

import app.brain.conversation_manager as conversation_module  # noqa: E402
from app.core.event_bus import EventBus  # noqa: E402
from app.speech import tts_service as tts_module  # noqa: E402
from app.speech.tts_service import TTSService  # noqa: E402


FAILURES = []


def check(name, fn):
    """Run one check and record the outcome."""

    started = time.perf_counter()

    try:

        fn()

        elapsed = time.perf_counter() - started

        print("[PASS] %-62s (%.2fs)" % (name, elapsed))

    except Exception as error:  # noqa: BLE001 - harness reports everything

        print("[FAIL] %-62s -> %s: %s" % (name, type(error).__name__, error))

        FAILURES.append(name)


class FakeContainer:
    """Container subset used by the real EventBus and a test logger."""

    def __init__(self):

        self.event_bus = EventBus()

        self.logger = logging.getLogger("speech-tests")

        # A deliberately failing Piper run is part of A3; do not print
        # its expected traceback as test noise.
        self.logger.addHandler(logging.NullHandler())
        self.logger.propagate = False


class FakeProcess:
    """Stand-in for the Piper subprocess; writes the output path."""

    def __init__(self, command):

        self.command = command

        self.returncode = None

        self.text = None

        self.output_path = Path(
            command[command.index("--output_file") + 1]
        )

    def communicate(self, text):

        self.text = text

        self.output_path.write_bytes(b"fake wav")

        time.sleep(0.03)

        self.returncode = 0

        return (None, None)


class FakePlayer:
    """Thread-safe playback probe that never touches an audio device."""

    def __init__(self):

        self._lock = threading.Lock()

        self.active = 0

        self.max_active = 0

        self.played = []

    def __call__(self, path):

        with self._lock:

            self.active += 1

            self.max_active = max(self.max_active, self.active)

        time.sleep(0.03)

        with self._lock:

            self.active -= 1

            self.played.append(path)

        return True


def record_events(bus):
    """Capture every speech lifecycle event."""

    events = []

    def make_handler(name):

        def handler(**payload):

            events.append((name, payload))

        return handler

    for name in (
        "speech.speaking_started",
        "speech.speaking_finished",
        "speech.error",
    ):

        bus.subscribe(name, make_handler(name))

    return events


def make_service(tmp, player, process_factory=FakeProcess):
    """Build a TTS service with a placeholder model and fake playback."""

    model = Path(tmp) / "voice.onnx"

    model.write_bytes(b"not a real model")

    service = TTSService(
        FakeContainer(),
        model_path=model,
        engine="subprocess",
    )

    service._play_audio = player

    return service


# ----------------------------------------------------------------------
# A. Shared use and lifecycle
# ----------------------------------------------------------------------

def test_shared_service_serializes_speech():

    with tempfile.TemporaryDirectory() as tmp:

        player = FakePlayer()

        service = make_service(tmp, player)

        events = record_events(service.event_bus)

        original_popen = tts_module.subprocess.Popen

        tts_module.subprocess.Popen = (
            lambda command, **kwargs: FakeProcess(command)
        )

        results = {}

        barrier = threading.Barrier(2)

        def speak(slot, text):

            barrier.wait()

            results[slot] = service.speak(text)

        threads = [
            threading.Thread(target=speak, args=(1, "first")),
            threading.Thread(target=speak, args=(2, "second")),
        ]

        try:

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

        finally:

            tts_module.subprocess.Popen = original_popen

        assert results == {1: True, 2: True}, results

        assert player.max_active == 1, (
            "two callers played audio at the same time (max active %d)"
            % player.max_active
        )

        assert len(player.played) == 2

        assert all(not Path(path).exists() for path in player.played), (
            "a temporary WAV was left behind"
        )

        started = [
            payload
            for name, payload in events
            if name == "speech.speaking_started"
        ]

        finished = [
            payload
            for name, payload in events
            if name == "speech.speaking_finished"
        ]

        assert len(started) == 2, events

        assert len(finished) == 2, events

        assert {event["text"] for event in started} == {"first", "second"}

        # Numbers are allocated under the same lock as playback, so the
        # completed utterances are exactly the started ones.
        assert {event["utterance"] for event in started} == {1, 2}

        assert (
            {event["utterance"] for event in finished}
            == {event["utterance"] for event in started}
        )

        print("       2 concurrent calls -> one at a time, one WAV each")


def test_empty_speech_is_ignored():

    with tempfile.TemporaryDirectory() as tmp:

        service = make_service(tmp, FakePlayer())

        events = record_events(service.event_bus)

        assert service.speak("") is False

        assert events == [], events

        assert service._utterance == 0


def test_failed_piper_reports_error_and_cleans_up():

    with tempfile.TemporaryDirectory() as tmp:

        player = FakePlayer()

        service = make_service(tmp, player)

        events = record_events(service.event_bus)

        class BrokenProcess(FakeProcess):

            def communicate(self, text):

                self.output_path.write_bytes(b"fake wav")

                self.returncode = 3

                return (None, None)

        created = []

        def broken_factory(command, **kwargs):

            process = BrokenProcess(command)

            created.append(process)

            return process

        original_popen = tts_module.subprocess.Popen

        tts_module.subprocess.Popen = broken_factory

        try:

            result = service.speak("this must fail")

        finally:

            tts_module.subprocess.Popen = original_popen

        assert result is False

        assert player.played == []

        assert [name for name, _ in events] == ["speech.error"]

        assert events[0][1]["kind"] == "tts"

        assert created and not created[0].output_path.exists(), (
            "failed speech left its temporary WAV behind"
        )

        print("       piper failure -> speech.error, no playback")


# ----------------------------------------------------------------------
# B. Injected shared TTS
# ----------------------------------------------------------------------

class StubLLM:
    """Return one valid parser response without loading llama.cpp."""

    def stream_generate(self, prompt):

        assert prompt

        return iter([
            '{"reply": "Hello from Greeto.", '
            '"emotion": "happy", "action": "answer"}'
        ])


class StubPromptManager:
    def build_prompt(self, **kwargs):

        assert kwargs["user_message"] == "hello"

        return "prompt"


class StubKnowledge:
    def search(self, text):

        return ""


class StubTTS:
    def __init__(self):

        self.spoken = []

    def speak(self, text):

        self.spoken.append(text)

        return True


def test_conversation_manager_uses_the_shared_service():

    container = FakeContainer()

    tts = StubTTS()

    received = []

    container.event_bus.subscribe(
        "llm_response_ready",
        lambda **payload: received.append(payload["response"]),
    )

    original_llm = conversation_module.LocalLLM
    original_prompt = conversation_module.PromptManager
    original_knowledge = conversation_module.KnowledgeBase

    conversation_module.LocalLLM = StubLLM
    conversation_module.PromptManager = StubPromptManager
    conversation_module.KnowledgeBase = StubKnowledge

    try:

        manager = conversation_module.ConversationManager(
            container,
            tts_service=tts,
        )

        assert manager.tts is tts

        manager.process_message("hello")

    finally:

        conversation_module.LocalLLM = original_llm
        conversation_module.PromptManager = original_prompt
        conversation_module.KnowledgeBase = original_knowledge

    assert tts.spoken == ["Hello from Greeto."], tts.spoken

    assert len(received) == 1

    assert received[0].reply == "Hello from Greeto."

    print("       ConversationManager used the injected TTSService")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main():

    print("Greeto shared speech-output checks")

    check("A1. shared TTS serializes concurrent calls",
          test_shared_service_serializes_speech)

    check("A2. empty text is ignored", test_empty_speech_is_ignored)

    check("A3. Piper failure is reported and cleaned up",
          test_failed_piper_reports_error_and_cleans_up)

    check("B1. ConversationManager uses the injected TTS",
          test_conversation_manager_uses_the_shared_service)

    print("-" * 74)

    if FAILURES:

        print("RESULT: %d FAILED -> %s" % (len(FAILURES), FAILURES))

        return 1

    print("RESULT: ALL CHECKS PASSED")

    return 0


if __name__ == "__main__":

    sys.exit(main())
