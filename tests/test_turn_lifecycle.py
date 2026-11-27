"""
Tests for turn_id tracking, stale-turn cancellation, and caption synchronization.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from app.brain.conversation_manager import ConversationManager
from app.brain.parser import LLMResponse
from app.core.event_bus import EventBus
from app.gui.kiosk_presenter import KioskPresenter
from app.speech.tts_service import TTSService


class DummyContainer:
    def __init__(self):
        self.event_bus = EventBus()
        self.logger = MagicMock()


class MockTTS:
    def __init__(self):
        self.spoken = []
        self._cancelled = False
        self._is_speaking = False

    def is_speaking(self) -> bool:
        return self._is_speaking

    def cancel(self):
        self._cancelled = True

    def speak(self, text, turn_id=None):
        if self._cancelled:
            return False
        self.spoken.append((text, turn_id))
        return True


class MockLLM:
    def __init__(self, stream_fn=None):
        self._stream_fn = stream_fn

    def stream_generate(self, prompt):
        if self._stream_fn:
            yield from self._stream_fn(prompt)
        else:
            yield '{"reply": "OK", "emotion": "neutral", "action": "answer"}'


def test_monotonic_turn_id_generation():
    container = DummyContainer()
    tts = MockTTS()
    llm = MockLLM()
    manager = ConversationManager(container, tts_service=tts, llm=llm)

    manager.knowledge = MagicMock()
    manager.knowledge.search = MagicMock(return_value="")

    manager.handle_speech("First query from visitor")
    time.sleep(0.05)
    first_turn = manager._current_turn_id
    assert first_turn >= 1

    time.sleep(0.35)
    manager.handle_speech("Second query from visitor")
    time.sleep(0.05)
    second_turn = manager._current_turn_id
    assert second_turn > first_turn


def test_stale_turn_cancellation_during_generation():
    """If a new turn starts while an earlier turn is generating, earlier turn is discarded."""
    container = DummyContainer()
    tts = MockTTS()
    barrier = threading.Event()

    def slow_generate(prompt):
        barrier.wait(timeout=2.0)
        yield '{"reply": "Slow reply", "emotion": "neutral", "action": "answer"}'

    llm = MockLLM(stream_fn=slow_generate)
    manager = ConversationManager(container, tts_service=tts, llm=llm)
    manager.knowledge = MagicMock()
    manager.knowledge.search = MagicMock(return_value="")

    # Start turn 1 in thread
    t1 = threading.Thread(target=manager.process_message, args=("Question 1", 1))
    manager._current_turn_id = 1
    t1.start()

    time.sleep(0.05)
    # Turn 2 arrives and supersedes turn 1
    manager._current_turn_id = 2
    # Release slow generation
    barrier.set()
    t1.join(timeout=2.0)

    # Turn 1 must have been aborted and never sent to TTS
    assert len(tts.spoken) == 0


def test_kiosk_presenter_caption_synchronized_with_speech_start(app):
    """Captions must not display on llm_response_ready; only on speech.speaking_started."""
    bus = EventBus()
    presenter = KioskPresenter(event_bus=bus)

    captions = []
    presenter.caption.connect(lambda text, duration: captions.append(text))

    # Trigger processing
    bus.publish("conversation.processing", text="Fee query", turn_id=1)
    app.processEvents()
    assert len(captions) == 1
    assert "check" in captions[0].lower()

    # LLM response ready emitted
    bus.publish(
        "llm_response_ready",
        response=LLMResponse(
            reply="The B.Tech fee is 1.25 Lakh per year.",
            emotion="happy",
            action="answer",
        ),
        turn_id=1,
    )
    app.processEvents()

    # Caption list must NOT have added the reply yet!
    assert len(captions) == 1

    # Audio playback starts
    bus.publish(
        "speech.speaking_started",
        text="The B.Tech fee is 1.25 Lakh per year.",
        utterance=1,
        turn_id=1,
    )
    app.processEvents()

    # Now caption is displayed!
    assert len(captions) == 2
    assert "1.25 Lakh" in captions[1]
    assert presenter.current_state() == "speaking"

    # Playback finishes
    bus.publish("speech.speaking_finished", utterance=1, turn_id=1)
    app.processEvents()
    assert presenter.current_state() == "happy"


def test_kiosk_presenter_stale_turn_discarded(app):
    """KioskPresenter discards response and speaking events from superseded turns."""
    bus = EventBus()
    presenter = KioskPresenter(event_bus=bus)

    captions = []
    presenter.caption.connect(lambda text, duration: captions.append(text))

    # Turn 1 processing
    bus.publish("conversation.processing", text="Query 1", turn_id=1)
    app.processEvents()

    # Turn 2 processing supersedes turn 1
    bus.publish("conversation.processing", text="Query 2", turn_id=2)
    app.processEvents()

    # Late response from turn 1 arrives
    bus.publish(
        "llm_response_ready",
        response=LLMResponse(reply="Stale reply 1", emotion="neutral", action="answer"),
        turn_id=1,
    )
    # Late speaking start from turn 1 arrives
    bus.publish("speech.speaking_started", text="Stale reply 1", utterance=1, turn_id=1)
    app.processEvents()

    # Stale reply must not have been shown
    assert all("Stale reply 1" not in c for c in captions)
    assert presenter.current_state() == "thinking"


def test_visitor_greeting_triggers_spoken_audio(app):
    """When a visitor is recognized, KioskPresenter requests TTS speech."""
    bus = EventBus()
    tts = MockTTS()
    container = DummyContainer()
    container.event_bus = bus
    manager = ConversationManager(container, tts_service=tts, llm=MockLLM())
    presenter = KioskPresenter(event_bus=bus)

    # Simulate visitor detection event
    f = MagicMock()
    f.visitor_id = "saksham_1"
    bus.publish("visitor.detected", name="Saksham", face=f)

    # Process Qt queued events and wait for background greeting thread to invoke TTS
    for _ in range(20):
        app.processEvents()
        if len(tts.spoken) > 0:
            break
        time.sleep(0.05)

    # Verify TTS was called with greeting text containing Saksham's name
    assert len(tts.spoken) == 1
    spoken_text, turn_id = tts.spoken[0]
    assert "Saksham" in spoken_text
    assert "Welcome back" in spoken_text
    assert turn_id >= 1


def test_invented_numeric_fee_is_replaced_with_grounded_answer():
    """A hallucinated fee not present in retrieved knowledge is not spoken."""
    container = DummyContainer()
    tts = MockTTS()

    def invented_fee(prompt):
        yield '{"reply": "The B.Tech CSE fee is ₹1,00,000 per year.", '
        yield '"emotion": "happy", "action": "answer"}'

    manager = ConversationManager(
        container, tts_service=tts, llm=MockLLM(stream_fn=invented_fee)
    )
    manager.knowledge = MagicMock()
    manager.knowledge.search = MagicMock(
        return_value="Fees are determined by the programme and admission category."
    )

    manager.process_message("What is the fee for B.Tech CSE?")

    assert len(tts.spoken) == 1
    spoken, _ = tts.spoken[0]
    assert "couldn't find" in spoken.lower()
    assert "1,00,000" not in spoken
    assert "1800-180-0840" in spoken


def test_short_response_uses_single_chunk():
    """Short responses (<= 250 chars) are synthesized in one single pass."""
    service = TTSService(
        DummyContainer(),
        model_path="models/tts/en_US-lessac-medium.onnx",
        engine="subprocess",
    )
    service._play_audio = MagicMock(return_value=True)

    from unittest.mock import patch

    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.command = command
            self.returncode = 0
            self.output_path = command[command.index("--output_file") + 1]

        def communicate(self, text):
            # Write non-empty dummy WAV
            with open(self.output_path, "wb") as f:
                f.write(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
            return (None, None)

    with patch("subprocess.Popen", side_effect=FakeProcess) as mock_popen:
        assert service.speak("Welcome to I.T.S Engineering College. I am delighted to help you.") is True

        # Exactly 1 subprocess call for both sentences!
        assert mock_popen.call_count == 1
        command = mock_popen.call_args[0][0]
        assert "--output_file" in command


def test_long_response_prefetches_next_sentence_during_playback():
    """Long replies synthesize the next sentence while the current one plays."""
    from unittest.mock import patch

    service = TTSService(
        DummyContainer(),
        model_path="models/tts/en_US-lessac-medium.onnx",
        engine="subprocess",
    )
    events = []

    service.event_bus.subscribe(
        "speech.speaking_started",
        lambda **kw: events.append(("start", kw["sentence_index"])),
    )
    service.event_bus.subscribe(
        "speech.speaking_finished",
        lambda **kw: events.append(("finish", kw["sentence_index"])),
    )

    sentences = [
        "Our admission process is simple and transparent for every applicant.",
        "You can apply online with your class twelve marksheets and id proof.",
        "Our placement cell has trained students for more than two decades.",
        "Scholarships are available for meritorious and deserving candidates.",
    ]
    index_by_text = {sentence: i for i, sentence in enumerate(sentences)}

    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.command = command
            self.returncode = 0
            self.output_path = command[command.index("--output_file") + 1]

        def communicate(self, text):
            i = index_by_text[text]
            events.append(("synth_begin", i))
            time.sleep(0.05)
            with open(self.output_path, "wb") as f:
                f.write(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
            events.append(("synth_end", i))
            return (None, None)

    def fake_play(path):
        plays = [e for e in events if e[0] == "play"]
        events.append(("play", len(plays)))
        time.sleep(0.08)
        return True

    service._play_audio = fake_play

    with patch("subprocess.Popen", side_effect=FakeProcess) as mock_popen:
        assert service.speak(" ".join(sentences)) is True

    assert mock_popen.call_count == 4

    # Every sentence is spoken in order, and each start is paired with a finish.
    assert events.count(("start", 0)) == 1
    assert events.count(("start", 3)) == 1
    assert [e for e in events if e[0] == "play"] == [
        ("play", 0),
        ("play", 1),
        ("play", 2),
        ("play", 3),
    ]

    # Sentence 1 is synthesized while sentence 0 is still playing.
    assert events.index(("play", 0)) < events.index(("synth_begin", 1))
    assert events.index(("synth_end", 1)) < events.index(("finish", 0))

