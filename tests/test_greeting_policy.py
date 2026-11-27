"""Greeting policy checks.

Covers the two rules that matter in front of the camera:

1. Unknown faces never produce a greeting (caption or TTS).
2. A known identity is greeted exactly once per sitting, even when the
   tracker flickers or re-detects the person under a new track ID. Only a
   real absence longer than the configured timeout starts a new sitting.
"""

from __future__ import annotations

import time

from app.core.event_bus import EventBus
from app.gui.kiosk_presenter import KioskPresenter
from app.vision.models import Face
from app.vision.vision_service import VisionService

from tests.test_recognition import (
    FakeCamera,
    FakeContainer,
    FakeDatabase,
    FakeModel,
    FakeVisitorDatabase,
    face,
    unit,
)


class StubFace:
    def __init__(self, visitor_id, name="Unknown"):
        self.visitor_id = visitor_id
        self.name = name


class RecordingTTS:
    def __init__(self):
        self.spoken = []

    def is_speaking(self) -> bool:
        return False

    def cancel(self):
        return None

    def speak(self, text, turn_id=None):
        self.spoken.append((text, turn_id))
        return True


def make_vision():
    container = FakeContainer()
    vision = VisionService(
        container,
        camera=FakeCamera(),
        embedding_model=FakeModel(),
        database=FakeDatabase({"Saksham": [unit(0)]}),
        visitor_database=FakeVisitorDatabase(),
    )
    return vision, container.event_bus


def moved_face(embedding, x):
    return Face(
        x=x,
        y=100,
        width=200,
        height=200,
        embedding=embedding,
        confidence=0.90,
    )


def detected_named(bus, name="visitor.detected"):
    return [payload for event, payload in bus.events if event == name]


def test_unknown_face_is_never_announced_as_known():
    vision, bus = make_vision()

    faces = vision.recognizer.recognize_all([face(unit(5))])
    vision.tracker.update(faces)
    vision._publish_events(faces)

    assert detected_named(bus, "visitor.detected") == []
    unknown = detected_named(bus, "visitor.unknown")
    assert len(unknown) == 1
    assert unknown[0]["greet"] is False


def test_known_face_greeted_once_across_frames_and_track_flicker():
    vision, bus = make_vision()

    start = time.monotonic()

    faces = vision.recognizer.recognize_all([face(unit(0))])
    vision.tracker.update(faces)
    vision._publish_events(faces, now=start)

    first = detected_named(bus)
    assert len(first) == 1
    assert first[0]["name"] == "Saksham"
    assert first[0]["greet"] is True
    sitting_id = first[0]["sitting_id"]

    # Same person, repeated recognition sweeps: no second announcement.
    vision._publish_events(faces, now=start + 1.0)
    vision._publish_events(faces, now=start + 2.0)
    assert len(detected_named(bus)) == 1

    # Tracker flickers and assigns a new track ID while the person is
    # still standing there. This is still the same sitting: announce the
    # track but do not greet again.
    flicker = vision.recognizer.recognize_all([moved_face(unit(0), 700)])
    vision.tracker.update(flicker)
    assert flicker[0].visitor_id != faces[0].visitor_id
    vision._publish_events(flicker, now=start + 3.0)

    events = detected_named(bus)
    assert len(events) == 2
    assert events[1]["greet"] is False
    assert events[1]["sitting_id"] == sitting_id

    # A brief absence shorter than the timeout is still the same sitting.
    vision._publish_events([], now=start + 10.0)
    vision._publish_events(flicker, now=start + 11.0)
    assert [event["greet"] for event in detected_named(bus)] == [True, False]

    # A genuine absence longer than the timeout ends the sitting, so the
    # visitor's next appearance is greeted once with a new sitting id.
    timeout = vision.greeting_absence_s
    vision._publish_events([], now=start + timeout + 20.0)

    returning = vision.recognizer.recognize_all([moved_face(unit(0), 450)])
    vision.tracker.update(returning)
    vision._publish_events(returning, now=start + timeout + 21.0)

    events = detected_named(bus)
    assert events[-1]["greet"] is True
    assert events[-1]["sitting_id"] != sitting_id


def test_presenter_ignores_unknown_and_non_greeting_events(app):
    bus = EventBus()
    presenter = KioskPresenter(event_bus=bus)

    captions = []
    greeting_requests = []

    presenter.caption.connect(lambda text, duration: captions.append(text))
    bus.subscribe(
        "greeting.speak_requested",
        lambda **payload: greeting_requests.append(payload),
    )

    bus.publish("visitor.unknown", face=StubFace(11, "Unknown"))
    app.processEvents()

    bus.publish(
        "visitor.detected",
        name="Saksham",
        face=StubFace(12, "Saksham"),
        greet=False,
        sitting_id=4,
    )
    app.processEvents()

    assert captions == []
    assert greeting_requests == []
    assert presenter.current_state() == "idle"

    # The first greeting=True event for a new sitting is presented and
    # spoken, and duplicate deliveries are ignored.
    for _ in range(3):
        bus.publish(
            "visitor.detected",
            name="Saksham",
            face=StubFace(12, "Saksham"),
            greet=True,
            sitting_id=5,
        )
    app.processEvents()

    assert len(greeting_requests) == 1
    assert greeting_requests[0]["name"] == "Saksham"
    assert "Saksham" in greeting_requests[0]["text"]
    assert presenter.current_state() == "greeting"

    presenter.shutdown()
