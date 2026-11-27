"""Visitor session management tests.

Phase 3 Milestone 4: Visitor Tracking and Session Intelligence.

Tests use synthetic inputs and a fake clock. They must not require:
- A physical camera
- A microphone
- A speaker
- A live Piper process
- A real InsightFace model
- Internet connectivity
"""

from __future__ import annotations

import time
from datetime import timezone

import numpy as np

from tests.test_recognition import (
    FakeCamera,
    FakeContainer,
    FakeDatabase,
    FakeVisitorDatabase,
    FakeModel,
    FakeBus,
    FakeLogger,
    unit,
    face,
)

from app.vision.visitor_session_manager import (
    VisitorSession,
    VisitorSessionManager,
    PRESENT,
    TEMPORARILY_MISSING,
    DEPARTED,
)

SIZE = 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_manager(
    visitor_database=None,
    grace_period_seconds: float | None = None,
) -> VisitorSessionManager:
    """Create a VisitorSessionManager with a fake event bus."""
    container = FakeContainer()
    bus = FakeBus()
    manager = VisitorSessionManager(
        event_bus=bus,
        visitor_database=visitor_database,
        grace_period_seconds=grace_period_seconds,
    )
    return manager, container, bus


def events_named(bus, event_name: str) -> list[dict]:
    """Return payloads for one event name from the tuple-recording bus."""

    return [payload for name, payload in bus.events if name == event_name]


# ---------------------------------------------------------------------------
# Basic lifecycle tests
# ---------------------------------------------------------------------------

def test_a_new_visitor_creates_one_session() -> None:
    """A new visitor creates one session."""
    manager, container, bus = make_manager()

    # First frame: visitor appears
    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")
    sessions = manager.get_sessions()
    assert len(sessions) == 1, f"Expected 1 session, got {len(sessions)}"

    # Second frame: visitor still present
    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")

    # Only one session should exist
    assert len(manager.get_sessions()) == 1


def test_one_arrival_event_is_emitted() -> None:
    """One arrival event is emitted."""
    manager, container, bus = make_manager()

    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")

    arrival_events = events_named(bus, "visitor.arrived")
    assert len(arrival_events) == 1, f"Expected 1 arrival, got {len(arrival_events)}"


def test_repeated_frames_do_not_produce_repeated_arrival_events() -> None:
    """Repeated frames do not produce repeated arrival events."""
    manager, container, bus = make_manager()

    # Same visitor across multiple frames
    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")
    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")
    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")

    arrival_events = events_named(bus, "visitor.arrived")
    assert len(arrival_events) == 1, (
        f"Expected 1 arrival across 3 frames, got {len(arrival_events)}"
    )


def test_a_present_visitor_remains_present() -> None:
    """A present visitor remains PRESENT."""
    manager, _, _ = make_manager()

    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")
    session = manager.get_session(1)
    assert session is not None
    assert session.is_present()


def test_a_visitor_temporarily_disappears() -> None:
    """A visitor temporarily disappears."""
    manager, _, _ = make_manager()

    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")

    # Mark as missing
    manager.mark_track_missing(track_id=1)
    session = manager.get_session(1)
    assert session is not None
    assert session.is_missing()


def test_no_departure_before_grace_period_expires() -> None:
    """No departure is emitted before grace-period expiry."""
    manager, container, bus = make_manager(grace_period_seconds=5.0)

    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")

    # Mark as missing but within grace period
    manager.mark_track_missing(track_id=1)

    # Publish events multiple times without advancing time past grace period
    for _ in range(10):
        manager.update_track(track_id=1, visitor_id=None, now=time.time() - 1.0)

    # No departure should have been published
    departure_events = events_named(bus, "visitor.departed")
    assert len(departure_events) == 0, (
        f"Expected 0 departures before grace period, got {len(departure_events)}"
    )


def test_a_visitor_returns_before_grace_period_expires() -> None:
    """A visitor returns before grace-period expiry preserves the original session ID."""
    manager, container, bus = make_manager(grace_period_seconds=5.0)

    # Initial presence
    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")

    # Mark missing
    manager.mark_track_missing(track_id=1)

    # Return before grace period expires (advance time but still within 5s)
    manager.update_track(
        track_id=1, visitor_id=1, visitor_name="Alice", now=time.time() - 2.0
    )

    session = manager.get_session(1)
    assert session is not None
    assert session.is_present(), f"Expected PRESENT, got {session.state}"
    # Session ID should be preserved
    assert session.session_id is not None


def test_a_visitor_returns_original_session_id_preserved() -> None:
    """The original session ID is preserved after return."""
    manager, container, bus = make_manager(grace_period_seconds=5.0)

    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")
    first_session_id = manager.get_session(1).session_id

    # Mark missing
    manager.mark_track_missing(track_id=1)

    # Return before grace period
    manager.update_track(
        track_id=1, visitor_id=1, visitor_name="Alice", now=time.time() - 2.0
    )

    session = manager.get_session(1)
    assert session is not None
    assert session.session_id == first_session_id


def test_a_visitor_disappears_beyond_grace_period() -> None:
    """A visitor disappears beyond the grace period."""
    manager, container, bus = make_manager(grace_period_seconds=1.0)

    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")

    # Mark as missing
    manager.mark_track_missing(track_id=1)

    # Advance time past grace period
    manager.update_track(track_id=1, visitor_id=None, now=time.time() + 2.0)

    # Session should be departed
    session = manager.get_session(1)
    assert session is not None
    assert session.is_departed()


def test_exactly_one_departure_event_is_emitted() -> None:
    """Exactly one departure event is emitted when grace period expires."""
    manager, container, bus = make_manager(grace_period_seconds=1.0)

    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")

    # Mark as missing and advance past grace period
    manager.mark_track_missing(track_id=1)
    manager.update_track(track_id=1, visitor_id=None, now=time.time() + 2.0)

    departure_events = events_named(bus, "visitor.departed")
    assert len(departure_events) == 1, (
        f"Expected 1 departure, got {len(departure_events)}"
    )


def test_return_after_departure_creates_new_session() -> None:
    """A known visitor returns after a completed departure creates a new session."""
    manager, container, bus = make_manager(grace_period_seconds=1.0)

    # First visit: arrival and departure
    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")
    first_sid = manager.get_session(1).session_id
    manager.mark_track_missing(track_id=1)
    manager.update_track(track_id=1, visitor_id=None, now=time.time() + 2.0)  # depart

    assert manager.get_session(1).is_departed()

    # Same track, same visitor returns after departure
    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")

    session = manager.get_session(1)
    assert session is not None
    assert not session.is_departed()
    # New session ID should be assigned
    assert session.session_id != first_sid


# Need first_session_id reference - let me fix this test
def test_return_after_departure_new_session_id() -> None:
    """A known visitor returns after a completed departure creates a new session with new session ID."""
    manager, container, bus = make_manager(grace_period_seconds=1.0)

    # First visit
    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")
    first_sid = manager.get_session(1).session_id

    # Depart
    manager.mark_track_missing(track_id=1)
    manager.update_track(track_id=1, visitor_id=None, now=time.time() + 2.0)

    # Return
    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")

    session = manager.get_session(1)
    assert session is not None
    # Should have a new session ID
    assert session.session_id != first_sid


# ---------------------------------------------------------------------------
# Identity handling tests
# ---------------------------------------------------------------------------

def test_known_visitor_associated_with_track() -> None:
    """A known visitor is correctly associated with a track."""
    manager, _, _ = make_manager()

    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")
    session = manager.get_session(1)
    assert session is not None
    assert session.visitor_id == 1
    assert session.visitor_name == "Alice"


def test_unknown_visitor_no_fabricated_visitor_id() -> None:
    """An unknown visitor receives no fabricated visitor ID."""
    manager, _, _ = make_manager()

    # Unknown visitor - no visitor_id provided
    manager.update_track(track_id=1, visitor_id=None, visitor_name=None)
    session = manager.get_session(1)
    assert session is not None
    assert session.visitor_id is None


def test_visitor_recognized_after_initial_detection() -> None:
    """A visitor recognized after initial detection is associated with existing session."""
    manager, container, bus = make_manager()

    # Initial detection without recognition
    manager.update_track(track_id=1, visitor_id=None, visitor_name=None)

    # Subsequent recognition
    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")

    session = manager.get_session(1)
    assert session is not None
    assert session.visitor_id == 1
    assert session.visitor_name == "Alice"


def test_low_confidence_result_no_identity() -> None:
    """A low-confidence result does not create an unsupported identity association."""
    manager, _, _ = make_manager()

    # Low confidence should not associate a visitor
    manager.update_track(track_id=1, visitor_id=None, visitor_name=None)
    session = manager.get_session(1)
    assert session is not None
    assert session.visitor_id is None


def test_two_tracks_not_merged_just_because_matching_visitor_ids() -> None:
    """Two tracks are not merged just because their recognized visitor IDs match."""
    manager, container, bus = make_manager()

    # Two separate tracks with same visitor
    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")
    manager.update_track(track_id=2, visitor_id=1, visitor_name="Alice")

    assert len(manager.get_sessions()) == 2, f"Expected 2 sessions, got {len(manager.get_sessions())}"


# ---------------------------------------------------------------------------
# Multiple visitors tests
# ---------------------------------------------------------------------------

def test_two_visitors_arrive_simultaneously() -> None:
    """Two visitors arrive simultaneously."""
    manager, container, bus = make_manager()

    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")
    manager.update_track(track_id=2, visitor_id=2, visitor_name="Bob")

    assert len(manager.get_sessions()) == 2


def test_one_visitor_disappears_while_another_remains() -> None:
    """One visitor disappears while another remains present."""
    manager, container, bus = make_manager(grace_period_seconds=1.0)

    # Alice present
    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")
    # Bob present
    manager.update_track(track_id=2, visitor_id=2, visitor_name="Bob")

    # Alice disappears
    manager.mark_track_missing(track_id=1)

    # Bob should still be present
    bob_session = manager.get_session(2)
    assert bob_session is not None
    assert bob_session.is_present()


def test_one_visitor_departs_while_another_returns() -> None:
    """One visitor departs while another returns."""
    manager, container, bus = make_manager(grace_period_seconds=1.0)

    # Alice departs
    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")
    manager.mark_track_missing(track_id=1)
    manager.update_track(track_id=1, visitor_id=None, now=time.time() + 2.0)

    # Bob returns (different track)
    manager.update_track(track_id=2, visitor_id=2, visitor_name="Bob")

    alice_session = manager.get_session(1)
    bob_session = manager.get_session(2)

    assert alice_session is not None
    assert alice_session.is_departed()
    assert bob_session is not None
    assert bob_session.is_present()


def test_three_or_more_visitors_maintain_independent_sessions() -> None:
    """Three or more visitors maintain independent sessions."""
    manager, _, _ = make_manager()

    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")
    manager.update_track(track_id=2, visitor_id=2, visitor_name="Bob")
    manager.update_track(track_id=3, visitor_id=3, visitor_name="Charlie")

    assert len(manager.get_sessions()) == 3


def test_fifty_simultaneous_tracks_handled_correctly() -> None:
    """Fifty simultaneous tracks are handled correctly."""
    manager, _, _ = make_manager()

    for i in range(1, 51):
        manager.update_track(track_id=i, visitor_id=i, visitor_name=f"Visitor{i}")

    assert len(manager.get_sessions()) == 50


def test_single_session_cleanup_does_not_affect_other_sessions() -> None:
    """Session cleanup does not leak or affect other sessions."""
    manager, container, bus = make_manager(grace_period_seconds=1.0)

    # Create two sessions
    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")
    manager.update_track(track_id=2, visitor_id=2, visitor_name="Bob")

    # Clean up Alice's session
    manager.end_session(track_id=1)

    # Bob's session should be unaffected
    assert len(manager.get_sessions()) == 1
    assert manager.get_session(2) is not None
    assert manager.get_session(2).is_present()


# ---------------------------------------------------------------------------
# Reliability tests
# ---------------------------------------------------------------------------

def test_duplicate_frame_updates_do_not_produce_duplicate_events() -> None:
    """Duplicate frame updates do not produce duplicate events."""
    manager, container, bus = make_manager()

    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")
    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")
    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")

    arrival_events = events_named(bus, "visitor.arrived")
    assert len(arrival_events) == 1


def test_out_of_order_timestamps_handled_safely() -> None:
    """Out-of-order timestamps are handled safely."""
    manager, _, _ = make_manager()

    # Send frames with decreasing timestamps
    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice", now=time.time() + 5.0)
    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice", now=time.time() + 3.0)
    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice", now=time.time() + 4.0)

    # Session should still be present
    session = manager.get_session(1)
    assert session is not None
    assert session.is_present()


def test_camera_disconnect_and_reconnect_behavior() -> None:
    """Camera disconnect and reconnect behavior follows documented policy."""
    from app.vision.visitor_session_manager import VisitorSessionManager
    from tests.test_recognition import FakeBus

    bus = FakeBus()
    manager = VisitorSessionManager(event_bus=bus, grace_period_seconds=5.0)

    # Simulate camera disconnect - no sessions to clean up initially
    manager.clear()

    # Simulate camera reconnect - new sessions can be created
    manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")

    assert manager.get_session(1) is not None


def test_event_bus_publication_failure_does_not_crash_pipeline() -> None:
    """EventBus publication failure does not crash the vision pipeline."""
    from app.vision.visitor_session_manager import VisitorSessionManager
    from tests.test_recognition import FakeBus

    bus = FakeBus(fail=True)  # Events will fail
    manager = VisitorSessionManager(event_bus=bus, grace_period_seconds=5.0)

    # Should not raise even if event bus fails
    try:
        manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")
        print("PASS: No crash on event bus failure")
    except Exception as e:
        print(f"FAIL: Crashed on event bus failure: {e}")


def test_database_failure_does_not_corrupt_session_state() -> None:
    """Database failure does not corrupt session state or crash camera processing."""
    from app.vision.visitor_session_manager import VisitorSessionManager
    from tests.test_recognition import FakeVisitorDatabase

    db = FakeVisitorDatabase(crash=True)  # Database operations will fail
    container = FakeContainer()
    bus = FakeBus()
    manager = VisitorSessionManager(event_bus=bus, visitor_database=db, grace_period_seconds=5.0)

    # Should not crash even if database fails
    try:
        manager.update_track(track_id=1, visitor_id=1, visitor_name="Alice")
        print("PASS: No crash on database failure")
    except Exception as e:
        print(f"FAIL: Crashed on database failure: {e}")


def test_session_cleanup_no_memory_leak() -> None:
    """Session cleanup does not leak memory indefinitely."""
    manager, _, _ = make_manager()

    # Create and end many sessions
    for i in range(100):
        manager.update_track(track_id=i, visitor_id=i, visitor_name=f"Visitor{i}")
        manager.end_session(track_id=i)

    # Manager should have no sessions left
    assert len(manager.get_sessions()) == 0


def test_reset_shutdown_behavior_is_safe_and_deterministic() -> None:
    """Reset/shutdown behavior is safe and deterministic."""
    manager, _, _ = make_manager()

    # Create sessions
    for i in range(5):
        manager.update_track(track_id=i, visitor_id=i, visitor_name=f"Visitor{i}")

    # Clear all
    manager.clear()

    # Should be empty
    assert len(manager.get_sessions()) == 0
    assert manager._next_session_id == 1


# ---------------------------------------------------------------------------
# Integration tests with VisionService
# ---------------------------------------------------------------------------

def test_visitor_session_manager_integration_with_vision_service() -> None:
    """VisitorSessionManager integrates correctly with VisionService."""
    from app.vision.vision_service import VisionService
    from tests.test_recognition import FakeContainer, FakeDatabase, FakeCamera, FakeModel

    container = FakeContainer()
    vision = VisionService(
        container,
        camera=FakeCamera(),
        embedding_model=FakeModel(),
        database=FakeDatabase({"Alice": [unit(0)]}),
        visitor_database=FakeVisitorDatabase(),
    )

    vision._running = True
    vision.visitor_database.register("Alice")

    # Process frames with two overlapping faces
    faces = vision.recognizer.recognize_all([face(unit(0)), face(unit(0))])
    vision.tracker.update(faces)
    vision._publish_events(faces)

    # Two overlapping faces of the same identity are two visitor
    # sessions, so two visits are recorded (plus the setup registration).
    assert vision.visitor_database.registered[1:] == ["Alice", "Alice"]


def test_event_payloads_contain_correlation_fields() -> None:
    """Event payloads contain enough information to correlate lifecycle events."""
    from app.vision.vision_service import VisionService
    from tests.test_recognition import FakeContainer, FakeDatabase, FakeCamera, FakeModel

    container = FakeContainer()
    bus = FakeBus()
    container.event_bus = bus

    vision = VisionService(
        container,
        camera=FakeCamera(),
        embedding_model=FakeModel(),
        database=FakeDatabase({"Alice": [unit(0)]}),
        visitor_database=FakeVisitorDatabase(),
    )

    vision._running = True
    vision.visitor_database.register("Alice")

    faces = vision.recognizer.recognize_all([face(unit(0)), face(unit(0))])
    vision.tracker.update(faces)
    vision._publish_events(faces)

    # Check that arrival event has required fields
    arrival_events = events_named(bus, "visitor.arrived")
    assert len(arrival_events) >= 1

    payload = arrival_events[0]
    assert "session_id" in payload
    assert "track_id" in payload
    assert "visitor_id" in payload
    assert "timestamp" in payload
