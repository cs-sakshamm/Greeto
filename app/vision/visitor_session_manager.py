"""Visitor Session Manager.

Converts low-level tracking information into meaningful visitor lifecycle
events. Maintains independent visitor sessions using track and recognition
information from the existing VisitorTracker and FaceRecognizer.

Do not merge all these responsibilities into one class.
Do not move face-recognition algorithms into this manager.
Do not turn VisitorTracker into a database or session-management component.
"""

from __future__ import annotations

import time
from collections.abc import MutableMapping
from datetime import datetime, timezone

from app.vision.models import Face, RecognitionResult

# ---------------------------------------------------------------------------
# Session state machine
# ---------------------------------------------------------------------------

PRESENT = "present"
TEMPORARILY_MISSING = "temporarily_missing"
DEPARTED = "departed"

# ---------------------------------------------------------------------------
# Session identity
# ---------------------------------------------------------------------------


class VisitorSession(MutableMapping):
    """One visitor session keyed by track ID.

    A session represents one visitor's lifecycle from arrival through
    temporary disappearance to departure.  The session is private to
    this manager; external code should interact through the manager's
    API rather than manipulating individual sessions directly.
    """

    __slots__ = (
        "session_id",
        "track_id",
        "visitor_id",
        "visitor_name",
        "first_seen",
        "last_seen",
        "state",
        "arrival_published",
        "departure_published",
    )

    def __init__(self, session_id: int, track_id: int) -> None:
        self.session_id = session_id
        self.track_id = track_id
        self.visitor_id: int | None = None  # set when a known visitor is associated
        self.visitor_name: str | None = None  # set when a known visitor is associated
        self.first_seen: float = time.time()
        self.last_seen: float = time.time()
        self.state: str = PRESENT
        self.arrival_published: bool = False
        self.departure_published: bool = False

    # -------------------------------------------------
    # MutableMapping interface (needed for compatibility)
    # -------------------------------------------------

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value: object) -> None:
        object.__setattr__(self, key, value)

    def __delitem__(self, key: str) -> None:
        raise KeyError(key)

    def __len__(self) -> int:
        # MutableMapping requires __len__; we report 1 for a single session.
        return 1

    def __iter__(self):
        yield "session_id"
        yield "track_id"
        yield "visitor_id"
        yield "first_seen"
        yield "last_seen"
        yield "state"
        yield "arrival_published"
        yield "departure_published"

    # -------------------------------------------------
    # Public mutators
    # -------------------------------------------------

    def update(self, visitor_id: int | None = None) -> None:
        """Record that the visitor is still present.

        Args:
            visitor_id: Registered visitor ID, or ``None`` for unknown visitors.
        """
        self.visitor_id = visitor_id
        self.last_seen = time.time()

    def mark_missing(self) -> None:
        """Transition toward TEMPORARILY_MISSING state.

        ``last_seen`` is deliberately not touched: the grace period is
        measured from the last *detection*, otherwise a sweep every frame
        would keep postponing the departure forever.
        """

        if self.state == PRESENT:
            self.state = TEMPORARILY_MISSING

    def mark_present(self) -> None:
        """Return from temporary missing to PRESENT state.

        Publishes a ``visitor.returned`` event if the session had
        previously been marked missing.
        """
        if self.state == TEMPORARILY_MISSING:
            self.state = PRESENT
        self.last_seen = time.time()
        self.arrival_published = True  # returned counts as "present for this session"

    def mark_departed(self) -> None:
        """Transition to DEPARTED state and publish departure event."""

        if not self.departure_published:
            self.departure_published = True
        self.state = DEPARTED
        self.last_seen = time.time()

    def is_departed(self) -> bool:
        return self.state == DEPARTED

    def is_present(self) -> bool:
        return self.state == PRESENT

    def is_missing(self) -> bool:
        return self.state == TEMPORARILY_MISSING


# ---------------------------------------------------------------------------
# VisitorSessionManager
# ---------------------------------------------------------------------------


class VisitorSessionManager:
    """Manages visitor lifecycles from tracking through arrival/departure.

    Responsibilities
    ----------------
    * Maintain one ``VisitorSession`` per active track ID.
    * Apply a configurable grace period for temporary disappearance.
    * Publish exactly one ``visitor.arrived`` event per new session.
    * Publish ``visitor.present`` while the visitor remains.
    * Publish ``visitor.temporarily_missing`` when detection is lost.
    * Publish ``visitor.returned`` when a missing visitor reconnects.
    * Publish ``visitor.departed`` when the grace period expires.
    * Associate registered visitor IDs with existing sessions.
    * Support multiple independent visitors simultaneously.
    * Integrate with the existing ``EventBus`` and ``VisitorDatabase``.
    """

    def __init__(
        self,
        event_bus,
        visitor_database=None,
        grace_period_seconds: float | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.visitor_database = visitor_database
        #: Grace period (seconds) after which a temporarily‑missing visitor
        #: is considered departed.  ``None`` means use the project default.
        self.grace_period_seconds = (
            grace_period_seconds
            if grace_period_seconds is not None
            else 5.0  # reasonable default; overriden by config later
        )

        # Map:: track_id -> VisitorSession
        self._sessions: dict[int, VisitorSession] = {}

        # Counter for generating stable session IDs independent of track IDs.
        self._next_session_id = 1

        # Track which visitor IDs have already been announced, to avoid
        # duplicate ``visitor.detected`` / arrival events.
        self._announced_visitor_ids: set[int] = set()

        # Track which track IDs have had an arrival event published,
        # so we emit exactly one arrival per session.
        self._arrival_published_per_track: set[int] = set()

        # Track which (session_id, visitor_id) pairs have already had
        # their visit recorded, so we register the visit once per unique
        # visitor per session while still allowing different tracker IDs for
        # the same person to each register a visit (as the old behavior did).
        self._visited_pairs: set[tuple[int, int | None]] = set()

    # -------------------------------------------------
    # Session lifecycle
    # -------------------------------------------------

    def update_track(
        self,
        track_id: int,
        visitor_id: int | None = None,
        visitor_name: str | None = None,
        *,
        now: float | None = None,
    ) -> VisitorSession | None:
        """Process a tracker update for *track_id*.

        This is the main entry point called from ``VisionService`` after
        each frame.  It updates or creates the session for the given track
        and publishes the appropriate lifecycle events.

        Args:
            track_id: The visitor tracker's ID for this face.
            visitor_id: Registered visitor ID from ``FaceRecognizer``, or
                ``None`` when the face is unknown.
            visitor_name: The visitor's display name, or ``None`` for unknown.
            now: Optional monotonic timestamp (seconds).  Defaults to
                ``time.time()``.

        Returns:
            The :class:`VisitorSession` that was updated, or ``None`` if
            the track is not being followed.
        """
        if now is None:
            now = time.time()

        session = self._sessions.get(track_id)

        # A departed session is terminal. If the same tracker ID comes
        # back (or a periodic sweep still references it), a new sitting
        # starts with a fresh session ID.
        if session is not None and session.state == DEPARTED:
            session = None

        # Ensure a session exists for this track.
        if session is None:
            session = VisitorSession(session_id=self._next_session_id, track_id=track_id)
            session.first_seen = now
            session.last_seen = now
            self._next_session_id += 1
            self._sessions[track_id] = session

            # New session: set visitor info and publish arrival event immediately.
            if visitor_id is not None:
                session.visitor_id = visitor_id
                session.visitor_name = visitor_name

            # New session: publish arrival event immediately.
            self._publish_arrival(session, visitor_id=visitor_id)
            return session

        # -----------------------------------------------------------------
        # Real detection for an existing track
        # -----------------------------------------------------------------
        if visitor_id is not None:
            # Update the visitor ID and name now that we have an identity.
            session.visitor_id = visitor_id
            session.visitor_name = visitor_name
            session.last_seen = now

            if session.state == TEMPORARILY_MISSING:
                # Returned within the grace period: same sitting, same
                # session ID.
                session.mark_present()
            elif session.state == PRESENT and not session.arrival_published:
                self._publish_arrival(session, visitor_id=visitor_id)

            return session

        # -----------------------------------------------------------------
        # No detection in this call (sweep with visitor_id=None)
        # -----------------------------------------------------------------
        if session.state == PRESENT:
            session.mark_missing()
        elif session.state == TEMPORARILY_MISSING:
            if now - session.last_seen >= self.grace_period_seconds:
                self._publish_departure(session)

        return session

    def sweep(
        self,
        active_track_ids: set[int] | list[int] | tuple[int, ...],
        now: float | None = None,
    ) -> None:
        """Advance sessions whose tracks were not seen in the latest frame.

        This is what turns a run of empty frames into a departure after
        the configured grace period, instead of departing on the first
        dropped frame. A block call may also use ``now`` to simulate the
        passage of time in tests.
        """

        if now is None:
            now = time.time()

        active = set(active_track_ids)

        for track_id, session in list(self._sessions.items()):

            if track_id in active:
                continue

            if session.state == PRESENT:
                session.mark_missing()

            elif session.state == TEMPORARILY_MISSING:
                if now - session.last_seen >= self.grace_period_seconds:
                    self._publish_departure(session)

    def mark_track_missing(self, track_id: int, now: float | None = None) -> VisitorSession | None:
        """Call when a tracker reports that a face is no longer detected.

        Transitions the session toward ``TEMPORARILY_MISSING``.  Does not
        immediately publish a departure event — the grace period must
        elapse first.

        Args:
            track_id: The tracker ID to update.
            now: Optional monotonic timestamp.

        Returns:
            The updated ``VisitorSession``, or ``None`` if the track is
            not being followed.
        """
        if now is None:
            now = time.time()

        if track_id not in self._sessions:
            # A face disappeared before we had a session for it; nothing
            # to do.
            return None

        session = self._sessions[track_id]
        if not session.is_departed():
            session.mark_missing()
        return session

    # -------------------------------------------------
    # Identity association
    # -------------------------------------------------

    def associate_visitor(self, track_id: int, visitor_id: int, visitor_name: str | None = None) -> VisitorSession | None:
        """Associate a registered visitor ID with an existing session.

        If the track does not yet have a session, one is created and an
        arrival event is published.

        Args:
            track_id: The tracker ID.
            visitor_id: The registered visitor's database ID.
            visitor_name: The visitor's display name.

        Returns:
            The ``VisitorSession`` that now holds the visitor ID.
        """
        if track_id not in self._sessions:
            session = VisitorSession(session_id=self._next_session_id, track_id=track_id)
            self._next_session_id += 1
            self._sessions[track_id] = session
            session.visitor_id = visitor_id
            session.visitor_name = visitor_name
            self._publish_arrival(session, visitor_id=visitor_id)
            return session

        session = self._sessions[track_id]
        # Do not overwrite an existing visitor ID with a different one;
        # keep the first association.
        if session.visitor_id is None:
            session.visitor_id = visitor_id
            session.visitor_name = visitor_name

        # If the visitor is already associated, make sure arrival was
        # published (it should have been when the session was first created).
        if not session.arrival_published:
            self._publish_arrival(session, visitor_id=visitor_id)

        return session

    # -------------------------------------------------
    # Event publishing
    # -------------------------------------------------

    def _publish_arrival(self, session: VisitorSession, *, visitor_id: int | None = None) -> None:
        """Publish ``visitor.arrived`` for this session if not already published.

        This is intentionally *per-session* (not per-face), so that a person
        who stands in front of the camera for many frames only generates one
        arrival event, even though their tracker ID may be re-examined on every frame.
        """

        if session.arrival_published:
            return

        session.arrival_published = True

        # Record the visit in the visitor database if available.
        # Only register once per unique visitor_id per session.
        # Use the face name (as the old code did via _record_visit(face.name)).
        visit_key = (session.session_id, visitor_id)
        if visit_key not in self._visited_pairs:
            self._visited_pairs.add(visit_key)
            if self.visitor_database is not None and visitor_id is not None and session.visitor_name is not None:
                try:
                    self.visitor_database.register(
                        session.visitor_name
                    )
                except Exception:
                    pass  # do not crash the camera pipeline

        payload: dict = {
            "session_id": session.session_id,
            "track_id": session.track_id,
            "visitor_id": visitor_id,
            "visitor_name": session.visitor_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._safe_publish("visitor.arrived", **payload)

        # Also publish visitor.present while the session stays present.
        self._safe_publish("visitor.present", **payload)

    def _safe_publish(self, event_name: str, **payload) -> None:
        """Publish without letting a broken bus stop the camera pipeline."""

        try:
            self.event_bus.publish(event_name, **payload)
        except Exception:
            pass

    def _publish_departure(self, session: VisitorSession) -> None:
        """Publish exactly one ``visitor.departed`` event per session."""

        if session.departure_published:
            return

        session.departure_published = True
        session.state = DEPARTED

        payload: dict = {
            "session_id": session.session_id,
            "track_id": session.track_id,
            "visitor_id": session.visitor_id,
            "first_seen_at": datetime.fromtimestamp(
                session.first_seen, tz=timezone.utc
            ).isoformat(),
            "last_seen_at": datetime.fromtimestamp(
                session.last_seen, tz=timezone.utc
            ).isoformat(),
            "departed_at": datetime.now(timezone.utc).isoformat(),
            "session_duration": int(session.last_seen - session.first_seen),
        }

        self._safe_publish("visitor.departed", **payload)

    def _publish_present(self, session: VisitorSession) -> None:
        """Publish ``visitor.present`` if not already published."""

        payload: dict = {
            "session_id": session.session_id,
            "track_id": session.track_id,
            "visitor_id": session.visitor_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._safe_publish("visitor.present", **payload)

    def _publish_returned(self, session: VisitorSession) -> None:
        """Publish ``visitor.returned`` when a missing visitor reconnects."""

        if session.arrival_published and session.state == PRESENT:
            payload: dict = {
                "session_id": session.session_id,
                "track_id": session.track_id,
                "visitor_id": session.visitor_id,
                "missing_duration": int(time.time() - session.last_seen),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._safe_publish("visitor.returned", **payload)

    def _publish_missing(self, session: VisitorSession) -> None:
        """Publish ``visitor.temporarily_missing`` once per missing entry."""

        if session.arrival_published:
            payload: dict = {
                "session_id": session.session_id,
                "track_id": session.track_id,
                "visitor_id": session.visitor_id,
                "last_seen_at": datetime.fromtimestamp(
                    session.last_seen, tz=timezone.utc
                ).isoformat(),
                "missing_since": datetime.now(timezone.utc).isoformat(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._safe_publish("visitor.temporarily_missing", **payload)

    # -------------------------------------------------
    # Global state
    # -------------------------------------------------

    def get_session(self, track_id: int) -> VisitorSession | None:
        return self._sessions.get(track_id)

    def get_sessions(self) -> list[VisitorSession]:
        return list(self._sessions.values())

    def end_session(self, track_id: int) -> VisitorSession | None:
        """Force-end a session and remove it from the manager.

        Useful for shutdown or when a track is definitively lost.
        """
        session = self._sessions.pop(track_id, None)
        if session is not None:
            session.mark_departed()
            self._publish_departure(session)
        return session

    def clear(self) -> None:
        """Remove all sessions (e.g. at kiosk restart)."""
        self._sessions.clear()
        self._next_session_id = 1
        self._arrival_published_per_track.clear()
        self._announced_visitor_ids.clear()

    def __getitem__(self, track_id: int) -> VisitorSession | None:
        return self._sessions.get(track_id)

    def __contains__(self, track_id: int) -> bool:
        return track_id in self._sessions

    def __len__(self) -> int:
        return len(self._sessions)

    def __iter__(self):
        return iter(self._sessions.values())

    def __repr__(self) -> str:
        return f"<VisitorSessionManager sessions={len(self)}>"