"""
Visitor Tracker.

Assigns stable, unique visitor IDs across consecutive frames using
simple centroid tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from app.vision.models import Face


@dataclass
class Track:
    """
    Internal tracking state.

    This class is private to the tracker and should never be
    used outside this file.
    """

    visitor_id: int
    center_x: int
    center_y: int
    missing_frames: int = 0


class VisitorTracker:
    """
    Tracks visitors between consecutive frames.

    The tracker only assigns stable visitor IDs.
    It does NOT perform:
        - face recognition
        - face detection
        - greetings
        - event publishing
    """

    def __init__(
        self,
        max_distance: float = 80.0,
        max_missing_frames: int = 30,
    ) -> None:

        self._max_distance = max_distance
        self._max_missing_frames = max_missing_frames

        self._next_visitor_id = 1

        self._tracks: dict[int, Track] = {}

    def update(
        self,
        faces: list[Face],
    ) -> list[Face]:
        """
        Update visitor tracking.

        Every face in one frame gets a distinct ID. A track that was
        already matched by another face is not reused in the same frame,
        even when both faces are close together.

        Args:
            faces:
                Faces detected in the current frame.

        Returns:
            Same list of faces with visitor_id assigned.
        """

        matched_tracks: set[int] = set()

        for face in faces:

            cx, cy = face.center

            visitor_id = self._find_best_track(
                cx,
                cy,
                matched_tracks,
            )

            if visitor_id is None:

                visitor_id = self._create_track(
                    cx,
                    cy,
                )

            else:

                self._update_track(
                    visitor_id,
                    cx,
                    cy,
                )

            face.visitor_id = visitor_id

            matched_tracks.add(visitor_id)

        self._cleanup(matched_tracks)

        return faces

    def _find_best_track(
        self,
        x: int,
        y: int,
        excluded: set[int] | None = None,
    ) -> int | None:
        """
        Find the closest existing track that is not already used.
        """

        excluded = excluded or set()

        best_distance = self._max_distance
        best_track = None

        for visitor_id, track in self._tracks.items():

            if visitor_id in excluded:

                continue

            distance = hypot(
                track.center_x - x,
                track.center_y - y,
            )

            if distance < best_distance:

                best_distance = distance
                best_track = visitor_id

        return best_track

    def _create_track(
        self,
        x: int,
        y: int,
    ) -> int:
        """
        Create a new visitor track.

        IDs increase monotonically for the lifetime of the tracker, so
        an expired ID is never handed to a different person.
        """

        visitor_id = self._next_visitor_id

        self._tracks[visitor_id] = Track(
            visitor_id=visitor_id,
            center_x=x,
            center_y=y,
        )

        self._next_visitor_id += 1

        return visitor_id

    def _update_track(
        self,
        visitor_id: int,
        x: int,
        y: int,
    ) -> None:
        """
        Update an existing track.
        """

        track = self._tracks[visitor_id]

        track.center_x = x
        track.center_y = y
        track.missing_frames = 0

    def _cleanup(
        self,
        matched_tracks: set[int],
    ) -> None:
        """
        Remove tracks that disappeared.
        """

        expired = []

        for visitor_id, track in self._tracks.items():

            if visitor_id in matched_tracks:

                continue

            track.missing_frames += 1

            if track.missing_frames > self._max_missing_frames:

                expired.append(visitor_id)

        for visitor_id in expired:
            del self._tracks[visitor_id]

    def clear(self) -> None:
        """
        Reset the tracker.

        IDs restart at 1 because this is a new tracking session; within
        a session they are never reused.
        """

        self._tracks.clear()
        self._next_visitor_id = 1

    def active_track_count(self) -> int:
        """
        Number of active visitor tracks.
        """

        return len(self._tracks)

    def has_active_tracks(self) -> bool:
        """
        Returns True if at least one visitor is being tracked.
        """

        return bool(self._tracks)

    def get_track_ids(self) -> list[int]:
        """
        Return all active visitor IDs.
        """

        return list(self._tracks.keys())
