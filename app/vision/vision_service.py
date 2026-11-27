"""Vision Service.

Optimized real-time vision pipeline.

Pipeline:

Camera
    ↓
FrameGrabber
    ↓
Vision Processing
    ↓
Detection (periodic)
    ↓
Recognition
    ↓
Tracking
    ↓
Display
    ↓
EventBus
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from app.core.container import ServiceContainer

from app.vision.camera_manager import CameraManager
from app.vision.embedding_model import EmbeddingModel
from app.vision.face_database import FaceDatabase
from app.vision.face_recognizer import FaceRecognizer
from app.vision.frame_grabber import FrameGrabber
from app.vision.models import Face
from app.vision.recognition import RecognitionConfig
from app.vision.visitor_tracker import VisitorTracker
from app.vision.visitor_database import VisitorDatabase
from app.vision.visitor_session_manager import VisitorSessionManager


class VisionService:
    """Optimized vision service."""

    def __init__(
        self,
        container: ServiceContainer,
        camera=None,
        embedding_model=None,
        database=None,
        visitor_database=None,
    ) -> None:

        self.container = container

        self.logger = container.logger

        self.event_bus = getattr(
            container,
            "event_bus",
            None,
        )

        # Injecting the camera/model/database exists for tests; the
        # application always uses the defaults below.
        self.camera = camera if camera is not None else CameraManager(
            container
        )

        self.grabber = FrameGrabber(
            self.camera
        )

        self.embedding_model = (
            embedding_model
            if embedding_model is not None
            else EmbeddingModel()
        )

        self.database = database if database is not None else FaceDatabase(
            self.embedding_model
        )

        recognition_config = RecognitionConfig.from_config(
            getattr(container, "config", None)
        )

        for warning in recognition_config.warnings:

            self.logger.warning(
                "Recognition config: %s",
                warning,
            )

        self.recognizer = FaceRecognizer(
            self.database,
            config=recognition_config,
        )

        # Set up visitor_database first, before creating the session manager,
        # because VisitorSessionManager needs it at construction time.
        if visitor_database is not None:

            self.visitor_database = visitor_database

        else:

            visitor_database_path = self._config_value(
                "visitors",
                "database_path",
                default="data/visitors.db",
            )

            legacy_visitor_path = self._config_value(
                "visitors",
                "legacy_json",
                default="data/visitors.json",
            )

            self.visitor_database = VisitorDatabase(
                database_path=visitor_database_path,
                legacy_path=legacy_visitor_path,
            )

        self.tracker = VisitorTracker()

        self.visitor_session_manager = VisitorSessionManager(
            event_bus=self.event_bus,
            visitor_database=self.visitor_database,
            grace_period_seconds=self._float_config(
                "sessions", "grace_period_seconds", default=5.0
            ),
        )

        # Greeting policy. A sitting is keyed by the canonical recognized
        # identity, not by a tracker id, so a flickering track cannot
        # produce a second greeting for the same person.
        self.greeting_enabled = self._bool_config(
            "greeting", "enabled", default=True
        )
        self.greeting_absence_s = max(
            1.0,
            self._float_config(
                "greeting", "absence_timeout_s", default=20.0
            ),
        )

        configured_min_confidence = self._config_value(
            "greeting", "min_confidence", default=None
        )
        try:
            self.greeting_min_confidence = (
                None
                if configured_min_confidence in (None, "")
                else float(configured_min_confidence)
            )
        except (TypeError, ValueError):
            self.greeting_min_confidence = None

        # identity key -> {id, name, first_seen, last_seen, greeted, tracks}
        self._sittings: dict[str, dict] = {}
        self._sitting_counter = 0
        self._sitting_lock = threading.Lock()

        # Unknown visitor IDs are announced once per track so the kiosk
        # is not asked to greet (it never does) or log them every cycle.
        self._announced_unknown: set[int] = set()

        self._visitor_present = False
        self._last_face_time = time.monotonic()

        self._running = False

        # Set when the application is shutting down. It also stops the
        # camera supervisor and the capture thread's reconnect loop.
        self._stop_event = threading.Event()

        self._supervisor: threading.Thread | None = None

        self._database_loaded = False

        self._camera_retry_interval = 5.0

        try:

            self._camera_retry_interval = float(
                container.config.get_value(
                    "camera",
                    "camera",
                    "open_retry_interval_s",
                    default=5.0,
                )
            )

        except (TypeError, ValueError, AttributeError):

            pass

        self._latest_frame = None

        # Performance control

        self.frame_counter = 0

        self.detection_interval = 3

        self.cached_faces: list[Face] = []

        self.last_time = time.perf_counter()

        self.fps = 0.0

    # -----------------------------------------------------

    def start(self) -> bool:
        """Start the vision pipeline.

        A missing or busy camera must not stop the receptionist from
        working: in that case start() returns False, a supervisor keeps
        retrying in the background, and the kiosk continues in
        voice-only degraded mode.
        """

        if self._running:

            return True

        self.logger.info(
            "Starting Vision Service..."
        )

        self._stop_event.clear()

        if self.camera.open():

            return self._start_pipeline()

        message = (
            getattr(self.camera, "last_error", "")
            or "The camera could not be opened."
        )

        self.logger.warning(
            "%s Starting in voice-only degraded mode; "
            "the camera will be retried every %.1fs.",
            message,
            self._camera_retry_interval,
        )

        self._publish(
            "vision.degraded",
            message=message,
        )

        self._start_camera_supervisor()

        return False

    # -----------------------------------------------------

    def _start_pipeline(self) -> bool:
        """Load the face database and start capturing frames."""

        try:

            self._load_database()

        except Exception as error:

            self.logger.exception(error)

            self.grabber.stop()

            self.camera.release()

            return False

        self.grabber.start()

        self._running = True

        self.logger.info(
            "Vision Service started."
        )

        return True

    def _load_database(self) -> None:

        if self._database_loaded:

            return

        self.database.load()

        try:

            identities, embeddings, rejected = self.recognizer.refresh()

            self.logger.info(
                "Recognition ready: %d identities, %d embeddings "
                "(threshold %.2f, %d rejected).",
                identities,
                embeddings,
                self.recognizer.threshold,
                rejected,
            )

        except Exception:

            # Recognition can still answer "unknown" without a usable
            # database; it must not take the camera pipeline down.
            self.logger.exception(
                "Could not prepare the recognition database."
            )

        self._database_loaded = True

    # -----------------------------------------------------

    def _start_camera_supervisor(self) -> None:
        """Retry opening the camera without blocking the GUI thread."""

        if self._supervisor is not None and self._supervisor.is_alive():

            return

        self._supervisor = threading.Thread(
            target=self._supervise_camera,
            daemon=True,
            name="CameraSupervisor",
        )

        self._supervisor.start()

    def _supervise_camera(self) -> None:

        while not self._stop_event.is_set():

            if self.camera.open():

                if self._start_pipeline():

                    self.logger.info(
                        "Camera recovered; vision pipeline started."
                    )

                    return

            self._stop_event.wait(self._camera_retry_interval)

    # -----------------------------------------------------

    def is_running(self) -> bool:

        return self._running

    # -----------------------------------------------------

    def _config_value(self, *keys, default=None):

        """Read a vision setting without requiring a real ConfigManager."""

        config = getattr(self.container, "config", None)

        if config is None:

            return default

        try:

            value = config.get_value(
                "vision",
                *keys,
                default=default,
            )

        except (AttributeError, TypeError, ValueError):

            return default

        return default if value in (None, "") else value

    def _float_config(self, *keys, default: float) -> float:
        """Read a positive float, falling back to the default."""

        try:
            value = float(self._config_value(*keys, default=default))
        except (TypeError, ValueError):
            return default

        return value if value > 0 else default

    def _bool_config(self, *keys, default: bool) -> bool:
        """Read a boolean setting from YAML without surprising coercion."""

        value = self._config_value(*keys, default=default)

        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "yes", "1", "on"):
                return True
            if lowered in ("false", "no", "0", "off"):
                return False

        return default

    # -----------------------------------------------------

    def health(self) -> dict:
        """Camera and capture diagnostics for the kiosk."""

        camera = None

        try:

            camera = self.camera.status()

        except Exception:

            camera = None

        grabber = None

        try:

            grabber = self.grabber.health()

        except Exception:

            grabber = None

        recognition = None

        try:

            recognition = self.recognizer.stats()

        except Exception:

            recognition = None

        visitors = None

        if self.visitor_database is not None:

            try:

                visitors = self.visitor_database.stats()

            except Exception:

                visitors = None

        return {

            "running": self._running,

            "camera": camera,

            "grabber": grabber,

            "recognition": recognition,

            "visitors": visitors,

        }

    # -----------------------------------------------------

    def _publish(self, event_name: str, **payload) -> None:

        if self.event_bus is None:

            return

        try:

            self.event_bus.publish(event_name, **payload)

        except Exception:

            self.logger.exception(event_name)

    # -----------------------------------------------------

    def stop(self):

        self._stop_event.set()

        self._running = False

        if not self.grabber.stop():

            self.logger.error(
                "Frame capture thread did not stop cleanly; "
                "the camera will only be released if it is safe."
            )

        supervisor = self._supervisor

        if supervisor is not None and supervisor.is_alive():

            # Opening a device can take a couple of seconds on Windows;
            # wait for the supervisor so it cannot open the camera
            # again after shutdown has released it.
            supervisor.join(timeout=5.0)

            if supervisor.is_alive():

                self.logger.warning(
                    "Camera supervisor did not stop within 5s; "
                    "shutting down anyway."
                )

            self._supervisor = None

        self.camera.release()

        self.logger.info(
            "Vision stopped."
        )

    # -----------------------------------------------------

    def process(
        self,
        annotate: bool = True,
    ) -> np.ndarray | None:

        if not self._running:

            return None

        frame = self.grabber.get_frame()

        if frame is None:

            return None

        self.frame_counter += 1

        try:

            # Heavy AI processing only periodically

            if (
                self.frame_counter
                % self.detection_interval
                == 0
            ):

                faces = self.embedding_model.detect(
                    frame
                )

                faces = self.recognizer.recognize_all(
                    faces
                )

                faces = self.tracker.update(
                    faces
                )

                self.cached_faces = faces

                self._publish_events(
                    faces
                )

            else:

                faces = self.cached_faces

            # The kiosk never displays camera frames. Skipping the
            # annotated copy avoids a full-frame copy and two draw
            # passes for every processed frame.
            if not annotate:

                return None

            output = frame.copy()

            self._draw_faces(
                output,
                faces,
            )

            self._draw_fps(
                output
            )

            self._latest_frame = output

            return output

        except Exception as error:

            self.logger.exception(error)

            return frame

    # -----------------------------------------------------

    def _draw_faces(
        self,
        frame,
        faces,
    ):

        for face in faces:

            color = (
                (0, 255, 0)
                if face.recognized
                else (0, 165, 255)
            )

            cv2.rectangle(
                frame,
                (
                    face.x,
                    face.y
                ),
                (
                    face.x + face.width,
                    face.y + face.height
                ),
                color,
                2,
            )

            # Show the similarity when the face was compared; fall back
            # to the detector score for faces rejected by a quality gate.
            score = face.recognition_score

            if score is None:

                score = face.confidence

            label = (
                f"{face.name} "
                f"{score:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (
                    face.x,
                    face.y - 10
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    # -----------------------------------------------------

    def _draw_fps(
        self,
        frame,
    ):

        now = time.perf_counter()

        elapsed = now - self.last_time

        if elapsed > 0:

            self.fps = (
                0.9 * self.fps
                + 0.1 * (1 / elapsed)
            )

        self.last_time = now

        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    # -----------------------------------------------------

    def _publish_events(
        self,
        faces: list[Face],
        now: float | None = None,
    ) -> None:

        if self.event_bus is None:

            return

        now = time.monotonic() if now is None else now

        # ---------------------------------------------------------
        # 1. Session lifecycle for every tracked face
        # ---------------------------------------------------------

        active_track_ids: set[int] = set()

        for face in faces:

            track_id = face.visitor_id

            if track_id is None:

                continue

            active_track_ids.add(track_id)

            session = self.visitor_session_manager.update_track(

                track_id=track_id,
                visitor_id=face.visitor_id,
                visitor_name=face.name,

            )

            if session is not None and session.visitor_id is not None:

                face.visitor_id = session.visitor_id

        # Tracks that were not seen this cycle move toward departure, but
        # only after the configured grace period - a single empty frame
        # must not end a sitting.
        self.visitor_session_manager.sweep(active_track_ids)

        # ---------------------------------------------------------
        # 2. End sittings only after a real absence
        # ---------------------------------------------------------

        with self._sitting_lock:

            expired = [
                key
                for key, sitting in self._sittings.items()
                if now - sitting["last_seen"] >= self.greeting_absence_s
            ]

            for key in expired:

                sitting = self._sittings.pop(key)

                self.logger.info(
                    "Sitting ended for %s after %.1fs unseen; the next "
                    "appearance is a new sitting.",
                    sitting["name"],
                    now - sitting["last_seen"],
                )

        if faces:

            self._last_face_time = now
            self._visitor_present = True

        elif (
            self._visitor_present
            and now - self._last_face_time >= self.greeting_absence_s
        ):

            self._visitor_present = False

            self._publish("visitor.left")

        # ---------------------------------------------------------
        # 3. Announce known identities once per sitting; never greet
        #    an unknown face.
        # ---------------------------------------------------------

        for face in faces:

            if face.visitor_id is None:

                continue

            score = face.recognition_score

            status = face.recognition_status

            identity = self._known_identity(face)

            if identity is not None:

                self._announce_known_visitor(
                    face,
                    identity,
                    score,
                    status,
                    now,
                )

            else:

                self._announce_unknown_visitor(face, score, status)

    # -----------------------------------------------------

    @staticmethod
    def _is_valid_identity_name(name) -> bool:
        """Reject empty and placeholder names from the recognizer."""

        if not isinstance(name, str):

            return False

        cleaned = name.strip()

        if not cleaned:

            return False

        return cleaned.casefold() not in {
            "unknown",
            "none",
            "visitor",
            "null",
            "n/a",
        }

    def _known_identity(self, face: Face) -> str | None:
        """Return the recognized identity, or None when not trustworthy."""

        if not face.recognized:

            return None

        name = face.name

        if not self._is_valid_identity_name(name):

            return None

        score = face.recognition_score

        if score is None:

            return None

        threshold = self.recognizer.threshold

        if self.greeting_min_confidence is not None:

            threshold = max(threshold, self.greeting_min_confidence)

        try:

            score = float(score)

        except (TypeError, ValueError):

            return None

        if score < threshold:

            return None

        return name.strip()

    def _announce_known_visitor(
        self,
        face: Face,
        identity: str,
        score,
        status,
        now: float,
    ) -> None:
        """Publish visitor.detected once per track in a sitting.

        ``greet`` is True only for the first identity detection of the
        sitting. Marking the sitting greeted happens atomically under the
        lock, before any Qt/TTS work is dispatched, so concurrent frames
        cannot produce duplicate greetings.
        """

        key = identity.casefold()

        announce_key = face.visitor_id

        if announce_key is None:

            announce_key = "untracked"

        with self._sitting_lock:

            sitting = self._sittings.get(key)

            new_sitting = sitting is None

            if sitting is None:

                self._sitting_counter += 1

                sitting = {
                    "id": self._sitting_counter,
                    "name": identity,
                    "first_seen": now,
                    "last_seen": now,
                    "greeted": False,
                    "tracks": set(),
                }

                self._sittings[key] = sitting

                self.logger.info(
                    "New sitting #%d for %s.",
                    sitting["id"],
                    identity,
                )

            sitting["last_seen"] = now

            if announce_key in sitting["tracks"]:

                return

            should_greet = self.greeting_enabled and not sitting["greeted"]

            if should_greet:

                sitting["greeted"] = True

            if len(sitting["tracks"]) >= 64:

                sitting["tracks"].clear()

            sitting["tracks"].add(announce_key)

            sitting_id = sitting["id"]

        self.event_bus.publish(

            "visitor.detected",

            name=identity,

            face=face,

            visitor_id=face.visitor_id,

            score=score,

            status=status,

            sitting_id=sitting_id,

            greet=should_greet,

            new_sitting=new_sitting,

        )

    def _announce_unknown_visitor(self, face: Face, score, status) -> None:
        """Publish an informational event for an unknown face, never a greeting."""

        track_id = face.visitor_id

        if track_id is None:

            return

        if track_id in self._announced_unknown:

            return

        self._announced_unknown.add(track_id)

        if len(self._announced_unknown) > 512:

            self._announced_unknown.clear()
            self._announced_unknown.add(track_id)

        self.event_bus.publish(

            "visitor.unknown",

            face=face,

            visitor_id=track_id,

            score=score,

            status=status,

            greet=False,

        )

    def _record_visit(self, name: str) -> None:
        """Store a visit without letting a storage error stop greeting."""

        if self.visitor_database is None:

            return

        try:

            self.visitor_database.register(name)

        except Exception:

            self.logger.exception(
                "Could not record the visit for %s.",
                name,
            )

    # -----------------------------------------------------

    def show(
        self,
        window_name="Vision",
    ) -> bool:
        """Display is handled by the Qt GUI.

        This method is kept only for compatibility.
        """

        return True

    # -----------------------------------------------------

    def run(self):
        """Starts the vision pipeline.

        Frame updates are handled by the Qt GUI
        (CameraWidget + QTimer), so there is no
        processing loop here.
        """

        if not self.start():

            return

        self.logger.info(
            "Vision service running under Qt GUI."
        )