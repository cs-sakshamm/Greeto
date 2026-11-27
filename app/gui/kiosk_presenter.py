"""
Kiosk presenter.

The single place where application events become facial expressions and
captions. It owns no widgets: it only emits Qt signals, so the window
decides how a state or caption is presented.

Thread safety
-------------
EventBus callbacks run on whichever thread published the event (vision
pump, speech thread, conversation thread, TTS thread). Each callback only
re-emits an internal Qt signal; the mapping itself runs in the GUI thread
because that signal is connected with ``Qt.QueuedConnection``. No widget
is ever touched from a worker thread.

Event names used here (the first four already existed):

    visitor.detected        name=, face=      VisionService
    visitor.unknown         face=             VisionService
    visitor.left            -                 VisionService   (added)
    camera.status           state=, ...        CameraManager   (added)
    speech_recognized       text=             SpeechService   (not used here)
    conversation.processing text=             ConversationManager (added)
    llm_response_ready      response=         ConversationManager
    speech.speaking_started text=, utterance= TTSService      (added)
    speech.speaking_finished utterance=       TTSService      (added)
    speech.error            kind=, message=   SpeechService / TTSService (added)
    conversation.error      message=          ConversationManager (added)
"""

from __future__ import annotations

import time

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal


class KioskPresenter(QObject):
    """Maps application events to face states and captions."""

    face_state = pyqtSignal(str)
    caption = pyqtSignal(str, int)
    caption_clear = pyqtSignal()

    # Internal transport from EventBus threads to the GUI thread.
    _dispatch = pyqtSignal(str, object)

    # Timings, in milliseconds. Class attributes so tests can shorten them.
    GREETING_HOLD_MS = 3800
    SUCCESS_HOLD_MS = 1400
    ERROR_RECOVERY_MS = 4200
    CAPTION_CLEAR_MS = 7000
    THINKING_WATCHDOG_MS = 45000
    ERROR_CAPTION_COOLDOWN_S = 6.0
    MAX_GREETED_VISITORS = 512

    EVENTS = (
        "visitor.detected",
        "visitor.unknown",
        "visitor.left",
        "camera.status",
        "vision.degraded",
        "conversation.processing",
        "llm_response_ready",
        "speech.speaking_started",
        "speech.speaking_finished",
        "speech.recording_started",
        "speech.transcription_started",
        "speech.no_speech_detected",
        "speech.error",
        "conversation.error",
        "speech.sentence_started",
        "speech.sentence_finished",
    )

    def __init__(
        self,
        event_bus=None,
        microphone_available=True,
        parent=None,
    ):

        super().__init__(parent)

        self._event_bus = event_bus

        self._handlers = {}

        self._alive = True

        self.microphone_available = bool(microphone_available)

        self.visitor_present = False

        # Camera health is diagnostics only: a missing camera must not
        # change the face or interrupt a conversation.
        self.camera_state = "unknown"

        self.camera_status = {}

        self.vision_degraded = False

        self.busy = False

        self._greeted = set()

        self._current_state = "idle"

        self._speaking_token = None

        self._active_turn_id = None

        self._pending_response = None

        self._last_error = 0.0

        self._sequence = 0

        self._pending = None

        self._dispatch.connect(self._handle, Qt.QueuedConnection)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timer)

        self._watchdog = QTimer(self)
        self._watchdog.setSingleShot(True)
        self._watchdog.timeout.connect(self._on_watchdog)

        if event_bus is not None:

            for name in self.EVENTS:

                handler = self._make_handler(name)

                self._handlers[name] = handler

                event_bus.subscribe(name, handler)

    # -----------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------

    def shutdown(self):
        """Release subscriptions and stop timers. Idempotent."""

        self._alive = False

        if self._event_bus is not None:

            for name, handler in self._handlers.items():

                try:

                    self._event_bus.unsubscribe(name, handler)

                except Exception:

                    pass

        self._handlers.clear()

        self._timer.stop()

        self._watchdog.stop()

    def current_state(self):
        """Active face state name."""

        return self._current_state

    # -----------------------------------------------------
    # EventBus bridge
    # -----------------------------------------------------

    def _make_handler(self, event_name):
        """Build a stable callback that forwards events to the GUI thread."""

        def handler(**payload):

            if self._alive:

                self._dispatch.emit(event_name, payload)

        return handler

    def _handle(self, event_name, payload):
        """Run one event on the GUI thread."""

        if not self._alive:

            return

        if event_name in ("visitor.detected", "visitor.unknown"):

            self._on_visitor(event_name, payload)

        elif event_name == "visitor.left":

            self._on_visitor_left()

        elif event_name == "camera.status":

            self._on_camera_status(payload)

        elif event_name == "vision.degraded":

            self.vision_degraded = True

        elif event_name == "conversation.processing":

            self._on_processing(payload)

        elif event_name == "llm_response_ready":

            self._on_response(payload)

        elif event_name == "speech.speaking_started":

            self._on_speaking_started(payload)

        elif event_name == "speech.speaking_finished":

            self._on_speaking_finished(payload)

        elif event_name == "speech.sentence_started":

            self._on_sentence_started(payload)

        elif event_name == "speech.sentence_finished":

            self._on_sentence_finished(payload)

        elif event_name == "speech.recording_started":

            self._on_recording_started()

        elif event_name == "speech.transcription_started":

            self._on_transcription_started()

        elif event_name == "speech.no_speech_detected":

            self._on_no_speech()

        elif event_name == "speech.error":

            self._on_error(str(payload.get("kind") or "unknown"))

        elif event_name == "conversation.error":

            self._on_error("conversation")

    # -----------------------------------------------------
    # Visitor
    # -----------------------------------------------------

    def _on_visitor(self, event_name, payload):

        self.visitor_present = True

        # Unknown faces never greet. They only keep the kiosk aware that
        # somebody is standing in front of the camera.
        if event_name != "visitor.detected":

            return

        # VisionService is authoritative: it sets greet=False for another
        # track belonging to an identity that was already greeted in this
        # sitting.
        if not payload.get("greet", True):

            return

        # A second visitor must not interrupt an answer in progress.
        if self.busy:

            return

        name = payload.get("name")

        greeting_text = self._greeting_text(name)

        if greeting_text is None:

            # Defence in depth: no recognized identity, no greeting.
            return

        sitting_id = payload.get("sitting_id")

        face = payload.get("face")

        visitor_id = getattr(face, "visitor_id", None)

        if sitting_id is not None:

            key = ("sitting", sitting_id)

        else:

            key = ("track", visitor_id if visitor_id is not None else id(face))

        if key in self._greeted:

            return

        self._greeted.add(key)

        if len(self._greeted) > self.MAX_GREETED_VISITORS:

            self._greeted.clear()
            self._greeted.add(key)

        self._cancel_schedule()

        self._set_state("greeting")

        self._show(greeting_text)

        # Request audible TTS speech for the visitor greeting. The name
        # is forwarded so ConversationManager can reject anything that is
        # not a recognized identity.
        if self._event_bus is not None:
            try:
                self._event_bus.publish(
                    "greeting.speak_requested",
                    text=greeting_text,
                    name=name.strip(),
                    sitting_id=sitting_id,
                )
            except Exception:
                pass

        self._schedule(self.GREETING_HOLD_MS, self._enter_listening_or_idle)

    def _on_visitor_left(self):

        self.visitor_present = False

        # If Greeto is still answering, finish the interaction first.
        if self.busy:

            return

        self._cancel_schedule()

        self._set_state("idle")

        self.caption_clear.emit()

    def _on_camera_status(self, payload):

        self.camera_state = str(payload.get("state") or "unknown")

        self.camera_status = dict(payload)

        if self.camera_state == "ready":

            self.vision_degraded = False

    @staticmethod
    def _greeting_text(name):
        """Return a greeting only for a valid, recognized identity.

        Unknown visitors deliberately get no greeting text at all, so
        there is no fallback path that could speak to an unregistered
        face.
        """

        name = (name or "").strip()

        if name and name.lower() not in (
            "unknown",
            "none",
            "visitor",
            "null",
            "n/a",
        ):

            return "Hello %s! Welcome back to I.T.S Engineering College." % name

        return None

    # -----------------------------------------------------
    # Conversation
    # -----------------------------------------------------

    def _on_processing(self, payload=None):

        if payload and isinstance(payload, dict) and "turn_id" in payload:
            self._active_turn_id = payload.get("turn_id")

        self.visitor_present = True

        self.busy = True

        self._cancel_schedule()

        self._set_state("thinking")

        self._show("Let me check that for you.", self.CAPTION_CLEAR_MS)

        # A failed or stuck request must not leave the face thinking forever.
        if self.THINKING_WATCHDOG_MS > 0:

            self._watchdog.start(int(self.THINKING_WATCHDOG_MS))

    def _on_response(self, payload):

        turn_id = payload.get("turn_id")
        if (
            turn_id is not None
            and self._active_turn_id is not None
            and turn_id < self._active_turn_id
        ):
            return

        self._pending_response = payload.get("response")

        # Do NOT show caption immediately. Captions synchronize with actual
        # audio playback start in _on_speaking_started.

    def _on_speaking_started(self, payload):

        turn_id = payload.get("turn_id")
        if (
            turn_id is not None
            and self._active_turn_id is not None
            and turn_id < self._active_turn_id
        ):
            return

        self.busy = True

        self._watchdog.stop()

        # A spoken greeting supersedes the visual-only greeting hold timer;
        # otherwise that timer could drop the face back to LISTENING while
        # the visitor greeting is still being played.
        self._cancel_schedule()

        self._speaking_token = payload.get("utterance")

        text = (payload.get("text") or "").strip()

        # Face state moves to speaking precisely when audio playback starts
        self._set_state("speaking")

        if text:
            # Display caption only when audio playback actually starts
            self._show(text, self.CAPTION_CLEAR_MS)

    def _on_speaking_finished(self, payload):

        utterance = payload.get("utterance")

        # A finish from an older utterance must not end a newer one.
        if (
            self._speaking_token is not None
            and utterance is not None
            and utterance != self._speaking_token
        ):

            return

        turn_id = payload.get("turn_id")
        if (
            turn_id is not None
            and self._active_turn_id is not None
            and turn_id < self._active_turn_id
        ):

            return

        if not self.busy and self._current_state != "speaking":

            return

        self._speaking_token = None

        self._watchdog.stop()

        self.busy = False

        self._set_state("happy")

        self._schedule(self.SUCCESS_HOLD_MS, self._enter_listening_or_idle)

    def _on_recording_started(self):
        if not self.busy and self.microphone_available:
            self._set_state("listening")

    def _on_transcription_started(self):
        if not self.busy:
            self._set_state("thinking")

    def _on_no_speech(self):
        if not self.busy and self._current_state in ("thinking", "listening"):
            self._enter_listening_or_idle()

    def _on_sentence_started(self, payload):
        """Handle sentence start for incremental TTS."""
        self._watchdog.stop()
        self._set_state("speaking")
        text = (payload.get("text") or "").strip()
        if text:
            self._show(text, self.CAPTION_CLEAR_MS)

    def _on_sentence_finished(self, payload):
        """Handle sentence finish for incremental TTS."""
        utterance = payload.get("utterance")
        # A finish from an older utterance must not end a newer one.
        if (
            self._speaking_token is not None
            and utterance is not None
            and utterance != self._speaking_token
        ):
            return
        # Just update the state to show we're between sentences
        # but don't necessarily go back to idle yet
        self._set_state("speaking")

    # -----------------------------------------------------
    # Errors
    # -----------------------------------------------------

    def _on_error(self, kind):

        self.busy = False

        self._speaking_token = None

        self._pending_response = None

        self._watchdog.stop()

        self._cancel_schedule()

        if kind == "microphone":

            self.microphone_available = False

        self._set_state("error")

        now = time.monotonic()

        # One caption per failure burst, not one per event.
        if now - self._last_error >= self.ERROR_CAPTION_COOLDOWN_S:

            self._last_error = now

            self._show(self._error_text(kind), self.CAPTION_CLEAR_MS)

        self._schedule(self.ERROR_RECOVERY_MS, self._enter_listening_or_idle)

    def _on_watchdog(self):

        if not self._alive:

            return

        self._on_error("timeout")

    @staticmethod
    def _error_text(kind):

        return {
            "microphone": (
                "I can't hear you right now. "
                "Please check the microphone."
            ),
            "stt": (
                "Sorry, I couldn't understand that. "
                "Please try again."
            ),
            "tts": (
                "Sorry, I couldn't speak that response. "
                "Please try again."
            ),
            "conversation": (
                "Sorry, I couldn't process that. "
                "Please try again."
            ),
            "timeout": (
                "Sorry, that took too long. "
                "Please try again."
            ),
        }.get(kind, "Sorry, something went wrong. Please try again.")

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------

    def _set_state(self, state):

        if state == self._current_state:

            return

        self._current_state = state

        self.face_state.emit(state)

    def _show(self, text, duration_ms=None):

        text = (text or "").strip()

        if not text:

            return

        self.caption.emit(
            text,
            int(self.CAPTION_CLEAR_MS if duration_ms is None else duration_ms),
        )

    def _enter_listening_or_idle(self):

        if self.visitor_present and self.microphone_available:

            self._set_state("listening")

            return

        self._set_state("idle")

        if not self.visitor_present:

            self.caption_clear.emit()

    def _schedule(self, delay_ms, action):

        self._sequence += 1

        self._pending = (self._sequence, action)

        self._timer.stop()

        if delay_ms <= 0:

            self._pending = None

            action()

            return

        self._timer.start(int(delay_ms))

    def _cancel_schedule(self):

        self._sequence += 1

        self._pending = None

        self._timer.stop()

    def _on_timer(self):

        if self._pending is None:

            return

        token, action = self._pending

        if token != self._sequence:

            return

        self._pending = None

        action()
