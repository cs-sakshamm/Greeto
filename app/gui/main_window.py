"""
Greeto kiosk window.

A fullscreen, borderless reception terminal: a large animated face, one
quiet caption line, and nothing else. The camera feed, visitor records,
logs and diagnostics are never shown.

The window owns the presentation only. Application semantics (which
expression, which caption) come from ``KioskPresenter``; the face comes
from ``RobotWidget``; the backdrop and caption are small dedicated widgets.

Development controls (never drawn on screen):

    Esc    leave kiosk mode; a second Esc closes the application
    F11    toggle fullscreen
    F10    toggle the diagnostics line (hidden by default)
"""

from __future__ import annotations

# Imported before PyQt5 on purpose: onnxruntime cannot initialise
# once the Qt DLLs are loaded (Windows). Importing VisionService
# first keeps the standalone launch of this module working.
# See docs/PROJECT_AUDIT.md.
from app.vision.vision_service import VisionService

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QVBoxLayout,
)

from app.gui.caption_widget import CaptionWidget
from app.gui.kiosk_background import KioskBackground
from app.gui.kiosk_presenter import KioskPresenter
from app.gui.robot_widget import RobotWidget
from app.gui.theme import KIOSK_STYLE
from app.gui.vision_pump import VisionPump


class MainWindow(QMainWindow):
    """Fullscreen kiosk interface."""

    # Layout proportions.
    SIDE_MARGIN = 48
    TOP_MARGIN = 28
    BOTTOM_MARGIN = 34

    def __init__(
        self,
        vision: VisionService,
        container=None,
        speech=None,
        fps: int = 30,
        pump_vision: bool = False,
        parent=None,
    ):

        super().__init__(parent)

        self.vision = vision

        self.container = container

        self.setWindowTitle("Greeto")

        # Kiosk: no title bar, no border, no window controls.
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)

        self.setStyleSheet(KIOSK_STYLE)

        self.setFocusPolicy(Qt.StrongFocus)

        self._kiosk = False

        self.build_ui(fps)

        # -------------------------------------------------
        # Application state integration
        # -------------------------------------------------

        event_bus = self._resolve_event_bus()

        self.presenter = None

        if event_bus is not None:

            self.presenter = KioskPresenter(
                event_bus,
                microphone_available=bool(
                    getattr(speech, "microphone_ok", True)
                ),
                parent=self,
            )

            self.presenter.face_state.connect(self._on_face_state)
            self.presenter.caption.connect(self._on_caption)
            self.presenter.caption_clear.connect(self._on_caption_clear)

        # -------------------------------------------------
        # Camera processing (off the GUI thread, never shown)
        # -------------------------------------------------

        self.vision_pump = None

        if pump_vision and vision is not None:

            self.vision_pump = VisionPump(vision, fps=fps, parent=self)

            self.vision_pump.start()

        self._refresh_diagnostics()

    # -----------------------------------------------------
    # UI
    # -----------------------------------------------------

    def build_ui(self, fps):

        self.background = KioskBackground()

        self.setCentralWidget(self.background)

        root = QVBoxLayout(self.background)

        root.setContentsMargins(
            self.SIDE_MARGIN,
            self.TOP_MARGIN,
            self.SIDE_MARGIN,
            self.BOTTOM_MARGIN,
        )

        root.setSpacing(16)

        # The face is the interface: it takes all remaining height.
        face_row = QHBoxLayout()
        face_row.setContentsMargins(0, 0, 0, 0)
        face_row.setSpacing(0)

        self.robot_widget = RobotWidget(fps=fps)

        self.robot_widget.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        face_row.addWidget(self.robot_widget, 1)

        root.addLayout(face_row, 1)

        # One caption line near the bottom, centred and width-limited.
        self.caption_widget = CaptionWidget()

        self.caption_widget.setMinimumHeight(92)

        caption_row = QHBoxLayout()
        caption_row.setContentsMargins(0, 0, 0, 0)
        caption_row.setSpacing(0)

        caption_row.addStretch(1)
        caption_row.addWidget(self.caption_widget, 8)
        caption_row.addStretch(1)

        root.addLayout(caption_row, 0)

        # Development-only diagnostics, hidden until F10.
        self.diagnostics = QLabel(self.background)

        self.diagnostics.setObjectName("Diagnostics")

        self.diagnostics.setStyleSheet(
            "QLabel#Diagnostics{background:transparent;color:#38E0FF;"
            "font-size:12px;}"
        )

        self.diagnostics.move(18, 12)
        self.diagnostics.hide()

        # Refreshes camera/vision health while the diagnostics line is
        # visible. One permanent timer keeps the widget tree stable.
        self._diag_timer = QTimer(self)

        self._diag_timer.setInterval(1000)

        self._diag_timer.timeout.connect(self._refresh_diagnostics)

        self._diag_timer.start()

        self._layout_caption()

    # -----------------------------------------------------

    def _resolve_event_bus(self):

        if self.container is not None:

            bus = getattr(self.container, "event_bus", None)

            if bus is not None:

                return bus

        container = getattr(self.vision, "container", None)

        return getattr(container, "event_bus", None)

    # -----------------------------------------------------

    def _layout_caption(self):

        height = max(92, int(self.height() * 0.15))

        self.caption_widget.setFixedHeight(height)

        self.caption_widget.setMaximumWidth(
            max(420, int(self.width() * 0.72))
        )

    def resizeEvent(self, event):

        super().resizeEvent(event)

        self._layout_caption()

        if self.diagnostics.isVisible():

            self.diagnostics.move(18, 12)

    # -----------------------------------------------------
    # Kiosk mode
    # -----------------------------------------------------

    def enter_kiosk(self):
        """Enter borderless fullscreen kiosk mode."""

        self._kiosk = True

        self.setCursor(Qt.BlankCursor)

        self.showFullScreen()

        self.setFocus()

    def leave_kiosk(self):
        """Return to a normal window (development use)."""

        self._kiosk = False

        self.unsetCursor()

        self.showNormal()

        self.setFocus()

    def is_kiosk(self):

        return self._kiosk

    # -----------------------------------------------------

    def keyPressEvent(self, event):

        key = event.key()

        if key == Qt.Key_Escape:

            if self.isFullScreen():

                self.leave_kiosk()

            else:

                self.close()

            return

        if key == Qt.Key_F11:

            if self.isFullScreen():

                self.leave_kiosk()

            else:

                self.enter_kiosk()

            return

        if key == Qt.Key_F10:

            self._toggle_diagnostics()

            return

        super().keyPressEvent(event)

    # -----------------------------------------------------
    # Presenter slots
    # -----------------------------------------------------

    def _on_face_state(self, state):

        self.robot_widget.set_state(state)

        self._refresh_diagnostics()

    def _on_caption(self, text, duration_ms):

        self.caption_widget.show_caption(text, duration_ms)

    def _on_caption_clear(self):

        self.caption_widget.clear_caption()

    # -----------------------------------------------------
    # Diagnostics (development only)
    # -----------------------------------------------------

    def _toggle_diagnostics(self):

        if self.diagnostics.isVisible():

            self.diagnostics.hide()

            return

        self._refresh_diagnostics()

        self.diagnostics.show()

        self.diagnostics.raise_()

    def _refresh_diagnostics(self):

        if not self.diagnostics.isVisible():

            return

        microphone = "ok"

        if self.presenter is not None and not self.presenter.microphone_available:

            microphone = "unavailable"

        visitors = "?"

        tracker = getattr(self.vision, "tracker", None)

        if tracker is not None:

            try:

                visitors = str(tracker.active_track_count())

            except Exception:

                visitors = "?"

        vision = "unknown"

        running = getattr(self.vision, "is_running", None)

        if callable(running):

            try:

                vision = "ready" if running() else "degraded"

            except Exception:

                vision = "unknown"

        self.diagnostics.setText(
            "state: %s    microphone: %s    visitors: %s    "
            "vision: %s    %s    %s"
            % (
                self.robot_widget.current_state(),
                microphone,
                visitors,
                vision,
                self._camera_diagnostics(),
                self._recognition_diagnostics(),
            )
        )

    def _recognition_diagnostics(self):

        recognizer = getattr(self.vision, "recognizer", None)

        stats = None

        if recognizer is not None and hasattr(recognizer, "stats"):

            try:

                stats = recognizer.stats()

            except Exception:

                stats = None

        if not stats:

            return "recognition: n/a"

        last = stats.get("last_status") or "idle"

        score = stats.get("last_score")

        if score is not None:

            last = "%s %.2f" % (last, score)

        return "recognition: %s ids / %s emb  threshold %.2f  last %s" % (
            stats.get("known_identities", 0),
            stats.get("known_embeddings", 0),
            stats.get("threshold", 0.0),
            last,
        )

    def _camera_diagnostics(self):

        status = {}

        if self.presenter is not None:

            status = self.presenter.camera_status or {}

        if not status:

            camera = getattr(self.vision, "camera", None)

            if camera is not None and hasattr(camera, "status"):

                try:

                    status = camera.status()

                except Exception:

                    status = {}

        if not status:

            return "camera: n/a"

        state = status.get("state") or "unknown"

        width = status.get("actual_width") or 0

        height = status.get("actual_height") or 0

        fps = status.get("actual_fps") or 0.0

        age = status.get("frame_age_s")

        failures = status.get("consecutive_failures") or 0

        reconnects = status.get("reconnects") or 0

        parts = ["camera: %s" % state]

        if width and height:

            parts.append("%sx%s @ %.1f fps" % (width, height, fps))

        if age is not None:

            parts.append("last frame %.1fs ago" % age)

        if failures:

            parts.append("%d failed reads" % failures)

        if reconnects:

            parts.append("%d reconnects" % reconnects)

        return "    ".join(parts)

    # -----------------------------------------------------
    # Shutdown
    # -----------------------------------------------------

    def closeEvent(self, event):

        try:

            self._diag_timer.stop()

        except Exception:

            pass

        # Stop the vision pump before releasing the camera it reads from.
        try:

            if self.vision_pump is not None:

                self.vision_pump.stop()

        except Exception:

            pass

        try:

            if self.presenter is not None:

                self.presenter.shutdown()

        except Exception:

            pass

        try:

            self.caption_widget.shutdown()

        except Exception:

            pass

        try:

            self.robot_widget.shutdown()

        except Exception:

            pass

        try:

            if self.vision is not None:

                self.vision.stop()

        except Exception:

            pass

        super().closeEvent(event)


# ---------------------------------------------------------
# Standalone Launch
# ---------------------------------------------------------

if __name__ == "__main__":

    import sys

    from app.core.container import ServiceContainer

    app = QApplication(sys.argv)

    container = ServiceContainer()

    vision = VisionService(container)

    # Keep the window usable in voice-only mode when no camera is
    # present; the vision service retries in the background.
    vision.start()

    window = MainWindow(
        vision=vision,
        container=container,
        pump_vision=True,
    )

    window.enter_kiosk()

    sys.exit(
        app.exec_()
    )
