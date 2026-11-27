"""
Greeto's animated digital face widget.

The mascot is drawn entirely with QPainter - eyes, pupils, eyelids,
eyebrows and mouth are separate pieces of geometry driven by a parameter
object. No image, SVG, GIF or video asset is used to build the face.

The public state API is unchanged from the previous implementation, so
MainWindow (and any future caller) keeps working as before.
"""

from __future__ import annotations

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QSizePolicy, QWidget

from app.gui.face_animation import FaceAnimator
from app.gui.face_renderer import FaceRenderer


class RobotWidget(QWidget):
    """
    Animated digital face.

    Rendering and animation are deliberately separated:

        RobotWidget  -> owns the QTimer through FaceAnimator
        FaceAnimator -> produces parameter numbers only
        FaceRenderer -> draws one frame from those numbers
    """

    MINIMUM_SIZE = 220
    PREFERRED_SIZE = 460

    def __init__(self, parent=None, fps=30):
        super().__init__(parent)

        self.setAttribute(Qt.WA_StyledBackground, True)

        self.setStyleSheet("""
        QWidget{
            background:transparent;
        }
        """)

        self.setFocusPolicy(Qt.NoFocus)

        self.setMinimumSize(self.MINIMUM_SIZE, self.MINIMUM_SIZE)

        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        self._renderer = FaceRenderer()

        self._animator = FaceAnimator(fps=fps, parent=self)

        self._animator.frame.connect(self.update)

        self._animator.set_state("idle")

        self._animator.start()

    # -----------------------------------------------------
    # Qt plumbing
    # -----------------------------------------------------

    def sizeHint(self):
        return QSize(self.PREFERRED_SIZE, self.PREFERRED_SIZE)

    def paintEvent(self, event):
        """Draw one frame using the current animation parameters."""

        painter = QPainter(self)

        try:

            self._renderer.paint(
                painter,
                self.rect(),
                self._animator.params,
            )

        finally:

            painter.end()

    def showEvent(self, event):
        """Resume animation when the face becomes visible."""

        self._animator.start()

        super().showEvent(event)

    def hideEvent(self, event):
        """Pause animation when the face is hidden, to save CPU."""

        self._animator.stop()

        super().hideEvent(event)

    def closeEvent(self, event):
        self.shutdown()

        super().closeEvent(event)

    def shutdown(self):
        """Stop the animation timer. Safe to call more than once."""

        self._animator.stop()

    # -----------------------------------------------------
    # State API
    # -----------------------------------------------------

    def set_state(self, state):
        """
        Set the facial expression.

        Unknown state names fall back to ``idle`` instead of raising, and
        selecting the state that is already active is a no-op.
        """

        return self._animator.set_state(state)

    def current_state(self):
        """Return the active state name."""

        return self._animator.state

    def parameters(self):
        """Expose the current animation parameters (read-only by use)."""

        return self._animator.params

    # -----------------------------------------------------

    def set_idle(self):

        return self.set_state("idle")

    # -----------------------------------------------------

    def set_listening(self):

        return self.set_state("listening")

    # -----------------------------------------------------

    def set_thinking(self):

        return self.set_state("thinking")

    # -----------------------------------------------------

    def set_speaking(self):

        return self.set_state("speaking")

    # -----------------------------------------------------

    def set_happy(self):

        return self.set_state("happy")

    # -----------------------------------------------------

    def set_greeting(self):

        return self.set_state("greeting")

    # -----------------------------------------------------

    def set_confused(self):

        return self.set_state("confused")

    # -----------------------------------------------------

    def set_error(self):

        return self.set_state("error")

    # -----------------------------------------------------

    def load_idle(self):
        """Legacy alias kept so older call sites keep working."""

        return self.set_state("idle")
