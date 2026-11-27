"""
Caption widget.

One quiet line of conversation near the bottom of the kiosk. It fades in
and out, wraps longer answers, and protects newer text from older timers.

The widget owns its auto-clear timer. Every caption update takes a new
token; a fade-out only blanks the label if its token is still the current
one, so a delayed clear can never erase a newer message.
"""

from __future__ import annotations

from PyQt5.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QGraphicsOpacityEffect, QLabel, QSizePolicy


class CaptionWidget(QLabel):
    """Minimal, fading caption line."""

    FADE_IN_MS = 320
    FADE_OUT_MS = 520

    # Long LLM answers stay readable without taking over the screen.
    MAX_CHARS = 260

    def __init__(self, parent=None):

        super().__init__(parent)

        self.setObjectName("Caption")

        self.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        self.setWordWrap(True)

        # Never interpret LLM text as markup.
        self.setTextFormat(Qt.PlainText)

        self.setTextInteractionFlags(Qt.NoTextInteraction)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.setStyleSheet(
            "QLabel#Caption{background:transparent;color:#C7DAE6;}"
        )

        self.setText("")

        self._effect = QGraphicsOpacityEffect(self)
        self._effect.setOpacity(0.0)

        self.setGraphicsEffect(self._effect)

        self._animation = QPropertyAnimation(self._effect, b"opacity", self)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._expire)

        # A caption update increments the token; an older fade-out that
        # finishes late checks it before blanking the label.
        self._token = 0

        self._apply_font()

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def show_caption(self, text, duration_ms=0):
        """
        Display ``text``. ``duration_ms`` 0 keeps it until replaced.

        Returns True when something was displayed.
        """

        text = " ".join(str(text or "").split())

        if not text:

            self.clear_caption()

            return False

        if len(text) > self.MAX_CHARS:

            text = text[: self.MAX_CHARS - 1].rstrip() + "\u2026"

        self._token += 1

        self._timer.stop()

        self.setText(text)

        self._fade_to(1.0, self.FADE_IN_MS)

        if duration_ms and duration_ms > 0:

            self._timer.start(int(duration_ms))

        return True

    def clear_caption(self):
        """Fade the caption out and empty it."""

        self._token += 1

        self._timer.stop()

        self._fade_to(0.0, self.FADE_OUT_MS, clear=True)

    def current_text(self):
        """Current caption text (empty when nothing is shown)."""

        return self.text()

    def opacity(self):
        """Current opacity, for tests."""

        return self._effect.opacity()

    def shutdown(self):
        """Stop timers and animations. Safe to call repeatedly."""

        self._timer.stop()

        self._animation.stop()

    # -----------------------------------------------------
    # Internals
    # -----------------------------------------------------

    def _apply_font(self):

        height = self.height() if self.height() > 0 else 120

        size = max(16, min(34, int(height * 0.22)))

        font = QFont("Segoe UI")
        font.setPixelSize(size)
        font.setWeight(QFont.Light)

        self.setFont(font)

    def _fade_to(self, target, duration_ms, clear=False):

        token = self._token

        self._animation.stop()

        try:

            self._animation.finished.disconnect()

        except TypeError:

            pass

        self._animation.setDuration(max(1, int(duration_ms)))
        self._animation.setStartValue(float(self._effect.opacity()))
        self._animation.setEndValue(float(target))

        if clear:

            self._animation.finished.connect(
                lambda: self._finish_clear(token)
            )

        self._animation.start()

    def _finish_clear(self, token):

        if token == self._token:

            self.setText("")

    def _expire(self):
        """Auto-clear timer fired for the current caption."""

        self.clear_caption()

    def resizeEvent(self, event):

        self._apply_font()

        super().resizeEvent(event)
