"""
Kiosk background.

A very dark, restrained backdrop: a soft radial light behind the face and
a vignette toward the edges. It exists so the face reads as the only
source of light on the terminal.

The gradients are rebuilt on resize only; painting allocates nothing.
"""

from __future__ import annotations

from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QBrush, QColor, QPainter, QRadialGradient
from PyQt5.QtWidgets import QWidget


class KioskBackground(QWidget):
    """Dark kiosk backdrop with subtle ambient light."""

    BASE = QColor(5, 8, 13)

    def __init__(self, parent=None):

        super().__init__(parent)

        self.setObjectName("KioskBackground")

        self.setAutoFillBackground(False)

        self._glow = None
        self._vignette = None

    # -----------------------------------------------------

    def _rebuild(self):

        width = max(1, self.width())
        height = max(1, self.height())

        glow = QRadialGradient(
            QPointF(width * 0.5, height * 0.38),
            max(width, height) * 0.80,
        )

        glow.setColorAt(0.0, QColor(17, 36, 51))
        glow.setColorAt(0.45, QColor(9, 18, 27))
        glow.setColorAt(1.0, self.BASE)

        self._glow = glow

        vignette = QRadialGradient(
            QPointF(width * 0.5, height * 0.5),
            max(width, height) * 0.74,
        )

        vignette.setColorAt(0.0, QColor(0, 0, 0, 0))
        vignette.setColorAt(0.62, QColor(0, 0, 0, 55))
        vignette.setColorAt(1.0, QColor(0, 0, 0, 175))

        self._vignette = vignette

    # -----------------------------------------------------

    def resizeEvent(self, event):

        self._rebuild()

        super().resizeEvent(event)

    # -----------------------------------------------------

    def paintEvent(self, event):

        if self._glow is None or self._vignette is None:

            self._rebuild()

        painter = QPainter(self)

        try:

            painter.setRenderHint(QPainter.Antialiasing, False)

            painter.fillRect(self.rect(), self.BASE)

            painter.setPen(Qt.NoPen)

            painter.setBrush(QBrush(self._glow))
            painter.drawRect(self.rect())

            painter.setBrush(QBrush(self._vignette))
            painter.drawRect(self.rect())

        finally:

            painter.end()
