from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy


class RobotWidget(QLabel):
    """
    Displays the receptionist robot image.

    Later this widget will support:
        • Idle animation
        • Listening animation
        • Thinking animation
        • Speaking animation
    """

    def __init__(self):
        super().__init__()

        self.setAlignment(Qt.AlignCenter)

        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        self.setStyleSheet("""
        QLabel{
            background:transparent;
        }
        """)

        self.robot_path = (
            Path(__file__).resolve().parent.parent.parent
            / "assets"
            / "robot.png"
        )

        self.original_pixmap = None

        self.load_robot()

    def load_robot(self):

        if self.robot_path.exists():

            self.original_pixmap = QPixmap(str(self.robot_path))

        else:

            self.setText("")

            return

        self.update_pixmap()

    def resizeEvent(self, event):

        super().resizeEvent(event)

        self.update_pixmap()

    def update_pixmap(self):

        if self.original_pixmap is None:
            return

        pix = self.original_pixmap.scaled(
            int(self.width() * 0.85),
            int(self.height() * 0.85),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.setPixmap(pix)