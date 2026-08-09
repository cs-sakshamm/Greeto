from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.gui.camera_widget import CameraWidget
from app.gui.robot_widget import RobotWidget
from app.gui.theme import STYLE

from app.vision.vision_service import VisionService


# ---------------------------------------------------------
# Glass Card
# ---------------------------------------------------------

class GlassCard(QFrame):
    """
    Rounded glass-like container.
    """

    def __init__(self):

        super().__init__()

        self.setObjectName("GlassCard")

        self.setStyleSheet("""
        QFrame#GlassCard{
            background:#151E2F;
            border:none;
            border-radius:28px;
        }
        """)

        shadow = QGraphicsDropShadowEffect(self)

        shadow.setBlurRadius(45)

        shadow.setOffset(0, 12)

        shadow.setColor(QColor(0, 0, 0, 120))

        self.setGraphicsEffect(shadow)


# ---------------------------------------------------------
# Main Window
# ---------------------------------------------------------

class MainWindow(QMainWindow):

    def __init__(
        self,
        vision: VisionService,
    ):

        super().__init__()

        self.vision = vision

        self.setWindowTitle("AI Admission Robot")

        self.resize(1700, 900)

        self.setMinimumSize(1400, 800)

        self.setStyleSheet(STYLE)

        self.build_ui()


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------

    def build_ui(self):

        central = QWidget()

        self.setCentralWidget(central)

        root = QHBoxLayout(central)

        root.setContentsMargins(30, 30, 30, 30)

        root.setSpacing(30)

        #
        # LEFT SIDE
        #

        self.camera_card = GlassCard()

        self.camera_card.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        camera_layout = QVBoxLayout(self.camera_card)

        camera_layout.setContentsMargins(
            15,
            15,
            15,
            15,
        )

        self.camera_widget = CameraWidget(
            self.vision,
        )

        self.camera_widget.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        camera_layout.addWidget(
            self.camera_widget
        )

        #
        # RIGHT SIDE
        #

        self.robot_card = GlassCard()

        self.robot_card.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        robot_layout = QVBoxLayout(
            self.robot_card
        )

        robot_layout.setContentsMargins(
            25,
            25,
            25,
            25,
        )

        robot_layout.addStretch()
        self.robot_widget = RobotWidget()

        self.robot_widget.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        robot_layout.addWidget(
            self.robot_widget,
            alignment=Qt.AlignCenter,
        )

        robot_layout.addStretch()

        #
        # Add both panels
        #

        root.addWidget(self.camera_card, 3)
        root.addWidget(self.robot_card, 2)
        self.camera_card.setMaximumWidth(1100)
        self.robot_card.setMinimumWidth(420)

        #
        # Remove focus rectangle
        #

        self.setFocusPolicy(
            Qt.NoFocus,
        )

        central.setFocusPolicy(
            Qt.NoFocus,
        )

        self.camera_widget.setFocusPolicy(
            Qt.NoFocus,
        )

        self.robot_widget.setFocusPolicy(
            Qt.NoFocus,
        )


# ---------------------------------------------------------
# Window Events
# ---------------------------------------------------------

    def closeEvent(self, event):

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

    if not vision.start():

        raise RuntimeError(
            "Unable to start VisionService."
        )

    window = MainWindow(
        vision=vision,
    )

    window.showMaximized()

    sys.exit(
        app.exec()
    )