from __future__ import annotations

import cv2

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel

from app.vision.vision_service import VisionService


class CameraWidget(QLabel):
    """
    Displays frames produced by VisionService.
    """

    def __init__(self, vision_service: VisionService):
        super().__init__()

        self.vision = vision_service

        self.setAlignment(Qt.AlignCenter)

        self.setStyleSheet("""
        QLabel{
            background:#0B1220;
            border-radius:22px;
        }
        """)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)      # ~30 FPS

    def update_frame(self):

        frame = self.vision.process()

        if frame is None:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape

        image = QImage(
            rgb.data,
            w,
            h,
            ch * w,
            QImage.Format_RGB888,
        )

        pixmap = QPixmap.fromImage(image)

        pixmap = pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation,
        )

        self.setPixmap(pixmap)