"""
Standalone preview for Greeto's animated digital face.

Runs the face on its own - no camera, microphone, LLM, Whisper, Piper or
visitor database is touched.

Usage:

    python scripts/face_preview.py
    python scripts/face_preview.py --demo
    python scripts/face_preview.py --size 1024x768 --fps 60

Keys:

    1  idle        5  happy
    2  listening   6  greeting
    3  thinking    7  confused
    4  speaking    8  error

    D  toggle automatic state cycling
    Q  quit  (Esc also quits)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not _VENV_PY.exists():
    _VENV_PY = ROOT / ".venv" / "bin" / "python"

if _VENV_PY.exists():
    try:
        if Path(sys.executable).resolve() != _VENV_PY.resolve():
            try:
                import PyQt5  # noqa: F401
            except ImportError:
                import subprocess
                sys.exit(subprocess.call([str(_VENV_PY), str(Path(__file__).resolve())] + sys.argv[1:]))
    except Exception:
        pass

if str(ROOT) not in sys.path:

    sys.path.insert(0, str(ROOT))

from PyQt5.QtCore import Qt, QTimer  # noqa: E402
from PyQt5.QtGui import QFont  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from app.gui.robot_widget import RobotWidget  # noqa: E402


STATE_KEYS = {
    Qt.Key_1: "idle",
    Qt.Key_2: "listening",
    Qt.Key_3: "thinking",
    Qt.Key_4: "speaking",
    Qt.Key_5: "happy",
    Qt.Key_6: "greeting",
    Qt.Key_7: "confused",
    Qt.Key_8: "error",
}

DEMO_ORDER = [
    "greeting",
    "idle",
    "listening",
    "thinking",
    "speaking",
    "happy",
    "confused",
    "error",
]


class PreviewWindow(QMainWindow):

    def __init__(self, width, height, fps, demo):
        super().__init__()

        self.setWindowTitle("Greeto - face preview")

        self.resize(width, height)

        central = QWidget()

        central.setStyleSheet("background:#0D1117;")

        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        layout.setContentsMargins(0, 0, 0, 12)

        self.face = RobotWidget(fps=fps)

        layout.addWidget(self.face, 1)

        self.caption = QLabel()

        self.caption.setAlignment(Qt.AlignCenter)

        font = QFont("Segoe UI", 12)

        self.caption.setFont(font)

        self.caption.setStyleSheet("color:#4A6076;background:transparent;")

        layout.addWidget(self.caption)

        self.face.set_state("idle")

        self._update_caption()

        self._demo_index = 0

        self._demo = QTimer(self)

        self._demo.setInterval(2600)

        self._demo.timeout.connect(self._advance_demo)

        if demo:

            self._demo.start()

    # -----------------------------------------------------

    def _update_caption(self):

        self.caption.setText(
            "state: %s        keys 1-8 select state, D demo, Q quit"
            % self.face.current_state()
        )

    def _set_state(self, state):

        self.face.set_state(state)

        self._update_caption()

    def _advance_demo(self):

        self._set_state(DEMO_ORDER[self._demo_index % len(DEMO_ORDER)])

        self._demo_index += 1

    # -----------------------------------------------------

    def keyPressEvent(self, event):

        key = event.key()

        if key in (Qt.Key_Q, Qt.Key_Escape):

            self.close()

            return

        if key == Qt.Key_D:

            if self._demo.isActive():

                self._demo.stop()

            else:

                self._demo.start()

            return

        state = STATE_KEYS.get(key)

        if state is not None:

            self._set_state(state)

            return

        super().keyPressEvent(event)


def _parse_size(text):

    try:

        width, height = text.lower().split("x")

        return int(width), int(height)

    except Exception:

        raise argparse.ArgumentTypeError(
            "size must look like 1024x768"
        )


def main(argv=None):

    parser = argparse.ArgumentParser(
        description="Preview Greeto's animated digital face."
    )

    parser.add_argument("--size", type=_parse_size, default=(900, 760))

    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument(
        "--demo",
        action="store_true",
        help="cycle through every expression automatically",
    )

    args = parser.parse_args(argv)

    app = QApplication(sys.argv[:1])

    window = PreviewWindow(args.size[0], args.size[1], args.fps, args.demo)

    window.show()

    return app.exec_()


if __name__ == "__main__":

    sys.exit(main())
