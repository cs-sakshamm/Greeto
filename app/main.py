"""
Main entry point.

Starts the complete AI Admission Robot application.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Auto-delegate to .venv if running under system Python without dependencies
_ROOT = Path(__file__).resolve().parents[1]
_VENV_PY = _ROOT / ".venv" / "Scripts" / "python.exe"
if not _VENV_PY.exists():
    _VENV_PY = _ROOT / ".venv" / "bin" / "python"

if _VENV_PY.exists():
    try:
        if Path(sys.executable).resolve() != _VENV_PY.resolve():
            try:
                import insightface  # noqa: F401
            except ImportError:
                sys.exit(subprocess.call([str(_VENV_PY), "-m", "app.main"] + sys.argv[1:]))
    except Exception:
        pass

# IMPORTANT - the import order below is deliberate.
#
# onnxruntime (pulled in through insightface) cannot initialise its
# native extension once the PyQt5 DLLs are loaded on Windows:
#
#   ImportError: DLL load failed while importing
#   onnxruntime_pybind11_state: A dynamic link library (DLL)
#   initialization routine failed.
#
# insightface hides that error inside a generic
# "Unable to import dependency onnxruntime." message.
# Vision therefore has to be imported before Qt.
# See docs/PROJECT_AUDIT.md - "onnxruntime / PyQt5 DLL load conflict".
from app.vision.vision_service import VisionService

from app.speech.speech_service import SpeechService
from app.speech.tts_service import TTSService
from app.brain.conversation_manager import ConversationManager

from PyQt5.QtWidgets import QApplication

from app.core.container import ServiceContainer

from app.gui.main_window import MainWindow


def main() -> int:
    """
    Application startup.

    Returns the process exit code.
    """

    # -------------------------------------------------
    # Qt Application
    #
    # Created before any Qt widget so that every widget,
    # timer and SVG renderer shares a single QApplication
    # and no second event loop is ever created.
    # -------------------------------------------------

    app = QApplication(sys.argv)

    # -------------------------------------------------
    # Core Container
    # -------------------------------------------------

    container = ServiceContainer()

    logger = container.logger

    logger.info(
        "===================================="
    )

    logger.info(
        "AI Admission Robot Starting..."
    )

    logger.info(
        "===================================="
    )

    # -------------------------------------------------
    # Initialize Services
    # -------------------------------------------------

    exit_code = 0

    vision = None
    speech = None

    try:

        # ---------------------------------------------
        # Initialize Services
        # ---------------------------------------------

        vision = VisionService(container)

        speech = SpeechService(container)

        # One speech-output service is shared by every component that
        # needs to speak, so two answers can never play over each other.
        tts = TTSService(container)

        # Load the Piper voice in the background so the first greeting or
        # answer does not pay the voice-model load inside its latency.
        tts.warm_up()

        # ConversationManager registers itself with the
        # EventBus during initialization.
        ConversationManager(container, tts_service=tts, vision_service=vision)

        # ---------------------------------------------
        # Start Backend Services
        # ---------------------------------------------

        vision_ready = vision.start()

        if not vision_ready:

            # Reception must keep working without a camera. The vision
            # service retries in the background and publishes
            # camera.status; speech and the local LLM stay available.
            logger.warning(
                "Vision Service is unavailable; continuing in "
                "voice-only degraded mode. Check that a camera is "
                "connected and that config/camera.yaml points to "
                "the right index."
            )

        speech.start()

        logger.info(
            "Backend services started (vision: %s).",
            "ready" if vision_ready else "degraded",
        )

        # ---------------------------------------------
        # Main Window (fullscreen kiosk)
        # ---------------------------------------------

        window = MainWindow(
            vision=vision,
            container=container,
            speech=speech,
            pump_vision=True,
        )

        window.enter_kiosk()

        logger.info(
            "GUI started."
        )

        # ---------------------------------------------
        # Event Loop
        # ---------------------------------------------

        exit_code = app.exec_()

    except KeyboardInterrupt:

        logger.info(
            "Interrupted by user."
        )

        exit_code = 0

    except Exception:

        logger.exception(
            "Fatal error during startup."
        )

        exit_code = 1

    finally:

        logger.info(
            "Stopping services..."
        )

        if speech is not None:

            speech.stop()

        if vision is not None:

            vision.stop()

        logger.info(
            "Shutdown complete."
        )

    return exit_code


if __name__ == "__main__":

    sys.exit(main())
