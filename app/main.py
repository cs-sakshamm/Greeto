"""
Main entry point.

Starts the complete AI Admission Robot application.
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app.core.container import ServiceContainer

from app.vision.vision_service import VisionService
from app.speech.speech_service import SpeechService
from app.brain.conversation_manager import ConversationManager

from app.gui.main_window import MainWindow


def main() -> None:
    """
    Application startup.
    """

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

    vision = VisionService(container)

    speech = SpeechService(container)

    brain = ConversationManager(container)

    # Prevent "unused variable" warnings.
    # The ConversationManager registers itself
    # with the EventBus during initialization.
    _ = brain

    # -------------------------------------------------
    # Start Backend Services
    # -------------------------------------------------

    if not vision.start():

        logger.error(
            "Failed to start Vision Service."
        )

        return

    speech.start()

    logger.info(
        "Backend services started."
    )

    # -------------------------------------------------
    # Qt Application
    # -------------------------------------------------

    app = QApplication(sys.argv)

    window = MainWindow(
        vision=vision,
    )

    window.showMaximized()

    logger.info(
        "GUI started."
    )

    # -------------------------------------------------
    # Event Loop
    # -------------------------------------------------

    try:

        exit_code = app.exec()

    except KeyboardInterrupt:

        exit_code = 0

    finally:

        logger.info(
            "Stopping services..."
        )

        speech.stop()

        vision.stop()

        logger.info(
            "Shutdown complete."
        )

    sys.exit(exit_code)


if __name__ == "__main__":

    main()