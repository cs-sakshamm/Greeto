"""
Dependency Injection Container.

Creates and stores shared services for the application.
"""

from pathlib import Path

from app.core.config_manager import ConfigManager, PROJECT_ROOT
from app.core.event_bus import EventBus
from app.core.logging_setup import LoggerManager
from app.core.state_manager import StateManager


class ServiceContainer:
    """
    Stores shared application services.
    """

    def __init__(self) -> None:

        self.config = ConfigManager()

        self.config.load()

        log_directory = Path(
            self.config.get_value(
                "logging",
                "logging",
                "log_directory",
                default="logs",
            )
        )

        # Logs always live inside the project, never relative to
        # whichever directory the application was started from.
        if not log_directory.is_absolute():

            log_directory = PROJECT_ROOT / log_directory

        self.logger_manager = LoggerManager(
            log_directory=log_directory,
            log_file=self.config.get_value(
                "logging",
                "logging",
                "log_file",
                default="system.log",
            ),
            level=self.config.get_value(
                "logging",
                "logging",
                "level",
                default="INFO",
            ),
            max_file_size=self.config.get_value(
                "logging",
                "logging",
                "max_file_size",
                default=1048576,
            ),
            backup_count=self.config.get_value(
                "logging",
                "logging",
                "backup_count",
                default=5,
            ),
        )

        self.logger = self.logger_manager.create_logger(
            "System"
        )

        self.event_bus = EventBus()

        self.state_manager = StateManager()