"""
Dependency Injection Container.

Creates and stores shared services for the application.
"""

from app.core.config_manager import ConfigManager
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

        self.logger_manager = LoggerManager(
            log_directory=self.config.get_value(
                "logging",
                "logging",
                "log_directory",
            ),
            log_file=self.config.get_value(
                "logging",
                "logging",
                "log_file",
            ),
            level=self.config.get_value(
                "logging",
                "logging",
                "level",
            ),
            max_file_size=self.config.get_value(
                "logging",
                "logging",
                "max_file_size",
            ),
            backup_count=self.config.get_value(
                "logging",
                "logging",
                "backup_count",
            ),
        )

        self.logger = self.logger_manager.create_logger(
            "System"
        )

        self.event_bus = EventBus()

        self.state_manager = StateManager()