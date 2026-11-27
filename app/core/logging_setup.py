"""
Logging configuration for the AI Admission Robot.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


class LoggerManager:
    """Creates and configures application loggers."""

    def __init__(
        self,
        log_directory: str,
        log_file: str,
        level: str = "INFO",
        max_file_size: int = 1_048_576,
        backup_count: int = 5,
    ):
        self.log_directory = Path(log_directory)
        self.log_file = log_file
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.max_file_size = max_file_size
        self.backup_count = backup_count

    def create_logger(self, name: str) -> logging.Logger:
        """Create or return a configured logger."""

        self.log_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger = logging.getLogger(name)

        if logger.handlers:
            return logger

        logger.setLevel(self.level)

        # Do not duplicate records through the root logger.
        logger.propagate = False

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        file_handler = RotatingFileHandler(
            self.log_directory / self.log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger