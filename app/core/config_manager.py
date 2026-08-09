"""
Configuration Manager

Loads YAML configuration files and provides a simple interface
for retrieving configuration values.
"""

from pathlib import Path

import yaml


class ConfigManager:
    """Loads and stores configuration from YAML files."""

    def __init__(self, config_directory: str = "config"):
        self.config_directory = Path(config_directory)
        self._configs = {}

    def load(self) -> None:
        """Load every YAML file inside the configuration directory."""

        if not self.config_directory.exists():
            raise FileNotFoundError(
                f"Configuration directory '{self.config_directory}' not found."
            )

        for yaml_file in self.config_directory.glob("*.yaml"):
            with yaml_file.open("r", encoding="utf-8") as file:
                self._configs[yaml_file.stem] = yaml.safe_load(file)

    def get(self, file_name: str):
        """Return an entire configuration file."""

        return self._configs.get(file_name)

    def get_value(self, file_name: str, *keys):
        """
        Retrieve a nested value.

        Example:
            get_value("camera", "camera", "index")
        """

        data = self._configs.get(file_name)

        if data is None:
            raise KeyError(f"Configuration '{file_name}' not loaded.")

        for key in keys:
            data = data[key]

        return data