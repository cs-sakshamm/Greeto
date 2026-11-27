"""
Configuration Manager

Loads YAML configuration files and provides a simple interface
for retrieving configuration values.
"""

from pathlib import Path

import yaml


# Repository root, independent of the current working directory.
# app/core/config_manager.py -> app/core -> app -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigManager:
    """Loads and stores configuration from YAML files."""

    def __init__(self, config_directory=None):

        if config_directory is None:

            self.config_directory = PROJECT_ROOT / "config"

        else:

            path = Path(config_directory)

            if not path.is_absolute():
                path = PROJECT_ROOT / path

            self.config_directory = path

        self._configs = {}

    def load(self) -> None:
        """Load every YAML file inside the configuration directory."""

        if not self.config_directory.exists():
            raise FileNotFoundError(
                f"Configuration directory '{self.config_directory}' not found."
            )

        for yaml_file in sorted(self.config_directory.glob("*.yaml")):
            with yaml_file.open("r", encoding="utf-8") as file:
                self._configs[yaml_file.stem] = yaml.safe_load(file)

    def get(self, file_name: str):
        """Return an entire configuration file."""

        return self._configs.get(file_name)

    def get_value(self, file_name: str, *keys, default=None):
        """
        Retrieve a nested value.

        Example:
            get_value("camera", "camera", "index")

        Returns ``default`` when the configuration file is not
        loaded, a key is missing, or the value itself is empty.
        """

        data = self._configs.get(file_name)

        if data is None:
            return default

        for key in keys:

            if not isinstance(data, dict) or key not in data:
                return default

            data = data[key]

        if data is None:
            return default

        return data