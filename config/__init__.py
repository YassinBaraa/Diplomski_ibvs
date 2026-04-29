"""Configuration management for IBVS pipeline."""
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages IBVS pipeline configuration from YAML file and environment overrides."""

    def __init__(self, config_path: str = None):
        """Initialize config manager.

        Args:
            config_path: Path to YAML config file. If None, uses default_config.yaml.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "default_config.yaml"
        else:
            config_path = Path(config_path)

        self.config_path = config_path
        self.config = self._load_config()
        self._apply_env_overrides()
        logger.info(f"Configuration loaded from {self.config_path}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config if config else {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")

    def _apply_env_overrides(self):
        """Apply environment variable overrides to config."""
        # Source overrides
        if "IBVS_SOURCE_TYPE" in os.environ:
            self.config["source"]["type"] = os.environ["IBVS_SOURCE_TYPE"]
        if "IBVS_VIDEO_PATH" in os.environ:
            self.config["source"]["video_path"] = os.environ["IBVS_VIDEO_PATH"]

        # Feature extraction overrides
        if "IBVS_MAX_FEATURES" in os.environ:
            try:
                self.config["feature_extraction"]["max_features"] = int(
                    os.environ["IBVS_MAX_FEATURES"]
                )
            except ValueError:
                logger.warning("Invalid IBVS_MAX_FEATURES, using default")

        # Visualization override
        if "IBVS_VISUALIZATION_ENABLED" in os.environ:
            enabled = os.environ["IBVS_VISUALIZATION_ENABLED"].lower() in ("true", "1", "yes")
            self.config["visualization"]["enabled"] = enabled

    def get(self, key_path: str, default=None):
        """Get config value using dot notation.

        Args:
            key_path: Dot-separated path, e.g., "feature_extraction.max_features"
            default: Default value if key not found

        Returns:
            Config value or default
        """
        keys = key_path.split(".")
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Config key not found: {key_path}")

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire config section.

        Args:
            section: Section name, e.g., "feature_extraction"

        Returns:
            Dictionary of config values
        """
        return self.config.get(section, {})

    def __repr__(self) -> str:
        return f"ConfigManager(path={self.config_path})"
