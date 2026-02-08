"""Configuration management utilities."""

import logging
from pathlib import Path
from typing import Dict

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/project.yaml") -> Dict:
    """Load project configuration from YAML file.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.debug(f"Loading configuration from: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration loaded successfully from {config_path}")
    return config


def validate_config(config: Dict) -> bool:
    """Validate configuration structure and required fields.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, raises ValueError otherwise

    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_top_level = ["features", "targets", "modeling"]

    for key in required_top_level:
        if key not in config:
            raise ValueError(f"Missing required configuration section: {key}")

    # Validate features section
    if "landsat" not in config.get("features", {}):
        raise ValueError("Missing 'features.landsat' configuration")

    if "terraclimate" not in config.get("features", {}):
        raise ValueError("Missing 'features.terraclimate' configuration")

    # Validate targets
    targets = config.get("targets", [])
    if not targets or len(targets) != 3:
        raise ValueError("Expected exactly 3 targets in configuration")

    logger.info("Configuration validation passed")
    return True


def get_feature_columns(config: Dict) -> list:
    """Extract all feature column names from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of feature column names
    """
    landsat = config.get("features", {}).get("landsat", [])
    terraclimate = config.get("features", {}).get("terraclimate", [])

    return landsat + terraclimate
