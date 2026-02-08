"""Test configuration loading and validation."""

import pytest
import yaml

from src.wqsa.utils.config import load_config, validate_config, get_feature_columns


def test_load_config():
    """Test that configuration loads successfully."""
    config = load_config("config/project.yaml")
    assert config is not None
    assert isinstance(config, dict)


def test_config_has_required_sections():
    """Test that config contains all required top-level sections."""
    config = load_config("config/project.yaml")
    
    required_sections = ["features", "targets", "modeling", "snowflake"]
    for section in required_sections:
        assert section in config, f"Missing required section: {section}"


def test_config_features():
    """Test that features are defined correctly."""
    config = load_config("config/project.yaml")
    
    assert "landsat" in config["features"]
    assert "terraclimate" in config["features"]
    
    assert isinstance(config["features"]["landsat"], list)
    assert isinstance(config["features"]["terraclimate"], list)
    
    assert len(config["features"]["landsat"]) > 0
    assert len(config["features"]["terraclimate"]) > 0


def test_config_targets():
    """Test that targets are defined correctly."""
    config = load_config("config/project.yaml")
    
    assert "targets" in config
    targets = config["targets"]
    
    assert len(targets) == 3
    assert "ALKALINITY" in targets
    assert "EC" in targets
    assert "DRP" in targets


def test_config_modeling():
    """Test that modeling configuration is valid."""
    config = load_config("config/project.yaml")
    
    modeling = config["modeling"]
    assert "cv_splits" in modeling
    assert modeling["cv_splits"] == 5
    
    assert "random_state" in modeling
    assert modeling["random_state"] == 42


def test_validate_config():
    """Test configuration validation."""
    config = load_config("config/project.yaml")
    assert validate_config(config) is True


def test_get_feature_columns():
    """Test extracting feature column names."""
    config = load_config("config/project.yaml")
    features = get_feature_columns(config)
    
    assert isinstance(features, list)
    assert len(features) > 0
    
    # Should have both landsat and terraclimate features
    expected_count = (
        len(config["features"]["landsat"]) + 
        len(config["features"]["terraclimate"])
    )
    assert len(features) == expected_count


def test_config_file_not_found():
    """Test that missing config raises error."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.yaml")


def test_invalid_config_missing_features():
    """Test that invalid config fails validation."""
    invalid_config = {
        "targets": ["ALKALINITY", "EC", "DRP"],
        "modeling": {}
    }
    
    with pytest.raises(ValueError, match="features"):
        validate_config(invalid_config)
