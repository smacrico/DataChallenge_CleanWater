"""Smoke tests for feature engineering modules."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.wqsa.utils.config import load_config


class TestFeaturesSmoke:
    """Smoke tests for feature engineering pipeline."""
    
    @pytest.fixture
    def config(self):
        """Load project configuration."""
        return load_config("config/project.yaml")
    
    @pytest.fixture
    def sample_data(self):
        """Create synthetic sample data for testing."""
        np.random.seed(42)
        
        n_samples = 10
        dates = pd.date_range("2023-01-01", periods=n_samples, freq="7D")
        
        data = pd.DataFrame({
            "STATION_KEY": [f"STATION_{i%3:03d}" for i in range(n_samples)],
            "SAMPLE_DATE": dates,
            "ALKALINITY": np.random.rand(n_samples) * 100,
            "EC": np.random.rand(n_samples) * 500,
            "DRP": np.random.rand(n_samples) * 0.5,
        })
        
        return data
    
    @pytest.fixture
    def landsat_data(self):
        """Create synthetic Landsat feature data."""
        np.random.seed(42)
        
        n_scenes = 30
        dates = pd.date_range("2022-12-01", periods=n_scenes, freq="5D")
        
        data = pd.DataFrame({
            "STATION_KEY": [f"STATION_{i%3:03d}" for i in range(n_scenes)],
            "SCENE_DATE": dates,
            "NDVI_MEAN_B250": np.random.rand(n_scenes),
            "NDBI_MEAN_B1K": np.random.rand(n_scenes),
            "NDWI_MEAN_B250": np.random.rand(n_scenes),
            "CLOUD_FRAC_B250": np.random.rand(n_scenes) * 0.5,
        })
        
        return data
    
    def test_landsat_join_logic(self, sample_data, landsat_data, config):
        """Test Landsat join logic with synthetic data."""
        # Simple join test
        result = pd.merge(
            sample_data,
            landsat_data,
            on="STATION_KEY",
            how="left"
        )
        
        assert len(result) > 0
        assert "NDVI_MEAN_B250" in result.columns
        assert "SCENE_DATE" in result.columns
    
    def test_feature_columns_from_config(self, config):
        """Test that all configured features can be identified."""
        landsat_features = config["features"]["landsat"]
        terraclimate_features = config["features"]["terraclimate"]
        
        assert "NDVI_MEAN_B250" in landsat_features
        assert "PPT_M0" in terraclimate_features
        
        all_features = landsat_features + terraclimate_features
        assert len(all_features) > 10  # Should have reasonable number of features
    
    def test_gold_table_structure(self, sample_data, config):
        """Test that gold table would have correct structure."""
        # Simulate gold table creation
        feature_cols = config["features"]["landsat"] + config["features"]["terraclimate"]
        target_cols = config["targets"]
        
        # Would need these columns
        required_cols = ["STATION_KEY", "SAMPLE_DATE"] + feature_cols + target_cols
        
        assert "ALKALINITY" in target_cols
        assert "EC" in target_cols
        assert "DRP" in target_cols
        
        assert len(feature_cols) > 0
