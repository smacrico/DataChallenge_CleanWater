"""Smoke tests for modeling pipeline."""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from src.wqsa.utils.config import load_config


class TestModelingSmoke:
    """Smoke tests for model training pipeline."""
    
    @pytest.fixture
    def config(self):
        """Load project configuration."""
        return load_config("config/project.yaml")
    
    @pytest.fixture
    def synthetic_training_data(self, config):
        """Create synthetic training data."""
        np.random.seed(42)
        
        n_samples = 100
        n_features = len(config["features"]["landsat"]) + len(config["features"]["terraclimate"])
        
        # Generate features
        X = np.random.rand(n_samples, n_features)
        
        # Generate targets with some correlation to features
        y_alkalinity = X[:, 0] * 50 + np.random.rand(n_samples) * 10 + 20
        y_ec = X[:, 1] * 300 + np.random.rand(n_samples) * 50 + 100
        y_drp = X[:, 2] * 0.3 + np.random.rand(n_samples) * 0.05
        
        # Station groups for GroupKFold
        groups = np.array([f"STATION_{i%5:03d}" for i in range(n_samples)])
        
        feature_names = config["features"]["landsat"] + config["features"]["terraclimate"]
        
        df = pd.DataFrame(X, columns=feature_names)
        df["STATION_KEY"] = groups
        df["ALKALINITY"] = y_alkalinity
        df["EC"] = y_ec
        df["DRP"] = y_drp
        
        return df
    
    def test_model_creation_and_fit(self, config):
        """Test that model can be created and fitted."""
        # Create simple model
        model = RandomForestRegressor(
            n_estimators=10,
            max_depth=5,
            random_state=42,
            n_jobs=1
        )
        
        # Fit on tiny dataset
        X = np.random.rand(20, 5)
        y = np.random.rand(20)
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == 20
        assert predictions.min() >= 0  # Should be positive for water quality params
    
    def test_groupkfold_split(self, synthetic_training_data, config):
        """Test GroupKFold splitting logic."""
        n_splits = config["modeling"]["cv_splits"]
        gkf = GroupKFold(n_splits=n_splits)
        
        X = synthetic_training_data.drop(columns=["STATION_KEY", "ALKALINITY", "EC", "DRP"])
        y = synthetic_training_data["ALKALINITY"]
        groups = synthetic_training_data["STATION_KEY"]
        
        fold_count = 0
        for train_idx, val_idx in gkf.split(X, y, groups):
            fold_count += 1
            
            # Check that groups don't overlap between train and val
            train_groups = set(groups.iloc[train_idx])
            val_groups = set(groups.iloc[val_idx])
            
            assert len(train_groups & val_groups) == 0  # No overlap
            assert len(train_idx) > 0
            assert len(val_idx) > 0
        
        assert fold_count == n_splits
    
    def test_cross_validation_workflow(self, synthetic_training_data, config):
        """Test complete cross-validation workflow."""
        feature_cols = config["features"]["landsat"] + config["features"]["terraclimate"]
        target = "ALKALINITY"
        
        X = synthetic_training_data[feature_cols]
        y = synthetic_training_data[target]
        groups = synthetic_training_data["STATION_KEY"]
        
        gkf = GroupKFold(n_splits=3)  # Use 3 for speed
        fold_scores = []
        
        for train_idx, val_idx in gkf.split(X, y, groups):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train simple model
            model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
            model.fit(X_train, y_train)
            
            # Predict and score
            y_pred = model.predict(X_val)
            score = r2_score(y_val, y_pred)
            fold_scores.append(score)
        
        mean_score = np.mean(fold_scores)
        
        # With synthetic correlated data, should get > 0.5 R²
        assert mean_score > 0.3  # Relaxed threshold for smoke test
        assert len(fold_scores) == 3
    
    def test_multi_target_prediction(self, synthetic_training_data, config):
        """Test prediction for all three targets."""
        feature_cols = config["features"]["landsat"] + config["features"]["terraclimate"]
        targets = config["targets"]
        
        X = synthetic_training_data[feature_cols]
        
        predictions = {}
        
        for target in targets:
            y = synthetic_training_data[target]
            
            model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
            model.fit(X, y)
            
            preds = model.predict(X)
            predictions[target] = preds
            
            # Basic sanity checks
            assert len(preds) == len(X)
            assert not np.any(np.isnan(preds))
            assert not np.any(np.isinf(preds))
        
        # All targets predicted
        assert len(predictions) == 3
        assert "ALKALINITY" in predictions
        assert "EC" in predictions
        assert "DRP" in predictions
