"""Model training with GroupKFold cross-validation.

Trains per-target regressors with spatial generalization (leave-location-out).
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from ..io.snowflake_session import create_snowpark_session
from ..utils.config import load_config
from ..utils.logging import setup_logging

# Try XGBoost, fallback to RandomForest
try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import RandomForestRegressor

    HAS_XGBOOST = False

logger = logging.getLogger(__name__)


def create_model(config: Dict, target_name: str):
    """Create a regressor based on configuration and availability.

    Args:
        config: Project configuration dict
        target_name: Name of the target variable

    Returns:
        Configured regressor instance
    """
    random_state = config.get("modeling", {}).get("random_state", 42)

    if HAS_XGBOOST and "xgboost" in config.get("modeling", {}).get("model_priority", ["xgboost"]):
        params = config.get("modeling", {}).get("xgboost", {})
        logger.info(f"Creating XGBRegressor for {target_name}")
        return XGBRegressor(**params)
    else:
        params = config.get("modeling", {}).get("random_forest", {})
        logger.info(f"Creating RandomForestRegressor for {target_name} (XGBoost not available)")
        return RandomForestRegressor(**params)


def train_cv_models(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    config: Dict,
    target_name: str,
) -> Tuple[List, np.ndarray, float]:
    """Train models using GroupKFold cross-validation.

    Args:
        X: Feature DataFrame
        y: Target series
        groups: Group identifiers for GroupKFold (e.g., STATION_KEY)
        config: Project configuration
        target_name: Name of the target variable

    Returns:
        Tuple of (fold_models, oof_predictions, cv_r2_score)
    """
    n_splits = config.get("modeling", {}).get("cv_splits", 5)
    gkf = GroupKFold(n_splits=n_splits)

    fold_models = []
    oof_predictions = np.zeros(len(X))
    fold_scores = []

    logger.info(f"Training {target_name} with {n_splits}-fold GroupKFold CV")

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train model
        model = create_model(config, target_name)
        model.fit(X_train, y_train)

        # Predict on validation fold
        y_pred = model.predict(X_val)
        oof_predictions[val_idx] = y_pred

        # Calculate fold R²
        fold_r2 = r2_score(y_val, y_pred)
        fold_scores.append(fold_r2)

        logger.info(f"  Fold {fold_idx}/{n_splits} - R²: {fold_r2:.4f}")

        fold_models.append(model)

    # Overall CV R²
    cv_r2 = r2_score(y, oof_predictions)
    logger.info(f"{target_name} CV R²: {cv_r2:.4f} (mean fold R²: {np.mean(fold_scores):.4f})")

    return fold_models, oof_predictions, cv_r2


def train_blender(
    oof_preds: Dict[str, np.ndarray],
    targets_df: pd.DataFrame,
    config: Dict,
) -> Ridge:
    """Train optional Ridge blender on out-of-fold predictions.

    Args:
        oof_preds: Dict mapping target names to OOF prediction arrays
        targets_df: DataFrame with true target values
        config: Project configuration

    Returns:
        Trained Ridge blender (or None if disabled)
    """
    if not config.get("modeling", {}).get("blender", {}).get("enabled", True):
        logger.info("Blender disabled in config")
        return None

    logger.info("Training Ridge blender on out-of-fold predictions")

    # Stack OOF predictions as features
    X_blend = np.column_stack([oof_preds[target] for target in targets_df.columns])
    y_blend = targets_df.values

    alpha = config.get("modeling", {}).get("blender", {}).get("alpha", 1.0)
    blender = Ridge(alpha=alpha, random_state=config.get("modeling", {}).get("random_state", 42))

    blender.fit(X_blend, y_blend)

    # Evaluate blender
    y_blend_pred = blender.predict(X_blend)
    blend_r2 = np.mean([
        r2_score(y_blend[:, i], y_blend_pred[:, i])
        for i in range(y_blend.shape[1])
    ])

    logger.info(f"Blender average R²: {blend_r2:.4f}")
    return blender


def main():
    """Main training pipeline."""
    setup_logging()
    logger.info("=" * 80)
    logger.info("WATER QUALITY SA - MODEL TRAINING")
    logger.info("=" * 80)

    # Load configuration
    config = load_config()

    # Create Snowpark session
    logger.info("Connecting to Snowflake...")
    session = create_snowpark_session()

    # Load TRAIN_GOLD
    train_table = config.get("snowflake", {}).get("tables", {}).get("train_gold", "TRAIN_GOLD")
    logger.info(f"Loading {train_table}...")
    train_df = session.table(train_table).to_pandas()
    logger.info(f"Loaded {len(train_df)} training samples")

    # Prepare features and targets
    landsat_features = config.get("features", {}).get("landsat", [])
    terraclimate_features = config.get("features", {}).get("terraclimate", [])
    feature_cols = landsat_features + terraclimate_features

    targets = config.get("targets", ["ALKALINITY", "EC", "DRP"])
    groups = train_df["STATION_KEY"]

    X = train_df[feature_cols]
    y_dict = {target: train_df[target] for target in targets}

    # Train per-target models
    all_fold_models = {}
    oof_predictions = {}
    cv_scores = {}

    for target in targets:
        logger.info(f"\nTraining models for target: {target}")
        fold_models, oof_preds, cv_r2 = train_cv_models(
            X, y_dict[target], groups, config, target
        )

        all_fold_models[target] = fold_models
        oof_predictions[target] = oof_preds
        cv_scores[target] = cv_r2

    # Calculate mean R² across targets
    mean_cv_r2 = np.mean(list(cv_scores.values()))
    logger.info(f"\nMean CV R² across targets: {mean_cv_r2:.4f}")

    # Train blender (optional)
    targets_df = train_df[targets]
    blender = train_blender(oof_predictions, targets_df, config)

    # Save models
    models_dir = Path(config.get("paths", {}).get("models", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)

    for target, fold_models in all_fold_models.items():
        for fold_idx, model in enumerate(fold_models):
            model_path = models_dir / f"{target}_fold{fold_idx}.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved: {model_path}")

    if blender is not None:
        blender_path = models_dir / "blender.pkl"
        joblib.dump(blender, blender_path)
        logger.info(f"Saved: {blender_path}")

    # Save CV scores metadata
    metadata = {
        "cv_scores": cv_scores,
        "mean_cv_r2": mean_cv_r2,
        "n_folds": config.get("modeling", {}).get("cv_splits", 5),
        "feature_count": len(feature_cols),
    }

    metadata_path = models_dir / "cv_metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f)
    logger.info(f"Saved: {metadata_path}")

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

    session.close()


if __name__ == "__main__":
    main()
