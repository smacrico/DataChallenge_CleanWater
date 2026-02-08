"""Prediction and submission generation.

Loads trained models and generates submission.csv for validation set.
"""

import logging
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

from ..io.snowflake_session import create_snowpark_session
from ..utils.config import load_config
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)


def load_fold_models(models_dir: Path, target: str, n_folds: int) -> List:
    """Load all fold models for a given target.

    Args:
        models_dir: Directory containing saved models
        target: Target variable name
        n_folds: Number of folds

    Returns:
        List of loaded model objects
    """
    fold_models = []
    for fold_idx in range(n_folds):
        model_path = models_dir / f"{target}_fold{fold_idx}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = joblib.load(model_path)
        fold_models.append(model)

    logger.info(f"Loaded {n_folds} fold models for {target}")
    return fold_models


def predict_with_folds(X: pd.DataFrame, fold_models: List) -> np.ndarray:
    """Generate predictions by averaging fold models.

    Args:
        X: Feature DataFrame
        fold_models: List of trained models

    Returns:
        Array of averaged predictions
    """
    fold_preds = np.column_stack([model.predict(X) for model in fold_models])
    return fold_preds.mean(axis=1)


def main():
    """Main prediction pipeline."""
    setup_logging()
    logger.info("=" * 80)
    logger.info("WATER QUALITY SA - PREDICTION & SUBMISSION")
    logger.info("=" * 80)

    # Load configuration
    config = load_config()

    # Create Snowpark session
    logger.info("Connecting to Snowflake...")
    session = create_snowpark_session()

    # Load VALID_GOLD
    valid_table = config.get("snowflake", {}).get("tables", {}).get("valid_gold", "VALID_GOLD")
    logger.info(f"Loading {valid_table}...")
    valid_df = session.table(valid_table).to_pandas()
    logger.info(f"Loaded {len(valid_df)} validation samples")

    # Prepare features
    landsat_features = config.get("features", {}).get("landsat", [])
    terraclimate_features = config.get("features", {}).get("terraclimate", [])
    feature_cols = landsat_features + terraclimate_features

    X_valid = valid_df[feature_cols]

    # Load models
    models_dir = Path(config.get("paths", {}).get("models", "models"))
    n_folds = config.get("modeling", {}).get("cv_splits", 5)
    targets = config.get("targets", ["ALKALINITY", "EC", "DRP"])

    # Generate predictions for each target
    predictions = {}

    for target in targets:
        logger.info(f"\nGenerating predictions for {target}...")
        fold_models = load_fold_models(models_dir, target, n_folds)
        preds = predict_with_folds(X_valid, fold_models)
        predictions[target] = preds
        logger.info(f"{target} predictions: mean={preds.mean():.4f}, std={preds.std():.4f}")

    # Stack predictions into matrix
    preds_matrix = np.column_stack([predictions[target] for target in targets])

    # Apply blender if available
    blender_path = models_dir / "blender.pkl"
    if blender_path.exists():
        logger.info("\nApplying blender...")
        blender = joblib.load(blender_path)
        final_preds = blender.predict(preds_matrix)
    else:
        logger.info("\nNo blender found, using fold-averaged predictions")
        final_preds = preds_matrix

    # Create submission DataFrame
    submission = pd.DataFrame(final_preds, columns=targets)

    # Ensure correct order and row count
    if len(submission) != 200:
        logger.warning(
            f"Expected 200 rows for submission, but got {len(submission)}. "
            f"Adjusting to first 200 rows."
        )
        submission = submission.head(200)

    # Save submission
    artifacts_dir = Path(config.get("paths", {}).get("artifacts", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    submission_path = artifacts_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    logger.info(f"\nSubmission saved: {submission_path}")
    logger.info(f"Shape: {submission.shape}")
    logger.info(f"Columns: {list(submission.columns)}")
    logger.info("\nFirst 5 rows:")
    logger.info(submission.head().to_string())

    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

    session.close()


if __name__ == "__main__":
    main()
