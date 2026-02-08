"""Gold table builder for TRAIN_GOLD and VALID_GOLD.

Assembles final feature tables with canonical column sets.
"""

import logging
from typing import Dict, List

from snowflake.snowpark import DataFrame, Session

logger = logging.getLogger(__name__)


def build_gold_table(
    session: Session,
    base_df: DataFrame,
    feature_columns: List[str],
    target_columns: List[str],
    table_name: str,
    mode: str = "overwrite",
) -> None:
    """Build and save a GOLD table with canonical feature set.

    Args:
        session: Active Snowpark session
        base_df: Source DataFrame with all joined features
        feature_columns: List of feature column names to include
        target_columns: List of target column names to include (for training)
        table_name: Name of the GOLD table to create
        mode: Save mode ('overwrite', 'append', 'errorifexists')

    Raises:
        ValueError: If required columns are missing from base_df
    """
    logger.info(f"Building GOLD table: {table_name}")

    # Verify all required columns exist
    available_cols = base_df.columns
    all_required = feature_columns + target_columns

    missing_cols = [col for col in all_required if col not in available_cols]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in base DataFrame: {', '.join(missing_cols)}"
        )

    # Select canonical columns
    gold_df = base_df.select(all_required)

    # Save to Snowflake table
    logger.info(f"Saving {table_name} with {len(all_required)} columns and mode={mode}")
    gold_df.write.mode(mode).save_as_table(table_name)

    row_count = gold_df.count()
    logger.info(f"{table_name} created successfully with {row_count} rows")


def build_train_gold(
    session: Session,
    train_df: DataFrame,
    config: Dict,
    table_name: str = "TRAIN_GOLD",
) -> None:
    """Build TRAIN_GOLD table with features and targets.

    Args:
        session: Active Snowpark session
        train_df: Training DataFrame with all features and targets
        config: Project configuration dict
        table_name: Name for the training GOLD table
    """
    # Extract feature columns from config
    landsat_features = config.get("features", {}).get("landsat", [])
    terraclimate_features = config.get("features", {}).get("terraclimate", [])
    feature_columns = landsat_features + terraclimate_features

    # Extract target columns
    target_columns = config.get("targets", ["ALKALINITY", "EC", "DRP"])

    # Add metadata columns
    metadata_cols = ["STATION_KEY", "SAMPLE_DATE"]
    all_columns = metadata_cols + feature_columns + target_columns

    logger.info(f"Building {table_name} with {len(feature_columns)} features and {len(target_columns)} targets")

    # Select and save
    gold_df = train_df.select(all_columns)
    gold_df.write.mode("overwrite").save_as_table(table_name)

    logger.info(f"{table_name} created successfully")


def build_valid_gold(
    session: Session,
    valid_df: DataFrame,
    config: Dict,
    table_name: str = "VALID_GOLD",
) -> None:
    """Build VALID_GOLD table with features (targets may be empty/null).

    Args:
        session: Active Snowpark session
        valid_df: Validation DataFrame with features
        config: Project configuration dict
        table_name: Name for the validation GOLD table
    """
    # Extract feature columns from config
    landsat_features = config.get("features", {}).get("landsat", [])
    terraclimate_features = config.get("features", {}).get("terraclimate", [])
    feature_columns = landsat_features + terraclimate_features

    # Metadata columns
    metadata_cols = ["STATION_KEY", "SAMPLE_DATE"]
    all_columns = metadata_cols + feature_columns

    logger.info(f"Building {table_name} with {len(feature_columns)} features")

    # Select and save
    gold_df = valid_df.select(all_columns)
    gold_df.write.mode("overwrite").save_as_table(table_name)

    logger.info(f"{table_name} created successfully")
