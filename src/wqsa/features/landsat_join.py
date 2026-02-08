"""Landsat feature join with anti-leakage constraints.

Implements non-future preference logic and fallback to nearest scene within ±60 days.
"""

import logging
from typing import Dict, Optional

from snowflake.snowpark import DataFrame, Session
from snowflake.snowpark import functions as F

logger = logging.getLogger(__name__)


def join_landsat_features(
    session: Session,
    samples_df: DataFrame,
    landsat_df: DataFrame,
    config: Dict,
    station_col: str = "STATION_KEY",
    sample_date_col: str = "SAMPLE_DATE",
    scene_date_col: str = "SCENE_DATE",
) -> DataFrame:
    """Join Landsat features to samples with anti-leakage constraints.

    Implements two-stage join:
    1. Prefer non-future scenes (on or before sample date) within 0-60 days lag
    2. Fallback to nearest scene within ±60 days if no non-future scene found

    Args:
        session: Active Snowpark session
        samples_df: DataFrame with sample data (STATION_KEY, SAMPLE_DATE)
        landsat_df: DataFrame with Landsat features (STATION_KEY, SCENE_DATE, features...)
        config: Configuration dict with landsat settings
        station_col: Station identifier column name
        sample_date_col: Sample date column name
        scene_date_col: Scene date column name

    Returns:
        DataFrame with joined Landsat features and metadata columns:
            - SCENE_GAP_DAYS: days between sample and scene (negative = future)
            - LANDSAT_LOW_COVERAGE_FLAG: 1 if coverage is poor, 0 otherwise
    """
    max_lag_days = config.get("landsat", {}).get("max_lag_days", 60)
    prefer_non_future = config.get("landsat", {}).get("prefer_non_future", True)

    logger.info(
        f"Joining Landsat features: max_lag={max_lag_days}, prefer_non_future={prefer_non_future}"
    )

    # Add aliases for clarity
    samples = samples_df.alias("s")
    landsat = landsat_df.alias("l")

    # Join on station
    joined = samples.join(
        landsat, samples[station_col] == landsat[station_col], join_type="left"
    )

    # Calculate scene gap in days (negative = future scene)
    joined = joined.with_column(
        "SCENE_GAP_DAYS", F.datediff("day", landsat[scene_date_col], samples[sample_date_col])
    )

    # Filter candidates within ±max_lag_days
    candidates = joined.filter(F.abs(F.col("SCENE_GAP_DAYS")) <= max_lag_days)

    if prefer_non_future:
        # Strategy 1: Try non-future scenes first (SCENE_GAP_DAYS >= 0)
        non_future = candidates.filter(F.col("SCENE_GAP_DAYS") >= 0)

        # Get best non-future scene (minimum gap)
        from snowflake.snowpark import Window

        window_spec = Window.partition_by(samples[station_col], samples[sample_date_col]).order_by(
            F.col("SCENE_GAP_DAYS").asc()
        )

        non_future_ranked = non_future.with_column("rn", F.row_number().over(window_spec))
        best_non_future = non_future_ranked.filter(F.col("rn") == 1).drop("rn")

        # Fallback: if no non-future scene, use nearest (min abs gap)
        window_spec_abs = Window.partition_by(
            samples[station_col], samples[sample_date_col]
        ).order_by(F.abs(F.col("SCENE_GAP_DAYS")).asc())

        all_ranked = candidates.with_column("rn", F.row_number().over(window_spec_abs))
        best_any = all_ranked.filter(F.col("rn") == 1).drop("rn")

        # Union strategy: use non-future if available, else fallback
        # NOTE: In Snowpark, we simulate this with conditional logic
        result = best_non_future.union_all(
            best_any.join(
                best_non_future.select(samples[station_col], samples[sample_date_col]),
                on=[station_col, sample_date_col],
                join_type="left_anti",
            )
        )
    else:
        # Strategy 2: Just use nearest scene (min abs gap)
        from snowflake.snowpark import Window

        window_spec = Window.partition_by(samples[station_col], samples[sample_date_col]).order_by(
            F.abs(F.col("SCENE_GAP_DAYS")).asc()
        )

        ranked = candidates.with_column("rn", F.row_number().over(window_spec))
        result = ranked.filter(F.col("rn") == 1).drop("rn")

    # Add low coverage flag (example: if CLOUD_FRAC_B250 > 0.3)
    result = result.with_column(
        "LANDSAT_LOW_COVERAGE_FLAG",
        F.when(F.col("CLOUD_FRAC_B250") > 0.3, F.lit(1)).otherwise(F.lit(0)),
    )

    logger.info("Landsat features joined successfully")
    return result
