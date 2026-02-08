"""Validation utilities for data and predictions."""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_dataframe_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    df_name: str = "DataFrame",
) -> bool:
    """Validate that DataFrame contains all required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        df_name: Name for error messages

    Returns:
        True if valid

    Raises:
        ValueError: If required columns are missing
    """
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"{df_name} is missing required columns: {', '.join(missing)}"
        )

    logger.debug(f"{df_name} validation passed: all required columns present")
    return True


def validate_no_nulls(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    df_name: str = "DataFrame",
) -> bool:
    """Validate that specified columns have no null values.

    Args:
        df: DataFrame to validate
        columns: List of column names to check (None = all columns)
        df_name: Name for error messages

    Returns:
        True if valid

    Raises:
        ValueError: If null values are found
    """
    if columns is None:
        columns = df.columns.tolist()

    null_counts = df[columns].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]

    if not cols_with_nulls.empty:
        error_msg = f"{df_name} contains null values:\n{cols_with_nulls}"
        raise ValueError(error_msg)

    logger.debug(f"{df_name} validation passed: no null values")
    return True


def validate_value_range(
    series: pd.Series,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    series_name: str = "Series",
) -> bool:
    """Validate that values are within specified range.

    Args:
        series: Pandas Series to validate
        min_val: Minimum allowed value (None = no minimum)
        max_val: Maximum allowed value (None = no maximum)
        series_name: Name for error messages

    Returns:
        True if valid

    Raises:
        ValueError: If values are outside range
    """
    if min_val is not None:
        below_min = series < min_val
        if below_min.any():
            raise ValueError(
                f"{series_name} contains {below_min.sum()} values below minimum {min_val}"
            )

    if max_val is not None:
        above_max = series > max_val
        if above_max.any():
            raise ValueError(
                f"{series_name} contains {above_max.sum()} values above maximum {max_val}"
            )

    logger.debug(f"{series_name} validation passed: values within range")
    return True


def validate_submission_format(
    df: pd.DataFrame,
    required_columns: List[str] = ["ALKALINITY", "EC", "DRP"],
    expected_rows: int = 200,
) -> bool:
    """Validate submission CSV format.

    Args:
        df: Submission DataFrame
        required_columns: Required column names in order
        expected_rows: Expected number of rows

    Returns:
        True if valid

    Raises:
        ValueError: If format is invalid
    """
    # Check columns
    if list(df.columns) != required_columns:
        raise ValueError(
            f"Invalid columns. Expected {required_columns}, got {list(df.columns)}"
        )

    # Check row count
    if len(df) != expected_rows:
        raise ValueError(
            f"Invalid row count. Expected {expected_rows}, got {len(df)}"
        )

    # Check for nulls
    if df.isnull().any().any():
        raise ValueError("Submission contains null values")

    # Check for infinite values
    if np.isinf(df.values).any():
        raise ValueError("Submission contains infinite values")

    logger.info("Submission format validation passed")
    return True
