"""
Data loading and preprocessing functions for water quality datasets.
"""
import os
from typing import Tuple, Optional, List
import pandas as pd
import numpy as np
from pathlib import Path


def load_water_quality_data(
    train_path: str = "../data/raw/train.csv",
    test_path: str = "../data/raw/test.csv",
    submission_template_path: str = "../data/raw/submission_template.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load water quality training and test datasets.

    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV
        submission_template_path: Path to submission template

    Returns:
        Tuple of (train_df, test_df, submission_template)
    """
    print("Loading water quality datasets...")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    submission_template = pd.read_csv(submission_template_path)

    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Submission template shape: {submission_template.shape}")

    return train_df, test_df, submission_template


def load_landsat_data(
    train_landsat_path: str = "../data/raw/train_landsat.csv",
    test_landsat_path: str = "../data/raw/test_landsat.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Landsat satellite imagery data.

    Args:
        train_landsat_path: Path to training Landsat data
        test_landsat_path: Path to test Landsat data

    Returns:
        Tuple of (train_landsat, test_landsat)
    """
    print("Loading Landsat datasets...")

    train_landsat = pd.read_csv(train_landsat_path)
    test_landsat = pd.read_csv(test_landsat_path)

    print(f"Training Landsat shape: {train_landsat.shape}")
    print(f"Test Landsat shape: {test_landsat.shape}")

    return train_landsat, test_landsat


def load_terraclimate_data(
    train_climate_path: str = "../data/raw/train_terraclimate.csv",
    test_climate_path: str = "../data/raw/test_terraclimate.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load TerraClimate weather and climate data.

    Args:
        train_climate_path: Path to training TerraClimate data
        test_climate_path: Path to test TerraClimate data

    Returns:
        Tuple of (train_climate, test_climate)
    """
    print("Loading TerraClimate datasets...")

    train_climate = pd.read_csv(train_climate_path)
    test_climate = pd.read_csv(test_climate_path)

    print(f"Training TerraClimate shape: {train_climate.shape}")
    print(f"Test TerraClimate shape: {test_climate.shape}")

    return train_climate, test_climate


def merge_all_datasets(
    water_df: pd.DataFrame,
    landsat_df: pd.DataFrame,
    climate_df: pd.DataFrame,
    merge_key: str = 'uid'
) -> pd.DataFrame:
    """
    Merge water quality, Landsat, and TerraClimate datasets.

    Args:
        water_df: Water quality DataFrame
        landsat_df: Landsat DataFrame
        climate_df: TerraClimate DataFrame
        merge_key: Column name to merge on

    Returns:
        Merged DataFrame
    """
    print(f"Merging datasets on '{merge_key}'...")

    # Merge water quality with Landsat
    merged = water_df.merge(landsat_df, on=merge_key, how='left')
    print(f"After Landsat merge: {merged.shape}")

    # Merge with TerraClimate
    merged = merged.merge(climate_df, on=merge_key, how='left')
    print(f"After TerraClimate merge: {merged.shape}")

    # Report missing values
    missing_pct = (merged.isnull().sum() / len(merged) * 100).round(2)
    if missing_pct.max() > 0:
        print("\nMissing values after merge:")
        print(missing_pct[missing_pct > 0].sort_values(ascending=False))

    return merged


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'median',
    fill_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Args:
        df: Input DataFrame
        strategy: Strategy for filling missing values ('median', 'mean', 'forward', 'value')
        fill_value: Value to use when strategy='value'

    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if strategy == 'median':
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
    elif strategy == 'mean':
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mean(), inplace=True)
    elif strategy == 'forward':
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
    elif strategy == 'value' and fill_value is not None:
        df.fillna(fill_value, inplace=True)

    print(f"Missing values handled using '{strategy}' strategy")
    return df


def remove_outliers(
    df: pd.DataFrame,
    columns: List[str],
    n_std: float = 3.0
) -> pd.DataFrame:
    """
    Remove outliers using standard deviation method.

    Args:
        df: Input DataFrame
        columns: Columns to check for outliers
        n_std: Number of standard deviations for outlier threshold

    Returns:
        DataFrame with outliers removed
    """
    df_clean = df.copy()
    initial_len = len(df_clean)

    for col in columns:
        if col in df_clean.columns and df_clean[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            lower_bound = mean - n_std * std
            upper_bound = mean + n_std * std

            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    removed = initial_len - len(df_clean)
    print(f"Removed {removed} outliers ({removed/initial_len*100:.2f}%)")

    return df_clean


def split_features_target(
    df: pd.DataFrame,
    target_col: str = 'target',
    drop_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target.

    Args:
        df: Input DataFrame
        target_col: Name of target column
        drop_cols: Additional columns to drop from features

    Returns:
        Tuple of (X, y)
    """
    if drop_cols is None:
        drop_cols = []

    # Columns to exclude from features
    exclude_cols = [target_col] + drop_cols

    # Remove columns that don't exist
    exclude_cols = [col for col in exclude_cols if col in df.columns]

    X = df.drop(columns=exclude_cols)
    y = df[target_col] if target_col in df.columns else None

    print(f"Features shape: {X.shape}")
    if y is not None:
        print(f"Target shape: {y.shape}")

    return X, y


def save_processed_data(
    df: pd.DataFrame,
    output_path: str,
    format: str = 'parquet'
) -> None:
    """
    Save processed data to file.

    Args:
        df: DataFrame to save
        output_path: Path to save file
        format: File format ('parquet', 'csv')
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if format == 'parquet':
        df.to_parquet(output_path, index=False)
    elif format == 'csv':
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"Saved processed data to {output_path}")


def load_processed_data(filepath: str) -> pd.DataFrame:
    """
    Load processed data from file.

    Args:
        filepath: Path to data file

    Returns:
        Loaded DataFrame
    """
    if filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    elif filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

    print(f"Loaded data from {filepath}: {df.shape}")
    return df
