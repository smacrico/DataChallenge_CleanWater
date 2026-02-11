"""
Feature engineering functions for water quality prediction.
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime


def create_temporal_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Create temporal features from date column.

    Args:
        df: Input DataFrame
        date_col: Name of date column

    Returns:
        DataFrame with added temporal features
    """
    df = df.copy()

    if date_col not in df.columns:
        print(f"Warning: {date_col} not found in DataFrame")
        return df

    # Convert to datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Extract temporal components
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    df['quarter'] = df[date_col].dt.quarter
    df['is_weekend'] = df[date_col].dt.dayofweek.isin([5, 6]).astype(int)

    # Cyclical encoding for month and day_of_year
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # Season
    df['season'] = df['month'] % 12 // 3 + 1  # 1=Winter, 2=Spring, 3=Summer, 4=Fall

    print(f"Created temporal features from {date_col}")
    return df


def create_spatial_features(df: pd.DataFrame,
                           lat_col: str = 'latitude',
                           lon_col: str = 'longitude') -> pd.DataFrame:
    """
    Create spatial features from latitude and longitude.

    Args:
        df: Input DataFrame
        lat_col: Name of latitude column
        lon_col: Name of longitude column

    Returns:
        DataFrame with added spatial features
    """
    df = df.copy()

    if lat_col not in df.columns or lon_col not in df.columns:
        print(f"Warning: {lat_col} or {lon_col} not found in DataFrame")
        return df

    # Radial distance from origin
    df['distance_from_origin'] = np.sqrt(df[lat_col]**2 + df[lon_col]**2)

    # Cyclical encoding for coordinates
    df['lat_sin'] = np.sin(np.radians(df[lat_col]))
    df['lat_cos'] = np.cos(np.radians(df[lat_col]))
    df['lon_sin'] = np.sin(np.radians(df[lon_col]))
    df['lon_cos'] = np.cos(np.radians(df[lon_col]))

    # Spatial binning
    df['lat_bin'] = pd.cut(df[lat_col], bins=10, labels=False)
    df['lon_bin'] = pd.cut(df[lon_col], bins=10, labels=False)

    # Spatial quadrant
    lat_median = df[lat_col].median()
    lon_median = df[lon_col].median()
    df['quadrant'] = ((df[lat_col] > lat_median).astype(int) * 2 +
                      (df[lon_col] > lon_median).astype(int))

    print(f"Created spatial features from {lat_col} and {lon_col}")
    return df


def create_landsat_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create spectral indices from Landsat bands.

    Args:
        df: DataFrame with Landsat band columns

    Returns:
        DataFrame with added spectral indices
    """
    df = df.copy()

    # Check for required bands
    required_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    available_bands = [b for b in required_bands if b in df.columns]

    if len(available_bands) < 4:
        print("Warning: Insufficient Landsat bands for index calculation")
        return df

    # NDVI (Normalized Difference Vegetation Index)
    if 'B5' in df.columns and 'B4' in df.columns:
        df['NDVI'] = (df['B5'] - df['B4']) / (df['B5'] + df['B4'] + 1e-8)

    # NDWI (Normalized Difference Water Index)
    if 'B3' in df.columns and 'B5' in df.columns:
        df['NDWI'] = (df['B3'] - df['B5']) / (df['B3'] + df['B5'] + 1e-8)

    # NBR (Normalized Burn Ratio)
    if 'B5' in df.columns and 'B7' in df.columns:
        df['NBR'] = (df['B5'] - df['B7']) / (df['B5'] + df['B7'] + 1e-8)

    # EVI (Enhanced Vegetation Index)
    if all(b in df.columns for b in ['B5', 'B4', 'B2']):
        df['EVI'] = 2.5 * ((df['B5'] - df['B4']) /
                           (df['B5'] + 6 * df['B4'] - 7.5 * df['B2'] + 1))

    # NDBI (Normalized Difference Built-up Index)
    if 'B6' in df.columns and 'B5' in df.columns:
        df['NDBI'] = (df['B6'] - df['B5']) / (df['B6'] + df['B5'] + 1e-8)

    # MNDWI (Modified Normalized Difference Water Index)
    if 'B3' in df.columns and 'B6' in df.columns:
        df['MNDWI'] = (df['B3'] - df['B6']) / (df['B3'] + df['B6'] + 1e-8)

    # SAVI (Soil Adjusted Vegetation Index)
    if 'B5' in df.columns and 'B4' in df.columns:
        L = 0.5
        df['SAVI'] = ((df['B5'] - df['B4']) / (df['B5'] + df['B4'] + L)) * (1 + L)

    # Brightness
    if all(f'B{i}' in df.columns for i in range(1, 8)):
        df['brightness'] = np.sqrt(sum(df[f'B{i}']**2 for i in range(1, 8)))

    print("Created Landsat spectral indices")
    return df


def create_climate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived climate features from TerraClimate data.

    Args:
        df: DataFrame with TerraClimate columns

    Returns:
        DataFrame with added climate features
    """
    df = df.copy()

    # Temperature range
    if 'tmax' in df.columns and 'tmin' in df.columns:
        df['temp_range'] = df['tmax'] - df['tmin']

    # Precipitation intensity
    if 'ppt' in df.columns and 'ppt_days' in df.columns:
        df['ppt_intensity'] = df['ppt'] / (df['ppt_days'] + 1)

    # Vapor Pressure Deficit
    if 'vpd' in df.columns:
        df['vpd_log'] = np.log1p(df['vpd'])

    # Soil moisture deficit
    if 'soil' in df.columns and 'def' in df.columns:
        df['soil_moisture_deficit'] = df['soil'] - df['def']

    # Water balance
    if 'ppt' in df.columns and 'pet' in df.columns:
        df['water_balance'] = df['ppt'] - df['pet']

    # Aridity index
    if 'ppt' in df.columns and 'pet' in df.columns:
        df['aridity_index'] = df['ppt'] / (df['pet'] + 1)

    print("Created climate-derived features")
    return df


def create_interaction_features(df: pd.DataFrame,
                                feature_pairs: Optional[List[tuple]] = None) -> pd.DataFrame:
    """
    Create interaction features between important variables.

    Args:
        df: Input DataFrame
        feature_pairs: List of tuples specifying feature pairs to interact

    Returns:
        DataFrame with added interaction features
    """
    df = df.copy()

    if feature_pairs is None:
        # Default interactions
        feature_pairs = []

        if 'NDVI' in df.columns and 'ppt' in df.columns:
            feature_pairs.append(('NDVI', 'ppt'))

        if 'NDWI' in df.columns and 'soil' in df.columns:
            feature_pairs.append(('NDWI', 'soil'))

        if 'temperature' in df.columns and 'elevation' in df.columns:
            feature_pairs.append(('temperature', 'elevation'))

    # Create multiplication interactions
    for feat1, feat2 in feature_pairs:
        if feat1 in df.columns and feat2 in df.columns:
            interaction_name = f'{feat1}_x_{feat2}'
            df[interaction_name] = df[feat1] * df[feat2]

    if feature_pairs:
        print(f"Created {len(feature_pairs)} interaction features")

    return df


def create_aggregation_features(df: pd.DataFrame,
                                group_col: str,
                                agg_cols: List[str],
                                agg_funcs: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
    """
    Create aggregation features grouped by a categorical column.

    Args:
        df: Input DataFrame
        group_col: Column to group by
        agg_cols: Columns to aggregate
        agg_funcs: Aggregation functions to apply

    Returns:
        DataFrame with added aggregation features
    """
    df = df.copy()

    if group_col not in df.columns:
        print(f"Warning: {group_col} not found in DataFrame")
        return df

    for col in agg_cols:
        if col not in df.columns:
            continue

        for func in agg_funcs:
            agg_feature = df.groupby(group_col)[col].transform(func)
            df[f'{col}_{group_col}_{func}'] = agg_feature

    print(f"Created aggregation features grouped by {group_col}")
    return df


def create_lag_features(df: pd.DataFrame,
                       columns: List[str],
                       lags: List[int] = [1, 7, 30],
                       group_col: Optional[str] = None) -> pd.DataFrame:
    """
    Create lag features for time series data.

    Args:
        df: Input DataFrame (must be sorted by time)
        columns: Columns to create lags for
        lags: List of lag periods
        group_col: Optional column to group by before creating lags

    Returns:
        DataFrame with added lag features
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        for lag in lags:
            if group_col:
                df[f'{col}_lag_{lag}'] = df.groupby(group_col)[col].shift(lag)
            else:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    print(f"Created lag features for {len(columns)} columns")
    return df


def create_rolling_features(df: pd.DataFrame,
                           columns: List[str],
                           windows: List[int] = [7, 14, 30],
                           functions: List[str] = ['mean', 'std'],
                           group_col: Optional[str] = None) -> pd.DataFrame:
    """
    Create rolling window features.

    Args:
        df: Input DataFrame (must be sorted by time)
        columns: Columns to create rolling features for
        windows: List of window sizes
        functions: Rolling functions to apply
        group_col: Optional column to group by

    Returns:
        DataFrame with added rolling features
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        for window in windows:
            for func in functions:
                if group_col:
                    rolling_feature = df.groupby(group_col)[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).agg(func)
                    )
                else:
                    rolling_feature = df[col].rolling(window=window, min_periods=1).agg(func)

                df[f'{col}_rolling_{window}_{func}'] = rolling_feature

    print(f"Created rolling features for {len(columns)} columns")
    return df
