"""
Utility functions for the water quality prediction project.
"""
import os
import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def setup_logging(log_dir: str = "../outputs/logs", log_name: str = "pipeline.log") -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_dir: Directory to save log files
        log_name: Name of the log file

    Returns:
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    return logger


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save dictionary as JSON file.

    Args:
        data: Dictionary to save
        filepath: Path to save JSON file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved JSON to {filepath}")


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file as dictionary.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        Dictionary containing RMSE, MAE, and R² scores
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }


def print_metrics(metrics: Dict[str, float], dataset_name: str = "Dataset") -> None:
    """
    Pretty print regression metrics.

    Args:
        metrics: Dictionary of metrics
        dataset_name: Name of the dataset being evaluated
    """
    print(f"\n{'='*50}")
    print(f"{dataset_name} Metrics:")
    print(f"{'='*50}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"R²:   {metrics['r2']:.4f}")
    print(f"{'='*50}\n")


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage of a pandas DataFrame by downcasting numeric types.

    Args:
        df: Input DataFrame
        verbose: Whether to print memory reduction info

    Returns:
        DataFrame with reduced memory usage
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB '
              f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')

    return df


def create_submission(predictions: np.ndarray,
                     template_path: str,
                     output_path: str,
                     target_col: str = 'target') -> pd.DataFrame:
    """
    Create submission file from predictions.

    Args:
        predictions: Array of predictions
        template_path: Path to submission template CSV
        output_path: Path to save submission
        target_col: Name of target column in submission

    Returns:
        Submission DataFrame
    """
    submission = pd.read_csv(template_path)
    submission[target_col] = predictions

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

    return submission


def load_data_safely(filepath: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Safely load data from various file formats.

    Args:
        filepath: Path to data file
        **kwargs: Additional arguments to pass to pandas read function

    Returns:
        DataFrame if successful, None otherwise
    """
    try:
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath, **kwargs)
        elif filepath.endswith('.parquet'):
            return pd.read_parquet(filepath, **kwargs)
        elif filepath.endswith('.xlsx'):
            return pd.read_excel(filepath, **kwargs)
        elif filepath.endswith('.json'):
            return pd.read_json(filepath, **kwargs)
        else:
            print(f"Unsupported file format: {filepath}")
            return None
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        return None


def get_feature_importance(model, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
    """
    Extract and sort feature importance from a trained model.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to return

    Returns:
        DataFrame with features and their importance scores
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })

    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
    return importance_df
