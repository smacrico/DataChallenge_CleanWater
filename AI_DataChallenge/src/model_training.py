"""
Model training and evaluation functions with optimized XGBoost configuration.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any
import pickle
import os

try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed")

try:
    from sklearn.model_selection import train_test_split, KFold, GroupKFold, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed")

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# BEST XGBoost HYPERPARAMETERS (optimized for water quality prediction)
BEST_XGB_PARAMS = {
    "max_depth": 9,
    "learning_rate": 0.035,
    "subsample": 0.82,
    "colsample_bytree": 0.78,
    "n_estimators": 900,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.1,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "eval_metric": "rmse",
    "random_state": 42,
    "n_jobs": -1
}


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 50,
    verbose: bool = True
) -> XGBRegressor:
    """
    Train XGBoost model with best parameters.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        params: Model parameters (defaults to BEST_XGB_PARAMS)
        early_stopping_rounds: Early stopping rounds
        verbose: Whether to print training progress

    Returns:
        Trained XGBoost model
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed")

    if params is None:
        params = BEST_XGB_PARAMS.copy()

    print("Training XGBoost with parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Initialize model
    model = XGBRegressor(**params)

    # Prepare evaluation set
    eval_set = [(X_train, y_train)]
    if X_val is not None and y_val is not None:
        eval_set.append((X_val, y_val))

    # Train model
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose
    )

    print(f"\nTraining completed. Best iteration: {model.best_iteration}")

    return model


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str = "Dataset"
) -> Dict[str, float]:
    """
    Evaluate model performance.

    Args:
        model: Trained model
        X: Features
        y: True target values
        dataset_name: Name for printing

    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }

    print(f"\n{'='*50}")
    print(f"{dataset_name} Metrics:")
    print(f"{'='*50}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"{'='*50}\n")

    return metrics


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: Optional[Dict[str, Any]] = None,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform K-Fold cross-validation.

    Args:
        X: Features
        y: Target
        params: Model parameters
        n_splits: Number of CV folds
        random_state: Random seed

    Returns:
        Dictionary with CV results
    """
    if params is None:
        params = BEST_XGB_PARAMS.copy()

    print(f"Performing {n_splits}-fold cross-validation...")

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_rmse_scores = []
    cv_r2_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        model = XGBRegressor(**params)
        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            early_stopping_rounds=50,
            verbose=False
        )

        y_pred = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
        r2 = r2_score(y_val_fold, y_pred)

        cv_rmse_scores.append(rmse)
        cv_r2_scores.append(r2)

        print(f"Fold {fold}: RMSE={rmse:.4f}, R²={r2:.4f}")

    results = {
        'mean_rmse': np.mean(cv_rmse_scores),
        'std_rmse': np.std(cv_rmse_scores),
        'mean_r2': np.mean(cv_r2_scores),
        'std_r2': np.std(cv_r2_scores),
        'all_rmse': cv_rmse_scores,
        'all_r2': cv_r2_scores
    }

    print(f"\nCross-Validation Results:")
    print(f"RMSE: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}")
    print(f"R²:   {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")

    return results


def spatial_cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    spatial_groups: pd.Series,
    params: Optional[Dict[str, Any]] = None,
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Perform spatial cross-validation using GroupKFold.

    Args:
        X: Features
        y: Target
        spatial_groups: Group labels for spatial grouping
        params: Model parameters
        n_splits: Number of CV folds

    Returns:
        Dictionary with CV results
    """
    if params is None:
        params = BEST_XGB_PARAMS.copy()

    print(f"Performing spatial {n_splits}-fold cross-validation...")

    group_kfold = GroupKFold(n_splits=n_splits)

    cv_rmse_scores = []
    cv_r2_scores = []

    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X, y, groups=spatial_groups), 1):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        model = XGBRegressor(**params)
        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            early_stopping_rounds=50,
            verbose=False
        )

        y_pred = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
        r2 = r2_score(y_val_fold, y_pred)

        cv_rmse_scores.append(rmse)
        cv_r2_scores.append(r2)

        print(f"Fold {fold}: RMSE={rmse:.4f}, R²={r2:.4f}")

    results = {
        'mean_rmse': np.mean(cv_rmse_scores),
        'std_rmse': np.std(cv_rmse_scores),
        'mean_r2': np.mean(cv_r2_scores),
        'std_r2': np.std(cv_r2_scores),
        'all_rmse': cv_rmse_scores,
        'all_r2': cv_r2_scores
    }

    print(f"\nSpatial Cross-Validation Results:")
    print(f"RMSE: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}")
    print(f"R²:   {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")

    return results


def hyperparameter_tuning_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 100,
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning using Optuna.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        n_trials: Number of Optuna trials
        timeout: Timeout in seconds

    Returns:
        Dictionary with best parameters and study results
    """
    if not OPTUNA_AVAILABLE:
        print("Optuna not available. Returning BEST_XGB_PARAMS.")
        return {'best_params': BEST_XGB_PARAMS}

    print(f"Starting Optuna hyperparameter tuning ({n_trials} trials)...")

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0),
            'n_estimators': 500,
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'eval_metric': 'rmse',
            'random_state': 42,
            'n_jobs': -1
        }

        model = XGBRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )

        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        return rmse

    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    print(f"\nBest trial RMSE: {study.best_value:.4f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    return {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'study': study
    }


def save_model(model, filepath: str) -> None:
    """
    Save trained model to disk.

    Args:
        model: Trained model
        filepath: Path to save model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved to {filepath}")


def load_model(filepath: str):
    """
    Load trained model from disk.

    Args:
        filepath: Path to saved model

    Returns:
        Loaded model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)

    print(f"Model loaded from {filepath}")
    return model


def get_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 20,
    importance_type: str = 'weight'
) -> pd.DataFrame:
    """
    Get feature importance from XGBoost model.

    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        top_n: Number of top features to return
        importance_type: Type of importance ('weight', 'gain', 'cover')

    Returns:
        DataFrame with feature importance
    """
    importance = model.get_booster().get_score(importance_type=importance_type)

    importance_df = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance': list(importance.values())
    })

    # Map feature indices to names if needed
    if importance_df['feature'].iloc[0].startswith('f'):
        feature_mapping = {f'f{i}': name for i, name in enumerate(feature_names)}
        importance_df['feature'] = importance_df['feature'].map(feature_mapping)

    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)

    return importance_df
