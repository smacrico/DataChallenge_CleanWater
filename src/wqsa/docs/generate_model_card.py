"""Generate Model Card documentation.

Creates a comprehensive model card with training metrics and metadata.
"""

import logging
from datetime import datetime
from pathlib import Path

import yaml

from ..utils.config import load_config
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)


def generate_model_card() -> str:
    """Generate Model Card markdown content.

    Returns:
        Model card content as markdown string
    """
    config = load_config()

    # Load CV metadata if available
    models_dir = Path(config.get("paths", {}).get("models", "models"))
    metadata_path = models_dir / "cv_metadata.yaml"

    if metadata_path.exists():
        with open(metadata_path) as f:
            cv_metadata = yaml.safe_load(f)
    else:
        cv_metadata = {}

    # Extract configuration details
    targets = config.get("targets", ["ALKALINITY", "EC", "DRP"])
    landsat_features = config.get("features", {}).get("landsat", [])
    terraclimate_features = config.get("features", {}).get("terraclimate", [])
    total_features = len(landsat_features) + len(terraclimate_features)

    n_folds = config.get("modeling", {}).get("cv_splits", 5)
    cv_scores = cv_metadata.get("cv_scores", {})
    mean_r2 = cv_metadata.get("mean_cv_r2", "N/A")

    # Build model card
    content = f"""# Model Card: Water Quality South Africa Predictor

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Model Overview

This model predicts three water quality parameters for South African water bodies:
- **Total Alkalinity** (mg/L CaCO₃)
- **Electrical Conductance (EC)** (µS/cm)
- **Dissolved Reactive Phosphorus (DRP)** (mg/L)

### Model Architecture

- **Type:** Ensemble of per-target regressors
- **Base Models:** {'XGBoost' if 'xgboost' in config.get('modeling', {}).get('model_priority', []) else 'Random Forest'}
- **Meta-Learner:** Ridge regression (blender)
- **Cross-Validation:** {n_folds}-fold GroupKFold (leave-location-out)

## Performance Metrics

### Cross-Validation R² Scores

| Target          | CV R²   |
|-----------------|---------|
| Alkalinity      | {cv_scores.get('ALKALINITY', 'N/A'):.4f if isinstance(cv_scores.get('ALKALINITY'), (int, float)) else 'N/A'} |
| EC              | {cv_scores.get('EC', 'N/A'):.4f if isinstance(cv_scores.get('EC'), (int, float)) else 'N/A'} |
| DRP             | {cv_scores.get('DRP', 'N/A'):.4f if isinstance(cv_scores.get('DRP'), (int, float)) else 'N/A'} |
| **Mean R²**     | **{mean_r2:.4f if isinstance(mean_r2, (int, float)) else 'N/A'}** |

### Validation Strategy

- **Method:** Spatial generalization via GroupKFold
- **Grouping:** By monitoring station (STATION_KEY)
- **Purpose:** Ensure model generalizes to unseen locations

## Features

### Total Feature Count: {total_features}

#### Landsat-Derived Features ({len(landsat_features)})
{chr(10).join(f"- {feat}" for feat in landsat_features)}

#### TerraClimate Features ({len(terraclimate_features)})
{chr(10).join(f"- {feat}" for feat in terraclimate_features)}

### Feature Engineering

**Landsat Join Strategy:**
- Prefer non-future scenes (on/before sample date)
- Fallback to nearest scene within ±60 days
- Compute SCENE_GAP_DAYS and cloud coverage flags

**TerraClimate Join Strategy:**
- Match by month (YYYY-MM)
- Fallback to previous month if current unavailable
- Rolling window aggregations (1/3/6 months)
- Seasonality encoding (sin/cos month)

## Data Sources

- **Landsat L2:** NDVI, NDWI, NDBI surface reflectance indices
- **TerraClimate:** Monthly climate variables (precipitation, runoff, VPD, deficit)
- **Training Data:** South African water quality monitoring stations
- **Data Policy:** Public/open datasets only (no proprietary data)

## Intended Use

### Primary Use Case
Predict water quality parameters at new spatial locations and time points for:
- Environmental monitoring
- Early warning systems
- Resource management planning

### Limitations
- Model trained on South African data; may not generalize to other regions
- Relies on satellite data availability (cloud cover may impact Landsat features)
- Temporal gaps in TerraClimate monthly data
- Predictions are estimates; not a substitute for direct laboratory measurements

## Ethical Considerations

- **Transparency:** All data sources are public
- **Fairness:** Model evaluated for spatial generalization across diverse locations
- **Privacy:** No personally identifiable information (PII) used
- **Environmental Impact:** Supports water resource conservation efforts

## Model Maintenance

- **Retraining:** Recommended quarterly with new monitoring data
- **Monitoring:** Track prediction accuracy on incoming samples
- **Updates:** Re-evaluate feature importance and add new satellite products as available

## Contact & Feedback

For questions, issues, or collaboration opportunities:
- GitHub: [water-quality-sa-predictor](https://github.com/your-org/water-quality-sa-predictor)
- Email: team@ey-challenge.example.com

---

*This model card follows the framework proposed by Mitchell et al. (2019) and Gebru et al. (2018).*
"""

    return content


def main():
    """Generate and save Model Card."""
    setup_logging()
    logger.info("Generating Model Card...")

    config = load_config()
    content = generate_model_card()

    # Save to artifacts
    artifacts_dir = Path(config.get("paths", {}).get("artifacts", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    output_path = artifacts_dir / "MODEL_CARD.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Model Card saved: {output_path}")
    print(content)


if __name__ == "__main__":
    main()
