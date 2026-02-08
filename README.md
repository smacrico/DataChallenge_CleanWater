# Water Quality Predictor - South Africa

[![CI](https://github.com/yourusername/water-quality-sa-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/water-quality-sa-predictor/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production-quality ML pipeline for predicting water quality metrics in South Africa using open datasets, Snowflake/Snowpark, and ensemble modeling.

## Overview

This repository implements a complete solution for the **EY AI & Data Challenge — Water Quality (South Africa)**, predicting three critical water quality targets at geographic locations and dates:

1. **Total Alkalinity**
2. **Electrical Conductance (EC)**
3. **Dissolved Reactive Phosphorus (DRP)**

### Key Constraints

- **Open Data Only**: Uses publicly available datasets (Landsat L2, TerraClimate)
- **No Data Leakage**: Anti-leakage temporal joins (prefer non-future Landsat scenes)
- **Spatial Generalization**: GroupKFold cross-validation by station (leave-location-out)
- **Single CSV Submission**: 200 rows × 3 columns in specified order
- **Evaluation Metric**: Mean R² across three targets

## Features

### Data Sources

- **Landsat L2**: NDVI, NDWI, NDBI with 250m/1km buffers + cloud filtering
- **TerraClimate**: Monthly precipitation, runoff, VPD, deficit with rolling windows (1/3/6 months)
- **Seasonality**: Month sine/cosine encoding

### Technical Stack

- **Data Platform**: Snowflake + Snowpark Python
- **ML Framework**: XGBoost (primary) / RandomForest (fallback)
- **Ensemble**: Ridge blending of fold predictions
- **Reproducibility**: Pinned dependencies, deterministic random seeds
- **Documentation**: Auto-generated Model Card, Business Plan, Video Script (MD/HTML/PDF)

## Quickstart

### Prerequisites

- Python 3.11+
- Snowflake account with appropriate permissions
- Git & Make

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/water-quality-sa-predictor.git
cd water-quality-sa-predictor

# Install dependencies
make install

# Copy and configure environment
cp .env.example .env
# Edit .env with your Snowflake credentials
```

### Snowflake Setup

Ensure your Snowflake account has:

1. **Database**: `AI_CHALLENGE_DB`
2. **Schema**: `PUBLIC`
3. **Warehouse**: `COMPUTE_WH` (or configured name)
4. **Stage**: `@AI_CHALLENGE_STAGE` with files:
   - `water_quality_training_dataset.csv`
   - `submission_template.csv`
   - `landsat_features/` (directory with Landsat CSVs)
   - `terraclimate_features/` (directory with TerraClimate CSVs)

Configure `.env`:

```bash
SF_ACCOUNT=your_account
SF_USER=your_user
SF_PASSWORD=your_password
SF_ROLE=SYSADMIN
SF_WAREHOUSE=COMPUTE_WH
SF_DATABASE=AI_CHALLENGE_DB
SF_SCHEMA=PUBLIC
SF_STAGE=@AI_CHALLENGE_STAGE
```

### Running the Pipeline

#### Option 1: Full Automated Pipeline

```bash
make all
```

This runs: linting → tests → feature engineering → training → prediction → docs generation → exports

#### Option 2: Step-by-Step

```bash
# Lint and format code
make lint

# Run tests
make test

# Train models with cross-validation
make train

# Generate predictions and submission.csv
make predict

# Generate documentation (Model Card, Business Plan, Video Script)
make docs

# Export documentation to HTML/PDF
make export
```

#### Option 3: Jupyter Notebooks

Launch Jupyter and run notebooks sequentially:

```bash
jupyter notebook
```

1. `01_ingest_and_stage_check.ipynb` - Verify Snowflake stage access
2. `02_join_landsat_terraclimate.ipynb` - Feature engineering
3. `03_build_gold_tables.ipynb` - Create TRAIN_GOLD, VALID_GOLD tables
4. `04_train_groupkfold_cv.ipynb` - Train models with cross-validation
5. `05_predict_and_submission.ipynb` - Generate submission.csv
6. `06_auto_docs_and_exports.ipynb` - Generate and export documentation

### Outputs

After running the pipeline:

- **Submission**: `artifacts/submission.csv` (200 rows × 3 targets)
- **Models**: `artifacts/models/` (fold models + blender)
- **Documentation**:
  - `artifacts/MODEL_CARD.{md,html,pdf}`
  - `artifacts/BUSINESS_PLAN_SNAPSHOT.{md,html,pdf}`
  - `artifacts/VIDEO_SCRIPT.{md,html,pdf}`

## Pipeline Architecture

### 1. Feature Engineering

#### Landsat Join (Anti-Leakage)
- **Preference**: Non-future scenes (0-60 days before sample date)
- **Fallback**: Nearest scene within ±60 days
- **Metrics**: Scene gap days, cloud fraction, coverage flags

#### TerraClimate Join
- **Primary**: Same month as sample
- **Fallback**: Previous month if unavailable
- **Features**: Rolling sums/means over 1/3/6 month windows

### 2. Gold Table Creation

Tables built in Snowflake:
- `TRAIN_GOLD`: Training data with all features
- `VALID_GOLD`: Validation data for submission

### 3. Model Training

- **CV Strategy**: GroupKFold by station (n_splits=5)
- **Per-Target Models**: XGBoost (or RandomForest fallback)
- **Ensemble**: Ridge blending of fold predictions (optional)
- **Metrics**: Per-fold R², overall CV R²

### 4. Prediction & Submission

- Load VALID_GOLD from Snowflake
- Average fold predictions
- Apply blender
- Format as submission.csv (200 rows, 3 columns)

## Development

### Code Quality

```bash
# Format code
make format

# Run linters
make lint

# Run all tests
make test

# Quick smoke test
make smoke
```

### Pre-commit Hooks

Pre-commit hooks automatically run on `git commit`:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Testing

```bash
# Run all tests with coverage
pytest --cov=src/wqsa tests/

# Run specific test
pytest tests/test_features_smoke.py -v
```

## CI/CD

### Continuous Integration

On every push/PR to `main`:
- Code linting (ruff, black, isort)
- Unit tests
- Notebook smoke tests

### Release Workflow

On git tag creation:
- Build submission.csv
- Generate documentation
- Attach artifacts to release

```bash
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

## Configuration

Edit `config/project.yaml` to adjust:

```yaml
landsat_max_lag_days: 60
prefer_non_future: true
tc_month_join: same_then_previous

features:
  landsat:
    - NDVI_MEAN_B250
    - NDBI_MEAN_B1K
    - NDWI_MEAN_B250
    - CLOUD_FRAC_B250
    - LANDSAT_LOW_COVERAGE_FLAG
    - SCENE_GAP_DAYS
  
  terraclimate:
    - PPT_M0
    - PPT_SUM_M1
    - PPT_SUM_M3
    - PPT_SUM_M6
    - Q_M0
    - Q_SUM_M1
    - Q_SUM_M3
    - VPD_M0
    - VPD_MEAN_M3
    - DEF_M0
    - DEF_MEAN_M3
    - MON_SIN
    - MON_COS
```

## Project Structure
