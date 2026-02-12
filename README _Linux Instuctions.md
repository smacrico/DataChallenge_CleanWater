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

See folder structure at the top of this README.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Security

For security concerns, see [SECURITY.md](SECURITY.md).

## Acknowledgments

- EY AI & Data Challenge organizers
- Landsat program (USGS)
- TerraClimate dataset (University of Idaho)
- Snowflake for data platform

---

## 📖 How the Pipeline Works: Step-by-Step

This section provides a detailed walkthrough of how the water quality prediction pipeline operates from data ingestion to final submission.

### Overview of Data Flow

```
Raw Data (Snowflake Stage)
    ↓
Stage Validation & Loading
    ↓
Feature Engineering (Landsat + TerraClimate Joins)
    ↓
Gold Table Creation (TRAIN_GOLD, VALID_GOLD)
    ↓
Model Training (GroupKFold CV)
    ↓
Prediction & Ensemble
    ↓
Submission Generation (submission.csv)
    ↓
Documentation Export (Model Card, Business Plan)
```

### Step 1: Data Ingestion & Stage Validation

**File**: `notebooks/01_ingest_and_stage_check.ipynb`  
**Purpose**: Verify Snowflake connectivity and stage file availability

**What Happens**:
1. **Snowpark Session Creation**: Establishes connection to Snowflake using credentials from `.env`
2. **Stage Verification**: Checks that `@AI_CHALLENGE_STAGE` contains:
   - `water_quality_training_dataset.csv` (training samples with targets)
   - `submission_template.csv` (200 validation samples)
   - `landsat_features/*.csv` (satellite imagery features)
   - `terraclimate_features/*.csv` (climate data features)
3. **Data Loading**: Reads CSV files into Snowflake DataFrames
4. **Schema Validation**: Verifies expected columns exist

**Key Functions**:
```python
from src.wqsa.io.snowflake_session import create_snowpark_session
from src.wqsa.io.staging_io import list_stage_files, load_csv_from_stage

session = create_snowpark_session()
files = list_stage_files(session, "@AI_CHALLENGE_STAGE")
train_df = load_csv_from_stage(session, "@AI_CHALLENGE_STAGE/water_quality_training_dataset.csv")
```

**Output**: Confirmation that all required data files are accessible

---

### Step 2: Feature Engineering - Landsat Join

**Module**: `src/wqsa/features/landsat_join.py`  
**Purpose**: Join satellite imagery features with anti-leakage constraints

**How It Works**:

#### Anti-Leakage Strategy
To prevent data leakage (using future information to predict the past), the join implements a two-stage preference system:

1. **Primary**: Non-future scenes (0-60 days before sample date)
   - Finds Landsat scenes captured **on or before** the water sample date
   - Selects the scene closest to the sample date (minimum gap)
   
2. **Fallback**: Nearest scene within ±60 days
   - If no non-future scene exists, uses the absolute nearest scene
   - Can be up to 60 days before or after the sample

#### Process Flow
```python
def join_landsat_features(samples_df, landsat_df, config):
    # 1. Join samples with Landsat on STATION_KEY
    joined = samples.join(landsat, on="STATION_KEY")
    
    # 2. Calculate SCENE_GAP_DAYS = SAMPLE_DATE - SCENE_DATE
    #    Positive values = scene captured before sample (good)
    #    Negative values = scene captured after sample (data leakage risk)
    joined = joined.with_column(
        "SCENE_GAP_DAYS", 
        datediff("day", SCENE_DATE, SAMPLE_DATE)
    )
    
    # 3. Filter candidates: |gap| <= 60 days
    candidates = joined.filter(abs(SCENE_GAP_DAYS) <= 60)
    
    # 4. Prefer non-future (SCENE_GAP_DAYS >= 0)
    non_future = candidates.filter(SCENE_GAP_DAYS >= 0)
    best_non_future = non_future.sort(SCENE_GAP_DAYS).limit(1)
    
    # 5. Fallback to nearest if no non-future exists
    nearest = candidates.sort(abs(SCENE_GAP_DAYS)).limit(1)
    
    # 6. Coalesce: use non_future if available, else nearest
    result = coalesce(best_non_future, nearest)
```

#### Features Added
- **NDVI_MEAN_B250**: Vegetation index (health) in 250m buffer
- **NDBI_MEAN_B1K**: Built-up index in 1km buffer (urbanization)
- **NDWI_MEAN_B250**: Water index in 250m buffer
- **CLOUD_FRAC_B250**: Cloud coverage percentage
- **LANDSAT_LOW_COVERAGE_FLAG**: Binary flag for poor scene quality
- **SCENE_GAP_DAYS**: Days between sample and scene (metadata)

---

### Step 3: Feature Engineering - TerraClimate Join

**Module**: `src/wqsa/features/terraclimate_join.py`  
**Purpose**: Join monthly climate data with temporal features

**How It Works**:

#### Month-Based Join Strategy
TerraClimate data is organized by month (YYYY-MM format). The join follows this logic:

1. **Extract Sample Month**: Convert `SAMPLE_DATE` to `YYYY-MM` format
2. **Same Month Join**: Match `SAMPLE_MONTH == TC_MONTH`
3. **Fallback to Previous**: If no match, use `TC_MONTH = SAMPLE_MONTH - 1`

#### Process Flow
```python
def join_terraclimate_features(samples_df, tc_df, config):
    # 1. Extract month from sample date
    samples = samples.with_column(
        "SAMPLE_MONTH", 
        to_char(SAMPLE_DATE, "YYYY-MM")
    )
    
    # 2. Try same-month join
    same_month = samples.join(
        tc, 
        (STATION_KEY matches) AND (SAMPLE_MONTH == TC_MONTH)
    )
    
    # 3. Calculate previous month for fallback
    samples = samples.with_column(
        "PREV_MONTH",
        to_char(add_months(SAMPLE_DATE, -1), "YYYY-MM")
    )
    
    # 4. Join on previous month for unmatched rows
    prev_month = samples.join(
        tc,
        (STATION_KEY matches) AND (PREV_MONTH == TC_MONTH)
    )
    
    # 5. Coalesce: prefer same month, fallback to previous
    result = coalesce(same_month, prev_month)
```

#### Features Added
- **Current Month (M0)**: `PPT_M0`, `Q_M0`, `VPD_M0`, `DEF_M0`
- **Rolling Sums**: `PPT_SUM_M1`, `PPT_SUM_M3`, `PPT_SUM_M6` (precipitation aggregates)
- **Rolling Means**: `VPD_MEAN_M3`, `DEF_MEAN_M3` (vapor pressure, water deficit)
- **Seasonality**: `MON_SIN`, `MON_COS` (cyclical month encoding)

**Why Rolling Windows?**  
Water quality responds to cumulative weather patterns. 3-month and 6-month rolling sums capture:
- Drought conditions (low precipitation)
- Flooding events (high runoff)
- Seasonal climate cycles

---

### Step 4: Gold Table Creation

**Module**: `src/wqsa/features/gold_builder.py`  
**Purpose**: Assemble canonical feature sets for modeling

**What Happens**:

1. **Column Selection**: Extracts only the features defined in `config/project.yaml`
2. **TRAIN_GOLD**: Training data with features + targets (ALKALINITY, EC, DRP)
3. **VALID_GOLD**: Validation data with features only (no targets)
4. **Data Quality Checks**:
   - Verifies no missing required columns
   - Logs row counts and feature statistics
   - Validates data types

```python
def build_train_gold(session, train_df, config):
    # 1. Get feature columns from config
    landsat_features = config["features"]["landsat"]
    tc_features = config["features"]["terraclimate"]
    all_features = landsat_features + tc_features
    
    # 2. Add target columns
    targets = ["ALKALINITY", "EC", "DRP"]
    
    # 3. Add metadata
    metadata = ["STATION_KEY", "SAMPLE_DATE", "SAMPLE_ID"]
    
    # 4. Select canonical columns
    columns = metadata + all_features + targets
    gold_df = train_df.select(columns)
    
    # 5. Save to Snowflake
    gold_df.write.mode("overwrite").save_as_table("TRAIN_GOLD")
```

**Output Tables**:
- `TRAIN_GOLD`: ~800-1000 rows with 19 features + 3 targets
- `VALID_GOLD`: 200 rows with 19 features (submission template)

---

### Step 5: Model Training with Cross-Validation

**Module**: `src/wqsa/modeling/train_cv.py`  
**Purpose**: Train robust models with spatial generalization

**Training Strategy**:

#### GroupKFold Cross-Validation
Regular K-Fold can leak information when multiple samples come from the same location. **GroupKFold** ensures all samples from a station stay together:

```
Fold 1: Stations [A, B, C] → Train   |   Stations [D, E] → Validate
Fold 2: Stations [A, B, D] → Train   |   Stations [C, E] → Validate
Fold 3: Stations [A, C, E] → Train   |   Stations [B, D] → Validate
...
```

**Why?** This tests if the model can predict water quality at **new locations** not seen during training.

#### Per-Target Modeling
Each water quality parameter gets its own model:
- **ALKALINITY_model**: XGBoost trained on ALKALINITY values
- **EC_model**: XGBoost trained on EC values  
- **DRP_model**: XGBoost trained on DRP values

**Why separate models?**  
Each target has different:
- Value ranges (ALKALINITY: 0-500, EC: 0-3000, DRP: 0-10)
- Feature importance (NDVI matters more for DRP than EC)
- Non-linear patterns

#### Training Process
```python
def train_cv_models(X, y, groups, config, target_name):
    # 1. Initialize GroupKFold
    gkf = GroupKFold(n_splits=5)
    
    # 2. For each fold
    fold_models = []
    oof_predictions = np.zeros(len(X))
    
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        # 3. Split data by station groups
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 4. Create and train model
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # 5. Generate out-of-fold predictions
        oof_predictions[val_idx] = model.predict(X_val)
        
        # 6. Save fold model
        joblib.dump(model, f"models/{target_name}_fold{fold_idx}.pkl")
        fold_models.append(model)
        
        # 7. Calculate fold R² score
        fold_r2 = r2_score(y_val, oof_predictions[val_idx])
        print(f"Fold {fold_idx} R²: {fold_r2:.4f}")
    
    # 8. Calculate overall CV R² (using all out-of-fold predictions)
    cv_r2 = r2_score(y, oof_predictions)
    print(f"Overall CV R² for {target_name}: {cv_r2:.4f}")
    
    return fold_models, oof_predictions, cv_r2
```

#### Optional: Ridge Blender
After training, a Ridge regression model can blend fold predictions:
```python
# Stack out-of-fold predictions from all folds
blender = Ridge(alpha=1.0)
blender.fit(oof_predictions.reshape(-1, 1), y)
```

**Benefit**: Smooths prediction variance and can improve final R² by 0.01-0.03

**Model Artifacts Saved**:
- `models/ALKALINITY_fold0.pkl` through `fold4.pkl`
- `models/EC_fold0.pkl` through `fold4.pkl`
- `models/DRP_fold0.pkl` through `fold4.pkl`
- `models/ALKALINITY_blender.pkl` (optional)
- `models/EC_blender.pkl` (optional)
- `models/DRP_blender.pkl` (optional)

---

### Step 6: Prediction & Submission Generation

**Module**: `src/wqsa/modeling/predict.py`  
**Purpose**: Generate predictions for validation set

**Prediction Pipeline**:

```python
def generate_submission():
    # 1. Load VALID_GOLD from Snowflake
    session = create_snowpark_session()
    valid_df = session.table("VALID_GOLD").to_pandas()
    
    # 2. Prepare features (same columns as training)
    features = config["features"]["landsat"] + config["features"]["terraclimate"]
    X_valid = valid_df[features]
    
    # 3. Load trained models for each target
    targets = ["ALKALINITY", "EC", "DRP"]
    predictions = {}
    
    for target in targets:
        # 4. Load all 5 fold models
        fold_models = []
        for fold_idx in range(5):
            model = joblib.load(f"models/{target}_fold{fold_idx}.pkl")
            fold_models.append(model)
        
        # 5. Generate predictions from each fold
        fold_preds = np.column_stack([
            model.predict(X_valid) for model in fold_models
        ])
        
        # 6. Average fold predictions
        avg_pred = fold_preds.mean(axis=1)
        
        # 7. Apply blender if available
        if os.path.exists(f"models/{target}_blender.pkl"):
            blender = joblib.load(f"models/{target}_blender.pkl")
            final_pred = blender.predict(avg_pred.reshape(-1, 1))
        else:
            final_pred = avg_pred
        
        predictions[target] = final_pred
    
    # 8. Create submission DataFrame
    submission = pd.DataFrame({
        "ALKALINITY": predictions["ALKALINITY"],
        "EC": predictions["EC"],
        "DRP": predictions["DRP"]
    })
    
    # 9. Ensure exact 200 rows in correct order
    assert len(submission) == 200, "Submission must have exactly 200 rows"
    
    # 10. Save to CSV
    submission.to_csv("artifacts/submission.csv", index=False)
    print("Submission saved to artifacts/submission.csv")
```

**Ensemble Strategy**: Averaging 5 fold models reduces overfitting and improves generalization

---

### Step 7: Documentation Generation

**Modules**: `src/wqsa/docs/`  
**Purpose**: Auto-generate model documentation

**Generated Documents**:

1. **MODEL_CARD.md**
   - Model architecture and hyperparameters
   - Feature importance rankings
   - Cross-validation performance metrics
   - Intended use cases and limitations
   - Training dataset statistics

2. **BUSINESS_PLAN_SNAPSHOT.md**
   - Deployment strategy for water quality monitoring
   - Cost-benefit analysis
   - Scaling considerations
   - Integration with existing systems

3. **VIDEO_SCRIPT.md**
   - Presentation script for explaining the solution
   - Key talking points about methodology
   - Visual suggestions for slides

**Export Formats**:
```bash
python -m src.wqsa.docs.export_to_html  # → artifacts/*.html
python -m src.wqsa.docs.export_to_pdf   # → artifacts/*.pdf
```

---

## 🔄 Pipeline Execution Guide

### Full Automated Pipeline

**Single Command**:
```bash
make all
```

**What It Does**:
1. ✅ **Lint**: Checks code quality with ruff, black, isort
2. ✅ **Test**: Runs pytest suite
3. 🔧 **Train**: Executes `train_cv.py` → saves models to `models/`
4. 🎯 **Predict**: Executes `predict.py` → generates `artifacts/submission.csv`
5. 📝 **Docs**: Generates Model Card, Business Plan, Video Script
6. 📄 **Export**: Converts documentation to HTML and PDF

**Estimated Runtime**: 10-30 minutes (depends on data size and compute resources)

---

### Manual Step-by-Step Execution

#### Prerequisites Check
```bash
# Verify Python version
python --version  # Should be 3.11+

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Verify Snowflake credentials
cat .env  # Check SF_ACCOUNT, SF_USER, etc.
```

#### Step 1: Install Dependencies
```bash
make install
# OR
pip install -r requirements.txt
pre-commit install
```

#### Step 2: Validate Data Access
```bash
# Run notebook to check Snowflake connection
jupyter notebook notebooks/01_ingest_and_stage_check.ipynb
```

**Expected Output**:
```
✓ Connected to Snowflake account: xyz12345
✓ Found @AI_CHALLENGE_STAGE
✓ Files in stage:
  - water_quality_training_dataset.csv (950 KB)
  - submission_template.csv (15 KB)
  - landsat_features/*.csv (120 files, 45 MB)
  - terraclimate_features/*.csv (200 files, 30 MB)
✓ Training data: 942 rows, 8 columns
```

#### Step 3: Feature Engineering
```bash
# Option A: Run programmatically
python -m src.wqsa.features.landsat_join
python -m src.wqsa.features.terraclimate_join
python -m src.wqsa.features.gold_builder

# Option B: Use notebook
jupyter notebook notebooks/02_join_and_build_gold.ipynb
```

**Expected Output**:
```
INFO: Joining Landsat features...
INFO: 942 samples joined with Landsat (850 non-future, 92 fallback)
INFO: Joining TerraClimate features...
INFO: 942 samples joined with TerraClimate (920 same-month, 22 prev-month)
INFO: Building TRAIN_GOLD... 942 rows, 25 columns
INFO: Building VALID_GOLD... 200 rows, 22 columns
✓ Gold tables created successfully
```

#### Step 4: Train Models
```bash
make train
# OR
python -m src.wqsa.modeling.train_cv
```

**Expected Output**:
```
================================================================================
WATER QUALITY SA - MODEL TRAINING
================================================================================
Training ALKALINITY models...
  Fold 0 - R²: 0.8421
  Fold 1 - R²: 0.8567
  Fold 2 - R²: 0.8312
  Fold 3 - R²: 0.8489
  Fold 4 - R²: 0.8403
  → CV R² (ALKALINITY): 0.8438

Training EC models...
  Fold 0 - R²: 0.9012
  ...
  → CV R² (EC): 0.8975

Training DRP models...
  ...
  → CV R² (DRP): 0.7823

Mean CV R² across targets: 0.8412
✓ Models saved to models/
```

#### Step 5: Generate Predictions
```bash
make predict
# OR
python -m src.wqsa.modeling.predict
```

**Expected Output**:
```
================================================================================
WATER QUALITY SA - PREDICTION & SUBMISSION
================================================================================
Loading VALID_GOLD... 200 rows
Loading models for ALKALINITY... 5 fold models + blender
Loading models for EC... 5 fold models + blender
Loading models for DRP... 5 fold models + blender
Generating predictions...
  ALKALINITY: [45.2, 67.8, 123.4, ...]
  EC: [567.3, 892.1, 234.5, ...]
  DRP: [1.2, 0.8, 3.4, ...]
✓ Submission saved to artifacts/submission.csv (200 rows × 3 columns)
```

#### Step 6: Generate Documentation
```bash
make docs
# OR
python -m src.wqsa.docs.generate_model_card
python -m src.wqsa.docs.generate_business_plan
python -m src.wqsa.docs.generate_video_script
```

**Expected Output**:
```
Generating Model Card...
  ✓ artifacts/MODEL_CARD.md
Generating Business Plan...
  ✓ artifacts/BUSINESS_PLAN_SNAPSHOT.md
Generating Video Script...
  ✓ artifacts/VIDEO_SCRIPT.md
```

#### Step 7: Export to HTML/PDF
```bash
make export
# OR
python -m src.wqsa.docs.export_to_html
python -m src.wqsa.docs.export_to_pdf
```

---

### Running with Bash Script (Linux/Mac/Git Bash)

```bash
chmod +x scripts/run_all.sh
./scripts/run_all.sh
```

**Interactive Prompts**:
- Warns if virtual environment not activated
- Checks for `.env` file
- Asks for confirmation before proceeding

---

### Troubleshooting Pipeline Issues

#### Issue: Snowflake Connection Failed
```
Error: Unable to connect to Snowflake account
```

**Solution**:
1. Verify `.env` file exists and has correct credentials
2. Check network connectivity: `ping <your-account>.snowflakecomputing.com`
3. Verify Snowflake account is active
4. Test credentials with SnowSQL: `snowsql -a <account> -u <user>`

#### Issue: Missing Stage Files
```
FileNotFoundError: @AI_CHALLENGE_STAGE/water_quality_training_dataset.csv not found
```

**Solution**:
```sql
-- Connect to Snowflake and run:
LIST @AI_CHALLENGE_STAGE;

-- Upload missing files:
PUT file://path/to/water_quality_training_dataset.csv @AI_CHALLENGE_STAGE;
```

#### Issue: Feature Engineering Fails
```
ValueError: Missing required columns: NDVI_MEAN_B250
```

**Solution**:
1. Verify Landsat CSV files in stage contain all expected columns
2. Check `config/project.yaml` feature definitions match data
3. Re-run data validation notebook

#### Issue: Training Crashes (Out of Memory)
```
MemoryError: Unable to allocate array
```

**Solution**:
1. Reduce `n_estimators` in `config/project.yaml` (e.g., 300 → 150)
2. Use smaller `max_depth` (e.g., 6 → 4)
3. Process targets sequentially instead of in parallel
4. Upgrade Snowflake warehouse size

#### Issue: Submission Has Wrong Shape
```
AssertionError: Submission must have exactly 200 rows
```

**Solution**:
1. Verify `VALID_GOLD` table has exactly 200 rows
2. Check for filtering errors in prediction code
3. Ensure submission template matches expected format

---

### Performance Optimization Tips

#### 1. **Snowflake Warehouse Sizing**
```python
# In .env file:
SF_WAREHOUSE=COMPUTE_WH_L  # Use Large warehouse for faster queries
```

#### 2. **Parallel Model Training**
```python
# Modify train_cv.py to use joblib parallelization
from joblib import Parallel, delayed

results = Parallel(n_jobs=3)(
    delayed(train_single_target)(target) for target in targets
)
```

#### 3. **Feature Caching**
```python
# Cache expensive Snowflake queries
@lru_cache
def load_train_gold():
    return session.table("TRAIN_GOLD").to_pandas()
```

#### 4. **Incremental Training**
```bash
# Train only one target at a time
python -m src.wqsa.modeling.train_cv --target ALKALINITY
python -m src.wqsa.modeling.train_cv --target EC
python -m src.wqsa.modeling.train_cv --target DRP
```

---

### Monitoring & Logging

#### View Logs
```bash
# Training logs
tail -f logs/train_cv.log

# Prediction logs
tail -f logs/predict.log
```

#### Log Levels
Adjust in `config/project.yaml`:
```yaml
logging:
  level: DEBUG  # Options: DEBUG, INFO, WARNING, ERROR
```

#### Performance Metrics
```python
# After training, check metrics:
import joblib
metadata = joblib.load("models/metadata.pkl")
print(f"Training time: {metadata['train_time_seconds']}")
print(f"Peak memory: {metadata['peak_memory_mb']} MB")
```

---

### Best Practices

1. **Always Run Tests Before Training**
   ```bash
   make test  # Ensures code quality
   ```

2. **Version Control Your Config**
   ```bash
   git add config/project.yaml
   git commit -m "Update feature set for experiment X"
   ```

3. **Validate Submission Format**
   ```python
   import pandas as pd
   sub = pd.read_csv("artifacts/submission.csv")
   assert sub.shape == (200, 3)
   assert list(sub.columns) == ["ALKALINITY", "EC", "DRP"]
   assert sub.isnull().sum().sum() == 0  # No missing values
   ```

4. **Archive Successful Runs**
   ```bash
   mkdir -p runs/run_$(date +%Y%m%d_%H%M%S)
   cp artifacts/submission.csv runs/run_*/
   cp models/*.pkl runs/run_*/
   ```

5. **Document Hyperparameter Changes**
   - Keep notes in `experiments.md`
   - Track CV R² scores for different configurations
   - Use git tags for best models: `git tag v1.0-best-score`

---

*Built with Snowpark 🐍 and XGBoost 🚀*
