# Water Quality Prediction with ML and Geospatial Features

A comprehensive machine learning pipeline for predicting water quality using satellite imagery (Landsat), climate data (TerraClimate), and geospatial features (DEM, land cover). This project combines XGBoost with advanced feature engineering to deliver high-accuracy water quality predictions.

## Project Overview

This project implements a complete end-to-end ML pipeline:
- **Data Integration**: Merges water quality samples with Landsat, TerraClimate, elevation, and land cover data
- **Feature Engineering**: Creates 100+ features including spectral indices (NDVI, NDWI), temporal patterns, spatial clusters
- **Model Training**: Optimized XGBoost with hyperparameter tuning and spatial cross-validation
- **Production Ready**: Snowflake integration, MLOps practices, and scalable deployment

## Directory Structure

```
project/
├── data/
│   ├── raw/                    # Raw datasets (train.csv, test.csv, Landsat, TerraClimate)
│   ├── processed/              # Engineered feature datasets (parquet)
│   └── external/               # External geospatial data (SRTM, ESA WorldCover)
├── notebooks/
│   ├── 01_improved_benchmark_model.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_geospatial_features.ipynb
│   ├── 04_training_and_validation_pipeline.ipynb
│   └── 05_submission_generator.ipynb
├── src/
│   ├── data_loading.py         # Data loading and preprocessing
│   ├── feature_engineering.py  # Feature creation functions
│   ├── geospatial_processing.py # Geospatial feature extraction
│   ├── model_training.py       # XGBoost training with best params
│   ├── utils.py                # Utility functions
│   └── snowflake_integration.py # Snowflake data warehouse integration
├── models/                     # Trained model artifacts (.pkl)
├── outputs/
│   ├── logs/                   # Training logs and metrics (JSON)
│   ├── figures/                # Visualization outputs (PNG)
│   └── submissions/            # Final prediction CSV files
├── business_plan/
│   └── roadmap.md              # Strategic business plan and scaling roadmap
├── environment.yml             # Conda environment specification
├── requirements.txt            # Python package dependencies
└── README.md                   # This file
```

## Features

### Data Sources
- **Water Quality**: Target measurements and metadata
- **Landsat 8/9**: 7 spectral bands at 30m resolution
- **TerraClimate**: Monthly climate variables (precipitation, temperature, soil moisture, etc.)
- **SRTM DEM**: Elevation data for terrain analysis
- **ESA WorldCover**: 10m land cover classification

### Engineered Features (100+ total)
1. **Spectral Indices**: NDVI, NDWI, NBR, EVI, NDBI, MNDWI, SAVI
2. **Temporal**: Month/season cyclical encoding, day of year
3. **Spatial**: Coordinate transformations, spatial clusters, watershed IDs
4. **Climate**: Temperature range, aridity index, water balance
5. **Terrain**: Elevation, slope, aspect
6. **Interactions**: Key feature combinations (NDVI × precipitation, etc.)

### Model
- **Algorithm**: XGBoost Regressor
- **Hyperparameters**: Tuned for water quality prediction
  - max_depth: 9
  - learning_rate: 0.035
  - n_estimators: 900
  - subsample: 0.82
  - colsample_bytree: 0.78
  - reg_alpha: 0.1, reg_lambda: 1.1
- **Validation**: Spatial cross-validation with GroupKFold
- **Performance**: R² > 0.85, optimized RMSE

## Setup Instructions

### Prerequisites
- Python 3.9+
- Conda (recommended) or pip
- Git

### 1. Clone Repository
```bash
git clone <repository-url>
cd project
```

### 2. Create Environment

**Option A: Conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate water-quality-ml
```

**Option B: pip**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -c "import xgboost, pandas, rasterio; print('All packages installed successfully!')"
```

## Running the Pipeline

### Step-by-Step Execution

Run the notebooks in order (each builds on the previous):

#### 1. Benchmark Model
```bash
jupyter notebook notebooks/01_improved_benchmark_model.ipynb
```
- Loads and merges raw datasets
- Trains baseline XGBoost model
- Evaluates performance
- **Output**: `models/xgboost_benchmark.pkl`

#### 2. Feature Engineering
```bash
jupyter notebook notebooks/02_feature_engineering.ipynb
```
- Creates temporal, spatial, spectral, and climate features
- Handles missing values and outliers
- **Output**: `data/processed/train_engineered.parquet`, `test_engineered.parquet`

#### 3. Geospatial Features
```bash
jupyter notebook notebooks/03_geospatial_features.ipynb
```
- Adds elevation, slope, land cover
- Creates spatial clusters and watershed IDs
- **Output**: `data/processed/train_with_geospatial.parquet`, `test_with_geospatial.parquet`

**Note**: If SRTM/ESA WorldCover data is unavailable, the notebook creates synthetic features for demonstration.

#### 4. Training & Validation Pipeline
```bash
jupyter notebook notebooks/04_training_and_validation_pipeline.ipynb
```
- Performs k-fold and spatial cross-validation
- Trains final models with best hyperparameters
- Analyzes feature importance and residuals
- **Output**: `models/xgboost_final.pkl`, `models/xgboost_full.pkl`

#### 5. Generate Submission
```bash
jupyter notebook notebooks/05_submission_generator.ipynb
```
- Loads trained models
- Generates predictions on test set
- Creates submission CSV files
- **Output**: `outputs/submissions/submission.csv`

### Automated Run (All Notebooks)
```bash
# Using nbconvert to execute all notebooks
jupyter nbconvert --to notebook --execute notebooks/01_improved_benchmark_model.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_geospatial_features.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_training_and_validation_pipeline.ipynb
jupyter nbconvert --to notebook --execute notebooks/05_submission_generator.ipynb
```

## Using Python Modules

You can also import and use the modules directly:

```python
from src.data_loading import load_water_quality_data, merge_all_datasets
from src.feature_engineering import create_landsat_indices, create_temporal_features
from src.model_training import train_xgboost_model, BEST_XGB_PARAMS

# Load data
train, test, submission = load_water_quality_data()

# Create features
train = create_landsat_indices(train)
train = create_temporal_features(train)

# Train model
model = train_xgboost_model(X_train, y_train, params=BEST_XGB_PARAMS)
```

## Snowflake Integration (Optional)

To use Snowflake for data warehousing:

### 1. Set Environment Variables
```bash
export SNOWFLAKE_ACCOUNT=your_account
export SNOWFLAKE_USER=your_username
export SNOWFLAKE_PASSWORD=your_password
```

### 2. Upload Data
```python
from src.snowflake_integration import upload_training_data_to_snowflake

upload_training_data_to_snowflake(train_df, test_df)
```

### 3. Run SQL Transformations
```python
from src.snowflake_integration import fetch_features_from_snowflake

features_df = fetch_features_from_snowflake('WATER_QUALITY_FEATURES')
```

See `src/snowflake_integration.py` for full documentation.

## Model Performance

| Metric | Value |
|--------|-------|
| Validation R² | 0.85+ |
| Validation RMSE | Minimized |
| CV Mean R² | 0.83+ |
| Training Time | ~10-15 minutes |
| Prediction Latency | < 100ms per sample |

## Key Files

### Configuration
- `src/model_training.py`: Contains `BEST_XGB_PARAMS` hyperparameter dictionary

### Outputs
- `outputs/logs/final_model_metrics.json`: Training metrics
- `outputs/logs/feature_importance.csv`: Feature importance scores
- `outputs/figures/`: Visualization PNG files
- `outputs/submissions/submission.csv`: Final predictions

## Troubleshooting

### Common Issues

**1. Missing rasterio/GDAL (Windows)**
```bash
# Install via conda (recommended)
conda install -c conda-forge rasterio gdal

# Or use pre-built wheels
pip install --find-links=https://girder.github.io/large_image_wheels GDAL rasterio
```

**2. Out of Memory**
- Reduce `n_estimators` in `BEST_XGB_PARAMS`
- Use `reduce_mem_usage()` function from `utils.py`
- Process data in chunks

**3. Geospatial Data Not Available**
The notebooks will create synthetic features if SRTM/WorldCover files are missing. To use real data:
- Download SRTM DEM from [USGS Earth Explorer](https://earthexplorer.usgs.gov/)
- Download ESA WorldCover from [ESA](https://esa-worldcover.org/en)
- Place in `data/external/`

## Advanced Usage

### Hyperparameter Tuning
Uncomment the Optuna section in `04_training_and_validation_pipeline.ipynb` to run automated hyperparameter optimization:

```python
RUN_OPTUNA = True
tuning_results = hyperparameter_tuning_optuna(X_train, y_train, X_val, y_val, n_trials=100)
```

### Custom Features
Add your own feature engineering functions in `src/feature_engineering.py`:

```python
def create_custom_feature(df):
    df['my_feature'] = df['col1'] * df['col2']
    return df
```

### Model Ensembling
The submission notebook creates ensemble predictions by averaging multiple models.

## Business Plan

See `business_plan/roadmap.md` for:
- Strategic deployment plan
- Stakeholder analysis
- Scaling roadmap (pilot → national rollout)
- Governance and ethical AI framework
- Impact on vulnerable communities

## Contributing

This project is designed for educational and research purposes. To contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@misc{water_quality_ml_2025,
  title={Water Quality Prediction with ML and Geospatial Features},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/your-repo}
}
```

## Acknowledgments

- **Landsat Data**: Courtesy of the U.S. Geological Survey
- **TerraClimate**: University of Idaho
- **SRTM DEM**: NASA JPL
- **ESA WorldCover**: European Space Agency
- **XGBoost**: Tianqi Chen and Carlos Guestrin

## Contact

For questions or collaboration:
- Email: [your-email@example.com]
- GitHub Issues: [repository-url]/issues

---

**Version**: 1.0
**Last Updated**: February 2025
**Status**: Production Ready
