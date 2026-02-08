# Data Directory

This directory is **not** used for storing actual data files. All data is managed in Snowflake stages.

## Snowflake Stage Layout

Data should be uploaded to the configured Snowflake stage:

```
@AI_CHALLENGE_STAGE/
├── water_quality_training_dataset.csv
├── submission_template.csv
├── landsat_features/
│   ├── station_001_landsat.parquet
│   ├── station_002_landsat.parquet
│   └── ...
└── terraclimate_features/
    ├── 2020-01_terraclimate.parquet
    ├── 2020-02_terraclimate.parquet
    └── ...
```

## Local Data (ignored by git)

If you need to work with local copies:
- Place files here: `data/raw/` or `data/processed/`
- These directories are in `.gitignore`
- **Never commit actual data to version control**

## Data Sources

All data must be from public/open sources:
- **Landsat L2**: USGS Earth Explorer or Google Earth Engine
- **TerraClimate**: TerraClimate.org monthly climate grid
- **Training data**: Provided by challenge organizers

## Uploading to Snowflake Stage

```sql
-- From Snowflake SQL
USE DATABASE AI_CHALLENGE_DB;
USE SCHEMA PUBLIC;

-- Create stage if needed
CREATE STAGE IF NOT EXISTS AI_CHALLENGE_STAGE;

-- Upload files (from SnowSQL or web UI)
PUT file://path/to/water_quality_training_dataset.csv @AI_CHALLENGE_STAGE/ AUTO_COMPRESS=FALSE;
PUT file://path/to/submission_template.csv @AI_CHALLENGE_STAGE/ AUTO_COMPRESS=FALSE;
PUT file://path/to/landsat_features/* @AI_CHALLENGE_STAGE/landsat_features/ AUTO_COMPRESS=FALSE;
PUT file://path/to/terraclimate_features/* @AI_CHALLENGE_STAGE/terraclimate_features/ AUTO_COMPRESS=FALSE;

-- List stage contents
LIST @AI_CHALLENGE_STAGE;
```

## Python Staging Helper

```python
from src.wqsa.io.staging_io import put_file_to_stage
from src.wqsa.io.snowflake_session import create_snowpark_session

session = create_snowpark_session()
put_file_to_stage(
    session, 
    "data/raw/training.csv",
    "@AI_CHALLENGE_STAGE/",
    overwrite=True
)
```
