"""
Snowflake data warehouse integration for water quality ML pipeline.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import os

try:
    import snowflake.connector
    from snowflake.connector.pandas_tools import write_pandas, pd_writer
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False
    print("Warning: snowflake-connector-python not installed")


class SnowflakeConnection:
    """
    Snowflake connection manager for water quality data pipeline.
    """

    def __init__(
        self,
        account: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        warehouse: str = 'COMPUTE_WH',
        database: str = 'WATER_QUALITY_DB',
        schema: str = 'PUBLIC',
        role: str = 'SYSADMIN'
    ):
        """
        Initialize Snowflake connection.

        Args:
            account: Snowflake account identifier
            user: Snowflake username
            password: Snowflake password
            warehouse: Warehouse name
            database: Database name
            schema: Schema name
            role: Role name
        """
        if not SNOWFLAKE_AVAILABLE:
            raise ImportError("snowflake-connector-python is not installed")

        # Load credentials from environment if not provided
        self.account = account or os.getenv('SNOWFLAKE_ACCOUNT')
        self.user = user or os.getenv('SNOWFLAKE_USER')
        self.password = password or os.getenv('SNOWFLAKE_PASSWORD')
        self.warehouse = warehouse
        self.database = database
        self.schema = schema
        self.role = role

        self.conn = None
        self.cursor = None

    def connect(self) -> None:
        """Establish connection to Snowflake."""
        if not all([self.account, self.user, self.password]):
            raise ValueError(
                "Missing Snowflake credentials. Set SNOWFLAKE_ACCOUNT, "
                "SNOWFLAKE_USER, and SNOWFLAKE_PASSWORD environment variables."
            )

        try:
            self.conn = snowflake.connector.connect(
                account=self.account,
                user=self.user,
                password=self.password,
                warehouse=self.warehouse,
                database=self.database,
                schema=self.schema,
                role=self.role
            )
            self.cursor = self.conn.cursor()
            print(f"Connected to Snowflake: {self.database}.{self.schema}")

        except Exception as e:
            print(f"Error connecting to Snowflake: {str(e)}")
            raise

    def disconnect(self) -> None:
        """Close Snowflake connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("Disconnected from Snowflake")

    def execute_query(self, query: str, fetch: bool = True) -> Optional[pd.DataFrame]:
        """
        Execute SQL query.

        Args:
            query: SQL query string
            fetch: Whether to fetch results

        Returns:
            DataFrame with results if fetch=True, None otherwise
        """
        if not self.conn:
            self.connect()

        try:
            self.cursor.execute(query)

            if fetch:
                columns = [col[0] for col in self.cursor.description]
                data = self.cursor.fetchall()
                df = pd.DataFrame(data, columns=columns)
                print(f"Query returned {len(df)} rows")
                return df
            else:
                print("Query executed successfully")
                return None

        except Exception as e:
            print(f"Error executing query: {str(e)}")
            raise

    def create_table(
        self,
        table_name: str,
        schema_dict: Dict[str, str],
        replace: bool = False
    ) -> None:
        """
        Create table in Snowflake.

        Args:
            table_name: Name of table to create
            schema_dict: Dictionary mapping column names to data types
            replace: Whether to replace existing table
        """
        if not self.conn:
            self.connect()

        # Build column definitions
        columns_sql = ", ".join([f"{col} {dtype}" for col, dtype in schema_dict.items()])

        # Create table SQL
        create_sql = f"CREATE {'OR REPLACE' if replace else ''} TABLE {table_name} ({columns_sql})"

        try:
            self.cursor.execute(create_sql)
            print(f"Table {table_name} created successfully")
        except Exception as e:
            print(f"Error creating table: {str(e)}")
            raise

    def upload_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        auto_create_table: bool = True,
        overwrite: bool = False
    ) -> None:
        """
        Upload pandas DataFrame to Snowflake table.

        Args:
            df: DataFrame to upload
            table_name: Target table name
            auto_create_table: Whether to auto-create table if it doesn't exist
            overwrite: Whether to truncate table before inserting
        """
        if not self.conn:
            self.connect()

        try:
            if overwrite:
                self.cursor.execute(f"TRUNCATE TABLE IF EXISTS {table_name}")

            # Use Snowflake's write_pandas utility
            success, nchunks, nrows, _ = write_pandas(
                conn=self.conn,
                df=df,
                table_name=table_name,
                auto_create_table=auto_create_table,
                overwrite=overwrite
            )

            if success:
                print(f"Uploaded {nrows} rows to {table_name} in {nchunks} chunks")
            else:
                print(f"Upload to {table_name} failed")

        except Exception as e:
            print(f"Error uploading DataFrame: {str(e)}")
            raise

    def download_table(self, table_name: str) -> pd.DataFrame:
        """
        Download entire table as DataFrame.

        Args:
            table_name: Name of table to download

        Returns:
            DataFrame with table contents
        """
        query = f"SELECT * FROM {table_name}"
        return self.execute_query(query)

    def create_feature_view(
        self,
        view_name: str,
        base_table: str,
        feature_transformations: Optional[str] = None
    ) -> None:
        """
        Create SQL view with feature transformations.

        Args:
            view_name: Name of view to create
            base_table: Base table name
            feature_transformations: SQL feature transformation logic
        """
        if feature_transformations is None:
            feature_transformations = "*"

        create_view_sql = f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT {feature_transformations}
        FROM {base_table}
        """

        self.execute_query(create_view_sql, fetch=False)
        print(f"View {view_name} created successfully")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


# Example SQL feature transformations for Snowflake
SQL_FEATURE_TRANSFORMS = """
    *,
    -- Temporal features
    EXTRACT(YEAR FROM date) AS year,
    EXTRACT(MONTH FROM date) AS month,
    EXTRACT(DAY FROM date) AS day,
    DAYOFYEAR(date) AS day_of_year,
    WEEKOFYEAR(date) AS week_of_year,
    QUARTER(date) AS quarter,

    -- Landsat indices
    (B5 - B4) / NULLIF((B5 + B4), 0) AS NDVI,
    (B3 - B5) / NULLIF((B3 + B5), 0) AS NDWI,
    (B5 - B7) / NULLIF((B5 + B7), 0) AS NBR,

    -- Climate features
    tmax - tmin AS temp_range,
    ppt / NULLIF(pet, 0) AS aridity_index,

    -- Spatial features
    SQRT(latitude * latitude + longitude * longitude) AS distance_from_origin
"""


def upload_training_data_to_snowflake(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    connection_params: Optional[Dict[str, Any]] = None
) -> None:
    """
    Upload training and test data to Snowflake.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        connection_params: Snowflake connection parameters
    """
    if connection_params is None:
        connection_params = {}

    with SnowflakeConnection(**connection_params) as sf:
        # Upload training data
        sf.upload_dataframe(
            train_df,
            table_name='WATER_QUALITY_TRAIN',
            auto_create_table=True,
            overwrite=True
        )

        # Upload test data
        sf.upload_dataframe(
            test_df,
            table_name='WATER_QUALITY_TEST',
            auto_create_table=True,
            overwrite=True
        )

        # Create feature view
        sf.create_feature_view(
            view_name='WATER_QUALITY_FEATURES',
            base_table='WATER_QUALITY_TRAIN',
            feature_transformations=SQL_FEATURE_TRANSFORMS
        )

    print("Data upload to Snowflake completed")


def fetch_features_from_snowflake(
    view_name: str = 'WATER_QUALITY_FEATURES',
    connection_params: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Fetch engineered features from Snowflake view.

    Args:
        view_name: Name of feature view
        connection_params: Snowflake connection parameters

    Returns:
        DataFrame with features
    """
    if connection_params is None:
        connection_params = {}

    with SnowflakeConnection(**connection_params) as sf:
        df = sf.download_table(view_name)

    return df


def run_snowflake_aggregations(
    connection_params: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Run complex aggregations in Snowflake and return results.

    Args:
        connection_params: Snowflake connection parameters

    Returns:
        DataFrame with aggregated features
    """
    if connection_params is None:
        connection_params = {}

    aggregation_query = """
    SELECT
        spatial_cluster,
        AVG(target) AS avg_target_by_cluster,
        STDDEV(target) AS std_target_by_cluster,
        COUNT(*) AS count_by_cluster,
        AVG(NDVI) AS avg_ndvi_by_cluster,
        AVG(ppt) AS avg_precipitation_by_cluster
    FROM WATER_QUALITY_FEATURES
    GROUP BY spatial_cluster
    """

    with SnowflakeConnection(**connection_params) as sf:
        agg_df = sf.execute_query(aggregation_query)

    return agg_df
