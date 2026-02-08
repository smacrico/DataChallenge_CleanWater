"""Snowflake session management and configuration.

This module provides utilities for creating and managing Snowpark sessions
from environment variables and ensuring proper database context.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


def create_snowpark_session(
    account: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    role: Optional[str] = None,
    warehouse: Optional[str] = None,
    database: Optional[str] = None,
    schema: Optional[str] = None,
) -> Session:
    """Create a Snowpark session from environment variables or explicit arguments.

    Args:
        account: Snowflake account identifier (defaults to SF_ACCOUNT env var)
        user: Snowflake username (defaults to SF_USER env var)
        password: Snowflake password (defaults to SF_PASSWORD env var)
        role: Snowflake role (defaults to SF_ROLE env var)
        warehouse: Snowflake warehouse (defaults to SF_WAREHOUSE env var)
        database: Snowflake database (defaults to SF_DATABASE env var)
        schema: Snowflake schema (defaults to SF_SCHEMA env var)

    Returns:
        Configured Snowpark Session object

    Raises:
        ValueError: If required connection parameters are missing
    """
    # Load environment variables from .env file
    load_dotenv()

    # Get connection parameters (prefer explicit args, fallback to env vars)
    conn_params = {
        "account": account or os.getenv("SF_ACCOUNT"),
        "user": user or os.getenv("SF_USER"),
        "password": password or os.getenv("SF_PASSWORD"),
        "role": role or os.getenv("SF_ROLE", "SYSADMIN"),
        "warehouse": warehouse or os.getenv("SF_WAREHOUSE", "COMPUTE_WH"),
        "database": database or os.getenv("SF_DATABASE", "AI_CHALLENGE_DB"),
        "schema": schema or os.getenv("SF_SCHEMA", "PUBLIC"),
    }

    # Validate required parameters
    required_params = ["account", "user", "password"]
    missing = [param for param in required_params if not conn_params.get(param)]
    if missing:
        raise ValueError(
            f"Missing required Snowflake connection parameters: {', '.join(missing)}. "
            f"Please set environment variables or pass explicit arguments."
        )

    logger.info(f"Creating Snowpark session for account: {conn_params['account']}")

    # Create session
    session = Session.builder.configs(conn_params).create()

    # Set context with USE statements
    if conn_params["database"]:
        session.sql(f"USE DATABASE {conn_params['database']}").collect()
        logger.info(f"Using database: {conn_params['database']}")

    if conn_params["schema"]:
        session.sql(f"USE SCHEMA {conn_params['schema']}").collect()
        logger.info(f"Using schema: {conn_params['schema']}")

    if conn_params["warehouse"]:
        session.sql(f"USE WAREHOUSE {conn_params['warehouse']}").collect()
        logger.info(f"Using warehouse: {conn_params['warehouse']}")

    logger.info("Snowpark session created successfully")
    return session


def get_stage_path(stage_var: str, default: str = "") -> str:
    """Get Snowflake stage path from environment variable.

    Args:
        stage_var: Environment variable name (e.g., 'STAGE_TRAIN_CSV')
        default: Default value if environment variable is not set

    Returns:
        Stage path string
    """
    load_dotenv()
    return os.getenv(stage_var, default)
