"""Staging I/O utilities for Snowflake stages.

Helpers for PUT/GET operations and stage file management.
"""

import logging
from pathlib import Path
from typing import Optional

from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


def put_file_to_stage(
    session: Session,
    local_path: str,
    stage_path: str,
    auto_compress: bool = False,
    overwrite: bool = True,
) -> None:
    """Upload a local file to a Snowflake stage.

    Args:
        session: Active Snowpark session
        local_path: Path to local file
        stage_path: Target stage path (e.g., '@MY_STAGE/folder/')
        auto_compress: Whether to automatically compress the file
        overwrite: Whether to overwrite existing files

    Raises:
        FileNotFoundError: If local file doesn't exist
    """
    local_file = Path(local_path)
    if not local_file.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    compress_clause = "AUTO_COMPRESS=TRUE" if auto_compress else "AUTO_COMPRESS=FALSE"
    overwrite_clause = "OVERWRITE=TRUE" if overwrite else "OVERWRITE=FALSE"

    put_cmd = f"PUT 'file://{local_file.as_posix()}' {stage_path} {compress_clause} {overwrite_clause}"

    logger.info(f"Uploading {local_path} to {stage_path}")
    session.sql(put_cmd).collect()
    logger.info("Upload completed successfully")


def get_file_from_stage(
    session: Session,
    stage_path: str,
    local_dir: str,
    pattern: Optional[str] = None,
) -> None:
    """Download file(s) from a Snowflake stage to local directory.

    Args:
        session: Active Snowpark session
        stage_path: Source stage path (e.g., '@MY_STAGE/file.csv')
        local_dir: Local directory to download to
        pattern: Optional file pattern to match
    """
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    get_cmd = f"GET {stage_path} 'file://{local_path.as_posix()}'"
    if pattern:
        get_cmd += f" PATTERN='{pattern}'"

    logger.info(f"Downloading from {stage_path} to {local_dir}")
    session.sql(get_cmd).collect()
    logger.info("Download completed successfully")


def list_stage_files(session: Session, stage_path: str, pattern: Optional[str] = None) -> list:
    """List files in a Snowflake stage.

    Args:
        session: Active Snowpark session
        stage_path: Stage path to list (e.g., '@MY_STAGE/folder/')
        pattern: Optional pattern to filter files

    Returns:
        List of file names in the stage
    """
    list_cmd = f"LIST {stage_path}"
    if pattern:
        list_cmd += f" PATTERN='{pattern}'"

    logger.info(f"Listing files in {stage_path}")
    result = session.sql(list_cmd).collect()

    files = [row["name"] for row in result]
    logger.info(f"Found {len(files)} files")
    return files
