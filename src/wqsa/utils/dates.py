"""Date and time utilities."""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd


def parse_date(date_str: str, format: str = "%Y-%m-%d") -> datetime:
    """Parse date string to datetime object.

    Args:
        date_str: Date string to parse
        format: Expected date format

    Returns:
        Parsed datetime object
    """
    return datetime.strptime(date_str, format)


def date_to_month_str(date: datetime) -> str:
    """Convert datetime to month string (YYYY-MM).

    Args:
        date: Datetime object

    Returns:
        Month string in YYYY-MM format
    """
    return date.strftime("%Y-%m")


def add_months(date: datetime, months: int) -> datetime:
    """Add or subtract months from a date.

    Args:
        date: Starting date
        months: Number of months to add (negative to subtract)

    Returns:
        New datetime object
    """
    # Simple approximation using 30-day months
    return date + timedelta(days=30 * months)


def days_between(date1: datetime, date2: datetime) -> int:
    """Calculate days between two dates.

    Args:
        date1: First date
        date2: Second date

    Returns:
        Number of days (positive if date1 > date2)
    """
    return (date1 - date2).days


def get_season(date: datetime) -> str:
    """Determine season from date (Southern Hemisphere).

    Args:
        date: Datetime object

    Returns:
        Season name (Summer, Autumn, Winter, Spring)
    """
    month = date.month

    if month in [12, 1, 2]:
        return "Summer"
    elif month in [3, 4, 5]:
        return "Autumn"
    elif month in [6, 7, 8]:
        return "Winter"
    else:
        return "Spring"


def encode_month_cyclical(date: datetime) -> tuple:
    """Encode month as sin/cos features for seasonality.

    Args:
        date: Datetime object

    Returns:
        Tuple of (sin_month, cos_month)
    """
    import math

    month = date.month
    sin_month = math.sin(2 * math.pi * month / 12)
    cos_month = math.cos(2 * math.pi * month / 12)

    return sin_month, cos_month
