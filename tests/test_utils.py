"""Test utility functions."""

import pytest
from datetime import datetime

from src.wqsa.utils.dates import (
    parse_date,
    date_to_month_str,
    days_between,
    encode_month_cyclical,
)
from src.wqsa.utils.validation import (
    validate_dataframe_columns,
    validate_no_nulls,
    validate_value_range,
    validate_submission_format,
)

import pandas as pd
import numpy as np


class TestDateUtils:
    """Test date utility functions."""
    
    def test_parse_date(self):
        """Test date parsing."""
        date = parse_date("2023-06-15")
        assert isinstance(date, datetime)
        assert date.year == 2023
        assert date.month == 6
        assert date.day == 15
    
    def test_date_to_month_str(self):
        """Test month string conversion."""
        date = datetime(2023, 6, 15)
        month_str = date_to_month_str(date)
        assert month_str == "2023-06"
    
    def test_days_between(self):
        """Test days calculation."""
        date1 = datetime(2023, 6, 20)
        date2 = datetime(2023, 6, 10)
        assert days_between(date1, date2) == 10
        assert days_between(date2, date1) == -10
    
    def test_encode_month_cyclical(self):
        """Test cyclical month encoding."""
        date_jan = datetime(2023, 1, 1)
        date_jul = datetime(2023, 7, 1)
        
        sin_jan, cos_jan = encode_month_cyclical(date_jan)
        sin_jul, cos_jul = encode_month_cyclical(date_jul)
        
        # Both should be floats between -1 and 1
        assert -1 <= sin_jan <= 1
        assert -1 <= cos_jan <= 1
        assert -1 <= sin_jul <= 1
        assert -1 <= cos_jul <= 1


class TestValidationUtils:
    """Test validation utility functions."""
    
    def test_validate_dataframe_columns_success(self):
        """Test successful column validation."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9]
        })
        
        assert validate_dataframe_columns(df, ["col1", "col2"]) is True
    
    def test_validate_dataframe_columns_missing(self):
        """Test column validation with missing columns."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6]
        })
        
        with pytest.raises(ValueError, match="missing required columns"):
            validate_dataframe_columns(df, ["col1", "col3"])
    
    def test_validate_no_nulls_success(self):
        """Test successful null validation."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6]
        })
        
        assert validate_no_nulls(df) is True
    
    def test_validate_no_nulls_failure(self):
        """Test null validation with nulls present."""
        df = pd.DataFrame({
            "col1": [1, None, 3],
            "col2": [4, 5, 6]
        })
        
        with pytest.raises(ValueError, match="contains null values"):
            validate_no_nulls(df)
    
    def test_validate_value_range_success(self):
        """Test successful value range validation."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        
        assert validate_value_range(series, min_val=0.0, max_val=10.0) is True
    
    def test_validate_value_range_below_min(self):
        """Test value range validation with values below minimum."""
        series = pd.Series([-1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="below minimum"):
            validate_value_range(series, min_val=0.0)
    
    def test_validate_value_range_above_max(self):
        """Test value range validation with values above maximum."""
        series = pd.Series([1.0, 2.0, 11.0])
        
        with pytest.raises(ValueError, match="above maximum"):
            validate_value_range(series, max_val=10.0)
    
    def test_validate_submission_format_success(self):
        """Test successful submission validation."""
        df = pd.DataFrame({
            "ALKALINITY": np.random.rand(200),
            "EC": np.random.rand(200),
            "DRP": np.random.rand(200)
        })
        
        assert validate_submission_format(df) is True
    
    def test_validate_submission_format_wrong_columns(self):
        """Test submission validation with wrong columns."""
        df = pd.DataFrame({
            "ALKALINITY": np.random.rand(200),
            "WRONG": np.random.rand(200),
            "DRP": np.random.rand(200)
        })
        
        with pytest.raises(ValueError, match="Invalid columns"):
            validate_submission_format(df)
    
    def test_validate_submission_format_wrong_row_count(self):
        """Test submission validation with wrong row count."""
        df = pd.DataFrame({
            "ALKALINITY": np.random.rand(100),
            "EC": np.random.rand(100),
            "DRP": np.random.rand(100)
        })
        
        with pytest.raises(ValueError, match="Invalid row count"):
            validate_submission_format(df)
    
    def test_validate_submission_format_with_nulls(self):
        """Test submission validation with null values."""
        df = pd.DataFrame({
            "ALKALINITY": [1.0, None] + list(np.random.rand(198)),
            "EC": list(np.random.rand(200)),
            "DRP": list(np.random.rand(200))
        })
        
        with pytest.raises(ValueError, match="null values"):
            validate_submission_format(df)
