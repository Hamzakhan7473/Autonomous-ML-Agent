"""Data ingestion and validation module."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatasetSchema:
    """Schema information for a dataset."""

    n_rows: int
    n_features: int
    n_categorical: int
    n_numerical: int
    n_missing: int
    missing_percentage: float
    target_type: str
    class_balance: dict[str, float] | None = None
    feature_types: dict[str, str] | None = None
    memory_usage_mb: float = 0.0


class DataIngester:
    """Handles data loading, validation, and schema extraction."""

    def __init__(self, random_state: int = 42):
        """Initialize the data ingester.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)

    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from various file formats.

        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments for pandas read functions

        Returns:
            Loaded DataFrame

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine file format and load accordingly
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path, **kwargs)
        elif file_path.suffix.lower() in [".parquet", ".pq"]:
            df = pd.read_parquet(file_path, **kwargs)
        elif file_path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path, **kwargs)
        elif file_path.suffix.lower() == ".json":
            df = pd.read_json(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        logger.info(f"Loaded dataset with shape: {df.shape}")
        return df

    def validate_data(
        self, df: pd.DataFrame, target_column: str
    ) -> tuple[bool, list[str]]:
        """Validate the dataset for ML tasks.

        Args:
            df: Input DataFrame
            target_column: Name of the target column

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check if target column exists
        if target_column not in df.columns:
            issues.append(f"Target column '{target_column}' not found in dataset")
            return False, issues

        # Check for empty dataset
        if df.empty:
            issues.append("Dataset is empty")
            return False, issues

        # Check minimum number of samples
        if len(df) < 10:
            issues.append(f"Dataset has only {len(df)} samples, minimum 10 required")

        # Check for constant target
        if df[target_column].nunique() == 1:
            issues.append("Target column has only one unique value")

        # Check for too many missing values in target
        target_missing = df[target_column].isna().sum()
        if target_missing > len(df) * 0.5:
            issues.append(
                f"Target column has {target_missing} missing values ({target_missing/len(df)*100:.1f}%)"
            )

        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > len(df) * 0.1:
            issues.append(
                f"Dataset has {duplicates} duplicate rows ({duplicates/len(df)*100:.1f}%)"
            )

        is_valid = len(issues) == 0
        return is_valid, issues

    def extract_schema(self, df: pd.DataFrame, target_column: str) -> DatasetSchema:
        """Extract schema information from the dataset.

        Args:
            df: Input DataFrame
            target_column: Name of the target column

        Returns:
            DatasetSchema object with metadata
        """
        # Basic counts
        n_rows, n_features = df.shape
        n_missing = df.isna().sum().sum()
        missing_percentage = (n_missing / (n_rows * n_features)) * 100

        # Feature type analysis
        feature_types = {}
        n_categorical = 0
        n_numerical = 0

        for col in df.columns:
            if col == target_column:
                continue

            if df[col].dtype == "object" or df[col].dtype.name == "category":
                feature_types[col] = "categorical"
                n_categorical += 1
            else:
                feature_types[col] = "numerical"
                n_numerical += 1

        # Target type analysis
        target_series = df[target_column]
        if target_series.dtype == "object" or target_series.dtype.name == "category":
            target_type = "categorical"
            class_balance = target_series.value_counts(normalize=True).to_dict()
        else:
            target_type = "numerical"
            class_balance = None

        # Memory usage
        memory_usage_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

        return DatasetSchema(
            n_rows=n_rows,
            n_features=n_features - 1,  # Exclude target
            n_categorical=n_categorical,
            n_numerical=n_numerical,
            n_missing=n_missing,
            missing_percentage=missing_percentage,
            target_type=target_type,
            class_balance=class_balance,
            feature_types=feature_types,
            memory_usage_mb=memory_usage_mb,
        )

    def get_data_summary(self, df: pd.DataFrame, target_column: str) -> dict[str, Any]:
        """Get comprehensive data summary for LLM consumption.

        Args:
            df: Input DataFrame
            target_column: Name of the target column

        Returns:
            Dictionary with data summary
        """
        schema = self.extract_schema(df, target_column)

        # Statistical summary for numerical features
        numerical_cols = [
            col for col, dtype in schema.feature_types.items() if dtype == "numerical"
        ]
        numerical_summary = {}

        if numerical_cols:
            numerical_summary = df[numerical_cols].describe().to_dict()

        # Categorical summary
        categorical_cols = [
            col for col, dtype in schema.feature_types.items() if dtype == "categorical"
        ]
        categorical_summary = {}

        for col in categorical_cols:
            categorical_summary[col] = {
                "unique_values": df[col].nunique(),
                "most_frequent": (
                    df[col].mode().iloc[0] if not df[col].mode().empty else None
                ),
                "missing_count": df[col].isna().sum(),
            }

        return {
            "schema": schema.__dict__,
            "numerical_summary": numerical_summary,
            "categorical_summary": categorical_summary,
            "target_summary": {
                "type": schema.target_type,
                "unique_values": df[target_column].nunique(),
                "class_balance": schema.class_balance,
                "missing_count": df[target_column].isna().sum(),
            },
            "data_quality": {
                "duplicate_rows": df.duplicated().sum(),
                "constant_features": [
                    col
                    for col in df.columns
                    if col != target_column and df[col].nunique() <= 1
                ],
                "high_missing_features": [
                    col
                    for col in df.columns
                    if col != target_column and df[col].isna().sum() > len(df) * 0.5
                ],
            },
        }


def analyze_data(
    file_path: str, target_column: str, **kwargs
) -> tuple[pd.DataFrame, DatasetSchema, dict[str, Any]]:
    """Convenience function to analyze a dataset.

    Args:
        file_path: Path to the data file
        target_column: Name of the target column
        **kwargs: Additional arguments for data loading

    Returns:
        Tuple of (DataFrame, Schema, Summary)
    """
    ingester = DataIngester()

    # Load data
    df = ingester.load_data(file_path, **kwargs)

    # Validate data
    is_valid, issues = ingester.validate_data(df, target_column)
    if not is_valid:
        logger.warning(f"Data validation issues: {issues}")

    # Extract schema and summary
    schema = ingester.extract_schema(df, target_column)
    summary = ingester.get_data_summary(df, target_column)

    return df, schema, summary
