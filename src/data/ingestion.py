"""
Data Ingestion Module for Autonomous Machine Learning Agent

This module handles data loading, validation, and initial analysis.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from pydantic import BaseModel, validator

logger = logging.getLogger(__name__)


class DatasetInfo(BaseModel):
    """Information about a dataset"""
    shape: Tuple[int, int]
    columns: List[str]
    target_column: str
    target_type: str
    target_distribution: Dict[str, int]
    feature_types: Dict[str, str]
    missing_values: Dict[str, int]
    memory_usage: float
    file_size: float
    
    @validator('shape')
    def validate_shape(cls, v):
        if v[0] == 0 or v[1] == 0:
            raise ValueError("Dataset cannot be empty")
        return v
    
    @validator('target_column')
    def validate_target_column(cls, v, values):
        if 'columns' in values and v not in values['columns']:
            raise ValueError(f"Target column '{v}' not found in dataset")
        return v


class DataIngestion:
    """
    Handles data loading, validation, and initial analysis
    """
    
    def __init__(self):
        self.supported_formats = ['.csv', '.parquet', '.xlsx', '.json']
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from various file formats
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_path.suffix.lower() == '.xlsx':
                df = pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            logger.info(f"Successfully loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame, target_column: str) -> DatasetInfo:
        """
        Validate dataset and extract information
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            Dataset information
        """
        # Basic validation
        if df.empty:
            raise ValueError("Dataset is empty")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Extract information
        info = DatasetInfo(
            shape=df.shape,
            columns=list(df.columns),
            target_column=target_column,
            target_type=str(df[target_column].dtype),
            target_distribution=df[target_column].value_counts().to_dict(),
            feature_types=self._get_feature_types(df, target_column),
            missing_values=df.isnull().sum().to_dict(),
            memory_usage=df.memory_usage(deep=True).sum() / 1024**2,  # MB
            file_size=0.0  # Will be set if file path is available
        )
        
        logger.info(f"Dataset validation completed: {info.shape[0]} rows, {info.shape[1]} columns")
        return info
    
    def _get_feature_types(self, df: pd.DataFrame, target_column: str) -> Dict[str, str]:
        """
        Categorize feature types
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            Dictionary mapping column names to feature types
        """
        feature_types = {}
        
        for col in df.columns:
            if col == target_column:
                continue
            
            dtype = df[col].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                if df[col].nunique() <= 10:
                    feature_types[col] = 'categorical_numeric'
                else:
                    feature_types[col] = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                feature_types[col] = 'datetime'
            elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                feature_types[col] = 'categorical'
            else:
                feature_types[col] = 'other'
        
        return feature_types
    
    def analyze_data_quality(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Analyze data quality issues
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            Data quality analysis
        """
        quality_report = {
            'missing_values': {},
            'duplicates': 0,
            'outliers': {},
            'data_types': {},
            'unique_values': {},
            'quality_score': 0.0
        }
        
        # Missing values analysis
        missing_counts = df.isnull().sum()
        quality_report['missing_values'] = missing_counts[missing_counts > 0].to_dict()
        
        # Duplicate analysis
        quality_report['duplicates'] = df.duplicated().sum()
        
        # Data type analysis
        quality_report['data_types'] = df.dtypes.to_dict()
        
        # Unique values analysis
        for col in df.columns:
            if col != target_column:
                quality_report['unique_values'][col] = df[col].nunique()
        
        # Outlier analysis for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != target_column:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers > 0:
                    quality_report['outliers'][col] = outliers
        
        # Calculate quality score
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = quality_report['duplicates']
        
        quality_score = 1.0 - (missing_cells + duplicate_rows) / total_cells
        quality_report['quality_score'] = max(0.0, quality_score)
        
        return quality_report
    
    def get_data_summary(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Get comprehensive data summary
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            Data summary
        """
        summary = {
            'basic_info': {
                'rows': df.shape[0],
                'columns': df.shape[1],
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
            },
            'target_info': {
                'column': target_column,
                'type': str(df[target_column].dtype),
                'unique_values': df[target_column].nunique(),
                'distribution': df[target_column].value_counts().to_dict(),
                'missing_count': df[target_column].isnull().sum()
            },
            'feature_info': {
                'numeric_features': list(df.select_dtypes(include=[np.number]).columns),
                'categorical_features': list(df.select_dtypes(include=['object']).columns),
                'datetime_features': list(df.select_dtypes(include=['datetime']).columns)
            },
            'quality_metrics': self.analyze_data_quality(df, target_column)
        }
        
        return summary
    
    def detect_data_issues(self, df: pd.DataFrame, target_column: str) -> List[str]:
        """
        Detect potential data issues
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            List of detected issues
        """
        issues = []
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        high_missing_cols = missing_counts[missing_counts > df.shape[0] * 0.5]
        if not high_missing_cols.empty:
            issues.append(f"High missing values (>50%) in columns: {list(high_missing_cols.index)}")
        
        # Check for class imbalance
        target_counts = df[target_column].value_counts()
        if len(target_counts) > 1:
            min_count = target_counts.min()
            max_count = target_counts.max()
            imbalance_ratio = max_count / min_count
            if imbalance_ratio > 10:
                issues.append(f"Severe class imbalance detected (ratio: {imbalance_ratio:.2f})")
        
        # Check for duplicate rows
        if df.duplicated().sum() > 0:
            issues.append(f"Duplicate rows detected: {df.duplicated().sum()}")
        
        # Check for constant columns
        constant_cols = []
        for col in df.columns:
            if col != target_column and df[col].nunique() == 1:
                constant_cols.append(col)
        if constant_cols:
            issues.append(f"Constant columns detected: {constant_cols}")
        
        # Check for high cardinality categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns
        high_cardinality_cols = []
        for col in categorical_cols:
            if col != target_column and df[col].nunique() > 100:
                high_cardinality_cols.append(col)
        if high_cardinality_cols:
            issues.append(f"High cardinality categorical features: {high_cardinality_cols}")
        
        return issues
    
    def suggest_preprocessing_steps(self, df: pd.DataFrame, target_column: str) -> List[str]:
        """
        Suggest preprocessing steps based on data analysis
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            List of suggested preprocessing steps
        """
        suggestions = []
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            suggestions.append("Handle missing values (imputation)")
        
        # Check for categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            suggestions.append("Encode categorical variables")
        
        # Check for numeric features that need scaling
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            suggestions.append("Scale numeric features")
        
        # Check for datetime features
        datetime_cols = df.select_dtypes(include=['datetime']).columns
        if len(datetime_cols) > 0:
            suggestions.append("Extract datetime features")
        
        # Check for outliers
        for col in numeric_cols:
            if col != target_column:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
                if outliers > 0:
                    suggestions.append("Handle outliers")
                    break
        
        # Check for class imbalance
        target_counts = df[target_column].value_counts()
        if len(target_counts) > 1:
            min_count = target_counts.min()
            max_count = target_counts.max()
            if max_count / min_count > 5:
                suggestions.append("Handle class imbalance")
        
        return suggestions
