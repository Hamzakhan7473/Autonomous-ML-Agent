"""
Data Preprocessing Module

This module handles data cleaning, feature engineering, and preprocessing
for the autonomous machine learning agent.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class PreprocessingConfig:
    """Configuration class for data preprocessing."""
    
    def __init__(
        self,
        handle_missing: str = "auto",
        encode_categorical: str = "onehot",
        scale_features: str = "standard",
        detect_outliers: bool = True,
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5,
        create_interactions: bool = False,
        create_polynomials: bool = False,
        random_state: int = 42,
    ):
        """
        Initialize preprocessing configuration.
        
        Args:
            handle_missing: Strategy for handling missing values ('auto', 'drop', 'impute')
            encode_categorical: Method for encoding categorical features ('onehot', 'label')
            scale_features: Method for scaling features ('standard', 'minmax', 'robust')
            detect_outliers: Whether to detect outliers
            outlier_method: Method for outlier detection ('iqr', 'zscore')
            outlier_threshold: Threshold for outlier detection
            create_interactions: Whether to create feature interactions
            create_polynomials: Whether to create polynomial features
            random_state: Random state for reproducibility
        """
        self.handle_missing = handle_missing
        self.encode_categorical = encode_categorical
        self.scale_features = scale_features
        self.detect_outliers = detect_outliers
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.create_interactions = create_interactions
        self.create_polynomials = create_polynomials
        self.random_state = random_state


class DataPreprocessor:
    """Main data preprocessing class for the autonomous ML agent."""

    def __init__(self, target_column: str, random_state: int = 42):
        """
        Initialize the data preprocessor.

        Args:
            target_column: Name of the target column
            random_state: Random state for reproducibility
        """
        self.target_column = target_column
        self.random_state = random_state
        self.preprocessing_pipeline = None
        self.feature_names = None
        self.categorical_features = None
        self.numerical_features = None
        self.target_encoder = None

    def analyze_data(self, df: pd.DataFrame) -> dict:
        """
        Analyze the dataset and return insights.

        Args:
            df: Input dataframe

        Returns:
            Dictionary containing data analysis results
        """
        analysis = {
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.to_dict(),
            "numerical_features": [],
            "categorical_features": [],
            "target_info": {},
        }

        # Separate features and target
        features = df.drop(columns=[self.target_column])

        # Identify feature types
        for col in features.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                analysis["numerical_features"].append(col)
            else:
                analysis["categorical_features"].append(col)

        # Target analysis
        target_col = df[self.target_column]
        analysis["target_info"] = {
            "dtype": str(target_col.dtype),
            "unique_values": target_col.nunique(),
            "missing_values": target_col.isnull().sum(),
            "value_counts": target_col.value_counts().to_dict(),
        }

        if pd.api.types.is_numeric_dtype(target_col):
            analysis["target_info"].update(
                {
                    "min": target_col.min(),
                    "max": target_col.max(),
                    "mean": target_col.mean(),
                    "std": target_col.std(),
                }
            )

        return analysis

    def create_preprocessing_pipeline(self, df: pd.DataFrame) -> Pipeline:
        """
        Create a preprocessing pipeline based on data analysis.

        Args:
            df: Input dataframe

        Returns:
            Preprocessing pipeline
        """
        # Analyze data
        analysis = self.analyze_data(df)

        # Store feature information
        self.numerical_features = analysis["numerical_features"]
        self.categorical_features = analysis["categorical_features"]

        # Create transformers
        transformers = []

        # Numerical features transformer
        if self.numerical_features:
            numerical_transformer = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(
                ("numerical", numerical_transformer, self.numerical_features)
            )

        # Categorical features transformer
        if self.categorical_features:
            categorical_transformer = Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="missing"),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )
            transformers.append(
                ("categorical", categorical_transformer, self.categorical_features)
            )

        # Create column transformer
        self.preprocessing_pipeline = ColumnTransformer(
            transformers=transformers, remainder="drop"
        )

        return self.preprocessing_pipeline

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit the preprocessing pipeline and transform the data.

        Args:
            df: Input dataframe

        Returns:
            Tuple of (X_transformed, y_transformed)
        """
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # Create and fit preprocessing pipeline
        if self.preprocessing_pipeline is None:
            self.create_preprocessing_pipeline(df)

        # Fit and transform features
        X_transformed = self.preprocessing_pipeline.fit_transform(X)

        # Handle target encoding with better error handling
        try:
            if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
                logger.info("Encoding categorical target with LabelEncoder")
                self.target_encoder = LabelEncoder()
                y_transformed = self.target_encoder.fit_transform(y)
            else:
                logger.info("Using numerical target as-is")
                y_transformed = y.values
                
            # Ensure y_transformed is a numpy array
            if not isinstance(y_transformed, np.ndarray):
                y_transformed = np.array(y_transformed)
                
            logger.info(f"Target shape: {y_transformed.shape}, dtype: {y_transformed.dtype}")
            
        except Exception as e:
            logger.error(f"Error in target encoding: {e}")
            # Fallback: try to convert to numeric
            try:
                y_numeric = pd.to_numeric(y, errors='coerce')
                y_transformed = y_numeric.values
                logger.info("Fallback: converted target to numeric")
            except Exception as e2:
                logger.error(f"Fallback target conversion failed: {e2}")
                raise ValueError(f"Could not process target column: {e}")

        # Store feature names
        self.feature_names = self._get_feature_names()

        return X_transformed, y_transformed

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform new data using fitted preprocessing pipeline.

        Args:
            df: Input dataframe

        Returns:
            Tuple of (X_transformed, y_transformed)
        """
        if self.preprocessing_pipeline is None:
            raise ValueError(
                "Preprocessing pipeline not fitted. Call fit_transform first."
            )

        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # Transform features
        X_transformed = self.preprocessing_pipeline.transform(X)

        # Transform target
        if self.target_encoder is not None:
            y_transformed = self.target_encoder.transform(y)
        else:
            y_transformed = y.values

        return X_transformed, y_transformed

    def _get_feature_names(self) -> list[str]:
        """Get feature names after preprocessing."""
        if self.preprocessing_pipeline is None:
            return []

        feature_names = []

        # Get numerical feature names
        if self.numerical_features:
            feature_names.extend(self.numerical_features)

        # Get categorical feature names after one-hot encoding
        if self.categorical_features:
            categorical_transformer = self.preprocessing_pipeline.named_transformers_[
                "categorical"
            ]
            onehot_encoder = categorical_transformer.named_steps["onehot"]
            categorical_feature_names = onehot_encoder.get_feature_names_out(
                self.categorical_features
            )
            feature_names.extend(categorical_feature_names)

        return (
            feature_names.tolist()
            if hasattr(feature_names, "tolist")
            else feature_names
        )

    def get_feature_importance_columns(self) -> list[str]:
        """Get column names for feature importance analysis."""
        return self.feature_names if self.feature_names else []


class SmartImputer(BaseEstimator, TransformerMixin):
    """Smart imputer that chooses the best imputation strategy."""

    def __init__(self, strategy="auto"):
        self.strategy = strategy
        self.imputer = None

    def fit(self, X, y=None):
        """Fit the imputer."""
        if self.strategy == "auto":
            # Choose strategy based on data characteristics
            missing_ratio = np.isnan(X).sum() / len(X)
            if missing_ratio > 0.5:
                strategy = "constant"
            elif X.dtype.kind in "fc":
                strategy = "mean"
            else:
                strategy = "most_frequent"
        else:
            strategy = self.strategy

        self.imputer = SimpleImputer(strategy=strategy)
        self.imputer.fit(X)
        return self

    def transform(self, X):
        """Transform the data."""
        return self.imputer.transform(X)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering transformer."""

    def __init__(self, create_interactions=True, create_polynomials=False):
        self.create_interactions = create_interactions
        self.create_polynomials = create_polynomials
        self.feature_names = None

    def fit(self, X, y=None):
        """Fit the feature engineer."""
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        """Transform the data."""
        X_transformed = X.copy()

        if self.create_interactions and X.shape[1] > 1:
            # Create pairwise interactions for numerical features
            for i in range(X.shape[1]):
                for j in range(i + 1, X.shape[1]):
                    interaction = X[:, i] * X[:, j]
                    X_transformed = np.column_stack([X_transformed, interaction])

        if self.create_polynomials:
            # Create polynomial features (degree 2)
            for i in range(X.shape[1]):
                polynomial = X[:, i] ** 2
                X_transformed = np.column_stack([X_transformed, polynomial])

        return X_transformed


def detect_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "iqr",
    threshold: float = 1.5,
) -> dict:
    """
    Detect outliers in the dataset.

    Args:
        df: Input dataframe
        columns: Columns to check for outliers (None for all numerical)
        method: Method to use ('iqr' or 'zscore')
        threshold: Threshold for outlier detection

    Returns:
        Dictionary with outlier information
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outliers = {}

    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            elif method == "zscore":
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_mask = z_scores > threshold
            else:
                continue

            outliers[col] = {
                "count": outlier_mask.sum(),
                "percentage": (outlier_mask.sum() / len(df)) * 100,
                "indices": df[outlier_mask].index.tolist(),
            }

    return outliers


def handle_missing_values(df: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Args:
        df: Input dataframe
        strategy: Strategy to use ('auto', 'drop', 'impute')

    Returns:
        Dataframe with handled missing values
    """
    if strategy == "auto":
        # Choose strategy based on missing value ratio
        missing_ratio = df.isnull().sum() / len(df)

        if missing_ratio.max() > 0.5:
            # Drop columns with more than 50% missing values
            columns_to_drop = missing_ratio[missing_ratio > 0.5].index
            df = df.drop(columns=columns_to_drop)
            logger.info(
                f"Dropped columns with >50% missing values: {columns_to_drop.tolist()}"
            )

        # Impute remaining missing values
        df = df.fillna(df.median())

    elif strategy == "drop":
        df = df.dropna()

    elif strategy == "impute":
        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns

        # Impute numerical columns with median
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

        # Impute categorical columns with mode
        for col in categorical_cols:
            df[col] = df[col].fillna(
                df[col].mode()[0] if len(df[col].mode()) > 0 else "missing"
            )

    return df


def encode_categorical_features(
    df: pd.DataFrame, method: str = "onehot"
) -> pd.DataFrame:
    """
    Encode categorical features.

    Args:
        df: Input dataframe
        method: Encoding method ('onehot', 'label', 'target')

    Returns:
        Dataframe with encoded categorical features
    """
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    if method == "onehot":
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    elif method == "label":
        for col in categorical_cols:
            df[col] = LabelEncoder().fit_transform(df[col])

    return df


def scale_features(df: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
    """
    Scale numerical features.

    Args:
        df: Input dataframe
        method: Scaling method ('standard', 'minmax', 'robust')

    Returns:
        Dataframe with scaled features
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
    elif method == "robust":
        from sklearn.preprocessing import RobustScaler

        scaler = RobustScaler()
    else:
        return df

    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df
