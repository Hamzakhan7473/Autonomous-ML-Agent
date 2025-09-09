"""Data preprocessing and feature engineering module."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing steps."""

    handle_missing: str = "auto"  # 'auto', 'drop', 'impute'
    imputation_strategy: str = "auto"  # 'auto', 'mean', 'median', 'mode', 'knn'
    categorical_encoding: str = "auto"  # 'auto', 'onehot', 'label', 'target'
    scaling: str = "auto"  # 'auto', 'standard', 'minmax', 'robust', 'none'
    feature_selection: bool = True
    outlier_detection: bool = True
    datetime_expansion: bool = True
    polynomial_features: bool = False
    polynomial_degree: int = 2


class SmartImputer:
    """Intelligent missing value imputation."""

    def __init__(self, strategy: str = "auto"):
        """Initialize the smart imputer.

        Args:
            strategy: Imputation strategy ('auto', 'mean', 'median', 'mode', 'knn')
        """
        self.strategy = strategy
        self.imputers = {}
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        """Fit imputers for each column type.

        Args:
            X: Input features
            y: Target variable (optional)
        """
        for col in X.columns:
            if X[col].isna().sum() == 0:
                continue

            if X[col].dtype in ["object", "category"]:
                # Categorical imputation
                if self.strategy == "auto":
                    strategy = "most_frequent"
                else:
                    strategy = self.strategy
                self.imputers[col] = SimpleImputer(strategy=strategy)
            else:
                # Numerical imputation
                if self.strategy == "auto":
                    # Choose strategy based on data distribution
                    if X[col].skew() > 2:
                        strategy = "median"  # Use median for skewed data
                    else:
                        strategy = "mean"
                elif self.strategy == "knn":
                    self.imputers[col] = KNNImputer(n_neighbors=5)
                    continue
                else:
                    strategy = self.strategy
                self.imputers[col] = SimpleImputer(strategy=strategy)

            self.imputers[col].fit(X[[col]])

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted imputers.

        Args:
            X: Input features

        Returns:
            Imputed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before transform")

        X_imputed = X.copy()

        for col, imputer in self.imputers.items():
            if isinstance(imputer, KNNImputer):
                # KNN imputer needs all numerical columns
                numerical_cols = X.select_dtypes(include=[np.number]).columns
                X_numerical = X[numerical_cols]
                X_imputed_numerical = pd.DataFrame(
                    imputer.transform(X_numerical),
                    columns=numerical_cols,
                    index=X.index,
                )
                X_imputed[numerical_cols] = X_imputed_numerical
            else:
                X_imputed[col] = imputer.transform(X[[col]]).flatten()

        return X_imputed

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class FeatureEngineer:
    """Advanced feature engineering capabilities."""

    def __init__(self, config: PreprocessingConfig):
        """Initialize feature engineer.

        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.feature_names = []
        self.is_fitted = False

    def create_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from datetime columns.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with datetime features
        """
        df_engineered = df.copy()

        for col in df.columns:
            if df[col].dtype == "datetime64[ns]" or "datetime" in str(df[col].dtype):
                try:
                    df_engineered[f"{col}_year"] = pd.to_datetime(df[col]).dt.year
                    df_engineered[f"{col}_month"] = pd.to_datetime(df[col]).dt.month
                    df_engineered[f"{col}_day"] = pd.to_datetime(df[col]).dt.day
                    df_engineered[f"{col}_dayofweek"] = pd.to_datetime(
                        df[col]
                    ).dt.dayofweek
                    df_engineered[f"{col}_hour"] = pd.to_datetime(df[col]).dt.hour

                    # Remove original datetime column
                    df_engineered = df_engineered.drop(columns=[col])

                    logger.info(f"Created datetime features for {col}")
                except Exception as e:
                    logger.warning(f"Could not create datetime features for {col}: {e}")

        return df_engineered

    def create_interaction_features(
        self, df: pd.DataFrame, max_features: int = 10
    ) -> pd.DataFrame:
        """Create interaction features between numerical columns.

        Args:
            df: Input DataFrame
            max_features: Maximum number of interaction features to create

        Returns:
            DataFrame with interaction features
        """
        df_engineered = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numerical_cols) < 2:
            return df_engineered

        # Create pairwise interactions
        interaction_count = 0
        for i, col1 in enumerate(numerical_cols):
            for col2 in numerical_cols[i + 1 :]:
                if interaction_count >= max_features:
                    break

                # Multiplication interaction
                df_engineered[f"{col1}_x_{col2}"] = df[col1] * df[col2]
                interaction_count += 1

                if interaction_count >= max_features:
                    break

        return df_engineered

    def remove_low_variance_features(
        self, df: pd.DataFrame, threshold: float = 0.01
    ) -> pd.DataFrame:
        """Remove features with low variance.

        Args:
            df: Input DataFrame
            threshold: Variance threshold

        Returns:
            DataFrame with low variance features removed
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        variances = df[numerical_cols].var()

        low_variance_cols = variances[variances < threshold].index.tolist()

        if low_variance_cols:
            logger.info(f"Removing low variance features: {low_variance_cols}")
            df = df.drop(columns=low_variance_cols)

        return df

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        """Fit the feature engineer."""
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data with feature engineering.

        Args:
            X: Input DataFrame

        Returns:
            Engineered DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted before transform")

        df_engineered = X.copy()

        # Datetime feature extraction
        if self.config.datetime_expansion:
            df_engineered = self.create_datetime_features(df_engineered)

        # Interaction features
        if self.config.polynomial_features:
            df_engineered = self.create_interaction_features(df_engineered)

        # Remove low variance features
        if self.config.feature_selection:
            df_engineered = self.remove_low_variance_features(df_engineered)

        self.feature_names = df_engineered.columns.tolist()
        return df_engineered

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class DataPreprocessor:
    """Main data preprocessing pipeline."""

    def __init__(self, config: PreprocessingConfig | None = None):
        """Initialize the preprocessor.

        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        self.imputer = SmartImputer(strategy=self.config.imputation_strategy)
        self.feature_engineer = FeatureEngineer(self.config)
        self.scaler = None
        self.encoders = {}
        self.is_fitted = False
        self.feature_names = []
        self.categorical_columns = []
        self.numerical_columns = []

    def _detect_column_types(self, X: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Detect categorical and numerical columns.

        Args:
            X: Input DataFrame

        Returns:
            Tuple of (categorical_columns, numerical_columns)
        """
        categorical_cols = []
        numerical_cols = []

        for col in X.columns:
            if X[col].dtype == "object" or X[col].dtype.name == "category":
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)

        return categorical_cols, numerical_cols

    def _setup_encoders(self, X: pd.DataFrame):
        """Setup encoders for categorical variables.

        Args:
            X: Input DataFrame
        """
        for col in self.categorical_columns:
            if self.config.categorical_encoding == "auto":
                # Auto-select encoding based on cardinality
                unique_count = X[col].nunique()
                if unique_count <= 10:
                    encoding = "onehot"
                else:
                    encoding = "label"
            else:
                encoding = self.config.categorical_encoding

            if encoding == "onehot":
                self.encoders[col] = OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, drop="first"
                )
            else:  # label encoding
                self.encoders[col] = LabelEncoder()

    def _setup_scaler(self, X: pd.DataFrame):
        """Setup scaler for numerical variables.

        Args:
            X: Input DataFrame
        """
        if self.config.scaling == "auto":
            # Auto-select scaling based on data distribution
            if len(self.numerical_columns) > 0:
                numerical_data = X[self.numerical_columns]
                if numerical_data.std().max() > 100:  # High variance
                    scaling = "standard"
                else:
                    scaling = "minmax"
            else:
                scaling = "none"
        else:
            scaling = self.config.scaling

        if scaling == "standard":
            self.scaler = StandardScaler()
        elif scaling == "minmax":
            from sklearn.preprocessing import MinMaxScaler

            self.scaler = MinMaxScaler()
        elif scaling == "robust":
            from sklearn.preprocessing import RobustScaler

            self.scaler = RobustScaler()
        else:
            self.scaler = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        """Fit the preprocessing pipeline.

        Args:
            X: Input features
            y: Target variable (optional)
        """
        # Detect column types
        self.categorical_columns, self.numerical_columns = self._detect_column_types(X)

        # Setup encoders and scaler
        self._setup_encoders(X)
        self._setup_scaler(X)

        # Fit imputer
        self.imputer.fit(X, y)

        # Fit feature engineer
        self.feature_engineer.fit(X, y)

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data through the preprocessing pipeline.

        Args:
            X: Input features

        Returns:
            Preprocessed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Step 1: Impute missing values
        X_processed = self.imputer.transform(X)

        # Step 2: Feature engineering
        X_processed = self.feature_engineer.transform(X_processed)

        # Step 3: Encode categorical variables
        for col in self.categorical_columns:
            if col in X_processed.columns and col in self.encoders:
                encoder = self.encoders[col]

                if isinstance(encoder, OneHotEncoder):
                    # One-hot encoding
                    encoded = encoder.transform(X_processed[[col]])
                    encoded_df = pd.DataFrame(
                        encoded,
                        columns=[
                            f"{col}_{cat}" for cat in encoder.categories_[0][1:]
                        ],  # Skip first category
                        index=X_processed.index,
                    )
                    X_processed = pd.concat(
                        [X_processed.drop(columns=[col]), encoded_df], axis=1
                    )
                else:
                    # Label encoding
                    X_processed[col] = encoder.transform(X_processed[col])

        # Step 4: Scale numerical variables
        if self.scaler is not None and len(self.numerical_columns) > 0:
            # Get numerical columns that still exist after feature engineering
            existing_numerical = [
                col for col in self.numerical_columns if col in X_processed.columns
            ]
            if existing_numerical:
                X_processed[existing_numerical] = self.scaler.transform(
                    X_processed[existing_numerical]
                )

        self.feature_names = X_processed.columns.tolist()
        return X_processed

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_feature_names(self) -> list[str]:
        """Get feature names after preprocessing."""
        return self.feature_names.copy()


def preprocess_data(
    X: pd.DataFrame,
    y: pd.Series | None = None,
    config: PreprocessingConfig | None = None,
) -> tuple[pd.DataFrame, DataPreprocessor]:
    """Convenience function to preprocess data.

    Args:
        X: Input features
        y: Target variable (optional)
        config: Preprocessing configuration

    Returns:
        Tuple of (preprocessed_data, fitted_preprocessor)
    """
    preprocessor = DataPreprocessor(config)
    X_processed = preprocessor.fit_transform(X, y)
    return X_processed, preprocessor
