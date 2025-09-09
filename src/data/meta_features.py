"""
Meta-Features Module

This module extracts dataset characteristics (meta-features) for meta-learning
in the autonomous ML agent.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

logger = logging.getLogger(__name__)


class MetaFeatureExtractor:
    """Extract meta-features from datasets for meta-learning."""

    def __init__(self, target_column: str):
        """
        Initialize meta-feature extractor.

        Args:
            target_column: Name of the target column
        """
        self.target_column = target_column
        self.meta_features = {}

    def extract_all_features(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Extract all meta-features from the dataset.

        Args:
            df: Input dataframe

        Returns:
            Dictionary of meta-features
        """
        self.meta_features = {}

        # Basic features
        self._extract_basic_features(df)

        # Statistical features
        self._extract_statistical_features(df)

        # Information theoretic features
        self._extract_information_theoretic_features(df)

        # Complexity features
        self._extract_complexity_features(df)

        # Feature-based features
        self._extract_feature_based_features(df)

        return self.meta_features

    def _extract_basic_features(self, df: pd.DataFrame):
        """Extract basic dataset features."""
        features = df.drop(columns=[self.target_column])
        target = df[self.target_column]

        self.meta_features.update(
            {
                "num_instances": len(df),
                "num_features": len(features.columns),
                "num_classes": target.nunique() if target.dtype == "object" else None,
                "missing_values_ratio": df.isnull().sum().sum()
                / (len(df) * len(df.columns)),
                "categorical_features_ratio": len(
                    features.select_dtypes(include=["object", "category"]).columns
                )
                / len(features.columns),
                "numerical_features_ratio": len(
                    features.select_dtypes(include=[np.number]).columns
                )
                / len(features.columns),
            }
        )

    def _extract_statistical_features(self, df: pd.DataFrame):
        """Extract statistical features."""
        features = df.drop(columns=[self.target_column])
        numerical_features = features.select_dtypes(include=[np.number])

        if len(numerical_features.columns) > 0:
            # Statistical measures for numerical features
            means = numerical_features.mean()
            stds = numerical_features.std()
            skews = numerical_features.skew()
            kurtoses = numerical_features.kurtosis()

            self.meta_features.update(
                {
                    "mean_mean": means.mean(),
                    "mean_std": means.std(),
                    "std_mean": stds.mean(),
                    "std_std": stds.std(),
                    "skew_mean": skews.mean(),
                    "skew_std": skews.std(),
                    "kurtosis_mean": kurtoses.mean(),
                    "kurtosis_std": kurtoses.std(),
                }
            )
        else:
            # Default values for datasets without numerical features
            self.meta_features.update(
                {
                    "mean_mean": 0.0,
                    "mean_std": 0.0,
                    "std_mean": 0.0,
                    "std_std": 0.0,
                    "skew_mean": 0.0,
                    "skew_std": 0.0,
                    "kurtosis_mean": 0.0,
                    "kurtosis_std": 0.0,
                }
            )

    def _extract_information_theoretic_features(self, df: pd.DataFrame):
        """Extract information theoretic features."""
        features = df.drop(columns=[self.target_column])
        target = df[self.target_column]

        # Entropy of target variable
        if target.dtype == "object":
            target_entropy = self._calculate_entropy(target)
            self.meta_features["target_entropy"] = target_entropy
        else:
            # For regression, use variance as a proxy for entropy
            self.meta_features["target_entropy"] = target.var()

        # Mutual information between features and target
        numerical_features = features.select_dtypes(include=[np.number])
        if len(numerical_features.columns) > 0:
            try:
                if target.dtype == "object":
                    mi_scores = mutual_info_classif(
                        numerical_features, target, random_state=42
                    )
                else:
                    mi_scores = mutual_info_regression(
                        numerical_features, target, random_state=42
                    )

                self.meta_features.update(
                    {
                        "mutual_info_mean": mi_scores.mean(),
                        "mutual_info_std": mi_scores.std(),
                        "mutual_info_max": mi_scores.max(),
                        "mutual_info_min": mi_scores.min(),
                    }
                )
            except Exception:
                self.meta_features.update(
                    {
                        "mutual_info_mean": 0.0,
                        "mutual_info_std": 0.0,
                        "mutual_info_max": 0.0,
                        "mutual_info_min": 0.0,
                    }
                )
        else:
            self.meta_features.update(
                {
                    "mutual_info_mean": 0.0,
                    "mutual_info_std": 0.0,
                    "mutual_info_max": 0.0,
                    "mutual_info_min": 0.0,
                }
            )

    def _extract_complexity_features(self, df: pd.DataFrame):
        """Extract complexity features."""
        features = df.drop(columns=[self.target_column])
        target = df[self.target_column]

        # Fisher's discriminant ratio
        if len(features.select_dtypes(include=[np.number]).columns) > 0:
            try:
                fisher_score = self._calculate_fisher_score(features, target)
                self.meta_features["fisher_score"] = fisher_score
            except Exception:
                self.meta_features["fisher_score"] = 0.0
        else:
            self.meta_features["fisher_score"] = 0.0

        # Class imbalance (for classification)
        if target.dtype == "object":
            class_counts = target.value_counts()
            imbalance_ratio = class_counts.max() / class_counts.min()
            self.meta_features["class_imbalance_ratio"] = imbalance_ratio
        else:
            self.meta_features["class_imbalance_ratio"] = 1.0

        # Feature correlation
        numerical_features = features.select_dtypes(include=[np.number])
        if len(numerical_features.columns) > 1:
            corr_matrix = numerical_features.corr().abs()
            # Remove diagonal elements
            corr_values = corr_matrix.values[
                np.triu_indices_from(corr_matrix.values, k=1)
            ]
            self.meta_features.update(
                {
                    "feature_correlation_mean": corr_values.mean(),
                    "feature_correlation_std": corr_values.std(),
                    "feature_correlation_max": corr_values.max(),
                }
            )
        else:
            self.meta_features.update(
                {
                    "feature_correlation_mean": 0.0,
                    "feature_correlation_std": 0.0,
                    "feature_correlation_max": 0.0,
                }
            )

    def _extract_feature_based_features(self, df: pd.DataFrame):
        """Extract feature-based complexity features."""
        features = df.drop(columns=[self.target_column])

        # Number of features with zero variance
        zero_var_features = features.columns[features.var() == 0]
        self.meta_features["zero_variance_features_ratio"] = len(
            zero_var_features
        ) / len(features.columns)

        # Number of features with low variance
        low_var_features = features.columns[features.var() < 0.01]
        self.meta_features["low_variance_features_ratio"] = len(low_var_features) / len(
            features.columns
        )

        # Feature sparsity
        numerical_features = features.select_dtypes(include=[np.number])
        if len(numerical_features.columns) > 0:
            sparsity = (numerical_features == 0).sum().sum() / (
                len(numerical_features) * len(numerical_features.columns)
            )
            self.meta_features["feature_sparsity"] = sparsity
        else:
            self.meta_features["feature_sparsity"] = 0.0

    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculate entropy of a categorical series."""
        value_counts = series.value_counts()
        probabilities = value_counts / len(series)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def _calculate_fisher_score(
        self, features: pd.DataFrame, target: pd.Series
    ) -> float:
        """Calculate Fisher's discriminant ratio."""
        numerical_features = features.select_dtypes(include=[np.number])
        if len(numerical_features.columns) == 0:
            return 0.0

        # For simplicity, use the first numerical feature
        feature = numerical_features.iloc[:, 0]

        if target.dtype == "object":
            # Classification case
            classes = target.unique()
            if len(classes) != 2:
                return 0.0

            class1_mask = target == classes[0]
            class2_mask = target == classes[1]

            mean1 = feature[class1_mask].mean()
            mean2 = feature[class2_mask].mean()
            var1 = feature[class1_mask].var()
            var2 = feature[class2_mask].var()

            if var1 + var2 == 0:
                return 0.0

            fisher_score = (mean1 - mean2) ** 2 / (var1 + var2)
            return fisher_score
        else:
            # Regression case - use correlation as proxy
            correlation = feature.corr(target)
            return abs(correlation) if not pd.isna(correlation) else 0.0


class DatasetProfiler:
    """Profile datasets to understand their characteristics."""

    def __init__(self, target_column: str):
        """
        Initialize dataset profiler.

        Args:
            target_column: Name of the target column
        """
        self.target_column = target_column
        self.meta_extractor = MetaFeatureExtractor(target_column)

    def profile_dataset(self, df: pd.DataFrame) -> dict:
        """
        Create a comprehensive profile of the dataset.

        Args:
            df: Input dataframe

        Returns:
            Dictionary containing dataset profile
        """
        profile = {
            "basic_info": self._get_basic_info(df),
            "target_analysis": self._analyze_target(df),
            "feature_analysis": self._analyze_features(df),
            "data_quality": self._assess_data_quality(df),
            "meta_features": self.meta_extractor.extract_all_features(df),
        }

        return profile

    def _get_basic_info(self, df: pd.DataFrame) -> dict:
        """Get basic dataset information."""
        return {
            "shape": df.shape,
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "dtypes": df.dtypes.value_counts().to_dict(),
        }

    def _analyze_target(self, df: pd.DataFrame) -> dict:
        """Analyze the target variable."""
        target = df[self.target_column]

        analysis = {
            "dtype": str(target.dtype),
            "missing_values": target.isnull().sum(),
            "missing_ratio": target.isnull().sum() / len(target),
        }

        if target.dtype == "object":
            # Classification
            analysis.update(
                {
                    "type": "classification",
                    "num_classes": target.nunique(),
                    "class_distribution": target.value_counts().to_dict(),
                    "most_common_class": (
                        target.mode()[0] if len(target.mode()) > 0 else None
                    ),
                }
            )
        else:
            # Regression
            analysis.update(
                {
                    "type": "regression",
                    "min": target.min(),
                    "max": target.max(),
                    "mean": target.mean(),
                    "std": target.std(),
                    "skewness": target.skew(),
                    "kurtosis": target.kurtosis(),
                }
            )

        return analysis

    def _analyze_features(self, df: pd.DataFrame) -> dict:
        """Analyze the features."""
        features = df.drop(columns=[self.target_column])

        numerical_features = features.select_dtypes(include=[np.number])
        categorical_features = features.select_dtypes(include=["object", "category"])

        analysis = {
            "total_features": len(features.columns),
            "numerical_features": len(numerical_features.columns),
            "categorical_features": len(categorical_features.columns),
            "numerical_features_list": numerical_features.columns.tolist(),
            "categorical_features_list": categorical_features.columns.tolist(),
        }

        # Analyze numerical features
        if len(numerical_features.columns) > 0:
            analysis["numerical_stats"] = {
                "mean": numerical_features.mean().to_dict(),
                "std": numerical_features.std().to_dict(),
                "min": numerical_features.min().to_dict(),
                "max": numerical_features.max().to_dict(),
                "skewness": numerical_features.skew().to_dict(),
                "kurtosis": numerical_features.kurtosis().to_dict(),
            }

        # Analyze categorical features
        if len(categorical_features.columns) > 0:
            cat_stats = {}
            for col in categorical_features.columns:
                cat_stats[col] = {
                    "unique_values": categorical_features[col].nunique(),
                    "most_common": (
                        categorical_features[col].mode()[0]
                        if len(categorical_features[col].mode()) > 0
                        else None
                    ),
                    "missing_values": categorical_features[col].isnull().sum(),
                }
            analysis["categorical_stats"] = cat_stats

        return analysis

    def _assess_data_quality(self, df: pd.DataFrame) -> dict:
        """Assess data quality."""
        quality = {
            "missing_values": df.isnull().sum().to_dict(),
            "missing_values_ratio": (df.isnull().sum() / len(df)).to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "duplicate_ratio": df.duplicated().sum() / len(df),
        }

        # Check for outliers in numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}

        for col in numerical_cols:
            if col != self.target_column:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_count = (
                    (df[col] < lower_bound) | (df[col] > upper_bound)
                ).sum()
                outliers[col] = {
                    "count": outlier_count,
                    "ratio": outlier_count / len(df),
                }

        quality["outliers"] = outliers

        return quality


class MetaLearningFeatures:
    """Generate features for meta-learning algorithms."""

    def __init__(self, target_column: str):
        """
        Initialize meta-learning features generator.

        Args:
            target_column: Name of the target column
        """
        self.target_column = target_column
        self.meta_extractor = MetaFeatureExtractor(target_column)

    def generate_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate meta-learning features from dataset.

        Args:
            df: Input dataframe

        Returns:
            Array of meta-learning features
        """
        meta_features = self.meta_extractor.extract_all_features(df)

        # Convert to array, handling None values
        feature_values = []
        feature_names = []

        for name, value in meta_features.items():
            if value is not None:
                feature_values.append(float(value))
                feature_names.append(name)

        return np.array(feature_values)

    def get_feature_names(self) -> list[str]:
        """Get the names of meta-learning features."""
        return [
            "num_instances",
            "num_features",
            "num_classes",
            "missing_values_ratio",
            "categorical_features_ratio",
            "numerical_features_ratio",
            "mean_mean",
            "mean_std",
            "std_mean",
            "std_std",
            "skew_mean",
            "skew_std",
            "kurtosis_mean",
            "kurtosis_std",
            "target_entropy",
            "mutual_info_mean",
            "mutual_info_std",
            "mutual_info_max",
            "mutual_info_min",
            "fisher_score",
            "class_imbalance_ratio",
            "feature_correlation_mean",
            "feature_correlation_std",
            "feature_correlation_max",
            "zero_variance_features_ratio",
            "low_variance_features_ratio",
            "feature_sparsity",
        ]
