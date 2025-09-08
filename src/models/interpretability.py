"""
Model Interpretability Module

This module provides model interpretation and explainability features
for the autonomous ML agent.
"""

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class ModelInterpreter:
    """Model interpretation and explainability utilities."""

    def __init__(self, model: BaseEstimator, feature_names: list[str] | None = None):
        """
        Initialize model interpreter.

        Args:
            model: Trained model to interpret
            feature_names: Names of features
        """
        self.model = model
        self.feature_names = feature_names
        self.feature_importance = None

    def get_feature_importance(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> dict[str, float]:
        """
        Get feature importance from the model.

        Args:
            X: Feature matrix
            y: Target vector (optional)

        Returns:
            Dictionary of feature importance scores
        """
        if hasattr(self.model, "feature_importances_"):
            # Tree-based models
            importance_scores = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            # Linear models
            importance_scores = np.abs(self.model.coef_)
            if len(importance_scores.shape) > 1:
                importance_scores = np.mean(importance_scores, axis=0)
        else:
            # Fallback to permutation importance
            importance_scores = self._calculate_permutation_importance(X, y)

        # Create feature names if not provided
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(len(importance_scores))]

        # Create importance dictionary
        self.feature_importance = dict(
            zip(self.feature_names, importance_scores, strict=False)
        )

        return self.feature_importance

    def _calculate_permutation_importance(
        self, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Calculate permutation importance."""
        from sklearn.inspection import permutation_importance

        try:
            result = permutation_importance(
                self.model, X, y, n_repeats=10, random_state=42, n_jobs=-1
            )
            return result.importances_mean
        except Exception:
            # Fallback to random importance
            logger.warning(
                "Could not calculate permutation importance, using random values"
            )
            return np.random.random(X.shape[1])

    def explain_prediction(self, X: np.ndarray, sample_idx: int = 0) -> dict[str, Any]:
        """
        Explain a single prediction.

        Args:
            X: Feature matrix
            sample_idx: Index of sample to explain

        Returns:
            Dictionary with explanation
        """
        if sample_idx >= len(X):
            raise ValueError(f"Sample index {sample_idx} out of range")

        sample = X[sample_idx : sample_idx + 1]
        prediction = self.model.predict(sample)[0]

        explanation = {
            "sample_index": sample_idx,
            "prediction": prediction,
            "feature_values": {},
        }

        # Add feature values
        if self.feature_names:
            for i, feature_name in enumerate(self.feature_names):
                if i < sample.shape[1]:
                    explanation["feature_values"][feature_name] = float(sample[0, i])
        else:
            for i in range(sample.shape[1]):
                explanation["feature_values"][f"feature_{i}"] = float(sample[0, i])

        # Add feature importance if available
        if self.feature_importance:
            explanation["feature_importance"] = self.feature_importance

        return explanation

    def get_global_explanation(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> dict[str, Any]:
        """
        Get global model explanation.

        Args:
            X: Feature matrix
            y: Target vector (optional)

        Returns:
            Dictionary with global explanation
        """
        # Get feature importance
        feature_importance = self.get_feature_importance(X, y)

        # Get model predictions
        predictions = self.model.predict(X)

        explanation = {
            "model_type": type(self.model).__name__,
            "feature_importance": feature_importance,
            "top_features": self._get_top_features(feature_importance, top_k=10),
            "prediction_stats": {
                "mean": float(np.mean(predictions)),
                "std": float(np.std(predictions)),
                "min": float(np.min(predictions)),
                "max": float(np.max(predictions)),
            },
        }

        # Add model-specific information
        if hasattr(self.model, "n_estimators"):
            explanation["n_estimators"] = self.model.n_estimators

        if hasattr(self.model, "max_depth"):
            explanation["max_depth"] = self.model.max_depth

        return explanation

    def _get_top_features(
        self, feature_importance: dict[str, float], top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Get top k most important features."""
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_features[:top_k]

    def generate_insights(self, X: np.ndarray, y: np.ndarray = None) -> str:
        """
        Generate human-readable insights about the model.

        Args:
            X: Feature matrix
            y: Target vector (optional)

        Returns:
            String with insights
        """
        # Get feature importance
        feature_importance = self.get_feature_importance(X, y)
        top_features = self._get_top_features(feature_importance, top_k=5)

        insights = []
        insights.append(
            f"The model is a {type(self.model).__name__} trained on {X.shape[0]} samples with {X.shape[1]} features."
        )

        if top_features:
            insights.append("The most important features are:")
            for i, (feature, importance) in enumerate(top_features, 1):
                insights.append(f"  {i}. {feature} (importance: {importance:.4f})")

        # Add model-specific insights
        if hasattr(self.model, "n_estimators"):
            insights.append(f"The model uses {self.model.n_estimators} estimators.")

        if hasattr(self.model, "max_depth"):
            insights.append(f"The maximum depth is {self.model.max_depth}.")

        return "\n".join(insights)


class SHAPExplainer:
    """SHAP-based model explainer."""

    def __init__(self, model: BaseEstimator, feature_names: list[str] | None = None):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained model to explain
            feature_names: Names of features
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None

    def create_explainer(self, X: np.ndarray):
        """Create SHAP explainer."""
        try:
            import shap

            # Choose appropriate explainer based on model type
            if hasattr(self.model, "predict_proba"):
                # Classification model
                self.explainer = (
                    shap.TreeExplainer(self.model)
                    if hasattr(self.model, "feature_importances_")
                    else shap.KernelExplainer(self.model.predict_proba, X[:100])
                )
            else:
                # Regression model
                self.explainer = (
                    shap.TreeExplainer(self.model)
                    if hasattr(self.model, "feature_importances_")
                    else shap.KernelExplainer(self.model.predict, X[:100])
                )

            logger.info("SHAP explainer created successfully")
        except ImportError:
            logger.warning("SHAP not available, falling back to basic interpretation")
            self.explainer = None
        except Exception as e:
            logger.warning(f"Could not create SHAP explainer: {e}")
            self.explainer = None

    def explain_sample(self, X: np.ndarray, sample_idx: int = 0) -> dict[str, Any]:
        """
        Explain a single sample using SHAP.

        Args:
            X: Feature matrix
            sample_idx: Index of sample to explain

        Returns:
            Dictionary with SHAP explanation
        """
        if self.explainer is None:
            return {"error": "SHAP explainer not available"}

        try:

            sample = X[sample_idx : sample_idx + 1]
            shap_values = self.explainer.shap_values(sample)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Take first class for classification

            explanation = {
                "sample_index": sample_idx,
                "shap_values": shap_values[0].tolist(),
                "base_value": (
                    float(self.explainer.expected_value)
                    if hasattr(self.explainer, "expected_value")
                    else 0.0
                ),
            }

            if self.feature_names:
                explanation["feature_names"] = self.feature_names

            return explanation
        except Exception as e:
            return {"error": f"SHAP explanation failed: {e}"}

    def get_feature_importance(self, X: np.ndarray) -> dict[str, float]:
        """
        Get feature importance using SHAP.

        Args:
            X: Feature matrix

        Returns:
            Dictionary of feature importance scores
        """
        if self.explainer is None:
            return {}

        try:

            # Use a subset for efficiency
            X_subset = X[: min(100, len(X))]
            shap_values = self.explainer.shap_values(X_subset)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Take first class for classification

            # Calculate mean absolute SHAP values
            mean_shap = np.mean(np.abs(shap_values), axis=0)

            # Create feature names if not provided
            if self.feature_names is None:
                self.feature_names = [f"feature_{i}" for i in range(len(mean_shap))]

            return dict(zip(self.feature_names, mean_shap, strict=False))
        except Exception as e:
            logger.warning(f"SHAP feature importance failed: {e}")
            return {}


class LimeExplainer:
    """LIME-based model explainer."""

    def __init__(self, model: BaseEstimator, feature_names: list[str] | None = None):
        """
        Initialize LIME explainer.

        Args:
            model: Trained model to explain
            feature_names: Names of features
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None

    def create_explainer(self, X: np.ndarray):
        """Create LIME explainer."""
        try:
            from lime.lime_tabular import LimeTabularExplainer

            self.explainer = LimeTabularExplainer(
                X,
                feature_names=self.feature_names,
                class_names=["class"] if hasattr(self.model, "predict_proba") else None,
                mode=(
                    "classification"
                    if hasattr(self.model, "predict_proba")
                    else "regression"
                ),
            )
            logger.info("LIME explainer created successfully")
        except ImportError:
            logger.warning("LIME not available")
            self.explainer = None
        except Exception as e:
            logger.warning(f"Could not create LIME explainer: {e}")
            self.explainer = None

    def explain_sample(
        self, X: np.ndarray, sample_idx: int = 0, num_features: int = 10
    ) -> dict[str, Any]:
        """
        Explain a single sample using LIME.

        Args:
            X: Feature matrix
            sample_idx: Index of sample to explain
            num_features: Number of features to include in explanation

        Returns:
            Dictionary with LIME explanation
        """
        if self.explainer is None:
            return {"error": "LIME explainer not available"}

        try:
            sample = X[sample_idx]
            explanation = self.explainer.explain_instance(
                sample,
                (
                    self.model.predict_proba
                    if hasattr(self.model, "predict_proba")
                    else self.model.predict
                ),
                num_features=num_features,
            )

            # Extract explanation data
            exp_list = explanation.as_list()

            return {
                "sample_index": sample_idx,
                "explanation": exp_list,
                "prediction": float(
                    self.model.predict(X[sample_idx : sample_idx + 1])[0]
                ),
            }
        except Exception as e:
            return {"error": f"LIME explanation failed: {e}"}


class ModelInsightsGenerator:
    """Generate comprehensive model insights."""

    def __init__(self, model: BaseEstimator, feature_names: list[str] | None = None):
        """
        Initialize insights generator.

        Args:
            model: Trained model
            feature_names: Names of features
        """
        self.model = model
        self.feature_names = feature_names
        self.interpreter = ModelInterpreter(model, feature_names)

    def generate_comprehensive_insights(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> str:
        """
        Generate comprehensive model insights.

        Args:
            X: Feature matrix
            y: Target vector (optional)

        Returns:
            String with comprehensive insights
        """
        insights = []

        # Basic model information
        insights.append("🤖 MODEL ANALYSIS")
        insights.append("=" * 50)
        insights.append(f"Model Type: {type(self.model).__name__}")
        insights.append(f"Dataset Size: {X.shape[0]} samples, {X.shape[1]} features")

        # Feature importance
        feature_importance = self.interpreter.get_feature_importance(X, y)
        if feature_importance:
            insights.append("\n🔍 FEATURE IMPORTANCE")
            insights.append("-" * 30)
            top_features = self.interpreter._get_top_features(
                feature_importance, top_k=10
            )
            for i, (feature, importance) in enumerate(top_features, 1):
                insights.append(f"{i:2d}. {feature:<20} | {importance:.4f}")

        # Model performance insights
        if y is not None:
            predictions = self.model.predict(X)
            insights.append("\n📊 PERFORMANCE INSIGHTS")
            insights.append("-" * 30)

            if hasattr(self.model, "predict_proba"):
                # Classification
                from sklearn.metrics import accuracy_score

                accuracy = accuracy_score(y, predictions)
                insights.append(f"Accuracy: {accuracy:.4f}")

                # Class distribution
                unique_classes, counts = np.unique(y, return_counts=True)
                insights.append(
                    f"Class Distribution: {dict(zip(unique_classes, counts, strict=False))}"
                )
            else:
                # Regression
                from sklearn.metrics import mean_squared_error, r2_score

                r2 = r2_score(y, predictions)
                rmse = np.sqrt(mean_squared_error(y, predictions))
                insights.append(f"R² Score: {r2:.4f}")
                insights.append(f"RMSE: {rmse:.4f}")

        # Model-specific insights
        insights.append("\n⚙️ MODEL CONFIGURATION")
        insights.append("-" * 30)

        model_params = self.model.get_params()
        important_params = ["n_estimators", "max_depth", "learning_rate", "C", "kernel"]

        for param in important_params:
            if param in model_params:
                insights.append(f"{param}: {model_params[param]}")

        # Recommendations
        insights.append("\n💡 RECOMMENDATIONS")
        insights.append("-" * 30)

        if feature_importance:
            top_feature = top_features[0][0] if top_features else None
            if top_feature:
                insights.append(
                    f"• Focus on {top_feature} as it's the most important feature"
                )

        if hasattr(self.model, "n_estimators") and self.model.n_estimators < 200:
            insights.append("• Consider increasing n_estimators for better performance")

        if (
            hasattr(self.model, "max_depth")
            and self.model.max_depth is not None
            and self.model.max_depth > 10
        ):
            insights.append("• Consider reducing max_depth to prevent overfitting")

        return "\n".join(insights)
