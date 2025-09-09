"""
Models Module

This module contains various machine learning models and model management
functionality for the autonomous ML agent.
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for managing machine learning models."""

    def __init__(self):
        self.models = {}
        self.model_configs = {}

    def register_model(self, name: str, model: BaseEstimator, config: dict = None):
        """Register a model in the registry."""
        self.models[name] = model
        if config:
            self.model_configs[name] = config
        logger.info(f"Registered model: {name}")

    def get_model(self, name: str) -> BaseEstimator | None:
        """Get a model from the registry."""
        return self.models.get(name)

    def list_models(self) -> list[str]:
        """List all registered models."""
        return list(self.models.keys())

    def remove_model(self, name: str):
        """Remove a model from the registry."""
        if name in self.models:
            del self.models[name]
            if name in self.model_configs:
                del self.model_configs[name]
            logger.info(f"Removed model: {name}")


class ModelFactory:
    """Factory for creating machine learning models."""

    @staticmethod
    def create_classification_models(
        random_state: int = 42,
    ) -> dict[str, BaseEstimator]:
        """Create a dictionary of classification models."""
        models = {
            "random_forest": RandomForestClassifier(
                n_estimators=100, random_state=random_state, n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100, random_state=random_state
            ),
            "extra_trees": ExtraTreesClassifier(
                n_estimators=100, random_state=random_state, n_jobs=-1
            ),
            "logistic_regression": LogisticRegression(
                random_state=random_state, max_iter=1000
            ),
            "svm": SVC(random_state=random_state, probability=True),
            "knn": KNeighborsClassifier(n_neighbors=5),
            "decision_tree": DecisionTreeClassifier(random_state=random_state),
            "naive_bayes": GaussianNB(),
            "mlp": MLPClassifier(
                hidden_layer_sizes=(100, 50), random_state=random_state, max_iter=1000
            ),
            "lda": LinearDiscriminantAnalysis(),
            "qda": QuadraticDiscriminantAnalysis(),
            "xgboost": XGBClassifier(
                n_estimators=100, random_state=random_state, eval_metric="logloss"
            ),
            "lightgbm": LGBMClassifier(
                n_estimators=100, random_state=random_state, verbose=-1
            ),
            "catboost": CatBoostClassifier(
                iterations=100, random_state=random_state, verbose=False
            ),
        }
        return models

    @staticmethod
    def create_regression_models(random_state: int = 42) -> dict[str, BaseEstimator]:
        """Create a dictionary of regression models."""
        models = {
            "random_forest": RandomForestRegressor(
                n_estimators=100, random_state=random_state, n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100, random_state=random_state
            ),
            "extra_trees": ExtraTreesRegressor(
                n_estimators=100, random_state=random_state, n_jobs=-1
            ),
            "linear_regression": LinearRegression(),
            "ridge": Ridge(random_state=random_state),
            "lasso": Lasso(random_state=random_state),
            "elastic_net": ElasticNet(random_state=random_state),
            "svr": SVR(),
            "knn": KNeighborsRegressor(n_neighbors=5),
            "decision_tree": DecisionTreeRegressor(random_state=random_state),
            "mlp": MLPRegressor(
                hidden_layer_sizes=(100, 50), random_state=random_state, max_iter=1000
            ),
            "xgboost": XGBRegressor(n_estimators=100, random_state=random_state),
            "lightgbm": LGBMRegressor(
                n_estimators=100, random_state=random_state, verbose=-1
            ),
            "catboost": CatBoostRegressor(
                iterations=100, random_state=random_state, verbose=False
            ),
        }
        return models


class EnsembleModel:
    """Ensemble model wrapper."""

    def __init__(
        self,
        models: dict[str, BaseEstimator],
        ensemble_method: str = "voting",
        weights: list[float] | None = None,
    ):
        """
        Initialize ensemble model.

        Args:
            models: Dictionary of models to ensemble
            ensemble_method: Method for ensembling ('voting', 'stacking', 'bagging')
            weights: Weights for voting ensemble
        """
        self.models = models
        self.ensemble_method = ensemble_method
        self.weights = weights
        self.ensemble = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the ensemble model."""
        if self.ensemble_method == "voting":
            # Determine if classification or regression
            if hasattr(list(self.models.values())[0], "predict_proba"):
                # Classification
                self.ensemble = VotingClassifier(
                    estimators=[(name, model) for name, model in self.models.items()],
                    voting="soft",
                    weights=self.weights,
                )
            else:
                # Regression
                self.ensemble = VotingRegressor(
                    estimators=[(name, model) for name, model in self.models.items()],
                    weights=self.weights,
                )

        self.ensemble.fit(X, y)
        self.is_fitted = True
        logger.info(f"Fitted ensemble model with {len(self.models)} models")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.ensemble.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (for classification)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if hasattr(self.ensemble, "predict_proba"):
            return self.ensemble.predict_proba(X)
        else:
            raise ValueError("Model does not support predict_proba")


class ModelEvaluator:
    """Model evaluation utilities."""

    @staticmethod
    def evaluate_classification_model(
        model: BaseEstimator, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict[str, float]:
        """Evaluate a classification model."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1": f1_score(y_test, y_pred, average="weighted"),
        }

        # ROC AUC for binary classification
        if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
            except Exception:
                metrics["roc_auc"] = None

        return metrics

    @staticmethod
    def evaluate_regression_model(
        model: BaseEstimator, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict[str, float]:
        """Evaluate a regression model."""
        from sklearn.metrics import (
            mean_absolute_error,
            mean_absolute_percentage_error,
            mean_squared_error,
            r2_score,
        )

        y_pred = model.predict(X_test)

        metrics = {
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }

        # MAPE (avoid division by zero)
        try:
            mape = mean_absolute_percentage_error(y_test, y_pred)
            metrics["mape"] = mape
        except Exception:
            metrics["mape"] = None

        return metrics


class ModelPersistence:
    """Model persistence utilities."""

    @staticmethod
    def save_model(model: BaseEstimator, filepath: str):
        """Save a model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        logger.info(f"Saved model to {filepath}")

    @staticmethod
    def load_model(filepath: str) -> BaseEstimator:
        """Load a model from disk."""
        model = joblib.load(filepath)
        logger.info(f"Loaded model from {filepath}")
        return model

    @staticmethod
    def save_model_metadata(metadata: dict, filepath: str):
        """Save model metadata to disk."""
        import json

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved model metadata to {filepath}")

    @staticmethod
    def load_model_metadata(filepath: str) -> dict:
        """Load model metadata from disk."""
        import json

        with open(filepath) as f:
            metadata = json.load(f)
        logger.info(f"Loaded model metadata from {filepath}")
        return metadata


class HyperparameterOptimizer:
    """Hyperparameter optimization utilities."""

    def __init__(
        self,
        model: BaseEstimator,
        param_grid: dict[str, list],
        cv: int = 5,
        scoring: str = "accuracy",
        n_jobs: int = -1,
    ):
        """
        Initialize hyperparameter optimizer.

        Args:
            model: Base model to optimize
            param_grid: Parameter grid for optimization
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of jobs for parallel processing
        """
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.best_model = None
        self.best_params = None
        self.best_score = None

    def optimize(self, X: np.ndarray, y: np.ndarray, method: str = "grid"):
        """
        Optimize hyperparameters.

        Args:
            X: Training features
            y: Training targets
            method: Optimization method ('grid', 'random', 'bayesian')
        """
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

        if method == "grid":
            search = GridSearchCV(
                self.model,
                self.param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=1,
            )
        elif method == "random":
            search = RandomizedSearchCV(
                self.model,
                self.param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=1,
                n_iter=20,  # Number of iterations for random search
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        search.fit(X, y)

        self.best_model = search.best_estimator_
        self.best_params = search.best_params_
        self.best_score = search.best_score_

        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best score: {self.best_score:.4f}")

        return self.best_model


class ModelComparison:
    """Model comparison utilities."""

    @staticmethod
    def compare_models(
        models: dict[str, BaseEstimator],
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = "accuracy",
    ) -> pd.DataFrame:
        """
        Compare multiple models using cross-validation.

        Args:
            models: Dictionary of models to compare
            X: Features
            y: Targets
            cv: Number of cross-validation folds
            scoring: Scoring metric

        Returns:
            DataFrame with comparison results
        """
        from sklearn.model_selection import cross_val_score

        results = []

        for name, model in models.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                results.append(
                    {
                        "model": name,
                        "mean_score": scores.mean(),
                        "std_score": scores.std(),
                        "min_score": scores.min(),
                        "max_score": scores.max(),
                    }
                )
                logger.info(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            except Exception as e:
                logger.warning(f"Failed to evaluate {name}: {e}")
                results.append(
                    {
                        "model": name,
                        "mean_score": None,
                        "std_score": None,
                        "min_score": None,
                        "max_score": None,
                    }
                )

        return pd.DataFrame(results)

    @staticmethod
    def create_leaderboard(results_df: pd.DataFrame) -> pd.DataFrame:
        """Create a leaderboard from comparison results."""
        if "mean_score" in results_df.columns:
            return results_df.sort_values("mean_score", ascending=False)
        return results_df


# Predefined parameter grids for common models
PARAM_GRIDS = {
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "gradient_boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 0.9, 1.0],
    },
    "logistic_regression": {
        "C": [0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
    },
    "svm": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    },
    "xgboost": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 0.9, 1.0],
    },
    "lightgbm": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "num_leaves": [31, 50, 100],
    },
}
