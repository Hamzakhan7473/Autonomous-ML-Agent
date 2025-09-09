"""
Hyperparameter Optimization Module

This module provides hyperparameter optimization functionality
for the autonomous ML agent.
"""

import logging

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

logger = logging.getLogger(__name__)


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

    def get_best_model(self) -> BaseEstimator | None:
        """Get the best model after optimization."""
        return self.best_model

    def get_best_params(self) -> dict | None:
        """Get the best parameters after optimization."""
        return self.best_params

    def get_best_score(self) -> float | None:
        """Get the best score after optimization."""
        return self.best_score


class OptunaOptimizer:
    """Optuna-based hyperparameter optimizer."""

    def __init__(
        self,
        model: BaseEstimator,
        cv: int = 5,
        scoring: str = "accuracy",
        n_trials: int = 100,
    ):
        """
        Initialize Optuna optimizer.

        Args:
            model: Base model to optimize
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_trials: Number of trials for optimization
        """
        self.model = model
        self.cv = cv
        self.scoring = scoring
        self.n_trials = n_trials
        self.best_model = None
        self.best_params = None
        self.best_score = None

    def optimize(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        """
        Optimize hyperparameters using Optuna.

        Args:
            X: Training features
            y: Training targets

        Returns:
            Best model
        """
        try:
            import optuna

            def objective(trial):
                # Define hyperparameter search space based on model type
                params = self._suggest_params(trial)

                # Create model with suggested parameters
                model = self.model.__class__(**params)

                # Perform cross-validation
                scores = cross_val_score(
                    model, X, y, cv=self.cv, scoring=self.scoring, n_jobs=-1
                )
                return scores.mean()

            # Create study
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.n_trials)

            # Get best parameters
            self.best_params = study.best_params
            self.best_score = study.best_value

            # Create best model
            self.best_model = self.model.__class__(**self.best_params)
            self.best_model.fit(X, y)

            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best score: {self.best_score:.4f}")

            return self.best_model

        except ImportError:
            logger.warning("Optuna not available, falling back to grid search")
            return self._fallback_optimization(X, y)
        except Exception as e:
            logger.warning(f"Optuna optimization failed: {e}")
            return self._fallback_optimization(X, y)

    def _suggest_params(self, trial):
        """Suggest hyperparameters based on model type."""
        model_name = self.model.__class__.__name__.lower()

        if "randomforest" in model_name:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": 42,
            }
        elif "gradientboosting" in model_name:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "random_state": 42,
            }
        elif "logisticregression" in model_name:
            return {
                "C": trial.suggest_float("C", 0.1, 100, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
                "random_state": 42,
            }
        elif "svc" in model_name:
            return {
                "C": trial.suggest_float("C", 0.1, 100, log=True),
                "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                "random_state": 42,
            }
        else:
            # Default parameter space
            return {"random_state": 42}

    def _fallback_optimization(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        """Fallback to basic optimization."""
        # Simple parameter grid
        param_grid = {"random_state": [42]}

        # Add model-specific parameters
        if hasattr(self.model, "n_estimators"):
            param_grid["n_estimators"] = [50, 100, 200]

        if hasattr(self.model, "max_depth"):
            param_grid["max_depth"] = [3, 5, 10]

        optimizer = HyperparameterOptimizer(
            self.model, param_grid, self.cv, self.scoring
        )
        return optimizer.optimize(X, y, method="grid")


class BayesianOptimizer:
    """Bayesian optimization for hyperparameters."""

    def __init__(
        self,
        model: BaseEstimator,
        cv: int = 5,
        scoring: str = "accuracy",
        n_iter: int = 50,
    ):
        """
        Initialize Bayesian optimizer.

        Args:
            model: Base model to optimize
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_iter: Number of iterations
        """
        self.model = model
        self.cv = cv
        self.scoring = scoring
        self.n_iter = n_iter
        self.best_model = None
        self.best_params = None
        self.best_score = None

    def optimize(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        """
        Optimize hyperparameters using Bayesian optimization.

        Args:
            X: Training features
            y: Training targets

        Returns:
            Best model
        """
        try:
            from skopt import gp_minimize

            # Define search space
            space = self._define_search_space()

            def objective(params):
                # Convert parameters to dictionary
                param_dict = self._params_to_dict(params, space)

                # Create model with parameters
                model = self.model.__class__(**param_dict)

                # Perform cross-validation
                scores = cross_val_score(
                    model, X, y, cv=self.cv, scoring=self.scoring, n_jobs=-1
                )
                return -scores.mean()  # Minimize negative score

            # Run optimization
            result = gp_minimize(objective, space, n_calls=self.n_iter, random_state=42)

            # Get best parameters
            self.best_params = self._params_to_dict(result.x, space)
            self.best_score = -result.fun

            # Create best model
            self.best_model = self.model.__class__(**self.best_params)
            self.best_model.fit(X, y)

            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best score: {self.best_score:.4f}")

            return self.best_model

        except ImportError:
            logger.warning("scikit-optimize not available, falling back to grid search")
            return self._fallback_optimization(X, y)
        except Exception as e:
            logger.warning(f"Bayesian optimization failed: {e}")
            return self._fallback_optimization(X, y)

    def _define_search_space(self):
        """Define search space for Bayesian optimization."""
        from skopt.space import Integer, Real

        model_name = self.model.__class__.__name__.lower()

        if "randomforest" in model_name:
            return [
                Integer(50, 300, name="n_estimators"),
                Integer(3, 20, name="max_depth"),
                Integer(2, 20, name="min_samples_split"),
                Integer(1, 10, name="min_samples_leaf"),
            ]
        elif "gradientboosting" in model_name:
            return [
                Integer(50, 300, name="n_estimators"),
                Real(0.01, 0.3, name="learning_rate"),
                Integer(3, 10, name="max_depth"),
                Real(0.6, 1.0, name="subsample"),
            ]
        else:
            return []

    def _params_to_dict(self, params, space):
        """Convert parameter list to dictionary."""
        param_names = [dim.name for dim in space]
        return dict(zip(param_names, params, strict=False))

    def _fallback_optimization(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        """Fallback to basic optimization."""
        param_grid = {"random_state": [42]}

        if hasattr(self.model, "n_estimators"):
            param_grid["n_estimators"] = [50, 100, 200]

        optimizer = HyperparameterOptimizer(
            self.model, param_grid, self.cv, self.scoring
        )
        return optimizer.optimize(X, y, method="grid")


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


def get_param_grid(model_name: str) -> dict[str, list]:
    """Get parameter grid for a specific model."""
    return PARAM_GRIDS.get(model_name.lower(), {})


def optimize_model(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    method: str = "grid",
    cv: int = 5,
    scoring: str = "accuracy",
    n_trials: int = 100,
) -> BaseEstimator:
    """
    Optimize a model's hyperparameters.

    Args:
        model: Model to optimize
        X: Training features
        y: Training targets
        method: Optimization method ('grid', 'random', 'optuna', 'bayesian')
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_trials: Number of trials (for optuna/bayesian)

    Returns:
        Optimized model
    """
    if method == "optuna":
        optimizer = OptunaOptimizer(model, cv, scoring, n_trials)
    elif method == "bayesian":
        optimizer = BayesianOptimizer(model, cv, scoring, n_trials)
    else:
        # Get parameter grid for the model
        model_name = model.__class__.__name__
        param_grid = get_param_grid(model_name)

        if not param_grid:
            # Default parameter grid
            param_grid = {"random_state": [42]}

        optimizer = HyperparameterOptimizer(model, param_grid, cv, scoring)

    return optimizer.optimize(X, y)
