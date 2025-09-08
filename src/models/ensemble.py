"""
Ensemble Models Module

This module provides ensemble model building functionality
for the autonomous ML agent.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    VotingClassifier, VotingRegressor,
    BaggingClassifier, BaggingRegressor,
    StackingClassifier, StackingRegressor
)
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

logger = logging.getLogger(__name__)


class EnsembleBuilder:
    """Build ensemble models from base models."""
    
    def __init__(self, base_models: Dict[str, BaseEstimator], 
                 ensemble_method: str = 'voting',
                 meta_model: Optional[BaseEstimator] = None,
                 weights: Optional[List[float]] = None):
        """
        Initialize ensemble builder.
        
        Args:
            base_models: Dictionary of base models
            ensemble_method: Method for ensembling ('voting', 'bagging', 'stacking')
            meta_model: Meta-model for stacking (optional)
            weights: Weights for voting ensemble (optional)
        """
        self.base_models = base_models
        self.ensemble_method = ensemble_method
        self.meta_model = meta_model
        self.weights = weights
        self.ensemble = None
        self.is_fitted = False
        
    def build_ensemble(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        """
        Build and fit ensemble model.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Fitted ensemble model
        """
        if self.ensemble_method == 'voting':
            self.ensemble = self._build_voting_ensemble()
        elif self.ensemble_method == 'bagging':
            self.ensemble = self._build_bagging_ensemble()
        elif self.ensemble_method == 'stacking':
            self.ensemble = self._build_stacking_ensemble()
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        # Fit the ensemble
        self.ensemble.fit(X, y)
        self.is_fitted = True
        
        logger.info(f"Built {self.ensemble_method} ensemble with {len(self.base_models)} base models")
        return self.ensemble
    
    def _build_voting_ensemble(self) -> BaseEstimator:
        """Build voting ensemble."""
        # Determine if classification or regression
        if hasattr(list(self.base_models.values())[0], 'predict_proba'):
            # Classification
            ensemble = VotingClassifier(
                estimators=[(name, model) for name, model in self.base_models.items()],
                voting='soft',
                weights=self.weights
            )
        else:
            # Regression
            ensemble = VotingRegressor(
                estimators=[(name, model) for name, model in self.base_models.items()],
                weights=self.weights
            )
        
        return ensemble
    
    def _build_bagging_ensemble(self) -> BaseEstimator:
        """Build bagging ensemble."""
        # Use the first model as base estimator
        base_estimator = list(self.base_models.values())[0]
        
        if hasattr(base_estimator, 'predict_proba'):
            # Classification
            ensemble = BaggingClassifier(
                base_estimator=base_estimator,
                n_estimators=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            # Regression
            ensemble = BaggingRegressor(
                base_estimator=base_estimator,
                n_estimators=10,
                random_state=42,
                n_jobs=-1
            )
        
        return ensemble
    
    def _build_stacking_ensemble(self) -> BaseEstimator:
        """Build stacking ensemble."""
        if self.meta_model is None:
            # Use default meta-model
            from sklearn.linear_model import LogisticRegression, LinearRegression
            
            if hasattr(list(self.base_models.values())[0], 'predict_proba'):
                self.meta_model = LogisticRegression(random_state=42)
            else:
                self.meta_model = LinearRegression()
        
        if hasattr(list(self.base_models.values())[0], 'predict_proba'):
            # Classification
            ensemble = StackingClassifier(
                estimators=[(name, model) for name, model in self.base_models.items()],
                final_estimator=self.meta_model,
                cv=5,
                n_jobs=-1
            )
        else:
            # Regression
            ensemble = StackingRegressor(
                estimators=[(name, model) for name, model in self.base_models.items()],
                final_estimator=self.meta_model,
                cv=5,
                n_jobs=-1
            )
        
        return ensemble
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call build_ensemble() first.")
        return self.ensemble.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (for classification)."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call build_ensemble() first.")
        if hasattr(self.ensemble, 'predict_proba'):
            return self.ensemble.predict_proba(X)
        else:
            raise ValueError("Ensemble does not support predict_proba")


class AutoEnsembleBuilder:
    """Automatically build optimal ensemble from multiple models."""
    
    def __init__(self, models: Dict[str, BaseEstimator], 
                 ensemble_methods: List[str] = None,
                 cv: int = 5,
                 scoring: str = 'accuracy'):
        """
        Initialize auto ensemble builder.
        
        Args:
            models: Dictionary of models to ensemble
            ensemble_methods: List of ensemble methods to try
            cv: Number of cross-validation folds
            scoring: Scoring metric
        """
        self.models = models
        self.ensemble_methods = ensemble_methods or ['voting', 'stacking']
        self.cv = cv
        self.scoring = scoring
        self.best_ensemble = None
        self.best_method = None
        self.best_score = None
        
    def find_best_ensemble(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        """
        Find the best ensemble configuration.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Best ensemble model
        """
        best_score = -np.inf
        best_ensemble = None
        best_method = None
        
        for method in self.ensemble_methods:
            try:
                logger.info(f"Testing {method} ensemble...")
                
                # Build ensemble
                if method == 'voting':
                    ensemble = self._build_voting_ensemble()
                elif method == 'stacking':
                    ensemble = self._build_stacking_ensemble()
                else:
                    continue
                
                # Evaluate ensemble
                scores = cross_val_score(ensemble, X, y, cv=self.cv, scoring=self.scoring, n_jobs=-1)
                mean_score = scores.mean()
                
                logger.info(f"{method} ensemble score: {mean_score:.4f} (+/- {scores.std() * 2:.4f})")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_ensemble = ensemble
                    best_method = method
                    
            except Exception as e:
                logger.warning(f"Failed to build {method} ensemble: {e}")
                continue
        
        if best_ensemble is None:
            raise ValueError("Could not build any ensemble")
        
        # Fit the best ensemble on full data
        best_ensemble.fit(X, y)
        
        self.best_ensemble = best_ensemble
        self.best_method = best_method
        self.best_score = best_score
        
        logger.info(f"Best ensemble: {best_method} with score {best_score:.4f}")
        return best_ensemble
    
    def _build_voting_ensemble(self) -> BaseEstimator:
        """Build voting ensemble."""
        if hasattr(list(self.models.values())[0], 'predict_proba'):
            return VotingClassifier(
                estimators=[(name, model) for name, model in self.models.items()],
                voting='soft'
            )
        else:
            return VotingRegressor(
                estimators=[(name, model) for name, model in self.models.items()]
            )
    
    def _build_stacking_ensemble(self) -> BaseEstimator:
        """Build stacking ensemble."""
        from sklearn.linear_model import LogisticRegression, LinearRegression
        
        if hasattr(list(self.models.values())[0], 'predict_proba'):
            meta_model = LogisticRegression(random_state=42)
            return StackingClassifier(
                estimators=[(name, model) for name, model in self.models.items()],
                final_estimator=meta_model,
                cv=5,
                n_jobs=-1
            )
        else:
            meta_model = LinearRegression()
            return StackingRegressor(
                estimators=[(name, model) for name, model in self.models.items()],
                final_estimator=meta_model,
                cv=5,
                n_jobs=-1
            )
    
    def get_best_ensemble(self) -> Optional[BaseEstimator]:
        """Get the best ensemble after optimization."""
        return self.best_ensemble
    
    def get_best_method(self) -> Optional[str]:
        """Get the best ensemble method."""
        return self.best_method
    
    def get_best_score(self) -> Optional[float]:
        """Get the best ensemble score."""
        return self.best_score


class WeightedEnsemble:
    """Weighted ensemble with learnable weights."""
    
    def __init__(self, models: Dict[str, BaseEstimator], 
                 weight_optimization: str = 'uniform'):
        """
        Initialize weighted ensemble.
        
        Args:
            models: Dictionary of models
            weight_optimization: Method for weight optimization ('uniform', 'cross_val', 'optimization')
        """
        self.models = models
        self.weight_optimization = weight_optimization
        self.weights = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the weighted ensemble."""
        if self.weight_optimization == 'uniform':
            self.weights = np.ones(len(self.models)) / len(self.models)
        elif self.weight_optimization == 'cross_val':
            self.weights = self._optimize_weights_cv(X, y)
        elif self.weight_optimization == 'optimization':
            self.weights = self._optimize_weights(X, y)
        else:
            raise ValueError(f"Unknown weight optimization method: {self.weight_optimization}")
        
        # Fit all base models
        for name, model in self.models.items():
            model.fit(X, y)
        
        self.is_fitted = True
        logger.info(f"Fitted weighted ensemble with weights: {self.weights}")
    
    def _optimize_weights_cv(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Optimize weights using cross-validation."""
        from sklearn.model_selection import cross_val_score
        
        # Get cross-validation scores for each model
        scores = []
        for name, model in self.models.items():
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
            scores.append(cv_scores.mean())
        
        # Convert scores to weights (higher score = higher weight)
        scores = np.array(scores)
        weights = scores / scores.sum()
        
        return weights
    
    def _optimize_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Optimize weights using optimization."""
        from sklearn.model_selection import train_test_split
        
        # Split data for optimization
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit all models
        predictions = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            predictions[name] = model.predict(X_val)
        
        # Optimize weights
        from scipy.optimize import minimize
        
        def objective(weights):
            # Normalize weights
            weights = np.abs(weights)
            weights = weights / weights.sum()
            
            # Weighted prediction
            weighted_pred = np.zeros_like(y_val, dtype=float)
            for i, (name, pred) in enumerate(predictions.items()):
                weighted_pred += weights[i] * pred
            
            # Calculate score
            if hasattr(list(self.models.values())[0], 'predict_proba'):
                # Classification
                from sklearn.metrics import accuracy_score
                return -accuracy_score(y_val, weighted_pred.round())
            else:
                # Regression
                from sklearn.metrics import r2_score
                return -r2_score(y_val, weighted_pred)
        
        # Initial weights
        initial_weights = np.ones(len(self.models)) / len(self.models)
        
        # Optimize
        result = minimize(objective, initial_weights, method='L-BFGS-B')
        
        # Normalize final weights
        weights = np.abs(result.x)
        weights = weights / weights.sum()
        
        return weights
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make weighted predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # Weighted combination
        weighted_pred = np.zeros_like(list(predictions.values())[0], dtype=float)
        for i, (name, pred) in enumerate(predictions.items()):
            weighted_pred += self.weights[i] * pred
        
        return weighted_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (for classification)."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        if not hasattr(list(self.models.values())[0], 'predict_proba'):
            raise ValueError("Base models do not support predict_proba")
        
        # Get probability predictions from all models
        proba_predictions = {}
        for name, model in self.models.items():
            proba_predictions[name] = model.predict_proba(X)
        
        # Weighted combination
        weighted_proba = np.zeros_like(list(proba_predictions.values())[0], dtype=float)
        for i, (name, proba) in enumerate(proba_predictions.items()):
            weighted_proba += self.weights[i] * proba
        
        return weighted_proba


def create_ensemble(models: Dict[str, BaseEstimator], 
                   method: str = 'voting',
                   **kwargs) -> BaseEstimator:
    """
    Create an ensemble from multiple models.
    
    Args:
        models: Dictionary of models
        method: Ensemble method ('voting', 'stacking', 'bagging', 'weighted')
        **kwargs: Additional arguments
        
    Returns:
        Ensemble model
    """
    if method == 'voting':
        return EnsembleBuilder(models, 'voting', **kwargs)
    elif method == 'stacking':
        return EnsembleBuilder(models, 'stacking', **kwargs)
    elif method == 'bagging':
        return EnsembleBuilder(models, 'bagging', **kwargs)
    elif method == 'weighted':
        return WeightedEnsemble(models, **kwargs)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")


def auto_ensemble(models: Dict[str, BaseEstimator], 
                  X: np.ndarray, 
                  y: np.ndarray,
                  methods: List[str] = None) -> BaseEstimator:
    """
    Automatically create the best ensemble.
    
    Args:
        models: Dictionary of models
        X: Training features
        y: Training targets
        methods: List of ensemble methods to try
        
    Returns:
        Best ensemble model
    """
    builder = AutoEnsembleBuilder(models, methods)
    return builder.find_best_ensemble(X, y)
