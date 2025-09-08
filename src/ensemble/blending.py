"""Ensemble blending methods."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BlendingConfig:
    """Configuration for ensemble blending."""
    method: str = 'weighted'  # 'weighted', 'stacking', 'voting'
    weights: Optional[List[float]] = None
    meta_model: str = 'linear'  # 'linear', 'ridge', 'lasso'
    cv_folds: int = 5
    random_state: int = 42


class BaseBlender(ABC):
    """Base class for ensemble blenders."""
    
    def __init__(self, config: BlendingConfig):
        """Initialize the blender.
        
        Args:
            config: Blending configuration
        """
        self.config = config
        self.is_fitted = False
        self.models = []
        self.weights = None
    
    @abstractmethod
    def fit(self, models: List[Any], X: pd.DataFrame, y: pd.Series) -> 'BaseBlender':
        """Fit the blender to the models and data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the blended ensemble."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions (for classification)."""
        pass


class WeightedBlender(BaseBlender):
    """Weighted ensemble blender."""
    
    def fit(self, models: List[Any], X: pd.DataFrame, y: pd.Series) -> 'WeightedBlender':
        """Fit the weighted blender.
        
        Args:
            models: List of trained models
            X: Training features
            y: Training target
            
        Returns:
            Fitted blender
        """
        self.models = models
        
        if self.config.weights is None:
            # Equal weights
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # Use provided weights
            if len(self.config.weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            self.weights = self.config.weights.copy()
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        self.is_fitted = True
        logger.info(f"Weighted blender fitted with weights: {self.weights}")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using weighted blending."""
        if not self.is_fitted:
            raise ValueError("Blender must be fitted before prediction")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        weighted_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            weighted_pred += weight * pred
        
        return weighted_pred
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions using weighted blending."""
        if not self.is_fitted:
            raise ValueError("Blender must be fitted before prediction")
        
        probabilities = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                probabilities.append(proba)
            else:
                # Convert predictions to probabilities for models without predict_proba
                pred = model.predict(X)
                if len(np.unique(pred)) == 2:  # Binary classification
                    proba = np.column_stack([1 - pred, pred])
                else:
                    # Multi-class - create one-hot encoding
                    unique_classes = np.unique(pred)
                    proba = np.zeros((len(pred), len(unique_classes)))
                    for i, class_val in enumerate(unique_classes):
                        proba[pred == class_val, i] = 1.0
                probabilities.append(proba)
        
        # Weighted average of probabilities
        weighted_proba = np.zeros_like(probabilities[0])
        for proba, weight in zip(probabilities, self.weights):
            weighted_proba += weight * proba
        
        return weighted_proba


class StackingBlender(BaseBlender):
    """Stacking ensemble blender."""
    
    def __init__(self, config: BlendingConfig):
        """Initialize stacking blender."""
        super().__init__(config)
        self.meta_model = None
        self.is_classification = None
    
    def fit(self, models: List[Any], X: pd.DataFrame, y: pd.Series) -> 'StackingBlender':
        """Fit the stacking blender.
        
        Args:
            models: List of trained models
            X: Training features
            y: Training target
            
        Returns:
            Fitted blender
        """
        self.models = models
        self.is_classification = models[0].config.is_classification
        
        # Create meta-features using cross-validation
        meta_features = self._create_meta_features(X, y)
        
        # Train meta-model
        self.meta_model = self._create_meta_model()
        self.meta_model.fit(meta_features, y)
        
        self.is_fitted = True
        logger.info(f"Stacking blender fitted with meta-model: {self.meta_model.__class__.__name__}")
        return self
    
    def _create_meta_features(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Create meta-features using cross-validation."""
        from sklearn.model_selection import StratifiedKFold, KFold
        
        # Setup cross-validation
        if self.is_classification:
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
        else:
            cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
        
        meta_features = np.zeros((len(X), len(self.models)))
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            
            for i, model in enumerate(self.models):
                # Create a copy of the model for this fold
                fold_model = model.__class__(model.config, **model.model.get_params())
                fold_model.fit(X_train, y_train)
                
                # Make predictions on validation set
                if self.is_classification and hasattr(fold_model, 'predict_proba'):
                    pred = fold_model.predict_proba(X_val)[:, 1]  # Use positive class probability
                else:
                    pred = fold_model.predict(X_val)
                
                meta_features[val_idx, i] = pred
        
        return meta_features
    
    def _create_meta_model(self):
        """Create meta-model based on configuration."""
        if self.config.meta_model == 'linear':
            from sklearn.linear_model import LinearRegression, LogisticRegression
            if self.is_classification:
                return LogisticRegression(random_state=self.config.random_state)
            else:
                return LinearRegression()
        elif self.config.meta_model == 'ridge':
            from sklearn.linear_model import Ridge, RidgeClassifier
            if self.is_classification:
                return RidgeClassifier(random_state=self.config.random_state)
            else:
                return Ridge(random_state=self.config.random_state)
        elif self.config.meta_model == 'lasso':
            from sklearn.linear_model import Lasso, LogisticRegression
            if self.is_classification:
                return LogisticRegression(penalty='l1', solver='liblinear', random_state=self.config.random_state)
            else:
                return Lasso(random_state=self.config.random_state)
        else:
            raise ValueError(f"Unknown meta-model: {self.config.meta_model}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using stacking."""
        if not self.is_fitted:
            raise ValueError("Blender must be fitted before prediction")
        
        # Get predictions from base models
        base_predictions = []
        for model in self.models:
            if self.is_classification and hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]  # Use positive class probability
            else:
                pred = model.predict(X)
            base_predictions.append(pred)
        
        # Stack predictions
        meta_features = np.column_stack(base_predictions)
        
        # Make final prediction using meta-model
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions using stacking."""
        if not self.is_fitted:
            raise ValueError("Blender must be fitted before prediction")
        
        if not self.is_classification:
            raise ValueError("predict_proba only available for classification models")
        
        # Get predictions from base models
        base_predictions = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]  # Use positive class probability
            else:
                pred = model.predict(X)
            base_predictions.append(pred)
        
        # Stack predictions
        meta_features = np.column_stack(base_predictions)
        
        # Make final prediction using meta-model
        if hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(meta_features)
        else:
            # Convert predictions to probabilities
            pred = self.meta_model.predict(meta_features)
            proba = np.column_stack([1 - pred, pred])
            return proba


class VotingBlender(BaseBlender):
    """Voting ensemble blender."""
    
    def __init__(self, config: BlendingConfig):
        """Initialize voting blender."""
        super().__init__(config)
        self.voting_type = 'soft'  # 'soft' or 'hard'
    
    def fit(self, models: List[Any], X: pd.DataFrame, y: pd.Series) -> 'VotingBlender':
        """Fit the voting blender.
        
        Args:
            models: List of trained models
            X: Training features
            y: Training target
            
        Returns:
            Fitted blender
        """
        self.models = models
        self.is_fitted = True
        logger.info(f"Voting blender fitted with {len(models)} models")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using voting."""
        if not self.is_fitted:
            raise ValueError("Blender must be fitted before prediction")
        
        if self.voting_type == 'hard':
            return self._hard_voting(X)
        else:
            return self._soft_voting(X)
    
    def _hard_voting(self, X: pd.DataFrame) -> np.ndarray:
        """Hard voting: majority vote."""
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Majority vote
        predictions_array = np.array(predictions)
        final_predictions = []
        
        for i in range(len(X)):
            votes = predictions_array[:, i]
            unique, counts = np.unique(votes, return_counts=True)
            final_predictions.append(unique[np.argmax(counts)])
        
        return np.array(final_predictions)
    
    def _soft_voting(self, X: pd.DataFrame) -> np.ndarray:
        """Soft voting: average of probabilities."""
        probabilities = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                probabilities.append(proba)
            else:
                # Convert predictions to probabilities
                pred = model.predict(X)
                if len(np.unique(pred)) == 2:  # Binary classification
                    proba = np.column_stack([1 - pred, pred])
                else:
                    # Multi-class - create one-hot encoding
                    unique_classes = np.unique(pred)
                    proba = np.zeros((len(pred), len(unique_classes)))
                    for i, class_val in enumerate(unique_classes):
                        proba[pred == class_val, i] = 1.0
                probabilities.append(proba)
        
        # Average probabilities
        avg_proba = np.mean(probabilities, axis=0)
        
        # Convert to predictions
        return np.argmax(avg_proba, axis=1)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions using voting."""
        if not self.is_fitted:
            raise ValueError("Blender must be fitted before prediction")
        
        probabilities = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                probabilities.append(proba)
            else:
                # Convert predictions to probabilities
                pred = model.predict(X)
                if len(np.unique(pred)) == 2:  # Binary classification
                    proba = np.column_stack([1 - pred, pred])
                else:
                    # Multi-class - create one-hot encoding
                    unique_classes = np.unique(pred)
                    proba = np.zeros((len(pred), len(unique_classes)))
                    for i, class_val in enumerate(unique_classes):
                        proba[pred == class_val, i] = 1.0
                probabilities.append(proba)
        
        # Average probabilities
        return np.mean(probabilities, axis=0)


class EnsembleBlender:
    """Main ensemble blender interface."""
    
    def __init__(self, config: Optional[BlendingConfig] = None):
        """Initialize ensemble blender.
        
        Args:
            config: Blending configuration
        """
        self.config = config or BlendingConfig()
        self.blender = None
    
    def create_blender(self, method: Optional[str] = None) -> BaseBlender:
        """Create a blender instance.
        
        Args:
            method: Blending method (overrides config)
            
        Returns:
            Blender instance
        """
        method = method or self.config.method
        
        if method == 'weighted':
            return WeightedBlender(self.config)
        elif method == 'stacking':
            return StackingBlender(self.config)
        elif method == 'voting':
            return VotingBlender(self.config)
        else:
            raise ValueError(f"Unknown blending method: {method}")
    
    def blend_models(self, models: List[Any], X: pd.DataFrame, y: pd.Series, 
                    method: Optional[str] = None) -> BaseBlender:
        """Blend models using the specified method.
        
        Args:
            models: List of trained models
            X: Training features
            y: Training target
            method: Blending method (overrides config)
            
        Returns:
            Fitted blender
        """
        self.blender = self.create_blender(method)
        self.blender.fit(models, X, y)
        return self.blender
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the fitted blender."""
        if self.blender is None:
            raise ValueError("No blender fitted. Call blend_models first.")
        
        return self.blender.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions using the fitted blender."""
        if self.blender is None:
            raise ValueError("No blender fitted. Call blend_models first.")
        
        return self.blender.predict_proba(X)


def create_ensemble(models: List[Any], X: pd.DataFrame, y: pd.Series, 
                  method: str = 'weighted', **kwargs) -> BaseBlender:
    """Convenience function to create an ensemble.
    
    Args:
        models: List of trained models
        X: Training features
        y: Training target
        method: Blending method
        **kwargs: Additional configuration parameters
        
    Returns:
        Fitted blender
    """
    config = BlendingConfig(method=method, **kwargs)
    blender = EnsembleBlender(config)
    return blender.blend_models(models, X, y, method)
