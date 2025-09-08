"""Model zoo with curated ML algorithms."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
import warnings

# Scikit-learn imports
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost not available. Install with: pip install catboost")

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    model_class: Any
    is_classification: bool
    hyperparameters: Dict[str, Any]
    priority: int = 1  # Higher priority = tried first
    requires_scaling: bool = True
    handles_categorical: bool = False
    memory_efficient: bool = True


class BaseModel(ABC):
    """Base class for all models in the zoo."""
    
    def __init__(self, config: ModelConfig, **kwargs):
        """Initialize the model.
        
        Args:
            config: Model configuration
            **kwargs: Additional model parameters
        """
        self.config = config
        self.model = config.model_class(**kwargs)
        self.is_fitted = False
        self.feature_names = []
        self.training_time = 0.0
        self.prediction_time = 0.0
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities (for classification)."""
        pass
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available."""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_.flatten())
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'name': self.config.name,
            'is_classification': self.config.is_classification,
            'is_fitted': self.is_fitted,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'feature_count': len(self.feature_names),
            'has_feature_importance': self.get_feature_importance() is not None
        }


class SklearnModel(BaseModel):
    """Wrapper for scikit-learn models."""
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'SklearnModel':
        """Fit the model to training data."""
        import time
        start_time = time.time()
        
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self.is_fitted = True
        
        self.training_time = time.time() - start_time
        logger.info(f"Fitted {self.config.name} in {self.training_time:.2f}s")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        import time
        start_time = time.time()
        
        predictions = self.model.predict(X)
        
        self.prediction_time = time.time() - start_time
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities (for classification)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if not self.config.is_classification:
            raise ValueError("predict_proba only available for classification models")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"{self.config.name} does not support predict_proba")
        
        return self.model.predict_proba(X)


class XGBoostModel(BaseModel):
    """Wrapper for XGBoost models."""
    
    def __init__(self, config: ModelConfig, **kwargs):
        """Initialize XGBoost model."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        super().__init__(config, **kwargs)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'XGBoostModel':
        """Fit the model to training data."""
        import time
        start_time = time.time()
        
        self.feature_names = X.columns.tolist()
        
        # Handle categorical features
        if self.config.handles_categorical:
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            self.model.fit(X, y, categorical_feature=categorical_features)
        else:
            self.model.fit(X, y)
        
        self.is_fitted = True
        self.training_time = time.time() - start_time
        logger.info(f"Fitted {self.config.name} in {self.training_time:.2f}s")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        import time
        start_time = time.time()
        
        predictions = self.model.predict(X)
        
        self.prediction_time = time.time() - start_time
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities (for classification)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if not self.config.is_classification:
            raise ValueError("predict_proba only available for classification models")
        
        return self.model.predict_proba(X)


class LightGBMModel(BaseModel):
    """Wrapper for LightGBM models."""
    
    def __init__(self, config: ModelConfig, **kwargs):
        """Initialize LightGBM model."""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")
        
        super().__init__(config, **kwargs)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LightGBMModel':
        """Fit the model to training data."""
        import time
        start_time = time.time()
        
        self.feature_names = X.columns.tolist()
        
        # Handle categorical features
        if self.config.handles_categorical:
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            self.model.fit(X, y, categorical_feature=categorical_features)
        else:
            self.model.fit(X, y)
        
        self.is_fitted = True
        self.training_time = time.time() - start_time
        logger.info(f"Fitted {self.config.name} in {self.training_time:.2f}s")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        import time
        start_time = time.time()
        
        predictions = self.model.predict(X)
        
        self.prediction_time = time.time() - start_time
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities (for classification)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if not self.config.is_classification:
            raise ValueError("predict_proba only available for classification models")
        
        return self.model.predict_proba(X)


class CatBoostModel(BaseModel):
    """Wrapper for CatBoost models."""
    
    def __init__(self, config: ModelConfig, **kwargs):
        """Initialize CatBoost model."""
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not available")
        
        super().__init__(config, **kwargs)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'CatBoostModel':
        """Fit the model to training data."""
        import time
        start_time = time.time()
        
        self.feature_names = X.columns.tolist()
        
        # Handle categorical features
        if self.config.handles_categorical:
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            self.model.fit(X, y, cat_features=categorical_features, verbose=False)
        else:
            self.model.fit(X, y, verbose=False)
        
        self.is_fitted = True
        self.training_time = time.time() - start_time
        logger.info(f"Fitted {self.config.name} in {self.training_time:.2f}s")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        import time
        start_time = time.time()
        
        predictions = self.model.predict(X)
        
        self.prediction_time = time.time() - start_time
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities (for classification)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if not self.config.is_classification:
            raise ValueError("predict_proba only available for classification models")
        
        return self.model.predict_proba(X)


class ModelZoo:
    """Registry of available models."""
    
    def __init__(self):
        """Initialize the model zoo."""
        self.models = {}
        self._register_models()
    
    def _register_models(self):
        """Register all available models."""
        # Classification models
        self._register_classification_models()
        
        # Regression models
        self._register_regression_models()
    
    def _register_classification_models(self):
        """Register classification models."""
        classification_models = [
            # Linear models
            ModelConfig(
                name="logistic_regression",
                model_class=LogisticRegression,
                is_classification=True,
                hyperparameters={
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga'],
                    'max_iter': [100, 1000]
                },
                priority=1
            ),
            
            # Tree-based models
            ModelConfig(
                name="random_forest",
                model_class=RandomForestClassifier,
                is_classification=True,
                hyperparameters={
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },
                priority=2,
                handles_categorical=False
            ),
            
            ModelConfig(
                name="gradient_boosting",
                model_class=GradientBoostingClassifier,
                is_classification=True,
                hyperparameters={
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                priority=2
            ),
            
            # Distance-based models
            ModelConfig(
                name="knn",
                model_class=KNeighborsClassifier,
                is_classification=True,
                hyperparameters={
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
                priority=3,
                requires_scaling=True
            ),
            
            # Neural networks
            ModelConfig(
                name="neural_network",
                model_class=MLPClassifier,
                is_classification=True,
                hyperparameters={
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                },
                priority=3,
                requires_scaling=True
            ),
            
            # Support Vector Machines
            ModelConfig(
                name="svm",
                model_class=SVC,
                is_classification=True,
                hyperparameters={
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01]
                },
                priority=3,
                requires_scaling=True
            ),
            
            # Naive Bayes
            ModelConfig(
                name="naive_bayes",
                model_class=GaussianNB,
                is_classification=True,
                hyperparameters={
                    'var_smoothing': [1e-9, 1e-8, 1e-7]
                },
                priority=4
            ),
            
            # Discriminant Analysis
            ModelConfig(
                name="lda",
                model_class=LinearDiscriminantAnalysis,
                is_classification=True,
                hyperparameters={
                    'solver': ['svd', 'lsqr', 'eigen'],
                    'shrinkage': [None, 'auto', 0.1, 0.5]
                },
                priority=4
            )
        ]
        
        # Add advanced models if available
        if XGBOOST_AVAILABLE:
            classification_models.append(
                ModelConfig(
                    name="xgboost",
                    model_class=xgb.XGBClassifier,
                    is_classification=True,
                    hyperparameters={
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0]
                    },
                    priority=1,
                    handles_categorical=True
                )
            )
        
        if LIGHTGBM_AVAILABLE:
            classification_models.append(
                ModelConfig(
                    name="lightgbm",
                    model_class=lgb.LGBMClassifier,
                    is_classification=True,
                    hyperparameters={
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0]
                    },
                    priority=1,
                    handles_categorical=True
                )
            )
        
        if CATBOOST_AVAILABLE:
            classification_models.append(
                ModelConfig(
                    name="catboost",
                    model_class=cb.CatBoostClassifier,
                    is_classification=True,
                    hyperparameters={
                        'iterations': [50, 100, 200],
                        'depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'l2_leaf_reg': [1, 3, 5]
                    },
                    priority=1,
                    handles_categorical=True
                )
            )
        
        # Register classification models
        for config in classification_models:
            self.models[config.name] = config
    
    def _register_regression_models(self):
        """Register regression models."""
        regression_models = [
            # Linear models
            ModelConfig(
                name="linear_regression",
                model_class=LinearRegression,
                is_classification=False,
                hyperparameters={
                    'fit_intercept': [True, False]
                },
                priority=1
            ),
            
            ModelConfig(
                name="ridge",
                model_class=Ridge,
                is_classification=False,
                hyperparameters={
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'fit_intercept': [True, False]
                },
                priority=1
            ),
            
            ModelConfig(
                name="lasso",
                model_class=Lasso,
                is_classification=False,
                hyperparameters={
                    'alpha': [0.1, 1.0, 10.0],
                    'fit_intercept': [True, False],
                    'max_iter': [1000, 2000]
                },
                priority=1
            ),
            
            # Tree-based models
            ModelConfig(
                name="random_forest_regressor",
                model_class=RandomForestRegressor,
                is_classification=False,
                hyperparameters={
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },
                priority=2
            ),
            
            ModelConfig(
                name="gradient_boosting_regressor",
                model_class=GradientBoostingRegressor,
                is_classification=False,
                hyperparameters={
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                priority=2
            ),
            
            # Distance-based models
            ModelConfig(
                name="knn_regressor",
                model_class=KNeighborsRegressor,
                is_classification=False,
                hyperparameters={
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
                priority=3,
                requires_scaling=True
            ),
            
            # Neural networks
            ModelConfig(
                name="neural_network_regressor",
                model_class=MLPRegressor,
                is_classification=False,
                hyperparameters={
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                },
                priority=3,
                requires_scaling=True
            ),
            
            # Support Vector Machines
            ModelConfig(
                name="svr",
                model_class=SVR,
                is_classification=False,
                hyperparameters={
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01]
                },
                priority=3,
                requires_scaling=True
            )
        ]
        
        # Add advanced models if available
        if XGBOOST_AVAILABLE:
            regression_models.append(
                ModelConfig(
                    name="xgboost_regressor",
                    model_class=xgb.XGBRegressor,
                    is_classification=False,
                    hyperparameters={
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0]
                    },
                    priority=1,
                    handles_categorical=True
                )
            )
        
        if LIGHTGBM_AVAILABLE:
            regression_models.append(
                ModelConfig(
                    name="lightgbm_regressor",
                    model_class=lgb.LGBMRegressor,
                    is_classification=False,
                    hyperparameters={
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0]
                    },
                    priority=1,
                    handles_categorical=True
                )
            )
        
        if CATBOOST_AVAILABLE:
            regression_models.append(
                ModelConfig(
                    name="catboost_regressor",
                    model_class=cb.CatBoostRegressor,
                    is_classification=False,
                    hyperparameters={
                        'iterations': [50, 100, 200],
                        'depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'l2_leaf_reg': [1, 3, 5]
                    },
                    priority=1,
                    handles_categorical=True
                )
            )
        
        # Register regression models
        for config in regression_models:
            self.models[config.name] = config
    
    def get_model(self, name: str, is_classification: bool) -> BaseModel:
        """Get a model instance by name.
        
        Args:
            name: Model name
            is_classification: Whether this is a classification task
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model not found or type mismatch
        """
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found in zoo")
        
        config = self.models[name]
        if config.is_classification != is_classification:
            raise ValueError(f"Model '{name}' is for {'classification' if config.is_classification else 'regression'}, "
                           f"but task is {'classification' if is_classification else 'regression'}")
        
        # Create appropriate model wrapper
        if name.startswith('xgboost'):
            return XGBoostModel(config)
        elif name.startswith('lightgbm'):
            return LightGBMModel(config)
        elif name.startswith('catboost'):
            return CatBoostModel(config)
        else:
            return SklearnModel(config)
    
    def list_models(self, is_classification: bool = None) -> List[str]:
        """List available models.
        
        Args:
            is_classification: Filter by task type (None for all)
            
        Returns:
            List of model names
        """
        if is_classification is None:
            return list(self.models.keys())
        
        return [name for name, config in self.models.items() 
                if config.is_classification == is_classification]
    
    def get_model_config(self, name: str) -> ModelConfig:
        """Get model configuration.
        
        Args:
            name: Model name
            
        Returns:
            Model configuration
        """
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found in zoo")
        
        return self.models[name]
    
    def get_recommended_models(self, is_classification: bool, n_samples: int, n_features: int) -> List[str]:
        """Get recommended models based on dataset characteristics.
        
        Args:
            is_classification: Whether this is a classification task
            n_samples: Number of samples
            n_features: Number of features
            
        Returns:
            List of recommended model names
        """
        available_models = self.list_models(is_classification)
        
        # Sort by priority (higher priority first)
        model_priorities = [(name, self.models[name].priority) for name in available_models]
        model_priorities.sort(key=lambda x: x[1], reverse=True)
        
        # For small datasets, prefer simpler models
        if n_samples < 1000:
            recommended = [name for name, _ in model_priorities 
                         if name in ['logistic_regression', 'linear_regression', 'naive_bayes', 'lda']]
        else:
            recommended = [name for name, _ in model_priorities]
        
        return recommended[:5]  # Return top 5 recommendations


# Global model zoo instance
model_zoo = ModelZoo()
