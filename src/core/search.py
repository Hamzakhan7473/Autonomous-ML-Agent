"""Hyperparameter optimization module."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings

# Hyperparameter optimization libraries
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available. Install with: pip install optuna")

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    warnings.warn("Scikit-optimize not available. Install with: pip install scikit-optimize")

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Configuration for hyperparameter search."""
    method: str = 'optuna'  # 'optuna', 'skopt', 'random'
    n_trials: int = 100
    timeout: int = 3600  # seconds
    cv_folds: int = 5
    scoring: str = 'auto'
    n_jobs: int = -1
    random_state: int = 42


class BaseOptimizer(ABC):
    """Base class for hyperparameter optimizers."""
    
    def __init__(self, config: SearchConfig):
        """Initialize the optimizer.
        
        Args:
            config: Search configuration
        """
        self.config = config
        self.best_score = -np.inf
        self.best_params = {}
        self.best_model = None
        self.trial_results = []
    
    @abstractmethod
    def optimize(self, model, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float, Dict[str, Any]]:
        """Optimize hyperparameters for a model.
        
        Args:
            model: Model to optimize
            X: Training features
            y: Training target
            
        Returns:
            Tuple of (best_model, best_score, best_params)
        """
        pass


class RandomSearchOptimizer(BaseOptimizer):
    """Random search hyperparameter optimization."""
    
    def optimize(self, model, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float, Dict[str, Any]]:
        """Perform random search optimization."""
        
        np.random.seed(self.config.random_state)
        
        # Get hyperparameter space from model config
        param_space = model.config.hyperparameters
        
        best_score = -np.inf
        best_params = {}
        best_model = None
        
        start_time = time.time()
        
        for trial in range(self.config.n_trials):
            if time.time() - start_time > self.config.timeout:
                logger.warning(f"Random search timeout reached after {trial} trials")
                break
            
            # Sample random parameters
            params = {}
            for param_name, param_values in param_space.items():
                if isinstance(param_values, list):
                    params[param_name] = np.random.choice(param_values)
                else:
                    params[param_name] = param_values
            
            try:
                # Create model with sampled parameters
                trial_model = model.__class__(model.config, **params)
                
                # Cross-validation
                cv_scores = self._cross_validate(trial_model, X, y)
                mean_score = np.mean(cv_scores)
                
                self.trial_results.append({
                    'trial': trial,
                    'params': params,
                    'score': mean_score,
                    'cv_scores': cv_scores
                })
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params.copy()
                    best_model = trial_model
                
                logger.debug(f"Trial {trial}: score={mean_score:.4f}, params={params}")
                
            except Exception as e:
                logger.warning(f"Trial {trial} failed: {e}")
                continue
        
        if best_model is None:
            raise ValueError("No successful trials completed")
        
        # Fit best model on full data
        best_model.fit(X, y)
        
        return best_model, best_score, best_params
    
    def _cross_validate(self, model, X: pd.DataFrame, y: pd.Series) -> List[float]:
        """Perform cross-validation."""
        if model.config.is_classification:
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
            scoring = self._get_scoring_metric(model.config.is_classification)
        else:
            cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
            scoring = self._get_scoring_metric(model.config.is_classification)
        
        scores = cross_val_score(model.model, X, y, cv=cv, scoring=scoring, n_jobs=self.config.n_jobs)
        return scores.tolist()
    
    def _get_scoring_metric(self, is_classification: bool) -> str:
        """Get scoring metric based on task type."""
        if self.config.scoring == 'auto':
            return 'accuracy' if is_classification else 'neg_mean_squared_error'
        return self.config.scoring


class OptunaOptimizer(BaseOptimizer):
    """Optuna-based hyperparameter optimization."""
    
    def __init__(self, config: SearchConfig):
        """Initialize Optuna optimizer."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")
        
        super().__init__(config)
        self.study = None
    
    def optimize(self, model, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float, Dict[str, Any]]:
        """Perform Optuna optimization."""
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state)
        )
        
        # Define objective function
        def objective(trial):
            # Sample parameters
            params = {}
            param_space = model.config.hyperparameters
            
            for param_name, param_values in param_space.items():
                if isinstance(param_values, list):
                    if all(isinstance(v, (int, float)) for v in param_values):
                        if all(isinstance(v, int) for v in param_values):
                            # Integer parameter
                            params[param_name] = trial.suggest_int(
                                param_name, 
                                min(param_values), 
                                max(param_values)
                            )
                        else:
                            # Float parameter
                            params[param_name] = trial.suggest_float(
                                param_name, 
                                min(param_values), 
                                max(param_values)
                            )
                    else:
                        # Categorical parameter
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                else:
                    params[param_name] = param_values
            
            try:
                # Create model with sampled parameters
                trial_model = model.__class__(model.config, **params)
                
                # Cross-validation
                cv_scores = self._cross_validate(trial_model, X, y)
                mean_score = np.mean(cv_scores)
                
                return mean_score
                
            except Exception as e:
                logger.warning(f"Optuna trial failed: {e}")
                return -np.inf
        
        # Optimize
        self.study.optimize(
            objective, 
            n_trials=self.config.n_trials,
            timeout=self.config.timeout
        )
        
        if not self.study.best_trial:
            raise ValueError("No successful trials completed")
        
        # Get best parameters
        best_params = self.study.best_params
        best_score = self.study.best_value
        
        # Create best model
        best_model = model.__class__(model.config, **best_params)
        best_model.fit(X, y)
        
        return best_model, best_score, best_params


class SkoptOptimizer(BaseOptimizer):
    """Scikit-optimize based hyperparameter optimization."""
    
    def __init__(self, config: SearchConfig):
        """Initialize Skopt optimizer."""
        if not SKOPT_AVAILABLE:
            raise ImportError("Scikit-optimize not available. Install with: pip install scikit-optimize")
        
        super().__init__(config)
    
    def optimize(self, model, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float, Dict[str, Any]]:
        """Perform Skopt optimization."""
        
        # Define search space
        param_space = model.config.hyperparameters
        dimensions = []
        param_names = []
        
        for param_name, param_values in param_space.items():
            if isinstance(param_values, list):
                if all(isinstance(v, (int, float)) for v in param_values):
                    if all(isinstance(v, int) for v in param_values):
                        # Integer parameter
                        dimensions.append(Integer(min(param_values), max(param_values)))
                    else:
                        # Float parameter
                        dimensions.append(Real(min(param_values), max(param_values)))
                else:
                    # Categorical parameter
                    dimensions.append(Categorical(param_values))
            else:
                # Fixed parameter
                dimensions.append(Categorical([param_values]))
            
            param_names.append(param_name)
        
        # Define objective function
        @use_named_args(dimensions=dimensions)
        def objective(**params):
            try:
                # Create model with sampled parameters
                trial_model = model.__class__(model.config, **params)
                
                # Cross-validation
                cv_scores = self._cross_validate(trial_model, X, y)
                mean_score = np.mean(cv_scores)
                
                return -mean_score  # Minimize negative score
                
            except Exception as e:
                logger.warning(f"Skopt trial failed: {e}")
                return np.inf
        
        # Optimize
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=self.config.n_trials,
            random_state=self.config.random_state
        )
        
        # Get best parameters
        best_params = dict(zip(param_names, result.x))
        best_score = -result.fun
        
        # Create best model
        best_model = model.__class__(model.config, **best_params)
        best_model.fit(X, y)
        
        return best_model, best_score, best_params
    
    def _cross_validate(self, model, X: pd.DataFrame, y: pd.Series) -> List[float]:
        """Perform cross-validation."""
        if model.config.is_classification:
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
            scoring = self._get_scoring_metric(model.config.is_classification)
        else:
            cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
            scoring = self._get_scoring_metric(model.config.is_classification)
        
        scores = cross_val_score(model.model, X, y, cv=cv, scoring=scoring, n_jobs=self.config.n_jobs)
        return scores.tolist()
    
    def _get_scoring_metric(self, is_classification: bool) -> str:
        """Get scoring metric based on task type."""
        if self.config.scoring == 'auto':
            return 'accuracy' if is_classification else 'neg_mean_squared_error'
        return self.config.scoring


class HyperparameterOptimizer:
    """Main hyperparameter optimization interface."""
    
    def __init__(self, 
                 model,
                 method: str = 'auto',
                 n_trials: int = 100,
                 timeout: int = 3600,
                 cv_folds: int = 5,
                 scoring: str = 'auto',
                 n_jobs: int = -1,
                 random_state: int = 42):
        """Initialize the hyperparameter optimizer.
        
        Args:
            model: Model to optimize
            method: Optimization method ('auto', 'optuna', 'skopt', 'random')
            n_trials: Number of trials
            timeout: Timeout in seconds
            cv_folds: Number of CV folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            random_state: Random seed
        """
        self.model = model
        
        # Auto-select method
        if method == 'auto':
            if OPTUNA_AVAILABLE:
                method = 'optuna'
            elif SKOPT_AVAILABLE:
                method = 'skopt'
            else:
                method = 'random'
        
        self.config = SearchConfig(
            method=method,
            n_trials=n_trials,
            timeout=timeout,
            cv_folds=cv_folds,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=random_state
        )
        
        # Create optimizer
        if method == 'optuna':
            self.optimizer = OptunaOptimizer(self.config)
        elif method == 'skopt':
            self.optimizer = SkoptOptimizer(self.config)
        else:
            self.optimizer = RandomSearchOptimizer(self.config)
    
    def optimize(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float, Dict[str, Any]]:
        """Optimize hyperparameters for the model.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            Tuple of (best_model, best_score, best_params)
        """
        logger.info(f"Starting hyperparameter optimization with {self.config.method}")
        logger.info(f"Model: {self.model.config.name}")
        logger.info(f"Trials: {self.config.n_trials}, Timeout: {self.config.timeout}s")
        
        start_time = time.time()
        
        try:
            best_model, best_score, best_params = self.optimizer.optimize(self.model, X, y)
            
            optimization_time = time.time() - start_time
            
            logger.info(f"Optimization completed in {optimization_time:.2f}s")
            logger.info(f"Best score: {best_score:.4f}")
            logger.info(f"Best parameters: {best_params}")
            
            return best_model, best_score, best_params
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise


def optimize_model(model, X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[Any, float, Dict[str, Any]]:
    """Convenience function to optimize a model.
    
    Args:
        model: Model to optimize
        X: Training features
        y: Training target
        **kwargs: Additional optimization parameters
        
    Returns:
        Tuple of (best_model, best_score, best_params)
    """
    optimizer = HyperparameterOptimizer(model, **kwargs)
    return optimizer.optimize(X, y)
