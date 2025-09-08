"""Model evaluation and metrics module."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import time
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)
from sklearn.metrics import classification_report, confusion_matrix
import warnings

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    cv_folds: int = 5
    scoring_metrics: List[str] = None
    stratified_cv: bool = True
    random_state: int = 42
    n_jobs: int = -1


class MetricsCalculator:
    """Calculate various evaluation metrics."""
    
    def __init__(self, is_classification: bool = True):
        """Initialize metrics calculator.
        
        Args:
            is_classification: Whether this is a classification task
        """
        self.is_classification = is_classification
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for classification)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        if self.is_classification:
            # Classification metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # AUC for binary classification
            if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
                try:
                    metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                except Exception as e:
                    logger.warning(f"Could not calculate AUC: {e}")
                    metrics['auc'] = 0.0
            else:
                metrics['auc'] = 0.0
            
            # Macro averages
            metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
            
        else:
            # Regression metrics
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # MAPE (handle division by zero)
            try:
                metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
            except Exception:
                metrics['mape'] = np.inf
            
            # Additional regression metrics
            metrics['mape_safe'] = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
        
        return metrics
    
    def get_primary_metric(self) -> str:
        """Get the primary metric for this task type."""
        return 'accuracy' if self.is_classification else 'r2'


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize model evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        self.metrics_calculator = None
    
    def evaluate(self, model, X: pd.DataFrame, y: pd.Series, 
                 cv_folds: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate a model comprehensively.
        
        Args:
            model: Trained model
            X: Features
            y: Target
            cv_folds: Number of CV folds (overrides config)
            
        Returns:
            Dictionary of evaluation results
        """
        cv_folds = cv_folds or self.config.cv_folds
        
        # Determine task type
        is_classification = model.config.is_classification
        self.metrics_calculator = MetricsCalculator(is_classification)
        
        # Cross-validation
        cv_results = self._cross_validate(model, X, y, cv_folds)
        
        # Single train-test split for detailed metrics
        detailed_results = self._detailed_evaluation(model, X, y)
        
        # Combine results
        results = {
            'cv_results': cv_results,
            'detailed_results': detailed_results,
            'primary_metric': self.metrics_calculator.get_primary_metric(),
            'task_type': 'classification' if is_classification else 'regression'
        }
        
        return results
    
    def _cross_validate(self, model, X: pd.DataFrame, y: pd.Series, cv_folds: int) -> Dict[str, Any]:
        """Perform cross-validation evaluation."""
        
        # Setup CV
        if model.config.is_classification and self.config.stratified_cv:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.config.random_state)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.config.random_state)
        
        # Get scoring metrics
        if self.config.scoring_metrics is None:
            if model.config.is_classification:
                scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            else:
                scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        else:
            scoring_metrics = self.config.scoring_metrics
        
        cv_results = {}
        
        for metric in scoring_metrics:
            try:
                scores = cross_val_score(
                    model.model, X, y, 
                    cv=cv, scoring=metric, 
                    n_jobs=self.config.n_jobs
                )
                cv_results[metric] = {
                    'scores': scores.tolist(),
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                }
            except Exception as e:
                logger.warning(f"Could not calculate {metric}: {e}")
                cv_results[metric] = None
        
        return cv_results
    
    def _detailed_evaluation(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform detailed evaluation on a single train-test split."""
        
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.config.random_state,
            stratify=y if model.config.is_classification else None
        )
        
        # Train model
        start_time = time.time()
        model_copy = model.__class__(model.config, **model.model.get_params())
        model_copy.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        start_time = time.time()
        y_pred = model_copy.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Get probabilities for classification
        y_pred_proba = None
        if model.config.is_classification and hasattr(model_copy, 'predict_proba'):
            try:
                y_pred_proba = model_copy.predict_proba(X_test)
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {e}")
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Additional analysis
        results = {
            'metrics': metrics,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'predictions': y_pred.tolist(),
            'true_values': y_test.tolist()
        }
        
        # Classification-specific analysis
        if model.config.is_classification:
            results['classification_report'] = classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            )
            results['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        
        # Feature importance
        if hasattr(model_copy, 'get_feature_importance'):
            feature_importance = model_copy.get_feature_importance()
            if feature_importance is not None:
                results['feature_importance'] = {
                    'values': feature_importance.tolist(),
                    'feature_names': model_copy.feature_names
                }
        
        return results
    
    def compare_models(self, results: List[Dict[str, Any]], 
                      primary_metric: Optional[str] = None) -> pd.DataFrame:
        """Compare multiple model results.
        
        Args:
            results: List of evaluation results
            primary_metric: Primary metric for comparison
            
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for i, result in enumerate(results):
            model_name = result.get('model_name', f'Model_{i+1}')
            
            # Get primary metric
            if primary_metric is None:
                primary_metric = result.get('primary_metric', 'accuracy')
            
            # Extract metrics
            cv_results = result.get('cv_results', {})
            detailed_results = result.get('detailed_results', {})
            
            row = {'model': model_name}
            
            # CV metrics
            for metric, metric_results in cv_results.items():
                if metric_results is not None:
                    row[f'{metric}_cv_mean'] = metric_results['mean']
                    row[f'{metric}_cv_std'] = metric_results['std']
            
            # Detailed metrics
            detailed_metrics = detailed_results.get('metrics', {})
            for metric, value in detailed_metrics.items():
                row[f'{metric}_test'] = value
            
            # Performance metrics
            row['training_time'] = detailed_results.get('training_time', 0)
            row['prediction_time'] = detailed_results.get('prediction_time', 0)
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by primary metric
        primary_col = f'{primary_metric}_cv_mean'
        if primary_col in df.columns:
            df = df.sort_values(primary_col, ascending=False)
        
        return df
    
    def get_best_model(self, results: List[Dict[str, Any]], 
                      primary_metric: Optional[str] = None) -> Dict[str, Any]:
        """Get the best model from evaluation results.
        
        Args:
            results: List of evaluation results
            primary_metric: Primary metric for selection
            
        Returns:
            Best model result
        """
        if not results:
            raise ValueError("No results provided")
        
        if primary_metric is None:
            primary_metric = results[0].get('primary_metric', 'accuracy')
        
        best_model = None
        best_score = -np.inf
        
        for result in results:
            cv_results = result.get('cv_results', {})
            metric_key = f'{primary_metric}_cv_mean'
            
            # Try different metric keys
            for key in [f'{primary_metric}_cv_mean', f'{primary_metric}_test', primary_metric]:
                if key in cv_results:
                    score = cv_results[key]['mean'] if isinstance(cv_results[key], dict) else cv_results[key]
                    break
            else:
                # Fallback to detailed results
                detailed_metrics = result.get('detailed_results', {}).get('metrics', {})
                score = detailed_metrics.get(primary_metric, -np.inf)
            
            if score > best_score:
                best_score = score
                best_model = result
        
        return best_model


def evaluate_model_performance(model, X: pd.DataFrame, y: pd.Series, 
                             config: Optional[EvaluationConfig] = None) -> Dict[str, Any]:
    """Convenience function to evaluate model performance.
    
    Args:
        model: Trained model
        X: Features
        y: Target
        config: Evaluation configuration
        
    Returns:
        Evaluation results
    """
    evaluator = ModelEvaluator(config)
    return evaluator.evaluate(model, X, y)


def compare_models(results: List[Dict[str, Any]], 
                  primary_metric: Optional[str] = None) -> pd.DataFrame:
    """Convenience function to compare models.
    
    Args:
        results: List of evaluation results
        primary_metric: Primary metric for comparison
        
    Returns:
        Model comparison DataFrame
    """
    evaluator = ModelEvaluator()
    return evaluator.compare_models(results, primary_metric)


def get_best_model(results: List[Dict[str, Any]], 
                  primary_metric: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to get the best model.
    
    Args:
        results: List of evaluation results
        primary_metric: Primary metric for selection
        
    Returns:
        Best model result
    """
    evaluator = ModelEvaluator()
    return evaluator.get_best_model(results, primary_metric)
