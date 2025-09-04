"""
Model Evaluation and Metrics Module

This module provides model evaluation and metrics functionality
for the autonomous ML agent.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, log_loss
)
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Model evaluation utilities."""
    
    def __init__(self, model: BaseEstimator = None):
        """
        Initialize model evaluator.
        
        Args:
            model: Model to evaluate (optional)
        """
        self.model = model
        
    def evaluate_classification_model(self, model: BaseEstimator, X_test: np.ndarray, 
                                   y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate a classification model."""
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        # ROC AUC for binary classification
        if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
                metrics['log_loss'] = log_loss(y_test, y_proba)
            except:
                metrics['roc_auc'] = None
                metrics['log_loss'] = None
        
        # Per-class metrics for multiclass
        if len(np.unique(y_test)) > 2:
            try:
                metrics['precision_macro'] = precision_score(y_test, y_pred, average='macro')
                metrics['recall_macro'] = recall_score(y_test, y_pred, average='macro')
                metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro')
            except:
                metrics['precision_macro'] = None
                metrics['recall_macro'] = None
                metrics['f1_macro'] = None
        
        return metrics
    
    def evaluate_regression_model(self, model: BaseEstimator, X_test: np.ndarray, 
                               y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate a regression model."""
        y_pred = model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        # MAPE (avoid division by zero)
        try:
            mape = mean_absolute_percentage_error(y_test, y_pred)
            metrics['mape'] = mape
        except:
            metrics['mape'] = None
        
        # Additional metrics
        metrics['mae_percentage'] = np.mean(np.abs((y_test - y_pred) / y_test)) * 100 if np.any(y_test != 0) else None
        
        return metrics
    
    def get_confusion_matrix(self, model: BaseEstimator, X_test: np.ndarray, 
                           y_test: np.ndarray) -> np.ndarray:
        """Get confusion matrix for classification."""
        y_pred = model.predict(X_test)
        return confusion_matrix(y_test, y_pred)
    
    def get_classification_report(self, model: BaseEstimator, X_test: np.ndarray, 
                                y_test: np.ndarray) -> str:
        """Get detailed classification report."""
        y_pred = model.predict(X_test)
        return classification_report(y_test, y_pred)
    
    def cross_validate_model(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                           cv: int = 5, scoring: str = 'accuracy') -> Dict[str, float]:
        """Perform cross-validation."""
        from sklearn.model_selection import cross_val_score
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'min_score': scores.min(),
            'max_score': scores.max(),
            'scores': scores.tolist()
        }
    
    def evaluate_model(self, model: BaseEstimator, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model and return comprehensive results."""
        # Determine if classification or regression
        if hasattr(model, 'predict_proba'):
            # Classification
            metrics = self.evaluate_classification_model(model, X_test, y_test)
            confusion_mat = self.get_confusion_matrix(model, X_test, y_test)
            classification_rep = self.get_classification_report(model, X_test, y_test)
            
            results = {
                'task_type': 'classification',
                'metrics': metrics,
                'confusion_matrix': confusion_mat.tolist(),
                'classification_report': classification_rep
            }
        else:
            # Regression
            metrics = self.evaluate_regression_model(model, X_test, y_test)
            
            results = {
                'task_type': 'regression',
                'metrics': metrics
            }
        
        # Add predictions
        y_pred = model.predict(X_test)
        results['predictions'] = y_pred.tolist()
        results['true_values'] = y_test.tolist()
        
        return results


class MetricsCalculator:
    """Calculate various metrics for model evaluation."""
    
    @staticmethod
    def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy."""
        return accuracy_score(y_true, y_pred)
    
    @staticmethod
    def calculate_precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
        """Calculate precision."""
        return precision_score(y_true, y_pred, average=average)
    
    @staticmethod
    def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
        """Calculate recall."""
        return recall_score(y_true, y_pred, average=average)
    
    @staticmethod
    def calculate_f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
        """Calculate F1 score."""
        return f1_score(y_true, y_pred, average=average)
    
    @staticmethod
    def calculate_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Calculate ROC AUC score."""
        return roc_auc_score(y_true, y_proba)
    
    @staticmethod
    def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate mean squared error."""
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate root mean squared error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate mean absolute error."""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² score."""
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate mean absolute percentage error."""
        return mean_absolute_percentage_error(y_true, y_pred)


class ModelComparison:
    """Compare multiple models."""
    
    def __init__(self):
        self.results = {}
        
    def add_model(self, name: str, model: BaseEstimator, X_test: np.ndarray, y_test: np.ndarray):
        """Add a model for comparison."""
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(model, X_test, y_test)
        self.results[name] = results
        
    def get_comparison_table(self) -> pd.DataFrame:
        """Get comparison table of all models."""
        comparison_data = []
        
        for name, results in self.results.items():
            row = {'model': name, 'task_type': results['task_type']}
            
            # Add metrics
            for metric_name, metric_value in results['metrics'].items():
                if metric_value is not None:
                    row[metric_name] = metric_value
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_best_model(self, metric: str = 'accuracy') -> str:
        """Get the best model based on a specific metric."""
        if not self.results:
            return None
        
        best_score = -np.inf
        best_model = None
        
        for name, results in self.results.items():
            if metric in results['metrics'] and results['metrics'][metric] is not None:
                score = results['metrics'][metric]
                if score > best_score:
                    best_score = score
                    best_model = name
        
        return best_model
    
    def plot_comparison(self, metrics: List[str] = None):
        """Plot comparison of models."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            df = self.get_comparison_table()
            
            if metrics is None:
                # Get numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                metrics = [col for col in numeric_cols if col != 'task_type']
            
            if len(metrics) == 0:
                logger.warning("No metrics to plot")
                return
            
            # Create subplots
            fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
            if len(metrics) == 1:
                axes = [axes]
            
            for i, metric in enumerate(metrics):
                if metric in df.columns:
                    df.plot(x='model', y=metric, kind='bar', ax=axes[i], title=metric.upper())
                    axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib not available for plotting")
        except Exception as e:
            logger.warning(f"Could not create plot: {e}")


class StatisticalSignificance:
    """Statistical significance testing for model comparison."""
    
    @staticmethod
    def mcnemar_test(model1_pred: np.ndarray, model2_pred: np.ndarray, 
                    y_true: np.ndarray) -> Dict[str, float]:
        """Perform McNemar's test for model comparison."""
        from scipy.stats import chi2
        
        # Create contingency table
        correct1 = (model1_pred == y_true)
        correct2 = (model2_pred == y_true)
        
        # Contingency table
        both_correct = np.sum(correct1 & correct2)
        both_incorrect = np.sum(~correct1 & ~correct2)
        only1_correct = np.sum(correct1 & ~correct2)
        only2_correct = np.sum(~correct1 & correct2)
        
        # McNemar's test statistic
        if only1_correct + only2_correct == 0:
            chi2_stat = 0
            p_value = 1.0
        else:
            chi2_stat = (abs(only1_correct - only2_correct) - 1) ** 2 / (only1_correct + only2_correct)
            p_value = 1 - chi2.cdf(chi2_stat, 1)
        
        return {
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def paired_t_test(scores1: np.ndarray, scores2: np.ndarray) -> Dict[str, float]:
        """Perform paired t-test for model comparison."""
        from scipy.stats import ttest_rel
        
        t_stat, p_value = ttest_rel(scores1, scores2)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }


def evaluate_model_performance(model: BaseEstimator, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate model performance and return comprehensive results.
    
    Args:
        model: Model to evaluate
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary with evaluation results
    """
    evaluator = ModelEvaluator()
    return evaluator.evaluate_model(model, X_test, y_test)


def compare_models(models: Dict[str, BaseEstimator], X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """
    Compare multiple models.
    
    Args:
        models: Dictionary of models
        X_test: Test features
        y_test: Test targets
        
    Returns:
        DataFrame with comparison results
    """
    comparison = ModelComparison()
    
    for name, model in models.items():
        comparison.add_model(name, model, X_test, y_test)
    
    return comparison.get_comparison_table()


def get_best_model(models: Dict[str, BaseEstimator], X_test: np.ndarray, y_test: np.ndarray,
                  metric: str = 'accuracy') -> str:
    """
    Get the best model based on a specific metric.
    
    Args:
        models: Dictionary of models
        X_test: Test features
        y_test: Test targets
        metric: Metric to optimize
        
    Returns:
        Name of the best model
    """
    comparison = ModelComparison()
    
    for name, model in models.items():
        comparison.add_model(name, model, X_test, y_test)
    
    return comparison.get_best_model(metric)
