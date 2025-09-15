"""
Model Interpretability and Feature Importance Analysis

This module provides comprehensive model interpretability features including
feature importance analysis and model explanations.
"""

import json
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.tree import export_text

logger = logging.getLogger(__name__)


@dataclass
class InterpretabilityConfig:
    """Configuration for model interpretability analysis."""
    
    feature_importance_methods: List[str] = None  # ["permutation", "shap", "coefficient"]
    max_features_to_show: int = 10
    generate_plots: bool = True
    plot_format: str = "png"  # "png", "svg", "pdf"
    plot_dpi: int = 300
    generate_explanations: bool = True
    explanation_depth: str = "detailed"  # "brief", "detailed", "comprehensive"


@dataclass
class FeatureImportance:
    """Feature importance analysis results."""
    
    feature_names: List[str]
    importance_scores: List[float]
    method: str
    model_name: str
    rank: int = 0


@dataclass
class ModelExplanation:
    """Model explanation and interpretability results."""
    
    model_name: str
    model_type: str
    feature_importance: Optional[FeatureImportance] = None
    partial_dependence: Optional[Dict[str, Any]] = None
    decision_path: Optional[str] = None
    confidence_analysis: Optional[Dict[str, Any]] = None
    error_analysis: Optional[Dict[str, Any]] = None


class FeatureImportanceAnalyzer:
    """Analyzer for feature importance across different models."""
    
    def __init__(self, config: InterpretabilityConfig = None):
        """Initialize the feature importance analyzer."""
        self.config = config or InterpretabilityConfig()
        self.methods = self.config.feature_importance_methods or ["permutation", "coefficient"]
    
    def analyze_feature_importance(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        y: pd.Series, 
        model_name: str,
        method: str = "auto"
    ) -> FeatureImportance:
        """Analyze feature importance for a given model."""
        
        try:
            if method == "auto":
                method = self._select_best_method(model, model_name)
            
            if method == "permutation":
                importance_scores = self._permutation_importance(model, X, y)
            elif method == "coefficient":
                importance_scores = self._coefficient_importance(model, X.columns.tolist())
            elif method == "tree_based":
                importance_scores = self._tree_based_importance(model, X.columns.tolist())
            else:
                logger.warning(f"Unknown importance method: {method}")
                importance_scores = self._default_importance(X.columns.tolist())
            
            # Sort features by importance
            feature_importance_pairs = list(zip(X.columns.tolist(), importance_scores))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            feature_names = [pair[0] for pair in feature_importance_pairs]
            importance_values = [pair[1] for pair in feature_importance_pairs]
            
            return FeatureImportance(
                feature_names=feature_names,
                importance_scores=importance_values,
                method=method,
                model_name=model_name
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze feature importance for {model_name}: {e}")
            return self._default_importance(X.columns.tolist(), model_name, method)
    
    def _select_best_method(self, model: Any, model_name: str) -> str:
        """Select the best feature importance method for the model."""
        
        model_type = type(model).__name__.lower()
        
        if "tree" in model_type or "forest" in model_type or "boost" in model_type:
            return "tree_based"
        elif "linear" in model_type or "logistic" in model_type or "regression" in model_type:
            return "coefficient"
        elif "neural" in model_type or "mlp" in model_type:
            return "permutation"
        else:
            return "permutation"
    
    def _permutation_importance(self, model: Any, X: pd.DataFrame, y: pd.Series) -> List[float]:
        """Calculate permutation importance."""
        
        try:
            # Use sklearn's permutation importance
            perm_importance = permutation_importance(
                model, X, y, n_repeats=5, random_state=42, n_jobs=-1
            )
            return perm_importance.importances_mean.tolist()
            
        except Exception as e:
            logger.error(f"Failed to calculate permutation importance: {e}")
            return [0.0] * len(X.columns)
    
    def _coefficient_importance(self, model: Any, feature_names: List[str]) -> List[float]:
        """Calculate coefficient-based importance for linear models."""
        
        try:
            if hasattr(model, "coef_"):
                if model.coef_.ndim == 1:
                    # Binary classification or regression
                    importance = np.abs(model.coef_)
                else:
                    # Multi-class classification
                    importance = np.abs(model.coef_).mean(axis=0)
                
                return importance.tolist()
            else:
                logger.warning("Model does not have coefficients")
                return [0.0] * len(feature_names)
                
        except Exception as e:
            logger.error(f"Failed to calculate coefficient importance: {e}")
            return [0.0] * len(feature_names)
    
    def _tree_based_importance(self, model: Any, feature_names: List[str]) -> List[float]:
        """Calculate tree-based feature importance."""
        
        try:
            if hasattr(model, "feature_importances_"):
                return model.feature_importances_.tolist()
            else:
                logger.warning("Model does not have feature_importances_")
                return [0.0] * len(feature_names)
                
        except Exception as e:
            logger.error(f"Failed to calculate tree-based importance: {e}")
            return [0.0] * len(feature_names)
    
    def _default_importance(self, feature_names: List[str], model_name: str = "unknown", method: str = "default") -> FeatureImportance:
        """Create default feature importance when analysis fails."""
        
        return FeatureImportance(
            feature_names=feature_names,
            importance_scores=[1.0 / len(feature_names)] * len(feature_names),
            method=method,
            model_name=model_name
        )


class ModelExplainer:
    """Comprehensive model explainer for different model types."""
    
    def __init__(self, config: InterpretabilityConfig = None):
        """Initialize the model explainer."""
        self.config = config or InterpretabilityConfig()
    
    def explain_model(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        y: pd.Series, 
        model_name: str,
        feature_analyzer: FeatureImportanceAnalyzer
    ) -> ModelExplanation:
        """Generate comprehensive model explanation."""
        
        try:
            model_type = self._get_model_type(model)
            
            # Feature importance analysis
            feature_importance = feature_analyzer.analyze_feature_importance(
                model, X, y, model_name
            )
            
            # Partial dependence analysis
            partial_dependence_data = self._analyze_partial_dependence(
                model, X, feature_importance
            )
            
            # Decision path analysis (for tree-based models)
            decision_path = self._analyze_decision_path(model, model_type)
            
            # Confidence analysis
            confidence_analysis = self._analyze_model_confidence(model, X, y)
            
            # Error analysis
            error_analysis = self._analyze_model_errors(model, X, y)
            
            return ModelExplanation(
                model_name=model_name,
                model_type=model_type,
                feature_importance=feature_importance,
                partial_dependence=partial_dependence_data,
                decision_path=decision_path,
                confidence_analysis=confidence_analysis,
                error_analysis=error_analysis
            )
            
        except Exception as e:
            logger.error(f"Failed to explain model {model_name}: {e}")
            return ModelExplanation(
                model_name=model_name,
                model_type="unknown",
                feature_importance=feature_analyzer._default_importance(X.columns.tolist(), model_name)
            )
    
    def _get_model_type(self, model: Any) -> str:
        """Get the type of the model."""
        
        model_type = type(model).__name__.lower()
        
        if "tree" in model_type or "forest" in model_type:
            return "tree_based"
        elif "linear" in model_type or "logistic" in model_type or "regression" in model_type:
            return "linear"
        elif "neural" in model_type or "mlp" in model_type:
            return "neural_network"
        elif "svm" in model_type:
            return "svm"
        elif "naive" in model_type:
            return "naive_bayes"
        elif "knn" in model_type or "neighbors" in model_type:
            return "knn"
        else:
            return "unknown"
    
    def _analyze_partial_dependence(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        feature_importance: FeatureImportance
    ) -> Dict[str, Any]:
        """Analyze partial dependence for top features."""
        
        try:
            # Select top features for partial dependence
            top_features = feature_importance.feature_names[:3]
            
            partial_dependence_data = {}
            
            for feature in top_features:
                if feature in X.columns:
                    try:
                        # Calculate partial dependence
                        pd_values = partial_dependence(
                            model, X, [feature], kind='average'
                        )
                        
                        partial_dependence_data[feature] = {
                            "values": pd_values["values"][0].tolist(),
                            "grid": pd_values["grid"][0].tolist(),
                            "feature_name": feature
                        }
                        
                    except Exception as e:
                        logger.warning(f"Failed to calculate partial dependence for {feature}: {e}")
                        continue
            
            return partial_dependence_data
            
        except Exception as e:
            logger.error(f"Failed to analyze partial dependence: {e}")
            return {}
    
    def _analyze_decision_path(self, model: Any, model_type: str) -> Optional[str]:
        """Analyze decision path for tree-based models."""
        
        try:
            if model_type == "tree_based" and hasattr(model, "tree_"):
                # Single decision tree
                tree_rules = export_text(model, feature_names=[f"feature_{i}" for i in range(model.n_features_)])
                return tree_rules[:1000]  # Limit length
            elif model_type == "tree_based" and hasattr(model, "estimators_"):
                # Random forest - get first tree
                first_tree = model.estimators_[0]
                tree_rules = export_text(first_tree, feature_names=[f"feature_{i}" for i in range(model.n_features_)])
                return tree_rules[:1000]  # Limit length
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to analyze decision path: {e}")
            return None
    
    def _analyze_model_confidence(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze model confidence and uncertainty."""
        
        try:
            confidence_analysis = {}
            
            # Get predictions
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X)
                predictions = model.predict(X)
                
                # Calculate confidence metrics
                max_probabilities = np.max(probabilities, axis=1)
                confidence_analysis.update({
                    "mean_confidence": float(np.mean(max_probabilities)),
                    "std_confidence": float(np.std(max_probabilities)),
                    "min_confidence": float(np.min(max_probabilities)),
                    "max_confidence": float(np.max(max_probabilities)),
                    "low_confidence_samples": int(np.sum(max_probabilities < 0.7)),
                    "high_confidence_samples": int(np.sum(max_probabilities > 0.9))
                })
            else:
                predictions = model.predict(X)
                confidence_analysis = {
                    "mean_confidence": 1.0,
                    "std_confidence": 0.0,
                    "note": "Model does not provide probability estimates"
                }
            
            return confidence_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze model confidence: {e}")
            return {"error": str(e)}
    
    def _analyze_model_errors(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze model errors and prediction patterns."""
        
        try:
            predictions = model.predict(X)
            
            # Calculate error metrics
            errors = predictions != y if hasattr(y, '__iter__') else []
            
            error_analysis = {
                "total_errors": int(np.sum(errors)) if len(errors) > 0 else 0,
                "error_rate": float(np.mean(errors)) if len(errors) > 0 else 0.0,
                "correct_predictions": int(np.sum(~errors)) if len(errors) > 0 else len(predictions)
            }
            
            # Analyze error patterns by feature (if categorical features exist)
            categorical_features = X.select_dtypes(include=['object', 'category']).columns
            if len(categorical_features) > 0:
                error_by_category = {}
                for feature in categorical_features[:2]:  # Limit to 2 features
                    error_by_category[feature] = {}
                    for category in X[feature].unique()[:5]:  # Limit to 5 categories
                        category_mask = X[feature] == category
                        category_errors = errors[category_mask] if len(errors) > 0 else []
                        error_by_category[feature][category] = {
                            "error_rate": float(np.mean(category_errors)) if len(category_errors) > 0 else 0.0,
                            "sample_count": int(np.sum(category_mask))
                        }
                
                error_analysis["error_by_category"] = error_by_category
            
            return error_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze model errors: {e}")
            return {"error": str(e)}


class ModelInterpreter:
    """Legacy ModelInterpreter class for backward compatibility."""
    
    def __init__(self, config: InterpretabilityConfig = None):
        """Initialize the model interpreter."""
        self.config = config or InterpretabilityConfig()
        self.feature_analyzer = FeatureImportanceAnalyzer(self.config)
        self.model_explainer = ModelExplainer(self.config)
    
    def explain_model(self, model: Any, X: pd.DataFrame, y: pd.Series, model_name: str) -> dict[str, Any]:
        """Explain a model (legacy interface)."""
        try:
            explanation = self.model_explainer.explain_model(
                model, X, y, model_name, self.feature_analyzer
            )
            return {
                "model_name": explanation.model_name,
                "model_type": explanation.model_type,
                "feature_importance": explanation.feature_importance,
                "confidence_analysis": explanation.confidence_analysis,
                "error_analysis": explanation.error_analysis
            }
        except Exception as e:
            logger.error(f"Failed to explain model {model_name}: {e}")
            return {"error": str(e)}
    
    def get_feature_importance(self, model: Any, X: pd.DataFrame, y: pd.Series, model_name: str) -> dict[str, Any]:
        """Get feature importance for a model."""
        try:
            feature_importance = self.feature_analyzer.analyze_feature_importance(
                model, X, y, model_name
            )
            return {
                "feature_names": feature_importance.feature_names,
                "importance_scores": feature_importance.importance_scores,
                "method": feature_importance.method,
                "model_name": feature_importance.model_name
            }
        except Exception as e:
            logger.error(f"Failed to get feature importance for {model_name}: {e}")
            return {"error": str(e)}