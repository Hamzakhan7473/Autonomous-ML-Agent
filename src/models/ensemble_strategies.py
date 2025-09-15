"""
LLM-Powered Ensemble Strategy Selection

This module implements intelligent ensemble strategy selection using LLMs
to analyze model performance and recommend optimal ensemble approaches.
"""

import json
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from src.ensemble.blending import EnsembleBlender, BlendingConfig
from src.models.meta_learning import MetaLearningEngine

logger = logging.getLogger(__name__)


@dataclass
class EnsembleStrategy:
    """Configuration for ensemble strategy."""
    
    method: str  # 'weighted', 'stacking', 'voting', 'blending'
    models: List[str]  # List of model names to include
    weights: Optional[List[float]] = None
    meta_model: str = "linear"
    cv_folds: int = 5
    performance_threshold: float = 0.7
    diversity_threshold: float = 0.3
    complexity_weight: float = 0.3
    performance_weight: float = 0.5
    speed_weight: float = 0.2


@dataclass
class ModelPerformance:
    """Model performance metrics for ensemble analysis."""
    
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    prediction_time: float
    feature_importance: Optional[Dict[str, float]] = None
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None


class EnsembleAnalyzer:
    """Analyzer for ensemble strategy recommendations."""
    
    def __init__(self, meta_learning_engine: MetaLearningEngine = None):
        """Initialize the ensemble analyzer."""
        self.meta_learning_engine = meta_learning_engine or MetaLearningEngine()
    
    def analyze_model_performance(self, leaderboard: pd.DataFrame) -> List[ModelPerformance]:
        """Analyze model performance from leaderboard data."""
        
        model_performances = []
        
        for _, row in leaderboard.iterrows():
            performance = ModelPerformance(
                model_name=row["model_name"],
                accuracy=row.get("accuracy", 0.0),
                precision=row.get("precision", 0.0),
                recall=row.get("recall", 0.0),
                f1_score=row.get("f1_score", 0.0),
                training_time=row.get("training_time", 0.0),
                prediction_time=row.get("prediction_time", 0.0),
                feature_importance=row.get("feature_importance", {}),
                predictions=row.get("predictions"),
                probabilities=row.get("probabilities")
            )
            model_performances.append(performance)
        
        return model_performances
    
    def calculate_model_diversity(self, model_performances: List[ModelPerformance]) -> Dict[str, float]:
        """Calculate diversity between models based on predictions."""
        
        diversity_scores = {}
        
        if len(model_performances) < 2:
            return diversity_scores
        
        # Calculate pairwise diversity based on predictions
        for i, model1 in enumerate(model_performances):
            for j, model2 in enumerate(model_performances[i+1:], i+1):
                if model1.predictions is not None and model2.predictions is not None:
                    # Calculate agreement rate
                    agreement = np.mean(model1.predictions == model2.predictions)
                    diversity = 1.0 - agreement
                    
                    pair_name = f"{model1.model_name}_vs_{model2.model_name}"
                    diversity_scores[pair_name] = diversity
        
        return diversity_scores
    
    def calculate_model_complexity(self, model_performances: List[ModelPerformance]) -> Dict[str, float]:
        """Calculate model complexity scores."""
        
        complexity_scores = {}
        
        for performance in model_performances:
            # Complexity based on training time and model type
            base_complexity = {
                "logistic_regression": 0.1,
                "linear_regression": 0.1,
                "knn": 0.2,
                "naive_bayes": 0.2,
                "decision_tree": 0.3,
                "random_forest": 0.6,
                "gradient_boosting": 0.7,
                "xgboost": 0.8,
                "lightgbm": 0.8,
                "mlp": 0.9,
                "svm": 0.8
            }
            
            model_name = performance.model_name.lower().replace(" ", "_")
            complexity = base_complexity.get(model_name, 0.5)
            
            # Adjust based on training time
            if performance.training_time > 10:
                complexity += 0.2
            elif performance.training_time > 5:
                complexity += 0.1
            
            complexity_scores[performance.model_name] = min(1.0, complexity)
        
        return complexity_scores
    
    def recommend_ensemble_strategy(
        self, 
        model_performances: List[ModelPerformance],
        meta_features: Dict[str, Any],
        llm_client = None,
        strategy_config: EnsembleStrategy = None
    ) -> EnsembleStrategy:
        """Recommend optimal ensemble strategy based on analysis."""
        
        # Analyze model characteristics
        diversity_scores = self.calculate_model_diversity(model_performances)
        complexity_scores = self.calculate_model_complexity(model_performances)
        
        # Filter models by performance threshold
        config = strategy_config or EnsembleStrategy(
            method="weighted",
            models=[],
            performance_threshold=0.7,
            diversity_threshold=0.3
        )
        
        qualified_models = [
            p for p in model_performances 
            if p.accuracy >= config.performance_threshold
        ]
        
        if not qualified_models:
            # Fall back to top performing models
            qualified_models = sorted(model_performances, key=lambda x: x.accuracy, reverse=True)[:3]
        
        # Calculate ensemble strategy score
        strategy_scores = self._calculate_strategy_scores(
            qualified_models, diversity_scores, complexity_scores, meta_features
        )
        
        # Get LLM recommendations if available
        if llm_client:
            llm_recommendations = self._get_llm_ensemble_recommendations(
                qualified_models, diversity_scores, complexity_scores, meta_features, llm_client
            )
            strategy_scores.update(llm_recommendations)
        
        # Select best strategy
        best_strategy = max(strategy_scores.keys(), key=lambda k: strategy_scores[k]["score"])
        
        # Create ensemble strategy configuration
        ensemble_strategy = EnsembleStrategy(
            method=best_strategy,
            models=[m.model_name for m in qualified_models[:5]],  # Top 5 models
            weights=self._calculate_optimal_weights(qualified_models[:5]),
            meta_model=strategy_scores[best_strategy].get("meta_model", "linear"),
            cv_folds=5,
            performance_threshold=config.performance_threshold,
            diversity_threshold=config.diversity_threshold
        )
        
        logger.info(f"Recommended ensemble strategy: {best_strategy} with models: {ensemble_strategy.models}")
        
        return ensemble_strategy
    
    def _calculate_strategy_scores(
        self, 
        models: List[ModelPerformance], 
        diversity_scores: Dict[str, float],
        complexity_scores: Dict[str, float],
        meta_features: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate scores for different ensemble strategies."""
        
        strategy_scores = {}
        
        # Dataset characteristics
        num_instances = meta_features.get("num_instances", 1000)
        num_features = meta_features.get("num_features", 10)
        complexity_ratio = num_features / max(num_instances, 1)
        
        # Weighted ensemble score
        avg_performance = np.mean([m.accuracy for m in models])
        avg_diversity = np.mean(list(diversity_scores.values())) if diversity_scores else 0.5
        avg_complexity = np.mean([complexity_scores[m.model_name] for m in models])
        
        strategy_scores["weighted"] = {
            "score": avg_performance * 0.6 + avg_diversity * 0.2 + (1 - avg_complexity) * 0.2,
            "meta_model": "linear",
            "reasoning": "Simple weighted average, good for similar performance models"
        }
        
        # Stacking ensemble score
        stacking_score = avg_performance * 0.5 + avg_diversity * 0.4 + (1 - avg_complexity * 0.8) * 0.1
        strategy_scores["stacking"] = {
            "score": stacking_score,
            "meta_model": "ridge" if complexity_ratio > 0.1 else "linear",
            "reasoning": "Meta-learning approach, good for diverse models"
        }
        
        # Voting ensemble score
        voting_score = avg_performance * 0.4 + avg_diversity * 0.5 + (1 - avg_complexity * 0.6) * 0.1
        strategy_scores["voting"] = {
            "score": voting_score,
            "meta_model": "none",
            "reasoning": "Democratic voting, good for high diversity"
        }
        
        # Blending ensemble score
        blending_score = avg_performance * 0.7 + avg_diversity * 0.2 + (1 - avg_complexity * 0.9) * 0.1
        strategy_scores["blending"] = {
            "score": blending_score,
            "meta_model": "lasso" if num_features > 20 else "ridge",
            "reasoning": "Advanced blending, good for high performance models"
        }
        
        return strategy_scores
    
    def _calculate_optimal_weights(self, models: List[ModelPerformance]) -> List[float]:
        """Calculate optimal weights for ensemble models."""
        
        # Weight based on performance and inverse complexity
        weights = []
        
        for model in models:
            # Performance weight (accuracy)
            perf_weight = model.accuracy
            
            # Complexity penalty (higher complexity = lower weight)
            complexity_penalty = 1.0 / (1.0 + model.training_time / 10.0)
            
            # Combined weight
            weight = perf_weight * complexity_penalty
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        return weights
    
    def _get_llm_ensemble_recommendations(
        self, 
        models: List[ModelPerformance],
        diversity_scores: Dict[str, float],
        complexity_scores: Dict[str, float],
        meta_features: Dict[str, Any],
        llm_client
    ) -> Dict[str, Dict[str, Any]]:
        """Get LLM-powered ensemble strategy recommendations."""
        
        try:
            prompt = self._create_ensemble_analysis_prompt(
                models, diversity_scores, complexity_scores, meta_features
            )
            
            llm_response = llm_client.generate_response(prompt)
            
            # Parse LLM response
            recommendations = self._parse_llm_ensemble_response(llm_response)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get LLM ensemble recommendations: {e}")
            return {}
    
    def _create_ensemble_analysis_prompt(
        self, 
        models: List[ModelPerformance],
        diversity_scores: Dict[str, float],
        complexity_scores: Dict[str, float],
        meta_features: Dict[str, Any]
    ) -> str:
        """Create prompt for LLM ensemble analysis."""
        
        prompt = f"""
You are an expert machine learning engineer specializing in ensemble methods. Analyze the following model performance data and recommend the optimal ensemble strategy.

Dataset Characteristics:
- Number of instances: {meta_features.get('num_instances', 'unknown')}
- Number of features: {meta_features.get('num_features', 'unknown')}
- Missing values ratio: {meta_features.get('missing_values_ratio', 0):.2f}
- Categorical features ratio: {meta_features.get('categorical_features_ratio', 0):.2f}

Model Performance Analysis:
"""
        
        for model in models:
            prompt += f"""
{model.model_name}:
  - Accuracy: {model.accuracy:.3f}
  - Precision: {model.precision:.3f}
  - Recall: {model.recall:.3f}
  - F1 Score: {model.f1_score:.3f}
  - Training Time: {model.training_time:.2f}s
  - Complexity Score: {complexity_scores.get(model.model_name, 0.5):.2f}
"""
        
        prompt += f"""
Model Diversity Analysis:
- Average diversity: {np.mean(list(diversity_scores.values())) if diversity_scores else 0:.3f}
- Diversity range: {min(diversity_scores.values()) if diversity_scores else 0:.3f} - {max(diversity_scores.values()) if diversity_scores else 0:.3f}

Please recommend the best ensemble strategy considering:
1. Model performance and accuracy
2. Diversity between models
3. Computational complexity and speed
4. Dataset characteristics
5. Balance between performance and interpretability

Available strategies:
- weighted: Simple weighted average of predictions
- stacking: Meta-learning approach with cross-validation
- voting: Majority or soft voting
- blending: Advanced blending with regularization

Provide your recommendation in JSON format:
{{
  "recommended_strategy": "strategy_name",
  "confidence": 0.0-1.0,
  "reasoning": "explanation",
  "alternative_strategies": [
    {{
      "strategy": "strategy_name",
      "score": 0.0-1.0,
      "reasoning": "explanation"
    }}
  ]
}}
"""
        
        return prompt
    
    def _parse_llm_ensemble_response(self, llm_response: str) -> Dict[str, Dict[str, Any]]:
        """Parse LLM response for ensemble recommendations."""
        
        try:
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                recommendations = json.loads(json_match.group())
                
                # Convert to strategy scores format
                strategy_scores = {}
                
                recommended_strategy = recommendations.get("recommended_strategy", "weighted")
                confidence = recommendations.get("confidence", 0.8)
                
                strategy_scores[recommended_strategy] = {
                    "score": confidence,
                    "meta_model": "linear",
                    "reasoning": recommendations.get("reasoning", "LLM recommendation"),
                    "llm_enhanced": True
                }
                
                # Add alternative strategies
                for alt in recommendations.get("alternative_strategies", []):
                    strategy_scores[alt["strategy"]] = {
                        "score": alt.get("score", 0.7),
                        "meta_model": "linear",
                        "reasoning": alt.get("reasoning", "LLM alternative"),
                        "llm_enhanced": True
                    }
                
                return strategy_scores
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to parse LLM ensemble response: {e}")
            return {}


class EnsembleExecutor:
    """Executor for ensemble strategies."""
    
    def __init__(self, ensemble_analyzer: EnsembleAnalyzer):
        """Initialize the ensemble executor."""
        self.analyzer = ensemble_analyzer
    
    def execute_ensemble_strategy(
        self, 
        strategy: EnsembleStrategy,
        trained_models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[Any, Dict[str, Any]]:
        """Execute the recommended ensemble strategy."""
        
        try:
            # Create ensemble blender
            blending_config = BlendingConfig(
                method=strategy.method,
                weights=strategy.weights,
                meta_model=strategy.meta_model,
                cv_folds=strategy.cv_folds
            )
            
            ensemble_blender = EnsembleBlender(blending_config)
            
            # Get models for ensemble
            ensemble_models = [trained_models[name] for name in strategy.models if name in trained_models]
            
            if not ensemble_models:
                logger.warning("No models available for ensemble")
                return None, {"error": "No models available"}
            
            # Train ensemble
            fitted_ensemble = ensemble_blender.blend_models(ensemble_models, X, y, strategy.method)
            
            # Evaluate ensemble
            ensemble_metrics = self._evaluate_ensemble(fitted_ensemble, X, y)
            
            ensemble_info = {
                "strategy": strategy.method,
                "models": strategy.models,
                "weights": strategy.weights,
                "metrics": ensemble_metrics,
                "config": blending_config.__dict__
            }
            
            logger.info(f"Ensemble executed successfully with {strategy.method} strategy")
            
            return fitted_ensemble, ensemble_info
            
        except Exception as e:
            logger.error(f"Failed to execute ensemble strategy: {e}")
            return None, {"error": str(e)}
    
    def _evaluate_ensemble(self, ensemble, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate ensemble performance."""
        
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # Make predictions
            predictions = ensemble.predict(X)
            
            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y, predictions),
                "precision": precision_score(y, predictions, average='weighted'),
                "recall": recall_score(y, predictions, average='weighted'),
                "f1_score": f1_score(y, predictions, average='weighted')
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate ensemble: {e}")
            return {"error": str(e)}


class EnsembleStrategyManager:
    """Main manager for ensemble strategy selection and execution."""
    
    def __init__(self, meta_learning_engine: MetaLearningEngine = None):
        """Initialize the ensemble strategy manager."""
        self.analyzer = EnsembleAnalyzer(meta_learning_engine)
        self.executor = EnsembleExecutor(self.analyzer)
    
    def recommend_and_execute_ensemble(
        self, 
        leaderboard: pd.DataFrame,
        trained_models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        meta_features: Dict[str, Any],
        llm_client = None,
        strategy_config: EnsembleStrategy = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Recommend and execute optimal ensemble strategy."""
        
        # Analyze model performance
        model_performances = self.analyzer.analyze_model_performance(leaderboard)
        
        # Recommend ensemble strategy
        recommended_strategy = self.analyzer.recommend_ensemble_strategy(
            model_performances, meta_features, llm_client, strategy_config
        )
        
        # Execute ensemble strategy
        ensemble_model, ensemble_info = self.executor.execute_ensemble_strategy(
            recommended_strategy, trained_models, X, y
        )
        
        return ensemble_model, {
            "strategy": recommended_strategy,
            "model_performances": model_performances,
            "ensemble_info": ensemble_info,
            "recommendation_reasoning": recommended_strategy.__dict__
        }
