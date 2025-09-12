"""
Leaderboard Module

This module provides a comprehensive leaderboard for displaying model performance,
insights, and comparisons in both CLI and web UI formats.
"""

import json
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Result for a single model."""
    
    model_name: str
    model_type: str
    best_score: float
    best_params: Dict[str, Any]
    metrics: Dict[str, float]
    training_time: float
    feature_importance: Optional[Dict[str, float]] = None
    cross_validation_scores: Optional[List[float]] = None
    test_predictions: Optional[np.ndarray] = None
    test_probabilities: Optional[np.ndarray] = None
    model_size_mb: Optional[float] = None
    inference_time_ms: Optional[float] = None


@dataclass
class LeaderboardConfig:
    """Configuration for the leaderboard."""
    
    primary_metric: str = "accuracy"
    secondary_metrics: List[str] = None
    show_feature_importance: bool = True
    show_training_time: bool = True
    show_model_size: bool = True
    show_inference_time: bool = True
    max_models_display: int = 10
    sort_descending: bool = True
    
    def __post_init__(self):
        if self.secondary_metrics is None:
            self.secondary_metrics = ["f1", "precision", "recall"]


class ModelLeaderboard:
    """Comprehensive model leaderboard."""
    
    def __init__(self, config: LeaderboardConfig = None):
        """Initialize the leaderboard.
        
        Args:
            config: Leaderboard configuration
        """
        self.config = config or LeaderboardConfig()
        self.results: List[ModelResult] = []
        self.best_model: Optional[ModelResult] = None
        self.task_type: Optional[str] = None
        
    def add_result(self, result: ModelResult):
        """Add a model result to the leaderboard.
        
        Args:
            result: Model result to add
        """
        self.results.append(result)
        
        # Update best model
        if self.best_model is None:
            self.best_model = result
        else:
            if self._is_better_score(result.best_score, self.best_model.best_score):
                self.best_model = result
        
        # Determine task type from first result
        if self.task_type is None:
            self.task_type = self._determine_task_type(result)
        
        logger.info(f"Added result for {result.model_name}: {result.best_score:.4f}")
    
    def _is_better_score(self, score1: float, score2: float) -> bool:
        """Determine if score1 is better than score2."""
        if self.config.sort_descending:
            return score1 > score2
        else:
            return score1 < score2
    
    def _determine_task_type(self, result: ModelResult) -> str:
        """Determine task type from model result."""
        if result.test_probabilities is not None:
            return "classification"
        elif "mse" in result.metrics or "rmse" in result.metrics:
            return "regression"
        else:
            return "unknown"
    
    def get_leaderboard_df(self) -> pd.DataFrame:
        """Get leaderboard as a pandas DataFrame.
        
        Returns:
            DataFrame with model results sorted by primary metric
        """
        if not self.results:
            return pd.DataFrame()
        
        # Prepare data for DataFrame
        data = []
        for result in self.results:
            row = {
                "rank": 0,  # Will be set after sorting
                "model_name": result.model_name,
                "model_type": result.model_type,
                "primary_score": result.best_score,
                "training_time": result.training_time,
                "model_size_mb": result.model_size_mb or 0,
                "inference_time_ms": result.inference_time_ms or 0,
            }
            
            # Add metrics
            for metric, value in result.metrics.items():
                row[f"metric_{metric}"] = value
            
            # Add cross-validation stats if available
            if result.cross_validation_scores:
                row["cv_mean"] = np.mean(result.cross_validation_scores)
                row["cv_std"] = np.std(result.cross_validation_scores)
                row["cv_min"] = np.min(result.cross_validation_scores)
                row["cv_max"] = np.max(result.cross_validation_scores)
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Sort by primary metric
        df = df.sort_values("primary_score", ascending=not self.config.sort_descending)
        
        # Add rank
        df["rank"] = range(1, len(df) + 1)
        
        # Reorder columns
        columns = ["rank", "model_name", "model_type", "primary_score"]
        if self.config.show_training_time:
            columns.append("training_time")
        if self.config.show_model_size:
            columns.append("model_size_mb")
        if self.config.show_inference_time:
            columns.append("inference_time_ms")
        
        # Add metric columns
        metric_columns = [col for col in df.columns if col.startswith("metric_")]
        columns.extend(sorted(metric_columns))
        
        # Add CV columns
        cv_columns = [col for col in df.columns if col.startswith("cv_")]
        columns.extend(sorted(cv_columns))
        
        df = df[columns]
        
        return df
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all models.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {}
        
        scores = [result.best_score for result in self.results]
        training_times = [result.training_time for result in self.results]
        
        summary = {
            "total_models": len(self.results),
            "best_model": self.best_model.model_name if self.best_model else None,
            "best_score": self.best_model.best_score if self.best_model else None,
            "score_stats": {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "median": np.median(scores),
            },
            "training_time_stats": {
                "mean": np.mean(training_times),
                "std": np.std(training_times),
                "min": np.min(training_times),
                "max": np.max(training_times),
                "total": np.sum(training_times),
            },
            "task_type": self.task_type,
        }
        
        return summary
    
    def print_cli_leaderboard(self):
        """Print leaderboard in CLI format."""
        if not self.results:
            print("No results available.")
            return
        
        df = self.get_leaderboard_df()
        
        print("\n" + "="*80)
        print("MODEL LEADERBOARD")
        print("="*80)
        
        # Print summary
        summary = self.get_summary_stats()
        print(f"\nSummary:")
        print(f"  Total Models: {summary['total_models']}")
        print(f"  Best Model: {summary['best_model']}")
        print(f"  Best Score: {summary['best_score']:.4f}")
        print(f"  Task Type: {summary['task_type']}")
        
        # Print leaderboard table
        print(f"\nLeaderboard (sorted by {self.config.primary_metric}):")
        print("-" * 80)
        
        # Format and print DataFrame
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 20)
        
        print(df.to_string(index=False, float_format='%.4f'))
    
    def get_model_insights(self, llm_client=None) -> str:
        """Generate natural language insights about the models.
        
        Args:
            llm_client: LLM client for generating insights
            
        Returns:
            Natural language insights
        """
        if not self.results:
            return "No model results available for analysis."
        
        summary = self.get_summary_stats()
        
        insights = f"""
        Model Performance Analysis:
        
        We evaluated {summary['total_models']} models on this {summary['task_type']} task.
        The best performing model was {summary['best_model']} with a score of {summary['best_score']:.4f}.
        
        Score Distribution:
        - Mean: {summary['score_stats']['mean']:.4f}
        - Standard Deviation: {summary['score_stats']['std']:.4f}
        - Range: {summary['score_stats']['min']:.4f} to {summary['score_stats']['max']:.4f}
        
        Training Efficiency:
        - Total training time: {summary['training_time_stats']['total']:.2f} seconds
        - Average training time: {summary['training_time_stats']['mean']:.2f} seconds
        """
        
        if llm_client:
            try:
                # Generate more detailed insights using LLM
                prompt = f"""
                Analyze these machine learning model results and provide insights:
                
                {insights}
                
                Please provide:
                1. Key performance patterns
                2. Model strengths and weaknesses
                3. Recommendations for improvement
                4. Deployment considerations
                
                Keep the response concise and actionable.
                """
                
                llm_insights = llm_client.generate_response(prompt)
                insights += f"\n\nDetailed Analysis:\n{llm_insights}"
            except Exception as e:
                logger.warning(f"Failed to generate LLM insights: {e}")
        
        return insights


# Alias for backward compatibility
Leaderboard = ModelLeaderboard