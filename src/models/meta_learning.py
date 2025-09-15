"""
Enhanced Meta-Learning Module for Hyperparameter Warm Starts

This module implements advanced meta-learning capabilities to provide
intelligent hyperparameter initialization based on similar datasets.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning warm starts."""
    
    similarity_threshold: float = 0.7
    max_warm_starts: int = 5
    weight_by_performance: bool = True
    weight_by_recency: bool = True
    min_samples_for_warm_start: int = 3
    feature_importance_weight: float = 0.3
    performance_weight: float = 0.4
    recency_weight: float = 0.3


class MetaLearningEngine:
    """Advanced meta-learning engine for hyperparameter warm starts."""
    
    def __init__(self, config: MetaLearningConfig = None, registry_path: str = "models/registry"):
        """Initialize the meta-learning engine.
        
        Args:
            config: Meta-learning configuration
            registry_path: Path to the model registry
        """
        self.config = config or MetaLearningConfig()
        self.registry_path = Path(registry_path)
        self.meta_features_scaler = StandardScaler()
        self.warm_start_cache = {}
        
    def get_hyperparameter_warm_starts(
        self, 
        meta_features: Dict[str, Any], 
        task_type: str,
        model_name: str,
        llm_client = None
    ) -> Dict[str, Any]:
        """Get intelligent hyperparameter warm starts based on meta-learning.
        
        Args:
            meta_features: Meta-features of the current dataset
            task_type: Type of ML task ('classification' or 'regression')
            model_name: Name of the model to get warm starts for
            llm_client: LLM client for generating recommendations
            
        Returns:
            Dictionary containing warm start configurations
        """
        try:
            # Load historical performance data
            historical_data = self._load_historical_data()
            
            if len(historical_data) < self.config.min_samples_for_warm_start:
                logger.info("Insufficient historical data for meta-learning warm starts")
                return self._get_default_warm_starts(model_name, task_type)
            
            # Find similar datasets
            similar_datasets = self._find_similar_datasets(meta_features, historical_data)
            
            if not similar_datasets:
                logger.info("No similar datasets found for warm starts")
                return self._get_default_warm_starts(model_name, task_type)
            
            # Extract hyperparameter configurations from similar datasets
            warm_starts = self._extract_warm_starts(similar_datasets, model_name, task_type)
            
            # Use LLM to generate intelligent recommendations if available
            if llm_client:
                warm_starts = self._enhance_with_llm_recommendations(
                    warm_starts, meta_features, model_name, task_type, llm_client
                )
            
            logger.info(f"Generated {len(warm_starts)} warm start configurations for {model_name}")
            return warm_starts
            
        except Exception as e:
            logger.error(f"Failed to generate meta-learning warm starts: {e}")
            return self._get_default_warm_starts(model_name, task_type)
    
    def _load_historical_data(self) -> List[Dict[str, Any]]:
        """Load historical model performance data."""
        historical_data = []
        
        try:
            metadata_file = self.registry_path / "metadata.json"
            if not metadata_file.exists():
                return historical_data
            
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            for model_id, model_meta in metadata.items():
                if "meta_features" in model_meta and "performance" in model_meta:
                    historical_data.append({
                        "model_id": model_id,
                        "meta_features": model_meta["meta_features"],
                        "performance": model_meta["performance"],
                        "timestamp": model_meta.get("timestamp", ""),
                        "model_type": model_meta.get("model_type", "")
                    })
                    
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            
        return historical_data
    
    def _find_similar_datasets(
        self, 
        current_meta_features: Dict[str, Any], 
        historical_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find datasets similar to the current one based on meta-features."""
        
        # Convert meta-features to numerical vectors
        current_vector = self._meta_features_to_vector(current_meta_features)
        historical_vectors = []
        
        for data in historical_data:
            vector = self._meta_features_to_vector(data["meta_features"])
            if vector is not None:
                historical_vectors.append((data, vector))
        
        if not historical_vectors:
            return []
        
        # Calculate similarities
        similarities = []
        for data, vector in historical_vectors:
            similarity = self._calculate_similarity(current_vector, vector)
            if similarity >= self.config.similarity_threshold:
                similarities.append((data, similarity))
        
        # Sort by similarity and performance
        similarities.sort(
            key=lambda x: (
                x[1],  # similarity
                x[0]["performance"].get("mean_score", 0),  # performance
            ),
            reverse=True
        )
        
        return [data for data, _ in similarities[:self.config.max_warm_starts]]
    
    def _meta_features_to_vector(self, meta_features: Dict[str, Any]) -> Optional[np.ndarray]:
        """Convert meta-features dictionary to numerical vector."""
        try:
            # Define the order of features for consistent vectorization
            feature_order = [
                "num_instances", "num_features", "num_classes",
                "missing_values_ratio", "categorical_features_ratio", "numerical_features_ratio",
                "mean_mean", "mean_std", "std_mean", "std_std",
                "skew_mean", "skew_std", "kurtosis_mean", "kurtosis_std",
                "target_entropy", "mutual_info_mean", "mutual_info_std",
                "mutual_info_max", "mutual_info_min", "fisher_score",
                "class_imbalance_ratio", "feature_correlation_mean",
                "feature_correlation_std", "feature_correlation_max",
                "zero_variance_features_ratio", "low_variance_features_ratio",
                "feature_sparsity"
            ]
            
            vector = []
            for feature in feature_order:
                value = meta_features.get(feature, 0.0)
                if value is None:
                    value = 0.0
                vector.append(float(value))
            
            return np.array(vector)
            
        except Exception as e:
            logger.error(f"Failed to convert meta-features to vector: {e}")
            return None
    
    def _calculate_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate similarity between two meta-feature vectors."""
        try:
            # Use cosine similarity for normalized features
            if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
                return 0.0
            
            similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def _extract_warm_starts(
        self, 
        similar_datasets: List[Dict[str, Any]], 
        model_name: str, 
        task_type: str
    ) -> Dict[str, Any]:
        """Extract hyperparameter warm starts from similar datasets."""
        
        warm_starts = {
            "configurations": [],
            "recommended_params": {},
            "confidence_scores": {}
        }
        
        # Get default parameter grids for the model
        default_params = self._get_default_hyperparameter_grid(model_name, task_type)
        
        # Extract successful configurations from similar datasets
        successful_configs = []
        for dataset in similar_datasets:
            performance = dataset.get("performance", {})
            if performance.get("mean_score", 0) > 0.7:  # Only consider good performers
                successful_configs.append({
                    "meta_features": dataset["meta_features"],
                    "performance": performance,
                    "similarity": self._calculate_similarity(
                        self._meta_features_to_vector(dataset["meta_features"]),
                        self._meta_features_to_vector(similar_datasets[0]["meta_features"])
                    )
                })
        
        if successful_configs:
            # Generate warm start configurations based on successful similar datasets
            warm_start_configs = self._generate_warm_start_configurations(
                successful_configs, default_params, model_name
            )
            
            warm_starts["configurations"] = warm_start_configs
            warm_starts["recommended_params"] = self._get_recommended_params(warm_start_configs)
            warm_starts["confidence_scores"] = self._calculate_confidence_scores(successful_configs)
        
        return warm_starts
    
    def _get_default_hyperparameter_grid(self, model_name: str, task_type: str) -> Dict[str, List]:
        """Get default hyperparameter grids for different models."""
        
        grids = {
            "random_forest": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [None, 5, 10, 15, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None]
            },
            "gradient_boosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7, 9],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "xgboost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7, 9],
                "min_child_weight": [1, 3, 5],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0]
            },
            "lightgbm": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7, 9],
                "num_leaves": [31, 50, 100],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0]
            },
            "logistic_regression": {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "lbfgs"],
                "max_iter": [100, 500, 1000]
            },
            "mlp": {
                "hidden_layer_sizes": [(50,), (100,), (100, 50), (200, 100)],
                "learning_rate": [0.001, 0.01, 0.1],
                "alpha": [0.0001, 0.001, 0.01],
                "max_iter": [200, 500, 1000]
            },
            "knn": {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan", "minkowski"]
            }
        }
        
        return grids.get(model_name, {})
    
    def _generate_warm_start_configurations(
        self, 
        successful_configs: List[Dict[str, Any]], 
        default_params: Dict[str, List], 
        model_name: str
    ) -> List[Dict[str, Any]]:
        """Generate warm start configurations based on successful similar datasets."""
        
        configurations = []
        
        # Create configurations based on dataset characteristics
        for config in successful_configs[:3]:  # Top 3 similar datasets
            meta_features = config["meta_features"]
            performance = config["performance"]
            
            # Generate configuration based on dataset characteristics
            warm_start_config = self._adapt_hyperparameters_to_dataset(
                meta_features, default_params, model_name
            )
            
            configurations.append({
                "params": warm_start_config,
                "expected_score": performance.get("mean_score", 0.8),
                "confidence": config["similarity"],
                "source_dataset": config["meta_features"]
            })
        
        return configurations
    
    def _adapt_hyperparameters_to_dataset(
        self, 
        meta_features: Dict[str, Any], 
        default_params: Dict[str, List], 
        model_name: str
    ) -> Dict[str, Any]:
        """Adapt hyperparameters based on dataset characteristics."""
        
        adapted_params = {}
        
        # Dataset size adaptations
        num_instances = meta_features.get("num_instances", 1000)
        num_features = meta_features.get("num_features", 10)
        
        if model_name in ["random_forest", "gradient_boosting", "xgboost", "lightgbm"]:
            # Tree-based models
            if num_instances < 1000:
                adapted_params["n_estimators"] = 50
            elif num_instances < 10000:
                adapted_params["n_estimators"] = 100
            else:
                adapted_params["n_estimators"] = 200
            
            # Max depth based on feature count
            if num_features < 10:
                adapted_params["max_depth"] = 5
            elif num_features < 50:
                adapted_params["max_depth"] = 10
            else:
                adapted_params["max_depth"] = 15
        
        elif model_name == "mlp":
            # Neural network adaptations
            if num_instances < 1000:
                adapted_params["hidden_layer_sizes"] = (50,)
            elif num_instances < 10000:
                adapted_params["hidden_layer_sizes"] = (100,)
            else:
                adapted_params["hidden_layer_sizes"] = (200, 100)
            
            adapted_params["max_iter"] = min(1000, num_instances // 10)
        
        elif model_name == "knn":
            # k-NN adaptations
            n_neighbors = min(11, max(3, int(np.sqrt(num_instances))))
            adapted_params["n_neighbors"] = n_neighbors
        
        # Add default values for unspecified parameters
        for param, values in default_params.items():
            if param not in adapted_params:
                adapted_params[param] = values[len(values) // 2]  # Middle value
        
        return adapted_params
    
    def _get_recommended_params(self, configurations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get recommended parameters based on weighted average of configurations."""
        
        if not configurations:
            return {}
        
        # Weight configurations by their confidence scores
        weights = [config["confidence"] for config in configurations]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return configurations[0]["params"]
        
        # For numerical parameters, calculate weighted average
        recommended_params = {}
        all_param_names = set()
        
        for config in configurations:
            all_param_names.update(config["params"].keys())
        
        for param_name in all_param_names:
            weighted_value = 0.0
            total_param_weight = 0.0
            
            for config, weight in zip(configurations, weights):
                if param_name in config["params"]:
                    value = config["params"][param_name]
                    if isinstance(value, (int, float)):
                        weighted_value += value * weight
                        total_param_weight += weight
                    else:
                        # For non-numerical parameters, use the most confident one
                        if weight > recommended_params.get(f"{param_name}_confidence", 0):
                            recommended_params[param_name] = value
                            recommended_params[f"{param_name}_confidence"] = weight
            
            if total_param_weight > 0 and isinstance(weighted_value / total_param_weight, (int, float)):
                recommended_params[param_name] = weighted_value / total_param_weight
        
        # Clean up confidence scores
        return {k: v for k, v in recommended_params.items() if not k.endswith("_confidence")}
    
    def _calculate_confidence_scores(self, successful_configs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate confidence scores for warm start recommendations."""
        
        if not successful_configs:
            return {"overall": 0.0}
        
        # Calculate overall confidence based on similarity and performance
        similarities = [config["similarity"] for config in successful_configs]
        performances = [config["performance"].get("mean_score", 0.8) for config in successful_configs]
        
        avg_similarity = np.mean(similarities)
        avg_performance = np.mean(performances)
        
        # Confidence is combination of similarity and performance
        confidence = (avg_similarity * 0.6 + avg_performance * 0.4)
        
        return {
            "overall": confidence,
            "similarity": avg_similarity,
            "performance": avg_performance,
            "sample_size": len(successful_configs)
        }
    
    def _enhance_with_llm_recommendations(
        self, 
        warm_starts: Dict[str, Any], 
        meta_features: Dict[str, Any], 
        model_name: str, 
        task_type: str,
        llm_client
    ) -> Dict[str, Any]:
        """Enhance warm starts with LLM-generated recommendations."""
        
        try:
            # Create prompt for LLM
            prompt = self._create_llm_prompt(warm_starts, meta_features, model_name, task_type)
            
            # Get LLM recommendations
            llm_response = llm_client.generate_response(prompt)
            
            # Parse LLM response and enhance warm starts
            enhanced_configs = self._parse_llm_recommendations(llm_response, warm_starts)
            
            if enhanced_configs:
                warm_starts["llm_recommendations"] = enhanced_configs
                warm_starts["llm_enhanced"] = True
            
            return warm_starts
            
        except Exception as e:
            logger.error(f"Failed to enhance warm starts with LLM: {e}")
            return warm_starts
    
    def _create_llm_prompt(
        self, 
        warm_starts: Dict[str, Any], 
        meta_features: Dict[str, Any], 
        model_name: str, 
        task_type: str
    ) -> str:
        """Create a prompt for LLM to generate hyperparameter recommendations."""
        
        prompt = f"""
You are an expert machine learning engineer. Based on the following dataset characteristics and similar successful configurations, recommend optimal hyperparameters for {model_name} on a {task_type} task.

Dataset Characteristics:
- Number of instances: {meta_features.get('num_instances', 'unknown')}
- Number of features: {meta_features.get('num_features', 'unknown')}
- Missing values ratio: {meta_features.get('missing_values_ratio', 0):.2f}
- Categorical features ratio: {meta_features.get('categorical_features_ratio', 0):.2f}
- Numerical features ratio: {meta_features.get('numerical_features_ratio', 0):.2f}
- Class imbalance ratio: {meta_features.get('class_imbalance_ratio', 1):.2f}

Similar Successful Configurations:
"""
        
        for i, config in enumerate(warm_starts.get("configurations", [])[:3]):
            prompt += f"\nConfiguration {i+1}:\n"
            for param, value in config["params"].items():
                prompt += f"  {param}: {value}\n"
            prompt += f"  Expected Score: {config['expected_score']:.3f}\n"
            prompt += f"  Confidence: {config['confidence']:.3f}\n"
        
        prompt += f"""
Please recommend optimal hyperparameters for {model_name} based on these insights. Consider:
1. Dataset size and complexity
2. Feature characteristics
3. Successful patterns from similar datasets
4. Best practices for {model_name} on {task_type} tasks

Provide your recommendations in JSON format with the parameter names and values.
"""
        
        return prompt
    
    def _parse_llm_recommendations(self, llm_response: str, warm_starts: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response to extract hyperparameter recommendations."""
        
        try:
            # Try to extract JSON from LLM response
            import re
            
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                recommendations = json.loads(json_match.group())
                return recommendations
            else:
                # If no JSON found, return empty dict
                return {}
                
        except Exception as e:
            logger.error(f"Failed to parse LLM recommendations: {e}")
            return {}
    
    def _get_default_warm_starts(self, model_name: str, task_type: str) -> Dict[str, Any]:
        """Get default warm starts when no similar datasets are found."""
        
        default_params = self._get_default_hyperparameter_grid(model_name, task_type)
        
        return {
            "configurations": [{
                "params": {param: values[len(values) // 2] for param, values in default_params.items()},
                "expected_score": 0.7,
                "confidence": 0.5,
                "source_dataset": "default"
            }],
            "recommended_params": {param: values[len(values) // 2] for param, values in default_params.items()},
            "confidence_scores": {"overall": 0.5, "similarity": 0.0, "performance": 0.7, "sample_size": 0},
            "llm_enhanced": False
        }


class MetaLearningAnalyzer:
    """Analyzer for meta-learning insights and recommendations."""
    
    def __init__(self, meta_learning_engine: MetaLearningEngine):
        """Initialize the meta-learning analyzer."""
        self.engine = meta_learning_engine
    
    def analyze_dataset_similarity(self, meta_features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how similar the current dataset is to historical datasets."""
        
        historical_data = self.engine._load_historical_data()
        
        if not historical_data:
            return {
                "similarity_analysis": "No historical data available",
                "recommendations": "Use default hyperparameters"
            }
        
        similarities = []
        for data in historical_data:
            vector1 = self.engine._meta_features_to_vector(meta_features)
            vector2 = self.engine._meta_features_to_vector(data["meta_features"])
            
            if vector1 is not None and vector2 is not None:
                similarity = self.engine._calculate_similarity(vector1, vector2)
                similarities.append({
                    "model_id": data["model_id"],
                    "similarity": similarity,
                    "performance": data["performance"],
                    "model_type": data["model_type"]
                })
        
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return {
            "total_historical_datasets": len(historical_data),
            "similar_datasets": len([s for s in similarities if s["similarity"] > 0.7]),
            "most_similar": similarities[0] if similarities else None,
            "average_similarity": np.mean([s["similarity"] for s in similarities]),
            "similarity_distribution": {
                "high_similarity": len([s for s in similarities if s["similarity"] > 0.8]),
                "medium_similarity": len([s for s in similarities if 0.5 < s["similarity"] <= 0.8]),
                "low_similarity": len([s for s in similarities if s["similarity"] <= 0.5])
            }
        }
    
    def get_model_performance_trends(self) -> Dict[str, Any]:
        """Analyze trends in model performance across different datasets."""
        
        historical_data = self.engine._load_historical_data()
        
        if not historical_data:
            return {"trends": "No historical data available"}
        
        # Group by model type
        model_performance = {}
        for data in historical_data:
            model_type = data.get("model_type", "unknown")
            performance = data.get("performance", {})
            
            if model_type not in model_performance:
                model_performance[model_type] = []
            
            model_performance[model_type].append(performance.get("mean_score", 0))
        
        # Calculate statistics for each model type
        trends = {}
        for model_type, scores in model_performance.items():
            if scores:
                trends[model_type] = {
                    "average_score": np.mean(scores),
                    "best_score": np.max(scores),
                    "worst_score": np.min(scores),
                    "std_score": np.std(scores),
                    "sample_count": len(scores)
                }
        
        return {
            "trends": trends,
            "best_performing_model": max(trends.keys(), key=lambda k: trends[k]["average_score"]) if trends else None,
            "most_consistent_model": min(trends.keys(), key=lambda k: trends[k]["std_score"]) if trends else None
        }
