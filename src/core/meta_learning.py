"""
Meta-Learning Module

This module implements meta-learning capabilities for warm starts in hyperparameter optimization.
It learns from previous runs to provide better initial hyperparameter configurations.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class RunMetadata:
    """Metadata for a single ML run."""
    
    run_id: str
    dataset_name: str
    dataset_size: int
    n_features: int
    n_categorical: int
    n_numerical: int
    missing_percentage: float
    target_type: str
    class_balance: Optional[Dict[str, float]]
    model_name: str
    best_params: Dict[str, Any]
    best_score: float
    optimization_method: str
    training_time: float
    timestamp: str


class MetaLearningDatabase:
    """Database for storing and retrieving meta-learning data."""
    
    def __init__(self, db_path: str = "./meta/runs.db"):
        """Initialize the meta-learning database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    dataset_name TEXT,
                    dataset_size INTEGER,
                    n_features INTEGER,
                    n_categorical INTEGER,
                    n_numerical INTEGER,
                    missing_percentage REAL,
                    target_type TEXT,
                    class_balance TEXT,
                    model_name TEXT,
                    best_params TEXT,
                    best_score REAL,
                    optimization_method TEXT,
                    training_time REAL,
                    timestamp TEXT
                )
            """)
            conn.commit()
    
    def store_run(self, metadata: RunMetadata):
        """Store a run's metadata in the database.
        
        Args:
            metadata: Run metadata to store
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO runs VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                metadata.run_id,
                metadata.dataset_name,
                metadata.dataset_size,
                metadata.n_features,
                metadata.n_categorical,
                metadata.n_numerical,
                metadata.missing_percentage,
                metadata.target_type,
                json.dumps(metadata.class_balance) if metadata.class_balance else None,
                metadata.model_name,
                json.dumps(metadata.best_params),
                metadata.best_score,
                metadata.optimization_method,
                metadata.training_time,
                metadata.timestamp
            ))
            conn.commit()
    
    def get_similar_runs(
        self, 
        target_metadata: Dict[str, Any], 
        model_name: str,
        top_k: int = 5
    ) -> List[RunMetadata]:
        """Get similar runs based on dataset characteristics.
        
        Args:
            target_metadata: Metadata of the target dataset
            model_name: Name of the model to find similar runs for
            top_k: Number of similar runs to return
            
        Returns:
            List of similar run metadata
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get all runs for the specified model
            cursor = conn.execute("""
                SELECT * FROM runs WHERE model_name = ?
            """, (model_name,))
            
            runs_data = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            runs_df = pd.DataFrame(runs_data, columns=columns)
        
        if runs_df.empty:
            return []
        
        # Convert to RunMetadata objects
        similar_runs = []
        for _, row in runs_df.iterrows():
            metadata = RunMetadata(
                run_id=row['run_id'],
                dataset_name=row['dataset_name'],
                dataset_size=row['dataset_size'],
                n_features=row['n_features'],
                n_categorical=row['n_categorical'],
                n_numerical=row['n_numerical'],
                missing_percentage=row['missing_percentage'],
                target_type=row['target_type'],
                class_balance=json.loads(row['class_balance']) if row['class_balance'] else None,
                model_name=row['model_name'],
                best_params=json.loads(row['best_params']),
                best_score=row['best_score'],
                optimization_method=row['optimization_method'],
                training_time=row['training_time'],
                timestamp=row['timestamp']
            )
            similar_runs.append(metadata)
        
        # Calculate similarity scores
        similarities = self._calculate_similarities(target_metadata, similar_runs)
        
        # Sort by similarity and return top_k
        sorted_runs = sorted(zip(similar_runs, similarities), key=lambda x: x[1], reverse=True)
        return [run for run, _ in sorted_runs[:top_k]]
    
    def _calculate_similarities(
        self, 
        target_metadata: Dict[str, Any], 
        runs: List[RunMetadata]
    ) -> List[float]:
        """Calculate similarity scores between target metadata and runs."""
        similarities = []
        
        # Extract numerical features for similarity calculation
        target_features = np.array([
            target_metadata.get('dataset_size', 0),
            target_metadata.get('n_features', 0),
            target_metadata.get('n_categorical', 0),
            target_metadata.get('n_numerical', 0),
            target_metadata.get('missing_percentage', 0),
        ])
        
        for run in runs:
            run_features = np.array([
                run.dataset_size,
                run.n_features,
                run.n_categorical,
                run.n_numerical,
                run.missing_percentage,
            ])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([target_features], [run_features])[0][0]
            similarities.append(similarity)
        
        return similarities
    
    def get_all_runs(self) -> List[RunMetadata]:
        """Get all runs from the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM runs")
            runs_data = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            runs_df = pd.DataFrame(runs_data, columns=columns)
        
        runs = []
        for _, row in runs_df.iterrows():
            metadata = RunMetadata(
                run_id=row['run_id'],
                dataset_name=row['dataset_name'],
                dataset_size=row['dataset_size'],
                n_features=row['n_features'],
                n_categorical=row['n_categorical'],
                n_numerical=row['n_numerical'],
                missing_percentage=row['missing_percentage'],
                target_type=row['target_type'],
                class_balance=json.loads(row['class_balance']) if row['class_balance'] else None,
                model_name=row['model_name'],
                best_params=json.loads(row['best_params']),
                best_score=row['best_score'],
                optimization_method=row['optimization_method'],
                training_time=row['training_time'],
                timestamp=row['timestamp']
            )
            runs.append(metadata)
        
        return runs


class MetaLearningOptimizer:
    """Meta-learning optimizer that provides warm starts for hyperparameter optimization."""
    
    def __init__(self, db_path: str = "./meta/runs.db"):
        """Initialize the meta-learning optimizer.
        
        Args:
            db_path: Path to the meta-learning database
        """
        self.db = MetaLearningDatabase(db_path)
        self.scaler = StandardScaler()
    
    def get_warm_start_params(
        self,
        model_name: str,
        dataset_metadata: Dict[str, Any],
        param_space: Dict[str, Any],
        n_suggestions: int = 3
    ) -> List[Dict[str, Any]]:
        """Get warm start parameter suggestions.
        
        Args:
            model_name: Name of the model to optimize
            dataset_metadata: Metadata of the current dataset
            param_space: Parameter space for optimization
            n_suggestions: Number of parameter suggestions to return
            
        Returns:
            List of parameter dictionaries for warm starts
        """
        # Get similar runs
        similar_runs = self.db.get_similar_runs(dataset_metadata, model_name, top_k=n_suggestions)
        
        if not similar_runs:
            logger.warning(f"No similar runs found for model {model_name}")
            return self._get_default_params(param_space, n_suggestions)
        
        # Extract and adapt parameters from similar runs
        warm_start_params = []
        for run in similar_runs:
            adapted_params = self._adapt_parameters(
                run.best_params, 
                param_space, 
                dataset_metadata, 
                run
            )
            warm_start_params.append(adapted_params)
        
        # Add some random variations
        if len(warm_start_params) < n_suggestions:
            additional_params = self._get_default_params(
                param_space, 
                n_suggestions - len(warm_start_params)
            )
            warm_start_params.extend(additional_params)
        
        return warm_start_params[:n_suggestions]
    
    def _adapt_parameters(
        self,
        source_params: Dict[str, Any],
        target_param_space: Dict[str, Any],
        target_metadata: Dict[str, Any],
        source_run: RunMetadata
    ) -> Dict[str, Any]:
        """Adapt parameters from a similar run to the current dataset.
        
        Args:
            source_params: Parameters from the similar run
            target_param_space: Parameter space for the current optimization
            target_metadata: Metadata of the current dataset
            source_run: Metadata of the source run
            
        Returns:
            Adapted parameters
        """
        adapted_params = {}
        
        for param_name, param_values in target_param_space.items():
            if param_name in source_params:
                source_value = source_params[param_name]
                
                # Adapt based on dataset size
                size_ratio = target_metadata.get('dataset_size', 1000) / source_run.dataset_size
                
                if param_name in ['n_estimators', 'max_iter']:
                    # Scale iteration-based parameters
                    adapted_value = int(source_value * min(size_ratio, 2.0))
                    adapted_value = max(adapted_value, min(param_values))
                    adapted_value = min(adapted_value, max(param_values))
                    adapted_params[param_name] = adapted_value
                
                elif param_name in ['learning_rate']:
                    # Adjust learning rate based on dataset size
                    if size_ratio > 1.5:
                        adapted_value = source_value * 0.8  # Smaller learning rate for larger datasets
                    else:
                        adapted_value = source_value
                    adapted_value = max(adapted_value, min(param_values))
                    adapted_value = min(adapted_value, max(param_values))
                    adapted_params[param_name] = adapted_value
                
                elif param_name in ['max_depth']:
                    # Adjust depth based on feature count
                    feature_ratio = target_metadata.get('n_features', 10) / source_run.n_features
                    if feature_ratio > 1.2:
                        adapted_value = min(source_value + 1, max(param_values))
                    else:
                        adapted_value = source_value
                    adapted_params[param_name] = adapted_value
                
                else:
                    # Use source value if it's in the target space
                    if source_value in param_values:
                        adapted_params[param_name] = source_value
                    else:
                        # Find closest value
                        adapted_params[param_name] = self._find_closest_value(source_value, param_values)
            else:
                # Use default value if parameter not found
                adapted_params[param_name] = param_values[0] if param_values else None
        
        return adapted_params
    
    def _find_closest_value(self, target_value: Any, param_values: List[Any]) -> Any:
        """Find the closest value in the parameter space."""
        if not param_values:
            return target_value
        
        try:
            # For numerical values
            if isinstance(target_value, (int, float)):
                numerical_values = [v for v in param_values if isinstance(v, (int, float))]
                if numerical_values:
                    return min(numerical_values, key=lambda x: abs(x - target_value))
            
            # For categorical values, return the first one
            return param_values[0]
        except:
            return param_values[0]
    
    def _get_default_params(self, param_space: Dict[str, Any], n_params: int) -> List[Dict[str, Any]]:
        """Get default parameter configurations."""
        default_params = []
        
        for _ in range(n_params):
            params = {}
            for param_name, param_values in param_space.items():
                if isinstance(param_values, list) and param_values:
                    # Choose middle value or random value
                    if len(param_values) > 2:
                        params[param_name] = param_values[len(param_values) // 2]
                    else:
                        params[param_name] = param_values[0]
                else:
                    params[param_name] = None
            default_params.append(params)
        
        return default_params
    
    def store_run_result(
        self,
        run_id: str,
        dataset_name: str,
        dataset_metadata: Dict[str, Any],
        model_name: str,
        best_params: Dict[str, Any],
        best_score: float,
        optimization_method: str,
        training_time: float
    ):
        """Store the result of a completed run.
        
        Args:
            run_id: Unique identifier for the run
            dataset_name: Name of the dataset
            dataset_metadata: Metadata of the dataset
            model_name: Name of the model
            best_params: Best hyperparameters found
            best_score: Best score achieved
            optimization_method: Method used for optimization
            training_time: Time taken for training
        """
        import datetime
        
        metadata = RunMetadata(
            run_id=run_id,
            dataset_name=dataset_name,
            dataset_size=dataset_metadata.get('dataset_size', 0),
            n_features=dataset_metadata.get('n_features', 0),
            n_categorical=dataset_metadata.get('n_categorical', 0),
            n_numerical=dataset_metadata.get('n_numerical', 0),
            missing_percentage=dataset_metadata.get('missing_percentage', 0),
            target_type=dataset_metadata.get('target_type', 'unknown'),
            class_balance=dataset_metadata.get('class_balance'),
            model_name=model_name,
            best_params=best_params,
            best_score=best_score,
            optimization_method=optimization_method,
            training_time=training_time,
            timestamp=datetime.datetime.now().isoformat()
        )
        
        self.db.store_run(metadata)
        logger.info(f"Stored run result for {model_name} on {dataset_name}")
    
    def get_model_performance_stats(self, model_name: str) -> Dict[str, Any]:
        """Get performance statistics for a model across all runs.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with performance statistics
        """
        all_runs = self.db.get_all_runs()
        model_runs = [run for run in all_runs if run.model_name == model_name]
        
        if not model_runs:
            return {}
        
        scores = [run.best_score for run in model_runs]
        training_times = [run.training_time for run in model_runs]
        
        return {
            'n_runs': len(model_runs),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'mean_training_time': np.mean(training_times),
            'std_training_time': np.std(training_times),
        }
    
    def get_dataset_performance_stats(self, dataset_name: str) -> Dict[str, Any]:
        """Get performance statistics for a dataset across all models.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with performance statistics
        """
        all_runs = self.db.get_all_runs()
        dataset_runs = [run for run in all_runs if run.dataset_name == dataset_name]
        
        if not dataset_runs:
            return {}
        
        # Group by model
        model_stats = {}
        for run in dataset_runs:
            if run.model_name not in model_stats:
                model_stats[run.model_name] = []
            model_stats[run.model_name].append(run.best_score)
        
        # Calculate statistics for each model
        stats = {}
        for model_name, scores in model_stats.items():
            stats[model_name] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'n_runs': len(scores),
            }
        
        return stats
