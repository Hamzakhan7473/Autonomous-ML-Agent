"""
MLflow Experiment Tracking and Model Registry Integration

This module provides comprehensive MLflow integration for:
- Experiment tracking with automatic logging
- Model registry management
- Reproducible experiment runs
- Artifact management and versioning
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
import mlflow.pytorch
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from mlflow.entities import RunStatus
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Comprehensive MLflow tracking and model registry management."""
    
    def __init__(self, 
                 experiment_name: str = "autonomous_ml_agent",
                 tracking_uri: Optional[str] = None,
                 registry_uri: Optional[str] = None):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (default: local file store)
            registry_uri: MLflow model registry URI
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or f"file://{Path.cwd() / 'meta' / 'mlflow'}"
        self.registry_uri = registry_uri
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create experiment if it doesn't exist
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment: {experiment_name}")
        except mlflow.exceptions.MlflowException:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
            logger.info(f"Using existing experiment: {experiment_name}")
        
        # Set active experiment
        mlflow.set_experiment(experiment_name)
        
        # Initialize client
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        
        # Create meta directory if it doesn't exist
        Path("meta").mkdir(exist_ok=True)
        Path("meta/mlflow").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
    
    def start_run(self, 
                  run_name: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None,
                  description: Optional[str] = None) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Dictionary of tags to add to the run
            description: Description of the run
            
        Returns:
            Run ID
        """
        tags = tags or {}
        tags.update({
            "framework": "autonomous_ml_agent",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        })
        
        if description:
            tags["description"] = description
        
        run = mlflow.start_run(run_name=run_name, tags=tags)
        self.current_run_id = run.info.run_id
        logger.info(f"Started MLflow run: {self.current_run_id}")
        return self.current_run_id
    
    def end_run(self, status: str = RunStatus.FINISHED):
        """End the current MLflow run."""
        mlflow.end_run(status=status)
        logger.info(f"Ended MLflow run: {self.current_run_id}")
    
    def log_data_info(self, 
                     data_path: str,
                     target_column: str,
                     data_shape: tuple,
                     feature_types: Dict[str, str],
                     missing_values: Dict[str, int],
                     class_distribution: Optional[Dict[str, int]] = None):
        """
        Log dataset information to MLflow.
        
        Args:
            data_path: Path to the dataset
            target_column: Name of the target column
            data_shape: Shape of the dataset (rows, columns)
            feature_types: Dictionary mapping feature names to types
            missing_values: Dictionary mapping feature names to missing value counts
            class_distribution: Distribution of target classes (for classification)
        """
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("target_column", target_column)
        mlflow.log_param("n_rows", data_shape[0])
        mlflow.log_param("n_features", data_shape[1])
        mlflow.log_param("feature_types", json.dumps(feature_types))
        mlflow.log_param("missing_values", json.dumps(missing_values))
        
        if class_distribution:
            mlflow.log_param("class_distribution", json.dumps(class_distribution))
            mlflow.log_metric("n_classes", len(class_distribution))
    
    def log_preprocessing_config(self, config: Dict[str, Any]):
        """Log preprocessing configuration."""
        mlflow.log_params(config)
    
    def log_model_performance(self, 
                            model_name: str,
                            metrics: Dict[str, float],
                            hyperparameters: Dict[str, Any],
                            training_time: float,
                            inference_time: float):
        """
        Log model performance metrics and parameters.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of performance metrics
            hyperparameters: Model hyperparameters
            training_time: Training time in seconds
            inference_time: Inference time in seconds
        """
        # Log model name and timing
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("training_time", training_time)
        mlflow.log_metric("inference_time", inference_time)
        
        # Log hyperparameters
        mlflow.log_params(hyperparameters)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        logger.info(f"Logged performance for {model_name}: {metrics}")
    
    def log_model(self, 
                 model: Any,
                 model_name: str,
                 model_type: str = "sklearn",
                 signature: Optional[Any] = None,
                 input_example: Optional[Any] = None,
                 conda_env: Optional[Dict] = None):
        """
        Log model to MLflow model registry.
        
        Args:
            model: The trained model
            model_name: Name for the model
            model_type: Type of model (sklearn, xgboost, lightgbm, etc.)
            signature: MLflow model signature
            input_example: Example input for the model
            conda_env: Conda environment specification
        """
        try:
            if model_type == "sklearn":
                mlflow.sklearn.log_model(
                    model, 
                    model_name,
                    signature=signature,
                    input_example=input_example,
                    conda_env=conda_env
                )
            elif model_type == "xgboost":
                mlflow.xgboost.log_model(
                    model,
                    model_name,
                    signature=signature,
                    input_example=input_example,
                    conda_env=conda_env
                )
            elif model_type == "lightgbm":
                mlflow.lightgbm.log_model(
                    model,
                    model_name,
                    signature=signature,
                    input_example=input_example,
                    conda_env=conda_env
                )
            elif model_type == "catboost":
                mlflow.catboost.log_model(
                    model,
                    model_name,
                    signature=signature,
                    input_example=input_example,
                    conda_env=conda_env
                )
            elif model_type == "pytorch":
                mlflow.pytorch.log_model(
                    model,
                    model_name,
                    signature=signature,
                    input_example=input_example,
                    conda_env=conda_env
                )
            elif model_type == "tensorflow":
                mlflow.tensorflow.log_model(
                    model,
                    model_name,
                    signature=signature,
                    input_example=input_example,
                    conda_env=conda_env
                )
            else:
                # Generic model logging
                mlflow.log_model(
                    model,
                    model_name,
                    signature=signature,
                    input_example=input_example,
                    conda_env=conda_env
                )
            
            logger.info(f"Logged {model_type} model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to log model {model_name}: {str(e)}")
            raise
    
    def log_artifacts(self, 
                     artifacts_dir: str,
                     artifact_path: Optional[str] = None):
        """
        Log artifacts (plots, data, etc.) to MLflow.
        
        Args:
            artifacts_dir: Directory containing artifacts
            artifact_path: Path within the run to store artifacts
        """
        mlflow.log_artifacts(artifacts_dir, artifact_path)
        logger.info(f"Logged artifacts from {artifacts_dir}")
    
    def log_figure(self, 
                  figure,
                  artifact_file: str,
                  artifact_path: Optional[str] = None):
        """
        Log a matplotlib figure to MLflow.
        
        Args:
            figure: Matplotlib figure object
            artifact_file: Name of the artifact file
            artifact_path: Path within the run to store the artifact
        """
        mlflow.log_figure(figure, artifact_file, artifact_path)
        logger.info(f"Logged figure: {artifact_file}")
    
    def log_text(self, 
                text: str,
                artifact_file: str,
                artifact_path: Optional[str] = None):
        """
        Log text content to MLflow.
        
        Args:
            text: Text content to log
            artifact_file: Name of the artifact file
            artifact_path: Path within the run to store the artifact
        """
        mlflow.log_text(text, artifact_file, artifact_path)
        logger.info(f"Logged text: {artifact_file}")
    
    def register_model(self, 
                     model_name: str,
                     model_version: Optional[str] = None,
                     description: Optional[str] = None,
                     tags: Optional[Dict[str, str]] = None) -> str:
        """
        Register model in MLflow model registry.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            description: Description of the model
            tags: Tags for the model
            
        Returns:
            Model version
        """
        try:
            model_uri = f"runs:/{self.current_run_id}/{model_name}"
            model_version = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                description=description,
                tags=tags or {}
            )
            
            logger.info(f"Registered model {model_name} version {model_version.version}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {str(e)}")
            raise
    
    def transition_model_stage(self, 
                             model_name: str,
                             version: str,
                             stage: str,
                             description: Optional[str] = None):
        """
        Transition model to a specific stage (Staging, Production, Archived).
        
        Args:
            model_name: Name of the model
            version: Model version
            stage: Target stage
            description: Description for the transition
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                description=description
            )
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
            
        except Exception as e:
            logger.error(f"Failed to transition model stage: {str(e)}")
            raise
    
    def get_best_model(self, 
                      metric: str = "accuracy",
                      ascending: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get the best model based on a metric.
        
        Args:
            metric: Metric to optimize
            ascending: Whether to sort ascending (for loss metrics)
            
        Returns:
            Dictionary with run info and model details
        """
        try:
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"]
            )
            
            if runs:
                best_run = runs[0]
                return {
                    "run_id": best_run.info.run_id,
                    "metric_value": best_run.data.metrics.get(metric),
                    "model_name": best_run.data.params.get("model_name"),
                    "tags": best_run.data.tags
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get best model: {str(e)}")
            return None
    
    def compare_runs(self, 
                    run_ids: List[str],
                    metrics: List[str]) -> pd.DataFrame:
        """
        Compare multiple runs in a DataFrame.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to include
            
        Returns:
            DataFrame with run comparison
        """
        try:
            runs_data = []
            for run_id in run_ids:
                run = self.client.get_run(run_id)
                run_data = {
                    "run_id": run_id,
                    "start_time": run.info.start_time,
                    "status": run.info.status
                }
                
                # Add metrics
                for metric in metrics:
                    run_data[metric] = run.data.metrics.get(metric, None)
                
                # Add key parameters
                run_data["model_name"] = run.data.params.get("model_name", "unknown")
                
                runs_data.append(run_data)
            
            return pd.DataFrame(runs_data)
            
        except Exception as e:
            logger.error(f"Failed to compare runs: {str(e)}")
            return pd.DataFrame()
    
    def export_experiment_results(self, 
                                output_path: str,
                                include_artifacts: bool = True) -> str:
        """
        Export experiment results to a directory.
        
        Args:
            output_path: Path to export results
            include_artifacts: Whether to include artifacts
            
        Returns:
            Path to exported results
        """
        try:
            export_path = Path(output_path)
            export_path.mkdir(parents=True, exist_ok=True)
            
            # Get all runs
            runs = self.client.search_runs(experiment_ids=[self.experiment_id])
            
            # Export run data
            runs_data = []
            for run in runs:
                run_info = {
                    "run_id": run.info.run_id,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "status": run.info.status,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags
                }
                runs_data.append(run_info)
            
            # Save runs data
            with open(export_path / "runs_data.json", "w") as f:
                json.dump(runs_data, f, indent=2, default=str)
            
            # Export artifacts if requested
            if include_artifacts:
                artifacts_path = export_path / "artifacts"
                artifacts_path.mkdir(exist_ok=True)
                
                for run in runs:
                    run_artifacts_path = artifacts_path / run.info.run_id
                    run_artifacts_path.mkdir(exist_ok=True)
                    
                    # Download artifacts (simplified - would need actual download logic)
                    pass
            
            logger.info(f"Exported experiment results to {export_path}")
            return str(export_path)
            
        except Exception as e:
            logger.error(f"Failed to export experiment results: {str(e)}")
            raise


def create_mlflow_signature(X_sample: pd.DataFrame, 
                          y_sample: pd.Series) -> Any:
    """
    Create MLflow model signature from sample data.
    
    Args:
        X_sample: Sample feature data
        y_sample: Sample target data
        
    Returns:
        MLflow model signature
    """
    from mlflow.models.signature import infer_signature
    
    return infer_signature(X_sample, y_sample)


def setup_mlflow_experiment(experiment_name: str = "autonomous_ml_agent") -> MLflowTracker:
    """
    Setup MLflow experiment with proper configuration.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Configured MLflowTracker instance
    """
    return MLflowTracker(experiment_name=experiment_name)
