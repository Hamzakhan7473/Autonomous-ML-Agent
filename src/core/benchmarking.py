"""
Comprehensive Model Benchmarking System

This module provides automated benchmarking capabilities for comparing
models across datasets with detailed performance tracking and reporting.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

from .mlflow_tracker import MLflowTracker
from .model_zoo import ModelZoo

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking experiments."""

    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    scoring_metrics: list[str] = None
    timeout_seconds: int = 300  # 5 minutes per model
    n_jobs: int = -1
    verbose: bool = True

    def __post_init__(self):
        if self.scoring_metrics is None:
            self.scoring_metrics = [
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
            ]


@dataclass
class ModelResult:
    """Results from benchmarking a single model."""

    model_name: str
    task_type: str
    cv_scores: dict[str, dict[str, float]]
    test_metrics: dict[str, float]
    training_time: float
    prediction_time: float
    total_time: float
    n_samples: int
    n_features: int
    hyperparameters: dict[str, Any]
    feature_importance: np.ndarray | None = None
    error_message: str | None = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class ModelBenchmark:
    """Comprehensive model benchmarking system."""

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        mlflow_tracker: MLflowTracker | None = None,
    ):
        """
        Initialize the benchmark system.

        Args:
            config: Benchmark configuration
            mlflow_tracker: Optional MLflow tracker for experiment logging
        """
        self.config = config or BenchmarkConfig()
        self.mlflow_tracker = mlflow_tracker
        self.model_zoo = ModelZoo()
        self.results = {}
        self.benchmark_history = []

        # Create output directories
        Path("benchmarks").mkdir(exist_ok=True)
        Path("benchmarks/reports").mkdir(exist_ok=True)
        Path("benchmarks/plots").mkdir(exist_ok=True)

    def benchmark_single_model(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        hyperparameters: dict[str, Any] | None = None,
        task_type: str | None = None,
    ) -> ModelResult:
        """
        Benchmark a single model with comprehensive evaluation.

        Args:
            model_name: Name of the model to benchmark
            X: Feature matrix
            y: Target vector
            hyperparameters: Model hyperparameters
            task_type: Task type ('classification' or 'regression')

        Returns:
            ModelResult object with comprehensive metrics
        """
        start_time = time.time()

        try:
            # Determine task type
            if task_type is None:
                task_type = (
                    "classification" if len(np.unique(y)) <= 20 else "regression"
                )

            # Get model configuration
            model_config = self.model_zoo.get_model_config(model_name)

            # Validate task type
            if model_config.is_classification != (task_type == "classification"):
                raise ValueError(
                    f"Model {model_name} is for {'classification' if model_config.is_classification else 'regression'}, "
                    f"but task is {task_type}"
                )

            # Get model instance
            model = self.model_zoo.get_model(model_name, model_config.is_classification)

            # Set hyperparameters if provided
            if hyperparameters:
                model.model.set_params(**hyperparameters)

            # Prepare data
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y if task_type == "classification" else None,
            )

            # Apply scaling if required
            if model_config.requires_scaling:
                scaler = StandardScaler()
                X_train_scaled = pd.DataFrame(
                    scaler.fit_transform(X_train),
                    columns=X_train.columns,
                    index=X_train.index,
                )
                X_test_scaled = pd.DataFrame(
                    scaler.transform(X_test), columns=X_test.columns, index=X_test.index
                )
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test

            # Cross-validation
            cv_scores = self._run_cross_validation(
                model, X_train_scaled, y_train, task_type
            )

            # Training
            train_start = time.time()
            model.fit(X_train_scaled, y_train)
            training_time = time.time() - train_start

            # Prediction
            pred_start = time.time()
            y_pred = model.predict(X_test_scaled)
            prediction_time = time.time() - pred_start

            # Test metrics
            test_metrics = self._calculate_test_metrics(
                y_test, y_pred, task_type, model, X_test_scaled
            )

            # Feature importance
            feature_importance = model.get_feature_importance()

            total_time = time.time() - start_time

            result = ModelResult(
                model_name=model_name,
                task_type=task_type,
                cv_scores=cv_scores,
                test_metrics=test_metrics,
                training_time=training_time,
                prediction_time=prediction_time,
                total_time=total_time,
                n_samples=X.shape[0],
                n_features=X.shape[1],
                hyperparameters=hyperparameters or {},
                feature_importance=feature_importance,
            )

            # Log to MLflow if available
            if self.mlflow_tracker:
                self._log_to_mlflow(result, X, y)

            logger.info(f"Benchmarked {model_name}: {test_metrics}")
            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error benchmarking {model_name}: {error_msg}")

            return ModelResult(
                model_name=model_name,
                task_type=task_type or "unknown",
                cv_scores={},
                test_metrics={},
                training_time=0,
                prediction_time=0,
                total_time=time.time() - start_time,
                n_samples=X.shape[0],
                n_features=X.shape[1],
                hyperparameters=hyperparameters or {},
                error_message=error_msg,
            )

    def _run_cross_validation(
        self, model: Any, X: pd.DataFrame, y: pd.Series, task_type: str
    ) -> dict[str, dict[str, float]]:
        """Run cross-validation and return scores."""
        # Setup cross-validation
        if task_type == "classification":
            cv = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )
            scoring_metrics = [
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
            ]
        else:
            cv = KFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )
            scoring_metrics = [
                "neg_mean_squared_error",
                "neg_mean_absolute_error",
                "r2",
            ]

        cv_scores = {}
        for metric in scoring_metrics:
            try:
                scores = cross_val_score(
                    model.model, X, y, cv=cv, scoring=metric, n_jobs=self.config.n_jobs
                )
                cv_scores[metric] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "scores": scores.tolist(),
                }
            except Exception as e:
                logger.warning(f"Failed to calculate {metric}: {str(e)}")
                cv_scores[metric] = {"mean": np.nan, "std": np.nan, "scores": []}

        return cv_scores

    def _calculate_test_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        task_type: str,
        model: Any,
        X_test: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate comprehensive test metrics."""
        metrics = {}

        if task_type == "classification":
            # Classification metrics
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
            metrics["precision"] = precision_score(
                y_true, y_pred, average="macro", zero_division=0
            )
            metrics["recall"] = recall_score(
                y_true, y_pred, average="macro", zero_division=0
            )
            metrics["f1_score"] = f1_score(
                y_true, y_pred, average="macro", zero_division=0
            )

            # ROC AUC for binary classification
            if len(np.unique(y_true)) == 2 and hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test)
                    if y_proba.shape[1] == 2:
                        metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
                    else:
                        metrics["roc_auc"] = roc_auc_score(
                            y_true, y_proba, multi_class="ovr"
                        )
                except Exception:
                    pass

            # Log loss
            try:
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)
                    metrics["log_loss"] = log_loss(y_true, y_proba)
            except Exception:
                pass

        else:
            # Regression metrics
            metrics["mse"] = mean_squared_error(y_true, y_pred)
            metrics["mae"] = mean_absolute_error(y_true, y_pred)
            metrics["r2_score"] = r2_score(y_true, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])

        return metrics

    def _log_to_mlflow(self, result: ModelResult, X: pd.DataFrame, y: pd.Series):
        """Log benchmark result to MLflow."""
        try:
            # Log model performance
            self.mlflow_tracker.log_model_performance(
                model_name=result.model_name,
                metrics=result.test_metrics,
                hyperparameters=result.hyperparameters,
                training_time=result.training_time,
                inference_time=result.prediction_time,
            )

            # Log data info
            self.mlflow_tracker.log_data_info(
                data_path="benchmark_data",
                target_column=y.name or "target",
                data_shape=X.shape,
                feature_types={col: str(dtype) for col, dtype in X.dtypes.items()},
                missing_values={col: int(X[col].isnull().sum()) for col in X.columns},
                class_distribution=(
                    y.value_counts().to_dict()
                    if result.task_type == "classification"
                    else None
                ),
            )

        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {str(e)}")

    def benchmark_multiple_models(
        self,
        model_names: list[str],
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str | None = None,
        hyperparameters: dict[str, dict[str, Any]] | None = None,
    ) -> list[ModelResult]:
        """
        Benchmark multiple models and return results.

        Args:
            model_names: List of model names to benchmark
            X: Feature matrix
            y: Target vector
            task_type: Task type
            hyperparameters: Dictionary mapping model names to their hyperparameters

        Returns:
            List of ModelResult objects
        """
        results = []
        hyperparameters = hyperparameters or {}

        for i, model_name in enumerate(model_names):
            logger.info(f"Benchmarking {model_name} ({i+1}/{len(model_names)})")

            model_params = hyperparameters.get(model_name, {})
            result = self.benchmark_single_model(
                model_name, X, y, model_params, task_type
            )

            results.append(result)
            self.results[model_name] = result

        # Store in history
        self.benchmark_history.append(
            {
                "timestamp": time.time(),
                "dataset_shape": X.shape,
                "task_type": task_type or "auto",
                "model_count": len(model_names),
                "results": [asdict(r) for r in results],
            }
        )

        return results

    def benchmark_all_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str | None = None,
        max_models: int | None = None,
    ) -> list[ModelResult]:
        """
        Benchmark all available models.

        Args:
            X: Feature matrix
            y: Target vector
            task_type: Task type
            max_models: Maximum number of models to benchmark

        Returns:
            List of ModelResult objects
        """
        # Determine task type
        if task_type is None:
            task_type = "classification" if len(np.unique(y)) <= 20 else "regression"

        # Get available models
        available_models = self.model_zoo.list_models(task_type == "classification")

        if max_models:
            available_models = available_models[:max_models]

        logger.info(f"Benchmarking {len(available_models)} models for {task_type}")

        return self.benchmark_multiple_models(available_models, X, y, task_type)

    def get_best_models(
        self, results: list[ModelResult], metric: str = "accuracy", top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Get top-k best performing models.

        Args:
            results: List of benchmark results
            metric: Metric to optimize
            top_k: Number of top models to return

        Returns:
            List of (model_name, score) tuples
        """
        # Filter successful results
        successful_results = [r for r in results if r.error_message is None]

        if not successful_results:
            return []

        # Sort by metric
        scored_results = []
        for result in successful_results:
            if metric in result.test_metrics:
                score = result.test_metrics[metric]
                scored_results.append((result.model_name, score))

        # Sort and return top-k
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:top_k]

    def create_comparison_report(
        self, results: list[ModelResult], output_path: str
    ) -> str:
        """
        Create a comprehensive comparison report.

        Args:
            results: List of benchmark results
            output_path: Path to save the report

        Returns:
            Path to saved report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create comparison DataFrame
        comparison_data = []
        for result in results:
            if result.error_message is None:
                row = {
                    "Model": result.model_name,
                    "Task Type": result.task_type,
                    "Training Time (s)": result.training_time,
                    "Prediction Time (s)": result.prediction_time,
                    "Total Time (s)": result.total_time,
                    "N Samples": result.n_samples,
                    "N Features": result.n_features,
                }

                # Add test metrics
                row.update(result.test_metrics)

                # Add CV scores (mean only)
                for metric, scores in result.cv_scores.items():
                    row[f"CV {metric} (mean)"] = scores["mean"]

                comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # Create HTML report
        html_content = self._generate_html_report(comparison_df, results)

        # Save report
        with open(output_path, "w") as f:
            f.write(html_content)

        # Save CSV
        csv_path = output_path.with_suffix(".csv")
        comparison_df.to_csv(csv_path, index=False)

        logger.info(f"Created comparison report at {output_path}")
        return str(output_path)

    def _generate_html_report(
        self, comparison_df: pd.DataFrame, results: list[ModelResult]
    ) -> str:
        """Generate HTML report content."""
        # Create performance plots
        plot_paths = self._create_performance_plots(results)

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .best {{ background-color: #d4edda; }}
                .error {{ background-color: #f8d7da; }}
                .plot {{ margin: 20px 0; }}
                .summary {{ background-color: #e9ecef; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Model Benchmark Report</h1>
            <p>Generated on: {pd.Timestamp.now()}</p>

            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total models benchmarked:</strong> {len(results)}</p>
                <p><strong>Successful runs:</strong> {len([r for r in results if r.error_message is None])}</p>
                <p><strong>Failed runs:</strong> {len([r for r in results if r.error_message is not None])}</p>
                <p><strong>Dataset shape:</strong> {results[0].n_samples if results else 0} samples × {results[0].n_features if results else 0} features</p>
            </div>

            <h2>Model Comparison</h2>
            {comparison_df.to_html(index=False, escape=False, classes='table')}

            <h2>Performance Visualizations</h2>
            {self._embed_plots_html(plot_paths)}

            <h2>Detailed Results</h2>
            {self._generate_detailed_results_html(results)}
        </body>
        </html>
        """

        return html_content

    def _create_performance_plots(self, results: list[ModelResult]) -> list[str]:
        """Create performance visualization plots."""
        plot_paths = []

        if not results:
            return plot_paths

        # Filter successful results
        successful_results = [r for r in results if r.error_message is None]

        if not successful_results:
            return plot_paths

        # Set style
        plt.style.use("seaborn-v0_8")

        # 1. Performance comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Model Performance Comparison", fontsize=16)

        # Extract data
        model_names = [r.model_name for r in successful_results]

        # Accuracy/R2 Score
        if successful_results[0].task_type == "classification":
            scores = [r.test_metrics.get("accuracy", 0) for r in successful_results]
            metric_name = "Accuracy"
        else:
            scores = [r.test_metrics.get("r2_score", 0) for r in successful_results]
            metric_name = "R² Score"

        axes[0, 0].bar(model_names, scores)
        axes[0, 0].set_title(f"{metric_name} Comparison")
        axes[0, 0].set_ylabel(metric_name)
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Training time
        training_times = [r.training_time for r in successful_results]
        axes[0, 1].bar(model_names, training_times)
        axes[0, 1].set_title("Training Time Comparison")
        axes[0, 1].set_ylabel("Time (seconds)")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # F1 Score (classification) or RMSE (regression)
        if successful_results[0].task_type == "classification":
            f1_scores = [r.test_metrics.get("f1_score", 0) for r in successful_results]
            axes[1, 0].bar(model_names, f1_scores)
            axes[1, 0].set_title("F1 Score Comparison")
            axes[1, 0].set_ylabel("F1 Score")
        else:
            rmse_scores = [r.test_metrics.get("rmse", 0) for r in successful_results]
            axes[1, 0].bar(model_names, rmse_scores)
            axes[1, 0].set_title("RMSE Comparison")
            axes[1, 0].set_ylabel("RMSE")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Prediction time
        prediction_times = [r.prediction_time for r in successful_results]
        axes[1, 1].bar(model_names, prediction_times)
        axes[1, 1].set_title("Prediction Time Comparison")
        axes[1, 1].set_ylabel("Time (seconds)")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()

        plot_path = Path("benchmarks/plots/performance_comparison.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plot_paths.append(str(plot_path))
        plt.close()

        # 2. Feature importance plot (if available)
        importance_results = [
            r for r in successful_results if r.feature_importance is not None
        ]
        if importance_results:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Get feature names (assuming all models have same features)
            feature_names = [
                f"Feature_{i}"
                for i in range(len(importance_results[0].feature_importance))
            ]

            # Plot feature importance for top models
            top_models = sorted(
                importance_results,
                key=lambda x: x.test_metrics.get(
                    "accuracy" if x.task_type == "classification" else "r2_score", 0
                ),
                reverse=True,
            )[:3]

            for _i, result in enumerate(top_models):
                ax.plot(
                    feature_names,
                    result.feature_importance,
                    marker="o",
                    label=f"{result.model_name}",
                    alpha=0.7,
                )

            ax.set_title("Feature Importance Comparison (Top 3 Models)")
            ax.set_xlabel("Features")
            ax.set_ylabel("Importance")
            ax.legend()
            ax.tick_params(axis="x", rotation=45)

            plt.tight_layout()

            plot_path = Path("benchmarks/plots/feature_importance.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plot_paths.append(str(plot_path))
            plt.close()

        return plot_paths

    def _embed_plots_html(self, plot_paths: list[str]) -> str:
        """Embed plots in HTML report."""
        if not plot_paths:
            return "<p>No plots available.</p>"

        html_plots = ""
        for plot_path in plot_paths:
            plot_name = Path(plot_path).name
            html_plots += f"""
            <div class="plot">
                <h3>{plot_name.replace('_', ' ').title()}</h3>
                <img src="{plot_path}" alt="{plot_name}" style="max-width: 100%; height: auto;">
            </div>
            """

        return html_plots

    def _generate_detailed_results_html(self, results: list[ModelResult]) -> str:
        """Generate detailed results HTML."""
        html = "<div class='detailed-results'>"

        for result in results:
            html += f"""
            <div class="model-result" style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px;">
                <h3>{result.model_name}</h3>
                <p><strong>Task Type:</strong> {result.task_type}</p>
                <p><strong>Status:</strong> {'✅ Success' if result.error_message is None else '❌ Failed'}</p>

                {f'<p><strong>Error:</strong> {result.error_message}</p>' if result.error_message else ''}

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div>
                        <h4>Test Metrics</h4>
                        <ul>
                            {''.join([f'<li><strong>{k}:</strong> {v:.4f}</li>' for k, v in result.test_metrics.items()])}
                        </ul>
                    </div>
                    <div>
                        <h4>Timing</h4>
                        <ul>
                            <li><strong>Training Time:</strong> {result.training_time:.4f}s</li>
                            <li><strong>Prediction Time:</strong> {result.prediction_time:.4f}s</li>
                            <li><strong>Total Time:</strong> {result.total_time:.4f}s</li>
                        </ul>
                    </div>
                </div>
            </div>
            """

        html += "</div>"
        return html

    def export_results(self, results: list[ModelResult], output_path: str) -> str:
        """
        Export benchmark results to JSON file.

        Args:
            results: List of benchmark results
            output_path: Path to save results

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert results to serializable format
        serializable_results = []
        for result in results:
            result_dict = asdict(result)
            # Convert numpy arrays to lists
            if result_dict["feature_importance"] is not None:
                result_dict["feature_importance"] = result_dict[
                    "feature_importance"
                ].tolist()
            serializable_results.append(result_dict)

        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)

        logger.info(f"Exported benchmark results to {output_path}")
        return str(output_path)

    def load_results(self, file_path: str) -> list[ModelResult]:
        """
        Load benchmark results from JSON file.

        Args:
            file_path: Path to the results file

        Returns:
            List of ModelResult objects
        """
        with open(file_path) as f:
            data = json.load(f)

        results = []
        for item in data:
            # Convert feature importance back to numpy array
            if item["feature_importance"] is not None:
                item["feature_importance"] = np.array(item["feature_importance"])

            results.append(ModelResult(**item))

        logger.info(f"Loaded {len(results)} benchmark results from {file_path}")
        return results


def run_benchmark_experiment(
    dataset_path: str,
    target_column: str,
    output_dir: str = "benchmarks",
    max_models: int | None = None,
    use_mlflow: bool = True,
) -> str:
    """
    Run a complete benchmark experiment on a dataset.

    Args:
        dataset_path: Path to the dataset
        target_column: Name of the target column
        output_dir: Output directory for results
        max_models: Maximum number of models to benchmark
        use_mlflow: Whether to use MLflow tracking

    Returns:
        Path to the generated report
    """
    # Load data
    data = pd.read_csv(dataset_path)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Setup MLflow if requested
    mlflow_tracker = None
    if use_mlflow:
        mlflow_tracker = MLflowTracker(
            experiment_name=f"benchmark_{Path(dataset_path).stem}"
        )
        mlflow_tracker.start_run(run_name=f"benchmark_{Path(dataset_path).stem}")

    # Run benchmark
    benchmark = ModelBenchmark(mlflow_tracker=mlflow_tracker)
    results = benchmark.benchmark_all_models(X, y, max_models=max_models)

    # Create report
    report_path = Path(output_dir) / f"benchmark_report_{Path(dataset_path).stem}.html"
    benchmark.create_comparison_report(results, str(report_path))

    # Export results
    results_path = (
        Path(output_dir) / f"benchmark_results_{Path(dataset_path).stem}.json"
    )
    benchmark.export_results(results, str(results_path))

    # End MLflow run
    if mlflow_tracker:
        mlflow_tracker.end_run()

    logger.info(f"Benchmark experiment completed. Report saved to: {report_path}")
    return str(report_path)
