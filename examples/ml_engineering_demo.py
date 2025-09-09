"""
ML Engineering Features Demo

This script demonstrates the advanced ML engineering capabilities including:
- MLflow experiment tracking
- Comprehensive model benchmarking
- Experiment reproducibility
- Model zoo with extensive algorithms
- Automated reporting and visualization
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.benchmarking import BenchmarkConfig, ModelBenchmark, run_benchmark_experiment
from core.ingest import DataIngester
from core.mlflow_tracker import setup_mlflow_experiment
from core.model_zoo import get_model_zoo
from core.reproducibility import ReproducibilityManager, ensure_reproducibility

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_dataset(
    n_samples: int = 1000, n_features: int = 10, task_type: str = "classification"
) -> str:
    """Create a sample dataset for demonstration."""
    np.random.seed(42)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    if task_type == "classification":
        # Create a classification target with some signal
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
        target_name = "target_class"
    else:
        # Create a regression target
        y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.1
        target_name = "target_value"

    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df[target_name] = y

    # Add some missing values and categorical features for realism
    df.loc[df.sample(50).index, "feature_0"] = np.nan
    df["feature_cat"] = np.random.choice(["A", "B", "C"], n_samples)

    # Save dataset
    dataset_path = "sample_dataset.csv"
    df.to_csv(dataset_path, index=False)

    logger.info(f"Created sample dataset: {dataset_path}")
    logger.info(f"Shape: {df.shape}, Target: {target_name}")

    return dataset_path, target_name


def demo_mlflow_tracking():
    """Demonstrate MLflow experiment tracking."""
    logger.info("🔬 Demonstrating MLflow Experiment Tracking")

    # Setup MLflow
    tracker = setup_mlflow_experiment("ml_engineering_demo")

    # Start a run
    tracker.start_run(
        run_name="demo_run",
        description="Demonstration of MLflow tracking capabilities",
        tags={"demo": "true", "framework": "autonomous_ml_agent"},
    )

    # Simulate some experiment data
    dataset_path, target_column = create_sample_dataset()

    # Log dataset information
    data = pd.read_csv(dataset_path)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    tracker.log_data_info(
        data_path=dataset_path,
        target_column=target_column,
        data_shape=data.shape,
        feature_types={col: str(dtype) for col, dtype in X.dtypes.items()},
        missing_values={col: int(X[col].isnull().sum()) for col in X.columns},
        class_distribution=(
            y.value_counts().to_dict() if len(y.unique()) <= 20 else None
        ),
    )

    # Log some mock model results
    mock_results = {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88,
        "f1_score": 0.85,
        "roc_auc": 0.91,
    }

    tracker.log_model_performance(
        model_name="demo_model",
        metrics=mock_results,
        hyperparameters={"C": 1.0, "max_iter": 1000},
        training_time=2.5,
        inference_time=0.01,
    )

    # Log some artifacts
    tracker.log_text(
        text="This is a demonstration of MLflow text logging",
        artifact_file="demo_notes.txt",
    )

    # End the run
    tracker.end_run()

    logger.info("✅ MLflow tracking demonstration completed")

    # Show experiment info
    best_model = tracker.get_best_model("accuracy")
    if best_model:
        logger.info(f"Best model: {best_model}")


def demo_model_benchmarking():
    """Demonstrate comprehensive model benchmarking."""
    logger.info("🏁 Demonstrating Model Benchmarking")

    # Create sample dataset
    dataset_path, target_column = create_sample_dataset(n_samples=500, n_features=5)

    # Setup MLflow for benchmarking
    tracker = setup_mlflow_experiment("benchmarking_demo")

    # Run comprehensive benchmark
    report_path = run_benchmark_experiment(
        dataset_path=dataset_path,
        target_column=target_column,
        output_dir="./benchmark_demo",
        max_models=5,  # Limit for demo
        use_mlflow=True,
    )

    logger.info(f"✅ Benchmark completed! Report: {report_path}")

    # Demonstrate manual benchmarking
    benchmark = ModelBenchmark(
        config=BenchmarkConfig(cv_folds=3, verbose=True), mlflow_tracker=tracker
    )

    # Load data
    data = pd.read_csv(dataset_path)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Get model zoo
    zoo = get_model_zoo()
    task_type = "classification" if len(y.unique()) <= 20 else "regression"
    available_models = zoo.list_models(task_type == "classification")[
        :3
    ]  # Top 3 for demo

    # Run benchmark
    results = benchmark.benchmark_multiple_models(available_models, X, y, task_type)

    # Get best models
    best_models = benchmark.get_best_models(results, metric="accuracy", top_k=3)
    logger.info(f"Top 3 models: {best_models}")

    # Create comparison report
    report_path = benchmark.create_comparison_report(
        results, "./benchmark_demo/manual_benchmark.html"
    )
    logger.info(f"Manual benchmark report: {report_path}")


def demo_reproducibility():
    """Demonstrate experiment reproducibility features."""
    logger.info("🔄 Demonstrating Experiment Reproducibility")

    # Setup reproducibility manager
    reproducibility_manager = ReproducibilityManager()

    # Create sample dataset
    dataset_path, target_column = create_sample_dataset()

    # Create an experiment
    experiment_id = reproducibility_manager.create_experiment(
        name="reproducibility_demo",
        description="Demonstration of experiment reproducibility",
        dataset_path=dataset_path,
        target_column=target_column,
        task_type="classification",
        models_to_test=["logistic_regression", "random_forest", "xgboost"],
        hyperparameters={
            "logistic_regression": {"C": 1.0, "max_iter": 1000},
            "random_forest": {"n_estimators": 100, "max_depth": 10},
            "xgboost": {"n_estimators": 100, "max_depth": 6},
        },
        preprocessing_config={
            "scaling": True,
            "handle_missing": "mean",
            "categorical_encoding": "onehot",
        },
    )

    logger.info(f"Created experiment: {experiment_id}")

    # Save some artifacts
    sample_data = pd.read_csv(dataset_path)
    reproducibility_manager.save_artifact(
        experiment_id, "sample_data", sample_data.head(), "csv"
    )

    # Save experiment configuration
    config = {"random_seed": 42, "cv_folds": 5, "test_size": 0.2}
    reproducibility_manager.save_artifact(
        experiment_id, "experiment_config", config, "json"
    )

    # Update experiment status
    reproducibility_manager.update_experiment_status(
        experiment_id,
        "completed",
        {"best_model": "xgboost", "best_score": 0.85, "total_time": 120.5},
    )

    # Create experiment report
    report_path = reproducibility_manager.create_experiment_report(experiment_id)
    logger.info(f"Experiment report: {report_path}")

    # Demonstrate reproduction
    reproduction_id = reproducibility_manager.reproduce_experiment(experiment_id)
    logger.info(f"Created reproduction experiment: {reproduction_id}")

    # List all experiments
    experiments = reproducibility_manager.list_experiments()
    logger.info(f"Total experiments: {len(experiments)}")

    # Export experiment
    export_path = reproducibility_manager.export_experiment(
        experiment_id, f"./experiment_export_{experiment_id}.zip"
    )
    logger.info(f"Exported experiment to: {export_path}")


def demo_model_zoo():
    """Demonstrate the comprehensive model zoo."""
    logger.info("🤖 Demonstrating Model Zoo")

    # Get model zoo
    zoo = get_model_zoo()

    # List all models
    all_models = zoo.list_models()
    logger.info(f"Total models available: {len(all_models)}")

    # List classification models
    classification_models = zoo.list_models(task_type="classification")
    logger.info(f"Classification models: {len(classification_models)}")

    # List regression models
    regression_models = zoo.list_models(task_type="regression")
    logger.info(f"Regression models: {len(regression_models)}")

    # Get model recommendations
    recommendations = zoo.get_recommended_models(
        is_classification=True, n_samples=1000, n_features=10
    )
    logger.info(f"Recommended models for 1000 samples, 10 features: {recommendations}")

    # Demonstrate model creation
    model = zoo.get_model("random_forest", is_classification=True)
    logger.info(f"Created model: {model}")

    # Demonstrate model pipeline creation
    pipeline = zoo.create_model_pipeline(
        "logistic_regression",
        preprocessing_steps=[("scaler", "StandardScaler")],
        C=1.0,
        max_iter=1000,
    )
    logger.info(f"Created pipeline: {pipeline}")


def demo_data_analysis():
    """Demonstrate data analysis capabilities."""
    logger.info("📊 Demonstrating Data Analysis")

    # Create sample dataset
    dataset_path, target_column = create_sample_dataset()

    # Analyze data
    ingester = DataIngester()
    data_info = ingester.analyze_data(dataset_path)

    logger.info("Data Analysis Results:")
    logger.info(f"  Shape: {data_info['shape']}")
    logger.info(f"  Target: {data_info['target_column']}")
    logger.info(f"  Data types: {data_info['dtypes']}")
    logger.info(f"  Missing values: {data_info.get('missing_values', {})}")

    # Load and show sample
    data = pd.read_csv(dataset_path)
    logger.info(f"Sample data:\n{data.head()}")


def main():
    """Run all demonstrations."""
    logger.info("🚀 Starting ML Engineering Features Demonstration")

    try:
        # Ensure reproducibility
        ensure_reproducibility(seed=42)

        # Run demonstrations
        demo_data_analysis()
        demo_model_zoo()
        demo_mlflow_tracking()
        demo_model_benchmarking()
        demo_reproducibility()

        logger.info("✅ All demonstrations completed successfully!")

        # Cleanup
        cleanup_files = [
            "sample_dataset.csv",
            "benchmark_demo",
            "experiments",
            "artifacts",
            "meta",
        ]

        for file_path in cleanup_files:
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    import shutil

                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)

        logger.info("🧹 Cleanup completed")

    except Exception as e:
        logger.error(f"❌ Error during demonstration: {str(e)}")
        raise


if __name__ == "__main__":
    main()
