#!/usr/bin/env python3
"""
Main entry point for the Autonomous Machine Learning Agent

This script provides a command-line interface for running the autonomous ML pipeline.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.core.orchestrator import AutonomousMLAgent, PipelineConfig
from src.utils.llm_client import LLMClient, LLMConfig, MockLLMClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("autonomous_ml.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


def load_config(config_path: str | None) -> dict:
    """Load configuration from YAML file"""
    if not config_path:
        return {}

    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Failed to load config file {config_path}: {e}")
        return {}


def validate_dataset(dataset_path: str, target_column: str) -> bool:
    """Validate dataset and target column"""
    try:
        df = pd.read_csv(dataset_path)

        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found in dataset")
            return False

        if df.empty:
            logger.error("Dataset is empty")
            return False

        logger.info(f"Dataset validated: {df.shape[0]} rows, {df.shape[1]} columns")
        return True

    except Exception as e:
        logger.error(f"Failed to validate dataset: {e}")
        return False


def print_results(results):
    """Print pipeline results in a formatted way"""
    print("\n" + "=" * 60)
    print("🤖 AUTONOMOUS ML PIPELINE RESULTS")
    print("=" * 60)

    print("\n📊 LEADERBOARD:")
    print("-" * 40)
    leaderboard = results.leaderboard
    for i, (_, row) in enumerate(leaderboard.head().iterrows(), 1):
        print(
            f"{i}. {row['model_name']:<20} | "
            f"Accuracy: {row.get('accuracy', 'N/A'):<6.3f} | "
            f"Precision: {row.get('precision', 'N/A'):<6.3f} | "
            f"Recall: {row.get('recall', 'N/A'):<6.3f} | "
            f"F1: {row.get('f1', 'N/A'):<6.3f}"
        )

    print(f"\n🏆 BEST MODEL: {results.best_model.__class__.__name__}")
    print(f"⏱️  TRAINING TIME: {results.training_time:.2f} seconds")
    print(f"🔄 TOTAL ITERATIONS: {results.total_iterations}")

    print("\n🔍 FEATURE IMPORTANCE (Top 10):")
    print("-" * 40)
    sorted_features = sorted(
        results.feature_importance.items(), key=lambda x: x[1], reverse=True
    )[:10]
    for feature, importance in sorted_features:
        print(f"{feature:<25} | {importance:.4f}")

    print("\n💡 INSIGHTS:")
    print("-" * 40)
    print(results.model_insights)

    print("\n" + "=" * 60)


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Autonomous Machine Learning Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py --dataset data/iris.csv --target species

  # With custom configuration
  python main.py --dataset data/credit.csv --target default --config config.yaml

  # With time budget
  python main.py --dataset data/housing.csv --target price --time-budget 1800

  # Using mock LLM for testing
  python main.py --dataset data/test.csv --target target --mock-llm
        """,
    )

    # Required arguments
    parser.add_argument(
        "--dataset", required=True, help="Path to the dataset file (CSV format)"
    )
    parser.add_argument("--target", required=True, help="Name of the target column")

    # Optional arguments
    parser.add_argument("--config", help="Path to configuration YAML file")
    parser.add_argument(
        "--optimization-metric",
        default="accuracy",
        choices=["accuracy", "precision", "recall", "f1", "auc"],
        help="Optimization metric (default: accuracy)",
    )
    parser.add_argument(
        "--time-budget",
        type=int,
        default=3600,
        help="Time budget in seconds (default: 3600)",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=10,
        help="Maximum number of models to train (default: 10)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-ensemble", action="store_true", help="Disable ensemble methods"
    )
    parser.add_argument(
        "--no-interpretability",
        action="store_true",
        help="Disable model interpretability",
    )
    parser.add_argument(
        "--no-meta-learning",
        action="store_true",
        help="Disable meta-learning warm starts",
    )
    parser.add_argument(
        "--mock-llm", action="store_true", help="Use mock LLM client for testing"
    )
    parser.add_argument(
        "--llm-provider",
        default="openai",
        choices=["openai", "anthropic"],
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--llm-model", default="gpt-4", help="LLM model name (default: gpt-4)"
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory for results (default: ./output)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate dataset
    if not validate_dataset(args.dataset, args.target):
        sys.exit(1)

    # Load configuration
    config = load_config(args.config)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize LLM client
        if args.mock_llm:
            MockLLMClient()
            logger.info("Using mock LLM client for testing")
        else:
            llm_config = LLMConfig(provider=args.llm_provider, model=args.llm_model)
            LLMClient(llm_config)
            logger.info(
                f"Using {args.llm_provider} LLM client with model {args.llm_model}"
            )

        # Create pipeline configuration
        pipeline_config = PipelineConfig(
            dataset_path=args.dataset,
            target_column=args.target,
            optimization_metric=args.optimization_metric,
            time_budget=args.time_budget,
            max_models=args.max_models,
            cross_validation_folds=args.cv_folds,
            random_state=args.random_state,
            enable_ensemble=not args.no_ensemble,
            enable_interpretability=not args.no_interpretability,
            enable_meta_learning=not args.no_meta_learning,
        )

        # Override with config file settings
        for key, value in config.items():
            if hasattr(pipeline_config, key):
                setattr(pipeline_config, key, value)

        logger.info("Starting autonomous ML pipeline")
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Target: {args.target}")
        logger.info(f"Optimization metric: {args.optimization_metric}")
        logger.info(f"Time budget: {args.time_budget} seconds")

        # Create and run the agent
        agent = AutonomousMLAgent(
            dataset_path=args.dataset,
            target_column=args.target,
            **vars(pipeline_config),
        )

        # Run the pipeline
        results = await agent.run()

        # Print results
        print_results(results)

        # Save results
        save_results(results, output_dir)

        logger.info("Pipeline completed successfully!")

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


def save_results(results, output_dir: Path):
    """Save pipeline results to files"""
    try:
        # Save leaderboard
        leaderboard_path = output_dir / "leaderboard.csv"
        results.leaderboard.to_csv(leaderboard_path, index=False)
        logger.info(f"Leaderboard saved to {leaderboard_path}")

        # Save feature importance
        feature_importance_path = output_dir / "feature_importance.json"
        import json

        with open(feature_importance_path, "w") as f:
            json.dump(results.feature_importance, f, indent=2)
        logger.info(f"Feature importance saved to {feature_importance_path}")

        # Save model insights
        insights_path = output_dir / "insights.txt"
        with open(insights_path, "w") as f:
            f.write(results.model_insights)
        logger.info(f"Model insights saved to {insights_path}")

        # Save best model
        model_path = output_dir / "best_model.pkl"
        import joblib

        joblib.dump(results.best_model, model_path)
        logger.info(f"Best model saved to {model_path}")

        # Save preprocessing pipeline
        pipeline_path = output_dir / "preprocessing_pipeline.pkl"
        joblib.dump(results.preprocessing_pipeline, pipeline_path)
        logger.info(f"Preprocessing pipeline saved to {pipeline_path}")

    except Exception as e:
        logger.error(f"Failed to save results: {e}")


if __name__ == "__main__":
    asyncio.run(main())
