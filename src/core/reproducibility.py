"""
Experiment Reproducibility and Artifact Management

This module provides comprehensive tools for ensuring experiment reproducibility,
managing artifacts, and maintaining experiment metadata for future reference.
"""

import hashlib
import json
import logging
import os
import pickle
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import git
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentInfo:
    """Information about the execution environment."""

    python_version: str
    platform: str
    architecture: str
    cpu_count: int
    memory_gb: float
    git_commit: str | None = None
    git_branch: str | None = None
    git_remote_url: str | None = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ExperimentMetadata:
    """Metadata for an experiment run."""

    experiment_id: str
    name: str
    description: str
    dataset_path: str
    dataset_hash: str
    target_column: str
    task_type: str
    models_tested: list[str]
    hyperparameters: dict[str, dict[str, Any]]
    preprocessing_config: dict[str, Any]
    environment_info: EnvironmentInfo
    random_seeds: dict[str, int]
    start_time: str
    end_time: str | None = None
    status: str = "running"
    results_summary: dict[str, Any] | None = None
    artifacts: list[str] = None

    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = []


class ReproducibilityManager:
    """Manages experiment reproducibility and artifact storage."""

    def __init__(
        self,
        experiments_dir: str = "experiments",
        artifacts_dir: str = "artifacts",
        metadata_dir: str = "meta",
    ):
        """
        Initialize the reproducibility manager.

        Args:
            experiments_dir: Directory to store experiment data
            artifacts_dir: Directory to store artifacts
            metadata_dir: Directory to store metadata
        """
        self.experiments_dir = Path(experiments_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.metadata_dir = Path(metadata_dir)

        # Create directories
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Initialize random seeds
        self.random_seeds = self._initialize_random_seeds()

        # Capture environment info
        self.environment_info = self._capture_environment_info()

    def _initialize_random_seeds(self) -> dict[str, int]:
        """Initialize random seeds for reproducibility."""
        base_seed = 42
        return {
            "numpy": base_seed,
            "random": base_seed,
            "sklearn": base_seed,
            "xgboost": base_seed,
            "lightgbm": base_seed,
            "catboost": base_seed,
            "torch": base_seed,
            "tensorflow": base_seed,
            "optuna": base_seed,
            "hyperopt": base_seed,
        }

    def _capture_environment_info(self) -> EnvironmentInfo:
        """Capture information about the execution environment."""
        try:
            # Git information
            git_commit = None
            git_branch = None
            git_remote_url = None

            try:
                repo = git.Repo(search_parent_directories=True)
                git_commit = repo.head.commit.hexsha
                git_branch = repo.active_branch.name
                if repo.remotes.origin:
                    git_remote_url = repo.remotes.origin.url
            except Exception:
                pass

            # System information
            memory_gb = 0
            try:
                import psutil

                memory_gb = psutil.virtual_memory().total / (1024**3)
            except ImportError:
                pass

            return EnvironmentInfo(
                python_version=sys.version,
                platform=platform.platform(),
                architecture=platform.architecture()[0],
                cpu_count=os.cpu_count() or 1,
                memory_gb=memory_gb,
                git_commit=git_commit,
                git_branch=git_branch,
                git_remote_url=git_remote_url,
            )
        except Exception as e:
            logger.warning(f"Failed to capture environment info: {str(e)}")
            return EnvironmentInfo(
                python_version=sys.version,
                platform=platform.platform(),
                architecture=platform.architecture()[0],
                cpu_count=os.cpu_count() or 1,
                memory_gb=0,
            )

    def set_random_seeds(self, seeds: dict[str, int] | None = None):
        """
        Set random seeds for all libraries.

        Args:
            seeds: Dictionary of library names to seed values
        """
        if seeds:
            self.random_seeds.update(seeds)

        # Set numpy seed
        np.random.seed(self.random_seeds["numpy"])

        # Set random seed
        import random

        random.seed(self.random_seeds["random"])

        # Set sklearn seed
        try:
            from sklearn.utils import check_random_state

            check_random_state(self.random_seeds["sklearn"])
        except ImportError:
            pass

        # Set other library seeds
        for lib, seed in self.random_seeds.items():
            if lib not in ["numpy", "random", "sklearn"]:
                try:
                    if lib == "xgboost":
                        import xgboost as xgb

                        xgb.set_config(verbosity=0)
                    elif lib == "lightgbm":
                        import lightgbm as lgb

                        lgb.set_config(verbosity=-1)
                    elif lib == "catboost":
                        import catboost as cb

                        cb.set_config(verbose=False)
                    elif lib == "torch":
                        import torch

                        torch.manual_seed(seed)
                    elif lib == "tensorflow":
                        import tensorflow as tf

                        tf.random.set_seed(seed)
                except ImportError:
                    pass

        logger.info(f"Set random seeds: {self.random_seeds}")

    def calculate_dataset_hash(self, dataset_path: str) -> str:
        """
        Calculate hash of dataset for reproducibility tracking.

        Args:
            dataset_path: Path to the dataset

        Returns:
            SHA256 hash of the dataset
        """
        hash_sha256 = hashlib.sha256()

        try:
            with open(dataset_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
        except Exception as e:
            logger.warning(f"Failed to calculate dataset hash: {str(e)}")
            return "unknown"

        return hash_sha256.hexdigest()

    def create_experiment(
        self,
        name: str,
        description: str,
        dataset_path: str,
        target_column: str,
        task_type: str,
        models_to_test: list[str],
        hyperparameters: dict[str, dict[str, Any]] | None = None,
        preprocessing_config: dict[str, Any] | None = None,
    ) -> str:
        """
        Create a new experiment with metadata tracking.

        Args:
            name: Experiment name
            description: Experiment description
            dataset_path: Path to the dataset
            target_column: Name of the target column
            task_type: Type of ML task ('classification' or 'regression')
            models_to_test: List of models to test
            hyperparameters: Model hyperparameters
            preprocessing_config: Preprocessing configuration

        Returns:
            Experiment ID
        """
        # Generate experiment ID
        experiment_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Calculate dataset hash
        dataset_hash = self.calculate_dataset_hash(dataset_path)

        # Create experiment metadata
        metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            name=name,
            description=description,
            dataset_path=dataset_path,
            dataset_hash=dataset_hash,
            target_column=target_column,
            task_type=task_type,
            models_tested=models_to_test,
            hyperparameters=hyperparameters or {},
            preprocessing_config=preprocessing_config or {},
            environment_info=self.environment_info,
            random_seeds=self.random_seeds,
            start_time=datetime.now().isoformat(),
        )

        # Create experiment directory
        experiment_dir = self.experiments_dir / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata_path = experiment_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2, default=str)

        # Save environment snapshot
        self._save_environment_snapshot(experiment_dir)

        logger.info(f"Created experiment: {experiment_id}")
        return experiment_id

    def _save_environment_snapshot(self, experiment_dir: Path):
        """Save environment snapshot for reproducibility."""
        # Save requirements
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
            )
            if result.returncode == 0:
                requirements_path = experiment_dir / "requirements.txt"
                with open(requirements_path, "w") as f:
                    f.write(result.stdout)
        except Exception as e:
            logger.warning(f"Failed to save requirements: {str(e)}")

        # Save environment info
        env_info_path = experiment_dir / "environment_info.json"
        with open(env_info_path, "w") as f:
            json.dump(asdict(self.environment_info), f, indent=2, default=str)

        # Save random seeds
        seeds_path = experiment_dir / "random_seeds.json"
        with open(seeds_path, "w") as f:
            json.dump(self.random_seeds, f, indent=2)

    def save_artifact(
        self,
        experiment_id: str,
        artifact_name: str,
        data: Any,
        artifact_type: str = "pickle",
    ) -> str:
        """
        Save an artifact for an experiment.

        Args:
            experiment_id: Experiment ID
            artifact_name: Name of the artifact
            data: Data to save
            artifact_type: Type of artifact ('pickle', 'json', 'csv', 'numpy', 'joblib')

        Returns:
            Path to saved artifact
        """
        experiment_dir = self.experiments_dir / experiment_id
        artifacts_dir = experiment_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        artifact_path = artifacts_dir / f"{artifact_name}.{artifact_type}"

        try:
            if artifact_type == "pickle":
                with open(artifact_path, "wb") as f:
                    pickle.dump(data, f)
            elif artifact_type == "json":
                with open(artifact_path, "w") as f:
                    json.dump(data, f, indent=2, default=str)
            elif artifact_type == "csv" and isinstance(data, pd.DataFrame):
                data.to_csv(artifact_path, index=False)
            elif artifact_type == "numpy" and isinstance(data, np.ndarray):
                np.save(artifact_path, data)
            elif artifact_type == "joblib":
                joblib.dump(data, artifact_path)
            else:
                raise ValueError(f"Unsupported artifact type: {artifact_type}")

            logger.info(f"Saved artifact: {artifact_path}")
            return str(artifact_path)

        except Exception as e:
            logger.error(f"Failed to save artifact {artifact_name}: {str(e)}")
            raise

    def load_artifact(
        self, experiment_id: str, artifact_name: str, artifact_type: str = "pickle"
    ) -> Any:
        """
        Load an artifact from an experiment.

        Args:
            experiment_id: Experiment ID
            artifact_name: Name of the artifact
            artifact_type: Type of artifact

        Returns:
            Loaded data
        """
        experiment_dir = self.experiments_dir / experiment_id
        artifact_path = (
            experiment_dir / "artifacts" / f"{artifact_name}.{artifact_type}"
        )

        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        try:
            if artifact_type == "pickle":
                with open(artifact_path, "rb") as f:
                    return pickle.load(f)
            elif artifact_type == "json":
                with open(artifact_path) as f:
                    return json.load(f)
            elif artifact_type == "csv":
                return pd.read_csv(artifact_path)
            elif artifact_type == "numpy":
                return np.load(artifact_path)
            elif artifact_type == "joblib":
                return joblib.load(artifact_path)
            else:
                raise ValueError(f"Unsupported artifact type: {artifact_type}")

        except Exception as e:
            logger.error(f"Failed to load artifact {artifact_name}: {str(e)}")
            raise

    def update_experiment_status(
        self,
        experiment_id: str,
        status: str,
        results_summary: dict[str, Any] | None = None,
    ):
        """
        Update experiment status and results.

        Args:
            experiment_id: Experiment ID
            status: New status ('running', 'completed', 'failed')
            results_summary: Summary of results
        """
        experiment_dir = self.experiments_dir / experiment_id
        metadata_path = experiment_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Experiment metadata not found: {metadata_path}")

        # Load and update metadata
        with open(metadata_path) as f:
            metadata_dict = json.load(f)

        metadata_dict["status"] = status
        metadata_dict["end_time"] = datetime.now().isoformat()

        if results_summary:
            metadata_dict["results_summary"] = results_summary

        # Save updated metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2, default=str)

        logger.info(f"Updated experiment {experiment_id} status to {status}")

    def list_experiments(self) -> list[dict[str, Any]]:
        """
        List all experiments.

        Returns:
            List of experiment metadata
        """
        experiments = []

        for experiment_dir in self.experiments_dir.iterdir():
            if experiment_dir.is_dir():
                metadata_path = experiment_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path) as f:
                            metadata = json.load(f)
                        experiments.append(metadata)
                    except Exception as e:
                        logger.warning(
                            f"Failed to load metadata for {experiment_dir.name}: {str(e)}"
                        )

        # Sort by start time (newest first)
        experiments.sort(key=lambda x: x["start_time"], reverse=True)

        return experiments

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        """
        Get experiment metadata.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment metadata
        """
        experiment_dir = self.experiments_dir / experiment_id
        metadata_path = experiment_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Experiment not found: {experiment_id}")

        with open(metadata_path) as f:
            return json.load(f)

    def reproduce_experiment(
        self, experiment_id: str, output_dir: str | None = None
    ) -> str:
        """
        Reproduce an experiment from its metadata.

        Args:
            experiment_id: Experiment ID to reproduce
            output_dir: Output directory for reproduction

        Returns:
            New experiment ID
        """
        # Load original experiment metadata
        original_metadata = self.get_experiment(experiment_id)

        # Create new experiment with reproduction prefix
        reproduction_name = f"reproduction_{original_metadata['name']}"
        reproduction_id = self.create_experiment(
            name=reproduction_name,
            description=f"Reproduction of experiment {experiment_id}",
            dataset_path=original_metadata["dataset_path"],
            target_column=original_metadata["target_column"],
            task_type=original_metadata["task_type"],
            models_to_test=original_metadata["models_tested"],
            hyperparameters=original_metadata["hyperparameters"],
            preprocessing_config=original_metadata["preprocessing_config"],
        )

        # Set the same random seeds
        self.set_random_seeds(original_metadata["random_seeds"])

        logger.info(f"Created reproduction experiment: {reproduction_id}")
        return reproduction_id

    def create_experiment_report(
        self, experiment_id: str, output_path: str | None = None
    ) -> str:
        """
        Create a comprehensive experiment report.

        Args:
            experiment_id: Experiment ID
            output_path: Output path for the report

        Returns:
            Path to the generated report
        """
        # Load experiment metadata
        metadata = self.get_experiment(experiment_id)

        # Generate report path
        if output_path is None:
            output_path = (
                self.experiments_dir / experiment_id / "experiment_report.html"
            )

        output_path = Path(output_path)

        # Create HTML report
        html_content = self._generate_experiment_report_html(metadata, experiment_id)

        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"Created experiment report: {output_path}")
        return str(output_path)

    def _generate_experiment_report_html(
        self, metadata: dict[str, Any], experiment_id: str
    ) -> str:
        """Generate HTML report for an experiment."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Experiment Report: {metadata['name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .section {{ margin: 20px 0; }}
                .metadata {{ background-color: #e9ecef; padding: 15px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .status {{ padding: 5px 10px; border-radius: 3px; font-weight: bold; }}
                .status.running {{ background-color: #fff3cd; color: #856404; }}
                .status.completed {{ background-color: #d4edda; color: #155724; }}
                .status.failed {{ background-color: #f8d7da; color: #721c24; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Experiment Report: {metadata['name']}</h1>
                <p><strong>Experiment ID:</strong> {experiment_id}</p>
                <p><strong>Status:</strong> <span class="status {metadata['status']}">{metadata['status'].upper()}</span></p>
                <p><strong>Description:</strong> {metadata['description']}</p>
            </div>

            <div class="section">
                <h2>Experiment Details</h2>
                <div class="metadata">
                    <p><strong>Start Time:</strong> {metadata['start_time']}</p>
                    <p><strong>End Time:</strong> {metadata.get('end_time', 'N/A')}</p>
                    <p><strong>Task Type:</strong> {metadata['task_type']}</p>
                    <p><strong>Dataset:</strong> {metadata['dataset_path']}</p>
                    <p><strong>Dataset Hash:</strong> <code>{metadata['dataset_hash'][:16]}...</code></p>
                    <p><strong>Target Column:</strong> {metadata['target_column']}</p>
                </div>
            </div>

            <div class="section">
                <h2>Models Tested</h2>
                <ul>
                    {''.join([f'<li>{model}</li>' for model in metadata['models_tested']])}
                </ul>
            </div>

            <div class="section">
                <h2>Environment Information</h2>
                <div class="metadata">
                    <p><strong>Python Version:</strong> {metadata['environment_info']['python_version']}</p>
                    <p><strong>Platform:</strong> {metadata['environment_info']['platform']}</p>
                    <p><strong>Architecture:</strong> {metadata['environment_info']['architecture']}</p>
                    <p><strong>CPU Count:</strong> {metadata['environment_info']['cpu_count']}</p>
                    <p><strong>Memory:</strong> {metadata['environment_info']['memory_gb']:.1f} GB</p>
                    <p><strong>Git Commit:</strong> {metadata['environment_info'].get('git_commit', 'N/A')}</p>
                    <p><strong>Git Branch:</strong> {metadata['environment_info'].get('git_branch', 'N/A')}</p>
                </div>
            </div>

            <div class="section">
                <h2>Random Seeds</h2>
                <table>
                    <tr><th>Library</th><th>Seed</th></tr>
                    {''.join([f'<tr><td>{lib}</td><td>{seed}</td></tr>' for lib, seed in metadata['random_seeds'].items()])}
                </table>
            </div>

            {self._generate_results_section_html(metadata)}

            <div class="section">
                <h2>Reproducibility</h2>
                <p>To reproduce this experiment:</p>
                <ol>
                    <li>Ensure you have the same dataset: <code>{metadata['dataset_path']}</code></li>
                    <li>Set the same random seeds as shown above</li>
                    <li>Use the same hyperparameters and preprocessing configuration</li>
                    <li>Run the experiment with the same environment</li>
                </ol>
            </div>
        </body>
        </html>
        """

        return html_content

    def _generate_results_section_html(self, metadata: dict[str, Any]) -> str:
        """Generate results section HTML."""
        if not metadata.get("results_summary"):
            return "<div class='section'><h2>Results</h2><p>No results available yet.</p></div>"

        results = metadata["results_summary"]
        html = """
        <div class="section">
            <h2>Results Summary</h2>
            <div class="metadata">
        """

        for key, value in results.items():
            html += f"<p><strong>{key}:</strong> {value}</p>"

        html += """
            </div>
        </div>
        """

        return html

    def export_experiment(self, experiment_id: str, output_path: str) -> str:
        """
        Export experiment as a portable archive.

        Args:
            experiment_id: Experiment ID
            output_path: Output path for the archive

        Returns:
            Path to the exported archive
        """
        import shutil

        experiment_dir = self.experiments_dir / experiment_id
        output_path = Path(output_path)

        if output_path.suffix == ".zip":
            shutil.make_archive(
                str(output_path.with_suffix("")), "zip", str(experiment_dir)
            )
        else:
            shutil.copytree(experiment_dir, output_path, dirs_exist_ok=True)

        logger.info(f"Exported experiment {experiment_id} to {output_path}")
        return str(output_path)


def ensure_reproducibility(seed: int = 42) -> ReproducibilityManager:
    """
    Ensure experiment reproducibility by setting up the reproducibility manager.

    Args:
        seed: Base random seed

    Returns:
        Configured ReproducibilityManager
    """
    manager = ReproducibilityManager()
    manager.set_random_seeds({"base": seed})

    logger.info("Reproducibility manager initialized")
    return manager


def create_experiment_checkpoint(
    experiment_id: str,
    checkpoint_name: str,
    data: dict[str, Any],
    reproducibility_manager: ReproducibilityManager,
) -> str:
    """
    Create a checkpoint during experiment execution.

    Args:
        experiment_id: Experiment ID
        checkpoint_name: Name of the checkpoint
        data: Data to checkpoint
        reproducibility_manager: Reproducibility manager instance

    Returns:
        Path to the checkpoint
    """
    checkpoint_path = reproducibility_manager.save_artifact(
        experiment_id, checkpoint_name, data, "json"
    )

    logger.info(f"Created checkpoint: {checkpoint_name}")
    return checkpoint_path
