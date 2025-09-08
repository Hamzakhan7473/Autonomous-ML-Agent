"""Autonomous ML Agent - AI-powered machine learning pipeline with LLM orchestration."""

__version__ = "0.1.0"
__author__ = "Autonomous ML Team"
__email__ = "team@autotab-ml.com"

from .core.orchestrator import AutonomousMLAgent, PipelineConfig, PipelineResults
from .core.ingest import analyze_data, DatasetSchema
from .core.preprocess import DataPreprocessor, PreprocessingConfig
from .core.model_zoo import model_zoo
from .core.search import HyperparameterOptimizer
from .core.evaluate import ModelEvaluator
from .agent_llm.planner import MLPlanner
from .utils.llm_client import LLMClient
from .ensemble.blending import EnsembleBlender, create_ensemble

__all__ = [
    "AutonomousMLAgent",
    "PipelineConfig", 
    "PipelineResults",
    "analyze_data",
    "DatasetSchema",
    "DataPreprocessor",
    "PreprocessingConfig",
    "model_zoo",
    "HyperparameterOptimizer",
    "ModelEvaluator",
    "MLPlanner",
    "LLMClient",
    "EnsembleBlender",
    "create_ensemble"
]
