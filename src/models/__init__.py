"""
Models Package

This package contains model-related modules for the autonomous ML agent.
"""

from .algorithms import ModelFactory
from .interpretability import ModelInterpreter, FeatureImportanceAnalyzer, ModelExplainer
from .meta_learning import MetaLearningEngine, MetaLearningConfig
from .ensemble_strategies import EnsembleStrategyManager, EnsembleStrategy
from .natural_language_summaries import NaturalLanguageSummarizer

__all__ = [
    "ModelFactory", 
    "ModelInterpreter", 
    "FeatureImportanceAnalyzer", 
    "ModelExplainer",
    "MetaLearningEngine", 
    "MetaLearningConfig",
    "EnsembleStrategyManager", 
    "EnsembleStrategy",
    "NaturalLanguageSummarizer"
]
