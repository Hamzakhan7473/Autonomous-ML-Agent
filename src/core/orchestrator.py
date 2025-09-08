"""
LLM Orchestrator for Autonomous Machine Learning Agent

This module orchestrates the entire ML pipeline using LLMs to:
- Generate and modify preprocessing code
- Select appropriate algorithms
- Optimize hyperparameters
- Refine the pipeline iteratively
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import time

import pandas as pd
import numpy as np
from pydantic import BaseModel

from ..utils.llm_client import LLMClient
from ..data.ingestion import DataIngestion
from ..data.preprocessing import DataPreprocessor
from ..data.meta_features import MetaFeatureExtractor
from ..models.algorithms import ModelFactory
from ..models.hyperopt import HyperparameterOptimizer
from ..models.ensemble import EnsembleBuilder
from ..evaluation.metrics import ModelEvaluator
from ..models.interpretability import ModelInterpreter
from ..evaluation.leaderboard import Leaderboard
from ..deployment.registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the ML pipeline"""
    dataset_path: str
    target_column: str
    optimization_metric: str = "accuracy"
    time_budget: int = 3600  # seconds
    max_models: int = 10
    cross_validation_folds: int = 5
    random_state: int = 42
    enable_ensemble: bool = True
    enable_interpretability: bool = True
    enable_meta_learning: bool = True


@dataclass
class PipelineResult:
    """Results from the ML pipeline"""
    best_model: Any
    leaderboard: pd.DataFrame
    preprocessing_pipeline: Any
    feature_importance: Dict[str, float]
    model_insights: str
    training_time: float
    total_iterations: int


class LLMOrchestrator:
    """
    Main orchestrator that uses LLMs to coordinate the ML pipeline
    """
    
    def __init__(self, config: PipelineConfig, llm_client: Optional[LLMClient] = None):
        self.config = config
        self.llm_client = llm_client or LLMClient()
        
        # Initialize components
        self.data_ingestion = DataIngestion()
        self.preprocessor = DataPreprocessor()
        self.meta_extractor = MetaFeatureExtractor()
        self.model_factory = ModelFactory()
        self.hyperopt = HyperparameterOptimizer()
        self.ensemble_builder = EnsembleBuilder()
        self.evaluator = ModelEvaluator()
        self.interpreter = ModelInterpreter()
        self.leaderboard = Leaderboard()
        self.registry = ModelRegistry()
        
        # Pipeline state
        self.data = None
        self.meta_features = None
        self.models = []
        self.results = []
        self.start_time = None
        
    async def run_pipeline(self) -> PipelineResult:
        """
        Run the complete autonomous ML pipeline orchestrated by LLMs
        """
        self.start_time = time.time()
        logger.info("Starting autonomous ML pipeline orchestrated by LLMs")
        
        try:
            # Step 1: Data Ingestion and Analysis
            await self._ingest_and_analyze_data()
            
            # Step 2: LLM-guided Data Preprocessing
            await self._preprocess_data_with_llm()
            
            # Step 3: LLM-guided Model Selection
            await self._select_models_with_llm()
            
            # Step 4: Hyperparameter Optimization with Meta-learning
            await self._optimize_hyperparameters()
            
            # Step 5: Model Training and Evaluation
            await self._train_and_evaluate_models()
            
            # Step 6: Ensemble Building (if enabled)
            if self.config.enable_ensemble:
                await self._build_ensemble()
            
            # Step 7: Model Interpretation
            if self.config.enable_interpretability:
                await self._interpret_models()
            
            # Step 8: Generate Insights and Recommendations
            insights = await self._generate_insights()
            
            # Step 9: Create Final Results
            result = self._create_final_result(insights)
            
            # Step 10: Save to Registry
            await self._save_to_registry(result)
            
            logger.info(f"Pipeline completed in {time.time() - self.start_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    async def _ingest_and_analyze_data(self):
        """Ingest data and extract meta-features"""
        logger.info("Step 1: Ingesting and analyzing data")
        
        # Load data
        self.data = self.data_ingestion.load_data(self.config.dataset_path)
        
        # Extract meta-features for LLM guidance
        self.meta_features = self.meta_extractor.extract_features(self.data, self.config.target_column)
        
        # Generate data analysis prompt for LLM
        analysis_prompt = self._create_data_analysis_prompt()
        analysis_response = await self.llm_client.generate_response(analysis_prompt)
        
        logger.info(f"Data analysis completed. Shape: {self.data.shape}")
        logger.info(f"Meta-features: {self.meta_features}")
    
    async def _preprocess_data_with_llm(self):
        """Use LLM to generate preprocessing code"""
        logger.info("Step 2: LLM-guided data preprocessing")
        
        # Create preprocessing prompt
        preprocessing_prompt = self._create_preprocessing_prompt()
        preprocessing_code = await self.llm_client.generate_code(preprocessing_prompt)
        
        # Execute the generated preprocessing code
        self.data = self.preprocessor.apply_llm_generated_preprocessing(
            self.data, preprocessing_code, self.config.target_column
        )
        
        logger.info("LLM-guided preprocessing completed")
    
    async def _select_models_with_llm(self):
        """Use LLM to select appropriate models"""
        logger.info("Step 3: LLM-guided model selection")
        
        # Create model selection prompt
        model_selection_prompt = self._create_model_selection_prompt()
        model_selection_response = await self.llm_client.generate_response(model_selection_prompt)
        
        # Parse LLM response and select models
        selected_models = self._parse_model_selection_response(model_selection_response)
        self.models = [self.model_factory.create_model(model_name) for model_name in selected_models]
        
        logger.info(f"Selected models: {selected_models}")
    
    async def _optimize_hyperparameters(self):
        """Optimize hyperparameters with meta-learning warm starts"""
        logger.info("Step 4: Hyperparameter optimization with meta-learning")
        
        # Get meta-learning warm starts if available
        warm_starts = await self._get_meta_learning_warm_starts()
        
        # Create optimization prompt
        optimization_prompt = self._create_optimization_prompt(warm_starts)
        optimization_strategy = await self.llm_client.generate_response(optimization_prompt)
        
        # Apply optimization strategy
        for model in self.models:
            optimized_params = self.hyperopt.optimize_with_llm_guidance(
                model, self.data, self.config.target_column, optimization_strategy, warm_starts
            )
            model.set_params(**optimized_params)
        
        logger.info("Hyperparameter optimization completed")
    
    async def _train_and_evaluate_models(self):
        """Train and evaluate all models"""
        logger.info("Step 5: Training and evaluating models")
        
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}: {model.__class__.__name__}")
            
            # Train model
            train_start = time.time()
            model.fit(self.data.drop(columns=[self.config.target_column]), self.data[self.config.target_column])
            train_time = time.time() - train_start
            
            # Evaluate model
            evaluation_result = self.evaluator.evaluate_model(
                model, self.data, self.config.target_column, self.config.cross_validation_folds
            )
            evaluation_result['training_time'] = train_time
            evaluation_result['model_name'] = model.__class__.__name__
            
            self.results.append(evaluation_result)
            
            # Check time budget
            if time.time() - self.start_time > self.config.time_budget:
                logger.warning("Time budget exceeded, stopping model training")
                break
        
        # Update leaderboard
        self.leaderboard.update_results(self.results)
    
    async def _build_ensemble(self):
        """Build ensemble of top models"""
        logger.info("Step 6: Building ensemble")
        
        # Get top models
        top_models = self.leaderboard.get_top_models(k=3)
        
        # Create ensemble prompt
        ensemble_prompt = self._create_ensemble_prompt(top_models)
        ensemble_strategy = await self.llm_client.generate_response(ensemble_prompt)
        
        # Build ensemble
        ensemble_model = self.ensemble_builder.build_ensemble_with_llm_guidance(
            top_models, self.data, self.config.target_column, ensemble_strategy
        )
        
        # Evaluate ensemble
        ensemble_result = self.evaluator.evaluate_model(
            ensemble_model, self.data, self.config.target_column, self.config.cross_validation_folds
        )
        ensemble_result['model_name'] = 'Ensemble'
        
        self.results.append(ensemble_result)
        self.leaderboard.update_results(self.results)
        
        logger.info("Ensemble building completed")
    
    async def _interpret_models(self):
        """Generate model interpretations"""
        logger.info("Step 7: Model interpretation")
        
        best_model = self.leaderboard.get_best_model()
        
        # Generate feature importance
        feature_importance = self.interpreter.get_feature_importance(
            best_model, self.data, self.config.target_column
        )
        
        # Generate model explanation
        explanation_prompt = self._create_explanation_prompt(best_model, feature_importance)
        model_explanation = await self.llm_client.generate_response(explanation_prompt)
        
        self.feature_importance = feature_importance
        self.model_explanation = model_explanation
        
        logger.info("Model interpretation completed")
    
    async def _generate_insights(self) -> str:
        """Generate insights and recommendations"""
        logger.info("Step 8: Generating insights")
        
        insights_prompt = self._create_insights_prompt()
        insights = await self.llm_client.generate_response(insights_prompt)
        
        return insights
    
    def _create_final_result(self, insights: str) -> PipelineResult:
        """Create final pipeline result"""
        best_model = self.leaderboard.get_best_model()
        
        return PipelineResult(
            best_model=best_model,
            leaderboard=self.leaderboard.get_leaderboard(),
            preprocessing_pipeline=self.preprocessor.get_pipeline(),
            feature_importance=self.feature_importance,
            model_insights=insights,
            training_time=time.time() - self.start_time,
            total_iterations=len(self.results)
        )
    
    async def _save_to_registry(self, result: PipelineResult):
        """Save results to model registry"""
        logger.info("Step 9: Saving to model registry")
        
        await self.registry.save_model(
            result.best_model,
            result.preprocessing_pipeline,
            self.meta_features,
            result.leaderboard,
            self.config
        )
        
        logger.info("Results saved to model registry")
    
    # Prompt creation methods
    def _create_data_analysis_prompt(self) -> str:
        """Create prompt for data analysis"""
        return f"""
        Analyze this dataset and provide insights:
        
        Dataset Info:
        - Shape: {self.data.shape}
        - Target column: {self.config.target_column}
        - Target distribution: {self.data[self.config.target_column].value_counts().to_dict()}
        - Meta-features: {self.meta_features}
        
        Please provide:
        1. Data quality assessment
        2. Potential preprocessing needs
        3. Feature engineering opportunities
        4. Model selection recommendations
        """
    
    def _create_preprocessing_prompt(self) -> str:
        """Create prompt for preprocessing code generation"""
        return f"""
        Generate Python code for preprocessing this dataset:
        
        Dataset characteristics:
        - Shape: {self.data.shape}
        - Columns: {list(self.data.columns)}
        - Target: {self.config.target_column}
        - Meta-features: {self.meta_features}
        
        Generate preprocessing code that handles:
        1. Missing values
        2. Categorical encoding
        3. Feature scaling
        4. Outlier detection
        5. Feature engineering
        
        Return only the Python code, no explanations.
        """
    
    def _create_model_selection_prompt(self) -> str:
        """Create prompt for model selection"""
        return f"""
        Select the best machine learning models for this dataset:
        
        Dataset characteristics:
        - Shape: {self.data.shape}
        - Target type: {self.data[self.config.target_column].dtype}
        - Target distribution: {self.data[self.config.target_column].value_counts().to_dict()}
        - Meta-features: {self.meta_features}
        - Optimization metric: {self.config.optimization_metric}
        
        Available models: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, kNN, MLP
        
        Select the top 5 models and explain why each is suitable.
        """
    
    def _create_optimization_prompt(self, warm_starts: Dict) -> str:
        """Create prompt for hyperparameter optimization strategy"""
        return f"""
        Design a hyperparameter optimization strategy:
        
        Dataset characteristics: {self.meta_features}
        Selected models: {[model.__class__.__name__ for model in self.models]}
        Meta-learning warm starts: {warm_starts}
        Time budget: {self.config.time_budget} seconds
        
        Design an optimization strategy that:
        1. Uses meta-learning warm starts effectively
        2. Balances exploration vs exploitation
        3. Respects time constraints
        4. Focuses on {self.config.optimization_metric}
        """
    
    def _create_ensemble_prompt(self, top_models: List) -> str:
        """Create prompt for ensemble strategy"""
        return f"""
        Design an ensemble strategy for these top models:
        
        Top models: {[model.__class__.__name__ for model in top_models]}
        Their performances: {[result[self.config.optimization_metric] for result in self.results[:3]]}
        
        Design an ensemble that:
        1. Combines the strengths of different models
        2. Handles overfitting
        3. Maintains interpretability
        4. Optimizes for {self.config.optimization_metric}
        """
    
    def _create_explanation_prompt(self, model, feature_importance: Dict) -> str:
        """Create prompt for model explanation"""
        return f"""
        Explain this model's behavior:
        
        Model: {model.__class__.__name__}
        Performance: {self.leaderboard.get_best_score()}
        Feature importance: {feature_importance}
        
        Provide a natural language explanation of:
        1. How the model makes decisions
        2. Which features are most important
        3. Model strengths and limitations
        4. Recommendations for improvement
        """
    
    def _create_insights_prompt(self) -> str:
        """Create prompt for generating insights"""
        return f"""
        Generate insights from this ML pipeline:
        
        Results summary:
        - Total models trained: {len(self.results)}
        - Best model: {self.leaderboard.get_best_model().__class__.__name__}
        - Best score: {self.leaderboard.get_best_score()}
        - Training time: {time.time() - self.start_time:.2f} seconds
        
        Provide actionable insights about:
        1. Model performance patterns
        2. Feature importance insights
        3. Recommendations for improvement
        4. Deployment considerations
        """
    
    async def _get_meta_learning_warm_starts(self) -> Dict:
        """Get meta-learning warm starts from registry"""
        if not self.config.enable_meta_learning:
            return {}
        
        try:
            warm_starts = await self.registry.get_meta_learning_warm_starts(self.meta_features)
            return warm_starts
        except Exception as e:
            logger.warning(f"Failed to get meta-learning warm starts: {e}")
            return {}
    
    def _parse_model_selection_response(self, response: str) -> List[str]:
        """Parse LLM response to extract selected models"""
        # Simple parsing - in production, use more robust parsing
        model_mapping = {
            'logistic regression': 'logistic_regression',
            'random forest': 'random_forest',
            'xgboost': 'xgboost',
            'lightgbm': 'lightgbm',
            'catboost': 'catboost',
            'knn': 'knn',
            'mlp': 'mlp'
        }
        
        selected_models = []
        response_lower = response.lower()
        
        for model_name, model_key in model_mapping.items():
            if model_name in response_lower:
                selected_models.append(model_key)
        
        # Default models if none selected
        if not selected_models:
            selected_models = ['logistic_regression', 'random_forest', 'xgboost']
        
        return selected_models[:self.config.max_models]


class AutonomousMLAgent:
    """
    High-level interface for the autonomous ML agent
    """
    
    def __init__(self, dataset_path: str, target_column: str, **kwargs):
        self.config = PipelineConfig(dataset_path=dataset_path, target_column=target_column, **kwargs)
        self.orchestrator = LLMOrchestrator(self.config)
    
    async def run(self) -> PipelineResult:
        """Run the complete autonomous ML pipeline"""
        return await self.orchestrator.run_pipeline()
    
    def run_sync(self) -> PipelineResult:
        """Synchronous version of run()"""
        return asyncio.run(self.run())
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Get the current leaderboard"""
        return self.orchestrator.leaderboard.get_leaderboard()
    
    def get_best_model(self):
        """Get the best performing model"""
        return self.orchestrator.leaderboard.get_best_model()
