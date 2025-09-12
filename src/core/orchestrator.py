"""
LLM Orchestrator for Autonomous Machine Learning Agent

This module orchestrates the entire ML pipeline using LLMs to:
- Generate and modify preprocessing code
- Select appropriate algorithms
- Optimize hyperparameters
- Refine the pipeline iteratively
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..agent_llm.planner import MLPlanner
from ..core.ingest import DatasetSchema, analyze_data
from ..core.model_zoo import ModelZoo
from ..data.ingestion import DataIngestion
from ..data.meta_features import MetaFeatureExtractor
from ..data.preprocessing import DataPreprocessor, PreprocessingConfig
from ..deployment.registry import ModelRegistry
from ..evaluation.leaderboard import Leaderboard
from ..evaluation.metrics import ModelEvaluator
from ..models.algorithms import ModelFactory
from ..models.ensemble import EnsembleBuilder
from ..models.hyperopt import HyperparameterOptimizer
from ..models.interpretability import ModelInterpreter
from ..utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the ML pipeline."""

    time_budget: int = 3600  # seconds
    optimization_metric: str = "auto"
    random_state: int = 42
    output_dir: str = "./results"
    save_models: bool = True
    save_results: bool = True
    verbose: bool = False


@dataclass
class PipelineResults:
    """Results from the ML pipeline execution."""

    best_model: Any
    best_score: float
    best_params: dict[str, Any]
    all_results: list[dict[str, Any]]
    preprocessing_config: Any
    execution_plan: Any
    execution_time: float
    data_summary: dict[str, Any]
    model_insights: str


class LLMOrchestrator:
    """
    Main orchestrator that uses LLMs to coordinate the ML pipeline
    """

    def __init__(self, config: PipelineConfig, llm_client: LLMClient | None = None):
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

    async def run_pipeline(self) -> PipelineResults:
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

            logger.info(
                f"Pipeline completed in {time.time() - self.start_time:.2f} seconds"
            )
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
        self.meta_features = self.meta_extractor.extract_features(
            self.data, self.config.target_column
        )

        # Generate data analysis prompt for LLM
        analysis_prompt = self._create_data_analysis_prompt()
        await self.llm_client.generate_response(analysis_prompt)

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
        model_selection_response = await self.llm_client.generate_response(
            model_selection_prompt
        )

        # Parse LLM response and select models
        selected_models = self._parse_model_selection_response(model_selection_response)
        self.models = [
            self.model_factory.create_model(model_name)
            for model_name in selected_models
        ]

        logger.info(f"Selected models: {selected_models}")

    async def _optimize_hyperparameters(self):
        """Optimize hyperparameters with meta-learning warm starts"""
        logger.info("Step 4: Hyperparameter optimization with meta-learning")

        # Get meta-learning warm starts if available
        warm_starts = await self._get_meta_learning_warm_starts()

        # Create optimization prompt
        optimization_prompt = self._create_optimization_prompt(warm_starts)
        optimization_strategy = await self.llm_client.generate_response(
            optimization_prompt
        )

        # Apply optimization strategy
        for model in self.models:
            optimized_params = self.hyperopt.optimize_with_llm_guidance(
                model,
                self.data,
                self.config.target_column,
                optimization_strategy,
                warm_starts,
            )
            model.set_params(**optimized_params)

        logger.info("Hyperparameter optimization completed")

    async def _train_and_evaluate_models(self):
        """Train and evaluate all models"""
        logger.info("Step 5: Training and evaluating models")

        for i, model in enumerate(self.models):
            logger.info(
                f"Training model {i+1}/{len(self.models)}: {model.__class__.__name__}"
            )

            # Train model
            train_start = time.time()
            model.fit(
                self.data.drop(columns=[self.config.target_column]),
                self.data[self.config.target_column],
            )
            train_time = time.time() - train_start

            # Evaluate model
            evaluation_result = self.evaluator.evaluate_model(
                model,
                self.data,
                self.config.target_column,
                self.config.cross_validation_folds,
            )
            evaluation_result["training_time"] = train_time
            evaluation_result["model_name"] = model.__class__.__name__

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
            ensemble_model,
            self.data,
            self.config.target_column,
            self.config.cross_validation_folds,
        )
        ensemble_result["model_name"] = "Ensemble"

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
        explanation_prompt = self._create_explanation_prompt(
            best_model, feature_importance
        )
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

    def _create_final_result(self, insights: str) -> PipelineResults:
        """Create final pipeline result"""
        best_model = self.leaderboard.get_best_model()

        return PipelineResults(
            best_model=best_model,
            leaderboard=self.leaderboard.get_leaderboard(),
            preprocessing_pipeline=self.preprocessor.get_pipeline(),
            feature_importance=self.feature_importance,
            model_insights=insights,
            training_time=time.time() - self.start_time,
            total_iterations=len(self.results),
        )

    async def _save_to_registry(self, result: PipelineResults):
        """Save results to model registry"""
        logger.info("Step 9: Saving to model registry")

        await self.registry.save_model(
            result.best_model,
            result.preprocessing_pipeline,
            self.meta_features,
            result.leaderboard,
            self.config,
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

    def _create_optimization_prompt(self, warm_starts: dict) -> str:
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

    def _create_ensemble_prompt(self, top_models: list) -> str:
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

    def _create_explanation_prompt(self, model, feature_importance: dict) -> str:
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

    async def _get_meta_learning_warm_starts(self) -> dict:
        """Get meta-learning warm starts from registry"""
        if not self.config.enable_meta_learning:
            return {}

        try:
            warm_starts = await self.registry.get_meta_learning_warm_starts(
                self.meta_features
            )
            return warm_starts
        except Exception as e:
            logger.warning(f"Failed to get meta-learning warm starts: {e}")
            return {}

    def _parse_model_selection_response(self, response: str) -> list[str]:
        """Parse LLM response to extract selected models"""
        # Simple parsing - in production, use more robust parsing
        model_mapping = {
            "logistic regression": "logistic_regression",
            "random forest": "random_forest",
            "xgboost": "xgboost",
            "lightgbm": "lightgbm",
            "catboost": "catboost",
            "knn": "knn",
            "mlp": "mlp",
        }

        selected_models = []
        response_lower = response.lower()

        for model_name, model_key in model_mapping.items():
            if model_name in response_lower:
                selected_models.append(model_key)

        # Default models if none selected
        if not selected_models:
            selected_models = ["logistic_regression", "random_forest", "xgboost"]

        return selected_models[: self.config.max_models]


class AutonomousMLAgent:
    """Main orchestrator for autonomous machine learning."""

    def __init__(
        self, config: PipelineConfig | None = None, llm_client: LLMClient | None = None
    ):
        """Initialize the autonomous ML agent."""
        self.config = config or PipelineConfig()
        self.llm_client = llm_client or LLMClient()
        self.planner = MLPlanner(self.llm_client)

        # Set random seed
        np.random.seed(self.config.random_state)

        # Initialize components
        self.preprocessor = None
        self.evaluator = ModelEvaluator()

        # Results storage
        self.results = None
        self.execution_history = []

    def run(
        self,
        dataset_path: str,
        target_column: str,
        config: PipelineConfig | None = None,
    ) -> PipelineResults:
        """Run the complete autonomous ML pipeline."""

        if config:
            self.config = config

        start_time = time.time()

        try:
            logger.info(f"Starting autonomous ML pipeline for {dataset_path}")

            # Step 1: Data Analysis
            logger.info("Step 1: Analyzing dataset...")
            df, schema, summary = self._analyze_data(dataset_path, target_column)

            # Step 2: LLM Planning
            logger.info("Step 2: Creating execution plan...")
            execution_plan = self._create_execution_plan(schema, summary)

            # Step 3: Data Preprocessing
            logger.info("Step 3: Preprocessing data...")
            X_processed, y_processed, preprocessor = self._preprocess_data(
                df, target_column, execution_plan
            )

            # Step 4: Model Training and Optimization
            logger.info("Step 4: Training and optimizing models...")
            model_results = self._train_models(
                X_processed, y_processed, execution_plan
            )

            # Step 5: Model Evaluation
            logger.info("Step 5: Evaluating models...")
            evaluation_results = self._evaluate_models(
                model_results, X_processed, y_processed
            )

            # Step 6: Generate Insights
            logger.info("Step 6: Generating insights...")
            insights = self._generate_insights(evaluation_results, execution_plan)

            # Step 7: Save Results
            if self.config.save_results:
                logger.info("Step 7: Saving results...")
                self._save_results(evaluation_results, execution_plan, schema, summary)

            execution_time = time.time() - start_time

            # Create results object
            self.results = PipelineResults(
                best_model=evaluation_results["best_model"],
                best_score=evaluation_results["best_score"],
                best_params=evaluation_results["best_params"],
                all_results=evaluation_results["all_results"],
                preprocessing_config=None,  # DataPreprocessor doesn't have config attribute
                execution_plan=execution_plan,
                execution_time=execution_time,
                data_summary=summary,
                model_insights=insights,
            )

            logger.info(
                f"Pipeline completed successfully in {execution_time:.2f} seconds"
            )

            return self.results

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise

    def _analyze_data(self, dataset_path: str, target_column: str):
        """Analyze the dataset."""
        return analyze_data(dataset_path, target_column)

    def _create_execution_plan(self, schema: DatasetSchema, summary: dict[str, Any]):
        """Create execution plan using LLM."""
        prior_runs = self._load_prior_runs()
        return self.planner.create_plan(schema, summary, prior_runs)

    def _preprocess_data(self, df: pd.DataFrame, target_column: str, execution_plan):
        """Preprocess the data according to the execution plan."""
        preprocess_config = self._create_preprocessing_config(execution_plan)
        self.preprocessor = DataPreprocessor(target_column, preprocess_config.random_state)

        X_processed, y_processed = self.preprocessor.fit_transform(df)

        return X_processed, y_processed, self.preprocessor

    def _create_preprocessing_config(self, execution_plan):
        """Create preprocessing configuration from execution plan."""
        config = PreprocessingConfig()

        for step in execution_plan.preprocessing_steps:
            step_name = step["step"]
            method = step["method"]

            if step_name == "missing_value_imputation":
                config.imputation_strategy = method
            elif step_name == "categorical_encoding":
                config.categorical_encoding = method
            elif step_name == "scaling":
                config.scaling = method
            elif step_name == "feature_selection":
                config.feature_selection = step.get("enabled", True)

        return config

    def _train_models(self, X: pd.DataFrame, y: pd.Series, execution_plan):
        """Train and optimize models according to the execution plan."""
        model_results = []
        time_per_model = self.config.time_budget // len(execution_plan.models_to_try)

        # Initialize model zoo
        model_zoo = ModelZoo()

        for i, model_name in enumerate(execution_plan.models_to_try):
            logger.info(
                f"Training model {i+1}/{len(execution_plan.models_to_try)}: {model_name}"
            )

            try:
                is_classification = execution_plan.models_to_try[
                    0
                ] in model_zoo.list_models(is_classification=True)
                model = model_zoo.get_model(model_name, is_classification)

                strategy = execution_plan.hyperparameter_strategies.get(
                    model_name, "random"
                )

                # Get default parameter grid for the model
                param_grid = self._get_default_param_grid(model_name)
                
                optimizer = HyperparameterOptimizer(
                    model=model.model,  # Get the actual sklearn model
                    param_grid=param_grid,
                    cv=5,
                    scoring="accuracy",
                    n_jobs=-1,
                )

                # Convert to numpy arrays if needed
                X_array = X.values if hasattr(X, 'values') else X
                y_array = y.values if hasattr(y, 'values') else y
                
                best_model, best_score, best_params = optimizer.optimize(X_array, y_array, method=strategy)

                model_results.append(
                    {
                        "model_name": model_name,
                        "model": best_model,
                        "score": best_score,
                        "params": best_params,
                        "training_time": 0.0,  # Placeholder - sklearn models don't track training time
                    }
                )

            except Exception as e:
                logger.error(f"Failed to train model {model_name}: {e}")
                continue

        return model_results

    def _get_default_param_grid(self, model_name: str) -> dict:
        """Get default parameter grid for a model."""
        param_grids = {
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10],
            },
            "logistic_regression": {
                "C": [0.1, 1.0, 10.0],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"],
            },
            "xgboost": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
            },
            "lightgbm": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
            },
            "catboost": {
                "iterations": [50, 100, 200],
                "depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
            },
            "knn": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
            },
            "mlp": {
                "hidden_layer_sizes": [(50,), (100,), (50, 50)],
                "activation": ["relu", "tanh"],
                "alpha": [0.0001, 0.001, 0.01],
            },
        }
        return param_grids.get(model_name, {})

    def _evaluate_models(
        self, model_results: list[dict[str, Any]], X: pd.DataFrame, y: pd.Series
    ):
        """Evaluate all trained models."""
        if not model_results:
            raise ValueError("No models were successfully trained")

        evaluation_results = []
        best_score = -np.inf
        best_model = None
        best_params = None
        best_model_name = None

        for result in model_results:
            try:
                eval_result = self.evaluator.evaluate(result["model"], X, y)
                eval_result["model_name"] = result["model_name"]
                eval_result["params"] = result["params"]
                eval_result["training_time"] = result["training_time"]

                evaluation_results.append(eval_result)

                primary_metric = eval_result["primary_metric"]
                cv_results = eval_result["cv_results"]

                if primary_metric in cv_results and cv_results[primary_metric]:
                    score = cv_results[primary_metric]["mean"]
                    if score > best_score:
                        best_score = score
                        best_model = result["model"]
                        best_params = result["params"]
                        best_model_name = result["model_name"]

            except Exception as e:
                logger.error(f"Failed to evaluate model {result['model_name']}: {e}")
                continue

        return {
            "all_results": evaluation_results,
            "best_model": best_model,
            "best_score": best_score,
            "best_params": best_params,
            "best_model_name": best_model_name,
        }

    def _generate_insights(self, evaluation_results: dict[str, Any], execution_plan):
        """Generate insights using LLM."""
        try:
            return self.planner.explain_results(evaluation_results, execution_plan)
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return "Insights generation failed due to technical issues."

    def _save_results(
        self,
        evaluation_results: dict[str, Any],
        execution_plan,
        schema: DatasetSchema,
        summary: dict[str, Any],
    ):
        """Save results to files."""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_data = {
            "evaluation_results": evaluation_results,
            "execution_plan": execution_plan.__dict__,
            "schema": schema.__dict__,
            "summary": summary,
            "config": self.config.__dict__,
            "timestamp": time.time(),
        }

        with open(output_path / "pipeline_results.json", "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        if evaluation_results["all_results"]:
            comparison_df = self.evaluator.compare_models(
                evaluation_results["all_results"]
            )
            comparison_df.to_csv(output_path / "model_comparison.csv", index=False)

        if self.config.save_models and evaluation_results["best_model"]:
            import joblib

            joblib.dump(
                evaluation_results["best_model"], output_path / "best_model.pkl"
            )
            joblib.dump(self.preprocessor, output_path / "preprocessor.pkl")

        logger.info(f"Results saved to {output_path}")

    def _load_prior_runs(self):
        """Load prior run results for meta-learning."""
        try:
            results_file = Path(self.config.output_dir) / "pipeline_results.json"
            if results_file.exists():
                with open(results_file) as f:
                    prior_data = json.load(f)
                    return [prior_data]
        except Exception as e:
            logger.warning(f"Could not load prior runs: {e}")

        return []

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the best model."""
        if self.results is None:
            raise ValueError("No trained model available. Run the pipeline first.")

        X_processed = self.preprocessor.transform(X)
        return self.results.best_model.predict(X_processed)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions (for classification)."""
        if self.results is None:
            raise ValueError("No trained model available. Run the pipeline first.")

        if not self.results.best_model.config.is_classification:
            raise ValueError(
                "Probability predictions only available for classification models"
            )

        X_processed = self.preprocessor.transform(X)
        return self.results.best_model.predict_proba(X_processed)

    def get_feature_importance(self) -> np.ndarray | None:
        """Get feature importance from the best model."""
        if self.results is None:
            raise ValueError("No trained model available. Run the pipeline first.")

        return self.results.best_model.get_feature_importance()

    def get_model_summary(self) -> dict[str, Any]:
        """Get summary of the best model."""
        if self.results is None:
            raise ValueError("No trained model available. Run the pipeline first.")

        return {
            "model_name": self.results.best_model.config.name,
            "best_score": self.results.best_score,
            "best_params": self.results.best_params,
            "execution_time": self.results.execution_time,
            "insights": self.results.model_insights,
        }


def run_autonomous_ml(
    dataset_path: str, target_column: str, config: PipelineConfig | None = None
) -> PipelineResults:
    """Convenience function to run autonomous ML pipeline."""
    agent = AutonomousMLAgent(config)
    return agent.run(dataset_path, target_column)
