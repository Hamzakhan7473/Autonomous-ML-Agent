"""LLM-based planning and orchestration module."""

import json
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..core.ingest import DatasetSchema
from ..utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class MLPlan:
    """Machine learning execution plan."""

    preprocessing_steps: list[dict[str, Any]]
    models_to_try: list[str]
    hyperparameter_strategies: dict[str, str]
    ensemble_strategy: str | None
    evaluation_metrics: list[str]
    reasoning: str
    confidence: float


class MLPlanner:
    """LLM-powered machine learning planner."""

    def __init__(self, llm_client: LLMClient):
        """Initialize the ML planner.

        Args:
            llm_client: LLM client for API calls
        """
        self.llm_client = llm_client
        self.plan_history = []

    def create_plan(
        self,
        schema: DatasetSchema,
        summary: dict[str, Any],
        prior_runs: list[dict[str, Any]] | None = None,
    ) -> MLPlan:
        """Create an ML execution plan based on dataset characteristics.

        Args:
            schema: Dataset schema information
            summary: Detailed data summary
            prior_runs: Previous run results for meta-learning

        Returns:
            ML execution plan
        """

        # Prepare context for LLM
        context = self._prepare_context(schema, summary, prior_runs)

        # Generate plan using LLM
        plan_response = self._generate_plan_with_llm(context, schema, summary)

        # Parse and validate plan
        plan = self._parse_plan_response(plan_response, schema, summary)

        # Store plan in history
        self.plan_history.append(
            {
                "schema": schema.__dict__,
                "summary": summary,
                "plan": plan,
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        )

        return plan

    def _prepare_context(
        self,
        schema: DatasetSchema,
        summary: dict[str, Any],
        prior_runs: list[dict[str, Any]] | None,
    ) -> str:
        """Prepare context for LLM planning."""

        context = f"""
Dataset Characteristics:
- Rows: {schema.n_rows:,}
- Features: {schema.n_features}
- Categorical features: {schema.n_categorical}
- Numerical features: {schema.n_numerical}
- Missing values: {schema.n_missing:,} ({schema.missing_percentage:.2f}%)
- Target type: {schema.target_type}
- Memory usage: {schema.memory_usage_mb:.2f} MB

Data Quality:
- Duplicate rows: {summary['data_quality']['duplicate_rows']}
- Constant features: {summary['data_quality']['constant_features']}
- High missing features: {summary['data_quality']['high_missing_features']}

Target Summary:
- Type: {summary['target_summary']['type']}
- Unique values: {summary['target_summary']['unique_values']}
- Missing count: {summary['target_summary']['missing_count']}
"""

        if schema.class_balance:
            context += "\nClass Balance:\n"
            for class_name, proportion in schema.class_balance.items():
                context += f"- {class_name}: {proportion:.3f}\n"

        if prior_runs:
            context += f"\nPrior Run Results ({len(prior_runs)} runs):\n"
            for i, run in enumerate(prior_runs[-3:]):  # Show last 3 runs
                context += f"Run {i+1}: Best model: {run.get('best_model', 'N/A')}, "
                context += f"Score: {run.get('best_score', 'N/A'):.4f}\n"

        return context

    def _generate_plan_with_llm(self, context: str, schema: DatasetSchema, summary: dict[str, Any]) -> str:
        """Generate ML plan using LLM."""

        prompt = f"""
You are an expert machine learning engineer. Based on the dataset characteristics below, create a comprehensive ML execution plan.

{context}

Please provide a JSON response with the following structure:
{{
    "preprocessing_steps": [
        {{
            "step": "missing_value_imputation",
            "method": "auto|mean|median|mode|knn",
            "reasoning": "Why this method is chosen"
        }},
        {{
            "step": "categorical_encoding",
            "method": "auto|onehot|label|target",
            "reasoning": "Why this method is chosen"
        }},
        {{
            "step": "scaling",
            "method": "auto|standard|minmax|robust|none",
            "reasoning": "Why this method is chosen"
        }},
        {{
            "step": "feature_selection",
            "enabled": true|false,
            "reasoning": "Why feature selection is needed/not needed"
        }}
    ],
    "models_to_try": [
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "neural_network"
    ],
    "hyperparameter_strategies": {{
        "logistic_regression": "random|bayesian",
        "random_forest": "random|bayesian",
        "xgboost": "bayesian",
        "lightgbm": "bayesian",
        "neural_network": "random"
    }},
    "ensemble_strategy": "blending|stacking|none",
    "evaluation_metrics": ["accuracy", "f1", "auc"],
    "reasoning": "Overall reasoning for the plan",
    "confidence": 0.85
}}

Guidelines:
1. Choose preprocessing methods based on data characteristics
2. Select models appropriate for dataset size and type
3. Use Bayesian optimization for complex models, random search for simple ones
4. Consider ensemble methods for high-stakes decisions
5. Select evaluation metrics appropriate for the task type
6. Provide clear reasoning for each decision
7. Set confidence between 0.0 and 1.0

Respond with only the JSON, no additional text.
"""

        try:
            response = self.llm_client.generate_response(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM planning failed: {e}")
            # Return a simple fallback JSON response
            return json.dumps({
                "preprocessing_steps": [
                    {
                        "step": "missing_value_imputation",
                        "method": "mean",
                        "reasoning": "Fallback: Using mean imputation for missing values"
                    },
                    {
                        "step": "scaling",
                        "method": "standard",
                        "reasoning": "Fallback: Using standard scaling"
                    }
                ],
                "models_to_try": ["random_forest", "logistic_regression"],
                "hyperparameter_strategies": {
                    "random_forest": "random",
                    "logistic_regression": "random"
                },
                "ensemble_strategy": "none",
                "evaluation_metrics": ["accuracy"],
                "reasoning": "Fallback plan due to LLM unavailability",
                "confidence": 0.5
            })

    def _parse_plan_response(self, response: str, schema: DatasetSchema, summary: dict[str, Any]) -> MLPlan:
        """Parse LLM response into MLPlan object."""

        try:
            # Clean response (remove markdown formatting if present)
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]

            plan_data = json.loads(response)

            # Validate required fields
            required_fields = [
                "preprocessing_steps",
                "models_to_try",
                "hyperparameter_strategies",
                "evaluation_metrics",
                "reasoning",
                "confidence",
            ]
            for field in required_fields:
                if field not in plan_data:
                    raise ValueError(f"Missing required field: {field}")

            return MLPlan(
                preprocessing_steps=plan_data["preprocessing_steps"],
                models_to_try=plan_data["models_to_try"],
                hyperparameter_strategies=plan_data["hyperparameter_strategies"],
                ensemble_strategy=plan_data.get("ensemble_strategy"),
                evaluation_metrics=plan_data["evaluation_metrics"],
                reasoning=plan_data["reasoning"],
                confidence=plan_data["confidence"],
            )

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Response was: {response}")
            # Fallback to rule-based planning
            return self._fallback_plan(schema, summary)

    def _parse_plan_response_direct(self, response: str) -> MLPlan:
        """Parse LLM response into MLPlan object without fallback."""
        try:
            # Clean response (remove markdown formatting if present)
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]

            plan_data = json.loads(response)

            # Validate required fields
            required_fields = [
                "preprocessing_steps",
                "models_to_try",
                "hyperparameter_strategies",
                "evaluation_metrics",
                "reasoning",
                "confidence",
            ]
            for field in required_fields:
                if field not in plan_data:
                    raise ValueError(f"Missing required field: {field}")

            return MLPlan(
                preprocessing_steps=plan_data["preprocessing_steps"],
                models_to_try=plan_data["models_to_try"],
                hyperparameter_strategies=plan_data["hyperparameter_strategies"],
                ensemble_strategy=plan_data.get("ensemble_strategy"),
                evaluation_metrics=plan_data["evaluation_metrics"],
                reasoning=plan_data["reasoning"],
                confidence=plan_data["confidence"],
            )

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Response was: {response}")
            raise e  # Re-raise the exception since we don't have fallback data

    def _fallback_plan(self, schema: DatasetSchema, summary: dict[str, Any]) -> MLPlan:
        """Create a fallback plan using rule-based logic."""

        logger.info("Using fallback rule-based planning")

        # Determine preprocessing steps
        preprocessing_steps = []

        # Missing value imputation
        if schema.missing_percentage > 0:
            if schema.missing_percentage < 5:
                method = "mean"
            elif schema.missing_percentage < 20:
                method = "median"
            else:
                method = "knn"
            preprocessing_steps.append(
                {
                    "step": "missing_value_imputation",
                    "method": method,
                    "reasoning": f"Missing values present ({schema.missing_percentage:.1f}%), using {method} imputation",
                }
            )

        # Categorical encoding
        if schema.n_categorical > 0:
            if schema.n_categorical < 5:
                method = "onehot"
            else:
                method = "label"
            preprocessing_steps.append(
                {
                    "step": "categorical_encoding",
                    "method": method,
                    "reasoning": f"Categorical features present ({schema.n_categorical}), using {method} encoding",
                }
            )

        # Scaling
        preprocessing_steps.append(
            {
                "step": "scaling",
                "method": "standard",
                "reasoning": "Standard scaling for numerical features",
            }
        )

        # Feature selection
        preprocessing_steps.append(
            {
                "step": "feature_selection",
                "enabled": schema.n_features > 20,
                "reasoning": f"Feature selection {'enabled' if schema.n_features > 20 else 'disabled'} due to feature count ({schema.n_features})",
            }
        )

        # Select models based on dataset characteristics
        models_to_try = []

        if schema.n_rows < 1000:
            # Small dataset - prefer simple models
            models_to_try = ["logistic_regression", "random_forest", "naive_bayes"]
        elif schema.n_rows < 10000:
            # Medium dataset
            models_to_try = [
                "logistic_regression",
                "random_forest",
                "xgboost",
                "gradient_boosting",
            ]
        else:
            # Large dataset
            models_to_try = [
                "random_forest",
                "xgboost",
                "lightgbm",
                "gradient_boosting",
            ]

        # Hyperparameter strategies
        hyperparameter_strategies = {}
        for model in models_to_try:
            if model in ["xgboost", "lightgbm", "gradient_boosting"]:
                hyperparameter_strategies[model] = "bayesian"
            else:
                hyperparameter_strategies[model] = "random"

        # Evaluation metrics
        if schema.target_type == "categorical":
            evaluation_metrics = ["accuracy", "f1", "precision", "recall"]
        else:
            evaluation_metrics = ["mse", "mae", "r2"]

        # Ensemble strategy
        ensemble_strategy = "blending" if len(models_to_try) > 2 else None

        return MLPlan(
            preprocessing_steps=preprocessing_steps,
            models_to_try=models_to_try,
            hyperparameter_strategies=hyperparameter_strategies,
            ensemble_strategy=ensemble_strategy,
            evaluation_metrics=evaluation_metrics,
            reasoning="Rule-based fallback plan based on dataset characteristics",
            confidence=0.7,
        )

    def refine_plan(self, plan: MLPlan, intermediate_results: dict[str, Any]) -> MLPlan:
        """Refine the plan based on intermediate results.

        Args:
            plan: Current plan
            intermediate_results: Results from partial execution

        Returns:
            Refined plan
        """

        # Analyze intermediate results
        context = f"""
Current Plan Performance:
{json.dumps(intermediate_results, indent=2)}

Original Plan Reasoning:
{plan.reasoning}

Please suggest refinements to improve performance. Consider:
1. Adjusting preprocessing steps
2. Adding/removing models
3. Changing hyperparameter strategies
4. Modifying ensemble approach

Provide a JSON response with the same structure as the original plan, but with refinements.
"""

        try:
            response = self.llm_client.generate_response(context)
            # For refinement, we don't have schema/summary, so we'll parse directly
            refined_plan = self._parse_plan_response_direct(response)
            return refined_plan
        except Exception as e:
            logger.error(f"Plan refinement failed: {e}")
            return plan  # Return original plan if refinement fails

    def explain_results(self, results: dict[str, Any], plan: MLPlan) -> str:
        """Generate natural language explanation of results.

        Args:
            results: Model evaluation results
            plan: Original execution plan

        Returns:
            Natural language explanation
        """

        context = f"""
Model Evaluation Results:
{json.dumps(results, indent=2)}

Original Plan:
{json.dumps(plan.__dict__, indent=2)}

Please provide a comprehensive explanation of the results, including:
1. Performance summary
2. Best performing models and why
3. Key insights from the analysis
4. Recommendations for improvement
5. Confidence in the results

Provide a clear, professional explanation suitable for stakeholders.
"""

        try:
            explanation = self.llm_client.generate_response(context)
            return explanation
        except Exception as e:
            logger.error(f"Result explanation failed: {e}")
            return "Unable to generate explanation due to technical issues."

    def get_plan_summary(self, plan: MLPlan) -> str:
        """Get a summary of the execution plan."""

        summary = f"""
ML Execution Plan Summary:
- Preprocessing Steps: {len(plan.preprocessing_steps)}
- Models to Try: {', '.join(plan.models_to_try)}
- Ensemble Strategy: {plan.ensemble_strategy or 'None'}
- Evaluation Metrics: {', '.join(plan.evaluation_metrics)}
- Confidence: {plan.confidence:.2f}

Reasoning: {plan.reasoning}
"""

        return summary
