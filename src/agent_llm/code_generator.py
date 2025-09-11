"""
LLM Code Generation Module

This module provides LLM-powered code generation capabilities for:
- Data preprocessing code generation
- Model training code generation
- Hyperparameter optimization code generation
- Feature engineering code generation
"""

import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from ..utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CodeGenerator:
    """LLM-powered code generator for ML tasks."""

    def __init__(self, llm_client: LLMClient):
        """Initialize the code generator.

        Args:
            llm_client: LLM client for code generation
        """
        self.llm_client = llm_client

    def generate_preprocessing_code(
        self,
        df_info: Dict[str, Any],
        target_column: str,
        preprocessing_requirements: List[str],
    ) -> Dict[str, Any]:
        """Generate preprocessing code based on data characteristics.

        Args:
            df_info: DataFrame information (shape, dtypes, missing values, etc.)
            target_column: Name of the target column
            preprocessing_requirements: List of preprocessing steps needed

        Returns:
            Dictionary containing generated code and execution results
        """
        prompt = f"""
You are an expert data scientist. Generate Python code for data preprocessing based on the following information:

Dataset Information:
- Shape: {df_info.get('shape', 'Unknown')}
- Columns: {df_info.get('columns', [])}
- Data types: {df_info.get('dtypes', {})}
- Missing values: {df_info.get('missing_values', {})}
- Target column: {target_column}
- Target type: {df_info.get('target_type', 'Unknown')}

Preprocessing Requirements:
{chr(10).join(f"- {req}" for req in preprocessing_requirements)}

Generate a complete Python function that:
1. Takes a pandas DataFrame as input
2. Performs all necessary preprocessing steps
3. Returns the preprocessed DataFrame
4. Handles missing values appropriately
5. Encodes categorical variables
6. Scales numerical features
7. Handles outliers if necessary
8. Includes proper error handling

The function should be named `preprocess_data` and should be production-ready.
Include necessary imports at the top.

Return only the Python code, no explanations.
"""

        try:
            result = self.llm_client.generate_code_with_execution(prompt)
            return {
                "code": result["code"],
                "execution_output": result.get("execution_output", ""),
                "execution_success": result.get("execution_success", False),
                "error": result.get("execution_error", ""),
            }
        except Exception as e:
            logger.error(f"Failed to generate preprocessing code: {e}")
            return {
                "code": "",
                "execution_output": "",
                "execution_success": False,
                "error": str(e),
            }

    def generate_feature_engineering_code(
        self,
        df_info: Dict[str, Any],
        target_column: str,
        feature_engineering_ideas: List[str],
    ) -> Dict[str, Any]:
        """Generate feature engineering code.

        Args:
            df_info: DataFrame information
            target_column: Name of the target column
            feature_engineering_ideas: List of feature engineering ideas

        Returns:
            Dictionary containing generated code and execution results
        """
        prompt = f"""
You are an expert feature engineer. Generate Python code for feature engineering based on the following information:

Dataset Information:
- Shape: {df_info.get('shape', 'Unknown')}
- Columns: {df_info.get('columns', [])}
- Data types: {df_info.get('dtypes', {})}
- Target column: {target_column}
- Target type: {df_info.get('target_type', 'Unknown')}

Feature Engineering Ideas:
{chr(10).join(f"- {idea}" for idea in feature_engineering_ideas)}

Generate a complete Python function that:
1. Takes a pandas DataFrame as input
2. Creates new features based on the ideas provided
3. Handles datetime features if present
4. Creates interaction features where appropriate
5. Handles categorical feature combinations
6. Returns the DataFrame with new features
7. Includes proper error handling

The function should be named `engineer_features` and should be production-ready.
Include necessary imports at the top.

Return only the Python code, no explanations.
"""

        try:
            result = self.llm_client.generate_code_with_execution(prompt)
            return {
                "code": result["code"],
                "execution_output": result.get("execution_output", ""),
                "execution_success": result.get("execution_success", False),
                "error": result.get("execution_error", ""),
            }
        except Exception as e:
            logger.error(f"Failed to generate feature engineering code: {e}")
            return {
                "code": "",
                "execution_output": "",
                "execution_success": False,
                "error": str(e),
            }

    def generate_model_training_code(
        self,
        model_name: str,
        hyperparameters: Dict[str, Any],
        training_data_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate model training code.

        Args:
            model_name: Name of the model to train
            hyperparameters: Hyperparameters for the model
            training_data_info: Information about training data

        Returns:
            Dictionary containing generated code and execution results
        """
        prompt = f"""
You are an expert machine learning engineer. Generate Python code for training a {model_name} model.

Model Information:
- Model: {model_name}
- Hyperparameters: {hyperparameters}

Training Data Information:
- Shape: {training_data_info.get('shape', 'Unknown')}
- Features: {training_data_info.get('n_features', 'Unknown')}
- Target type: {training_data_info.get('target_type', 'Unknown')}

Generate a complete Python function that:
1. Takes training features (X) and target (y) as input
2. Creates and configures the {model_name} model with the specified hyperparameters
3. Trains the model
4. Returns the trained model
5. Includes proper error handling and logging
6. Handles both classification and regression tasks appropriately

The function should be named `train_model` and should be production-ready.
Include necessary imports at the top.

Return only the Python code, no explanations.
"""

        try:
            result = self.llm_client.generate_code_with_execution(prompt)
            return {
                "code": result["code"],
                "execution_output": result.get("execution_output", ""),
                "execution_success": result.get("execution_success", False),
                "error": result.get("execution_error", ""),
            }
        except Exception as e:
            logger.error(f"Failed to generate model training code: {e}")
            return {
                "code": "",
                "execution_output": "",
                "execution_success": False,
                "error": str(e),
            }

    def generate_hyperparameter_optimization_code(
        self,
        model_name: str,
        optimization_method: str,
        param_space: Dict[str, Any],
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """Generate hyperparameter optimization code.

        Args:
            model_name: Name of the model to optimize
            optimization_method: Method for optimization (random, bayesian, optuna)
            param_space: Parameter space for optimization
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary containing generated code and execution results
        """
        prompt = f"""
You are an expert in hyperparameter optimization. Generate Python code for optimizing a {model_name} model.

Optimization Information:
- Model: {model_name}
- Method: {optimization_method}
- Parameter space: {param_space}
- CV folds: {cv_folds}

Generate a complete Python function that:
1. Takes training features (X) and target (y) as input
2. Implements {optimization_method} optimization for the {model_name} model
3. Uses the specified parameter space
4. Performs {cv_folds}-fold cross-validation
5. Returns the best model, best parameters, and best score
6. Includes proper error handling and progress tracking
7. Uses appropriate optimization library (Optuna, scikit-optimize, etc.)

The function should be named `optimize_hyperparameters` and should be production-ready.
Include necessary imports at the top.

Return only the Python code, no explanations.
"""

        try:
            result = self.llm_client.generate_code_with_execution(prompt)
            return {
                "code": result["code"],
                "execution_output": result.get("execution_output", ""),
                "execution_success": result.get("execution_success", False),
                "error": result.get("execution_error", ""),
            }
        except Exception as e:
            logger.error(f"Failed to generate hyperparameter optimization code: {e}")
            return {
                "code": "",
                "execution_output": "",
                "execution_success": False,
                "error": str(e),
            }

    def generate_evaluation_code(
        self,
        model_name: str,
        task_type: str,
        metrics: List[str],
    ) -> Dict[str, Any]:
        """Generate model evaluation code.

        Args:
            model_name: Name of the model to evaluate
            task_type: Type of task (classification/regression)
            metrics: List of metrics to calculate

        Returns:
            Dictionary containing generated code and execution results
        """
        prompt = f"""
You are an expert in model evaluation. Generate Python code for evaluating a {model_name} model.

Evaluation Information:
- Model: {model_name}
- Task type: {task_type}
- Metrics: {metrics}

Generate a complete Python function that:
1. Takes a trained model, test features (X_test), and test target (y_test) as input
2. Makes predictions on the test set
3. Calculates all specified metrics: {metrics}
4. Returns a dictionary with all metric values
5. Handles both classification and regression tasks appropriately
6. Includes proper error handling
7. Provides detailed evaluation results

The function should be named `evaluate_model` and should be production-ready.
Include necessary imports at the top.

Return only the Python code, no explanations.
"""

        try:
            result = self.llm_client.generate_code_with_execution(prompt)
            return {
                "code": result["code"],
                "execution_output": result.get("execution_output", ""),
                "execution_success": result.get("execution_success", False),
                "error": result.get("execution_error", ""),
            }
        except Exception as e:
            logger.error(f"Failed to generate evaluation code: {e}")
            return {
                "code": "",
                "execution_output": "",
                "execution_success": False,
                "error": str(e),
            }

    def generate_ensemble_code(
        self,
        models: List[str],
        ensemble_method: str,
        task_type: str,
    ) -> Dict[str, Any]:
        """Generate ensemble model code.

        Args:
            models: List of model names to ensemble
            ensemble_method: Method for ensembling (voting, stacking, blending)
            task_type: Type of task (classification/regression)

        Returns:
            Dictionary containing generated code and execution results
        """
        prompt = f"""
You are an expert in ensemble methods. Generate Python code for creating an ensemble model.

Ensemble Information:
- Models: {models}
- Method: {ensemble_method}
- Task type: {task_type}

Generate a complete Python function that:
1. Takes a list of trained models as input
2. Creates an ensemble using {ensemble_method} method
3. Handles both classification and regression tasks appropriately
4. Returns the ensemble model
5. Includes proper error handling
6. Optimizes ensemble weights if applicable

The function should be named `create_ensemble` and should be production-ready.
Include necessary imports at the top.

Return only the Python code, no explanations.
"""

        try:
            result = self.llm_client.generate_code_with_execution(prompt)
            return {
                "code": result["code"],
                "execution_output": result.get("execution_output", ""),
                "execution_success": result.get("execution_success", False),
                "error": result.get("execution_error", ""),
            }
        except Exception as e:
            logger.error(f"Failed to generate ensemble code: {e}")
            return {
                "code": "",
                "execution_output": "",
                "execution_success": False,
                "error": str(e),
            }

    def execute_generated_code(
        self, code: str, context: Dict[str, Any], timeout: int = 30
    ) -> Dict[str, Any]:
        """Execute generated code with provided context.

        Args:
            code: Generated code to execute
            context: Context variables (DataFrames, models, etc.)
            timeout: Execution timeout in seconds

        Returns:
            Dictionary containing execution results
        """
        try:
            # Prepare execution context
            context_code = self._prepare_execution_context(context)
            
            # Combine context and generated code
            full_code = f"{context_code}\n\n{code}"
            
            # Execute in sandbox
            result = self.llm_client.execute_code_in_sandbox(full_code, timeout)
            
            return {
                "output": result["output"],
                "error": result["error"],
                "success": result["success"],
                "exit_code": result["exit_code"],
            }
        except Exception as e:
            logger.error(f"Failed to execute generated code: {e}")
            return {
                "output": "",
                "error": str(e),
                "success": False,
                "exit_code": -1,
            }

    def _prepare_execution_context(self, context: Dict[str, Any]) -> str:
        """Prepare execution context as Python code."""
        context_code = "import pandas as pd\nimport numpy as np\n"
        context_code += "from sklearn.model_selection import train_test_split\n"
        context_code += "from sklearn.metrics import *\n"
        context_code += "from sklearn.ensemble import *\n"
        context_code += "from sklearn.linear_model import *\n"
        context_code += "from sklearn.tree import *\n"
        context_code += "from sklearn.neighbors import *\n"
        context_code += "from sklearn.svm import *\n"
        context_code += "from sklearn.neural_network import *\n"
        context_code += "from sklearn.preprocessing import *\n"
        context_code += "import joblib\n"
        context_code += "import logging\n\n"

        # Add context variables
        for key, value in context.items():
            if isinstance(value, pd.DataFrame):
                context_code += f"{key} = pd.DataFrame({value.to_dict()})\n"
            elif isinstance(value, (list, dict, str, int, float)):
                context_code += f"{key} = {repr(value)}\n"

        return context_code

    def validate_generated_code(self, code: str) -> Dict[str, Any]:
        """Validate generated code for syntax and basic issues.

        Args:
            code: Generated code to validate

        Returns:
            Dictionary containing validation results
        """
        try:
            # Basic syntax check
            compile(code, "<string>", "exec")
            
            # Check for common issues
            issues = []
            
            # Check for imports
            if "import" not in code:
                issues.append("No imports found")
            
            # Check for function definition
            if "def " not in code:
                issues.append("No function definition found")
            
            # Check for return statement
            if "return" not in code:
                issues.append("No return statement found")
            
            return {
                "valid": True,
                "issues": issues,
                "error": None,
            }
        except SyntaxError as e:
            return {
                "valid": False,
                "issues": [],
                "error": f"Syntax error: {str(e)}",
            }
        except Exception as e:
            return {
                "valid": False,
                "issues": [],
                "error": f"Validation error: {str(e)}",
            }
