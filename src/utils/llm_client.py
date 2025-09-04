"""
LLM Client for Autonomous Machine Learning Agent

This module provides a unified interface for interacting with various LLM APIs
including OpenAI GPT, Anthropic Claude, and others.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time

import openai
import anthropic
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM client"""
    provider: str = "openai"  # openai, anthropic, local
    model: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 2000
    api_key: Optional[str] = None
    retry_attempts: int = 3
    retry_delay: float = 1.0


class LLMClient:
    """
    Unified client for interacting with various LLM providers
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._setup_client()
    
    def _setup_client(self):
        """Setup the appropriate LLM client based on provider"""
        api_key = self.config.api_key or os.getenv(f"{self.config.provider.upper()}_API_KEY")
        
        if not api_key:
            raise ValueError(f"API key not found for provider: {self.config.provider}")
        
        if self.config.provider == "openai":
            openai.api_key = api_key
            self.client = ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
        elif self.config.provider == "anthropic":
            self.client = ChatAnthropic(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                anthropic_api_key=api_key
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
    
    async def generate_response(self, prompt: str, system_message: Optional[str] = None) -> str:
        """
        Generate a response from the LLM
        
        Args:
            prompt: The user prompt
            system_message: Optional system message for context
            
        Returns:
            Generated response text
        """
        for attempt in range(self.config.retry_attempts):
            try:
                messages = []
                
                if system_message:
                    messages.append(SystemMessage(content=system_message))
                
                messages.append(HumanMessage(content=prompt))
                
                response = await self.client.agenerate([messages])
                return response.generations[0][0].text.strip()
                
            except Exception as e:
                logger.warning(f"LLM request failed (attempt {attempt + 1}): {str(e)}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise
    
    async def generate_code(self, prompt: str, language: str = "python") -> str:
        """
        Generate code from the LLM
        
        Args:
            prompt: The code generation prompt
            language: Programming language for the code
            
        Returns:
            Generated code
        """
        system_message = f"""You are an expert {language} programmer. 
        Generate only valid, production-ready {language} code.
        Do not include explanations or comments unless specifically requested.
        Return only the code."""
        
        code_prompt = f"""
        {prompt}
        
        Generate clean, efficient {language} code that follows best practices.
        """
        
        response = await self.generate_response(code_prompt, system_message)
        
        # Extract code from response (remove markdown if present)
        if "```" in response:
            code_blocks = response.split("```")
            for block in code_blocks:
                if block.strip().startswith(language) or block.strip().startswith("py"):
                    return block.strip()[len(language):].strip()
        
        return response
    
    async def generate_structured_response(self, prompt: str, schema: Dict) -> Dict:
        """
        Generate a structured response following a specific schema
        
        Args:
            prompt: The prompt
            schema: Expected response schema
            
        Returns:
            Structured response as dictionary
        """
        system_message = f"""You are an expert AI assistant. 
        Respond with valid JSON that follows this schema: {json.dumps(schema, indent=2)}
        Ensure all required fields are present and properly formatted."""
        
        json_prompt = f"""
        {prompt}
        
        Respond with valid JSON only.
        """
        
        response = await self.generate_response(json_prompt, system_message)
        
        try:
            # Extract JSON from response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()
            
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response: {response}")
            raise
    
    async def analyze_dataset(self, data_info: Dict) -> Dict:
        """
        Analyze dataset characteristics using LLM
        
        Args:
            data_info: Dictionary containing dataset information
            
        Returns:
            Analysis results
        """
        prompt = f"""
        Analyze this dataset and provide insights:
        
        Dataset Information:
        - Shape: {data_info.get('shape', 'Unknown')}
        - Target column: {data_info.get('target_column', 'Unknown')}
        - Target type: {data_info.get('target_type', 'Unknown')}
        - Target distribution: {data_info.get('target_distribution', 'Unknown')}
        - Feature types: {data_info.get('feature_types', 'Unknown')}
        - Missing values: {data_info.get('missing_values', 'Unknown')}
        
        Provide analysis in the following JSON format:
        {{
            "data_quality_score": float,
            "preprocessing_needs": [list of preprocessing steps],
            "feature_engineering_opportunities": [list of opportunities],
            "recommended_models": [list of model names],
            "potential_issues": [list of issues],
            "optimization_suggestions": [list of suggestions]
        }}
        """
        
        return await self.generate_structured_response(prompt, {
            "data_quality_score": "float",
            "preprocessing_needs": ["list"],
            "feature_engineering_opportunities": ["list"],
            "recommended_models": ["list"],
            "potential_issues": ["list"],
            "optimization_suggestions": ["list"]
        })
    
    async def generate_preprocessing_code(self, data_info: Dict) -> str:
        """
        Generate preprocessing code based on dataset characteristics
        
        Args:
            data_info: Dictionary containing dataset information
            
        Returns:
            Generated preprocessing code
        """
        prompt = f"""
        Generate Python code for preprocessing this dataset:
        
        Dataset characteristics:
        - Shape: {data_info.get('shape', 'Unknown')}
        - Columns: {data_info.get('columns', 'Unknown')}
        - Target: {data_info.get('target_column', 'Unknown')}
        - Feature types: {data_info.get('feature_types', 'Unknown')}
        - Missing values: {data_info.get('missing_values', 'Unknown')}
        
        Generate preprocessing code that handles:
        1. Missing value imputation
        2. Categorical encoding
        3. Feature scaling/normalization
        4. Outlier detection and handling
        5. Feature engineering
        6. Data validation
        
        The code should:
        - Be production-ready and efficient
        - Handle both training and inference scenarios
        - Include proper error handling
        - Return a preprocessing pipeline that can be saved and reused
        """
        
        return await self.generate_code(prompt, "python")
    
    async def select_models(self, data_info: Dict, optimization_metric: str) -> List[str]:
        """
        Select appropriate models based on dataset characteristics
        
        Args:
            data_info: Dictionary containing dataset information
            optimization_metric: Target optimization metric
            
        Returns:
            List of selected model names
        """
        prompt = f"""
        Select the best machine learning models for this dataset:
        
        Dataset characteristics:
        - Shape: {data_info.get('shape', 'Unknown')}
        - Target type: {data_info.get('target_type', 'Unknown')}
        - Target distribution: {data_info.get('target_distribution', 'Unknown')}
        - Feature types: {data_info.get('feature_types', 'Unknown')}
        - Optimization metric: {optimization_metric}
        
        Available models: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, kNN, MLP
        
        Select the top 5 models and provide reasoning in JSON format:
        {{
            "selected_models": [list of model names],
            "reasoning": {{
                "model_name": "reasoning for selection"
            }},
            "expected_performance": {{
                "model_name": "expected performance characteristics"
            }}
        }}
        """
        
        response = await self.generate_structured_response(prompt, {
            "selected_models": ["list"],
            "reasoning": {"object"},
            "expected_performance": {"object"}
        })
        
        return response.get("selected_models", [])
    
    async def optimize_hyperparameters(self, model_name: str, data_info: Dict, time_budget: int) -> Dict:
        """
        Generate hyperparameter optimization strategy
        
        Args:
            model_name: Name of the model
            data_info: Dictionary containing dataset information
            time_budget: Time budget in seconds
            
        Returns:
            Optimization strategy
        """
        prompt = f"""
        Design a hyperparameter optimization strategy for {model_name}:
        
        Dataset characteristics:
        - Shape: {data_info.get('shape', 'Unknown')}
        - Target type: {data_info.get('target_type', 'Unknown')}
        - Time budget: {time_budget} seconds
        
        Provide optimization strategy in JSON format:
        {{
            "search_method": "random|bayesian|grid",
            "max_iterations": int,
            "parameter_ranges": {{
                "param_name": [min, max] or [list of values]
            }},
            "early_stopping": boolean,
            "cv_folds": int,
            "optimization_metric": string
        }}
        """
        
        return await self.generate_structured_response(prompt, {
            "search_method": "string",
            "max_iterations": "integer",
            "parameter_ranges": {"object"},
            "early_stopping": "boolean",
            "cv_folds": "integer",
            "optimization_metric": "string"
        })
    
    async def generate_ensemble_strategy(self, models: List[str], performances: List[float]) -> Dict:
        """
        Generate ensemble strategy for top models
        
        Args:
            models: List of model names
            performances: List of model performances
            
        Returns:
            Ensemble strategy
        """
        prompt = f"""
        Design an ensemble strategy for these models:
        
        Models and performances:
        {dict(zip(models, performances))}
        
        Provide ensemble strategy in JSON format:
        {{
            "ensemble_method": "stacking|blending|voting",
            "base_models": [list of model names],
            "meta_model": "model name for meta-learner",
            "weights": [list of weights for each model],
            "cross_validation": boolean,
            "expected_improvement": float
        }}
        """
        
        return await self.generate_structured_response(prompt, {
            "ensemble_method": "string",
            "base_models": ["list"],
            "meta_model": "string",
            "weights": ["list"],
            "cross_validation": "boolean",
            "expected_improvement": "float"
        })
    
    async def interpret_model(self, model_name: str, feature_importance: Dict, performance: float) -> str:
        """
        Generate model interpretation and explanation
        
        Args:
            model_name: Name of the model
            feature_importance: Dictionary of feature importance scores
            performance: Model performance score
            
        Returns:
            Model interpretation text
        """
        prompt = f"""
        Provide a comprehensive interpretation of this model:
        
        Model: {model_name}
        Performance: {performance}
        Feature Importance: {feature_importance}
        
        Provide a natural language explanation covering:
        1. How the model makes decisions
        2. Which features are most important and why
        3. Model strengths and limitations
        4. Potential improvements
        5. Business implications
        """
        
        return await self.generate_response(prompt)
    
    async def generate_insights(self, results: Dict) -> str:
        """
        Generate insights from ML pipeline results
        
        Args:
            results: Dictionary containing pipeline results
            
        Returns:
            Generated insights
        """
        prompt = f"""
        Generate actionable insights from this ML pipeline:
        
        Results summary:
        - Total models trained: {results.get('total_models', 'Unknown')}
        - Best model: {results.get('best_model', 'Unknown')}
        - Best score: {results.get('best_score', 'Unknown')}
        - Training time: {results.get('training_time', 'Unknown')}
        - Feature importance: {results.get('feature_importance', 'Unknown')}
        
        Provide insights about:
        1. Model performance patterns
        2. Feature importance insights
        3. Recommendations for improvement
        4. Deployment considerations
        5. Next steps for optimization
        """
        
        return await self.generate_response(prompt)


class MockLLMClient:
    """
    Mock LLM client for testing and development
    """
    
    def __init__(self, responses: Optional[Dict] = None):
        self.responses = responses or {}
        self.call_count = 0
    
    async def generate_response(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Mock response generation"""
        self.call_count += 1
        
        if "preprocessing" in prompt.lower():
            return self._get_mock_preprocessing_code()
        elif "model selection" in prompt.lower():
            return "Selected models: Logistic Regression, Random Forest, XGBoost"
        elif "optimization" in prompt.lower():
            return "Use Bayesian optimization with 50 iterations"
        elif "ensemble" in prompt.lower():
            return "Use stacking with Random Forest as meta-learner"
        else:
            return "Mock LLM response"
    
    async def generate_code(self, prompt: str, language: str = "python") -> str:
        """Mock code generation"""
        return self._get_mock_preprocessing_code()
    
    def _get_mock_preprocessing_code(self) -> str:
        """Get mock preprocessing code"""
        return '''
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

def preprocess_data(df, target_column):
    # Handle missing values
    numeric_imputer = SimpleImputer(strategy='mean')
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Impute missing values
    df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
    df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in categorical_cols:
        if col != target_column:
            df[col] = le.fit_transform(df[col])
    
    # Scale numeric features
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df
'''
