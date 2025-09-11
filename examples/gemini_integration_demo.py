"""
Demo script showing how to use Google Cloud Gemini API with the Autonomous ML Agent
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.gemini_client import GeminiClient, get_gemini_client
from utils.llm_client import LLMClient

# Try to import config, fallback to basic config if not available
try:
    from config.gemini_config import get_gemini_config, get_ml_agent_prompt
except ImportError:
    # Fallback configuration
    def get_gemini_config():
        return {
            "api_key": os.getenv("GOOGLE_API_KEY"),
            "model_name": "gemini-1.5-pro",
            "max_tokens": 4000,
            "temperature": 0.7
        }
    
    def get_ml_agent_prompt(prompt_type):
        prompts = {
            "data_analysis": "You are an expert data scientist. Analyze the provided dataset and provide recommendations.",
            "pipeline_planning": "You are an expert ML engineer. Create a detailed machine learning pipeline plan.",
            "result_explanation": "You are an expert data scientist explaining ML results to stakeholders."
        }
        return prompts.get(prompt_type, "You are an expert AI assistant.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_basic_gemini_usage():
    """Demonstrate basic Gemini API usage."""
    print("🚀 Basic Gemini API Usage Demo")
    print("=" * 50)
    
    try:
        # Initialize Gemini client
        client = get_gemini_client(use_mock=False)
        
        # Simple text generation
        prompt = "Explain machine learning in simple terms for a business audience."
        response = client.generate_response(prompt, max_tokens=500)
        
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print()
        
    except Exception as e:
        logger.error(f"Error in basic demo: {e}")
        print("Using mock client for demo...")
        client = get_gemini_client(use_mock=True)
        response = client.generate_response(prompt)
        print(f"Mock Response: {response}")


def demo_structured_response():
    """Demonstrate structured response generation."""
    print("📊 Structured Response Demo")
    print("=" * 50)
    
    try:
        client = get_gemini_client(use_mock=False)
        
        # Define schema for data analysis
        schema = {
            "type": "object",
            "properties": {
                "data_type": {"type": "string", "enum": ["classification", "regression"]},
                "recommended_models": {"type": "array", "items": {"type": "string"}},
                "preprocessing_steps": {"type": "array", "items": {"type": "string"}},
                "confidence_score": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["data_type", "recommended_models", "confidence_score"]
        }
        
        prompt = """
        Analyze this dataset: 1000 samples, 20 features, target variable is binary (0/1), 
        15% missing values, mix of numerical and categorical features.
        """
        
        response = client.generate_structured_response(prompt, schema)
        
        print(f"Prompt: {prompt}")
        print(f"Structured Response: {response}")
        print()
        
    except Exception as e:
        logger.error(f"Error in structured demo: {e}")
        print("Using mock client for demo...")
        client = get_gemini_client(use_mock=True)
        response = client.generate_structured_response(prompt, schema)
        print(f"Mock Response: {response}")


def demo_ml_agent_integration():
    """Demonstrate ML agent integration with Gemini."""
    print("🤖 ML Agent Integration Demo")
    print("=" * 50)
    
    try:
        # Initialize LLM client with Gemini as primary
        llm_client = LLMClient(primary_provider="gemini")
        
        # Test data analysis
        data_description = """
        Dataset: Customer churn prediction
        - 10,000 customers
        - 15 features (age, income, usage patterns, etc.)
        - Binary target: churned (1) or retained (0)
        - 5% missing values in income column
        - Mix of numerical and categorical features
        """
        
        # Get ML agent prompt
        prompt_template = get_ml_agent_prompt("data_analysis")
        full_prompt = f"{prompt_template}\n\nDataset Description:\n{data_description}"
        
        # Generate analysis
        analysis = llm_client.generate_structured_response(
            full_prompt,
            schema={
                "type": "object",
                "properties": {
                    "data_type": {"type": "string"},
                    "preprocessing_steps": {"type": "array", "items": {"type": "string"}},
                    "recommended_models": {"type": "array", "items": {"type": "string"}},
                    "feature_engineering": {"type": "array", "items": {"type": "string"}},
                    "evaluation_metrics": {"type": "array", "items": {"type": "string"}},
                    "confidence_score": {"type": "number"}
                }
            }
        )
        
        print("Data Analysis Results:")
        print(f"Data Type: {analysis.get('data_type', 'N/A')}")
        print(f"Recommended Models: {analysis.get('recommended_models', [])}")
        print(f"Preprocessing Steps: {analysis.get('preprocessing_steps', [])}")
        print(f"Confidence Score: {analysis.get('confidence_score', 'N/A')}")
        print()
        
    except Exception as e:
        logger.error(f"Error in ML agent demo: {e}")
        print("Demo failed - check your API key configuration")


def demo_pipeline_planning():
    """Demonstrate ML pipeline planning with Gemini."""
    print("🔧 Pipeline Planning Demo")
    print("=" * 50)
    
    try:
        client = get_gemini_client(use_mock=False)
        
        # Mock data analysis results
        data_analysis = {
            "data_type": "classification",
            "target_column": "churned",
            "preprocessing_steps": ["handle_missing_values", "encode_categorical"],
            "model_recommendations": ["RandomForest", "XGBoost", "LogisticRegression"],
            "confidence_score": 0.85
        }
        
        # Generate pipeline plan
        pipeline_plan = client.generate_ml_pipeline_plan(data_analysis)
        
        print("Generated Pipeline Plan:")
        print(f"Pipeline Steps: {pipeline_plan.get('pipeline_steps', [])}")
        print(f"Models to Try: {pipeline_plan.get('models_to_try', [])}")
        print(f"Evaluation Metrics: {pipeline_plan.get('evaluation_metrics', [])}")
        print(f"Expected Runtime: {pipeline_plan.get('expected_runtime', 'N/A')} minutes")
        print()
        
    except Exception as e:
        logger.error(f"Error in pipeline planning demo: {e}")
        print("Using mock client for demo...")
        client = get_gemini_client(use_mock=True)
        pipeline_plan = client.generate_ml_pipeline_plan(data_analysis)
        print(f"Mock Pipeline Plan: {pipeline_plan}")


def demo_result_explanation():
    """Demonstrate model result explanation."""
    print("📈 Result Explanation Demo")
    print("=" * 50)
    
    try:
        client = get_gemini_client(use_mock=False)
        
        # Mock model results
        model_results = {
            "best_model": "XGBoost",
            "accuracy": 0.89,
            "precision": 0.87,
            "recall": 0.91,
            "f1_score": 0.89,
            "feature_importance": {
                "monthly_usage": 0.25,
                "customer_age": 0.20,
                "support_tickets": 0.15,
                "payment_delay": 0.12
            }
        }
        
        # Generate explanation
        explanation = client.explain_model_results(model_results)
        
        print("Model Results Explanation:")
        print(explanation)
        print()
        
    except Exception as e:
        logger.error(f"Error in result explanation demo: {e}")
        print("Using mock client for demo...")
        client = get_gemini_client(use_mock=True)
        explanation = client.explain_model_results(model_results)
        print(f"Mock Explanation: {explanation}")


def demo_available_models():
    """Demonstrate getting available Gemini models."""
    print("🔍 Available Models Demo")
    print("=" * 50)
    
    try:
        client = get_gemini_client(use_mock=False)
        models = client.get_available_models()
        
        print("Available Gemini Models:")
        for model in models:
            print(f"  - {model}")
        print()
        
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        print("Using mock client for demo...")
        client = get_gemini_client(use_mock=True)
        models = client.get_available_models()
        print(f"Mock Available Models: {models}")


def demo_configuration():
    """Demonstrate Gemini configuration."""
    print("⚙️ Configuration Demo")
    print("=" * 50)
    
    config = get_gemini_config()
    print("Current Gemini Configuration:")
    for key, value in config.items():
        if key == "api_key" and value:
            print(f"  {key}: {'*' * 10}...{str(value)[-4:]}")
        else:
            print(f"  {key}: {value}")
    print()


def main():
    """Run all Gemini integration demos."""
    print("🎯 Google Cloud Gemini API Integration Demo")
    print("=" * 60)
    print()
    
    # Check if API key is configured
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️  GOOGLE_API_KEY not found in environment variables.")
        print("   Set your API key: export GOOGLE_API_KEY='your_api_key_here'")
        print("   Or create a .env file with: GOOGLE_API_KEY=your_api_key_here")
        print("   Running demos with mock client...")
        print()
    
    # Run demos
    demo_configuration()
    demo_available_models()
    demo_basic_gemini_usage()
    demo_structured_response()
    demo_ml_agent_integration()
    demo_pipeline_planning()
    demo_result_explanation()
    
    print("✅ All demos completed!")
    print()
    print("Next steps:")
    print("1. Set up your Google Cloud API key")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run the ML agent with Gemini: python -m src.cli --provider gemini")


if __name__ == "__main__":
    main()
