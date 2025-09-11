"""
Comprehensive Demo of Autonomous ML Agent

This demo showcases the complete autonomous ML pipeline using the Iris dataset,
demonstrating all the key features including:
- LLM orchestration with code generation
- Meta-learning warm starts
- Advanced preprocessing
- Model training and optimization
- Ensemble methods
- Leaderboard and insights
- Model export and deployment
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from core.orchestrator import AutonomousMLAgent, PipelineConfig
from core.meta_learning import MetaLearningOptimizer
from agent_llm.code_generator import CodeGenerator
from evaluation.leaderboard import ModelLeaderboard, ModelResult
from utils.llm_client import LLMClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    # Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target_names[iris.target]
    
    # Add some missing values and noise to make it more realistic
    np.random.seed(42)
    
    # Add missing values
    missing_indices = np.random.choice(df.index, size=10, replace=False)
    df.loc[missing_indices, 'sepal length (cm)'] = np.nan
    
    # Add some outliers
    outlier_indices = np.random.choice(df.index, size=5, replace=False)
    df.loc[outlier_indices, 'petal width (cm)'] *= 2
    
    # Add some categorical features
    df['sepal_size'] = pd.cut(df['sepal length (cm)'], bins=3, labels=['small', 'medium', 'large'])
    df['petal_size'] = pd.cut(df['petal length (cm)'], bins=3, labels=['small', 'medium', 'large'])
    
    return df


async def demo_llm_code_generation():
    """Demonstrate LLM code generation capabilities."""
    print("\n" + "="*60)
    print("DEMO: LLM Code Generation")
    print("="*60)
    
    # Initialize LLM client
    llm_client = LLMClient(primary_provider="openai")
    code_generator = CodeGenerator(llm_client)
    
    # Create sample data info
    df = create_sample_dataset()
    df_info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'target_type': 'categorical'
    }
    
    # Generate preprocessing code
    print("\nGenerating preprocessing code...")
    preprocessing_requirements = [
        "Handle missing values",
        "Encode categorical variables", 
        "Scale numerical features",
        "Detect and handle outliers"
    ]
    
    result = code_generator.generate_preprocessing_code(
        df_info, 'species', preprocessing_requirements
    )
    
    print(f"Generated Code:\n{result['code']}")
    print(f"Execution Success: {result['execution_success']}")
    if result['execution_output']:
        print(f"Execution Output:\n{result['execution_output']}")
    
    # Generate feature engineering code
    print("\nGenerating feature engineering code...")
    feature_ideas = [
        "Create interaction features",
        "Add polynomial features",
        "Create ratio features"
    ]
    
    result = code_generator.generate_feature_engineering_code(
        df_info, 'species', feature_ideas
    )
    
    print(f"Generated Code:\n{result['code']}")
    print(f"Execution Success: {result['execution_success']}")


def demo_meta_learning():
    """Demonstrate meta-learning capabilities."""
    print("\n" + "="*60)
    print("DEMO: Meta-Learning Warm Starts")
    print("="*60)
    
    # Initialize meta-learning optimizer
    meta_optimizer = MetaLearningOptimizer()
    
    # Simulate some previous runs
    print("\nSimulating previous runs...")
    previous_runs = [
        {
            'run_id': 'run_001',
            'dataset_name': 'iris_similar_1',
            'dataset_size': 150,
            'n_features': 4,
            'n_categorical': 0,
            'n_numerical': 4,
            'missing_percentage': 5.0,
            'target_type': 'categorical',
            'model_name': 'random_forest',
            'best_params': {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 2},
            'best_score': 0.95,
            'optimization_method': 'random',
            'training_time': 2.5,
            'timestamp': '2024-01-01T10:00:00'
        },
        {
            'run_id': 'run_002', 
            'dataset_name': 'iris_similar_2',
            'dataset_size': 200,
            'n_features': 5,
            'n_categorical': 1,
            'n_numerical': 4,
            'missing_percentage': 3.0,
            'target_type': 'categorical',
            'model_name': 'random_forest',
            'best_params': {'n_estimators': 150, 'max_depth': 7, 'min_samples_split': 3},
            'best_score': 0.92,
            'optimization_method': 'bayesian',
            'training_time': 3.2,
            'timestamp': '2024-01-02T10:00:00'
        }
    ]
    
    # Store previous runs
    for run_data in previous_runs:
        meta_optimizer.store_run_result(**run_data)
    
    # Get warm start parameters for current dataset
    current_dataset_metadata = {
        'dataset_size': 150,
        'n_features': 6,
        'n_categorical': 2,
        'n_numerical': 4,
        'missing_percentage': 6.7,
        'target_type': 'categorical'
    }
    
    param_space = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 2, 4]
    }
    
    print("\nGetting warm start parameters...")
    warm_start_params = meta_optimizer.get_warm_start_params(
        'random_forest', current_dataset_metadata, param_space, n_suggestions=3
    )
    
    print("Warm start parameter suggestions:")
    for i, params in enumerate(warm_start_params):
        print(f"  Suggestion {i+1}: {params}")
    
    # Get performance statistics
    stats = meta_optimizer.get_model_performance_stats('random_forest')
    print(f"\nModel performance statistics: {stats}")


def demo_ensemble_methods():
    """Demonstrate ensemble methods."""
    print("\n" + "="*60)
    print("DEMO: Advanced Ensemble Methods")
    print("="*60)
    
    from ensemble.blending import EnsembleBlender, BlendingConfig
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    # Create sample data
    df = create_sample_dataset()
    X = df.drop('species', axis=1)
    y = df['species']
    
    # Encode target
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Handle categorical features
    X_processed = pd.get_dummies(X, columns=['sepal_size', 'petal_size'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train base models
    print("\nTraining base models...")
    models = []
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models.append(rf)
    print(f"Random Forest accuracy: {rf.score(X_test, y_test):.4f}")
    
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    models.append(gb)
    print(f"Gradient Boosting accuracy: {gb.score(X_test, y_test):.4f}")
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    models.append(lr)
    print(f"Logistic Regression accuracy: {lr.score(X_test, y_test):.4f}")
    
    # Test different ensemble methods
    print("\nTesting ensemble methods...")
    
    # Weighted blending
    config = BlendingConfig(method="weighted", weights=[0.4, 0.4, 0.2])
    blender = EnsembleBlender(config)
    weighted_ensemble = blender.blend_models(models, X_train, y_train)
    weighted_score = weighted_ensemble.predict(X_test)
    weighted_accuracy = np.mean(weighted_score == y_test)
    print(f"Weighted Ensemble accuracy: {weighted_accuracy:.4f}")
    
    # Stacking
    config = BlendingConfig(method="stacking", meta_model="linear")
    blender = EnsembleBlender(config)
    stacking_ensemble = blender.blend_models(models, X_train, y_train)
    stacking_score = stacking_ensemble.predict(X_test)
    stacking_accuracy = np.mean(stacking_score == y_test)
    print(f"Stacking Ensemble accuracy: {stacking_accuracy:.4f}")
    
    # Voting
    config = BlendingConfig(method="voting")
    blender = EnsembleBlender(config)
    voting_ensemble = blender.blend_models(models, X_train, y_train)
    voting_score = voting_ensemble.predict(X_test)
    voting_accuracy = np.mean(voting_score == y_test)
    print(f"Voting Ensemble accuracy: {voting_accuracy:.4f}")


def demo_leaderboard():
    """Demonstrate leaderboard functionality."""
    print("\n" + "="*60)
    print("DEMO: Model Leaderboard")
    print("="*60)
    
    # Create sample results
    results = [
        ModelResult(
            model_name="Random Forest",
            model_type="ensemble",
            best_score=0.95,
            best_params={"n_estimators": 100, "max_depth": 5},
            metrics={"accuracy": 0.95, "f1": 0.94, "precision": 0.93, "recall": 0.95},
            training_time=2.5,
            feature_importance={"petal length (cm)": 0.4, "petal width (cm)": 0.3, "sepal length (cm)": 0.2, "sepal width (cm)": 0.1},
            cross_validation_scores=[0.94, 0.96, 0.95, 0.94, 0.95],
            model_size_mb=0.5,
            inference_time_ms=1.2
        ),
        ModelResult(
            model_name="Gradient Boosting",
            model_type="ensemble", 
            best_score=0.93,
            best_params={"n_estimators": 100, "learning_rate": 0.1},
            metrics={"accuracy": 0.93, "f1": 0.92, "precision": 0.91, "recall": 0.93},
            training_time=3.2,
            feature_importance={"petal length (cm)": 0.45, "petal width (cm)": 0.25, "sepal length (cm)": 0.2, "sepal width (cm)": 0.1},
            cross_validation_scores=[0.92, 0.94, 0.93, 0.92, 0.94],
            model_size_mb=0.3,
            inference_time_ms=0.8
        ),
        ModelResult(
            model_name="Logistic Regression",
            model_type="linear",
            best_score=0.90,
            best_params={"C": 1.0, "penalty": "l2"},
            metrics={"accuracy": 0.90, "f1": 0.89, "precision": 0.88, "recall": 0.90},
            training_time=0.5,
            feature_importance={"petal length (cm)": 0.5, "petal width (cm)": 0.3, "sepal length (cm)": 0.15, "sepal width (cm)": 0.05},
            cross_validation_scores=[0.89, 0.91, 0.90, 0.89, 0.91],
            model_size_mb=0.1,
            inference_time_ms=0.3
        )
    ]
    
    # Create leaderboard
    leaderboard = ModelLeaderboard()
    for result in results:
        leaderboard.add_result(result)
    
    # Print CLI leaderboard
    leaderboard.print_cli_leaderboard()
    
    # Generate insights
    print("\nGenerating model insights...")
    insights = leaderboard.get_model_insights()
    print(f"\nInsights:\n{insights}")
    
    # Export to JSON
    output_path = Path("demo_results")
    output_path.mkdir(exist_ok=True)
    leaderboard.export_to_json(str(output_path / "leaderboard.json"))
    print(f"\nLeaderboard exported to {output_path / 'leaderboard.json'}")


async def demo_complete_pipeline():
    """Demonstrate the complete autonomous ML pipeline."""
    print("\n" + "="*60)
    print("DEMO: Complete Autonomous ML Pipeline")
    print("="*60)
    
    # Create sample dataset
    df = create_sample_dataset()
    
    # Save dataset
    output_path = Path("demo_results")
    output_path.mkdir(exist_ok=True)
    df.to_csv(output_path / "iris_demo.csv", index=False)
    
    # Configure pipeline
    config = PipelineConfig(
        time_budget=300,  # 5 minutes
        optimization_metric="accuracy",
        random_state=42,
        output_dir=str(output_path),
        save_models=True,
        save_results=True,
        verbose=True
    )
    
    # Initialize agent
    llm_client = LLMClient(primary_provider="openai")
    agent = AutonomousMLAgent(config, llm_client)
    
    print("\nStarting autonomous ML pipeline...")
    start_time = time.time()
    
    try:
        # Run the pipeline
        results = agent.run(
            dataset_path=str(output_path / "iris_demo.csv"),
            target_column="species",
            config=config
        )
        
        execution_time = time.time() - start_time
        
        print(f"\nPipeline completed in {execution_time:.2f} seconds")
        print(f"Best model: {results.best_model}")
        print(f"Best score: {results.best_score:.4f}")
        print(f"Best parameters: {results.best_params}")
        print(f"Execution time: {results.execution_time:.2f} seconds")
        
        # Print insights
        print(f"\nModel insights:\n{results.model_insights}")
        
        # Test prediction
        print("\nTesting prediction on new data...")
        sample_data = df.drop('species', axis=1).head(5)
        predictions = agent.predict(sample_data)
        print(f"Sample predictions: {predictions}")
        
        if hasattr(agent.results.best_model, 'predict_proba'):
            probabilities = agent.predict_proba(sample_data)
            print(f"Sample probabilities:\n{probabilities}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        logger.error(f"Pipeline execution failed: {e}")


async def main():
    """Run all demos."""
    print("🤖 Autonomous Machine Learning Agent - Comprehensive Demo")
    print("=" * 80)
    
    try:
        # Demo 1: LLM Code Generation
        await demo_llm_code_generation()
        
        # Demo 2: Meta-Learning
        demo_meta_learning()
        
        # Demo 3: Ensemble Methods
        demo_ensemble_methods()
        
        # Demo 4: Leaderboard
        demo_leaderboard()
        
        # Demo 5: Complete Pipeline
        await demo_complete_pipeline()
        
        print("\n" + "="*80)
        print("✅ All demos completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.error(f"Demo execution failed: {e}")


if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Some demos may not work properly.")
        print("   Set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
    
    # Run demos
    asyncio.run(main())
