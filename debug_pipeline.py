#!/usr/bin/env python3
"""Debug script to test the ML pipeline step by step."""

import sys
import os
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from src.core.orchestrator import AutonomousMLAgent, PipelineConfig
from src.core.ingest import analyze_data
from src.utils.llm_client import LLMClient

def test_pipeline():
    print("🔍 Testing ML Pipeline Step by Step")
    print("=" * 50)
    
    # Step 1: Test data analysis
    print("\n1. Testing data analysis...")
    try:
        df, schema, summary = analyze_data("examples/sample_dataset.csv", "target")
        print(f"✅ Data analysis successful: {df.shape}")
        print(f"   Target type: {schema.target_type}")
        print(f"   Features: {schema.n_features}")
    except Exception as e:
        print(f"❌ Data analysis failed: {e}")
        return
    
    # Step 2: Test LLM client
    print("\n2. Testing LLM client...")
    try:
        llm_client = LLMClient()
        print(f"✅ LLM client created: {llm_client is not None}")
    except Exception as e:
        print(f"❌ LLM client failed: {e}")
        llm_client = None
    
    # Step 3: Test pipeline config
    print("\n3. Testing pipeline config...")
    try:
        config = PipelineConfig(
            time_budget=60,
            optimization_metric="accuracy",
            random_state=42,
            output_dir="./results",
            save_models=True,
            save_results=True,
            verbose=True
        )
        print(f"✅ Pipeline config created")
    except Exception as e:
        print(f"❌ Pipeline config failed: {e}")
        return
    
    # Step 4: Test agent creation
    print("\n4. Testing agent creation...")
    try:
        agent = AutonomousMLAgent(config, llm_client)
        print(f"✅ Agent created successfully")
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return
    
    # Step 5: Test execution plan creation
    print("\n5. Testing execution plan creation...")
    try:
        execution_plan = agent._create_execution_plan(schema, summary)
        print(f"✅ Execution plan created")
        print(f"   Models to try: {execution_plan.models_to_try}")
        print(f"   Preprocessing steps: {len(execution_plan.preprocessing_steps)}")
    except Exception as e:
        print(f"❌ Execution plan creation failed: {e}")
        return
    
    # Step 6: Test data preprocessing
    print("\n6. Testing data preprocessing...")
    try:
        X_processed, y_processed, preprocessor = agent._preprocess_data(df, "target", execution_plan)
        print(f"✅ Data preprocessing successful")
        print(f"   X shape: {X_processed.shape}")
        print(f"   y shape: {y_processed.shape}")
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        return
    
    # Step 7: Test model training (just one model)
    print("\n7. Testing model training...")
    try:
        from src.core.model_zoo import ModelZoo
        from src.models.hyperopt import HyperparameterOptimizer
        
        model_zoo = ModelZoo()
        model_name = execution_plan.models_to_try[0]
        print(f"   Testing model: {model_name}")
        
        is_classification = model_name in model_zoo.list_models(is_classification=True)
        model = model_zoo.get_model(model_name, is_classification)
        print(f"   Model created: {type(model)}")
        
        param_grid = agent._get_default_param_grid(model_name)
        print(f"   Parameter grid: {param_grid}")
        
        optimizer = HyperparameterOptimizer(
            model=model.model,
            param_grid=param_grid,
            cv=3,  # Reduced for testing
            scoring="accuracy",
            n_jobs=1,  # Reduced for testing
        )
        
        X_array = X_processed.values if hasattr(X_processed, 'values') else X_processed
        y_array = y_processed.values if hasattr(y_processed, 'values') else y_processed
        
        print(f"   X array shape: {X_array.shape}")
        print(f"   y array shape: {y_array.shape}")
        
        best_model, best_score, best_params = optimizer.optimize(X_array, y_array, method="random")
        print(f"✅ Model training successful!")
        print(f"   Best score: {best_score:.4f}")
        print(f"   Best params: {best_params}")
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n🎉 All tests passed! The pipeline should work.")

if __name__ == "__main__":
    test_pipeline()
