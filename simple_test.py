#!/usr/bin/env python3
"""
Simple test script to verify the autonomous ML agent installation
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test basic imports without relative imports."""
    print("🧪 Testing basic imports...")
    
    try:
        # Test individual modules
        import numpy as np
        print("✅ numpy imported")
        
        import pandas as pd
        print("✅ pandas imported")
        
        import sklearn
        print("✅ scikit-learn imported")
        
        import xgboost
        print("✅ xgboost imported")
        
        import lightgbm
        print("✅ lightgbm imported")
        
        import catboost
        print("✅ catboost imported")
        
        import torch
        print("✅ torch imported")
        
        import tensorflow as tf
        print("✅ tensorflow imported")
        
        import optuna
        print("✅ optuna imported")
        
        import openai
        print("✅ openai imported")
        
        import anthropic
        print("✅ anthropic imported")
        
        import google.generativeai as genai
        print("✅ google-generativeai imported")
        
        import e2b
        print("✅ e2b imported")
        
        import fastapi
        print("✅ fastapi imported")
        
        import uvicorn
        print("✅ uvicorn imported")
        
        import streamlit
        print("✅ streamlit imported")
        
        import matplotlib
        print("✅ matplotlib imported")
        
        import seaborn
        print("✅ seaborn imported")
        
        import plotly
        print("✅ plotly imported")
        
        import pydantic
        print("✅ pydantic imported")
        
        import yaml
        print("✅ pyyaml imported")
        
        import dotenv
        print("✅ python-dotenv imported")
        
        import pytest
        print("✅ pytest imported")
        
        import black
        print("✅ black imported")
        
        import flake8
        print("✅ flake8 imported")
        
        import mypy
        print("✅ mypy imported")
        
        import isort
        print("✅ isort imported")
        
        import tqdm
        print("✅ tqdm imported")
        
        import joblib
        print("✅ joblib imported")
        
        import mlflow
        print("✅ mlflow imported")
        
        import wandb
        print("✅ wandb imported")
        
        import requests
        print("✅ requests imported")
        
        import aiohttp
        print("✅ aiohttp imported")
        
        import jupyter
        print("✅ jupyter imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_our_modules():
    """Test our custom modules."""
    print("\n🔧 Testing our custom modules...")
    
    try:
        # Test LLM client
        from utils.llm_client import LLMClient
        print("✅ LLM client imported")
        
        # Test meta-learning
        from core.meta_learning import MetaLearningOptimizer
        print("✅ Meta-learning optimizer imported")
        
        # Test leaderboard
        from evaluation.leaderboard import ModelLeaderboard
        print("✅ Leaderboard imported")
        
        # Test ensemble
        from ensemble.blending import EnsembleBlender
        print("✅ Ensemble blender imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Custom module test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality."""
    print("\n⚙️ Testing basic functionality...")
    
    try:
        # Test LLM client initialization
        from utils.llm_client import LLMClient
        client = LLMClient(primary_provider="openai")
        print("✅ LLM client initialized")
        
        # Test meta-learning optimizer
        from core.meta_learning import MetaLearningOptimizer
        optimizer = MetaLearningOptimizer()
        print("✅ Meta-learning optimizer initialized")
        
        # Test leaderboard
        from evaluation.leaderboard import ModelLeaderboard
        leaderboard = ModelLeaderboard()
        print("✅ Leaderboard initialized")
        
        # Test ensemble blender
        from ensemble.blending import EnsembleBlender
        blender = EnsembleBlender()
        print("✅ Ensemble blender initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🤖 Autonomous ML Agent - Simple Installation Test")
    print("=" * 60)
    
    # Test basic imports
    imports_ok = test_basic_imports()
    
    # Test our modules
    modules_ok = test_our_modules()
    
    # Test basic functionality
    functionality_ok = test_basic_functionality()
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print(f"  Basic Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"  Custom Modules: {'✅ PASS' if modules_ok else '❌ FAIL'}")
    print(f"  Functionality: {'✅ PASS' if functionality_ok else '❌ FAIL'}")
    
    if all([imports_ok, modules_ok, functionality_ok]):
        print("\n🎉 All tests passed! Installation is successful.")
        print("\n📋 Next steps:")
        print("1. Set up your API keys:")
        print("   export OPENAI_API_KEY='your-key-here'")
        print("   export GEMINI_API_KEY='your-key-here'")
        print("   export E2B_API_KEY='your-key-here'")
        print("2. Run the demo: python examples/comprehensive_demo.py")
        print("3. Check README_ENHANCED.md for detailed usage")
        return True
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
