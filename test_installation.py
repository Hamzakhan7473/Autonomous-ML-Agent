#!/usr/bin/env python3
"""
Test script to verify the autonomous ML agent installation
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing module imports...")
    
    try:
        # Test core modules
        from core.orchestrator import AutonomousMLAgent, PipelineConfig
        print("✅ Core orchestrator imported")
        
        from core.meta_learning import MetaLearningOptimizer
        print("✅ Meta-learning module imported")
        
        # Test LLM modules
        from utils.llm_client import LLMClient
        print("✅ LLM client imported")
        
        from agent_llm.code_generator import CodeGenerator
        print("✅ Code generator imported")
        
        # Test evaluation modules
        from evaluation.leaderboard import ModelLeaderboard
        print("✅ Leaderboard module imported")
        
        # Test ensemble modules
        from ensemble.blending import EnsembleBlender
        print("✅ Ensemble module imported")
        
        # Test service modules
        from service.model_service import ModelService
        print("✅ Model service imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without API calls."""
    print("\n🔧 Testing basic functionality...")
    
    try:
        # Test LLM client initialization (without API calls)
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

def test_dependencies():
    """Test that all required dependencies are available."""
    print("\n📚 Testing dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'xgboost', 'lightgbm', 'catboost',
        'torch', 'tensorflow', 'optuna', 'hyperopt', 'scikit-optimize',
        'shap', 'lime', 'eli5', 'openai', 'anthropic', 'google.generativeai',
        'e2b', 'fastapi', 'uvicorn', 'streamlit', 'flask', 'matplotlib',
        'seaborn', 'plotly', 'pydantic', 'pyyaml', 'python-dotenv',
        'pytest', 'black', 'flake8', 'mypy', 'isort', 'tqdm', 'joblib',
        'mlflow', 'wandb', 'requests', 'aiohttp', 'jupyter'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {missing_packages}")
        return False
    else:
        print("\n✅ All required packages are available")
        return True

def main():
    """Run all tests."""
    print("🤖 Autonomous ML Agent - Installation Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test basic functionality
    functionality_ok = test_basic_functionality()
    
    # Test dependencies
    dependencies_ok = test_dependencies()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"  Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"  Functionality: {'✅ PASS' if functionality_ok else '❌ FAIL'}")
    print(f"  Dependencies: {'✅ PASS' if dependencies_ok else '❌ FAIL'}")
    
    if all([imports_ok, functionality_ok, dependencies_ok]):
        print("\n🎉 All tests passed! Installation is successful.")
        print("\n📋 Next steps:")
        print("1. Set up your API keys in .env file")
        print("2. Run the demo: python examples/comprehensive_demo.py")
        print("3. Check README_ENHANCED.md for detailed usage")
        return True
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
