#!/usr/bin/env python3
"""
API Keys Setup Script for Autonomous ML Agent
This script helps you set up your API keys securely.
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create .env file with API key placeholders."""
    env_file = Path(".env")
    
    if env_file.exists():
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ").lower()
        if response != 'y':
            print("Keeping existing .env file.")
            return
    
    env_content = """# Google Cloud Gemini API
GOOGLE_API_KEY=your_google_api_key_here

# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# MLflow Configuration
MLFLOW_TRACKING_URI=sqlite:///meta/mlflow.db
MLFLOW_EXPERIMENT_NAME=autonomous_ml_agent

# Database Configuration
DATABASE_URL=sqlite:///meta/agent.db

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here

# Model Configuration
DEFAULT_MODEL_PROVIDER=gemini
DEFAULT_MODEL_NAME=gemini-1.5-pro

# Performance
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=300

# Data Processing
MAX_FILE_SIZE_MB=100
SUPPORTED_FILE_TYPES=csv,xlsx,json,parquet
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f"✅ Created .env file at {env_file.absolute()}")
    print("📝 Please edit the .env file and replace the placeholder values with your actual API keys.")

def check_api_keys():
    """Check if API keys are properly configured."""
    print("🔍 Checking API key configuration...")
    
    keys_to_check = [
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY", 
        "ANTHROPIC_API_KEY"
    ]
    
    all_configured = True
    
    for key in keys_to_check:
        value = os.getenv(key)
        if value and value != f"your_{key.lower()}_here":
            print(f"✅ {key}: Configured")
        else:
            print(f"❌ {key}: Not configured")
            all_configured = False
    
    if all_configured:
        print("\n🎉 All API keys are configured!")
        return True
    else:
        print("\n⚠️  Some API keys are missing or not configured.")
        print("Please edit your .env file and add your actual API keys.")
        return False

def test_connections():
    """Test API connections."""
    print("\n🧪 Testing API connections...")
    
    try:
        # Test Gemini
        from src.utils.llm_client import LLMClient
        
        print("Testing Gemini...")
        llm_client = LLMClient(primary_provider="gemini")
        connection_results = llm_client.test_connection("gemini")
        
        if connection_results.get("gemini", False):
            print("✅ Gemini: Connection successful")
        else:
            print("❌ Gemini: Connection failed")
            
    except Exception as e:
        print(f"❌ Error testing connections: {e}")

def main():
    """Main setup function."""
    print("🚀 Autonomous ML Agent - API Keys Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Please run this script from the project root directory.")
        sys.exit(1)
    
    # Create .env file
    create_env_file()
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check configuration
    if check_api_keys():
        test_connections()
    else:
        print("\n📋 Next steps:")
        print("1. Edit the .env file with your actual API keys")
        print("2. Run this script again to test connections")
        print("3. Or run: python -m src.cli test-llm --provider gemini")

if __name__ == "__main__":
    main()

