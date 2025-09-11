#!/bin/bash

# Autonomous ML Agent Environment Setup Script

echo "🤖 Setting up Autonomous ML Agent Environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p meta models results data/raw data/artifacts
touch data/raw/.gitkeep data/artifacts/.gitkeep meta/.gitkeep models/.gitkeep results/.gitkeep

# Set up environment variables template
echo "🔑 Creating environment template..."
cat > .env.template << EOF
# LLM API Configuration
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GEMINI_API_KEY=your_gemini_key_here

# Code Execution
E2B_API_KEY=your_e2b_key_here

# Model Registry
MODEL_REGISTRY_PATH=./models
META_LEARNING_DB=./meta/runs.db

# Compute Resources
MAX_MEMORY_GB=16
MAX_CPU_CORES=8
GPU_ENABLED=true
EOF

echo "✅ Environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Copy .env.template to .env and fill in your API keys"
echo "2. Run the demo: python examples/comprehensive_demo.py"
echo "3. Check the README_ENHANCED.md for detailed usage instructions"
echo ""
echo "🚀 Happy ML-ing!"
