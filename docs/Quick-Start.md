# 🚀 Quick Start Guide

Get up and running with the Autonomous ML Agent in 5 minutes!

## **Prerequisites**

- **Python 3.10+**
- **Node.js 18+**
- **Git**
- **LLM API Key** (E2B recommended)

## **Step 1: Clone the Repository**

```bash
git clone https://github.com/Hamzakhan7473/Autonomous-ML-Agent.git
cd Autonomous-ML-Agent
```

## **Step 2: Install Dependencies**

### **Backend Dependencies**
```bash
# Install Python dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### **Frontend Dependencies**
```bash
cd frontend
npm install
cd ..
```

## **Step 3: Configure API Keys**

### **Get E2B API Key**
1. Visit [E2B](https://e2b.dev/)
2. Sign up and create an API key
3. Copy your API key

### **Setup Environment**
```bash
# Copy environment template
cp env.example .env

# Edit .env file and add your API key
echo "E2B_API_KEY=your_e2b_api_key_here" >> .env
echo "DEFAULT_LLM_PROVIDER=e2b" >> .env
```

## **Step 4: Start the Services**

### **Terminal 1: Start Backend**
```bash
python3 -m uvicorn src.service.app:app --host 0.0.0.0 --port 8000 --reload
```

### **Terminal 2: Start Frontend**
```bash
cd frontend
npm run dev
```

## **Step 5: Access the Application**

- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## **Step 6: Run Your First Pipeline**

### **Using the Web Interface**

1. **Open the Application**: Navigate to http://localhost:3000
2. **Upload Dataset**: Drag and drop a CSV file or click to browse
3. **Configure Pipeline**: Set time budget and optimization metric
4. **Start Pipeline**: Click "Start Pipeline" and monitor progress
5. **View Results**: See model performance and insights

### **Using the API**

```bash
# Upload dataset
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_dataset.csv"

# Run pipeline
curl -X POST "http://localhost:8000/pipeline/run" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "data/raw/your_dataset.csv",
    "target_column": "",
    "time_budget": 1800
  }'

# Check status
curl -X GET "http://localhost:8000/pipeline/status/task_id_here"
```

### **Using Python Script**

```python
from src import AutonomousMLAgent, PipelineConfig

# Initialize agent
agent = AutonomousMLAgent(
    config=PipelineConfig(
        time_budget=1800,  # 30 minutes
        optimization_metric="accuracy"
    )
)

# Run pipeline
results = agent.run(
    dataset_path="data/raw/your_dataset.csv",
    target_column=""  # Auto-detect target column
)

# Get results
print(f"Best model: {results.best_model_name}")
print(f"Best score: {results.best_score}")
print(f"Execution time: {results.execution_time:.2f} seconds")
```

## **Sample Datasets**

### **Iris Dataset (Classification)**
```python
from sklearn.datasets import load_iris
import pandas as pd

# Load and save iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.to_csv('data/raw/iris.csv', index=False)
```

### **Boston Housing Dataset (Regression)**
```python
from sklearn.datasets import load_boston
import pandas as pd

# Load and save boston dataset
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['target'] = boston.target
df.to_csv('data/raw/boston.csv', index=False)
```

## **Common Use Cases**

### **1. Binary Classification**
```python
# Dataset: Customer churn prediction
config = PipelineConfig(
    time_budget=3600,  # 1 hour
    optimization_metric="f1"  # F1 score for imbalanced data
)
```

### **2. Multi-class Classification**
```python
# Dataset: Image classification
config = PipelineConfig(
    time_budget=7200,  # 2 hours
    optimization_metric="accuracy"
)
```

### **3. Regression**
```python
# Dataset: House price prediction
config = PipelineConfig(
    time_budget=3600,  # 1 hour
    optimization_metric="r2"  # R-squared for regression
)
```

## **Troubleshooting**

### **Common Issues**

#### **1. API Key Not Working**
```bash
# Check if API key is set
python -c "import os; print('API Key:', 'SET' if os.getenv('OPENROUTER_API_KEY') else 'NOT SET')"

# Test API connection
python -c "
from src.utils.llm_client import LLMClient
client = LLMClient()
print('LLM Client:', 'OK' if client else 'FAILED')
"
```

#### **2. Port Already in Use**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

#### **3. Dependencies Issues**
```bash
# Reinstall dependencies
pip uninstall -y autonomous-ml-agent
pip install -e .

# Clear npm cache
cd frontend
rm -rf node_modules package-lock.json
npm install
```

#### **4. File Upload Issues**
```bash
# Check file permissions
ls -la data/raw/

# Create directory if missing
mkdir -p data/raw
```

### **Debug Mode**

```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Start with verbose output
python3 -m uvicorn src.service.app:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

## **Next Steps**

### **1. Explore Advanced Features**
- [Model Training](Model-Training.md) - Learn about algorithm selection and hyperparameter optimization
- [LLM Integration](LLM-Integration.md) - Understand LLM orchestration
- [API Reference](API-Reference.md) - Complete API documentation

### **2. Customize Your Setup**
- [Configuration](Configuration.md) - Environment variables and settings
- [Development Setup](Development-Setup.md) - Local development environment
- [Production Deployment](Production-Deployment.md) - Deploy to production

### **3. Contribute**
- [Contributing Guide](CONTRIBUTING.md) - How to contribute to the project
- [Code of Conduct](CODE_OF_CONDUCT.md) - Community guidelines
- [GitHub Issues](https://github.com/Hamzakhan7473/Autonomous-ML-Agent/issues) - Report bugs or request features

## **Performance Tips**

### **1. Dataset Size**
- **Small datasets** (< 10K rows): Use 30-60 minutes time budget
- **Medium datasets** (10K-100K rows): Use 1-2 hours time budget
- **Large datasets** (> 100K rows): Use 2-4 hours time budget

### **2. Model Selection**
- **Binary classification**: Focus on F1 score for imbalanced data
- **Multi-class**: Use accuracy or macro F1
- **Regression**: Use R-squared or RMSE

### **3. Feature Engineering**
- Let the LLM handle automatic feature engineering
- Provide domain knowledge through data column names
- Use meaningful feature names for better LLM understanding

## **Support**

### **Getting Help**
- 📧 **Email**: hamzakhan@taxora.ai
- 💼 **LinkedIn**: [Abu Hamza Khan](https://www.linkedin.com/in/abuhamzakhan/)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Hamzakhan7473/Autonomous-ML-Agent/discussions)
- 🐛 **Issues**: [GitHub Issues](https://github.com/Hamzakhan7473/Autonomous-ML-Agent/issues)

### **Community**
- Join our Discord server
- Follow us on Twitter
- Star the repository on GitHub

---

**🎉 Congratulations!** You've successfully set up and run your first autonomous ML pipeline. The system is now ready to handle your machine learning tasks automatically!

**Ready for more?** Check out the [User Manual](User-Manual.md) for detailed usage instructions and advanced features.

