# 📖 User Manual

## **Table of Contents**

1. [Getting Started](#getting-started)
2. [Web Interface](#web-interface)
3. [API Usage](#api-usage)
4. [Python SDK](#python-sdk)
5. [Data Preparation](#data-preparation)
6. [Pipeline Configuration](#pipeline-configuration)
7. [Model Selection](#model-selection)
8. [Results Interpretation](#results-interpretation)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## **Getting Started**

### **What is the Autonomous ML Agent?**

The Autonomous ML Agent is an intelligent machine learning pipeline that automatically:
- Analyzes your datasets
- Selects appropriate algorithms
- Optimizes hyperparameters
- Trains multiple models
- Provides insights and recommendations

### **Key Features**

- **🤖 LLM-Powered**: Uses Large Language Models for intelligent decision-making
- **🔄 End-to-End Automation**: Complete pipeline from data to insights
- **📊 Multi-Model Training**: Tests multiple algorithms automatically
- **⚡ Hyperparameter Optimization**: Intelligent parameter tuning
- **📈 Performance Monitoring**: Real-time progress tracking
- **🎯 Customizable**: Flexible configuration options

## **Web Interface**

### **Dashboard Overview**

The web interface provides an intuitive dashboard with four main sections:

1. **Data Upload**: Upload and analyze your datasets
2. **Pipeline Configuration**: Configure ML pipeline settings
3. **Execution Monitoring**: Monitor pipeline progress
4. **Results Visualization**: View and analyze results

### **Step 1: Data Upload**

#### **Supported File Formats**
- **CSV files** (`.csv`) - Most common format
- **Excel files** (`.xlsx`, `.xls`) - Limited support
- **JSON files** (`.json`) - Structured data

#### **File Requirements**
- **Minimum size**: 100 rows
- **Maximum size**: 1GB (recommended)
- **Encoding**: UTF-8
- **Delimiter**: Comma (`,`)

#### **Upload Process**
1. **Drag and Drop**: Simply drag your file onto the upload area
2. **Browse Files**: Click "Browse" to select a file from your computer
3. **Auto-Analysis**: The system automatically analyzes your dataset
4. **Review Results**: Check the dataset summary and detected target column

#### **Dataset Analysis Results**
After upload, you'll see:
- **File Information**: Name, size, and path
- **Dataset Shape**: Number of rows and columns
- **Target Column**: Automatically detected target variable
- **Data Types**: Column types and missing values
- **Preview**: First few rows of your data

### **Step 2: Pipeline Configuration**

#### **Basic Configuration**

**Time Budget**
- **Short**: 30 minutes - Quick exploration
- **Medium**: 1-2 hours - Balanced approach
- **Long**: 3-4 hours - Comprehensive analysis

**Optimization Metric**
- **Auto**: Automatically select based on task type
- **Accuracy**: For balanced classification
- **F1 Score**: For imbalanced classification
- **Precision**: For high-precision requirements
- **Recall**: For high-recall requirements
- **ROC AUC**: For overall classification performance
- **R-squared**: For regression tasks
- **RMSE**: For regression with error focus

**Random State**
- **Fixed seed** (e.g., 42): Reproducible results
- **Random**: Different results each time

#### **Advanced Configuration**

**Model Selection**
- **Auto**: Let the system choose models
- **Custom**: Select specific algorithms
- **Ensemble**: Enable ensemble methods

**Preprocessing Options**
- **Missing Values**: Auto-handle missing data
- **Categorical Encoding**: One-hot or label encoding
- **Feature Scaling**: Standard or min-max scaling
- **Feature Selection**: Automatic feature selection

### **Step 3: Execution Monitoring**

#### **Real-time Progress**
- **Status Indicators**: Running, completed, failed
- **Progress Bar**: Visual progress tracking
- **Time Elapsed**: Current execution time
- **Estimated Completion**: Remaining time estimate

#### **Pipeline Steps**
1. **Data Analysis**: Analyzing dataset characteristics
2. **LLM Planning**: Creating execution plan
3. **Preprocessing**: Cleaning and transforming data
4. **Model Training**: Training multiple algorithms
5. **Hyperparameter Optimization**: Tuning parameters
6. **Evaluation**: Testing model performance
7. **Results Generation**: Creating insights and recommendations

#### **Status Messages**
- **"Analyzing dataset..."**: Initial data analysis
- **"Creating execution plan..."**: LLM planning phase
- **"Training models..."**: Model training in progress
- **"Optimizing hyperparameters..."**: Parameter tuning
- **"Evaluating results..."**: Performance assessment
- **"Generating insights..."**: Final analysis

### **Step 4: Results Visualization**

#### **Performance Summary**
- **Best Model**: Top-performing algorithm
- **Best Score**: Performance metric value
- **Execution Time**: Total pipeline time
- **Models Trained**: Number of algorithms tested

#### **Model Comparison Table**
| Model | Score | Time | Parameters |
|-------|-------|------|------------|
| Random Forest | 0.95 | 45s | n_estimators=100 |
| XGBoost | 0.94 | 38s | learning_rate=0.1 |
| Logistic Regression | 0.92 | 12s | C=1.0 |

#### **Feature Importance**
- **Bar Chart**: Visual feature importance
- **Numerical Values**: Exact importance scores
- **Interpretation**: LLM-generated explanations

#### **Insights and Recommendations**
- **Performance Analysis**: Why certain models performed well
- **Feature Insights**: Which features are most important
- **Improvement Suggestions**: How to enhance performance
- **Deployment Considerations**: Production recommendations

## **API Usage**

### **Authentication**

Currently, no authentication is required. In production, consider implementing API keys or OAuth.

### **Base URL**

```
Development: http://localhost:8000
Production: https://your-domain.com
```

### **Endpoints Overview**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/analyze` | POST | Analyze dataset |
| `/pipeline/run` | POST | Start pipeline |
| `/pipeline/status/{task_id}` | GET | Get task status |
| `/predict` | POST | Make predictions |
| `/model/info/{task_id}` | GET | Get model info |
| `/results/{task_id}` | GET | Download results |
| `/tasks` | GET | List all tasks |

### **Complete Workflow Example**

#### **1. Upload and Analyze Dataset**

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@dataset.csv"
```

**Response:**
```json
{
  "filename": "dataset.csv",
  "dataset_path": "data/raw/dataset.csv",
  "target_column": "target",
  "shape": [1000, 10],
  "columns": ["feature1", "feature2", "target"],
  "target_type": "classification",
  "missing_percentage": 5.2
}
```

#### **2. Start Pipeline**

```bash
curl -X POST "http://localhost:8000/pipeline/run" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "data/raw/dataset.csv",
    "target_column": "target",
    "time_budget": 1800,
    "optimization_metric": "accuracy"
  }'
```

**Response:**
```json
{
  "task_id": "task_1703123456",
  "status": "running",
  "message": "Pipeline started successfully",
  "estimated_time": 1800
}
```

#### **3. Monitor Progress**

```bash
curl -X GET "http://localhost:8000/pipeline/status/task_1703123456"
```

**Response (Running):**
```json
{
  "task_id": "task_1703123456",
  "status": "running",
  "progress": 0.65,
  "message": "Training models... 1170.5s elapsed"
}
```

**Response (Completed):**
```json
{
  "task_id": "task_1703123456",
  "status": "completed",
  "progress": 1.0,
  "message": "Pipeline completed successfully",
  "results": {
    "best_model": "RandomForestClassifier",
    "best_score": 0.95,
    "execution_time": 1800.5,
    "models_trained": 7
  }
}
```

#### **4. Make Predictions**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task_1703123456",
    "features": [[1.0, 2.0, 3.0, 4.0]]
  }'
```

**Response:**
```json
{
  "predictions": [0],
  "probabilities": [[0.8, 0.2]],
  "confidence": 0.8
}
```

#### **5. Get Model Information**

```bash
curl -X GET "http://localhost:8000/model/info/task_1703123456"
```

**Response:**
```json
{
  "task_id": "task_1703123456",
  "model_info": {
    "model_name": "RandomForestClassifier",
    "best_score": 0.95,
    "best_params": {
      "n_estimators": 100,
      "max_depth": 10
    },
    "execution_time": 1800.5
  },
  "feature_importance": [0.3, 0.25, 0.2, 0.15, 0.1]
}
```

## **Python SDK**

### **Installation**

```bash
pip install autonomous-ml-agent
```

### **Basic Usage**

```python
from src import AutonomousMLAgent, PipelineConfig

# Initialize agent
agent = AutonomousMLAgent(
    config=PipelineConfig(
        time_budget=3600,  # 1 hour
        optimization_metric="accuracy"
    )
)

# Run pipeline
results = agent.run(
    dataset_path="data/raw/dataset.csv",
    target_column=""  # Auto-detect
)

# Get results
print(f"Best model: {results.best_model_name}")
print(f"Best score: {results.best_score}")
print(f"Execution time: {results.execution_time:.2f} seconds")
```

### **Advanced Configuration**

```python
# Custom configuration
config = PipelineConfig(
    time_budget=7200,  # 2 hours
    optimization_metric="f1",  # F1 score
    random_state=42,  # Reproducible results
    output_dir="./results",  # Custom output directory
    save_models=True,  # Save trained models
    save_results=True,  # Save results
    verbose=True  # Verbose logging
)

agent = AutonomousMLAgent(config)
results = agent.run("dataset.csv", "target")
```

### **Making Predictions**

```python
# Make predictions on new data
import pandas as pd

# Load new data
new_data = pd.read_csv("new_data.csv")

# Make predictions
predictions = agent.predict(new_data)
print(f"Predictions: {predictions}")

# Get probabilities (for classification)
if hasattr(agent.results.best_model, 'predict_proba'):
    probabilities = agent.predict_proba(new_data)
    print(f"Probabilities: {probabilities}")
```

### **Feature Importance**

```python
# Get feature importance
importance = agent.get_feature_importance()
print(f"Feature importance: {importance}")

# Get model summary
summary = agent.get_model_summary()
print(f"Model summary: {summary}")
```

## **Data Preparation**

### **Data Quality Requirements**

#### **Minimum Requirements**
- **Rows**: At least 100 samples
- **Columns**: At least 2 features + target
- **Missing Values**: Less than 50% missing per column
- **Target Variable**: Clearly defined and consistent

#### **Data Types**
- **Numerical**: Continuous or discrete numbers
- **Categorical**: Text labels or categories
- **Target Variable**: Must be consistent type

### **Common Data Issues**

#### **1. Missing Values**
```python
# Check missing values
df.isnull().sum()

# Handle missing values
df.fillna(df.median(), inplace=True)  # Numerical
df.fillna(df.mode().iloc[0], inplace=True)  # Categorical
```

#### **2. Categorical Encoding**
```python
# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['category_column'])

# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category_column'] = le.fit_transform(df['category_column'])
```

#### **3. Feature Scaling**
```python
# Standard scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
```

### **Best Practices**

#### **1. Data Cleaning**
- Remove duplicates
- Handle outliers appropriately
- Ensure consistent data types
- Validate data ranges

#### **2. Feature Engineering**
- Create meaningful feature names
- Remove irrelevant features
- Consider domain knowledge
- Avoid data leakage

#### **3. Target Variable**
- Ensure balanced classes (for classification)
- Check for data leakage
- Validate target variable quality
- Consider class imbalance

## **Pipeline Configuration**

### **Time Budget Guidelines**

#### **Small Datasets** (< 10K rows)
- **Quick**: 30 minutes
- **Standard**: 60 minutes
- **Comprehensive**: 120 minutes

#### **Medium Datasets** (10K-100K rows)
- **Quick**: 60 minutes
- **Standard**: 120 minutes
- **Comprehensive**: 240 minutes

#### **Large Datasets** (> 100K rows)
- **Quick**: 120 minutes
- **Standard**: 240 minutes
- **Comprehensive**: 480 minutes

### **Optimization Metrics**

#### **Classification Metrics**
- **Accuracy**: Overall correctness (balanced data)
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **ROC AUC**: Area under ROC curve

#### **Regression Metrics**
- **R-squared**: Proportion of variance explained
- **RMSE**: Root mean squared error
- **MAE**: Mean absolute error
- **MSE**: Mean squared error

### **Model Selection Strategy**

#### **Automatic Selection**
The system automatically selects models based on:
- Dataset characteristics
- Task type (classification/regression)
- Data size and complexity
- Time budget constraints

#### **Manual Selection**
You can specify models to use:
```python
config = PipelineConfig(
    models_to_try=[
        "random_forest",
        "xgboost",
        "logistic_regression"
    ]
)
```

## **Model Selection**

### **Available Algorithms**

#### **Classification**
- **Logistic Regression**: Linear classifier
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting
- **LightGBM**: Light gradient boosting
- **CatBoost**: Categorical boosting
- **k-NN**: k-nearest neighbors
- **SVM**: Support vector machine
- **Naive Bayes**: Probabilistic classifier
- **Neural Network**: Multi-layer perceptron

#### **Regression**
- **Linear Regression**: Linear model
- **Random Forest**: Ensemble regression
- **XGBoost**: Gradient boosting regression
- **LightGBM**: Light gradient boosting regression
- **CatBoost**: Categorical boosting regression
- **k-NN**: k-nearest neighbors regression
- **Ridge**: L2 regularized linear regression
- **Lasso**: L1 regularized linear regression
- **Neural Network**: Multi-layer perceptron regression

### **Algorithm Selection Guidelines**

#### **Small Datasets** (< 1K rows)
- **Linear models**: Logistic/Linear Regression
- **Simple ensemble**: Random Forest
- **Avoid**: Complex models that may overfit

#### **Medium Datasets** (1K-10K rows)
- **Ensemble methods**: Random Forest, XGBoost
- **Linear models**: Logistic/Linear Regression
- **Neural networks**: For complex patterns

#### **Large Datasets** (> 10K rows)
- **Gradient boosting**: XGBoost, LightGBM, CatBoost
- **Ensemble methods**: Random Forest
- **Neural networks**: For very complex patterns

### **Hyperparameter Optimization**

#### **Strategies**
- **Random Search**: Random parameter sampling
- **Grid Search**: Exhaustive parameter search
- **Bayesian Optimization**: Intelligent parameter search

#### **Time Allocation**
- **Quick**: 20% of time budget
- **Standard**: 40% of time budget
- **Comprehensive**: 60% of time budget

## **Results Interpretation**

### **Performance Metrics**

#### **Classification Results**
```python
# Example classification results
{
    "accuracy": 0.95,
    "precision": 0.94,
    "recall": 0.96,
    "f1_score": 0.95,
    "roc_auc": 0.98
}
```

#### **Regression Results**
```python
# Example regression results
{
    "r2_score": 0.87,
    "rmse": 0.15,
    "mae": 0.12,
    "mse": 0.023
}
```

### **Model Comparison**

#### **Performance Ranking**
Models are ranked by the primary optimization metric:
1. **Best Model**: Highest performance
2. **Runner-up**: Second best
3. **Baseline**: Simple baseline model

#### **Trade-offs**
Consider:
- **Performance vs Speed**: Faster models may have lower accuracy
- **Interpretability vs Performance**: Complex models may be less interpretable
- **Memory vs Performance**: Large models may require more memory

### **Feature Importance**

#### **Interpretation**
- **High importance**: Features that strongly influence predictions
- **Low importance**: Features that have little impact
- **Zero importance**: Features that don't contribute

#### **Actionable Insights**
- **Feature selection**: Remove low-importance features
- **Feature engineering**: Focus on high-importance features
- **Data collection**: Prioritize important features

### **Insights and Recommendations**

#### **Performance Analysis**
- **Model strengths**: What the model does well
- **Model limitations**: Areas for improvement
- **Data quality**: Issues with the dataset

#### **Improvement Suggestions**
- **Feature engineering**: New features to create
- **Data collection**: Additional data to gather
- **Model selection**: Alternative algorithms to try

#### **Deployment Considerations**
- **Production requirements**: Scalability and performance
- **Monitoring**: Key metrics to track
- **Maintenance**: Regular retraining schedule

## **Best Practices**

### **1. Data Preparation**
- **Clean data**: Remove duplicates and handle missing values
- **Feature engineering**: Create meaningful features
- **Validation**: Split data for training and testing
- **Documentation**: Document data sources and transformations

### **2. Pipeline Configuration**
- **Time budget**: Allocate sufficient time for comprehensive analysis
- **Optimization metric**: Choose metric that aligns with business goals
- **Random state**: Use fixed seed for reproducible results
- **Output directory**: Organize results systematically

### **3. Model Selection**
- **Start simple**: Begin with linear models
- **Progress to complex**: Try ensemble methods
- **Consider trade-offs**: Balance performance and interpretability
- **Validate results**: Use cross-validation and holdout sets

### **4. Results Interpretation**
- **Context matters**: Consider business context
- **Multiple metrics**: Don't rely on single metric
- **Feature importance**: Understand what drives predictions
- **Limitations**: Acknowledge model limitations

### **5. Production Deployment**
- **Model versioning**: Track model versions
- **Performance monitoring**: Monitor model performance
- **Data drift**: Detect changes in data distribution
- **Regular retraining**: Update models periodically

## **Troubleshooting**

### **Common Issues**

#### **1. Pipeline Fails to Start**
**Symptoms**: Pipeline status remains "idle"
**Solutions**:
- Check API key configuration
- Verify dataset file exists
- Ensure sufficient disk space
- Check system resources

#### **2. Poor Model Performance**
**Symptoms**: Low accuracy scores
**Solutions**:
- Check data quality
- Increase time budget
- Try different optimization metrics
- Consider feature engineering

#### **3. Long Execution Times**
**Symptoms**: Pipeline takes too long
**Solutions**:
- Reduce time budget
- Use fewer models
- Optimize dataset size
- Check system resources

#### **4. Memory Issues**
**Symptoms**: Out of memory errors
**Solutions**:
- Reduce dataset size
- Use chunked processing
- Increase system memory
- Optimize data types

### **Debug Mode**

#### **Enable Debug Logging**
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
```

#### **Verbose Output**
```python
config = PipelineConfig(verbose=True)
```

#### **Check System Status**
```bash
curl -X GET "http://localhost:8000/health"
```

### **Performance Optimization**

#### **1. System Resources**
- **CPU**: Use multi-core systems
- **Memory**: Allocate sufficient RAM
- **Storage**: Use SSD for better I/O
- **Network**: Ensure stable internet connection

#### **2. Data Optimization**
- **File format**: Use efficient formats (CSV, Parquet)
- **Data types**: Use appropriate data types
- **Compression**: Compress large datasets
- **Sampling**: Use representative samples

#### **3. Configuration Tuning**
- **Time budget**: Allocate time based on dataset size
- **Model selection**: Choose models appropriate for data size
- **Parallel processing**: Enable multi-threading
- **Caching**: Cache intermediate results

---

This comprehensive user manual provides everything you need to effectively use the Autonomous ML Agent. For additional help, refer to the [FAQ](FAQ.md) or contact support.

