# 🌐 API Reference

## **Overview**

The Autonomous ML Agent provides a comprehensive REST API built with FastAPI. The API enables programmatic access to all ML pipeline functionality including data analysis, model training, and predictions.

## **Base URL**

```
Development: http://localhost:8000
Production: https://your-domain.com
```

## **Authentication**

Currently, the API does not require authentication for development. In production, consider implementing:
- API key authentication
- OAuth 2.0
- JWT tokens

## **Content Types**

- **Request**: `application/json` for JSON payloads, `multipart/form-data` for file uploads
- **Response**: `application/json`

## **Error Handling**

All errors follow a consistent format:

```json
{
  "error": "Error message",
  "status_code": 400,
  "detail": "Additional error details"
}
```

## **API Endpoints**

### **Health Check**

#### `GET /health`

Check the health status of the API service.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1703123456.789,
  "llm_client_available": true
}
```

**Status Codes:**
- `200 OK`: Service is healthy
- `503 Service Unavailable`: Service is unhealthy

---

### **Dataset Analysis**

#### `POST /analyze`

Analyze an uploaded dataset and return metadata.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Body**: Form data with file upload

**Parameters:**
- `file` (required): CSV file to analyze

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

**Example:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@dataset.csv"
```

**Status Codes:**
- `200 OK`: Analysis completed successfully
- `400 Bad Request`: Invalid file or analysis failed
- `422 Unprocessable Entity`: File format not supported

---

### **Pipeline Execution**

#### `POST /pipeline/run`

Start a new ML pipeline execution.

**Request:**
```json
{
  "dataset_path": "data/raw/dataset.csv",
  "target_column": "target",
  "time_budget": 3600,
  "optimization_metric": "auto",
  "random_state": 42,
  "output_dir": "./results",
  "save_models": true,
  "save_results": true,
  "verbose": false
}
```

**Parameters:**
- `dataset_path` (required): Path to the dataset file
- `target_column` (required): Name of the target column
- `time_budget` (optional): Time budget in seconds (default: 3600)
- `optimization_metric` (optional): Optimization metric (default: "auto")
- `random_state` (optional): Random seed (default: 42)
- `output_dir` (optional): Output directory (default: "./results")
- `save_models` (optional): Save trained models (default: true)
- `save_results` (optional): Save results (default: true)
- `verbose` (optional): Verbose logging (default: false)

**Response:**
```json
{
  "task_id": "task_1703123456",
  "status": "running",
  "message": "Pipeline started successfully",
  "estimated_time": 3600
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/pipeline/run" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "data/raw/dataset.csv",
    "target_column": "target",
    "time_budget": 1800
  }'
```

**Status Codes:**
- `200 OK`: Pipeline started successfully
- `400 Bad Request`: Invalid request parameters
- `500 Internal Server Error`: Pipeline startup failed

---

### **Task Status**

#### `GET /pipeline/status/{task_id}`

Get the current status of a pipeline task.

**Parameters:**
- `task_id` (required): Task identifier

**Response (Running):**
```json
{
  "task_id": "task_1703123456",
  "status": "running",
  "progress": 0.45,
  "message": "Pipeline running... 1620.5s elapsed"
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
    "status": "completed",
    "best_model": "RandomForestClassifier",
    "best_score": 0.95,
    "execution_time": 1800.5,
    "models_trained": 7,
    "message": "Pipeline completed successfully"
  }
}
```

**Response (Failed):**
```json
{
  "task_id": "task_1703123456",
  "status": "failed",
  "progress": 0.0,
  "message": "Pipeline failed",
  "error": "Error message details"
}
```

**Example:**
```bash
curl -X GET "http://localhost:8000/pipeline/status/task_1703123456"
```

**Status Codes:**
- `200 OK`: Status retrieved successfully
- `404 Not Found`: Task not found

---

### **Predictions**

#### `POST /predict`

Make predictions using a trained model.

**Request:**
```json
{
  "task_id": "task_1703123456",
  "features": [
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0]
  ],
  "feature_names": ["feature1", "feature2", "feature3", "feature4"]
}
```

**Parameters:**
- `task_id` (required): Task identifier of completed pipeline
- `features` (required): Array of feature arrays for prediction
- `feature_names` (optional): Names of features (for debugging)

**Response:**
```json
{
  "predictions": [0, 1],
  "probabilities": [[0.8, 0.2], [0.3, 0.7]],
  "confidence": 0.8
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task_1703123456",
    "features": [[1.0, 2.0, 3.0, 4.0]]
  }'
```

**Status Codes:**
- `200 OK`: Predictions generated successfully
- `400 Bad Request`: Model not ready or invalid input
- `404 Not Found`: Task not found
- `500 Internal Server Error`: Prediction failed

---

### **Model Information**

#### `GET /model/info/{task_id}`

Get detailed information about a trained model.

**Parameters:**
- `task_id` (required): Task identifier

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
    "execution_time": 1800.5,
    "insights": "Model insights and recommendations..."
  },
  "feature_importance": [0.3, 0.25, 0.2, 0.15, 0.1],
  "preprocessing_config": {
    "imputation_strategy": "median",
    "categorical_encoding": "onehot",
    "scaling": "standard"
  }
}
```

**Example:**
```bash
curl -X GET "http://localhost:8000/model/info/task_1703123456"
```

**Status Codes:**
- `200 OK`: Model information retrieved successfully
- `400 Bad Request`: Model not ready
- `404 Not Found`: Task not found

---

### **Results Download**

#### `GET /results/{task_id}`

Download pipeline results as a JSON file.

**Parameters:**
- `task_id` (required): Task identifier

**Response:**
- **Content-Type**: `application/json`
- **Body**: Complete pipeline results JSON file

**Example:**
```bash
curl -X GET "http://localhost:8000/results/task_1703123456" \
  -o results_task_1703123456.json
```

**Status Codes:**
- `200 OK`: Results file retrieved successfully
- `400 Bad Request`: Results not ready
- `404 Not Found`: Task or results file not found

---

### **Task Management**

#### `GET /tasks`

List all active tasks.

**Response:**
```json
{
  "tasks": [
    {
      "task_id": "task_1703123456",
      "status": "completed",
      "start_time": 1703123456.789,
      "elapsed_time": 1800.5
    },
    {
      "task_id": "task_1703127890",
      "status": "running",
      "start_time": 1703127890.123,
      "elapsed_time": 300.0
    }
  ]
}
```

**Example:**
```bash
curl -X GET "http://localhost:8000/tasks"
```

**Status Codes:**
- `200 OK`: Task list retrieved successfully

---

## **Data Models**

### **DatasetInfo**
```json
{
  "filename": "string",
  "dataset_path": "string",
  "target_column": "string",
  "shape": [number, number],
  "columns": ["string"],
  "target_type": "string",
  "missing_percentage": number
}
```

### **PipelineRequest**
```json
{
  "dataset_path": "string",
  "target_column": "string",
  "time_budget": number,
  "optimization_metric": "string",
  "random_state": number,
  "output_dir": "string",
  "save_models": boolean,
  "save_results": boolean,
  "verbose": boolean
}
```

### **PipelineResponse**
```json
{
  "task_id": "string",
  "status": "string",
  "message": "string",
  "estimated_time": number
}
```

### **TaskStatus**
```json
{
  "task_id": "string",
  "status": "string",
  "progress": number,
  "message": "string",
  "results": object,
  "error": "string"
}
```

### **PredictionRequest**
```json
{
  "task_id": "string",
  "features": [[number]],
  "feature_names": ["string"]
}
```

### **PredictionResponse**
```json
{
  "predictions": [number],
  "probabilities": [[number]],
  "confidence": number
}
```

## **Rate Limiting**

Currently, no rate limiting is implemented. Consider implementing:
- Request rate limiting per IP
- API key-based rate limiting
- Burst protection

## **CORS Configuration**

The API supports CORS with the following configuration:
- **Allowed Origins**: `*` (all origins)
- **Allowed Methods**: All HTTP methods
- **Allowed Headers**: All headers
- **Credentials**: Enabled

## **API Documentation**

Interactive API documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## **SDK Examples**

### **Python SDK**
```python
import requests

# Upload and analyze dataset
with open('dataset.csv', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/analyze', files=files)
    dataset_info = response.json()

# Run pipeline
pipeline_request = {
    "dataset_path": dataset_info['dataset_path'],
    "target_column": dataset_info['target_column'],
    "time_budget": 1800
}
response = requests.post('http://localhost:8000/pipeline/run', json=pipeline_request)
task_info = response.json()
task_id = task_info['task_id']

# Check status
response = requests.get(f'http://localhost:8000/pipeline/status/{task_id}')
status = response.json()

# Make predictions
prediction_request = {
    "task_id": task_id,
    "features": [[1.0, 2.0, 3.0, 4.0]]
}
response = requests.post('http://localhost:8000/predict', json=prediction_request)
predictions = response.json()
```

### **JavaScript SDK**
```javascript
// Upload and analyze dataset
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const analyzeResponse = await fetch('http://localhost:8000/analyze', {
  method: 'POST',
  body: formData
});
const datasetInfo = await analyzeResponse.json();

// Run pipeline
const pipelineRequest = {
  dataset_path: datasetInfo.dataset_path,
  target_column: datasetInfo.target_column,
  time_budget: 1800
};

const runResponse = await fetch('http://localhost:8000/pipeline/run', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(pipelineRequest)
});
const taskInfo = await runResponse.json();
const taskId = taskInfo.task_id;

// Check status
const statusResponse = await fetch(`http://localhost:8000/pipeline/status/${taskId}`);
const status = await statusResponse.json();

// Make predictions
const predictionRequest = {
  task_id: taskId,
  features: [[1.0, 2.0, 3.0, 4.0]]
};

const predictResponse = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(predictionRequest)
});
const predictions = await predictResponse.json();
```

## **Best Practices**

### **1. Error Handling**
- Always check HTTP status codes
- Handle network errors gracefully
- Implement retry logic for transient failures

### **2. Performance**
- Use appropriate time budgets for your datasets
- Monitor task status instead of blocking
- Implement progress indicators for long-running tasks

### **3. Security**
- Validate input data before sending requests
- Use HTTPS in production
- Implement proper authentication when needed

### **4. Monitoring**
- Log API requests and responses
- Monitor API performance metrics
- Set up alerts for failed requests

---

This API reference provides comprehensive documentation for integrating with the Autonomous ML Agent. For additional examples and use cases, see the [User Manual](User-Manual.md).
