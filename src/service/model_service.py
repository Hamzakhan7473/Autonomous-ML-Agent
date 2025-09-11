"""
Model Service Module

This module provides a FastAPI service for deploying and serving trained models
with automatic preprocessing and inference capabilities.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..core.orchestrator import AutonomousMLAgent, PipelineConfig
from ..evaluation.leaderboard import ModelLeaderboard
from ..utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    
    features: List[Union[float, int, str]] = Field(..., description="Feature values for prediction")
    feature_names: Optional[List[str]] = Field(None, description="Names of features")


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""
    
    data: List[List[Union[float, int, str]]] = Field(..., description="Batch of feature vectors")
    feature_names: Optional[List[str]] = Field(None, description="Names of features")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    
    prediction: Union[int, float, str] = Field(..., description="Predicted value")
    probability: Optional[Dict[str, float]] = Field(None, description="Prediction probabilities")
    confidence: Optional[float] = Field(None, description="Prediction confidence")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    
    predictions: List[Union[int, float, str]] = Field(..., description="Batch predictions")
    probabilities: Optional[List[Dict[str, float]]] = Field(None, description="Batch probabilities")
    inference_time_ms: float = Field(..., description="Total inference time in milliseconds")


class ModelInfo(BaseModel):
    """Model information response."""
    
    model_name: str = Field(..., description="Name of the model")
    model_type: str = Field(..., description="Type of the model")
    task_type: str = Field(..., description="Type of ML task")
    accuracy: float = Field(..., description="Model accuracy")
    feature_names: List[str] = Field(..., description="Feature names")
    preprocessing_steps: List[str] = Field(..., description="Preprocessing steps")
    created_at: str = Field(..., description="Model creation timestamp")
    version: str = Field(..., description="Model version")


class ModelService:
    """Model service for handling model deployment and inference."""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """Initialize the model service.
        
        Args:
            model_path: Path to the saved model
            config_path: Path to the model configuration
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        self.model = None
        self.preprocessor = None
        self.config = None
        self.model_info = None
        self._load_model()
    
    def _load_model(self):
        """Load the model and preprocessor."""
        try:
            # Load model
            self.model = joblib.load(self.model_path / "best_model.pkl")
            
            # Load preprocessor
            preprocessor_path = self.model_path / "preprocessor.pkl"
            if preprocessor_path.exists():
                self.preprocessor = joblib.load(preprocessor_path)
            
            # Load configuration
            if self.config_path and self.config_path.exists():
                with open(self.config_path) as f:
                    config_data = json.load(f)
                    self.config = config_data
            
            # Create model info
            self.model_info = self._create_model_info()
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _create_model_info(self) -> ModelInfo:
        """Create model information."""
        return ModelInfo(
            model_name=getattr(self.model, '__class__', {}).get('__name__', 'Unknown'),
            model_type="unknown",
            task_type="unknown",
            accuracy=getattr(self.model, 'score', 0.0),
            feature_names=getattr(self.preprocessor, 'feature_names_in_', []),
            preprocessing_steps=getattr(self.preprocessor, 'steps', []),
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            version="1.0.0"
        )
    
    def predict(self, features: List[Union[float, int, str]], feature_names: Optional[List[str]] = None) -> PredictionResponse:
        """Make a single prediction.
        
        Args:
            features: Feature values
            feature_names: Feature names
            
        Returns:
            Prediction response
        """
        start_time = time.time()
        
        try:
            # Create DataFrame
            if feature_names:
                df = pd.DataFrame([features], columns=feature_names)
            else:
                df = pd.DataFrame([features])
            
            # Preprocess if preprocessor is available
            if self.preprocessor:
                df = self.preprocessor.transform(df)
            
            # Make prediction
            prediction = self.model.predict(df)[0]
            
            # Get probabilities if available
            probability = None
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(df)[0]
                classes = getattr(self.model, 'classes_', range(len(proba)))
                probability = {str(cls): float(prob) for cls, prob in zip(classes, proba)}
            
            # Calculate confidence
            confidence = None
            if probability:
                confidence = max(probability.values())
            
            inference_time = (time.time() - start_time) * 1000
            
            return PredictionResponse(
                prediction=prediction,
                probability=probability,
                confidence=confidence,
                inference_time_ms=inference_time
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def predict_batch(self, data: List[List[Union[float, int, str]]], feature_names: Optional[List[str]] = None) -> BatchPredictionResponse:
        """Make batch predictions.
        
        Args:
            data: Batch of feature vectors
            feature_names: Feature names
            
        Returns:
            Batch prediction response
        """
        start_time = time.time()
        
        try:
            # Create DataFrame
            if feature_names:
                df = pd.DataFrame(data, columns=feature_names)
            else:
                df = pd.DataFrame(data)
            
            # Preprocess if preprocessor is available
            if self.preprocessor:
                df = self.preprocessor.transform(df)
            
            # Make predictions
            predictions = self.model.predict(df).tolist()
            
            # Get probabilities if available
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(df)
                classes = getattr(self.model, 'classes_', range(proba.shape[1]))
                probabilities = [
                    {str(cls): float(prob) for cls, prob in zip(classes, row)}
                    for row in proba
                ]
            
            inference_time = (time.time() - start_time) * 1000
            
            return BatchPredictionResponse(
                predictions=predictions,
                probabilities=probabilities,
                inference_time_ms=inference_time
            )
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
    
    def get_model_info(self) -> ModelInfo:
        """Get model information."""
        return self.model_info


def create_model_service_app(model_path: str, config_path: Optional[str] = None) -> FastAPI:
    """Create FastAPI application for model service.
    
    Args:
        model_path: Path to the saved model
        config_path: Path to the model configuration
        
    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="Autonomous ML Model Service",
        description="FastAPI service for serving autonomous ML models",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize model service
    model_service = ModelService(model_path, config_path)
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Autonomous ML Model Service",
            "version": "1.0.0",
            "status": "running"
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": time.time()}
    
    @app.get("/model/info", response_model=ModelInfo)
    async def get_model_info():
        """Get model information."""
        return model_service.get_model_info()
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """Make a single prediction."""
        return model_service.predict(request.features, request.feature_names)
    
    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    async def predict_batch(request: BatchPredictionRequest):
        """Make batch predictions."""
        return model_service.predict_batch(request.data, request.feature_names)
    
    @app.post("/predict/file")
    async def predict_from_file(file: UploadFile = File(...)):
        """Make predictions from uploaded CSV file."""
        try:
            # Read CSV file
            content = await file.read()
            df = pd.read_csv(pd.io.common.StringIO(content.decode('utf-8')))
            
            # Convert to list of lists
            data = df.values.tolist()
            feature_names = df.columns.tolist()
            
            # Make predictions
            response = model_service.predict_batch(data, feature_names)
            
            # Add predictions to DataFrame
            df['prediction'] = response.predictions
            if response.probabilities:
                prob_df = pd.DataFrame(response.probabilities)
                df = pd.concat([df, prob_df], axis=1)
            
            # Return as CSV
            csv_content = df.to_csv(index=False)
            return {
                "predictions": csv_content,
                "inference_time_ms": response.inference_time_ms
            }
            
        except Exception as e:
            logger.error(f"File prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"File prediction failed: {str(e)}")
    
    @app.get("/model/feature-importance")
    async def get_feature_importance():
        """Get feature importance if available."""
        try:
            if hasattr(model_service.model, 'feature_importances_'):
                importance = model_service.model.feature_importances_
                feature_names = model_service.model_info.feature_names
                
                if len(feature_names) == len(importance):
                    return {
                        "feature_importance": dict(zip(feature_names, importance.tolist()))
                    }
                else:
                    return {"feature_importance": dict(enumerate(importance.tolist()))}
            else:
                return {"message": "Feature importance not available for this model"}
                
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")
    
    @app.get("/model/performance")
    async def get_model_performance():
        """Get model performance metrics."""
        try:
            return {
                "accuracy": model_service.model_info.accuracy,
                "model_name": model_service.model_info.model_name,
                "task_type": model_service.model_info.task_type,
                "created_at": model_service.model_info.created_at,
                "version": model_service.model_info.version
            }
        except Exception as e:
            logger.error(f"Failed to get model performance: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get model performance: {str(e)}")
    
    return app


def create_deployment_script(model_path: str, config_path: Optional[str] = None, port: int = 8000) -> str:
    """Generate deployment script for the model service.
    
    Args:
        model_path: Path to the saved model
        config_path: Path to the model configuration
        port: Port for the service
        
    Returns:
        Deployment script content
    """
    script_content = f"""#!/bin/bash
# Auto-generated deployment script for Autonomous ML Model Service

set -e

echo "🚀 Deploying Autonomous ML Model Service..."

# Check if model exists
if [ ! -d "{model_path}" ]; then
    echo "❌ Model not found at {model_path}"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Start the service
echo "🌐 Starting model service on port {port}..."
python -c "
from src.service.model_service import create_model_service_app
import uvicorn

app = create_model_service_app('{model_path}', '{config_path or ""}')
uvicorn.run(app, host='0.0.0.0', port={port})
"

echo "✅ Model service deployed successfully!"
echo "📊 API Documentation: http://localhost:{port}/docs"
echo "🔍 Health Check: http://localhost:{port}/health"
"""
    
    return script_content


def create_dockerfile(model_path: str, config_path: Optional[str] = None) -> str:
    """Generate Dockerfile for the model service.
    
    Args:
        model_path: Path to the saved model
        config_path: Path to the model configuration
        
    Returns:
        Dockerfile content
    """
    dockerfile_content = f"""FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY {model_path}/ ./model/

# Copy model configuration if exists
{f"COPY {config_path} ./config.json" if config_path else ""}

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Start the service
CMD ["python", "-c", "
from src.service.model_service import create_model_service_app
import uvicorn

app = create_model_service_app('./model', {'./config.json' if config_path else 'None'})
uvicorn.run(app, host='0.0.0.0', port=8000)
"]
"""
    
    return dockerfile_content
