"""
Pipeline Export and Artifact Generation

This module provides comprehensive pipeline export capabilities including
Python packages, FastAPI services, and deployment-ready artifacts.
"""

import json
import logging
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
import zipfile

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for pipeline export."""
    
    export_format: str = "fastapi"  # "fastapi", "python_package", "docker", "all"
    output_dir: str = "exports"
    include_metadata: bool = True
    include_examples: bool = True
    include_tests: bool = True
    compress_output: bool = True
    api_version: str = "v1"
    service_name: str = "ml_pipeline_service"
    description: str = "Auto-generated ML Pipeline Service"


@dataclass
class PipelineArtifact:
    """Represents a complete ML pipeline artifact."""
    
    model: Any
    preprocessing_pipeline: Any
    feature_names: List[str]
    target_column: str
    task_type: str
    model_name: str
    performance_metrics: Dict[str, float]
    meta_features: Dict[str, Any]
    model_explanation: Optional[Any] = None
    ensemble_config: Optional[Dict[str, Any]] = None


class PipelineExporter:
    """Main pipeline exporter for creating reusable artifacts."""
    
    def __init__(self, config: ExportConfig = None):
        """Initialize the pipeline exporter."""
        self.config = config or ExportConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_pipeline(
        self, 
        pipeline_artifact: PipelineArtifact,
        llm_client = None
    ) -> Dict[str, str]:
        """Export pipeline in the specified format(s)."""
        
        export_results = {}
        
        try:
            # Create timestamp-based directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = self.output_dir / f"{self.config.service_name}_{timestamp}"
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Save pipeline artifact
            self._save_pipeline_artifact(pipeline_artifact, export_dir)
            
            # Generate exports based on format
            if self.config.export_format in ["fastapi", "all"]:
                export_results["fastapi"] = self._export_fastapi_service(
                    pipeline_artifact, export_dir, llm_client
                )
            
            if self.config.export_format in ["python_package", "all"]:
                export_results["python_package"] = self._export_python_package(
                    pipeline_artifact, export_dir, llm_client
                )
            
            if self.config.export_format in ["docker", "all"]:
                export_results["docker"] = self._export_docker_service(
                    pipeline_artifact, export_dir, llm_client
                )
            
            # Compress if requested
            if self.config.compress_output:
                compressed_path = self._compress_export(export_dir)
                export_results["compressed"] = str(compressed_path)
            
            # Generate deployment scripts
            deployment_scripts = self._generate_deployment_scripts(
                pipeline_artifact, export_dir, llm_client
            )
            export_results["deployment_scripts"] = deployment_scripts
            
            logger.info(f"Pipeline exported successfully to {export_dir}")
            return export_results
            
        except Exception as e:
            logger.error(f"Failed to export pipeline: {e}")
            raise
    
    def _save_pipeline_artifact(self, artifact: PipelineArtifact, export_dir: Path):
        """Save the pipeline artifact components."""
        
        # Save model
        model_path = export_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(artifact.model, f)
        
        # Save preprocessing pipeline
        preprocessing_path = export_dir / "preprocessing.pkl"
        with open(preprocessing_path, "wb") as f:
            pickle.dump(artifact.preprocessing_pipeline, f)
        
        # Save metadata
        metadata = {
            "feature_names": artifact.feature_names,
            "target_column": artifact.target_column,
            "task_type": artifact.task_type,
            "model_name": artifact.model_name,
            "performance_metrics": artifact.performance_metrics,
            "meta_features": artifact.meta_features,
            "export_timestamp": datetime.now().isoformat(),
            "export_config": asdict(self.config)
        }
        
        metadata_path = export_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save model explanation if available
        if artifact.model_explanation:
            explanation_path = export_dir / "model_explanation.json"
            with open(explanation_path, "w") as f:
                # Convert to serializable format
                explanation_data = self._serialize_model_explanation(artifact.model_explanation)
                json.dump(explanation_data, f, indent=2)
    
    def _serialize_model_explanation(self, explanation: Any) -> Dict[str, Any]:
        """Serialize model explanation to JSON-serializable format."""
        
        try:
            if hasattr(explanation, "__dict__"):
                explanation_dict = explanation.__dict__.copy()
                
                # Convert numpy arrays to lists
                for key, value in explanation_dict.items():
                    if isinstance(value, np.ndarray):
                        explanation_dict[key] = value.tolist()
                    elif hasattr(value, "__dict__"):
                        explanation_dict[key] = self._serialize_model_explanation(value)
                
                return explanation_dict
            else:
                return {"explanation": str(explanation)}
                
        except Exception as e:
            logger.warning(f"Failed to serialize model explanation: {e}")
            return {"explanation": str(explanation)}
    
    def _export_fastapi_service(
        self, 
        artifact: PipelineArtifact, 
        export_dir: Path,
        llm_client = None
    ) -> str:
        """Export pipeline as FastAPI service."""
        
        service_dir = export_dir / "fastapi_service"
        service_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate FastAPI application
        fastapi_code = self._generate_fastapi_code(artifact, llm_client)
        
        # Write FastAPI files
        app_file = service_dir / "main.py"
        with open(app_file, "w") as f:
            f.write(fastapi_code["main"])
        
        # Write requirements
        requirements_file = service_dir / "requirements.txt"
        with open(requirements_file, "w") as f:
            f.write(fastapi_code["requirements"])
        
        # Write Dockerfile
        dockerfile = service_dir / "Dockerfile"
        with open(dockerfile, "w") as f:
            f.write(fastapi_code["dockerfile"])
        
        # Copy model artifacts
        shutil.copy(export_dir / "model.pkl", service_dir / "model.pkl")
        shutil.copy(export_dir / "preprocessing.pkl", service_dir / "preprocessing.pkl")
        shutil.copy(export_dir / "metadata.json", service_dir / "metadata.json")
        
        return str(service_dir)
    
    def _generate_fastapi_code(self, artifact: PipelineArtifact, llm_client = None) -> Dict[str, str]:
        """Generate FastAPI service code."""
        
        service_name = self.config.service_name
        api_version = self.config.api_version
        
        main_code = f'''"""
{self.config.description}

Auto-generated FastAPI service for ML pipeline deployment.
Generated on: {datetime.now().isoformat()}
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="{service_name}",
    description="{self.config.description}",
    version="{api_version}",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and preprocessing pipeline
MODEL_PATH = Path(__file__).parent / "model.pkl"
PREPROCESSING_PATH = Path(__file__).parent / "preprocessing.pkl"
METADATA_PATH = Path(__file__).parent / "metadata.json"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {{e}}")
    model = None

try:
    with open(PREPROCESSING_PATH, "rb") as f:
        preprocessing_pipeline = pickle.load(f)
    logger.info("Preprocessing pipeline loaded successfully")
except Exception as e:
    logger.error(f"Failed to load preprocessing pipeline: {{e}}")
    preprocessing_pipeline = None

try:
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    logger.info("Metadata loaded successfully")
except Exception as e:
    logger.error(f"Failed to load metadata: {{e}}")
    metadata = {{}}

# Pydantic models
class PredictionRequest(BaseModel):
    """Request model for predictions."""
    features: Dict[str, Union[float, int, str]] = Field(
        ..., 
        description="Feature values for prediction"
    )

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: Union[float, int, str] = Field(..., description="Model prediction")
    probability: Optional[float] = Field(None, description="Prediction probability")
    confidence: Optional[float] = Field(None, description="Prediction confidence")

class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str = Field(..., description="Name of the model")
    task_type: str = Field(..., description="Type of ML task")
    feature_names: List[str] = Field(..., description="List of feature names")
    performance_metrics: Dict[str, float] = Field(..., description="Model performance metrics")

@app.get("/")
async def root():
    """Root endpoint."""
    return {{
        "service": "{service_name}",
        "version": "{api_version}",
        "status": "healthy",
        "model_loaded": model is not None
    }}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {{
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessing_loaded": preprocessing_pipeline is not None
    }}

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return ModelInfo(
        model_name=metadata.get("model_name", "unknown"),
        task_type=metadata.get("task_type", "unknown"),
        feature_names=metadata.get("feature_names", []),
        performance_metrics=metadata.get("performance_metrics", {{}})
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction."""
    if not model or not preprocessing_pipeline:
        raise HTTPException(status_code=500, detail="Model or preprocessing pipeline not loaded")
    
    try:
        # Convert request to DataFrame
        df = pd.DataFrame([request.features])
        
        # Ensure all required features are present
        feature_names = metadata.get("feature_names", [])
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required features: {{list(missing_features)}}"
            )
        
        # Reorder columns to match training data
        df = df[feature_names]
        
        # Apply preprocessing
        if hasattr(preprocessing_pipeline, 'transform'):
            df_processed = preprocessing_pipeline.transform(df)
        else:
            df_processed = df.values
        
        # Make prediction
        prediction = model.predict(df_processed)
        
        # Get probability if available
        probability = None
        confidence = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(df_processed)
                if probabilities.ndim == 1:
                    probability = float(probabilities[0])
                else:
                    probability = float(np.max(probabilities[0]))
                confidence = float(np.max(probabilities[0]))
            except Exception as e:
                logger.warning(f"Failed to get prediction probability: {{e}}")
        
        # Convert prediction to appropriate type
        if isinstance(prediction, np.ndarray):
            prediction_value = prediction[0]
        else:
            prediction_value = prediction
        
        return PredictionResponse(
            prediction=float(prediction_value) if isinstance(prediction_value, (int, float, np.number)) else str(prediction_value),
            probability=probability,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {{e}}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {{str(e)}}")

@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    """Make batch predictions."""
    if not model or not preprocessing_pipeline:
        raise HTTPException(status_code=500, detail="Model or preprocessing pipeline not loaded")
    
    try:
        # Convert requests to DataFrame
        features_list = [req.features for req in requests]
        df = pd.DataFrame(features_list)
        
        # Ensure all required features are present
        feature_names = metadata.get("feature_names", [])
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required features: {{list(missing_features)}}"
            )
        
        # Reorder columns to match training data
        df = df[feature_names]
        
        # Apply preprocessing
        if hasattr(preprocessing_pipeline, 'transform'):
            df_processed = preprocessing_pipeline.transform(df)
        else:
            df_processed = df.values
        
        # Make predictions
        predictions = model.predict(df_processed)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(df_processed)
            except Exception as e:
                logger.warning(f"Failed to get prediction probabilities: {{e}}")
        
        # Format responses
        responses = []
        for i, prediction in enumerate(predictions):
            probability = None
            confidence = None
            
            if probabilities is not None:
                if probabilities.ndim == 1:
                    probability = float(probabilities[i])
                else:
                    probability = float(np.max(probabilities[i]))
                confidence = float(np.max(probabilities[i]))
            
            responses.append(PredictionResponse(
                prediction=float(prediction) if isinstance(prediction, (int, float, np.number)) else str(prediction),
                probability=probability,
                confidence=confidence
            ))
        
        return responses
        
    except Exception as e:
        logger.error(f"Batch prediction error: {{e}}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {{str(e)}}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        
        requirements = f'''fastapi==0.104.1
uvicorn[standard]==0.24.0
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
pydantic==2.5.0
python-multipart==0.0.6
'''
        
        dockerfile = f'''FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        
        return {
            "main": main_code,
            "requirements": requirements,
            "dockerfile": dockerfile
        }
    
    def _export_python_package(
        self, 
        artifact: PipelineArtifact, 
        export_dir: Path,
        llm_client = None
    ) -> str:
        """Export pipeline as Python package."""
        
        package_dir = export_dir / "python_package"
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate package code
        package_code = self._generate_python_package_code(artifact, llm_client)
        
        # Create package structure
        package_name = self.config.service_name.replace("-", "_").replace(" ", "_")
        src_dir = package_dir / package_name
        src_dir.mkdir(parents=True, exist_ok=True)
        
        # Write package files
        init_file = src_dir / "__init__.py"
        with open(init_file, "w") as f:
            f.write(package_code["init"])
        
        predictor_file = src_dir / "predictor.py"
        with open(predictor_file, "w") as f:
            f.write(package_code["predictor"])
        
        # Write setup.py
        setup_file = package_dir / "setup.py"
        with open(setup_file, "w") as f:
            f.write(package_code["setup"])
        
        # Write requirements
        requirements_file = package_dir / "requirements.txt"
        with open(requirements_file, "w") as f:
            f.write(package_code["requirements"])
        
        # Copy model artifacts
        models_dir = src_dir / "models"
        models_dir.mkdir(exist_ok=True)
        shutil.copy(export_dir / "model.pkl", models_dir / "model.pkl")
        shutil.copy(export_dir / "preprocessing.pkl", models_dir / "preprocessing.pkl")
        shutil.copy(export_dir / "metadata.json", models_dir / "metadata.json")
        
        return str(package_dir)
    
    def _generate_python_package_code(self, artifact: PipelineArtifact, llm_client = None) -> Dict[str, str]:
        """Generate Python package code."""
        
        package_name = self.config.service_name.replace("-", "_").replace(" ", "_")
        
        init_code = f'''"""
{self.config.description}

Auto-generated Python package for ML pipeline deployment.
"""

from .predictor import MLPredictor

__version__ = "{self.config.api_version}"
__author__ = "AutoML Pipeline Generator"
__description__ = "{self.config.description}"

__all__ = ["MLPredictor"]
'''
        
        predictor_code = f'''"""
ML Predictor class for making predictions with the trained model.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MLPredictor:
    """
    ML Predictor for making predictions with the trained model.
    
    This class provides a simple interface for loading the trained model
    and making predictions on new data.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ML Predictor.
        
        Args:
            model_path: Path to the model directory. If None, uses default path.
        """
        if model_path is None:
            model_path = Path(__file__).parent / "models"
        else:
            model_path = Path(model_path)
        
        self.model_path = model_path
        self.model = None
        self.preprocessing_pipeline = None
        self.metadata = None
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model artifacts."""
        try:
            # Load model
            model_file = self.model_path / "model.pkl"
            with open(model_file, "rb") as f:
                self.model = pickle.load(f)
            
            # Load preprocessing pipeline
            preprocessing_file = self.model_path / "preprocessing.pkl"
            with open(preprocessing_file, "rb") as f:
                self.preprocessing_pipeline = pickle.load(f)
            
            # Load metadata
            metadata_file = self.model_path / "metadata.json"
            with open(metadata_file) as f:
                self.metadata = json.load(f)
            
            logger.info("Model artifacts loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {{e}}")
            raise
    
    def predict(self, features: Dict[str, Union[float, int, str]]) -> Dict[str, Any]:
        """
        Make a prediction.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Dictionary containing prediction results
        """
        if not self.model or not self.preprocessing_pipeline:
            raise ValueError("Model or preprocessing pipeline not loaded")
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([features])
            
            # Ensure all required features are present
            feature_names = self.metadata.get("feature_names", [])
            missing_features = set(feature_names) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {{list(missing_features)}}")
            
            # Reorder columns to match training data
            df = df[feature_names]
            
            # Apply preprocessing
            if hasattr(self.preprocessing_pipeline, 'transform'):
                df_processed = self.preprocessing_pipeline.transform(df)
            else:
                df_processed = df.values
            
            # Make prediction
            prediction = self.model.predict(df_processed)
            
            # Get probability if available
            probability = None
            confidence = None
            if hasattr(self.model, "predict_proba"):
                try:
                    probabilities = self.model.predict_proba(df_processed)
                    if probabilities.ndim == 1:
                        probability = float(probabilities[0])
                    else:
                        probability = float(np.max(probabilities[0]))
                    confidence = float(np.max(probabilities[0]))
                except Exception as e:
                    logger.warning(f"Failed to get prediction probability: {{e}}")
            
            # Convert prediction to appropriate type
            if isinstance(prediction, np.ndarray):
                prediction_value = prediction[0]
            else:
                prediction_value = prediction
            
            return {{
                "prediction": float(prediction_value) if isinstance(prediction_value, (int, float, np.number)) else str(prediction_value),
                "probability": probability,
                "confidence": confidence,
                "model_name": self.metadata.get("model_name", "unknown"),
                "task_type": self.metadata.get("task_type", "unknown")
            }}
            
        except Exception as e:
            logger.error(f"Prediction error: {{e}}")
            raise
    
    def predict_batch(self, features_list: List[Dict[str, Union[float, int, str]]]) -> List[Dict[str, Any]]:
        """
        Make batch predictions.
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            List of prediction results
        """
        if not self.model or not self.preprocessing_pipeline:
            raise ValueError("Model or preprocessing pipeline not loaded")
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(features_list)
            
            # Ensure all required features are present
            feature_names = self.metadata.get("feature_names", [])
            missing_features = set(feature_names) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {{list(missing_features)}}")
            
            # Reorder columns to match training data
            df = df[feature_names]
            
            # Apply preprocessing
            if hasattr(self.preprocessing_pipeline, 'transform'):
                df_processed = self.preprocessing_pipeline.transform(df)
            else:
                df_processed = df.values
            
            # Make predictions
            predictions = self.model.predict(df_processed)
            
            # Get probabilities if available
            probabilities = None
            if hasattr(self.model, "predict_proba"):
                try:
                    probabilities = self.model.predict_proba(df_processed)
                except Exception as e:
                    logger.warning(f"Failed to get prediction probabilities: {{e}}")
            
            # Format responses
            results = []
            for i, prediction in enumerate(predictions):
                probability = None
                confidence = None
                
                if probabilities is not None:
                    if probabilities.ndim == 1:
                        probability = float(probabilities[i])
                    else:
                        probability = float(np.max(probabilities[i]))
                    confidence = float(np.max(probabilities[i]))
                
                results.append({{
                    "prediction": float(prediction) if isinstance(prediction, (int, float, np.number)) else str(prediction),
                    "probability": probability,
                    "confidence": confidence,
                    "model_name": self.metadata.get("model_name", "unknown"),
                    "task_type": self.metadata.get("task_type", "unknown")
                }})
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction error: {{e}}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary containing model information
        """
        if not self.metadata:
            raise ValueError("Metadata not loaded")
        
        return {{
            "model_name": self.metadata.get("model_name", "unknown"),
            "task_type": self.metadata.get("task_type", "unknown"),
            "feature_names": self.metadata.get("feature_names", []),
            "performance_metrics": self.metadata.get("performance_metrics", {{}}),
            "meta_features": self.metadata.get("meta_features", {{}})
        }}
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names.
        
        Returns:
            List of feature names
        """
        if not self.metadata:
            raise ValueError("Metadata not loaded")
        
        return self.metadata.get("feature_names", [])
'''
        
        setup_code = f'''"""
Setup script for {package_name} package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="{package_name}",
    version="{self.config.api_version}",
    author="AutoML Pipeline Generator",
    author_email="",
    description="{self.config.description}",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
    ],
)
'''
        
        requirements = '''pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
'''
        
        return {
            "init": init_code,
            "predictor": predictor_code,
            "setup": setup_code,
            "requirements": requirements
        }
    
    def _export_docker_service(
        self, 
        artifact: PipelineArtifact, 
        export_dir: Path,
        llm_client = None
    ) -> str:
        """Export pipeline as Docker service."""
        
        docker_dir = export_dir / "docker_service"
        docker_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy FastAPI service
        fastapi_dir = export_dir / "fastapi_service"
        if fastapi_dir.exists():
            shutil.copytree(fastapi_dir, docker_dir, dirs_exist_ok=True)
        
        # Generate docker-compose.yml
        docker_compose = f'''version: '3.8'

services:
  {self.config.service_name}:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
    volumes:
      - ./models:/app/models:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - {self.config.service_name}
    restart: unless-stopped
'''
        
        docker_compose_file = docker_dir / "docker-compose.yml"
        with open(docker_compose_file, "w") as f:
            f.write(docker_compose)
        
        # Generate nginx configuration
        nginx_config = '''events {
    worker_connections 1024;
}

http {
    upstream ml_service {
        server ml_pipeline_service:8000;
    }

    server {
        listen 80;
        
        location / {
            proxy_pass http://ml_service;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
'''
        
        nginx_file = docker_dir / "nginx.conf"
        with open(nginx_file, "w") as f:
            f.write(nginx_config)
        
        return str(docker_dir)
    
    def _generate_deployment_scripts(
        self, 
        artifact: PipelineArtifact, 
        export_dir: Path,
        llm_client = None
    ) -> Dict[str, str]:
        """Generate deployment scripts."""
        
        scripts = {}
        
        # Generate deployment script
        deploy_script = f'''#!/bin/bash

# Deployment script for {self.config.service_name}
# Generated on: {datetime.now().isoformat()}

set -e

echo "Deploying {self.config.service_name}..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Build and start services
echo "Building Docker images..."
docker-compose build

echo "Starting services..."
docker-compose up -d

# Wait for service to be ready
echo "Waiting for service to be ready..."
sleep 10

# Health check
echo "Performing health check..."
if curl -f http://localhost:8000/health; then
    echo "Service is healthy and ready!"
    echo "API documentation available at: http://localhost:8000/docs"
    echo "Health check endpoint: http://localhost:8000/health"
else
    echo "Health check failed. Check logs with: docker-compose logs"
    exit 1
fi

echo "Deployment completed successfully!"
'''
        
        deploy_file = export_dir / "deploy.sh"
        with open(deploy_file, "w") as f:
            f.write(deploy_script)
        deploy_file.chmod(0o755)
        scripts["deploy"] = str(deploy_file)
        
        # Generate test script
        test_script = f'''#!/usr/bin/env python3

"""
Test script for {self.config.service_name}
Generated on: {datetime.now().isoformat()}
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    try:
        response = requests.get(f"{{BASE_URL}}/health")
        response.raise_for_status()
        print("✓ Health check passed")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {{e}}")
        return False

def test_model_info():
    """Test model info endpoint."""
    try:
        response = requests.get(f"{{BASE_URL}}/model/info")
        response.raise_for_status()
        info = response.json()
        print(f"✓ Model info retrieved: {{info.get('model_name', 'unknown')}}")
        return True
    except Exception as e:
        print(f"✗ Model info failed: {{e}}")
        return False

def test_prediction():
    """Test prediction endpoint."""
    try:
        # Get feature names
        response = requests.get(f"{{BASE_URL}}/model/info")
        response.raise_for_status()
        info = response.json()
        feature_names = info.get("feature_names", [])
        
        if not feature_names:
            print("✗ No feature names available")
            return False
        
        # Create test data
        test_data = {{}}
        for feature in feature_names[:5]:  # Test with first 5 features
            test_data[feature] = 0.0
        
        # Make prediction
        response = requests.post(
            f"{{BASE_URL}}/predict",
            json={{"features": test_data}}
        )
        response.raise_for_status()
        result = response.json()
        print(f"✓ Prediction successful: {{result.get('prediction', 'unknown')}}")
        return True
    except Exception as e:
        print(f"✗ Prediction failed: {{e}}")
        return False

def main():
    """Run all tests."""
    print("Testing {self.config.service_name}...")
    
    tests = [
        test_health,
        test_model_info,
        test_prediction
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\\nTests passed: {{passed}}/{{len(tests)}}")
    
    if passed == len(tests):
        print("All tests passed! ✓")
        sys.exit(0)
    else:
        print("Some tests failed! ✗")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        test_file = export_dir / "test_service.py"
        with open(test_file, "w") as f:
            f.write(test_script)
        test_file.chmod(0o755)
        scripts["test"] = str(test_file)
        
        return scripts
    
    def _compress_export(self, export_dir: Path) -> Path:
        """Compress the export directory."""
        
        compressed_file = export_dir.parent / f"{export_dir.name}.zip"
        
        with zipfile.ZipFile(compressed_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in export_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(export_dir)
                    zipf.write(file_path, arcname)
        
        return compressed_file
