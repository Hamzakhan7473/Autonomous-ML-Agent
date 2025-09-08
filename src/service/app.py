"""FastAPI service for the Autonomous ML Agent."""

import os
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import time
import json

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from ..core.orchestrator import AutonomousMLAgent, PipelineConfig, PipelineResults
from ..core.ingest import analyze_data
from ..utils.llm_client import LLMClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Autonomous ML Agent API",
    description="AI-powered autonomous machine learning pipeline with LLM orchestration",
    version="1.0.0",
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

# Global variables
active_agents = {}
llm_client = None

# Pydantic models
class DatasetInfo(BaseModel):
    """Dataset information model."""
    filename: str
    target_column: str
    shape: tuple
    columns: List[str]
    target_type: str
    missing_percentage: float


class PipelineRequest(BaseModel):
    """Pipeline execution request model."""
    dataset_path: str
    target_column: str
    time_budget: int = 3600
    optimization_metric: str = "auto"
    random_state: int = 42
    output_dir: str = "./results"
    save_models: bool = True
    save_results: bool = True
    verbose: bool = False


class PipelineResponse(BaseModel):
    """Pipeline execution response model."""
    task_id: str
    status: str
    message: str
    estimated_time: Optional[int] = None


class PredictionRequest(BaseModel):
    """Prediction request model."""
    task_id: str
    features: List[List[float]]
    feature_names: Optional[List[str]] = None


class PredictionResponse(BaseModel):
    """Prediction response model."""
    predictions: List[Union[float, int]]
    probabilities: Optional[List[List[float]]] = None
    confidence: Optional[float] = None


class TaskStatus(BaseModel):
    """Task status model."""
    task_id: str
    status: str
    progress: float
    message: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    global llm_client
    
    try:
        llm_client = LLMClient()
        logger.info("LLM client initialized successfully")
    except Exception as e:
        logger.warning(f"LLM client initialization failed: {e}")
        llm_client = None


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "llm_client_available": llm_client is not None
    }


# Dataset analysis endpoint
@app.post("/analyze", response_model=DatasetInfo)
async def analyze_dataset(file: UploadFile = File(...)):
    """Analyze uploaded dataset."""
    
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Analyze dataset
        df, schema, summary = analyze_data(temp_path, "target")  # Will need to detect target column
        
        # Clean up temp file
        os.remove(temp_path)
        
        return DatasetInfo(
            filename=file.filename,
            target_column="target",  # This should be detected automatically
            shape=df.shape,
            columns=df.columns.tolist(),
            target_type=schema.target_type,
            missing_percentage=schema.missing_percentage
        )
        
    except Exception as e:
        logger.error(f"Dataset analysis failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Pipeline execution endpoint
@app.post("/pipeline/run", response_model=PipelineResponse)
async def run_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """Run the autonomous ML pipeline."""
    
    try:
        # Generate task ID
        task_id = f"task_{int(time.time())}"
        
        # Create pipeline configuration
        config = PipelineConfig(
            time_budget=request.time_budget,
            optimization_metric=request.optimization_metric,
            random_state=request.random_state,
            output_dir=request.output_dir,
            save_models=request.save_models,
            save_results=request.save_results,
            verbose=request.verbose
        )
        
        # Initialize agent
        agent = AutonomousMLAgent(config, llm_client)
        active_agents[task_id] = {
            "agent": agent,
            "status": "running",
            "start_time": time.time(),
            "config": config,
            "request": request
        }
        
        # Run pipeline in background
        background_tasks.add_task(run_pipeline_background, task_id, request.dataset_path, request.target_column)
        
        return PipelineResponse(
            task_id=task_id,
            status="running",
            message="Pipeline started successfully",
            estimated_time=request.time_budget
        )
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Task status endpoint
@app.get("/pipeline/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get the status of a pipeline task."""
    
    if task_id not in active_agents:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = active_agents[task_id]
    
    if task_info["status"] == "running":
        # Check if task is still running
        elapsed_time = time.time() - task_info["start_time"]
        progress = min(elapsed_time / task_info["config"].time_budget, 1.0)
        
        return TaskStatus(
            task_id=task_id,
            status="running",
            progress=progress,
            message=f"Pipeline running... {elapsed_time:.1f}s elapsed"
        )
    
    elif task_info["status"] == "completed":
        return TaskStatus(
            task_id=task_id,
            status="completed",
            progress=1.0,
            message="Pipeline completed successfully",
            results=task_info.get("results")
        )
    
    elif task_info["status"] == "failed":
        return TaskStatus(
            task_id=task_id,
            status="failed",
            progress=0.0,
            message="Pipeline failed",
            error=task_info.get("error")
        )
    
    else:
        return TaskStatus(
            task_id=task_id,
            status="unknown",
            progress=0.0,
            message="Unknown status"
        )


# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    """Make predictions using a trained model."""
    
    if request.task_id not in active_agents:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = active_agents[request.task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Model not ready for predictions")
    
    try:
        agent = task_info["agent"]
        
        # Convert features to DataFrame
        if request.feature_names:
            df = pd.DataFrame(request.features, columns=request.feature_names)
        else:
            df = pd.DataFrame(request.features)
        
        # Make predictions
        predictions = agent.predict(df)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(agent.results.best_model, 'predict_proba'):
            try:
                probabilities = agent.predict_proba(df).tolist()
            except Exception as e:
                logger.warning(f"Could not get probabilities: {e}")
        
        # Calculate confidence (for classification)
        confidence = None
        if probabilities is not None and len(probabilities[0]) == 2:  # Binary classification
            confidence = max(probabilities[0])
        
        return PredictionResponse(
            predictions=predictions.tolist(),
            probabilities=probabilities,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model information endpoint
@app.get("/model/info/{task_id}")
async def get_model_info(task_id: str):
    """Get information about a trained model."""
    
    if task_id not in active_agents:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = active_agents[task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Model not ready")
    
    try:
        agent = task_info["agent"]
        summary = agent.get_model_summary()
        
        return {
            "task_id": task_id,
            "model_info": summary,
            "feature_importance": agent.get_feature_importance().tolist() if agent.get_feature_importance() is not None else None,
            "preprocessing_config": agent.preprocessor.config.__dict__ if agent.preprocessor else None
        }
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Results download endpoint
@app.get("/results/{task_id}")
async def download_results(task_id: str):
    """Download pipeline results."""
    
    if task_id not in active_agents:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = active_agents[task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Results not ready")
    
    try:
        results_path = Path(task_info["config"].output_dir) / "pipeline_results.json"
        
        if not results_path.exists():
            raise HTTPException(status_code=404, detail="Results file not found")
        
        return FileResponse(
            path=str(results_path),
            filename=f"results_{task_id}.json",
            media_type="application/json"
        )
        
    except Exception as e:
        logger.error(f"Failed to download results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# List active tasks endpoint
@app.get("/tasks")
async def list_tasks():
    """List all active tasks."""
    
    tasks = []
    for task_id, task_info in active_agents.items():
        tasks.append({
            "task_id": task_id,
            "status": task_info["status"],
            "start_time": task_info["start_time"],
            "elapsed_time": time.time() - task_info["start_time"]
        })
    
    return {"tasks": tasks}


# Background task function
async def run_pipeline_background(task_id: str, dataset_path: str, target_column: str):
    """Run pipeline in background."""
    
    try:
        task_info = active_agents[task_id]
        agent = task_info["agent"]
        
        # Run pipeline
        results = agent.run(dataset_path, target_column)
        
        # Update task status
        task_info["status"] = "completed"
        task_info["results"] = results.__dict__
        
        logger.info(f"Pipeline {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline {task_id} failed: {e}")
        task_info["status"] = "failed"
        task_info["error"] = str(e)


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
