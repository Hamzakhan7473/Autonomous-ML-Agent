# 🏗️ System Architecture

## **Overview**

The Autonomous ML Agent follows a modular, microservices architecture designed for scalability, maintainability, and extensibility. The system is built around three main layers: **Frontend**, **Backend API**, and **ML Pipeline**.

## **High-Level Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                     │
├─────────────────────────────────────────────────────────────────┤
│  Next.js Frontend (React + TypeScript + Tailwind CSS)          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │   Landing   │ │  Dashboard  │ │   Upload    │ │  Results    │ │
│  │    Page     │ │   Component │ │ Component   │ │ Visualization│ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  FastAPI Backend (Python + Pydantic + Uvicorn)                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │   Health    │ │   Upload    │ │  Pipeline   │ │  Prediction │ │
│  │   Check     │ │   Handler   │ │  Runner     │ │  Service    │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ML Pipeline Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │   Data      │ │   LLM       │ │   Model     │ │  Evaluation │ │
│  │ Processing  │ │ Orchestrator│ │  Training   │ │   & Metrics │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     External Services                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │  OpenRouter │ │   OpenAI    │ │ Anthropic   │ │   Gemini    │ │
│  │     API     │ │     API     │ │     API     │ │     API     │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## **Core Components**

### **1. Frontend Layer (Next.js)**

The frontend is built with modern web technologies for optimal user experience:

#### **Architecture**
```
frontend/src/
├── app/                    # Next.js App Router
│   ├── globals.css        # Global styles
│   ├── layout.tsx         # Root layout
│   └── page.tsx           # Landing page
├── components/            # Reusable components
│   ├── dashboard/         # Dashboard-specific components
│   ├── landing/           # Landing page components
│   ├── layout/            # Layout components
│   └── ui/               # Base UI components
└── lib/                  # Utilities and configuration
    ├── api.ts            # API client
    ├── config.ts         # App configuration
    └── types.ts          # TypeScript definitions
```

#### **Key Features**
- **Real-time Updates**: Live pipeline progress tracking
- **File Upload**: Drag-and-drop with auto-detection
- **Responsive Design**: Mobile-first approach
- **Type Safety**: Full TypeScript coverage
- **State Management**: React hooks and context

### **2. Backend API Layer (FastAPI)**

The backend provides a robust REST API with async processing:

#### **Architecture**
```
src/service/
├── app.py                 # Main FastAPI application
├── docker/               # Containerization
└── middleware/           # Custom middleware (future)
```

#### **Key Features**
- **Async Processing**: Background task execution
- **Task Persistence**: File-based task storage
- **Error Handling**: Comprehensive error management
- **CORS Support**: Cross-origin resource sharing
- **Auto Documentation**: OpenAPI/Swagger docs

### **3. ML Pipeline Layer (Python)**

The core ML functionality is organized into specialized modules:

#### **Architecture**
```
src/
├── core/                 # Core ML pipeline components
│   ├── orchestrator.py   # Main pipeline orchestrator
│   ├── ingest.py        # Data ingestion and analysis
│   ├── models.py        # Model definitions
│   └── evaluate.py      # Model evaluation
├── agent_llm/           # LLM integration layer
│   └── planner.py       # LLM-powered planning
├── data/                # Data processing
│   ├── ingestion.py     # Data loading utilities
│   ├── preprocessing.py # Data preprocessing
│   └── meta_features.py # Feature extraction
├── models/              # ML algorithms
│   ├── algorithms.py    # Algorithm implementations
│   ├── ensemble.py      # Ensemble methods
│   └── hyperopt.py      # Hyperparameter optimization
├── evaluation/          # Model evaluation
│   ├── metrics.py       # Evaluation metrics
│   └── leaderboard.py   # Model comparison
└── utils/               # Utilities
    └── llm_client.py    # LLM client abstraction
```

## **Data Flow Architecture**

### **1. Request Flow**
```
User Upload → Frontend → Backend API → ML Pipeline → LLM Services
     ↓              ↓           ↓            ↓            ↓
File Storage → Task Queue → Background → Model Training → Results
     ↓              ↓           ↓            ↓            ↓
Auto-detect → Progress → Status Update → Evaluation → Frontend
```

### **2. Pipeline Execution Flow**
```
Data Ingestion → LLM Planning → Preprocessing → Model Training → Evaluation
      ↓              ↓              ↓              ↓              ↓
  Schema Analysis → Execution Plan → Feature Eng. → Hyperopt → Metrics
      ↓              ↓              ↓              ↓              ↓
  Meta-features → Model Selection → Scaling → Ensemble → Insights
```

## **Component Interactions**

### **1. Frontend-Backend Communication**

```typescript
// Frontend API Client
const apiClient = {
  uploadFile: (file: File) => POST('/analyze', formData),
  runPipeline: (config: PipelineConfig) => POST('/pipeline/run', config),
  getTaskStatus: (taskId: string) => GET(`/pipeline/status/${taskId}`),
  makePrediction: (request: PredictionRequest) => POST('/predict', request)
}
```

### **2. Backend-ML Pipeline Communication**

```python
# Backend Service
class PipelineService:
    def __init__(self):
        self.llm_client = LLMClient(primary_provider="openrouter")
        self.agent = AutonomousMLAgent(config, self.llm_client)
    
    async def run_pipeline(self, request: PipelineRequest):
        # Background task execution
        return await self.agent.run(
            request.dataset_path, 
            request.target_column
        )
```

### **3. ML Pipeline-LLM Communication**

```python
# LLM Orchestrator
class LLMOrchestrator:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.planner = MLPlanner(llm_client)
    
    async def create_execution_plan(self, schema, summary):
        return await self.planner.create_plan(schema, summary)
```

## **Technology Stack**

### **Frontend Technologies**
- **Next.js 15**: React framework with App Router
- **TypeScript**: Type-safe JavaScript development
- **Tailwind CSS**: Utility-first CSS framework
- **Framer Motion**: Animation and gesture library
- **React Hooks**: State management and side effects

### **Backend Technologies**
- **FastAPI**: Modern, fast web framework for APIs
- **Python 3.10+**: Core programming language
- **Pydantic**: Data validation using Python type hints
- **Uvicorn**: Lightning-fast ASGI server
- **Pandas**: Data manipulation and analysis

### **ML Technologies**
- **Scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Light gradient boosting machine
- **CatBoost**: Categorical boosting
- **Joblib**: Model serialization

### **LLM Integration**
- **OpenRouter**: Unified LLM API access
- **OpenAI**: GPT models
- **Anthropic**: Claude models
- **Google**: Gemini models

### **Infrastructure**
- **Docker**: Containerization platform
- **SQLite**: Lightweight database
- **GitHub Actions**: CI/CD pipeline
- **Nginx**: Reverse proxy (production)

## **Design Patterns**

### **1. Microservices Architecture**
- **Separation of Concerns**: Each layer has distinct responsibilities
- **Loose Coupling**: Components communicate through well-defined interfaces
- **Independent Scaling**: Each service can be scaled independently

### **2. Event-Driven Architecture**
- **Asynchronous Processing**: Background task execution
- **Event Sourcing**: Task state changes are tracked
- **Message Passing**: Components communicate through events

### **3. Plugin Architecture**
- **Extensible Models**: Easy addition of new ML algorithms
- **LLM Provider Abstraction**: Support for multiple LLM providers
- **Configurable Pipeline**: Dynamic pipeline configuration

### **4. Repository Pattern**
- **Data Access Abstraction**: Consistent data access interface
- **Model Registry**: Centralized model storage and retrieval
- **Task Persistence**: File-based task state management

## **Security Architecture**

### **1. API Security**
- **Input Validation**: Pydantic models for request validation
- **CORS Configuration**: Controlled cross-origin access
- **Error Handling**: Secure error messages without information leakage

### **2. Data Security**
- **File Upload Validation**: File type and size restrictions
- **Temporary Storage**: Secure file handling
- **API Key Management**: Environment variable configuration

### **3. Infrastructure Security**
- **Container Security**: Docker best practices
- **Network Security**: Internal service communication
- **Secret Management**: Secure API key storage

## **Scalability Considerations**

### **1. Horizontal Scaling**
- **Stateless Services**: No server-side session storage
- **Load Balancing**: Multiple backend instances
- **Database Sharding**: Distributed data storage

### **2. Performance Optimization**
- **Async Processing**: Non-blocking I/O operations
- **Caching Strategy**: Result caching and memoization
- **Resource Management**: Memory and CPU optimization

### **3. Monitoring and Observability**
- **Health Checks**: Service health monitoring
- **Logging**: Structured logging throughout the system
- **Metrics**: Performance and usage metrics

## **Deployment Architecture**

### **Development Environment**
```
┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │
│   (npm run dev) │    │   (uvicorn)     │
│   Port: 3000    │    │   Port: 8000    │
└─────────────────┘    └─────────────────┘
```

### **Production Environment**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx         │    │   FastAPI       │    │   Database      │
│   (Load Balancer)│    │   (Multiple)    │    │   (PostgreSQL)  │
│   Port: 80/443  │    │   Port: 8000    │    │   Port: 5432    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## **Future Architecture Enhancements**

### **1. Advanced Features**
- **WebSocket Support**: Real-time bidirectional communication
- **Distributed Computing**: Multi-node ML pipeline execution
- **Model Versioning**: Advanced model registry with versioning

### **2. Scalability Improvements**
- **Message Queue**: Redis/RabbitMQ for task queuing
- **Microservices**: Split into smaller, focused services
- **Kubernetes**: Container orchestration platform

### **3. Integration Enhancements**
- **MLflow Integration**: Advanced experiment tracking
- **Cloud Providers**: AWS/Azure/GCP integration
- **CI/CD Pipeline**: Automated testing and deployment

---

This architecture provides a solid foundation for the Autonomous ML Agent while maintaining flexibility for future enhancements and scalability requirements.

