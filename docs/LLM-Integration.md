# 🤖 LLM Integration

## **Overview**

The Autonomous ML Agent leverages Large Language Models (LLMs) to orchestrate the entire machine learning pipeline. This integration enables intelligent decision-making, code generation, and automated optimization throughout the ML workflow.

## **Supported LLM Providers**

### **E2B (Recommended)**
- **Provider**: E2B API
- **Models**: Access to multiple LLM providers through E2B sandbox
- **Benefits**: Secure code execution, sandboxed environment, unified API
- **Setup**: Requires `E2B_API_KEY`

### **OpenAI**
- **Provider**: OpenAI API
- **Models**: GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
- **Benefits**: High-quality responses, reliable service
- **Setup**: Requires `OPENAI_API_KEY`

### **Anthropic**
- **Provider**: Anthropic API
- **Models**: Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku
- **Benefits**: Excellent reasoning capabilities, long context
- **Setup**: Requires `ANTHROPIC_API_KEY`

### **Google Gemini**
- **Provider**: Google AI API
- **Models**: Gemini Pro, Gemini Ultra
- **Benefits**: Multimodal capabilities, competitive pricing
- **Setup**: Requires `GEMINI_API_KEY`

## **Architecture**

### **LLM Client Abstraction Layer**

```python
# src/utils/llm_client.py
class BaseLLMClient(ABC):
    """Base class for LLM clients."""
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def generate_structured_response(self, prompt: str, schema: dict, **kwargs) -> dict:
        """Generate a structured response following a schema."""
        pass
```

### **Unified LLM Client**

```python
class LLMClient:
    """Unified LLM client with fallback support."""
    
    def __init__(self, primary_provider: str = "openrouter", **kwargs):
        self.primary_provider = primary_provider
        self.clients = {}
        self._initialize_clients(**kwargs)
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response with fallback support."""
        try:
            return self.clients[self.primary_provider].generate_response(prompt, **kwargs)
        except Exception as e:
            logger.warning(f"Primary provider failed: {e}")
            # Try fallback providers
            for provider, client in self.clients.items():
                if provider != self.primary_provider:
                    try:
                        return client.generate_response(prompt, **kwargs)
                    except Exception:
                        continue
            raise Exception("All LLM providers failed")
```

## **LLM Integration Points**

### **1. Data Analysis and Planning**

The LLM analyzes dataset characteristics and creates execution plans:

```python
# src/agent_llm/planner.py
class MLPlanner:
    """LLM-powered ML pipeline planner."""
    
    def create_plan(self, schema: DatasetSchema, summary: dict, prior_runs: list = None) -> MLPlan:
        """Create execution plan using LLM."""
        
        prompt = f"""
        Analyze this dataset and create an ML execution plan:
        
        Dataset Schema:
        - Shape: {schema.shape}
        - Target type: {schema.target_type}
        - Missing values: {schema.missing_percentage}%
        - Categorical features: {schema.n_categorical}
        - Numerical features: {schema.n_features}
        
        Data Summary:
        {summary}
        
        Prior Runs (if available):
        {prior_runs}
        
        Create a comprehensive execution plan including:
        1. Preprocessing steps
        2. Model selection
        3. Hyperparameter strategies
        4. Evaluation approach
        """
        
        response = self.llm_client.generate_structured_response(prompt, ML_PLAN_SCHEMA)
        return MLPlan.from_dict(response)
```

### **2. Preprocessing Code Generation**

The LLM generates preprocessing code based on data characteristics:

```python
def generate_preprocessing_code(self, schema: DatasetSchema, summary: dict) -> str:
    """Generate preprocessing code using LLM."""
    
    prompt = f"""
    Generate Python code for preprocessing this dataset:
    
    Dataset characteristics:
    - Shape: {schema.shape}
    - Target type: {schema.target_type}
    - Missing values: {schema.missing_percentage}%
    - Categorical features: {schema.n_categorical}
    - Numerical features: {schema.n_features}
    
    Generate preprocessing code that handles:
    1. Missing value imputation
    2. Categorical encoding
    3. Feature scaling
    4. Outlier detection
    5. Feature engineering
    
    Return only the Python code, no explanations.
    """
    
    code = self.llm_client.generate_response(prompt)
    return self._extract_code_from_response(code)
```

### **3. Model Selection**

The LLM selects appropriate models based on data characteristics:

```python
def select_models(self, schema: DatasetSchema, summary: dict) -> list[str]:
    """Select models using LLM."""
    
    prompt = f"""
    Select the best machine learning models for this dataset:
    
    Dataset characteristics:
    - Shape: {schema.shape}
    - Target type: {schema.target_type}
    - Target distribution: {summary.get('target_distribution', 'Unknown')}
    - Feature types: {summary.get('feature_types', 'Unknown')}
    
    Available models:
    - Classification: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, kNN, MLP, SVM, Naive Bayes, LDA
    - Regression: Linear Regression, Random Forest, XGBoost, LightGBM, CatBoost, kNN, MLP, Ridge, Lasso
    
    Select the top 5 models and explain why each is suitable.
    """
    
    response = self.llm_client.generate_response(prompt)
    return self._parse_model_selection(response)
```

### **4. Hyperparameter Optimization Strategy**

The LLM designs optimization strategies:

```python
def create_optimization_strategy(self, model_name: str, schema: DatasetSchema, prior_runs: list) -> dict:
    """Create hyperparameter optimization strategy."""
    
    prompt = f"""
    Design a hyperparameter optimization strategy for {model_name}:
    
    Dataset characteristics:
    - Shape: {schema.shape}
    - Target type: {schema.target_type}
    - Feature count: {schema.n_features}
    
    Prior runs (if available):
    {prior_runs}
    
    Design an optimization strategy that:
    1. Uses meta-learning warm starts effectively
    2. Balances exploration vs exploitation
    3. Respects time constraints
    4. Focuses on the primary metric
    """
    
    response = self.llm_client.generate_structured_response(prompt, OPTIMIZATION_STRATEGY_SCHEMA)
    return response
```

### **5. Results Interpretation**

The LLM generates insights and explanations:

```python
def explain_results(self, results: dict, execution_plan: MLPlan) -> str:
    """Generate insights using LLM."""
    
    prompt = f"""
    Analyze these ML pipeline results and provide insights:
    
    Execution Plan:
    - Models tried: {execution_plan.models_to_try}
    - Preprocessing steps: {execution_plan.preprocessing_steps}
    - Optimization strategies: {execution_plan.hyperparameter_strategies}
    
    Results:
    - Best model: {results.get('best_model_name', 'Unknown')}
    - Best score: {results.get('best_score', 'Unknown')}
    - Execution time: {results.get('execution_time', 'Unknown')}
    - All results: {results.get('all_results', [])}
    
    Provide actionable insights about:
    1. Model performance patterns
    2. Feature importance insights
    3. Recommendations for improvement
    4. Deployment considerations
    """
    
    return self.llm_client.generate_response(prompt)
```

## **E2B Integration**

### **Setup and Configuration**

1. **Get API Key**: Visit [E2B](https://e2b.dev/) and create an API key
2. **Configure Environment**: Add to your `.env` file:
   ```bash
   E2B_API_KEY=your_e2b_api_key_here
   DEFAULT_LLM_PROVIDER=e2b
   ```

### **E2B Client Implementation**

```python
class E2BClient(BaseLLMClient):
    """E2B API client for secure code execution and LLM integration."""
    
    def __init__(self, api_key: str = None, template_id: str = "base"):
        self.api_key = api_key or os.getenv("E2B_API_KEY")
        self.template_id = template_id
        self.client = e2b.Client(self.api_key)
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        # Create sandbox environment
        sandbox = self.client.sandbox.create(template=self.template_id)
        
        # Execute LLM query in sandbox
        result = sandbox.run_code(f"""
        import openai
        client = openai.OpenAI()
        
        response = client.chat.completions.create(
            model="{kwargs.get('model', 'gpt-4o-mini')}",
            messages=[{{"role": "user", "content": "{prompt}"}}],
            max_tokens={kwargs.get("max_tokens", 2000)},
            temperature={kwargs.get("temperature", 0.1)}
        )
        print(response.choices[0].message.content)
        """)
        
        # Clean up sandbox
        sandbox.close()
        
        return result.stdout.strip()
```

### **Supported Models**

E2B provides access to models from multiple providers through secure sandbox execution:

#### **OpenAI Models**
- `gpt-4o` - Latest GPT-4 model
- `gpt-4o-mini` - Cost-effective GPT-4 variant
- `gpt-4-turbo` - High-performance model
- `gpt-3.5-turbo` - Fast and economical

#### **Anthropic Models**
- `claude-3-opus` - Most capable Claude model
- `claude-3-sonnet` - Balanced performance
- `claude-3-haiku` - Fast and efficient

#### **Google Models**
- `gemini-pro` - Google's flagship model
- `gemini-pro-vision` - Multimodal capabilities

#### **Meta Models**
- `llama-3-8b` - Open source option
- `llama-3-70b` - Larger open source model

### **Model Selection Guidelines**

```python
def select_model_for_task(task_type: str, complexity: str, budget: str) -> str:
    """Select appropriate model based on task requirements."""
    
    if budget == "low":
        return "gpt-4o-mini"  # Cost-effective
    elif complexity == "high":
        return "gpt-4o"  # Best quality
    elif task_type == "reasoning":
        return "claude-3-sonnet"  # Excellent reasoning
    elif task_type == "code_generation":
        return "gpt-4o"  # Best code generation
    else:
        return "gpt-4o-mini"  # Default
```

## **Prompt Engineering**

### **Best Practices**

1. **Clear Instructions**: Provide specific, actionable instructions
2. **Context Provision**: Include relevant dataset characteristics
3. **Format Specification**: Specify output format clearly
4. **Examples**: Provide examples when helpful
5. **Error Handling**: Include fallback instructions

### **Prompt Templates**

#### **Data Analysis Prompt**
```python
DATA_ANALYSIS_PROMPT = """
Analyze this dataset and provide insights:

Dataset Info:
- Shape: {shape}
- Target column: {target_column}
- Target distribution: {target_distribution}
- Meta-features: {meta_features}

Please provide:
1. Data quality assessment
2. Potential preprocessing needs
3. Feature engineering opportunities
4. Model selection recommendations
"""
```

#### **Model Selection Prompt**
```python
MODEL_SELECTION_PROMPT = """
Select the best machine learning models for this dataset:

Dataset characteristics:
- Shape: {shape}
- Target type: {target_type}
- Target distribution: {target_distribution}
- Meta-features: {meta_features}
- Optimization metric: {optimization_metric}

Available models: {available_models}

Select the top 5 models and explain why each is suitable.
"""
```

#### **Results Interpretation Prompt**
```python
RESULTS_INTERPRETATION_PROMPT = """
Analyze these ML pipeline results and provide insights:

Results summary:
- Total models trained: {total_models}
- Best model: {best_model}
- Best score: {best_score}
- Training time: {training_time}

Provide actionable insights about:
1. Model performance patterns
2. Feature importance insights
3. Recommendations for improvement
4. Deployment considerations
"""
```

## **Error Handling and Fallbacks**

### **Provider Fallback Strategy**

```python
def generate_response_with_fallback(self, prompt: str, **kwargs) -> str:
    """Generate response with automatic fallback."""
    
    providers = [self.primary_provider] + [p for p in self.clients.keys() if p != self.primary_provider]
    
    for provider in providers:
        try:
            return self.clients[provider].generate_response(prompt, **kwargs)
        except Exception as e:
            logger.warning(f"Provider {provider} failed: {e}")
            continue
    
    raise Exception("All LLM providers failed")
```

### **Retry Logic**

```python
def generate_response_with_retry(self, prompt: str, max_retries: int = 3, **kwargs) -> str:
    """Generate response with retry logic."""
    
    for attempt in range(max_retries):
        try:
            return self.generate_response(prompt, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
```

## **Performance Optimization**

### **Caching**

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_llm_response(self, prompt_hash: str, provider: str, model: str) -> str:
    """Cache LLM responses for identical prompts."""
    # Implementation here
```

### **Batch Processing**

```python
def batch_generate_responses(self, prompts: list[str], **kwargs) -> list[str]:
    """Generate multiple responses in batch."""
    # Implementation for batch processing
```

### **Streaming Responses**

```python
def stream_response(self, prompt: str, **kwargs):
    """Stream LLM response for long outputs."""
    # Implementation for streaming
```

## **Monitoring and Logging**

### **LLM Usage Tracking**

```python
class LLMUsageTracker:
    """Track LLM usage and costs."""
    
    def track_request(self, provider: str, model: str, tokens_used: int, cost: float):
        """Track LLM request metrics."""
        # Implementation here
    
    def get_usage_stats(self) -> dict:
        """Get usage statistics."""
        # Implementation here
```

### **Performance Metrics**

```python
class LLMPerformanceMonitor:
    """Monitor LLM performance metrics."""
    
    def track_response_time(self, provider: str, response_time: float):
        """Track response times."""
        # Implementation here
    
    def track_success_rate(self, provider: str, success: bool):
        """Track success rates."""
        # Implementation here
```

## **Testing LLM Integration**

### **Unit Tests**

```python
def test_llm_client_initialization():
    """Test LLM client initialization."""
    client = LLMClient(primary_provider="openrouter")
    assert "openrouter" in client.clients
    assert client.primary_provider == "openrouter"

def test_fallback_mechanism():
    """Test fallback mechanism."""
    client = LLMClient(primary_provider="openrouter")
    # Mock primary provider failure
    client.clients["openrouter"].generate_response = Mock(side_effect=Exception("API Error"))
    
    # Should fallback to other providers
    response = client.generate_response("Test prompt")
    assert response is not None
```

### **Integration Tests**

```python
def test_end_to_end_llm_pipeline():
    """Test complete LLM-powered pipeline."""
    # Test data analysis
    schema = DatasetSchema(shape=(1000, 10), target_type="classification")
    plan = planner.create_plan(schema, {})
    assert plan is not None
    
    # Test model selection
    models = planner.select_models(schema, {})
    assert len(models) > 0
    
    # Test results interpretation
    results = {"best_model": "RandomForest", "best_score": 0.95}
    insights = planner.explain_results(results, plan)
    assert len(insights) > 0
```

## **Best Practices**

### **1. Model Selection**
- Use cost-effective models for development
- Use high-quality models for production
- Consider task-specific model strengths

### **2. Prompt Engineering**
- Be specific and clear in instructions
- Provide relevant context
- Use structured output formats
- Include error handling instructions

### **3. Error Handling**
- Implement fallback mechanisms
- Use retry logic with exponential backoff
- Monitor and log errors
- Provide meaningful error messages

### **4. Performance**
- Cache frequent responses
- Use batch processing when possible
- Monitor response times and costs
- Optimize prompt length

### **5. Security**
- Never expose API keys in code
- Use environment variables
- Implement rate limiting
- Monitor usage for anomalies

---

This comprehensive LLM integration enables the Autonomous ML Agent to make intelligent decisions throughout the ML pipeline, from data analysis to model selection and results interpretation.

