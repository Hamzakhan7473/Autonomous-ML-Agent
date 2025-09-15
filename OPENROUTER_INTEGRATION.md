# 🚀 OpenRouter Integration Guide

## **Overview**

This document describes the integration of [OpenRouter](https://openrouter.ai/) into the Autonomous ML Agent system. OpenRouter provides access to multiple LLM providers through a single API, making it easier to switch between different models and providers.

## **What is OpenRouter?**

OpenRouter is a unified API that provides access to multiple LLM providers including:
- **OpenAI**: GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
- **Anthropic**: Claude 3 Sonnet, Claude 3 Haiku, Claude 3 Opus
- **Google**: Gemini Pro, Gemini Ultra
- **Meta**: Llama 2, Llama 3
- **And many more providers**

## **Benefits of OpenRouter Integration**

### **🎯 Cost Optimization**
- Access to competitive pricing across providers
- Easy model comparison and switching
- Transparent cost tracking

### **🔧 Simplified API Management**
- Single API key for multiple providers
- Unified authentication system
- Consistent response format

### **⚡ Performance & Reliability**
- Automatic failover between providers
- Latency optimization
- Uptime monitoring

### **🔒 Enhanced Security**
- Centralized API key management
- Usage tracking and limits
- Rate limiting and abuse prevention

## **Integration Details**

### **Files Modified**

1. **`src/utils/llm_client.py`**
   - Added `OpenRouterClient` class
   - Integrated with unified `LLMClient` system
   - Added model discovery functionality

2. **`src/service/app.py`**
   - Updated to prefer OpenRouter as default provider
   - Enhanced result serialization for better API responses

3. **`env.example`**
   - Added `OPENROUTER_API_KEY` configuration
   - Added `DEFAULT_LLM_PROVIDER` setting

4. **`API_KEYS_SETUP.md`**
   - Updated with OpenRouter setup instructions
   - Added testing commands

### **New Features**

#### **OpenRouterClient Class**
```python
class OpenRouterClient(BaseLLMClient):
    """OpenRouter API client for accessing multiple LLM providers."""
    
    def __init__(self, api_key: str | None = None, model: str = "openai/gpt-4o-mini"):
        # Initialize with OpenRouter API endpoint
        
    def generate_response(self, prompt: str, **kwargs: Any) -> str:
        # Generate responses using OpenRouter API
        
    def generate_structured_response(self, prompt: str, schema: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        # Generate structured JSON responses
        
    def get_available_models(self) -> list[dict[str, Any]]:
        # Get list of available models from OpenRouter
```

#### **Supported Models**
The OpenRouter client supports a wide range of models:

**OpenAI Models:**
- `openai/gpt-4o` - Latest GPT-4 model
- `openai/gpt-4o-mini` - Cost-effective GPT-4 variant
- `openai/gpt-4-turbo` - High-performance model
- `openai/gpt-3.5-turbo` - Fast and economical

**Anthropic Models:**
- `anthropic/claude-3-opus` - Most capable Claude model
- `anthropic/claude-3-sonnet` - Balanced performance
- `anthropic/claude-3-haiku` - Fast and efficient

**Google Models:**
- `google/gemini-pro` - Google's flagship model
- `google/gemini-pro-vision` - Multimodal capabilities

**Meta Models:**
- `meta-llama/llama-3-8b` - Open source option
- `meta-llama/llama-3-70b` - Larger open source model

## **Setup Instructions**

### **Step 1: Get OpenRouter API Key**

1. Visit [OpenRouter](https://openrouter.ai/)
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key for your `.env` file

### **Step 2: Configure Environment**

Add to your `.env` file:
```env
# OpenRouter API Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here
DEFAULT_LLM_PROVIDER=openrouter
```

### **Step 3: Test Integration**

```bash
# Test OpenRouter integration
python3 test_openrouter.py

# Test specific models
python3 -c "
from src.utils.llm_client import OpenRouterClient
client = OpenRouterClient(model='openai/gpt-4o-mini')
response = client.generate_response('Hello, world!')
print(response)
"
```

## **Usage Examples**

### **Basic Usage**
```python
from src.utils.llm_client import LLMClient

# Initialize with OpenRouter as primary provider
client = LLMClient(primary_provider="openrouter")

# Generate response
response = client.generate_response(
    "Explain machine learning in simple terms",
    max_tokens=200
)
print(response)
```

### **Model Selection**
```python
from src.utils.llm_client import OpenRouterClient

# Use specific model
client = OpenRouterClient(model="anthropic/claude-3-sonnet")

# Get available models
models = client.get_available_models()
for model in models[:5]:
    print(f"- {model['id']}: ${model.get('pricing', {}).get('prompt', 'N/A')}/1M tokens")
```

### **Structured Responses**
```python
# Define schema for ML pipeline configuration
schema = {
    "type": "object",
    "properties": {
        "algorithm": {"type": "string"},
        "hyperparameters": {"type": "object"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["algorithm", "hyperparameters", "confidence"]
}

# Generate structured response
response = client.generate_structured_response(
    "Recommend a machine learning algorithm for binary classification",
    schema
)
print(response)
```

## **Configuration Options**

### **Environment Variables**

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key | Required |
| `DEFAULT_LLM_PROVIDER` | Primary LLM provider | `openrouter` |

### **Model Parameters**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | Model identifier | `openai/gpt-4o-mini` |
| `max_tokens` | Maximum response tokens | `2000` |
| `temperature` | Response randomness | `0.1` |
| `timeout` | Request timeout (seconds) | `30` |

### **Headers**

The OpenRouter client automatically includes:
- `HTTP-Referer`: Your site URL for rankings
- `X-Title`: Your application title
- `Authorization`: Bearer token with API key

## **Testing & Validation**

### **Test Suite**
The `test_openrouter.py` script provides comprehensive testing:

```bash
python3 test_openrouter.py
```

**Test Coverage:**
- ✅ Direct OpenRouter client functionality
- ✅ Unified LLM client integration
- ✅ Structured response generation
- ✅ Model discovery and listing
- ✅ Error handling and fallbacks

### **Integration Tests**
```bash
# Test with ML pipeline
curl -X POST "http://localhost:8000/pipeline/run" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "data/raw/test_final.csv",
    "target_column": "",
    "time_budget": 300,
    "optimization_metric": "accuracy"
  }'
```

## **Best Practices**

### **Model Selection**
- **Development**: Use `openai/gpt-4o-mini` for cost-effective testing
- **Production**: Use `openai/gpt-4o` or `anthropic/claude-3-sonnet` for best quality
- **Specialized Tasks**: Choose models optimized for specific domains

### **Cost Management**
- Monitor usage through OpenRouter dashboard
- Set spending limits for API keys
- Use appropriate models for task complexity

### **Error Handling**
- Implement fallback to other providers
- Handle rate limiting gracefully
- Monitor API health and availability

### **Security**
- Never commit API keys to version control
- Use environment variables for configuration
- Rotate API keys regularly
- Monitor usage for anomalies

## **Troubleshooting**

### **Common Issues**

**1. Authentication Errors**
```
❌ OpenRouter API error: 401 Unauthorized
```
**Solution**: Verify your API key is correct and active

**2. Model Not Found**
```
❌ Model 'invalid-model' not found
```
**Solution**: Check available models with `client.get_available_models()`

**3. Rate Limiting**
```
❌ Rate limit exceeded
```
**Solution**: Implement exponential backoff or switch to another provider

**4. Network Issues**
```
❌ Connection timeout
```
**Solution**: Increase timeout or check network connectivity

### **Debug Mode**
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

client = LLMClient(primary_provider="openrouter")
```

## **Performance Metrics**

### **Expected Performance**
- **Response Time**: 1-5 seconds (depending on model)
- **Throughput**: 100-1000 requests/minute (depending on plan)
- **Availability**: 99.9% uptime SLA

### **Cost Comparison**
| Model | Cost per 1M tokens | Best For |
|-------|-------------------|----------|
| `openai/gpt-4o-mini` | $0.15 | Development, testing |
| `openai/gpt-4o` | $5.00 | Production, high quality |
| `anthropic/claude-3-haiku` | $0.25 | Fast responses |
| `anthropic/claude-3-sonnet` | $3.00 | Balanced performance |

## **Future Enhancements**

### **Planned Features**
- [ ] Automatic model selection based on task type
- [ ] Cost optimization recommendations
- [ ] Advanced caching strategies
- [ ] Multi-model ensemble responses
- [ ] Real-time performance monitoring

### **Integration Roadmap**
- [ ] Add support for streaming responses
- [ ] Implement function calling capabilities
- [ ] Add multimodal model support
- [ ] Integrate with MLflow for experiment tracking

## **Support & Resources**

### **Documentation**
- [OpenRouter API Docs](https://openrouter.ai/docs/api-reference/authentication)
- [Available Models](https://openrouter.ai/models)
- [Pricing Information](https://openrouter.ai/pricing)

### **Community**
- [OpenRouter Discord](https://discord.gg/openrouter)
- [GitHub Issues](https://github.com/openrouter-ai/openrouter/issues)

### **Support**
- Email: support@openrouter.ai
- Discord: #support channel

---

## **Conclusion**

The OpenRouter integration significantly enhances the Autonomous ML Agent's capabilities by providing:

1. **Unified Access** to multiple LLM providers
2. **Cost Optimization** through competitive pricing
3. **Simplified Management** with single API key
4. **Enhanced Reliability** with automatic failover
5. **Future-Proof Architecture** for easy model switching

This integration makes the system more robust, cost-effective, and maintainable while providing access to the latest and most capable language models available.

**Ready to get started?** Follow the setup instructions above and run `python3 test_openrouter.py` to verify your integration!


