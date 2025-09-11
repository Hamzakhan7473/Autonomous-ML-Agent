# 🔐 API Keys Setup Guide

## **Step 1: Create .env File**

Create a `.env` file in your project root directory:

```bash
# In your terminal, run:
touch .env
```

## **Step 2: Add Your API Keys**

Edit the `.env` file and add your actual API keys:

```env
# Google Cloud Gemini API
GOOGLE_API_KEY=your_actual_google_api_key_here

# OpenAI API  
OPENAI_API_KEY=your_actual_openai_api_key_here

# Anthropic API
ANTHROPIC_API_KEY=your_actual_anthropic_api_key_here

# Other configurations
MLFLOW_TRACKING_URI=sqlite:///meta/mlflow.db
DEFAULT_MODEL_PROVIDER=gemini
```

## **Step 3: Get Your API Keys**

### **Google Gemini API Key:**
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click "Get API Key"
4. Create a new API key
5. Copy the key to your `.env` file

### **OpenAI API Key:**
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign in to your account
3. Go to API Keys section
4. Create a new secret key
5. Copy the key to your `.env` file

### **Anthropic API Key:**
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Sign in to your account
3. Go to API Keys section
4. Create a new API key
5. Copy the key to your `.env` file

## **Step 4: Test Your Setup**

```bash
# Test Gemini
python -m src.cli test-llm --provider gemini

# Test all providers
python -m src.cli test-llm --provider openai
python -m src.cli test-llm --provider anthropic

# Run full demo
python examples/gemini_integration_demo.py
```

## **Security Best Practices**

### **✅ DO:**
- Use `.env` file for local development
- Add `.env` to `.gitignore` (already done)
- Use environment variables in production
- Rotate your API keys regularly
- Use different keys for different environments

### **❌ DON'T:**
- Never commit API keys to version control
- Don't share API keys in chat/email
- Don't hardcode keys in your code
- Don't use the same key for all environments

## **File Structure**

```
autonomous_ml_agent/
├── .env                    # Your API keys (DO NOT COMMIT)
├── .env.template          # Template file (safe to commit)
├── .gitignore             # Already includes .env
├── src/
└── examples/
```

## **Troubleshooting**

### **If API keys aren't loaded:**
```bash
# Check if .env file exists
ls -la .env

# Check if keys are loaded
python -c "import os; print('GOOGLE_API_KEY:', 'SET' if os.getenv('GOOGLE_API_KEY') else 'NOT SET')"
```

### **If you get authentication errors:**
1. Verify your API key is correct
2. Check if the key has proper permissions
3. Ensure you're using the right environment (dev/prod)
4. Check if your account has sufficient credits/quota

## **Production Deployment**

For production, use environment variables or secure secret management:

```bash
# Example for production
export GOOGLE_API_KEY="prod_key_here"
export OPENAI_API_KEY="prod_key_here"
export ANTHROPIC_API_KEY="prod_key_here"
```

Or use cloud secret management services like:
- AWS Secrets Manager
- Google Secret Manager
- Azure Key Vault
- HashiCorp Vault

