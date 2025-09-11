"""LLM client for API integration."""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Base class for LLM clients."""

    @abstractmethod
    def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def generate_structured_response(
        self, prompt: str, schema: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        """Generate a structured response following a schema."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4"):
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model to use
        """
        try:
            import openai

            self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.model = model
        except ImportError as e:
            raise ImportError(
                "OpenAI library not available. Install with: pip install openai"
            ) from e
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenAI client: {e}") from e

    def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from OpenAI."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert machine learning engineer.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 2000),
                timeout=kwargs.get("timeout", 30),
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def generate_structured_response(
        self, prompt: str, schema: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        """Generate a structured response."""

        structured_prompt = f"""
{prompt}

Please respond with a valid JSON object that matches this schema:
{json.dumps(schema, indent=2)}

Respond with only the JSON, no additional text.
"""

        response = self.generate_response(structured_prompt, **kwargs)

        try:
            # Clean response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]

            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structured response: {e}")
            logger.error(f"Response was: {response}")
            raise


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client."""

    def __init__(
        self, api_key: str | None = None, model: str = "claude-3-sonnet-20240229"
    ):
        """Initialize Anthropic client.

        Args:
            api_key: Anthropic API key
            model: Model to use
        """
        try:
            import anthropic

            self.client = anthropic.Anthropic(
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
            )
            self.model = model
        except ImportError as e:
            raise ImportError(
                "Anthropic library not available. Install with: pip install anthropic"
            ) from e
        except Exception as e:
            raise ValueError(f"Failed to initialize Anthropic client: {e}") from e

    def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from Claude."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 2000),
                temperature=kwargs.get("temperature", 0.1),
                messages=[{"role": "user", "content": prompt}],
                timeout=kwargs.get("timeout", 30),
            )

            return response.content[0].text.strip()

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def generate_structured_response(
        self, prompt: str, schema: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        """Generate a structured response."""

        structured_prompt = f"""
{prompt}

Please respond with a valid JSON object that matches this schema:
{json.dumps(schema, indent=2)}

Respond with only the JSON, no additional text.
"""

        response = self.generate_response(structured_prompt, **kwargs)

        try:
            # Clean response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]

            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structured response: {e}")
            logger.error(f"Response was: {response}")
            raise


class GeminiClient(BaseLLMClient):
    """Google Gemini API client with code execution support."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash"):
        """Initialize Gemini client.

        Args:
            api_key: Gemini API key
            model: Model to use
        """
        try:
            import google.generativeai as genai

            self.client = genai.configure(api_key=api_key or os.getenv("GEMINI_API_KEY"))
            self.model = model
        except ImportError as e:
            raise ImportError(
                "Google Generative AI library not available. Install with: pip install google-generativeai"
            ) from e
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini client: {e}") from e

    def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from Gemini."""
        try:
            import google.generativeai as genai

            model = genai.GenerativeModel(self.model)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get("temperature", 0.1),
                    max_output_tokens=kwargs.get("max_tokens", 2000),
                ),
            )
            return response.text.strip()

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def generate_code_with_execution(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Generate and execute code using Gemini's code execution tool."""
        try:
            import google.generativeai as genai

            model = genai.GenerativeModel(self.model)
            
            # Configure tools for code execution
            tools = [genai.protos.Tool(code_execution=genai.protos.ToolCodeExecution())]
            
            response = model.generate_content(
                prompt,
                tools=tools,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get("temperature", 0.1),
                    max_output_tokens=kwargs.get("max_tokens", 4000),
                ),
            )

            # Extract code and execution results
            result = {
                "text": response.text or "",
                "code": "",
                "execution_output": "",
                "execution_success": False
            }

            for part in response.candidates[0].content.parts:
                if hasattr(part, 'executable_code') and part.executable_code:
                    result["code"] = part.executable_code.code
                if hasattr(part, 'code_execution_result') and part.code_execution_result:
                    result["execution_output"] = part.code_execution_result.output
                    result["execution_success"] = True

            return result

        except Exception as e:
            logger.error(f"Gemini code execution error: {e}")
            return {
                "text": "",
                "code": "",
                "execution_output": f"Error: {str(e)}",
                "execution_success": False
            }

    def generate_structured_response(
        self, prompt: str, schema: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        """Generate a structured response."""
        structured_prompt = f"""
{prompt}

Please respond with a valid JSON object that matches this schema:
{json.dumps(schema, indent=2)}

Respond with only the JSON, no additional text.
"""

        response = self.generate_response(structured_prompt, **kwargs)

        try:
            # Clean response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]

            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structured response: {e}")
            logger.error(f"Response was: {response}")
            raise


class E2BClient:
    """E2B sandbox client for code execution."""

    def __init__(self, api_key: str | None = None):
        """Initialize E2B client.

        Args:
            api_key: E2B API key
        """
        try:
            from e2b import Sandbox

            self.api_key = api_key or os.getenv("E2B_API_KEY")
            self.sandbox = None
        except ImportError as e:
            raise ImportError(
                "E2B library not available. Install with: pip install e2b"
            ) from e

    def create_sandbox(self):
        """Create a new E2B sandbox."""
        try:
            from e2b import Sandbox

            self.sandbox = Sandbox(api_key=self.api_key)
            logger.info("E2B sandbox created successfully")
            return self.sandbox
        except Exception as e:
            logger.error(f"Failed to create E2B sandbox: {e}")
            raise

    def execute_code(self, code: str, timeout: int = 30) -> dict[str, Any]:
        """Execute code in E2B sandbox."""
        if not self.sandbox:
            self.create_sandbox()

        try:
            result = self.sandbox.run_python(code, timeout=timeout)
            return {
                "output": result.stdout,
                "error": result.stderr,
                "success": result.exit_code == 0,
                "exit_code": result.exit_code
            }
        except Exception as e:
            logger.error(f"E2B code execution failed: {e}")
            return {
                "output": "",
                "error": str(e),
                "success": False,
                "exit_code": -1
            }

    def install_packages(self, packages: list[str]):
        """Install packages in the sandbox."""
        if not self.sandbox:
            self.create_sandbox()

        try:
            for package in packages:
                self.sandbox.run_python(f"import subprocess; subprocess.run(['pip', 'install', '{package}'])")
            logger.info(f"Installed packages: {packages}")
        except Exception as e:
            logger.error(f"Failed to install packages: {e}")

    def close(self):
        """Close the sandbox."""
        if self.sandbox:
            self.sandbox.close()
            self.sandbox = None


class LLMClient:
    """Unified LLM client with fallback support and code execution capabilities."""

    def __init__(self, primary_provider: str = "openai", **kwargs: Any):
        """Initialize LLM client.

        Args:
            primary_provider: Primary LLM provider ('openai', 'anthropic', 'gemini')
            **kwargs: Any: Provider-specific configuration
        """
        self.primary_provider = primary_provider
        self.clients = {}
        self.e2b_client = None
        self._initialize_clients(**kwargs)

    def _initialize_clients(self, **kwargs: Any):
        """Initialize available LLM clients."""

        # Try OpenAI
        try:
            self.clients["openai"] = OpenAIClient(**kwargs)
            logger.info("OpenAI client initialized")
        except Exception as e:
            logger.warning(f"OpenAI client not available: {e}")

        # Try Anthropic
        try:
            self.clients["anthropic"] = AnthropicClient(**kwargs)
            logger.info("Anthropic client initialized")
        except Exception as e:
            logger.warning(f"Anthropic client not available: {e}")

        # Try Gemini
        try:
            self.clients["gemini"] = GeminiClient(**kwargs)
            logger.info("Gemini client initialized")
        except Exception as e:
            logger.warning(f"Gemini client not available: {e}")

        # Try E2B
        try:
            self.e2b_client = E2BClient(**kwargs)
            logger.info("E2B client initialized")
        except Exception as e:
            logger.warning(f"E2B client not available: {e}")

        if not self.clients:
            raise ValueError(
                "No LLM clients available. Please check your API keys and dependencies."
            )

    def generate_response(
        self, prompt: str, provider: str | None = None, **kwargs: Any
    ) -> str:
        """Generate a response from the specified or best available provider.

        Args:
            prompt: Input prompt
            provider: Specific provider to use
            **kwargs: Any: Additional parameters

        Returns:
            Generated response
        """

        # Determine which provider to use
        if provider and provider in self.clients:
            selected_provider = provider
        elif self.primary_provider in self.clients:
            selected_provider = self.primary_provider
        else:
            # Use first available provider
            selected_provider = list(self.clients.keys())[0]

        try:
            response = self.clients[selected_provider].generate_response(
                prompt, **kwargs
            )
            logger.debug(f"Generated response using {selected_provider}")
            return response
        except Exception as e:
            logger.error(f"Error with {selected_provider}: {e}")

            # Try fallback providers
            for fallback_provider in self.clients:
                if fallback_provider != selected_provider:
                    try:
                        response = self.clients[fallback_provider].generate_response(
                            prompt, **kwargs
                        )
                        logger.info(f"Used fallback provider: {fallback_provider}")
                        return response
                    except Exception as fallback_error:
                        logger.error(
                            f"Fallback {fallback_provider} also failed: {fallback_error}"
                        )

            raise RuntimeError("All LLM providers failed") from None

    def generate_structured_response(
        self,
        prompt: str,
        schema: dict[str, Any],
        provider: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a structured response following a schema.

        Args:
            prompt: Input prompt
            schema: Expected response schema
            provider: Specific provider to use
            **kwargs: Any: Additional parameters

        Returns:
            Structured response as dictionary
        """

        # Determine which provider to use
        if provider and provider in self.clients:
            selected_provider = provider
        elif self.primary_provider in self.clients:
            selected_provider = self.primary_provider
        else:
            # Use first available provider
            selected_provider = list(self.clients.keys())[0]

        try:
            response = self.clients[selected_provider].generate_structured_response(
                prompt, schema, **kwargs
            )
            logger.debug(f"Generated structured response using {selected_provider}")
            return response
        except Exception as e:
            logger.error(f"Error with {selected_provider}: {e}")

            # Try fallback providers
            for fallback_provider in self.clients:
                if fallback_provider != selected_provider:
                    try:
                        response = self.clients[
                            fallback_provider
                        ].generate_structured_response(prompt, schema, **kwargs)
                        logger.info(f"Used fallback provider: {fallback_provider}")
                        return response
                    except Exception as fallback_error:
                        logger.error(
                            f"Fallback {fallback_provider} also failed: {fallback_error}"
                        )

            raise RuntimeError("All LLM providers failed") from None

    def generate_code_with_execution(
        self, prompt: str, provider: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Generate and execute code using the best available provider.

        Args:
            prompt: Input prompt for code generation
            provider: Specific provider to use
            **kwargs: Any: Additional parameters

        Returns:
            Dictionary with code, execution results, and metadata
        """
        # Prefer Gemini for code execution if available
        if provider == "gemini" and "gemini" in self.clients:
            try:
                return self.clients["gemini"].generate_code_with_execution(prompt, **kwargs)
            except Exception as e:
                logger.error(f"Gemini code execution failed: {e}")

        # Try Gemini as fallback
        if "gemini" in self.clients:
            try:
                return self.clients["gemini"].generate_code_with_execution(prompt, **kwargs)
            except Exception as e:
                logger.error(f"Gemini code execution failed: {e}")

        # Fallback to E2B if available
        if self.e2b_client:
            try:
                # Generate code using any available LLM
                code_prompt = f"""
{prompt}

Please generate Python code to solve this problem. Return only the Python code, no explanations or markdown formatting.
"""
                code_response = self.generate_response(code_prompt, provider, **kwargs)
                
                # Clean the response to extract code
                code = self._extract_code_from_response(code_response)
                
                # Execute in E2B sandbox
                execution_result = self.e2b_client.execute_code(code, timeout=kwargs.get("timeout", 30))
                
                return {
                    "text": code_response,
                    "code": code,
                    "execution_output": execution_result["output"],
                    "execution_error": execution_result["error"],
                    "execution_success": execution_result["success"],
                    "provider": "e2b_fallback"
                }
            except Exception as e:
                logger.error(f"E2B code execution failed: {e}")

        # Final fallback - just generate code without execution
        try:
            code_response = self.generate_response(prompt, provider, **kwargs)
            code = self._extract_code_from_response(code_response)
            
            return {
                "text": code_response,
                "code": code,
                "execution_output": "",
                "execution_error": "Code execution not available",
                "execution_success": False,
                "provider": "code_generation_only"
            }
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return {
                "text": "",
                "code": "",
                "execution_output": "",
                "execution_error": str(e),
                "execution_success": False,
                "provider": "error"
            }

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response."""
        response = response.strip()
        
        # Remove markdown code blocks
        if response.startswith("```python"):
            response = response[9:]
        elif response.startswith("```"):
            response = response[3:]
        
        if response.endswith("```"):
            response = response[:-3]
        
        return response.strip()

    def execute_code_in_sandbox(self, code: str, timeout: int = 30) -> dict[str, Any]:
        """Execute code in E2B sandbox.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds

        Returns:
            Execution results
        """
        if not self.e2b_client:
            return {
                "output": "",
                "error": "E2B client not available",
                "success": False,
                "exit_code": -1
            }

        return self.e2b_client.execute_code(code, timeout)

    def install_packages_in_sandbox(self, packages: list[str]):
        """Install packages in E2B sandbox.

        Args:
            packages: List of package names to install
        """
        if self.e2b_client:
            self.e2b_client.install_packages(packages)

    def close_sandbox(self):
        """Close E2B sandbox."""
        if self.e2b_client:
            self.e2b_client.close()

    def get_available_providers(self) -> list[str]:
        """Get list of available providers."""
        return list(self.clients.keys())

    def test_connection(self, provider: str | None = None) -> bool:
        """Test connection to LLM provider.

        Args:
            provider: Provider to test (None for all)

        Returns:
            True if connection successful
        """

        test_prompt = "Respond with 'OK' if you can read this message."

        if provider:
            providers_to_test = [provider] if provider in self.clients else []
        else:
            providers_to_test = list(self.clients.keys())

        results = {}
        for prov in providers_to_test:
            try:
                response = self.clients[prov].generate_response(
                    test_prompt, max_tokens=10
                )
                results[prov] = "OK" in response
            except Exception as e:
                results[prov] = False
                logger.error(f"Connection test failed for {prov}: {e}")

        return results


def create_llm_client(provider: str = "openai", **kwargs: Any) -> LLMClient:
    """Convenience function to create an LLM client.

    Args:
        provider: Primary provider
        **kwargs: Any: Provider-specific configuration

    Returns:
        LLM client instance
    """
    return LLMClient(primary_provider=provider, **kwargs)
