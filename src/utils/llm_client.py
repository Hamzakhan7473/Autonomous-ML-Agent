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


class LLMClient:
    """Unified LLM client with fallback support."""

    def __init__(self, primary_provider: str = "openai", **kwargs: Any):
        """Initialize LLM client.

        Args:
            primary_provider: Primary LLM provider ('openai', 'anthropic')
            **kwargs: Any: Provider-specific configuration
        """
        self.primary_provider = primary_provider
        self.clients = {}
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
