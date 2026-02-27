"""Groq LLM Provider implementation.

Uses langchain-groq to interface with Groq's high-speed inference API.

Model Selection Rationale:
- Default: llama-3.3-70b-versatile
  - Best balance of quality and speed for conversational tasks.
  - Supports large context windows (128k tokens).
  - Optimized on Groq's LPU hardware for low latency.
- Alternative: llama3-8b-8192 (faster but less capable)
"""

import logging
from typing import List

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage

from app.core.llm.base import LLMProvider, LLMProviderError
from app.core.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    """LLM provider implementation using Groq API via langchain-groq.

    Groq offers extremely low latency inference, making it ideal for
    real-time conversational applications like this honeypot.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        temperature: float = 0.8,
        max_tokens: int = 512,
    ):
        """Initialize the Groq provider.

        Args:
            api_key: Groq API key. Defaults to config.GROQ_API_KEY.
            model: Model identifier. Defaults to config.GROQ_MODEL.
            temperature: Default sampling temperature.
            max_tokens: Default max tokens for responses.
        """
        self._api_key = api_key or GROQ_API_KEY
        self._model = model or GROQ_MODEL
        self._default_temperature = temperature
        self._default_max_tokens = max_tokens

        # Initialize the LangChain Groq chat model
        self._chat = ChatGroq(
            model=self._model,
            api_key=self._api_key,
            temperature=self._default_temperature,
            max_tokens=self._default_max_tokens,
        )

        logger.info(f"GroqProvider initialized with model: {self._model}")

    @property
    def model_name(self) -> str:
        """Return the Groq model identifier."""
        return self._model

    async def generate_response(
        self,
        messages: List[BaseMessage],
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """Generate a response using Groq's API.

        Args:
            messages: List of LangChain message objects (System, Human, AI).
            temperature: Override default temperature if provided.
            max_tokens: Override default max_tokens if provided.

        Returns:
            The generated response text.

        Raises:
            LLMProviderError: If Groq API call fails.
        """
        # Apply overrides if provided
        effective_temp = temperature if temperature is not None else self._default_temperature
        effective_max = max_tokens if max_tokens is not None else self._default_max_tokens

        # Recreate chat instance if parameters differ from defaults
        # This is necessary because ChatGroq doesn't support per-call overrides natively
        if effective_temp != self._default_temperature or effective_max != self._default_max_tokens:
            chat = ChatGroq(
                model=self._model,
                api_key=self._api_key,
                temperature=effective_temp,
                max_tokens=effective_max,
            )
        else:
            chat = self._chat

        try:
            response = await chat.ainvoke(messages)
            content = response.content

            if not content:
                logger.warning("Empty response from Groq LLM")
                return ""

            return str(content)

        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            raise LLMProviderError(
                message=f"Groq generation failed: {str(e)}",
                provider="groq",
                original_error=e,
            )
