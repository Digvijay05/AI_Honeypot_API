"""Abstract base class for LLM providers.

This module defines the contract that all LLM providers must implement,
enabling provider-agnostic usage across the application.
"""

from abc import ABC, abstractmethod
from typing import List
import logging

from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All concrete providers (Groq, OpenAI, etc.) must implement this interface.
    This ensures the agent logic remains decoupled from specific LLM backends.
    """

    @abstractmethod
    async def generate_response(
        self,
        messages: List[BaseMessage],
        temperature: float = 0.8,
        max_tokens: int = 512,
    ) -> str:
        """Generate a response given a list of messages.

        Args:
            messages: Conversation history as LangChain message objects.
            temperature: Sampling temperature (higher = more creative).
            max_tokens: Maximum tokens in the response.

        Returns:
            The generated text content.

        Raises:
            LLMProviderError: If the underlying API call fails.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the identifier of the model being used."""
        pass


class LLMProviderError(Exception):
    """Raised when an LLM provider encounters an error."""

    def __init__(self, message: str, provider: str, original_error: Exception = None):
        super().__init__(message)
        self.provider = provider
        self.original_error = original_error
