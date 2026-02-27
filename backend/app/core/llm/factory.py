"""LLM Provider Factory.

Provides a centralized way to obtain the configured LLM provider instance.
This module is the single entry point for agent code to access LLM capabilities.

Usage:
    from app.core.llm import get_llm_provider

    provider = get_llm_provider()
    response = await provider.generate_response(messages)
"""

import logging
from functools import lru_cache

from app.core.llm.base import LLMProvider
from app.core.llm.ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    """Get the configured LLM provider instance.

    Currently returns OllamaProvider as the default (gemma3:27b-cloud).
    Future: Could read from config to select between providers.

    Returns:
        An initialized LLMProvider instance (cached singleton).
    """
    logger.info("Initializing LLM provider (OllamaProvider)")
    return OllamaProvider()

