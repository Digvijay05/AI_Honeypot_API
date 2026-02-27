"""Ollama Cloud LLM Provider implementation.

Uses Ollama Cloud API for remote inference.

Model Selection:
- Default: gemma3:27b-cloud
  - Google's Gemma 3 27B parameter model hosted on Ollama Cloud
  - Good balance of capability and cost for conversational tasks

API Reference:
- Endpoint: https://ollama.com/api/chat
- Auth: Bearer token via OLLAMA_API_KEY
- Request format: {"model": "...", "messages": [...], "stream": false}
"""

import logging
import time
from typing import List, Dict, Any

import httpx
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from app.core.llm.base import LLMProvider, LLMProviderError
from app.core.config import OLLAMA_API_KEY, OLLAMA_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES: int = 3
RETRY_DELAY_SECONDS: float = 1.0
REQUEST_TIMEOUT_SECONDS: float = 60.0


def _convert_messages(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    """Convert LangChain messages to Ollama chat format.
    
    Ollama expects: {"role": "system|user|assistant", "content": "..."}
    """
    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
        else:
            # Fallback: treat as user message
            result.append({"role": "user", "content": str(msg.content)})
    return result


class OllamaProvider(LLMProvider):
    """LLM provider implementation using Ollama Cloud API.

    Ollama Cloud provides hosted inference for open-source models,
    offering a simple API compatible with chat-style interactions.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        base_url: str = None,
        temperature: float = 0.8,
        max_tokens: int = 512,
    ):
        """Initialize the Ollama Cloud provider.

        Args:
            api_key: Ollama API key. Defaults to config.OLLAMA_API_KEY.
            model: Model identifier. Defaults to config.OLLAMA_MODEL.
            base_url: Ollama Cloud base URL. Defaults to config.OLLAMA_BASE_URL.
            temperature: Default sampling temperature.
            max_tokens: Default max tokens for responses.
        """
        self._api_key = api_key or OLLAMA_API_KEY
        self._model = model or OLLAMA_MODEL
        self._base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")
        self._default_temperature = temperature
        self._default_max_tokens = max_tokens
        
        # Construct chat endpoint
        self._chat_endpoint = f"{self._base_url}/api/chat"
        
        logger.info(f"OllamaProvider initialized: model={self._model}, endpoint={self._chat_endpoint}")

    @property
    def model_name(self) -> str:
        """Return the Ollama model identifier."""
        return self._model

    async def generate_response(
        self,
        messages: List[BaseMessage],
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """Generate a response using Ollama Cloud API.

        Args:
            messages: List of LangChain message objects (System, Human, AI).
            temperature: Override default temperature if provided.
            max_tokens: Override default max_tokens if provided.

        Returns:
            The generated response text.

        Raises:
            LLMProviderError: If Ollama API call fails after retries.
        """
        effective_temp = temperature if temperature is not None else self._default_temperature
        effective_max = max_tokens if max_tokens is not None else self._default_max_tokens

        # Convert messages to Ollama format
        ollama_messages = _convert_messages(messages)

        # Build request payload
        payload = {
            "model": self._model,
            "messages": ollama_messages,
            "stream": False,  # We want the full response, not streaming
            "options": {
                "temperature": effective_temp,
                "num_predict": effective_max,  # Ollama uses num_predict for max tokens
            }
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()
                
                async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                    response = await client.post(
                        self._chat_endpoint,
                        json=payload,
                        headers=headers,
                    )
                
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"Ollama request completed in {elapsed_ms:.0f}ms, status={response.status_code}")

                if response.status_code == 200:
                    data = response.json()
                    # Ollama returns: {"message": {"role": "assistant", "content": "..."}}
                    content = data.get("message", {}).get("content", "")
                    
                    if not content:
                        logger.warning("Empty response from Ollama Cloud")
                        return ""
                    
                    return str(content)
                
                elif response.status_code == 401:
                    raise LLMProviderError(
                        message="Ollama Cloud authentication failed. Check OLLAMA_API_KEY.",
                        provider="ollama",
                    )
                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    logger.warning(f"Ollama rate limited, retrying in {RETRY_DELAY_SECONDS * (attempt + 1)}s")
                    await self._sleep(RETRY_DELAY_SECONDS * (attempt + 1))
                    continue
                else:
                    error_text = response.text[:200]
                    last_error = LLMProviderError(
                        message=f"Ollama API error {response.status_code}: {error_text}",
                        provider="ollama",
                    )
                    logger.warning(f"Ollama request failed: {response.status_code}")

            except httpx.TimeoutException as e:
                last_error = LLMProviderError(
                    message=f"Ollama request timed out after {REQUEST_TIMEOUT_SECONDS}s",
                    provider="ollama",
                    original_error=e,
                )
                logger.warning(f"Ollama timeout on attempt {attempt + 1}")
            except httpx.RequestError as e:
                last_error = LLMProviderError(
                    message=f"Ollama network error: {str(e)}",
                    provider="ollama",
                    original_error=e,
                )
                logger.warning(f"Ollama network error on attempt {attempt + 1}: {e}")

            # Wait before retry (except on last attempt)
            if attempt < MAX_RETRIES - 1:
                await self._sleep(RETRY_DELAY_SECONDS * (attempt + 1))

        # All retries exhausted
        logger.error(f"Ollama API call failed after {MAX_RETRIES} attempts")
        raise last_error or LLMProviderError(
            message="Ollama API call failed after retries",
            provider="ollama",
        )

    async def _sleep(self, seconds: float) -> None:
        """Async sleep helper for retry delays."""
        import asyncio
        await asyncio.sleep(seconds)
