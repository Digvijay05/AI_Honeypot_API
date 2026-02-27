"""Application configuration from environment variables."""

import os
import sys
from dotenv import load_dotenv

load_dotenv()


def _require_env(key: str) -> str:
    """Enforce presence of required environment variable. Fail if missing."""
    value = os.getenv(key)
    if not value:
        sys.exit(f"FATAL: Required environment variable '{key}' is not set.")
    return value


def _optional_env(key: str, default: str = None) -> str:
    """Get optional environment variable with fallback."""
    return os.getenv(key, default)


# Required secrets - no defaults, fail if missing
HONEYPOT_API_KEY: str = _require_env("HONEYPOT_API_KEY")

# Ollama Cloud Configuration (Primary LLM Provider)
OLLAMA_API_KEY: str = _require_env("OLLAMA_API_KEY")
OLLAMA_MODEL: str = _optional_env("OLLAMA_MODEL", "gemma3:27b-cloud")
OLLAMA_BASE_URL: str = _optional_env("OLLAMA_BASE_URL", "https://ollama.com")

# Groq Configuration (Optional - for fallback)
GROQ_API_KEY: str = _optional_env("GROQ_API_KEY", "")
GROQ_MODEL: str = _optional_env("GROQ_MODEL", "llama-3.3-70b-versatile")

# OpenAI Configuration (Optional - for fallback or legacy use)
OPENAI_API_KEY: str = _optional_env("OPENAI_API_KEY", "")

# Session settings
MAX_SESSION_MESSAGES: int = int(os.getenv("MAX_SESSION_MESSAGES", "20"))
