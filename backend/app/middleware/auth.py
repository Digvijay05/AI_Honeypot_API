"""API key authentication middleware."""

from fastapi import Header, HTTPException

from app.core.config import HONEYPOT_API_KEY


def verify_api_key(x_api_key: str = Header(...)) -> str:
    """Verify API key from request header.

    Args:
        x_api_key: The API key from x-api-key header.

    Returns:
        The validated API key.

    Raises:
        HTTPException: If API key is invalid.
    """
    if x_api_key != HONEYPOT_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key
