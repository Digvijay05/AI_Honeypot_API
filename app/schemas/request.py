"""Pydantic models for API request/response schemas.

Per TODO.md Phase 3:
- Add examples to HoneypotRequest and HoneypotResponse schema.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Individual message in a conversation."""

    sender: str = Field(..., description="The sender of the message", examples=["scammer", "agent"])
    text: str = Field(..., description="The message content", examples=["Your account is blocked!"])
    timestamp: int = Field(..., description="Epoch milliseconds timestamp", examples=[1770005528731])


class Metadata(BaseModel):
    """Optional metadata for the request."""

    channel: Optional[str] = Field(None, description="Communication channel", examples=["sms", "whatsapp", "email"])
    language: Optional[str] = Field(None, description="Language code", examples=["en", "hi"])
    locale: Optional[str] = Field(None, description="Locale code", examples=["en-US", "en-IN"])


class HoneypotRequest(BaseModel):
    """Request payload for honeypot message endpoint."""

    sessionId: str = Field(
        ...,
        description="Unique session identifier",
        examples=["sess-abc123-def456"],
    )
    message: Message = Field(..., description="The incoming message to process")
    conversationHistory: List[Message] = Field(
        default=[],
        description="Previous messages in the conversation",
    )
    metadata: Optional[Metadata] = Field(None, description="Optional request metadata")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sessionId": "sess-abc123-def456",
                    "message": {
                        "sender": "scammer",
                        "text": "Your bank account has been suspended. Verify immediately!",
                        "timestamp": 1770005528731,
                    },
                    "conversationHistory": [],
                    "metadata": {"channel": "sms", "language": "en", "locale": "en-US"},
                }
            ]
        }
    }


class HoneypotResponse(BaseModel):
    """Response payload for honeypot message endpoint."""

    status: str = Field(..., description="Response status", examples=["success", "error"])
    reply: str = Field(
        ...,
        description="The honeypot agent's reply",
        examples=["Oh dear, what should I do? Can you explain more slowly?"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "success",
                    "reply": "Oh my, that sounds serious! What do I need to do?",
                }
            ]
        }
    }
