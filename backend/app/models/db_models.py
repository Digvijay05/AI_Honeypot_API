"""Database models for persistence.

Defines SQLAlchemy ORM models for sessions, messages, scam analyses,
and extracted entities.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


class Session(Base):
    """Honeypot conversation session."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    session_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    channel: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    language: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    locale: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)

    messages: Mapped[List["MessageRecord"]] = relationship("MessageRecord", back_populates="session")
    scam_analyses: Mapped[List["ScamAnalysis"]] = relationship("ScamAnalysis", back_populates="session")
    extracted_entities: Mapped[List["ExtractedEntity"]] = relationship("ExtractedEntity", back_populates="session")


class MessageRecord(Base):
    """Individual message in a conversation."""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("sessions.id"), index=True)
    sender: Mapped[str] = mapped_column(String(50))
    text: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    session: Mapped["Session"] = relationship("Session", back_populates="messages")


class ScamAnalysis(Base):
    """Scam analysis result for a message."""

    __tablename__ = "scam_analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("sessions.id"), index=True)
    message_text: Mapped[str] = mapped_column(Text)
    is_scam: Mapped[bool] = mapped_column(Boolean)
    confidence: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text)
    analyzed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    session: Mapped["Session"] = relationship("Session", back_populates="scam_analyses")


class ExtractedEntity(Base):
    """Extracted intelligence entity from conversation.

    Tracks: Phone Numbers, UPI IDs, Crypto Addresses, URLs.
    """

    __tablename__ = "extracted_entities"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("sessions.id"), index=True)
    entity_type: Mapped[str] = mapped_column(String(50))  # phone, upi, crypto, url
    entity_value: Mapped[str] = mapped_column(Text)
    source_text: Mapped[str] = mapped_column(Text)
    extracted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    session: Mapped["Session"] = relationship("Session", back_populates="extracted_entities")
