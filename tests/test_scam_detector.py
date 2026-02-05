"""Unit tests for scam detection engine."""

import os
import sys

# Set env vars before imports
os.environ["HONEYPOT_API_KEY"] = "test-key"
os.environ["OPENAI_API_KEY"] = "test-key"

import pytest

from app.scam_detector import detect_scam, ScamSignal
from app.scam_detector.rules import match_rules, SCAM_RULES


class TestScamRules:
    """Test individual rule matching."""

    def test_upi_id_detection(self) -> None:
        """Test UPI ID pattern detection."""
        text = "Please send payment to scammer@ybl"
        matched = match_rules(text)
        rule_names = [r.name for r in matched]
        assert "upi_id" in rule_names

    def test_urgency_detection(self) -> None:
        """Test urgency phrase detection."""
        text = "Verify immediately or your account will be blocked"
        matched = match_rules(text)
        rule_names = [r.name for r in matched]
        assert "urgency_verify" in rule_names or "account_blocked" in rule_names

    def test_otp_request_detection(self) -> None:
        """Test OTP request detection."""
        text = "Please share your OTP for verification"
        matched = match_rules(text)
        rule_names = [r.name for r in matched]
        assert "otp_request" in rule_names

    def test_legal_threat_detection(self) -> None:
        """Test legal threat detection."""
        text = "We will take legal action against you"
        matched = match_rules(text)
        rule_names = [r.name for r in matched]
        assert "legal_threat" in rule_names

    def test_normal_message_no_match(self) -> None:
        """Test that normal messages don't match scam rules."""
        text = "Hey, how are you doing today?"
        matched = match_rules(text)
        assert len(matched) == 0


class TestScamScorer:
    """Test scoring and threshold logic."""

    def test_scam_detected_above_threshold(self) -> None:
        """Test that high-signal messages trigger scam detection."""
        text = "Your account is blocked! Verify immediately with OTP. Send money to scam@ybl"
        result = detect_scam(text)
        assert result.is_scam is True
        assert result.confidence >= 0.5
        assert len(result.matched_rules) > 0

    def test_no_scam_below_threshold(self) -> None:
        """Test that low-signal messages don't trigger scam detection."""
        text = "Hello, can you help me with my order?"
        result = detect_scam(text)
        assert result.is_scam is False
        assert result.confidence < 0.5

    def test_returns_scam_signal_type(self) -> None:
        """Test that detect_scam returns ScamSignal dataclass."""
        result = detect_scam("test message")
        assert isinstance(result, ScamSignal)
        assert hasattr(result, "is_scam")
        assert hasattr(result, "confidence")
        assert hasattr(result, "matched_rules")

    def test_confidence_is_capped(self) -> None:
        """Test that confidence never exceeds 1.0."""
        text = "URGENT! Account blocked! Legal action! Send OTP to scam@ybl immediately! Pay fine of Rs 5000!"
        result = detect_scam(text)
        assert result.confidence <= 1.0
