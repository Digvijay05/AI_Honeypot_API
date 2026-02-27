"""Scam detection engine.

Combines regex-based rule matching with weighted scoring to produce
a scam detection signal. Each rule has a pattern and a weight;
higher weight = stronger scam indicator.
"""

import re
from dataclasses import dataclass, field
from typing import List, Pattern


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

@dataclass
class ScamRule:
    """A single scam detection rule."""

    name: str
    pattern: Pattern[str]
    weight: float
    description: str


# Compile all patterns with IGNORECASE for robustness
SCAM_RULES: List[ScamRule] = [
    # UPI Patterns
    ScamRule(
        name="upi_id",
        pattern=re.compile(r"[a-zA-Z0-9._-]+@(ybl|paytm|okaxis|oksbi|okhdfcbank|axl|upi|ibl)", re.IGNORECASE),
        weight=0.3,
        description="UPI ID detected",
    ),
    ScamRule(
        name="upi_link",
        pattern=re.compile(r"upi://pay\?", re.IGNORECASE),
        weight=0.4,
        description="UPI payment link detected",
    ),
    # Urgency Phrases
    ScamRule(
        name="urgency_verify",
        pattern=re.compile(r"\b(verify\s*(immediately|now|urgently)|urgent\s*verification)\b", re.IGNORECASE),
        weight=0.25,
        description="Urgency to verify",
    ),
    ScamRule(
        name="urgency_action",
        pattern=re.compile(r"\b(action\s*required|immediate\s*action|act\s*now|within\s*\d+\s*(hours?|minutes?))\b", re.IGNORECASE),
        weight=0.25,
        description="Urgency to act",
    ),
    ScamRule(
        name="account_blocked",
        pattern=re.compile(r"\b(account\s*(blocked|suspended|frozen|locked|disabled)|block(ed|ing)\s*your\s*account)\b", re.IGNORECASE),
        weight=0.35,
        description="Account threat",
    ),
    # Threat Language
    ScamRule(
        name="legal_threat",
        pattern=re.compile(r"\b(legal\s*action|arrest\s*warrant|police|court\s*notice|lawsuit|prosecution)\b", re.IGNORECASE),
        weight=0.4,
        description="Legal threat detected",
    ),
    ScamRule(
        name="penalty_threat",
        pattern=re.compile(r"\b(penalty|fine|fee|charge)\s*(of\s*)?(rs\.?|₹|inr)?\s*\d+", re.IGNORECASE),
        weight=0.3,
        description="Financial penalty threat",
    ),
    # Sensitive Info Requests
    ScamRule(
        name="otp_request",
        pattern=re.compile(r"\b(otp|one\s*time\s*password|verification\s*code|pin|cvv)\b", re.IGNORECASE),
        weight=0.35,
        description="OTP/PIN request",
    ),
    ScamRule(
        name="bank_details",
        pattern=re.compile(r"\b(bank\s*details?|account\s*number|ifsc|card\s*number|atm\s*pin)\b", re.IGNORECASE),
        weight=0.35,
        description="Bank details request",
    ),
    ScamRule(
        name="password_request",
        pattern=re.compile(r"\b(password|login\s*credentials?|username\s*and\s*password)\b", re.IGNORECASE),
        weight=0.35,
        description="Password request",
    ),
    # Impersonation
    ScamRule(
        name="bank_impersonation",
        pattern=re.compile(r"\b(sbi|hdfc|icici|axis|rbi|reserve\s*bank|bank\s*of\s*india)\s*(bank|customer\s*care|support)?\b", re.IGNORECASE),
        weight=0.2,
        description="Bank name mentioned",
    ),
    ScamRule(
        name="govt_impersonation",
        pattern=re.compile(r"\b(income\s*tax|it\s*department|customs|cyber\s*cell|police|government)\b", re.IGNORECASE),
        weight=0.25,
        description="Government impersonation",
    ),
    # Financial Pressure
    ScamRule(
        name="transfer_request",
        pattern=re.compile(r"\b(transfer|send|pay)\s*(money|amount|rs\.?|₹|inr)?\s*\d*\s*(to|immediately)?\b", re.IGNORECASE),
        weight=0.3,
        description="Money transfer request",
    ),
    ScamRule(
        name="refund_bait",
        pattern=re.compile(r"\b(refund|cashback|reward|prize|lottery|winner)\b", re.IGNORECASE),
        weight=0.25,
        description="Refund/reward bait",
    ),
    # Phone Numbers (Indian format)
    ScamRule(
        name="phone_number",
        pattern=re.compile(r"\b(\+91[\s-]?)?[6-9]\d{9}\b"),
        weight=0.15,
        description="Phone number detected",
    ),
    # Suspicious URLs
    ScamRule(
        name="suspicious_url",
        pattern=re.compile(r"https?://(?!www\.(google|microsoft|apple|amazon|facebook|twitter|instagram)\.com)[^\s]+\.(xyz|tk|ml|ga|cf|gq|top|click|link|info)/", re.IGNORECASE),
        weight=0.4,
        description="Suspicious URL detected",
    ),
    ScamRule(
        name="shortened_url",
        pattern=re.compile(r"https?://(bit\.ly|tinyurl|t\.co|goo\.gl|ow\.ly|is\.gd|buff\.ly)/[^\s]+", re.IGNORECASE),
        weight=0.25,
        description="Shortened URL detected",
    ),
]


# ---------------------------------------------------------------------------
# Rule matching
# ---------------------------------------------------------------------------

def match_rules(text: str) -> List[ScamRule]:
    """Match text against all scam rules.

    Args:
        text: The message text to analyze.

    Returns:
        List of matched ScamRule objects.
    """
    matched = []
    for rule in SCAM_RULES:
        if rule.pattern.search(text):
            matched.append(rule)
    return matched


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# Threshold for triggering scam detection
SCAM_THRESHOLD: float = 0.5


@dataclass
class ScamSignal:
    """Result of scam detection analysis."""

    is_scam: bool
    confidence: float
    matched_rules: List[str] = field(default_factory=list)


def calculate_score(matched: List[ScamRule]) -> float:
    """Calculate aggregate scam score from matched rules.

    Uses diminishing returns: each additional match adds less.
    Score is capped at 1.0.

    Args:
        matched: List of matched ScamRule objects.

    Returns:
        Float score between 0.0 and 1.0.
    """
    if not matched:
        return 0.0

    total = sum(rule.weight for rule in matched)
    # Apply soft cap using tanh-like function
    return min(1.0, total)


def detect_scam(text: str) -> ScamSignal:
    """Analyze text for scam intent using regex rules.

    Args:
        text: The message text to analyze.

    Returns:
        ScamSignal with is_scam boolean, confidence score, and matched rules.
    """
    matched = match_rules(text)
    score = calculate_score(matched)

    return ScamSignal(
        is_scam=score >= SCAM_THRESHOLD,
        confidence=round(score, 2),
        matched_rules=[rule.name for rule in matched],
    )
