"""Intelligence extraction from conversations.

Extracts structured data (UPI IDs, phone numbers, URLs, etc.) from message text.
Output matches the exact GUVI schema for final callback.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List

# Regex patterns for extraction
PATTERNS = {
    "upiIds": re.compile(
        r"[a-zA-Z0-9._-]+@(ybl|paytm|okaxis|oksbi|okhdfcbank|axl|upi|ibl|apl|waaxis|freecharge|icici|kotak|indus)",
        re.IGNORECASE,
    ),
    "phoneNumbers": re.compile(
        r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b",
    ),
    "phishingLinks": re.compile(
        r"https?://[^\s<>\"']+",
        re.IGNORECASE,
    ),
    "bankAccounts": re.compile(
        # Indian bank account numbers are typically 9-18 digits
        # Match only when in context of "account" or explicit mention
        r"\b\d{9,18}\b",
    ),
}

# Keywords that indicate suspicious/scam content
SUSPICIOUS_KEYWORDS = [
    "otp", "verify", "blocked", "suspended", "urgent", "immediately",
    "arrest", "police", "legal action", "fine", "penalty", "refund",
    "cashback", "lottery", "winner", "prize", "kyc", "update",
    "link click", "download", "install", "remote", "anydesk", "teamviewer",
]


@dataclass
class ExtractedIntelligence:
    """Container for extracted intelligence matching GUVI schema."""

    bankAccounts: List[str] = field(default_factory=list)
    upiIds: List[str] = field(default_factory=list)
    phishingLinks: List[str] = field(default_factory=list)
    phoneNumbers: List[str] = field(default_factory=list)
    suspiciousKeywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[str]]:
        """Convert to dictionary for JSON serialization."""
        return {
            "bankAccounts": self.bankAccounts,
            "upiIds": self.upiIds,
            "phishingLinks": self.phishingLinks,
            "phoneNumbers": self.phoneNumbers,
            "suspiciousKeywords": self.suspiciousKeywords,
        }

    def merge(self, other: "ExtractedIntelligence") -> None:
        """Merge another ExtractedIntelligence into this one, deduplicating."""
        self.bankAccounts = list(set(self.bankAccounts + other.bankAccounts))
        self.upiIds = list(set(self.upiIds + other.upiIds))
        self.phishingLinks = list(set(self.phishingLinks + other.phishingLinks))
        self.phoneNumbers = list(set(self.phoneNumbers + other.phoneNumbers))
        self.suspiciousKeywords = list(set(self.suspiciousKeywords + other.suspiciousKeywords))


def extract_intelligence(text: str) -> ExtractedIntelligence:
    """Extract intelligence entities from text.

    Args:
        text: The message text to analyze.

    Returns:
        ExtractedIntelligence with all found entities.
    """
    result = ExtractedIntelligence()
    text_lower = text.lower()

    # Extract UPI IDs
    for match in PATTERNS["upiIds"].finditer(text):
        result.upiIds.append(match.group(0).lower())

    # Extract phone numbers
    phone_matches = PATTERNS["phoneNumbers"].findall(text)
    for phone in phone_matches:
        # Normalize: remove spaces, dashes, +91 prefix
        normalized = re.sub(r"[\s\-+]", "", phone)
        if normalized.startswith("91") and len(normalized) > 10:
            normalized = normalized[2:]
        result.phoneNumbers.append(normalized)

    # Extract URLs (potential phishing links)
    url_matches = PATTERNS["phishingLinks"].findall(text)
    for url in url_matches:
        # Exclude known safe domains
        if not any(safe in url.lower() for safe in ["google.com", "microsoft.com", "apple.com"]):
            result.phishingLinks.append(url)

    # Extract potential bank account numbers (very context-dependent)
    # Only extract if "account" is mentioned nearby
    if "account" in text_lower or "a/c" in text_lower or "bank" in text_lower:
        account_matches = PATTERNS["bankAccounts"].findall(text)
        for acc in account_matches:
            if len(acc) >= 9:  # Minimum length for bank accounts
                result.bankAccounts.append(acc)

    # Extract suspicious keywords
    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in text_lower:
            result.suspiciousKeywords.append(keyword)

    # Deduplicate all lists
    result.bankAccounts = list(set(result.bankAccounts))
    result.upiIds = list(set(result.upiIds))
    result.phishingLinks = list(set(result.phishingLinks))
    result.phoneNumbers = list(set(result.phoneNumbers))
    result.suspiciousKeywords = list(set(result.suspiciousKeywords))

    return result


def merge_intel_buffer(
    existing: Dict[str, List[str]],
    new_intel: ExtractedIntelligence,
) -> Dict[str, List[str]]:
    """Merge new intelligence into existing buffer, deduplicating.

    Args:
        existing: Existing intel buffer from session.
        new_intel: Newly extracted intelligence.

    Returns:
        Merged intel buffer dictionary.
    """
    merged = {
        "bankAccounts": list(set(existing.get("bankAccounts", []) + new_intel.bankAccounts)),
        "upiIds": list(set(existing.get("upiIds", []) + new_intel.upiIds)),
        "phishingLinks": list(set(existing.get("phishingLinks", []) + new_intel.phishingLinks)),
        "phoneNumbers": list(set(existing.get("phoneNumbers", []) + new_intel.phoneNumbers)),
        "suspiciousKeywords": list(set(existing.get("suspiciousKeywords", []) + new_intel.suspiciousKeywords)),
    }
    return merged
