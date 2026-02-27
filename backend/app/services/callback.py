"""Final callback client for GUVI evaluation server.

Sends extracted intelligence to the evaluation endpoint.
CRITICAL: Missing this callback = automatic disqualification.

Payload must EXACTLY match GUVI schema:
{
  "sessionId": string,
  "scamDetected": boolean,
  "totalMessagesExchanged": integer,
  "extractedIntelligence": {
    "bankAccounts": string[],
    "upiIds": string[],
    "phishingLinks": string[],
    "phoneNumbers": string[],
    "suspiciousKeywords": string[]
  },
  "agentNotes": string
}
"""

import logging
from typing import Any, Dict

import httpx

logger = logging.getLogger(__name__)

CALLBACK_URL = "https://hackathon.guvi.in/api/updateHoneyPotFinalResult"
CALLBACK_TIMEOUT = 5.0  # seconds (per GUVI spec: timeout ≤ 5 seconds)


async def send_callback(
    session_id: str,
    intel_buffer: Dict[str, Any],
    scam_detected: bool,
    total_messages: int,
    agent_notes: str,
) -> bool:
    """Send final intelligence to the GUVI evaluation server.

    This callback is triggered ONLY when all three conditions are met:
    1. scamDetected = true
    2. Engagement is complete (max messages reached)
    3. Intelligence has been fully extracted (accumulated in intel_buffer)

    Args:
        session_id: The session identifier.
        intel_buffer: Extracted intelligence matching GUVI schema.
        scam_detected: Whether scam was detected in this session.
        total_messages: Total message count exchanged in session.
        agent_notes: Summary notes about the scam engagement.

    Returns:
        True if callback was successful, False otherwise.
    """
    # Ensure extractedIntelligence has all required fields (no nulls, empty arrays OK)
    extracted_intelligence = {
        "bankAccounts": intel_buffer.get("bankAccounts", []) or [],
        "upiIds": intel_buffer.get("upiIds", []) or [],
        "phishingLinks": intel_buffer.get("phishingLinks", []) or [],
        "phoneNumbers": intel_buffer.get("phoneNumbers", []) or [],
        "suspiciousKeywords": intel_buffer.get("suspiciousKeywords", []) or [],
    }

    # Construct payload matching GUVI schema EXACTLY
    payload = {
        "sessionId": session_id,
        "scamDetected": scam_detected,
        "totalMessagesExchanged": total_messages,
        "extractedIntelligence": extracted_intelligence,
        "agentNotes": agent_notes,
    }

    logger.info(f"Sending final callback for session {session_id}")
    logger.debug(f"Callback payload: {payload}")

    try:
        async with httpx.AsyncClient(timeout=CALLBACK_TIMEOUT) as client:
            response = await client.post(
                CALLBACK_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                logger.info(f"Callback successful for session {session_id}")
                return True
            else:
                logger.error(
                    f"Callback failed for session {session_id}: "
                    f"status={response.status_code}, body={response.text}"
                )
                return False

    except httpx.TimeoutException:
        logger.error(f"Callback timeout for session {session_id}")
        return False
    except httpx.RequestError as e:
        logger.error(f"Callback request failed for session {session_id}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during callback for session {session_id}: {e}")
        return False
