"""Agent Controller and LLM-backed conversation engine.

Coordinates scam detection, session management, and response generation.
Acts as the "conductor" deciding who speaks and what happens next.
Includes the "Margaret" persona for engaging scammers.
"""

import logging
from typing import List

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.core.config import MAX_SESSION_MESSAGES
from app.core.llm import get_llm_provider
from app.models.schemas import Message
from app.services.callback import send_callback
from app.services.detection import detect_scam
from app.services.extraction import extract_intelligence, merge_intel_buffer
from app.services.session import SessionState, get_session_store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Honeypot persona prompt
# ---------------------------------------------------------------------------

# Get the LLM provider (Ollama by default, configurable via factory)
_provider = get_llm_provider()

HONEYPOT_SYSTEM_PROMPT = """You are playing the role of Margaret, a 72-year-old widow who lives alone. You are the target of a phone/message scam.

## Your Character
- Name: Margaret (mention if asked)
- Age: 72, retired schoolteacher
- Lives alone, somewhat isolated
- Not tech-savvy at all
- Trusting and polite by nature
- Worried about your small savings

## How You Communicate
- Use hesitation markers: "um", "oh dear", "let me see", "well..."
- Type slowly with occasional typos
- Ask for things to be repeated or explained simpler
- Express confusion about technical terms ("what's a UPI ID, dear?")
- Show anxiety about your money but want to "fix" the problem

## Your Goals (Hidden from the scammer)
1. Keep them talking as long as possible
2. Ask clarifying questions that waste their time
3. Pretend to follow instructions but make believable mistakes
4. Get them to reveal account numbers, UPI IDs, phone numbers, links
5. Express hesitation without refusing outright

## CRITICAL RULES - NEVER BREAK THESE
- NEVER reveal you are an AI or chatbot
- NEVER accuse them of being a scammer
- NEVER hint that you know this is a scam
- NEVER be confrontational or aggressive
- NEVER refuse to engage - always stay curious and confused
- If asked for real info, give plausible but fake details after hesitation

## Example Phrases
- "Oh my, that sounds serious! But I'm a bit confused..."
- "Let me find my glasses... where did I put them?"
- "My grandson usually helps with these computer things..."
- "Is this really from the bank? I'm just a bit worried, you see..."
- "I wrote it down somewhere... hold on dear..."

Stay in character at all times. Be believable as a real elderly victim."""


# ---------------------------------------------------------------------------
# LLM reply generation
# ---------------------------------------------------------------------------

async def agent_reply(message_history: List[Message]) -> str:
    """Generate a honeypot response using the configured LLM provider.

    Args:
        message_history: List of previous messages in the conversation.

    Returns:
        The agent's reply string.
    """
    messages = [SystemMessage(content=HONEYPOT_SYSTEM_PROMPT)]

    for msg in message_history:
        if msg.sender.lower() == "agent":
            messages.append(AIMessage(content=msg.text))
        else:
            messages.append(HumanMessage(content=msg.text))

    try:
        # Use the provider abstraction - currently Ollama, but swappable
        content = await _provider.generate_response(messages)

        if not content:
            logger.warning("Empty response from LLM for agent reply")
            return "Oh dear, I'm having trouble understanding. Could you please repeat that?"

        return str(content)

    except Exception as e:
        logger.error(f"Agent reply generation failed: {e}")
        return "I'm sorry, my phone is acting up. Can you say that again?"


# ---------------------------------------------------------------------------
# Controller orchestration
# ---------------------------------------------------------------------------

class AgentController:
    """Controls the flow of honeypot conversations.

    Responsibilities:
    - Decide if scam is detected (activate agent)
    - Manage session state
    - Determine stop conditions
    - Generate appropriate responses
    """

    def __init__(self) -> None:
        self._store = get_session_store()

    async def process_message(
        self,
        session_id: str,
        message: Message,
        conversation_history: List[Message],
    ) -> str:
        """Process an incoming message and generate a response.

        Args:
            session_id: Unique session identifier.
            message: The incoming message.
            conversation_history: Previous messages in the conversation.

        Returns:
            The response string to send back.
        """
        # 1. Get or create session
        session = await self._store.get(session_id)
        if session is None:
            session = SessionState(session_id=session_id)
            logger.info(f"Created new session: {session_id}")

        # 2. Run scam detection
        scam_result = detect_scam(message.text)
        logger.debug(f"Scam detection: {scam_result}")

        # 3. Update session state
        session.message_count += 1
        if scam_result.is_scam:
            session.scam_score = max(session.scam_score, scam_result.confidence)
            session.is_scam = True
            session.agent_active = True
            logger.info(f"Scam detected for session {session_id}, agent activated")

        # 4. Extract intelligence from message
        intel = extract_intelligence(message.text)
        session.intel_buffer = merge_intel_buffer(session.intel_buffer, intel)
        logger.debug(f"Intel buffer updated: {session.intel_buffer}")

        # 5. Decision engine
        if not session.agent_active:
            # Normal message - minimal response
            await self._store.save(session)
            return "Okay."

        # 6. Check stop conditions
        if session.message_count >= MAX_SESSION_MESSAGES:
            logger.info(f"Session {session_id} reached max messages ({MAX_SESSION_MESSAGES})")
            # Send final callback before ending
            # Guard: only send when scam detected AND not already sent
            if not session.callback_sent and session.is_scam:
                agent_notes = self._generate_agent_notes(session)
                success = await send_callback(
                    session_id=session_id,
                    intel_buffer=session.intel_buffer,
                    scam_detected=True,
                    total_messages=session.message_count,
                    agent_notes=agent_notes,
                )
                if success:
                    session.callback_sent = True
                    logger.info(f"Final callback sent for session {session_id}")
            await self._store.save(session)
            return "I'm sorry, I need to go now. Thank you for calling."

        if session.callback_sent:
            logger.info(f"Session {session_id} already sent callback")
            await self._store.save(session)
            return "I already reported this. Please don't call again."

        # 7. Generate agent response
        full_history = list(conversation_history) + [message]
        try:
            reply = await agent_reply(full_history)
        except Exception as e:
            logger.error(f"Agent reply failed: {e}")
            reply = "I'm sorry, I didn't understand that. Could you repeat?"

        # 8. Save session
        await self._store.save(session)

        return reply

    def _generate_agent_notes(self, session: SessionState) -> str:
        """Generate summary notes for the GUVI callback.

        Creates a human-readable summary of the scam engagement based on
        extracted intelligence and detection results.

        Args:
            session: The session state with accumulated intelligence.

        Returns:
            A summary string for the agentNotes field.
        """
        notes: List[str] = []

        # Summarize detected tactics based on keywords
        keywords = session.intel_buffer.get("suspiciousKeywords", [])
        if keywords:
            # Group by tactic type
            urgency_words = [k for k in keywords if k in ["urgent", "immediately", "verify now"]]
            threat_words = [k for k in keywords if k in ["arrest", "police", "legal action", "fine", "penalty", "blocked", "suspended"]]
            lure_words = [k for k in keywords if k in ["refund", "cashback", "lottery", "winner", "prize"]]

            if urgency_words:
                notes.append("Scammer used urgency tactics")
            if threat_words:
                notes.append("Scammer used threatening language")
            if lure_words:
                notes.append("Scammer used financial lure tactics")

        # Summarize extracted data
        if session.intel_buffer.get("upiIds"):
            notes.append("Attempted UPI payment extraction")
        if session.intel_buffer.get("bankAccounts"):
            notes.append("Bank account details extracted")
        if session.intel_buffer.get("phoneNumbers"):
            notes.append("Scammer shared phone contact")
        if session.intel_buffer.get("phishingLinks"):
            notes.append("Phishing links detected")

        # Add scam score context
        if session.scam_score >= 0.8:
            notes.append("High confidence scam detection")
        elif session.scam_score >= 0.5:
            notes.append("Medium confidence scam detection")

        # Return compiled notes or default
        if notes:
            return ". ".join(notes) + "."
        return "Scam engagement session completed without specific indicators."


# Global controller instance
_controller: AgentController | None = None


def get_controller() -> AgentController:
    """Get or create the controller singleton."""
    global _controller
    if _controller is None:
        _controller = AgentController()
    return _controller
