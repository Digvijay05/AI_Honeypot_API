"""End-to-End scam flow test.

Simulates the complete lifecycle:
1. First message (safe) -> minimal reply
2. Scam message -> agent activated, intel extracted
3. Max messages -> callback triggered

Per TODO.md Phase 9:
- Simulate the evaluator.
- Verify complete conversation lifecycle with single callback.
"""

import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

# Patch config before importing app
with patch.dict(
    "os.environ",
    {
        "HONEYPOT_API_KEY": "test-key",
        "OPENAI_API_KEY": "test-openai-key",
        "MAX_SESSION_MESSAGES": "3",  # Low for testing
        "LOG_FORMAT": "simple",
    },
):
    from app.main import app

client = TestClient(app)


class TestE2EScamFlow:
    """End-to-end tests for scam conversation lifecycle."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset session store before each test."""
        # Clear session store using public API
        from app.session import get_session_store
        store = get_session_store()
        store._store.clear()

    def test_full_scam_lifecycle(self):
        """Test complete lifecycle: safe -> scam -> callback."""
        session_id = "e2e-test-session-001"
        headers = {"x-api-key": "test-key"}
        ts = "2026-02-04T22:00:00Z"

        # Mock both agent_reply and send_callback
        with patch("app.agent.controller.agent_reply", new_callable=AsyncMock) as mock_agent, \
             patch("app.agent.controller.send_callback", new_callable=AsyncMock) as mock_callback:

            mock_agent.return_value = "Oh dear, what's happening? My bank account?"
            mock_callback.return_value = True

            # Turn 1: Safe message
            resp1 = client.post(
                "/api/honeypot/message",
                json={
                    "sessionId": session_id,
                    "message": {"sender": "scammer", "text": "Hello there!", "timestamp": ts},
                    "conversationHistory": [],
                },
                headers=headers,
            )
            assert resp1.status_code == 200
            data1 = resp1.json()
            assert data1["status"] == "success"
            assert data1["reply"] == "Okay."  # Non-scam minimal reply
            mock_agent.assert_not_called()

            # Turn 2: Scam message with UPI
            resp2 = client.post(
                "/api/honeypot/message",
                json={
                    "sessionId": session_id,
                    "message": {"sender": "scammer", "text": "Your account is blocked! Send 500 to fraud@ybl", "timestamp": ts},
                    "conversationHistory": [
                        {"sender": "scammer", "text": "Hello there!", "timestamp": ts},
                        {"sender": "agent", "text": "Okay.", "timestamp": ts},
                    ],
                },
                headers=headers,
            )
            assert resp2.status_code == 200
            data2 = resp2.json()
            assert data2["status"] == "success"
            assert data2["reply"] == "Oh dear, what's happening? My bank account?"
            mock_agent.assert_called_once()  # Agent was activated

            # Turn 3: Reach max messages (3), trigger callback
            mock_agent.reset_mock()
            resp3 = client.post(
                "/api/honeypot/message",
                json={
                    "sessionId": session_id,
                    "message": {"sender": "scammer", "text": "Send money NOW to 9876543210!", "timestamp": ts},
                    "conversationHistory": [
                        {"sender": "scammer", "text": "Hello there!", "timestamp": ts},
                        {"sender": "agent", "text": "Okay.", "timestamp": ts},
                        {"sender": "scammer", "text": "Your account is blocked! Send 500 to fraud@ybl", "timestamp": ts},
                        {"sender": "agent", "text": "Oh dear, what's happening? My bank account?", "timestamp": ts},
                    ],
                },
                headers=headers,
            )
            assert resp3.status_code == 200
            data3 = resp3.json()
            assert data3["status"] == "success"
            # At max messages, should get the stopping message
            assert "go now" in data3["reply"].lower() or "thank you" in data3["reply"].lower()

            # Verify callback was triggered
            mock_callback.assert_called_once()
            call_args = mock_callback.call_args
            assert call_args.kwargs["session_id"] == session_id
            assert call_args.kwargs["scam_detected"] is True

            # Verify intel was extracted (UPI and phone number)
            intel = call_args.kwargs["intel_buffer"]
            assert "fraud@ybl" in intel.get("upiIds", [])
            assert "9876543210" in intel.get("phoneNumbers", [])

    def test_callback_only_sent_once(self):
        """Verify callback is not sent multiple times for same session."""
        session_id = "e2e-test-session-002"
        headers = {"x-api-key": "test-key"}
        ts = "2026-02-04T22:00:00Z"

        with patch("app.agent.controller.agent_reply", new_callable=AsyncMock) as mock_agent, \
             patch("app.agent.controller.send_callback", new_callable=AsyncMock) as mock_callback:

            mock_agent.return_value = "Oh no, is my money safe?"
            mock_callback.return_value = True

            # Send 4 messages (above MAX_SESSION_MESSAGES=3)
            for i in range(4):
                client.post(
                    "/api/honeypot/message",
                    json={
                        "sessionId": session_id,
                        "message": {"sender": "scammer", "text": f"Urgent! Pay now to scam{i}@ybl", "timestamp": ts},
                        "conversationHistory": [],
                    },
                    headers=headers,
                )

            # Callback should have been called exactly once
            assert mock_callback.call_count == 1

    def test_no_callback_for_non_scam_session(self):
        """Non-scam sessions should not trigger callback."""
        session_id = "e2e-test-session-003"
        headers = {"x-api-key": "test-key"}
        ts = "2026-02-04T22:00:00Z"

        with patch("app.agent.controller.send_callback", new_callable=AsyncMock) as mock_callback:
            # Send only safe messages
            for i in range(5):
                client.post(
                    "/api/honeypot/message",
                    json={
                        "sessionId": session_id,
                        "message": {"sender": "user", "text": f"Hello, how are you? {i}", "timestamp": ts},
                        "conversationHistory": [],
                    },
                    headers=headers,
                )

            # Callback should never be called for non-scam
            mock_callback.assert_not_called()
