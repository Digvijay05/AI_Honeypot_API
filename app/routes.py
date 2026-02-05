"""Routes for Honeypot API.

Delegates processing to AgentController for orchestration.
"""

from fastapi import APIRouter, Depends, Request

from app.agent import get_controller
from app.middleware.auth import verify_api_key
from app.schemas.request import HoneypotRequest, HoneypotResponse

router = APIRouter()


# -----------------------------
# Health Check (No Auth Required)
# -----------------------------
@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint for container orchestration.

    No authentication required. Used by Docker HEALTHCHECK and load balancers.

    Returns:
        Status dict indicating the service is healthy.
    """
    return {"status": "healthy"}

# -----------------------------
# Main Honeypot API (Evaluation)
# -----------------------------
@router.post("/api/honeypot/message", response_model=HoneypotResponse)
async def honeypot(
    req: HoneypotRequest,
    _: str = Depends(verify_api_key),
) -> HoneypotResponse:
    """Process incoming message and generate honeypot response.

    Args:
        req: The honeypot request containing message and history.

    Returns:
        HoneypotResponse with status and agent reply.
    """
    controller = get_controller()
    reply = await controller.process_message(
        session_id=req.sessionId,
        message=req.message,
        conversation_history=req.conversationHistory,
    )

    return HoneypotResponse(
        status="success",
        reply=reply,
    )


# -----------------------------------
# Root POST Endpoint for GUVI Tester
# -----------------------------------
@router.post("/", response_model=HoneypotResponse)
async def root_honeypot(
    req: HoneypotRequest,
    _: str = Depends(verify_api_key),
) -> HoneypotResponse:
    """Process incoming message at root path for GUVI tester compatibility.

    The GUVI Agentic Honey-Pot API Tester posts to the root URL.
    This endpoint mirrors /api/honeypot/message for compatibility.

    Args:
        req: The honeypot request containing message and history.

    Returns:
        HoneypotResponse with status and agent reply.
    """
    controller = get_controller()
    reply = await controller.process_message(
        session_id=req.sessionId,
        message=req.message,
        conversation_history=req.conversationHistory,
    )

    return HoneypotResponse(
        status="success",
        reply=reply,
    )


# -----------------------------------
# GUVI Honeypot Endpoint Tester (FINAL)
# -----------------------------------
@router.api_route("/api/honeypot/test", methods=["GET", "POST"])
async def test_endpoint(
    request: Request,
    _: str = Depends(verify_api_key),
) -> dict:
    """Test endpoint to verify honeypot is reachable.

    Args:
        request: The incoming request.

    Returns:
        Status dict indicating endpoint is reachable.
    """
    # IMPORTANT:
    # - Do NOT read body
    # - Do NOT validate body
    # - Just respond OK
    return {
        "status": "ok",
        "message": "Honeypot endpoint reachable",
    }
