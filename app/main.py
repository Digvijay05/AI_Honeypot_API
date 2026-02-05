"""FastAPI application entry point.

Per TODO.md Phase 3:
- Add examples to HoneypotRequest and HoneypotResponse schema.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.db import init_db_async
from app.logging import setup_logging
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown."""
    # Startup: Configure logging first
    setup_logging()
    # Startup: Initialize database
    await init_db_async()
    yield
    # Shutdown: Cleanup (if needed)


app = FastAPI(
    title="Agentic Honeypot API",
    version="1.0.0",
    description="A backend system designed to engage scammers in automated, believable conversations.",
    lifespan=lifespan,
)
app.include_router(router)
