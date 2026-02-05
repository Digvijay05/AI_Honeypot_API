# ==============================================================================
# Dockerfile for Honeypot API
# ==============================================================================
# Multi-stage build with non-root user and health check.
# Production-ready for hackathon evaluation.
# ==============================================================================

# Stage 1: Build dependencies
FROM python:3.10-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/install -r requirements.txt

# Stage 2: Production image
FROM python:3.10-slim

# Metadata
LABEL maintainer="Honeypot Team"
LABEL description="Agentic Honeypot API for scam detection and engagement"
LABEL version="1.0.0"

# Environment settings for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/usr/local/lib/python3.10/site-packages

# Create non-root user for security
RUN groupadd -r honeypot && useradd -r -g honeypot honeypot

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local/lib/python3.10/site-packages/

# Copy application code
COPY app ./app

# Set ownership to non-root user
RUN chown -R honeypot:honeypot /app

# Switch to non-root user
USER honeypot

# Expose port
EXPOSE 8000

# Health check - verifies the API is responding
# Interval: Check every 30s
# Timeout: Fail if no response in 10s
# Start-period: Wait 10s before first check
# Retries: Mark unhealthy after 3 consecutive failures
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
