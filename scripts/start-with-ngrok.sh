#!/usr/bin/env bash
# ==============================================================================
# Start Honeypot API with Docker and expose via ngrok
# ==============================================================================
# Prerequisites:
# - Docker and Docker Compose installed
# - ngrok installed and authenticated (ngrok config add-authtoken YOUR_TOKEN)
# - .env file with HONEYPOT_API_KEY and OPENAI_API_KEY
# ==============================================================================

set -e

PORT=${1:-8000}
REGION=${2:-in}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

success() { echo -e "${GREEN}[OK]${NC} $1"; }
info() { echo -e "${CYAN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo ""
echo -e "${MAGENTA}========================================"
echo " Honeypot API - Docker + ngrok Startup"
echo -e "========================================${NC}"
echo ""

# Check prerequisites
info "Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    error "Docker is not installed"
    echo "  Install from: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker info &> /dev/null; then
    error "Docker daemon is not running"
    echo "  Start Docker Desktop or run: sudo systemctl start docker"
    exit 1
fi
success "Docker is available"

# Check ngrok
if ! command -v ngrok &> /dev/null; then
    error "ngrok is not installed"
    echo "  Install from: https://ngrok.com/download"
    echo "  Then run: ngrok config add-authtoken YOUR_AUTHTOKEN"
    exit 1
fi
success "ngrok is available"

# Check .env file
if [ ! -f ".env" ]; then
    error ".env file not found"
    echo "  Copy .env.example to .env and fill in your secrets:"
    echo "    cp .env.example .env"
    echo "    nano .env"
    exit 1
fi
success ".env file exists"

# Start Docker containers
echo ""
info "Starting Docker containers..."

docker compose up -d --build

if [ $? -ne 0 ]; then
    error "Failed to start Docker containers"
    exit 1
fi

success "Docker containers started"

# Wait for backend to be healthy
info "Waiting for backend to be healthy..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s "http://localhost:${PORT}/health" | grep -q "healthy"; then
        success "Backend is healthy"
        break
    fi
    
    attempt=$((attempt + 1))
    if [ $attempt -ge $max_attempts ]; then
        error "Backend did not become healthy within 60 seconds"
        echo "  Check logs: docker compose logs honeypot-backend"
        exit 1
    fi
    sleep 2
done

# Get API key from .env for display
API_KEY=$(grep "^HONEYPOT_API_KEY=" .env | cut -d'=' -f2)

# Start ngrok
echo ""
info "Starting ngrok tunnel..."
echo "  Local: http://localhost:${PORT}"
echo "  Region: ${REGION}"
echo ""

echo -e "${GREEN}========================================"
echo " ngrok will display your public URL"
echo -e "========================================${NC}"
echo ""
echo "Once ngrok starts, your public URL will be shown (e.g., https://xxxx.ngrok-free.app)"
echo ""
echo -e "${YELLOW}Test commands (replace URL with your actual ngrok URL):${NC}"
echo ""
echo -e "  ${CYAN}# Health check (no auth):${NC}"
echo "  curl https://YOUR-URL.ngrok-free.app/health"
echo ""
echo -e "  ${CYAN}# API test (with auth):${NC}"
echo "  curl -X POST https://YOUR-URL.ngrok-free.app/api/honeypot/test -H 'x-api-key: ${API_KEY}'"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop ngrok (containers keep running).${NC}"
echo ""

# Start ngrok (this blocks until Ctrl+C)
ngrok http "${PORT}" --region "${REGION}"
