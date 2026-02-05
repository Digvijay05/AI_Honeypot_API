<#
.SYNOPSIS
    Start Honeypot API with Docker and expose via ngrok.

.DESCRIPTION
    This script starts the Docker containers and creates an ngrok tunnel
    to expose the API publicly for GUVI evaluation.

.NOTES
    Prerequisites:
    - Docker Desktop running
    - ngrok installed and authenticated (ngrok config add-authtoken YOUR_TOKEN)
    - .env file with HONEYPOT_API_KEY and OPENAI_API_KEY

.EXAMPLE
    .\scripts\start-with-ngrok.ps1
#>

param(
    [int]$Port = 8000,
    [string]$Region = "in",
    [switch]$NoBuild
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Success { param($Message) Write-Host "[OK] $Message" -ForegroundColor Green }
function Write-Info { param($Message) Write-Host "[INFO] $Message" -ForegroundColor Cyan }
function Write-Warn { param($Message) Write-Host "[WARN] $Message" -ForegroundColor Yellow }
function Write-Err { param($Message) Write-Host "[ERROR] $Message" -ForegroundColor Red }

Write-Host ""
Write-Host "========================================" -ForegroundColor Magenta
Write-Host " Honeypot API - Docker + ngrok Startup" -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta
Write-Host ""

# Check prerequisites
Write-Info "Checking prerequisites..."

# Check ngrok
try {
    $null = ngrok version 2>&1
    Write-Success "ngrok is available"
} catch {
    Write-Err "ngrok is not installed"
    Write-Host "  Install from: https://ngrok.com/download"
    Write-Host "  Then run: ngrok config add-authtoken YOUR_AUTHTOKEN"
    exit 1
}

# Check Docker
try {
    $null = docker version 2>&1
    Write-Success "Docker is available"
} catch {
    Write-Err "Docker is not running or not installed"
    Write-Host "  Install from: https://docs.docker.com/desktop/install/windows-install/"
    exit 1
}



# Check .env file
if (-not (Test-Path ".env")) {
    Write-Err ".env file not found"
    Write-Host "  Copy .env.example to .env and fill in your secrets:"
    Write-Host "    cp .env.example .env"
    Write-Host "    notepad .env"
    exit 1
}
Write-Success ".env file exists"

# Start Docker containers
Write-Host ""
Write-Info "Starting Docker containers..."

if ($NoBuild) {
    docker compose up -d
} else {
    docker compose up -d --build
}

if ($LASTEXITCODE -ne 0) {
    Write-Err "Failed to start Docker containers"
    exit 1
}

Write-Success "Docker containers started"

# Wait for backend to be healthy
Write-Info "Waiting for backend to be healthy..."
$maxAttempts = 30
$attempt = 0

while ($attempt -lt $maxAttempts) {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:$Port/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.status -eq "healthy") {
            Write-Success "Backend is healthy"
            break
        }
    } catch {
        # Ignore errors, keep trying
    }
    
    $attempt++
    if ($attempt -ge $maxAttempts) {
        Write-Err "Backend did not become healthy within 60 seconds"
        Write-Host "  Check logs: docker compose logs honeypot-backend"
        exit 1
    }
    Start-Sleep -Seconds 2
}

# Get API key from .env for display
$apiKey = (Get-Content .env | Where-Object { $_ -match "^HONEYPOT_API_KEY=" }) -replace "HONEYPOT_API_KEY=", ""

# Start ngrok
Write-Host ""
Write-Info "Starting ngrok tunnel..."
Write-Host "  Local: http://localhost:$Port"
Write-Host "  Region: $Region"
Write-Host ""

Write-Host "========================================" -ForegroundColor Green
Write-Host " ngrok will display your public URL" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Once ngrok starts, your public URL will be shown (e.g., https://xxxx.ngrok-free.app)"
Write-Host ""
Write-Host "Test commands (replace URL with your actual ngrok URL):" -ForegroundColor Yellow
Write-Host ""
Write-Host "  # Health check (no auth):" -ForegroundColor Gray
Write-Host "  curl https://YOUR-URL.ngrok-free.app/health"
Write-Host ""
Write-Host "  # API test (with auth):" -ForegroundColor Gray
Write-Host "  curl -X POST https://YOUR-URL.ngrok-free.app/api/honeypot/test -H 'x-api-key: $apiKey'"
Write-Host ""
Write-Host "Press Ctrl+C to stop ngrok (containers keep running)." -ForegroundColor Yellow
Write-Host ""

# Start ngrok (this blocks until Ctrl+C)
ngrok http $Port --region $Region
