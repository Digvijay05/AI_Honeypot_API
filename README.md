# Agentic Honeypot API

An agentic AI system that engages scammers in automated, believable conversations, extracts intelligence (UPI IDs, phone numbers, phishing links), and reports findings via a mandatory callback to the GUVI evaluation server.

## Architecture

```
AI_honeypot_API/
├── backend/                        # FastAPI REST API
│   ├── app/
│   │   ├── main.py                 # FastAPI entrypoint + lifespan
│   │   ├── api/routes.py           # HTTP endpoints
│   │   ├── core/
│   │   │   ├── config.py           # Environment-based configuration
│   │   │   └── llm/                # LLM provider abstraction
│   │   │       ├── base.py         # Abstract LLMProvider interface
│   │   │       ├── factory.py      # Provider factory (singleton)
│   │   │       ├── ollama_provider.py
│   │   │       └── groq_provider.py
│   │   ├── models/
│   │   │   ├── schemas.py          # Pydantic request/response models
│   │   │   └── db_models.py        # SQLAlchemy ORM models
│   │   ├── services/
│   │   │   ├── agent.py            # AgentController + "Margaret" persona
│   │   │   ├── detection.py        # Regex-based scam detection engine
│   │   │   ├── extraction.py       # Intelligence extraction (UPI, phone, URL)
│   │   │   ├── session.py          # Session state + in-memory/Redis store
│   │   │   └── callback.py         # GUVI final result callback client
│   │   ├── middleware/auth.py      # x-api-key authentication
│   │   ├── logging/config.py       # JSON/simple logging configuration
│   │   └── db.py                   # SQLAlchemy async engine setup
│   ├── tests/                      # pytest test suite
│   ├── Dockerfile                  # Multi-stage production image
│   ├── requirements.txt            # Pinned Python dependencies
│   └── .env.example                # Environment variable template
├── frontend/                       # Reserved for dashboard UI
│   └── README.md
├── docker-compose.yml              # Backend + Redis orchestration
├── .gitignore
└── README.md
```

## Key Components

| Component | File | Purpose |
|-----------|------|---------|
| **Scam Detection** | `services/detection.py` | 17 regex rules with weighted scoring (threshold: 0.5) |
| **Agent Controller** | `services/agent.py` | Orchestrates detection → engagement → extraction → callback |
| **Intelligence Extraction** | `services/extraction.py` | Extracts UPI IDs, phone numbers, URLs, bank accounts, keywords |
| **Session Management** | `services/session.py` | In-memory (dev) or Redis (prod) session store |
| **GUVI Callback** | `services/callback.py` | Sends final intelligence to `hackathon.guvi.in` |
| **LLM Agent** | `services/agent.py` | "Margaret" persona via Ollama Cloud / Groq |

## Setup

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (for production)
- An LLM API key (Ollama Cloud or Groq)

### Local Development

```bash
# 1. Navigate to backend
cd backend

# 2. Create virtual environment
python -m venv .venv

# 3. Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt

# 5. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 6. Run the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
# From project root (where docker-compose.yml is)
docker compose up --build

# Or run in background
docker compose up -d --build

# View logs
docker compose logs -f honeypot-backend

# Stop
docker compose down
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HONEYPOT_API_KEY` | **Yes** | — | API key for `x-api-key` header auth |
| `OLLAMA_API_KEY` | **Yes** | — | Ollama Cloud API key |
| `OLLAMA_MODEL` | No | `gemma3:27b-cloud` | Ollama model identifier |
| `OLLAMA_BASE_URL` | No | `https://ollama.com` | Ollama Cloud endpoint |
| `GROQ_API_KEY` | No | `""` | Groq API key (fallback) |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Groq model identifier |
| `OPENAI_API_KEY` | No | `""` | OpenAI API key (legacy) |
| `REDIS_URL` | No | — | Redis URL (set by Docker Compose) |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `LOG_FORMAT` | No | `json` | `json` or `simple` |
| `MAX_SESSION_MESSAGES` | No | `20` | Max messages before session ends |

## API Usage

### Request

```
POST /api/honeypot/message
Header: x-api-key: <your-api-key>
Content-Type: application/json
```

```json
{
  "sessionId": "sess-abc123-def456",
  "message": {
    "sender": "scammer",
    "text": "Your bank account has been suspended. Verify immediately!",
    "timestamp": 1770005528731
  },
  "conversationHistory": [],
  "metadata": {
    "channel": "sms",
    "language": "en",
    "locale": "en-US"
  }
}
```

### Response

```json
{
  "status": "success",
  "reply": "Oh my, that sounds serious! What do I need to do?"
}
```

### Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/api/honeypot/message` | Yes | Main honeypot endpoint |
| `POST` | `/` | Yes | Root endpoint (GUVI tester compat) |
| `GET/POST` | `/api/honeypot/test` | Yes | Reachability check |
| `GET` | `/health` | No | Container health check |

## GUVI Final Callback

The system sends a mandatory callback to `https://hackathon.guvi.in/api/updateHoneyPotFinalResult` when **all three conditions** are met:

1. **`scamDetected = true`** — scam detection engine triggered
2. **Engagement complete** — session reached `MAX_SESSION_MESSAGES`
3. **Intelligence extracted** — accumulated throughout the conversation

```json
{
  "sessionId": "sess-abc123-def456",
  "scamDetected": true,
  "totalMessagesExchanged": 20,
  "extractedIntelligence": {
    "bankAccounts": [],
    "upiIds": ["scammer@ybl"],
    "phishingLinks": [],
    "phoneNumbers": ["9876543210"],
    "suspiciousKeywords": ["verify", "blocked", "otp"]
  },
  "agentNotes": "Scammer used urgency tactics. Attempted UPI payment extraction."
}
```

## Testing

```bash
cd backend
python -m pytest tests/ -v --tb=short
```

## License

Private — GUVI Hackathon submission.
