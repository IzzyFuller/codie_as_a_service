# Codie-as-a-Service

![CI](https://github.com/IzzyFuller/codie_as_a_service/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/IzzyFuller/codie_as_a_service/branch/main/graph/badge.svg)](https://codecov.io/gh/IzzyFuller/codie_as_a_service)
![Python](https://img.shields.io/badge/python-3.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A microservice that gives AI agents persistent identity and memory. Each agent gets isolated memory, a behavioral frame, and a 5-phase ReAct orchestration pipeline. Point it at any agent's memory directory, and it runs.

## How It Works

Every request flows through five phases:

1. **HYDRATE** — Summarize the agent's identity from memory files into a system prompt
2. **EXTEND** — Enrich the user's instruction with relevant context from memory (uses tools)
3. **PROCESS** — Execute the instruction using the LLM with available tools
4. **VALIDATE** — Check if the response satisfies the instruction; retry if not (max 3 iterations)
5. **SYNTHESIZE** — Persist the full interaction to session memory (no LLM call)

HYDRATE and EXTEND run once (skipped on retry). PROCESS and VALIDATE loop until done.

## Quick Start

```bash
# Clone and install
git clone https://github.com/IzzyFuller/codie_as_a_service.git
cd codie_as_a_service
uv sync

# Copy environment config
cp .env.example .env

# Start the HTTP server (uses Claude CLI by default)
./scripts/start-local-http.sh

# Optionally create an agent on startup
./scripts/start-local-http.sh myagent
```

Test it:

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: local-dev-key" \
  -d '{"agent_id": "myagent", "session_id": "1", "message": "Hello!"}' \
  --no-buffer
```

## Running the Pub/Sub Service

For RabbitMQ-based async messaging instead of HTTP:

```bash
# Start RabbitMQ (Docker)
docker run -d --name rabbitmq -p 5672:5672 rabbitmq:3-management

# Start the service
STORAGE_DIR=./data/agents \
  PROMPTS_DIR=./prompts \
  PROMPT_NAMES=codie_as_a_service_system \
  LLM_ADAPTER=claude_cli \
  uv run python -m codie_as_a_service.main_pubsub
```

Or with a local LLM on Apple Silicon:

```bash
STORAGE_DIR=./data/agents \
  PROMPTS_DIR=./prompts \
  PROMPT_NAMES=codie_as_a_service_system \
  LLM_ADAPTER=local \
  MODEL_NAME=HuggingFaceTB/SmolLM3-3B \
  DEVICE=mps \
  uv run python -m codie_as_a_service.main_pubsub
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `STORAGE_DIR` | `./data/agents` | Agent memory root directory |
| `STORAGE_PATH_TEMPLATE` | `agents/{agent_id}` | Path template within storage dir |
| `LLM_ADAPTER` | `claude_cli` | `claude_cli` or `local` |
| `MODEL_NAME` | — | HuggingFace model ID (required when `local`) |
| `DEVICE` | `mps` | `mps`, `cuda`, or `cpu` (when `local`) |
| `PROMPTS_DIR` | `./prompts` | Directory containing prompt files |
| `PROMPT_NAMES` | — | Comma-separated prompt file names |
| `API_KEY` | — | HTTP endpoint auth key |
| `HTTP_HOST` | `0.0.0.0` | HTTP bind address |
| `HTTP_PORT` | `8080` | HTTP port |
| `BROKER_URL` | `amqp://guest:guest@localhost:5672/` | RabbitMQ connection URL |
| `REQUEST_SUBSCRIPTION` | `agent.requests` | RabbitMQ request queue |
| `RESPONSE_TOPIC` | `agent.responses` | RabbitMQ response exchange |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

See `.env.example` for a copyable template.

## Agent Memory Structure

Each agent needs a directory with at least these files:

```
{agent_id}/
├── me.md                # Who the agent is (identity)
├── frame.md             # Behavioral constraints (required)
├── context_anchors.md   # Priority-ranked current context
└── current_session.md   # Working memory (written by SYNTHESIZE)
```

`frame.md` is required — the service returns an error if it's missing. The others are read during HYDRATE to build the agent's identity summary.

## HTTP API

**`POST /chat`** — Streaming chat (requires `X-API-Key` header)

```json
{
  "agent_id": "demo-user",
  "session_id": "unique-session-id",
  "message": "Hello, who are you?"
}
```

Returns SSE stream:
- `event: response` — `{"output": "...", "session_id": "...", "done": true}`
- `event: done` — `{"usage": {...}}`
- `event: error` — `{"message": "..."}`

**`GET /health`** — Health check (no auth required)

## Project Structure

```
src/codie_as_a_service/
├── core/                  # Domain models, protocols, phase output models
├── services/
│   ├── agent/             # ReActOrchestrator (phase loop)
│   ├── memory/            # MemoryService (read/write/list)
│   └── phases/            # LLMPhaseDefinition, TextLLMPhaseDefinition, SynthesizePhaseDefinition
├── adapters/
│   ├── auth/              # API key authentication
│   ├── llm/               # Claude CLI adapter, Local MLX adapter
│   ├── prompts/           # File-based prompt loading
│   ├── storage/           # Local filesystem storage
│   └── messaging/         # RabbitMQ pub/sub handler
├── main_http.py           # HTTP entry point (FastAPI + SSE)
└── main_pubsub.py         # Pub/Sub entry point (RabbitMQ)
```

## Development

### Running Tests

```bash
# All tests (excludes pub/sub — needs Docker)
uv run pytest -k "not pubsub" -v

# With coverage
uv run pytest -k "not pubsub" --cov=src/codie_as_a_service --cov-report=term-missing

# Pub/Sub integration tests (requires Docker RabbitMQ)
uv run pytest -k "pubsub" -v
```

### Code Quality

```bash
# Format
uv run ruff format .

# Lint
uv run ruff check .

# Type check
uv run mypy src/
```

### Test Architecture

Tests use a `TestApp` abstraction that hides adapter wiring. Tests describe behavior from the client perspective — they don't know about internal services, handlers, or adapters. Both LLM adapters (Claude CLI, Local) are parameterized and run against every test.

### Gradio Demo

```bash
# Terminal 1: start HTTP server
./scripts/start-local-http.sh

# Terminal 2: start Gradio UI
uv run --group demo python demo/app.py
# Opens at http://localhost:7860
```

## License

MIT
