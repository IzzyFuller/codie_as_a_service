# Codie-as-a-Service Demo

Two demo options:
1. **HTTP Streaming** - Gradio chat UI with SSE streaming responses
2. **Pub/Sub Message-Driven** - CLI client with async message processing via RabbitMQ

Both demonstrate per-agent memory persistence.

## Prerequisites

- Docker installed and running
- Main dependencies installed: `uv sync` (from repo root)
- `.env` file configured (copy from `.env.example`)
- Local LLM model downloaded (e.g., SmolLM3-3B via HuggingFace)

## Option 1: HTTP Streaming Demo (Gradio UI)

From the **repo root**:

```bash
set -a && source .env && set +a && ./demo/run-http-demo.sh
```

This single command:
1. Starts GCS emulator (Docker container)
2. Creates the memory bucket
3. Loads the local LLM model
4. Starts the HTTP service (port 8080)
5. Starts the Gradio demo UI (port 7860)

Then open http://localhost:7860

Press `Ctrl+C` to stop all services.

## Option 2: Pub/Sub Message-Driven Demo (CLI)

From the **repo root**:

```bash
set -a && source .env && set +a && ./demo/run-pubsub-demo.sh
```

This single command:
1. Starts GCS emulator (Docker container)
2. Starts RabbitMQ (Docker container)
3. Creates queues and exchanges
4. Creates demo agent
5. Starts the RabbitMQ agent service
6. Launches CLI client

Type messages at the `You:` prompt. Use `/quit` to exit.

Press `Ctrl+C` to stop all services.

## Environment Variables

Create a `.env` file in the repo root (see `.env.example`):

```bash
# Required - Local LLM
MODEL_NAME=HuggingFaceTB/SmolLM3-3B
DEVICE=mps  # mps (Apple Silicon), cuda, or cpu

# Required - Authentication
API_KEY=your-secret-api-key

# Required - Prompt configuration
PROMPTS_DIR=./prompts
PROMPT_NAMES=codie_as_a_service_system

# Optional - Override defaults
HTTP_HOST=0.0.0.0
HTTP_PORT=8080
GCS_BUCKET_NAME=deep-agent-memory
```

## Demo Features

- **Streaming responses**: Watch the agent think in real-time
- **Per-user memory**: Each agent_id gets isolated memory storage
- **Session persistence**: Memory persists across conversations for the same user
- **API key authentication**: HTTP endpoint protected by X-API-Key header

## Testing Memory Persistence

1. Chat with the agent using default agent ID
2. Tell it something memorable (e.g., "My favorite color is blue")
3. Close the chat and reopen
4. Ask "What's my favorite color?" - it should remember

## Troubleshooting

**Docker not running:**
```
ERROR: GCS emulator failed to start
```
Start Docker and try again.

**Missing dependencies:**
```
ModuleNotFoundError: No module named 'gradio'
```
Run `uv sync --group demo` from repo root.

**Port already in use:**
Check for existing processes on ports 4443, 5672, 8080, or 7860.

**Model loading slow:**
First run downloads the model from HuggingFace. Subsequent runs use cached model.

## Architecture

### HTTP Streaming (Option 1)

```
User Browser (localhost:7860)
       │
       ▼
   Gradio UI (demo/app.py)
       │ HTTP POST /chat (X-API-Key header)
       ▼
   HTTP Service (main_http.py:8080)
       │
       ├──▶ Local LLM (Transformers)
       │
       └──▶ GCS Emulator (localhost:4443)
            └── agents/{agent_id}/*.md
```

### Pub/Sub Message-Driven (Option 2)

```
   CLI Client (pubsub_cli.py)
       │
       │ publish request
       ▼
   RabbitMQ (localhost:5672)
   ┌───────────────────────────────┐
   │  agent.requests (queue)       │
   └───────────────────────────────┘
       │
       │ consume
       ▼
   Agent Service (main_pubsub.py)
       │
       ├──▶ Local LLM (Transformers)
       │
       ├──▶ GCS Emulator (localhost:4443)
       │         └── agents/{agent_id}/*.md
       │
       │ publish response
       ▼
   ┌───────────────────────────────┐
   │  agent.responses (exchange)   │
   └───────────────────────────────┘
       │
       │ consume response
       ▼
   CLI Client (pubsub_cli.py)
```

## Note

This is for **demo purposes only** - not intended for production deployment.
