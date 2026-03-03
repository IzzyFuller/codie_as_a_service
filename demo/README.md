# Codie-as-a-Service Demo

Two demo options:
1. **HTTP Streaming** - Gradio chat UI with SSE streaming responses
2. **Pub/Sub Message-Driven** - CLI client with async message processing via RabbitMQ

Both demonstrate per-agent memory persistence.

## Prerequisites

- Docker installed and running (for Pub/Sub emulator)
- Main dependencies installed: `uv sync` (from repo root)
- `.env` file configured (copy from `.env.example`)
- Local LLM model downloaded (e.g., SmolLM3-3B via HuggingFace)

## Option 1: HTTP Streaming Demo (Gradio UI)

From the **repo root**:

```bash
set -a && source .env && set +a && ./demo/run-http-demo.sh
```

This single command:
1. Loads the local LLM model
2. Starts the HTTP service (port 8080)
3. Starts the Gradio demo UI (port 7860)

Then open http://localhost:7860

Press `Ctrl+C` to stop all services.

## Option 2: Pub/Sub Message-Driven Demo (CLI)

From the **repo root**:

```bash
# 1. Set up infrastructure first
./demo/setup-infrastructure.sh

# 2. Run the demo
set -a && source .env && set +a && ./demo/run-pubsub-demo.sh
```

This:
1. Starts RabbitMQ (Docker container)
2. Creates queues and exchanges
3. Creates demo agent
4. Starts the RabbitMQ agent service
5. Launches CLI client

Type messages at the `You:` prompt. Use `/quit` to exit.

Press `Ctrl+C` to stop all services.

## Environment Variables

Create a `.env` file in the repo root (see `.env.example`):

```bash
# Required - Storage
STORAGE_DIR=./data/agents

# Required - Prompt configuration
PROMPTS_DIR=./prompts
PROMPT_NAMES=codie_as_a_service_system

# Optional - Override defaults
HTTP_HOST=0.0.0.0
HTTP_PORT=8080
```

## Demo Features

- **Streaming responses**: Watch the agent think in real-time
- **Per-user memory**: Each agent_id gets isolated memory storage
- **Session persistence**: Memory persists across conversations for the same user

## Testing Memory Persistence

1. Chat with the agent using default agent ID
2. Tell it something memorable (e.g., "My favorite color is blue")
3. Close the chat and reopen
4. Ask "What's my favorite color?" - it should remember

## Troubleshooting

**Missing dependencies:**
```
ModuleNotFoundError: No module named 'gradio'
```
Run `uv sync --group demo` from repo root.

**Port already in use:**
Check for existing processes on ports 5672, 8080, or 7860.

**Model loading slow:**
First run downloads the model from HuggingFace. Subsequent runs use cached model.

## Architecture

### HTTP Streaming (Option 1)

```
User Browser (localhost:7860)
       │
       ▼
   Gradio UI (demo/app.py)
       │ HTTP POST /chat
       ▼
   HTTP Service (main_http.py:8080)
       │
       ├──▶ Local LLM (Transformers)
       │
       └──▶ Local Filesystem (./data/agents/)
            └── {agent_id}/*.md
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
       ├──▶ Local Filesystem (./data/agents/)
       │         └── {agent_id}/*.md
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
