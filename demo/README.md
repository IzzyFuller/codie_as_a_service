# Deep Agent Demo

Two demo options for the deep agent service:
1. **HTTP Streaming** - Gradio chat UI with SSE streaming responses
2. **Pub/Sub Message-Driven** - CLI client with async message processing

Both demonstrate per-user memory persistence.

## Prerequisites

- Docker installed and running
- Main dependencies installed: `uv sync --no-dev` (from repo root)
- `.env` file configured (see below)

## Option 1: HTTP Streaming Demo (Gradio UI)

From the **repo root**:

```bash
set -a && source .env && set +a && ./demo/run-http-demo.sh
```

This single command:
1. Starts GCS emulator (Docker container)
2. Creates the memory bucket
3. Starts the HTTP service (port 8080)
4. Starts the Gradio demo UI (port 7860)

Then open http://localhost:7860

Press `Ctrl+C` to stop all services.

## Option 2: Pub/Sub Message-Driven Demo (CLI)

From the **repo root**:

```bash
set -a && source .env && set +a && ./demo/run-pubsub-demo.sh
```

This single command:
1. Starts GCS emulator (Docker container)
2. Starts Pub/Sub emulator (Docker container)
3. Creates topics and subscriptions
4. Creates demo user
5. Starts the Pub/Sub agent service
6. Launches CLI client

Type messages at the `You:` prompt. Use `/quit` to exit.

Press `Ctrl+C` to stop all services.

## Environment Variables

Create a `.env` file in the repo root with:

```bash
# Required - LLM API
OPENROUTER_API_KEY=your-openrouter-key
OPENROUTER_MODEL=anthropic/claude-sonnet-4-20250514

# Required - Langfuse (prompt management)
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_PROMPT_LABEL=codie_as_a_service

# Required - Prompt configuration
PROMPT_NAMES=codie_as_a_service_system

# Optional - Override defaults
HTTP_HOST=0.0.0.0
HTTP_PORT=8080
GCS_BUCKET_NAME=deep-agent-memory
```

## Manual Startup (Alternative)

If you prefer to run services separately:

**Terminal 1** - Start GCS emulator and HTTP service:
```bash
./scripts/start-local.sh
```

**Terminal 2** - Start demo UI (preferred):
```bash
set -a && source .env && set +a && ./demo/run-http-demo.sh
```

Or for Pub/Sub CLI demo:
```bash
set -a && source .env && set +a && ./demo/run-pubsub-demo.sh
```

## Demo Features

- **Streaming responses**: Watch the agent think in real-time
- **Per-user memory**: Each user_id gets isolated memory storage
- **Session persistence**: Memory persists across conversations for the same user

## Testing Memory Persistence

1. Chat with the agent using default user ID
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
Run `uv sync --no-dev` from repo root.

**Port already in use:**
Check for existing processes on ports 4443, 8080, or 7860.

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
       ├──▶ Anthropic API (via OpenRouter)
       │
       └──▶ GCS Emulator (localhost:4443)
            └── users/{user_id}/*.md
```

### Pub/Sub Message-Driven (Option 2)

```
   CLI Client (pubsub_cli.py)
       │
       │ publish request
       ▼
   Pub/Sub Emulator (localhost:8085)
   ┌───────────────────────────────┐
   │  agent-requests (topic)       │
   │       │                       │
   │       ▼                       │
   │  agent-requests-sub           │
   └───────────────────────────────┘
       │
       │ subscribe
       ▼
   Agent Service (main_pubsub.py)
       │
       ├──▶ Anthropic API (via OpenRouter)
       │
       ├──▶ GCS Emulator (localhost:4443)
       │         └── users/{user_id}/*.md
       │
       │ publish response
       ▼
   ┌───────────────────────────────┐
   │  agent-responses (topic)      │
   │       │                       │
   │       ▼                       │
   │  agent-responses-sub          │
   └───────────────────────────────┘
       │
       │ poll for response
       ▼
   CLI Client (pubsub_cli.py)
```

## Note

This is for **demo purposes only** - not intended for production deployment.
