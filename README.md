# Codie-as-a-Service

![CI](https://github.com/IzzyFuller/codie_as_a_service/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/IzzyFuller/codie_as_a_service/branch/main/graph/badge.svg)](https://codecov.io/gh/IzzyFuller/codie_as_a_service)
![Python](https://img.shields.io/badge/python-3.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Microservice for per-agent agent identity and memory - a prototype for "Codie-as-a-Service".

## Overview

A standalone microservice that enables per-agent agent identity and memory isolation. Each agent gets their own memory space, enabling personalized AI agents at scale.

## Architecture

### Hexagonal Architecture (Ports & Adapters)

```
codie_as_a_service/
├── src/codie_as_a_service/
│   ├── core/                  # Domain layer (models, protocols)
│   ├── services/              # Business logic layer
│   │   ├── agent/            # ReAct agent implementation
│   │   ├── memory/           # Agent memory management
│   │   └── tools/            # Agent tool system
│   ├── adapters/             # Infrastructure layer
│   │   ├── auth/             # Authentication adapters
│   │   ├── llm/              # LLM adapters (Claude CLI, Local Transformers)
│   │   ├── prompts/          # File-based prompt templates
│   │   ├── storage/          # Local filesystem storage adapter
│   │   └── messaging/        # RabbitMQ adapter
│   └── api/                  # Entry points
└── tests/                    # Mirror structure for tests
```

### Design Principles

- **Hexagonal Architecture**: Ports & adapters pattern for easy swapping of infrastructure
- **TDD**: 100% test coverage with mock-at-boundaries strategy
- **Message-Driven**: RabbitMQ for async execution (no threading complexity)

## Technology Stack

### Core Dependencies
- **Python**: 3.13.x
- **FastAPI**: HTTP API with streaming SSE responses
- **Pydantic**: Type-safe models throughout
- **pika**: RabbitMQ client for pub/sub messaging

### LLM Adapters
- **Claude CLI**: Uses Claude Code CLI (`claude -p`) - recommended for local dev
- **Transformers**: Native HuggingFace model loading (supports local models like SmolLM3)

### Storage
- **Local Filesystem**: File-based memory storage with configurable path templates

### Dev Dependencies
- **pytest**: Testing framework with coverage and async support
- **ruff**: Fast linting and formatting
- **mypy**: Static type checking
- **Docker**: RabbitMQ for integration tests

## POC Feature Scope

### In Scope (Minimal Viable Feature Set)

1. **Agent Identity & Isolation**: Each agent gets unique `agent_id` with isolated memory
2. **Basic Memory System**:
   - `current_session.md` - Working memory for active conversation
   - `context_anchors.md` - Priority-ranked current context
   - Simple key-value memory operations (read/write)
3. **Simple ReAct Agent**:
   - Reason → Act → Observe loop (max 10 iterations)
   - Native Transformers integration (local models)
   - Structured outputs via Pydantic
4. **Session Management**: Create, resume, and end user sessions
5. **Tool System**: 3 tools (read_memory, write_memory, + 1 domain tool)
6. **Message Handler**: Process RunAgentRequest from Pub/Sub

### Out of Scope (Future Enhancements)

- Customer-level multi-tenancy (`customer_id`)
- Advanced memory (deep merge, semantic search, compaction)
- Planning system (task decomposition, status tracking)
- Cost tracking and token attribution
- Rate limiting
- Multiple prompt versions/labels
- Dynamic tool loading, MCP integration

## Setup

### Prerequisites

- **Python 3.13.x** - Required (strict version constraint)
- **uv** - Package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Claude Code CLI** - For `claude_cli` LLM adapter ([install guide](https://docs.anthropic.com/en/docs/claude-code))
- **Docker** - Only needed for running Pub/Sub integration tests

### Installation

```bash
# Clone repository
git clone https://github.com/IzzyFuller/codie_as_a_service.git
cd codie_as_a_service

# Install all dependencies (main + dev + demo)
uv sync

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings
```

### Configuration

See `.env.example` for all available options. Key settings:

```bash
# Required
STORAGE_DIR=./data/agents           # Where agent memory is stored
API_KEY=your-secret-api-key         # For HTTP endpoint authentication
```

## Running Locally

### Quick Start

```bash
# 1. Copy and configure environment
cp .env.example .env
# Edit .env - defaults work for local mode

# 2. Start the HTTP server with local storage + Claude CLI
./scripts/start-local-http.sh

# 3. Optionally create a user on startup
./scripts/start-local-http.sh myuser
```

This starts the service with:
- **Storage**: Local filesystem (`./data/agents/`)
- **LLM**: Claude Code CLI (uses your installed `claude` command)
- **API**: http://localhost:8080

### Test the API

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: local-dev-key" \
  -d '{"agent_id": "test", "session_id": "1", "message": "Hello!"}' \
  --no-buffer
```

### Running with Gradio UI

```bash
# Start HTTP server in one terminal
./scripts/start-local-http.sh

# Start Gradio UI in another terminal
uv run --group demo python demo/app.py
# Opens at http://localhost:7860
```

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `STORAGE_DIR` | `./data/agents` | Path for local storage |
| `LLM_ADAPTER` | `claude_cli` | `claude_cli` or `local` |
| `MODEL_NAME` | - | HuggingFace model (when `LLM_ADAPTER=local`) |
| `DEVICE` | `mps` | `mps`, `cuda`, or `cpu` (when `LLM_ADAPTER=local`) |
| `API_KEY` | - | API key for authentication |

See `.env.example` for all options.

### HTTP Streaming API

The service exposes a streaming chat endpoint using Server-Sent Events (SSE):

**Endpoint:** `POST /chat`

**Request:**
```json
{
  "agent_id": "demo-user",
  "session_id": "unique-session-id",
  "message": "Hello, who are you?"
}
```

**Response:** SSE stream with events:
- `event: text` - Streamed response text chunks
- `event: tool_use` - Tool invocation notifications
- `event: error` - Error messages
- `event: done` - Stream completion

**Example with curl:**
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"agent_id": "demo-user", "session_id": "test", "message": "Hello!"}' \
  --no-buffer
```

Note: The `/chat` endpoint requires authentication via `X-API-Key` header. The `/health` endpoint is open.

## Testing

### Running Tests

```bash
# Run all tests with coverage
uv run pytest --cov=src/codie_as_a_service --cov-report=term-missing

# Run specific test file
uv run pytest tests/services/agent/test_react_agent.py

# Run with verbose output
uv run pytest -v
```

### Test Strategy

- **Mock at boundaries**: LLM model, RabbitMQ
- **Don't mock our logic**: ReAct loop, memory service, tool registry
- **TestApp abstraction**: Adapter-agnostic test interface

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type check
uv run mypy src/
```

## Memory Structure

Each agent gets isolated memory on the local filesystem:

```
data/agents/
└── {agent_id}/
    ├── current_session.md       # Working memory
    ├── context_anchors.md       # Priority context
    └── conversations/
        └── {session_id}.json    # Conversation history
```

## Development Workflow

1. **Write failing test** (`tests/`)
2. **Implement minimum code** to pass (`src/`)
3. **Refactor** for clarity
4. **Run linting** and type checks
5. **Commit** with descriptive message

## Project Status

**Current Phase**: Core functionality complete

**Implemented**:
- Hexagonal architecture with swappable adapters
- ReAct agent with tool calling (read/write memory)
- HTTP streaming API with API key authentication
- RabbitMQ pub/sub messaging
- 44 tests with 100% coverage

## Contributing

Follow established patterns:
- TDD workflow mandatory
- 100% test coverage required
- Hexagonal architecture principles
- Mock at boundaries, not internal logic

## License

MIT
