# Deep Agent Service

![CI](https://github.com/IzzyFuller/deep_agent_service/workflows/CI/badge.svg)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/IzzyFuller/100b39151bc1c8397fb6a684ade5e588/raw/coverage.json)](https://gist.github.com/IzzyFuller/100b39151bc1c8397fb6a684ade5e588)
![Python](https://img.shields.io/badge/python-3.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Microservice for per-user agent identity and memory - a prototype for "Codie-as-a-Service".

## Overview

This service extracts deep agent functionality from python-monorepo and preprocessing-service into a standalone microservice that enables per-user agent identity and memory isolation. Each user gets their own memory space (similar to how Codie maintains memory for Izzy), positioning us to provide personalized AI agents at scale.

## Architecture

### Hexagonal Architecture (Ports & Adapters)

```
deep_agent_service/
├── src/deep_agent_service/
│   ├── core/                  # Domain layer (models, protocols)
│   ├── services/              # Business logic layer
│   │   ├── agent/            # ReAct agent implementation
│   │   ├── memory/           # User memory management
│   │   └── tools/            # Agent tool system
│   ├── adapters/             # Infrastructure layer
│   │   ├── llm/              # Anthropic SDK adapter
│   │   ├── prompts/          # Langfuse integration
│   │   ├── storage/          # GCS & Firestore adapters
│   │   └── messaging/        # Pub/Sub message handling
│   └── api/                  # Entry points
└── tests/                    # Mirror structure for tests
```

### Design Principles

- **Proportional Response**: Simple POC approach (~500-800 LOC) vs complex production system (3,405 LOC)
- **Archaeological Engineering**: Reuses proven patterns from preprocessing-service
- **TDD**: 100% test coverage target with mock-at-boundaries strategy
- **Message-Driven**: Pub/Sub for async execution (no threading complexity)

## Technology Stack

### Core Dependencies
- **Python**: 3.13.x
- **Anthropic SDK**: Direct LLM integration (no LangChain/LangGraph overhead)
- **Langfuse**: Version-controlled prompts
- **Google Cloud**: Storage (GCS) for user memory files, Firestore for session metadata
- **Pydantic**: Type-safe models throughout
- **synapse**: Stack-agnostic pub/sub library

### Dev Dependencies
- **pytest**: Testing framework with coverage and async support
- **ruff**: Fast linting and formatting
- **mypy**: Static type checking
- **httpx**: HTTP client for test mocking

## POC Feature Scope

### ✅ In Scope (Minimal Viable Feature Set)

1. **User Identity & Isolation**: Each user gets unique `user_id` with isolated memory
2. **Basic Memory System**:
   - `current_session.md` - Working memory for active conversation
   - `context_anchors.md` - Priority-ranked current context
   - Simple key-value memory operations (read/write)
3. **Simple ReAct Agent**:
   - Reason → Act → Observe loop (max 10 iterations)
   - Direct Anthropic SDK integration
   - Structured outputs via Pydantic
4. **Session Management**: Create, resume, and end user sessions
5. **Tool System**: 3 tools (read_memory, write_memory, + 1 domain tool)
6. **Message Handler**: Process RunAgentRequest from Pub/Sub

### ⛔ Out of Scope (Future Enhancements)

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
- **Docker** - For running GCS and Pub/Sub emulators
- API keys: Anthropic, Langfuse

### Installation

```bash
# Clone repository
git clone https://github.com/IzzyFuller/deep_agent_service.git
cd deep-agent-service

# Install all dependencies (main + dev + demo)
uv sync --all-groups

# Set required environment variables
export ANTHROPIC_API_KEY="your-key"
export LANGFUSE_PUBLIC_KEY="your-public-key"
export LANGFUSE_SECRET_KEY="your-secret-key"
```

### Configuration

Set environment variables:

```bash
export ANTHROPIC_API_KEY="your-key"
export LANGFUSE_PUBLIC_KEY="your-public-key"
export LANGFUSE_SECRET_KEY="your-secret-key"
export GCP_PROJECT_ID="your-project"
export MEMORY_BUCKET="gs://your-memory-bucket"
```

## Running the Demo

### Quick Start

```bash
# 1. Start emulators and create demo user
./demo/setup-infrastructure.sh

# 2. Run the HTTP streaming demo with Gradio UI
./demo/run-http-demo.sh
# Opens at http://localhost:7860

# 3. Or run the Pub/Sub CLI demo
./demo/run-pubsub-demo.sh
```

### HTTP Streaming API

The service exposes a streaming chat endpoint using Server-Sent Events (SSE):

**Endpoint:** `POST /chat`

**Request:**
```json
{
  "user_id": "demo-user",
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
  -d '{"user_id": "demo-user", "session_id": "test", "message": "Hello!"}' \
  --no-buffer
```

### Gradio UI

The demo includes a web interface for interactive testing:
- Real-time streaming responses
- User ID switching (test different memory contexts)
- Conversation history display

### Memory Sync Scripts

For advanced users testing with custom memory:

```bash
# Copy Codie's memory entities to a GCS user
./scripts/copy-codie-memory.sh izzy

# Sync session notes back from GCS to local
python scripts/sync-memory-from-gcs.py izzy
```

## Testing

### Running Tests

```bash
# Run all tests with coverage
uv run pytest --cov=src/deep_agent_service --cov-report=term-missing

# Run specific test file
uv run pytest tests/services/agent/test_react_agent.py

# Run with verbose output
uv run pytest -v
```

### Test Strategy

- **Mock at boundaries**: GCS, Firestore, Anthropic API, Langfuse
- **Don't mock our logic**: ReAct loop, memory service, tool registry
- **Emulators**: Docker-based Pub/Sub and GCS emulators for integration tests
- **Fixtures**: Reusable test data in `tests/conftest.py`

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

Each user gets isolated memory in GCS:

```
gs://deep-agent-memory-{env}/
└── users/
    └── {user_id}/
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

**Current Phase**: Initial setup and architecture definition

**Next Steps**:
1. Define core Pydantic models (AgentState, Message, MemoryEntry)
2. Implement protocol interfaces
3. Build adapters with mocked tests
4. Implement services layer
5. Create integration tests
6. Validate with real Anthropic API

## Contributing

Follow established patterns:
- TDD workflow mandatory
- 100% test coverage required
- Hexagonal architecture principles
- Mock at boundaries, not internal logic

## License

MIT
