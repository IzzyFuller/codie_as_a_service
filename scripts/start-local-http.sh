#!/bin/bash
# Start the HTTP server with local storage and Claude CLI
#
# Prerequisites:
#   - Claude Code CLI installed (`claude` command available)
#   - .env file configured (or use defaults below)
#
# Usage:
#   ./scripts/start-local-http.sh [agent_id]
#
# Arguments:
#   agent_id    Optional agent to create if not exists (creates minimal memory)

set -e

cd "$(dirname "$0")/.."

# Load .env if it exists
if [ -f .env ]; then
    echo "Loading .env file..."
    set -a
    source .env
    set +a
fi

# Set defaults for local mode
export STORAGE_DIR=${STORAGE_DIR:-./data/agents}
export LLM_ADAPTER=${LLM_ADAPTER:-claude_cli}
export HTTP_HOST=${HTTP_HOST:-0.0.0.0}
export HTTP_PORT=${HTTP_PORT:-8080}
export PROMPTS_DIR=${PROMPTS_DIR:-./prompts}
export PROMPT_NAMES=${PROMPT_NAMES:-codie_as_a_service_system}
export API_KEY=${API_KEY:-local-dev-key}

# Create storage directory if needed
mkdir -p "$STORAGE_DIR"

# Create agent if specified
if [ -n "$1" ]; then
    AGENT_DIR="$STORAGE_DIR/$1"
    if [ ! -d "$AGENT_DIR" ]; then
        echo "Creating agent: $1"
        mkdir -p "$AGENT_DIR"
        echo "# Identity" > "$AGENT_DIR/me.md"
        echo "# Context Anchors" > "$AGENT_DIR/context_anchors.md"
        echo "# Current Session" > "$AGENT_DIR/current_session.md"
        echo "Agent $1 created with minimal memory."
    else
        echo "Agent $1 already exists."
    fi
fi

echo ""
echo "==========================================="
echo "  Starting HTTP Server (Local Mode)"
echo "==========================================="
echo ""
echo "  Storage:   local ($STORAGE_DIR)"
echo "  LLM:       $LLM_ADAPTER"
echo "  API:       http://localhost:$HTTP_PORT"
echo "  API Key:   $API_KEY"
echo ""
echo "  Test with:"
echo "    curl -X POST http://localhost:$HTTP_PORT/chat \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -H 'X-API-Key: $API_KEY' \\"
echo "      -d '{\"agent_id\": \"test\", \"session_id\": \"1\", \"message\": \"Hello!\"}'"
echo ""
echo "  Press Ctrl+C to stop"
echo "==========================================="
echo ""

# Run the server
uv run python -m codie_as_a_service.main_http
