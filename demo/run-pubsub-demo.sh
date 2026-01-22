#!/bin/bash
# Run the Pub/Sub CLI demo
#
# Prerequisites:
#   Run ./demo/setup-infrastructure.sh first
#
# Usage:
#   ./demo/run-pubsub-demo.sh

set -e

cd "$(dirname "$0")/.."

# Check if infrastructure is set up
if [ ! -f demo/demo.env ]; then
    echo "ERROR: Infrastructure not set up. Run ./demo/setup-infrastructure.sh first"
    exit 1
fi

# Check if emulators are running
if ! docker ps --format '{{.Names}}' | grep -q "^gcs-emulator-demo$"; then
    echo "ERROR: GCS emulator not running. Run ./demo/setup-infrastructure.sh first"
    exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -q "^pubsub-emulator-demo$"; then
    echo "ERROR: Pub/Sub emulator not running. Run ./demo/setup-infrastructure.sh first"
    exit 1
fi

# Load environment
source demo/demo.env

# Load .env for API keys
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down..."
    if [ -n "$AGENT_PID" ]; then
        kill $AGENT_PID 2>/dev/null || true
    fi
    echo "Demo stopped. Infrastructure still running."
    echo "Stop infrastructure with: ./demo/stop-infrastructure.sh"
}

trap cleanup EXIT INT TERM

echo ""
echo "==========================================="
echo "  Starting Pub/Sub CLI Demo"
echo "==========================================="
echo ""
echo "  GCS Emulator:   $STORAGE_EMULATOR_HOST"
echo "  Pub/Sub:        $PUBSUB_EMULATOR_HOST"
echo "  User:           demo-user"
echo ""
echo "  Press Ctrl+C to stop demo (infrastructure stays running)"
echo "==========================================="
echo ""

# Ensure demo dependencies are installed
uv sync --group demo --quiet

# Start the agent service in background
# UV_NO_SYNC=1 prevents uv bug where torch is reinstalled on every run (macOS 26 platform tag mismatch)
echo "Starting Pub/Sub agent service..."
UV_NO_SYNC=1 uv run python -m deep_agent_service.main_pubsub &
AGENT_PID=$!

# Give agent time to start
sleep 3

echo "Agent service running (PID: $AGENT_PID)"
echo ""

# Run CLI client in foreground
UV_NO_SYNC=1 uv run python demo/pubsub_cli.py
