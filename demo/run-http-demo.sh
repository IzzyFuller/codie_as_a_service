#!/bin/bash
# Run the HTTP demo with Gradio UI
#
# Prerequisites:
#   Run ./demo/setup-infrastructure.sh first
#
# Usage:
#   ./demo/run-http-demo.sh

set -e

cd "$(dirname "$0")/.."

# Check if infrastructure is set up
if [ ! -f demo/demo.env ]; then
    echo "ERROR: Infrastructure not set up. Run ./demo/setup-infrastructure.sh first"
    exit 1
fi

# Check if GCS emulator is running
if ! docker ps --format '{{.Names}}' | grep -q "^gcs-emulator-demo$"; then
    echo "ERROR: GCS emulator not running. Run ./demo/setup-infrastructure.sh first"
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

# Configuration
HTTP_PORT=${HTTP_PORT:-8080}
DEMO_PORT=${DEMO_PORT:-7860}

# Cleanup function
cleanup() {
    echo ""
    echo "=== Shutting down ==="
    kill $HTTP_PID 2>/dev/null || true
    kill $DEMO_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT

echo ""
echo "==========================================="
echo "  Starting HTTP Demo with Gradio UI"
echo "==========================================="
echo ""
echo "  GCS Emulator:   $STORAGE_EMULATOR_HOST"
echo "  HTTP API:       http://localhost:$HTTP_PORT"
echo "  Demo UI:        http://localhost:$DEMO_PORT"
echo "  User:           demo-user"
echo ""
echo "  Press Ctrl+C to stop"
echo "==========================================="
echo ""

# Start the HTTP service in background
echo "Starting HTTP service..."
poetry run python -m deep_agent_service.main_http &
HTTP_PID=$!

# Wait for HTTP service to be ready
echo "Waiting for HTTP service..."
for i in {1..15}; do
    if curl -s "http://localhost:${HTTP_PORT}/health" > /dev/null 2>&1; then
        echo "HTTP service is ready."
        break
    fi
    if [ $i -eq 15 ]; then
        echo "WARNING: HTTP service health check timed out (may still be starting)"
    fi
    sleep 1
done

# Start the Gradio demo UI
echo "Starting Gradio UI..."
poetry run python demo/app.py &
DEMO_PID=$!

echo ""
echo "  Demo UI ready at: http://localhost:$DEMO_PORT"
echo ""

# Wait for any process to exit
wait
