#!/bin/bash
# Start codie_as_a_service locally with GCS emulator
#
# Prerequisites:
#   - Docker installed and running
#   - .env file with ANTHROPIC_API_KEY and Langfuse credentials
#
# Usage:
#   ./scripts/start-local.sh

set -e

# Configuration
GCS_EMULATOR_PORT=4443
GCS_BUCKET_NAME="deep-agent-memory"
HTTP_PORT=8080

echo "=== Deep Agent Service Local Startup ==="

# Load .env if it exists
if [ -f .env ]; then
    echo "Loading .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Check for required credentials
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set. Add it to .env or export it."
    exit 1
fi

# Stop any existing emulator
echo "Cleaning up existing GCS emulator..."
docker rm -f gcs-emulator-local 2>/dev/null || true

# Start GCS emulator
echo "Starting GCS emulator on port $GCS_EMULATOR_PORT..."
docker run -d \
    --name gcs-emulator-local \
    -p ${GCS_EMULATOR_PORT}:4443 \
    fsouza/fake-gcs-server \
    -scheme http \
    -port 4443

# Wait for emulator to be ready
echo "Waiting for GCS emulator to be ready..."
for i in {1..10}; do
    if curl -s "http://localhost:${GCS_EMULATOR_PORT}/storage/v1/b" > /dev/null 2>&1; then
        echo "GCS emulator is ready."
        break
    fi
    if [ $i -eq 10 ]; then
        echo "ERROR: GCS emulator failed to start"
        exit 1
    fi
    sleep 1
done

# Create bucket
echo "Creating bucket: $GCS_BUCKET_NAME..."
curl -s -X POST "http://localhost:${GCS_EMULATOR_PORT}/storage/v1/b?project=local-project" \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"${GCS_BUCKET_NAME}\"}" > /dev/null

echo "Bucket created."

# Set environment for emulator
export STORAGE_EMULATOR_HOST="http://localhost:${GCS_EMULATOR_PORT}"
export GCS_BUCKET_NAME="$GCS_BUCKET_NAME"
export HTTP_PORT="$HTTP_PORT"

echo ""
echo "=== Starting HTTP Server ==="
echo "GCS Emulator: $STORAGE_EMULATOR_HOST"
echo "Bucket: $GCS_BUCKET_NAME"
echo "HTTP Port: $HTTP_PORT"
echo ""

# Run the server
python -m codie_as_a_service.main_http
