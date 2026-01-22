#!/bin/bash
# Copy Codie's memory entities into GCS emulator for a specified user
#
# Usage:
#   ./scripts/copy-codie-memory.sh [user_id]
#
# Default user_id: izzy

set -e

USER_ID="${1:-izzy}"
CODIE_MEMORY_DIR="${CODIE_MEMORY_PATH:-/Users/izzyfuller/Codie/memory}"

# Configuration
GCS_EMULATOR_PORT=${GCS_EMULATOR_PORT:-4443}
GCS_BUCKET_NAME=${GCS_BUCKET_NAME:-"deep-agent-memory"}
GCS_URL="http://localhost:${GCS_EMULATOR_PORT}"

echo "=== Copying Codie's Memory to User: $USER_ID ==="
echo ""

# Check if emulator is running
if ! curl -s "${GCS_URL}/storage/v1/b" > /dev/null 2>&1; then
    echo "ERROR: GCS emulator not running. Start it with: ./demo/setup-infrastructure.sh"
    exit 1
fi

# Check if Codie memory directory exists
if [ ! -d "$CODIE_MEMORY_DIR" ]; then
    echo "ERROR: Codie memory directory not found at $CODIE_MEMORY_DIR"
    exit 1
fi

# Function to upload a file to GCS emulator
upload_file() {
    local key=$1
    local source_file=$2
    local blob_path="users/${USER_ID}/${key}.md"

    if [ ! -f "$source_file" ]; then
        echo "  SKIP: $source_file (not found)"
        return
    fi

    echo "  Uploading: ${blob_path}"

    # Use the fake-gcs-server upload API with file content
    curl -s -X POST \
        "${GCS_URL}/upload/storage/v1/b/${GCS_BUCKET_NAME}/o?uploadType=media&name=${blob_path}" \
        -H "Content-Type: text/markdown" \
        --data-binary "@${source_file}" > /dev/null
}

echo "Uploading core memory files..."
upload_file "me" "${CODIE_MEMORY_DIR}/me.md"
upload_file "context_anchors" "${CODIE_MEMORY_DIR}/context_anchors.md"
upload_file "current_session" "${CODIE_MEMORY_DIR}/current_session.md"

echo ""
echo "=== Codie's Memory Copied Successfully ==="
echo "User ID: $USER_ID"
echo "Memory files:"
echo "  - users/${USER_ID}/me.md"
echo "  - users/${USER_ID}/context_anchors.md"
echo "  - users/${USER_ID}/current_session.md"
echo ""
echo "Test with Gradio demo:"
echo "  ./demo/run-http-demo.sh"
echo "  Then use user_id: ${USER_ID}"
