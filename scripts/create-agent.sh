#!/bin/bash
# Create a demo agent with initial memory files in GCS emulator
#
# Prerequisites:
#   - GCS emulator running (start with ./scripts/start-local.sh first)
#
# Usage:
#   ./scripts/create-agent.sh [agent_id] [--reset]
#
# Options:
#   --reset    Force recreate agent files even if they exist
#
# Default agent_id: demo-user

set -e

# Parse arguments
RESET_FLAG=false
AGENT_ID="demo-user"

for arg in "$@"; do
    case $arg in
        --reset)
            RESET_FLAG=true
            ;;
        *)
            AGENT_ID="$arg"
            ;;
    esac
done

# Configuration
GCS_EMULATOR_PORT=${GCS_EMULATOR_PORT:-4443}
GCS_BUCKET_NAME=${GCS_BUCKET_NAME:-"deep-agent-memory"}
GCS_URL="http://localhost:${GCS_EMULATOR_PORT}"

echo "=== Creating Demo Agent: $AGENT_ID ==="

# Check if emulator is running
if ! curl -s "${GCS_URL}/storage/v1/b" > /dev/null 2>&1; then
    echo "ERROR: GCS emulator not running. Start it with: ./scripts/start-local.sh"
    exit 1
fi

# Function to check if a file exists in GCS
file_exists() {
    local blob_path=$1
    local response
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        "${GCS_URL}/storage/v1/b/${GCS_BUCKET_NAME}/o/${blob_path}")
    [ "$response" = "200" ]
}

# Check if user already exists (by checking me.md)
USER_EXISTS=false
ME_BLOB_PATH="agents%2F${AGENT_ID}%2Fme.md"  # URL-encoded path

# Debug: Show what we're checking
echo "Checking for existing agent at: ${GCS_URL}/storage/v1/b/${GCS_BUCKET_NAME}/o/${ME_BLOB_PATH}"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
    "${GCS_URL}/storage/v1/b/${GCS_BUCKET_NAME}/o/${ME_BLOB_PATH}")
echo "Response code: $RESPONSE"

if file_exists "$ME_BLOB_PATH"; then
    USER_EXISTS=true
fi

if [ "$USER_EXISTS" = true ] && [ "$RESET_FLAG" = false ]; then
    echo "Agent '$AGENT_ID' already exists. Memory preserved."
    echo "Use --reset flag to force recreate: ./scripts/create-agent.sh $AGENT_ID --reset"
    exit 0
fi

if [ "$RESET_FLAG" = true ]; then
    echo "Reset flag set - recreating agent files..."
fi

# Function to upload a file to GCS emulator
upload_file() {
    local key=$1
    local content=$2
    local blob_path="agents/${AGENT_ID}/${key}.md"

    echo "  Uploading: ${blob_path}"

    # Use the fake-gcs-server upload API
    curl -s -X POST \
        "${GCS_URL}/upload/storage/v1/b/${GCS_BUCKET_NAME}/o?uploadType=media&name=${blob_path}" \
        -H "Content-Type: text/markdown" \
        -d "${content}" > /dev/null
}

# Create me.md - Core identity document
ME_CONTENT="# Demo Agent Identity

## Who I Am
I am a helpful AI assistant for demo purposes.

## Core Principles
- Be helpful and accurate
- Explain my reasoning
- Ask clarifying questions when needed

## Preferences
- Clear, concise communication
- Step-by-step explanations for complex topics
"

# Create context_anchors.md - Working memory pointers
CONTEXT_ANCHORS_CONTENT="# Context Anchors - Working Memory

## Current Focus
- Getting started with the deep agent service
- Exploring capabilities

## Notes
This is a fresh user session.
"

# Create current_session.md - Session notes
CURRENT_SESSION_CONTENT="# Current Session

## Session Start
New session initialized.

## Notes
Ready to begin interaction.
"

echo "Creating memory files..."
upload_file "me" "$ME_CONTENT"
upload_file "context_anchors" "$CONTEXT_ANCHORS_CONTENT"
upload_file "current_session" "$CURRENT_SESSION_CONTENT"

echo ""
echo "=== Demo Agent Created Successfully ==="
echo "User ID: $AGENT_ID"
echo "Memory files:"
echo "  - agents/${AGENT_ID}/me.md"
echo "  - agents/${AGENT_ID}/context_anchors.md"
echo "  - agents/${AGENT_ID}/current_session.md"
echo ""
echo "Test with:"
echo "  curl -X POST http://localhost:8080/chat \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"agent_id\": \"${AGENT_ID}\", \"session_id\": \"test-session\", \"message\": \"Hello!\"}'"
