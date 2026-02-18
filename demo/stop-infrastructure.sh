#!/bin/bash
# Stop demo infrastructure (emulators)
#
# Usage:
#   ./demo/stop-infrastructure.sh

set -e

echo "=== Stopping Demo Infrastructure ==="

# Stop Pub/Sub emulator
if docker ps --format '{{.Names}}' | grep -q "^pubsub-emulator-demo$"; then
    echo "Stopping Pub/Sub emulator..."
    docker rm -f pubsub-emulator-demo
else
    echo "Pub/Sub emulator not running"
fi

# Clean up env file
if [ -f demo/demo.env ]; then
    rm demo/demo.env
    echo "Removed demo/demo.env"
fi

echo ""
echo "Infrastructure stopped."
