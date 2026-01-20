#!/bin/bash
# Stop demo infrastructure (emulators)
#
# Usage:
#   ./demo/stop-infrastructure.sh
#
# Note: This does NOT delete the Docker volume, so data persists.
# To reset all data: docker volume rm gcs-demo-data

set -e

echo "=== Stopping Demo Infrastructure ==="

# Stop GCS emulator
if docker ps --format '{{.Names}}' | grep -q "^gcs-emulator-demo$"; then
    echo "Stopping GCS emulator..."
    docker rm -f gcs-emulator-demo
else
    echo "GCS emulator not running"
fi

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
echo ""
echo "Data is preserved in Docker volume 'gcs-demo-data'."
echo "To delete all data: docker volume rm gcs-demo-data"
