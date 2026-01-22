#!/usr/bin/env python3
"""Sync session notes from GCS emulator back to local memory.

This script fetches session notes from the GCS emulator (Docker) and
identifies new entries that can be merged into local memory.

Usage:
    python scripts/sync-memory-from-gcs.py [user_id]

    # Show new notes from GCS that aren't in local memory
    python scripts/sync-memory-from-gcs.py demo-user

    # Output as JSON for programmatic use
    python scripts/sync-memory-from-gcs.py demo-user --json

Environment variables:
    CODIE_MEMORY_PATH: Path to local memory directory (required)
    STORAGE_EMULATOR_HOST: GCS emulator URL (default: http://localhost:4443)
    GCS_BUCKET_NAME: GCS bucket name (default: deep-agent-memory)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import httpx

# Configuration
GCS_EMULATOR_URL = os.environ.get("STORAGE_EMULATOR_HOST", "http://localhost:4443")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "deep-agent-memory")
CODIE_MEMORY_PATH = os.environ.get("CODIE_MEMORY_PATH")
if not CODIE_MEMORY_PATH:
    print("ERROR: CODIE_MEMORY_PATH environment variable is required", file=sys.stderr)
    sys.exit(1)
CODIE_MEMORY_DIR = Path(CODIE_MEMORY_PATH)


def fetch_gcs_session(user_id: str) -> str | None:
    """Fetch current_session.md from GCS emulator for user."""
    url = f"{GCS_EMULATOR_URL}/storage/v1/b/{GCS_BUCKET_NAME}/o/users%2F{user_id}%2Fcurrent_session.md?alt=media"

    try:
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
        return response.text
    except httpx.HTTPError as e:
        print(f"Error fetching from GCS: {e}", file=sys.stderr)
        return None


def read_local_session() -> str | None:
    """Read Codie's local current_session.md."""
    session_path = CODIE_MEMORY_DIR / "current_session.md"

    if not session_path.exists():
        print(f"Local session file not found: {session_path}", file=sys.stderr)
        return None

    return session_path.read_text()


def parse_session_notes(content: str) -> list[dict]:
    """Parse session notes from markdown content.

    Returns list of dicts with keys: type, importance, timestamp, content
    """
    notes = []

    # Pattern: ### TYPE - IMPORTANCE (TIMESTAMP)
    # Followed by content until next ### or end
    pattern = r"### (\w+) - (\w+) \(([^)]+)\)\n(.*?)(?=\n### |\Z)"

    for match in re.finditer(pattern, content, re.DOTALL):
        note_type, importance, timestamp, note_content = match.groups()
        notes.append(
            {
                "type": note_type.lower(),
                "importance": importance.lower(),
                "timestamp": timestamp,
                "content": note_content.strip(),
            }
        )

    return notes


def find_new_notes(gcs_notes: list[dict], local_notes: list[dict]) -> list[dict]:
    """Find notes in GCS that aren't in local memory.

    Uses timestamp as unique identifier.
    """
    local_timestamps = {note["timestamp"] for note in local_notes}
    return [note for note in gcs_notes if note["timestamp"] not in local_timestamps]


def main():
    parser = argparse.ArgumentParser(description="Sync session notes from GCS to local")
    parser.add_argument(
        "user_id",
        nargs="?",
        default="demo-user",
        help="GCS user ID (default: demo-user)",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    # Fetch both versions
    print(f"Fetching session from GCS for user: {args.user_id}", file=sys.stderr)
    gcs_content = fetch_gcs_session(args.user_id)
    if not gcs_content:
        sys.exit(1)

    print("Reading local session...", file=sys.stderr)
    local_content = read_local_session()
    if not local_content:
        sys.exit(1)

    # Parse notes
    gcs_notes = parse_session_notes(gcs_content)
    local_notes = parse_session_notes(local_content)

    print(
        f"GCS notes: {len(gcs_notes)}, Local notes: {len(local_notes)}", file=sys.stderr
    )

    # Find new notes
    new_notes = find_new_notes(gcs_notes, local_notes)

    if not new_notes:
        print("\nNo new notes to sync - GCS and local are in sync!", file=sys.stderr)
        return

    print(f"\nFound {len(new_notes)} new note(s) to sync:\n", file=sys.stderr)

    if args.json:
        print(json.dumps(new_notes, indent=2))
    else:
        for note in new_notes:
            print(
                f"### {note['type'].upper()} - {note['importance'].upper()} ({note['timestamp']})"
            )
            print(note["content"])
            print()


if __name__ == "__main__":
    main()
