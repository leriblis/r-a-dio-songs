#!/usr/bin/env bash

set -euo pipefail

BASEDIR=$(cd "$(dirname "$0")" && pwd)
cd "$BASEDIR"

ARCHIVE="songs_db.json.xz"
PLAIN="songs_db.json"

if [ -f "$PLAIN" ]; then
  echo "Re-archiving $PLAIN..."
  xz -z -f "$PLAIN"
  echo "Created $ARCHIVE"
elif [ -f "$ARCHIVE" ]; then
  echo "$ARCHIVE already exists. Nothing to do."
else
  echo "Error: neither $PLAIN nor $ARCHIVE exists."
  exit 1
fi

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git restore --staged "$PLAIN" 2>/dev/null || true
fi

echo "Cleanup complete."
