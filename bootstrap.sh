#!/bin/bash
# OpenFork DGN Client Bootstrap Script
# Downloads all client files from GitHub repository
# Usage: curl -sL https://raw.githubusercontent.com/besch/openfork_client/main/bootstrap.sh | bash

set -e

BASE_URL="https://raw.githubusercontent.com/besch/openfork_client/main"

echo "=== OpenFork DGN Client Bootstrap ==="
echo "Downloading manifest..."

# Download manifest first
curl -sL $BASE_URL/manifest.txt -o manifest.txt

echo "Downloading client files..."
while IFS= read -r file || [[ -n "$file" ]]; do
  # Skip empty lines or comments
  [[ -z "$file" || "$file" =~ ^# ]] && continue
  
  # Ensure directory exists
  dir=$(dirname "$file")
  if [ "$dir" != "." ]; then
    mkdir -p "$dir"
  fi
  
  echo "  -> Downloading $file"
  curl -sL "$BASE_URL/$file" -o "$file"
done < manifest.txt


# Install Python dependencies if we're in a fresh environment
if [ "$INSTALL_DEPS" = "true" ]; then
  echo "Installing Python dependencies..."
  pip install --quiet -r requirements.txt 2>/dev/null || true
fi

echo "✓ DGN client files downloaded successfully"
