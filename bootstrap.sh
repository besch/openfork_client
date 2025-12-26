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
  
  # Detect Python executable
  PYTHON_EXE=$(command -v python3 || command -v python || echo "")
  
  if [ -n "$PYTHON_EXE" ]; then
    echo "Using $PYTHON_EXE to install dependencies..."
    # Try with --break-system-packages first (for PEP 668 / Ubuntu 24+)
    $PYTHON_EXE -m pip install --quiet --break-system-packages -r requirements.txt 2>/dev/null || \
    $PYTHON_EXE -m pip install --quiet -r requirements.txt 2>/dev/null || \
    echo "Warning: pip install failed with $PYTHON_EXE. Retrying with --user..."
    $PYTHON_EXE -m pip install --quiet --user -r requirements.txt 2>/dev/null || \
    echo "Failed to install dependencies automatically."
  else
    echo "Warning: Python not found, skipping dependency installation."
  fi
fi

echo "✓ DGN client files downloaded successfully"
