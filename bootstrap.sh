#!/bin/bash
# OpenFork DGN Client Bootstrap Script
# Dynamically downloads all client files from GitHub repository using GitHub API
# Usage: curl -sL https://raw.githubusercontent.com/besch/openfork_client/main/bootstrap.sh | bash

set -e

REPO="besch/openfork_client"
BRANCH="main"
BASE_URL="https://raw.githubusercontent.com/$REPO/$BRANCH"
API_URL="https://api.github.com/repos/$REPO/git/trees/$BRANCH?recursive=1"

# Detect Python executable early for JSON parsing
if [ -x "/usr/bin/python" ]; then
  PYTHON_EXE="/usr/bin/python"
elif command -v python &> /dev/null; then
  PYTHON_EXE=$(command -v python)
elif command -v python3 &> /dev/null; then
  PYTHON_EXE=$(command -v python3)
else
  # Fallback to python3 and hope it works or user has it
  PYTHON_EXE="python3"
fi

echo "=== OpenFork DGN Client Bootstrap ==="
echo "Fetching file list from GitHub API..."

# Fetch the repository tree from GitHub API
# This returns all files in the repo without needing a manifest
TREE_JSON=$(curl -sL "$API_URL")

if [ -z "$TREE_JSON" ] || [[ "$TREE_JSON" == *"rate limit"* ]]; then
  echo "Warning: GitHub API unavailable or rate limited. Falling back to manifest..."
  curl -sL "$BASE_URL/manifest.txt" -o manifest.txt
  while IFS= read -r file || [[ -n "$file" ]]; do
    [[ -z "$file" || "$file" =~ ^# ]] && continue
    dir=$(dirname "$file")
    [ "$dir" != "." ] && mkdir -p "$dir"
    curl -sL "$BASE_URL/$file" -o "$file"
  done < manifest.txt
else
  echo "Downloading client files..."
  
  # Parse the JSON tree and download relevant files
  # Filter: .py, .sh, .txt (requirements), .json (workflows), excluding tests and docs
  # Parse the JSON tree and download relevant files
  # Use Python for robust JSON parsing (handles minified or pretty-printed JSON)
  # Filter: .py, .sh, .txt (requirements), .json (workflows), excluding tests and docs
  echo "$TREE_JSON" | $PYTHON_EXE -c "
import sys, json, re
try:
    data = json.load(sys.stdin)
    files = [item['path'] for item in data.get('tree', [])]
    # Filter files
    pattern = re.compile(r'.*\.(py|sh|txt|json)$')
    for f in files:
        if pattern.match(f) and not f.startswith('test') and not f.startswith('docs') and '__pycache__' not in f and 'generate_manifest' not in f and not f.endswith('.spec'):
             print(f)
except Exception as e:
    pass
" | \
    while IFS= read -r file; do
      # Ensure directory exists
      dir=$(dirname "$file")
      if [ "$dir" != "." ]; then
        mkdir -p "$dir"
      fi
      
      echo "  -> Downloading $file"
      curl -sL "$BASE_URL/$file" -o "$file" 2>/dev/null || echo "    (skipped)"
    done
fi

# Install Python dependencies if requested
if [ "$INSTALL_DEPS" = "true" ]; then
  echo "Installing Python dependencies..."
  
  if [ -n "$PYTHON_EXE" ]; then
    echo "Using $PYTHON_EXE to install dependencies..."
    # Try with --break-system-packages first (for PEP 668 / Ubuntu 24+)
    $PYTHON_EXE -m pip install --quiet --break-system-packages -r requirements.txt 2>/dev/null || \
    $PYTHON_EXE -m pip install --quiet -r requirements.txt 2>/dev/null || \
    $PYTHON_EXE -m pip install --quiet --user -r requirements.txt 2>/dev/null || \
    echo "Warning: Failed to install dependencies automatically."
  else
    echo "Warning: Python not found, skipping dependency installation."
  fi
fi

echo "✓ DGN client files downloaded successfully"
