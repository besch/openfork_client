#!/bin/bash
# OpenFork DGN Client Bootstrap Script
# Dynamically downloads all client files from GitHub repository using GitHub API
# Usage: curl -sL https://raw.githubusercontent.com/besch/openfork_client/main/bootstrap.sh | bash

set -e

REPO="besch/openfork_client"
BRANCH="${OPENFORK_CLIENT_SCRIPT_REF:-main}"
BASE_URL="https://raw.githubusercontent.com/$REPO/$BRANCH"
API_URL="https://api.github.com/repos/$REPO/git/trees/$BRANCH?recursive=1"

download_openfork_file() {
  local url="$1"
  local dest="$2"
  curl --fail --location --proto '=https' --tlsv1.2 --silent --show-error "$url" -o "$dest"
}

echo "=== OpenFork DGN Client Bootstrap ==="
echo "Fetching file list from GitHub API..."

# Fetch the repository tree from GitHub API
# This returns all files in the repo without needing a manifest
TREE_JSON=$(curl --fail --location --proto '=https' --tlsv1.2 --silent --show-error "$API_URL" || true)

if [ -z "$TREE_JSON" ] || [[ "$TREE_JSON" == *"rate limit"* ]]; then
  echo "Warning: GitHub API unavailable or rate limited. Falling back to manifest..."
  download_openfork_file "$BASE_URL/manifest.txt" manifest.txt
  count=0
  while IFS= read -r file || [[ -n "$file" ]]; do
    file=$(echo "$file" | tr -d '\r')
    [[ -z "$file" || "$file" =~ ^# ]] && continue
    dir=$(dirname "$file")
    [ "$dir" != "." ] && mkdir -p "$dir"
    (download_openfork_file "$BASE_URL/$file" "$file" || echo "Failed: $file") &
    
    count=$((count+1))
    if [ $count -ge 20 ]; then
      wait
      count=0
    fi
  done < manifest.txt
  wait
else
  echo "Downloading client files..."
  
  # Parse the JSON tree and download relevant files
  # Filter: .py, .sh, .txt (requirements), .json (workflows), excluding tests and docs
  # Note: Using POSIX-compatible regex patterns for cross-platform support (Linux/Mac/Windows Git Bash)
  echo "$TREE_JSON" | grep -E '"path":[[:space:]]*"[^"]+\.(py|sh|txt|json)"' | \
    sed 's/.*"path":[[:space:]]*"\([^"]*\)".*/\1/' | \
    grep -v "^test" | grep -v "^docs" | grep -v "__pycache__" | \
    grep -v "generate_manifest" | grep -v "\.spec$" | \
    {
      count=0
      while IFS= read -r file; do
        file=$(echo "$file" | tr -d '\r')
        # Ensure directory exists
        dir=$(dirname "$file")
        if [ "$dir" != "." ]; then
          mkdir -p "$dir"
        fi
        
        echo "  -> Downloading $file"
        (download_openfork_file "$BASE_URL/$file" "$file" 2>/dev/null || echo "    (skipped)") &
        
        count=$((count+1))
        if [ $count -ge 20 ]; then
          wait
          count=0
        fi
      done
      wait
    }
fi

# Install Python dependencies if requested
if [ "$INSTALL_DEPS" = "true" ]; then
  echo "Installing Python dependencies..."
  
  # Detect Python executable
  if [ -x "/usr/bin/python" ]; then
    PYTHON_EXE="/usr/bin/python"
  elif command -v python &> /dev/null; then
    PYTHON_EXE=$(command -v python)
  elif command -v python3 &> /dev/null; then
    PYTHON_EXE=$(command -v python3)
  else
    PYTHON_EXE=""
  fi
  
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
