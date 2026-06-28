#!/bin/bash
# OpenFork DGN Client Bootstrap Script
# Dynamically downloads all client files from GitHub repository.
# Usage: curl -sL https://raw.githubusercontent.com/besch/openfork_client/main/bootstrap.sh | bash

set -e

REPO="besch/openfork_client"
REQUESTED_REF="${OPENFORK_CLIENT_SCRIPT_REF:-main}"
RAW_BASE="https://raw.githubusercontent.com/$REPO"
COMMIT_API_URL="https://api.github.com/repos/$REPO/commits/$REQUESTED_REF"

DOWNLOAD_FAILURES=$(mktemp)
PATH_LIST=$(mktemp)
TREE_JSON=$(mktemp)
MANIFEST_FILE=$(mktemp)
RESOLVED_REF_FILE=$(mktemp)
STAGING_DIR=$(mktemp -d)
MANAGED_TOPS=$(mktemp)

cleanup() {
  rm -f "$DOWNLOAD_FAILURES" "$PATH_LIST" "$TREE_JSON" "$MANIFEST_FILE" "$RESOLVED_REF_FILE" "$MANAGED_TOPS"
  rm -rf "$STAGING_DIR"
}
trap cleanup EXIT

find_json_python() {
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
  elif command -v python >/dev/null 2>&1; then
    command -v python
  else
    echo ""
  fi
}

download_openfork_file() {
  local url="$1"
  local dest="$2"
  local tmp="${dest}.tmp.$$"
  local attempt
  local max_attempts="${OPENFORK_BOOTSTRAP_DOWNLOAD_ATTEMPTS:-5}"
  local delay_seconds

  mkdir -p "$(dirname "$dest")"

  for attempt in $(seq 1 "$max_attempts"); do
    if curl --fail --location --proto '=https' --tlsv1.2 --silent --show-error "$url" -o "$tmp"; then
      mv "$tmp" "$dest"
      return 0
    fi

    rm -f "$tmp"
    if [ "$attempt" -lt "$max_attempts" ]; then
      delay_seconds=$((attempt * 2))
      echo "Warning: download failed for $url (attempt $attempt/$max_attempts); retrying in ${delay_seconds}s..." >&2
      sleep "$delay_seconds"
    fi
  done

  rm -f "$tmp"
  echo "$dest" >> "$DOWNLOAD_FAILURES"
  return 1
}

wait_for_download_batch() {
  wait || true
  if [ -s "$DOWNLOAD_FAILURES" ]; then
    echo "ERROR: Failed to download required DGN client file(s):"
    sed 's/^/  - /' "$DOWNLOAD_FAILURES"
    exit 1
  fi
}

copy_openfork_archive() {
  local archive_file
  local extract_dir
  local source_root
  local file
  local failures=0
  archive_file=$(mktemp)
  extract_dir=$(mktemp -d)

  echo "Downloading client archive from GitHub..."
  if ! curl --fail --location --proto '=https' --tlsv1.2 --silent --show-error \
    "https://codeload.github.com/$REPO/tar.gz/$RESOLVED_REF" -o "$archive_file"; then
    rm -f "$archive_file"
    rm -rf "$extract_dir"
    return 1
  fi

  if ! tar -xzf "$archive_file" -C "$extract_dir"; then
    rm -f "$archive_file"
    rm -rf "$extract_dir"
    return 1
  fi

  source_root=$(find "$extract_dir" -mindepth 1 -maxdepth 1 -type d | head -n 1)
  if [ ! -d "$source_root" ]; then
    rm -f "$archive_file"
    rm -rf "$extract_dir"
    return 1
  fi

  while IFS= read -r file || [ -n "$file" ]; do
    file=$(echo "$file" | tr -d '\r')
    [ -z "$file" ] && continue

    if [ -f "$source_root/$file" ]; then
      mkdir -p "$(dirname "$STAGING_DIR/$file")"
      cp -a "$source_root/$file" "$STAGING_DIR/$file"
    else
      echo "Warning: archive missing required client file: $file" >&2
      failures=$((failures + 1))
    fi
  done < "$PATH_LIST"

  rm -f "$archive_file"
  rm -rf "$extract_dir"

  [ "$failures" -eq 0 ]
}

resolve_ref() {
  local json_python="$1"
  local commit_json
  commit_json=$(mktemp)

  if curl --fail --location --proto '=https' --tlsv1.2 --silent --show-error "$COMMIT_API_URL" -o "$commit_json"; then
    "$json_python" - "$commit_json" "$RESOLVED_REF_FILE" <<'PY' || true
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    data = json.load(handle)

sha = data.get("sha")
if sha:
    with open(sys.argv[2], "w", encoding="utf-8") as handle:
        handle.write(sha)
PY
  fi

  rm -f "$commit_json"

  if [ -s "$RESOLVED_REF_FILE" ]; then
    cat "$RESOLVED_REF_FILE"
  else
    echo "$REQUESTED_REF"
  fi
}

write_paths_from_tree() {
  local json_python="$1"
  local tree_json="$2"
  local output_file="$3"

  "$json_python" - "$tree_json" "$output_file" <<'PY'
import json
import sys

tree_json, output_file = sys.argv[1], sys.argv[2]

with open(tree_json, "r", encoding="utf-8") as handle:
    data = json.load(handle)

if data.get("truncated"):
    print("GitHub tree response was truncated.", file=sys.stderr)
    sys.exit(2)

paths = []
for item in data.get("tree", []):
    if item.get("type") != "blob":
        continue

    path = item.get("path", "")
    if not path.endswith((".py", ".sh", ".txt", ".json")):
        continue
    if path.startswith("test") or path.startswith("docs"):
        continue
    if "__pycache__" in path:
        continue
    if "generate_manifest" in path or path.endswith(".spec"):
        continue

    paths.append(path)

with open(output_file, "w", encoding="utf-8") as handle:
    handle.write("\n".join(sorted(paths)))
    handle.write("\n")
PY
}

write_paths_from_manifest() {
  local manifest_file="$1"
  local output_file="$2"

  sed 's/\r$//' "$manifest_file" \
    | grep -v '^[[:space:]]*$' \
    | grep -v '^[[:space:]]*#' \
    | sort > "$output_file"
}

echo "=== OpenFork DGN Client Bootstrap ==="
echo "Repository: $REPO"
echo "Requested ref: $REQUESTED_REF"

JSON_PYTHON=$(find_json_python)
if [ -n "$JSON_PYTHON" ]; then
  RESOLVED_REF=$(resolve_ref "$JSON_PYTHON")
else
  RESOLVED_REF="$REQUESTED_REF"
fi

BASE_URL="$RAW_BASE/$RESOLVED_REF"
API_URL="https://api.github.com/repos/$REPO/git/trees/$RESOLVED_REF?recursive=1"

echo "Resolved ref: $RESOLVED_REF"
echo "Fetching file list from GitHub API..."

USE_MANIFEST=false
if [ -n "$JSON_PYTHON" ] && curl --fail --location --proto '=https' --tlsv1.2 --silent --show-error "$API_URL" -o "$TREE_JSON"; then
  if ! write_paths_from_tree "$JSON_PYTHON" "$TREE_JSON" "$PATH_LIST"; then
    echo "Warning: Could not use GitHub tree API response. Falling back to manifest..."
    USE_MANIFEST=true
  fi
else
  echo "Warning: GitHub API unavailable or Python JSON parser missing. Falling back to manifest..."
  USE_MANIFEST=true
fi

if [ "$USE_MANIFEST" = "true" ]; then
  download_openfork_file "$BASE_URL/manifest.txt" "$MANIFEST_FILE"
  wait_for_download_batch
  write_paths_from_manifest "$MANIFEST_FILE" "$PATH_LIST"
fi

FILE_COUNT=$(grep -cve '^[[:space:]]*$' "$PATH_LIST" || true)
if [ "$FILE_COUNT" -le 0 ]; then
  echo "ERROR: Bootstrap file list is empty for ref '$RESOLVED_REF'."
  exit 1
fi

if [ "${OPENFORK_BOOTSTRAP_ARCHIVE:-true}" = "true" ] && copy_openfork_archive; then
  echo "Copied $FILE_COUNT client files from GitHub archive."
else
  echo "Warning: archive bootstrap unavailable; falling back to individual raw file downloads."
  echo "Downloading $FILE_COUNT client files..."
  count=0
  while IFS= read -r file || [ -n "$file" ]; do
    file=$(echo "$file" | tr -d '\r')
    [ -z "$file" ] && continue

    echo "  -> Downloading $file"
    download_openfork_file "$BASE_URL/$file" "$STAGING_DIR/$file" &

    count=$((count + 1))
    if [ "$count" -ge 20 ]; then
      wait_for_download_batch
      count=0
    fi
  done < "$PATH_LIST"
  wait_for_download_batch
fi

BOOTSTRAP_ROOT=$(pwd -P)
SHOULD_PRUNE=false
if [ "${OPENFORK_BOOTSTRAP_PRUNE:-auto}" = "true" ]; then
  SHOULD_PRUNE=true
elif [ "${OPENFORK_BOOTSTRAP_PRUNE:-auto}" = "auto" ] && [ "$BOOTSTRAP_ROOT" = "/opt/dgn-client" ]; then
  SHOULD_PRUNE=true
fi

if [ "$SHOULD_PRUNE" = "true" ]; then
  while IFS= read -r file || [ -n "$file" ]; do
    file=$(echo "$file" | tr -d '\r')
    [ -z "$file" ] && continue
    top="${file%%/*}"
    [ -n "$top" ] && echo "$top"
  done < "$PATH_LIST" | sort -u > "$MANAGED_TOPS"

  while IFS= read -r top || [ -n "$top" ]; do
    case "$top" in
      ""|"."|".."|/*)
        continue
        ;;
    esac
    rm -rf "$top"
  done < "$MANAGED_TOPS"
else
  echo "Skipping managed-file prune outside /opt/dgn-client. Set OPENFORK_BOOTSTRAP_PRUNE=true to force."
fi

cp -a "$STAGING_DIR"/. .

find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

cat > .installed-ref << EOF
repo=$REPO
requested_ref=$REQUESTED_REF
resolved_ref=$RESOLVED_REF
file_count=$FILE_COUNT
installed_at=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
EOF

# Install Python dependencies if requested
if [ "$INSTALL_DEPS" = "true" ]; then
  echo "Installing Python dependencies..."

  # Detect Python executable
  if [ -x "/usr/bin/python" ]; then
    PYTHON_EXE="/usr/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_EXE=$(command -v python)
  elif command -v python3 >/dev/null 2>&1; then
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
