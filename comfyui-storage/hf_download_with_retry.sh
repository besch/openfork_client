#!/usr/bin/env bash
set -euo pipefail

repo_id="$1"
shift

attempts="${HF_DOWNLOAD_ATTEMPTS:-3}"
status=1

cleanup_incomplete_blobs() {
    local cache_root="${HF_HOME:-${HOME}/.cache/huggingface}/hub"
    find "${cache_root}" -type f -name "*.incomplete" -delete 2>/dev/null || true
}

for attempt in $(seq 1 "${attempts}"); do
    echo "Hugging Face download attempt ${attempt}/${attempts}: ${repo_id} $*"
    set +e
    huggingface-cli download "${repo_id}" "$@"
    status=$?
    set -e

    if [ "${status}" -eq 0 ]; then
        exit 0
    fi

    cleanup_incomplete_blobs
    if [ "${attempt}" -lt "${attempts}" ]; then
        sleep_seconds=$((attempt * 20))
        echo "Download failed with exit ${status}; retrying in ${sleep_seconds}s..."
        sleep "${sleep_seconds}"
    fi
done

echo "ERROR: Hugging Face download failed after ${attempts} attempts: ${repo_id} $*" >&2
echo "If the log mentions Errno 5 or .incomplete files, retry the build after checking Docker builder disk and I/O health." >&2
exit "${status}"
