#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/data/dataset/Objects365_2020}"
JOBS="${JOBS:-1}"

if [[ ! -d "$ROOT" ]]; then
  echo "Dataset root not found: $ROOT" >&2
  exit 1
fi

extract_one() {
  local archive="$1"
  local target
  target="$(dirname "$archive")"
  local marker="${archive}.extracted"

  if [[ -f "$marker" ]]; then
    echo "[skip] $archive"
    return 0
  fi

  echo "[extract] $archive -> $target"
  tar -xzf "$archive" -C "$target"
  touch "$marker"
}

export -f extract_one

find "$ROOT" -type f -name '*.tar.gz' | sort | while read -r archive; do
  extract_one "$archive"
done

echo "[done] extraction finished under $ROOT"
