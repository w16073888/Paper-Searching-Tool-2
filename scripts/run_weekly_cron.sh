#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/scu101/miniconda3/envs/mobile-cup/bin/python}"

export HOTSPOT_CONFIG="${HOTSPOT_CONFIG:-$PROJECT_ROOT/config.server.yaml}"
export HF_HOME="${HF_HOME:-/home/scu101/models/hf-cache}"
export QWEN_LOCAL_API_KEY="${QWEN_LOCAL_API_KEY:-local-qwen}"
export ARXIV_WEEKLY_DATA="${ARXIV_WEEKLY_DATA:-$PROJECT_ROOT/data/arxiv-weekly}"
export ARXIV_WEEKLY_CACHE="${ARXIV_WEEKLY_CACHE:-$ARXIV_WEEKLY_DATA/cache}"

mkdir -p "$ARXIV_WEEKLY_DATA/state"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/daily_arxiv_sync.py" "$@"
