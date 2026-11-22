#!/usr/bin/env bash
# Full study pull from data.binance.vision. Each step logs and the next still runs if one fails.
set -uo pipefail
cd "$(dirname "$0")/.."
START=${START:-2025-11}; END=${END:-2026-10}; VOL_START=${VOL_START:-2025-09}
step() { echo "=== $(date -u +%FT%TZ) $*"; uv run pathrisk --data data "$@" || echo "=== STEP FAILED: $*"; }
step symbols
step listing
step fetch --dataset volume --start "$VOL_START" --end "$END"
step fetch --dataset mark   --start "$START"     --end "$END"
step fetch --dataset last   --start "$START"     --end "$END"
step delistings
echo "=== $(date -u +%FT%TZ) done"
