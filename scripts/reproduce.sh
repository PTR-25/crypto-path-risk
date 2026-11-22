#!/usr/bin/env bash
# Reproduce the paper end to end from public data.
#   1. download (data.binance.vision, about 31 GB, roughly 3 h on a home connection)
#   2. delisting cut and balanced cohort
#   3. the three analyses (mark price, balanced cohort, last traded price)
#   4. figures and tables, then the PDF (needs `tectonic`)
# Re-running is safe: downloaded files are cached and skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

START_M=${START_M:-2025-11}; END_M=${END_M:-2026-10}; VOL_START_M=${VOL_START_M:-2025-09}
START=${START:-2025-11-01}; END=${END:-2026-10-31}
LEV=${LEV:-2,3,5,10,20,25,50}
MARGINS=paper/data/venue_leverage.csv  # per-contract Binance maintenance rate and max leverage
run() { uv run pathrisk --data data "$@"; }

uv sync --all-groups
START=$START_M END=$END_M VOL_START=$VOL_START_M ./scripts/download_all.sh
run cohort --start "$START" --end "$END"
mkdir -p results/logs
run analyze --dataset mark --start "$START" --end "$END" --leverages "$LEV" --margins "$MARGINS"
run analyze --dataset mark --start "$START" --end "$END" --leverages "$LEV" --margins "$MARGINS" --symbols-file data/cohort.txt --tag mark-cohort
run analyze --dataset last --start "$START" --end "$END" --leverages "$LEV" --margins "$MARGINS"
uv run python -I scripts/make_paper_assets.py
(cd paper && tectonic -X compile main.tex && tectonic -X compile linkedin.tex)
echo "done: paper/main.pdf (technical) and paper/linkedin.pdf (reader edition)"
