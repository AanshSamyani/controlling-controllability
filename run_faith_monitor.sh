#!/usr/bin/env bash
# D2: CoT FAITHFULNESS + MONITORABILITY, base vs fine-tuned.
# Hint paradigm: inject a hint at a wrong answer; among items where the hint flips
# the answer, measure (1) faithfulness = does the CoT admit the hint, and
# (2) monitorability = can a gpt-4.1-mini monitor reading ONLY the CoT catch it.
#
#   nohup bash run_faith_monitor.sh > faith.log 2>&1 &
#   tail -f faith.log
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
log() { echo "[$(date '+%F %T')] $*"; }

source "$ROOT/workspace_env.sh"
export PYTHONUNBUFFERED=1
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env" || true
if [ -f .env ]; then set -a; source .env; set +a; fi
: "${OPENAI_API_KEY:?set OPENAI_API_KEY in .env}"

BASE="${BASE_MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
MERGED="out/qwen3-4b-cotctrl-merged"
[ -f "$MERGED/config.json" ] || { log "ERROR: $MERGED missing — run run_finetune.sh first"; exit 1; }

log "STEP 0: uv sync (gen)"
uv sync --extra gen

log "STEP 1: BASE hinted+clean rollouts"
[ -f data/hint.base.jsonl ] || uv run python cotctrl/eval/hint_rollouts.py --model "$BASE"   --out data/hint.base.jsonl --limit 150

log "STEP 2: FINE-TUNED hinted+clean rollouts"
[ -f data/hint.ft.jsonl ] || uv run python cotctrl/eval/hint_rollouts.py --model "$MERGED" --out data/hint.ft.jsonl --limit 150

log "STEP 3: score faithfulness + monitorability (base vs fine-tuned)"
uv run python cotctrl/eval/hint_score.py data/hint.base.jsonl data/hint.ft.jsonl

log "DONE. data/hint.{base,ft}.jsonl"
