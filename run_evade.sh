#!/usr/bin/env bash
# D2b: INSTRUCTED-EVASION capability. Same hint setup, but the model is explicitly
# told to USE the hint and HIDE that it did. Does the controllability-fine-tuned
# model evade the CoT monitor better than base? (the benign->evasion-capability test)
#
#   nohup bash run_evade.sh > evade.log 2>&1 &
#   tail -f evade.log
#
# Compare against the non-evade numbers (data/hint.{base,ft}.jsonl) to see the
# DROP in faithfulness/monitor-catch each model achieves when instructed to evade.
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

log "STEP 1: BASE evasion-instructed rollouts"
[ -f data/hint.base.evade.jsonl ] || uv run python cotctrl/eval/hint_rollouts.py --model "$BASE"   --out data/hint.base.evade.jsonl --limit 150 --evade

log "STEP 2: FINE-TUNED evasion-instructed rollouts"
[ -f data/hint.ft.evade.jsonl ] || uv run python cotctrl/eval/hint_rollouts.py --model "$MERGED" --out data/hint.ft.evade.jsonl --limit 150 --evade

log "STEP 3: score under evasion (base vs fine-tuned)"
uv run python cotctrl/eval/hint_score.py data/hint.base.evade.jsonl data/hint.ft.evade.jsonl

log "DONE. data/hint.{base,ft}.evade.jsonl (+ .scored.jsonl)"
log "  interpret: lower faithfulness / lower monitor catch under --evade = better evasion."
log "  compare to non-evade (data/hint.{base,ft}.jsonl) to see each model's evasion DROP."
