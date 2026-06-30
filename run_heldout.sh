#!/usr/bin/env bash
# Held-out CONSTRAINT-TYPE generalization: evaluate BASE and FINE-TUNED models on
# constraint types that were NEVER in training (heldout_benchmark + heldout_transfer:
# uppercase/lowercase/alternating case, numbered/JSON structure, ignore-question,
# boundary/end/insert mirrors). Answers: did SFT teach the general meta-skill of
# CoT control, or just the 16 trained constraints?
#
#   nohup bash run_heldout.sh > heldout.log 2>&1 &
#   tail -f heldout.log
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

log "STEP 1: build held-out pairs (constraint types never trained)"
[ -f data/pairs.heldout.jsonl ] || uv run python cotctrl/build_pairs.py --heldout 270

log "STEP 2: BASE model rollouts + judge on held-out"
[ -f data/rollouts.heldout.base.jsonl ] || uv run python cotctrl/qwen_rollouts.py --model "$BASE"   --pairs data/pairs.heldout.jsonl --out data/rollouts.heldout.base.jsonl --batch_size 256
[ -f data/judged.heldout.base.jsonl ]   || uv run python cotctrl/cot_judge.py     --rollouts data/rollouts.heldout.base.jsonl --out data/judged.heldout.base.jsonl

log "STEP 3: FINE-TUNED model rollouts + judge on held-out"
[ -f data/rollouts.heldout.ft.jsonl ] || uv run python cotctrl/qwen_rollouts.py --model "$MERGED" --pairs data/pairs.heldout.jsonl --out data/rollouts.heldout.ft.jsonl --batch_size 256
[ -f data/judged.heldout.ft.jsonl ]   || uv run python cotctrl/cot_judge.py     --rollouts data/rollouts.heldout.ft.jsonl --out data/judged.heldout.ft.jsonl

log "STEP 4: held-out generalization — base vs fine-tuned"
uv run python cotctrl/analyze.py data/judged.heldout.base.jsonl data/judged.heldout.ft.jsonl
echo "--- per-constraint (base) ---";      uv run python cotctrl/analyze.py data/judged.heldout.base.jsonl | sed -n '/by constraint_key/,$p'
echo "--- per-constraint (fine-tuned) ---"; uv run python cotctrl/analyze.py data/judged.heldout.ft.jsonl  | sed -n '/by constraint_key/,$p'
log "DONE. data/judged.heldout.{base,ft}.jsonl"
