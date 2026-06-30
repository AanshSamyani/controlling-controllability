#!/usr/bin/env bash
# Generate base-Qwen rollouts on train/val/test (NO benchmark, NO vLLM server).
# Train rollouts feed make_sft_data.py (GPT-4.1-mini rewrites them to satisfy each
# constraint -> SFT targets). val/test rollouts are the baseline; this script
# skips any output that already exists.
#
#   nohup bash run_rollouts.sh > rollouts.log 2>&1 &
#   tail -f rollouts.log
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
log() { echo "[$(date '+%F %T')] $*"; }

# persist caches under /workspace; unbuffered logs
source "$ROOT/workspace_env.sh"
export PYTHONUNBUFFERED=1
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env" || true
if [ -f .env ]; then set -a; source .env; set +a; fi   # OPENAI_API_KEY for the val/test judge

BS=256   # vLLM batches well; large chunks = higher throughput on the H100

log "STEP 0: uv sync (core + gen/vLLM)"
uv sync --extra gen

log "STEP 1: build train/val/test pairs (skip if present)"
[ -f data/pairs.train.jsonl ] || uv run python build_pairs.py --n_train 1000 --n_val 150 --n_test 250

log "STEP 2: rollouts (skip any that already exist)"
[ -f data/rollouts.train.jsonl ] || uv run python qwen_rollouts.py --pairs data/pairs.train.jsonl --out data/rollouts.train.jsonl --batch_size $BS
[ -f data/rollouts.val.jsonl ]   || uv run python qwen_rollouts.py --pairs data/pairs.val.jsonl   --out data/rollouts.val.jsonl   --batch_size $BS
[ -f data/rollouts.test.jsonl ]  || uv run python qwen_rollouts.py --pairs data/pairs.test.jsonl  --out data/rollouts.test.jsonl  --batch_size $BS

log "STEP 3: judge val/test for the baseline numbers (train needs no judging)"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY in .env for the judge step}"
[ -f data/judged.val.jsonl ]  || uv run python cot_judge.py --rollouts data/rollouts.val.jsonl  --out data/judged.val.jsonl
[ -f data/judged.test.jsonl ] || uv run python cot_judge.py --rollouts data/rollouts.test.jsonl --out data/judged.test.jsonl

log "DONE."
log "  rollouts: data/rollouts.{train,val,test}.jsonl"
log "  baseline judged: data/judged.{val,test}.jsonl"
log "  next: make_sft_data.py on data/rollouts.train.jsonl to build SFT targets"
