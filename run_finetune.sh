#!/usr/bin/env bash
# Fine-tune Qwen3 for CoT controllability and evaluate the fine-tuned model on
# val/test, comparing to the baseline. Pipeline:
#   train rollouts -> SFT data (gpt-4.1-mini rewrite) -> LoRA SFT -> merge ->
#   rollouts+judge on val/test with the fine-tuned model -> baseline-vs-ft table.
#
# Requires: data/rollouts.train.jsonl (from run_rollouts.sh) and OPENAI_API_KEY.
#   nohup bash run_finetune.sh > finetune.log 2>&1 &
#   tail -f finetune.log
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
log() { echo "[$(date '+%F %T')] $*"; }

source "$ROOT/workspace_env.sh"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation OOM
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env" || true
if [ -f .env ]; then set -a; source .env; set +a; fi
: "${OPENAI_API_KEY:?set OPENAI_API_KEY in .env}"

BASE="${BASE_MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
ADAPTER="out/qwen3-4b-cotctrl-lora"
MERGED="out/qwen3-4b-cotctrl-merged"

log "STEP 0: uv sync (gen + train)"
uv sync --extra gen --extra train

[ -f data/rollouts.train.jsonl ] || { log "ERROR: data/rollouts.train.jsonl missing — run run_rollouts.sh first"; exit 1; }

log "STEP 1: build SFT data (gpt-4.1-mini rewrites train rollouts to obey each constraint; slow)"
[ -f data/sft.train.jsonl ] || uv run python cotctrl/make_sft_data.py \
    --rollouts data/rollouts.train.jsonl --out data/sft.train.jsonl
log "  SFT examples: $(wc -l < data/sft.train.jsonl)"

log "STEP 2: LoRA SFT (r=32, alpha=64)"
[ -f "$ADAPTER/adapter_config.json" ] || uv run python cotctrl/train_sft.py \
    --model "$BASE" --train_file data/sft.train.jsonl --output_dir "$ADAPTER" \
    --lora --lora_r 32 --lora_alpha 64 --epochs 3 --lr 1e-4 \
    --batch_size 1 --grad_accum 32 --max_seq_length 8192

log "STEP 3: merge LoRA -> $MERGED (fast inference)"
[ -f "$MERGED/config.json" ] || uv run python cotctrl/merge_lora.py --base "$BASE" --adapter "$ADAPTER" --out "$MERGED"

log "STEP 4: eval fine-tuned model on val + test (same pairs as baseline)"
[ -f data/rollouts.val.ft.jsonl ]  || uv run python cotctrl/qwen_rollouts.py --model "$MERGED" --pairs data/pairs.val.jsonl  --out data/rollouts.val.ft.jsonl  --batch_size 256
[ -f data/judged.val.ft.jsonl ]    || uv run python cotctrl/cot_judge.py --rollouts data/rollouts.val.ft.jsonl  --out data/judged.val.ft.jsonl
[ -f data/rollouts.test.ft.jsonl ] || uv run python cotctrl/qwen_rollouts.py --model "$MERGED" --pairs data/pairs.test.jsonl --out data/rollouts.test.ft.jsonl --batch_size 256
[ -f data/judged.test.ft.jsonl ]   || uv run python cotctrl/cot_judge.py --rollouts data/rollouts.test.ft.jsonl --out data/judged.test.ft.jsonl

log "STEP 5: baseline vs fine-tuned"
echo "===== VAL ====="; uv run python cotctrl/analyze.py data/judged.val.jsonl  data/judged.val.ft.jsonl
echo "===== TEST ====="; uv run python cotctrl/analyze.py data/judged.test.jsonl data/judged.test.ft.jsonl
log "DONE. adapter=$ADAPTER merged=$MERGED  judged: data/judged.{val,test}.ft.jsonl"
