#!/usr/bin/env bash
# Full experiment suite for one model (default Qwen3-14B), thinking mode throughout.
# Reuses the model-agnostic pairs in data/ for cross-model comparability; all
# model-specific outputs go to data/$MODEL_TAG/ and out/$MODEL_TAG-cotctrl-*.
#
#   nohup bash run_all.sh > all_14b.log 2>&1 &
#   tail -f all_14b.log
#
# Override the model:  BASE_MODEL=Qwen/Qwen3-8B MODEL_TAG=8b nohup bash run_all.sh ...
# Idempotent (skip-if-exists), so a drop/restart resumes. Excludes the heavy real
# CoT-Control benchmark (run_experiment.sh covers that separately).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
log() { echo "[$(date '+%F %T')] $*"; }

source "$ROOT/workspace_env.sh"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env" || true
if [ -f .env ]; then set -a; source .env; set +a; fi
: "${OPENAI_API_KEY:?set OPENAI_API_KEY in .env}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-14B}"
TAG="${MODEL_TAG:-14b}"
D="data/$TAG"
ADAPTER="out/$TAG-cotctrl-lora"
MERGED="out/$TAG-cotctrl-merged"
RBS=128                      # vLLM rollout chunk (smaller for the bigger model)
mkdir -p "$D"

log "model=$BASE_MODEL  tag=$TAG  out=$D"
uv sync --extra gen --extra train

# ---- shared, model-agnostic pairs (build once, reuse across models) --------
[ -f data/pairs.test.jsonl ]    || uv run python cotctrl/build_pairs.py --n_train 1000 --n_val 150 --n_test 250
[ -f data/pairs.heldout.jsonl ] || uv run python cotctrl/build_pairs.py --heldout 270

# ---- 1. baseline rollouts (thinking) + judge val/test ----------------------
log "STAGE 1: baseline rollouts (thinking) + judge"
[ -f $D/rollouts.train.jsonl ] || uv run python cotctrl/qwen_rollouts.py --model "$BASE_MODEL" --pairs data/pairs.train.jsonl --out $D/rollouts.train.jsonl --batch_size $RBS
[ -f $D/rollouts.val.jsonl ]   || uv run python cotctrl/qwen_rollouts.py --model "$BASE_MODEL" --pairs data/pairs.val.jsonl   --out $D/rollouts.val.jsonl   --batch_size $RBS
[ -f $D/rollouts.test.jsonl ]  || uv run python cotctrl/qwen_rollouts.py --model "$BASE_MODEL" --pairs data/pairs.test.jsonl  --out $D/rollouts.test.jsonl  --batch_size $RBS
[ -f $D/judged.val.jsonl ]  || uv run python cotctrl/cot_judge.py --rollouts $D/rollouts.val.jsonl  --out $D/judged.val.jsonl
[ -f $D/judged.test.jsonl ] || uv run python cotctrl/cot_judge.py --rollouts $D/rollouts.test.jsonl --out $D/judged.test.jsonl

# ---- 2. SFT data (on-policy: rewrite THIS model's train rollouts) -----------
log "STAGE 2: SFT data (gpt-4.1-mini rewrite)"
[ -f $D/sft.train.jsonl ] || uv run python cotctrl/make_sft_data.py --rollouts $D/rollouts.train.jsonl --out $D/sft.train.jsonl
log "  SFT examples: $(wc -l < $D/sft.train.jsonl)"

# ---- 3. LoRA SFT + merge (batch 2, grad-accum 8) ---------------------------
log "STAGE 3: LoRA SFT (r=32 a=64, batch 2 x grad-accum 8)"
[ -f $ADAPTER/adapter_config.json ] || uv run python cotctrl/train_sft.py \
    --model "$BASE_MODEL" --train_file $D/sft.train.jsonl --output_dir $ADAPTER \
    --lora --lora_r 32 --lora_alpha 64 --epochs 3 --lr 1e-4 \
    --batch_size 2 --grad_accum 8 --max_seq_length 8192
[ -f $MERGED/config.json ] || uv run python cotctrl/merge_lora.py --base "$BASE_MODEL" --adapter $ADAPTER --out $MERGED

# ---- 4. fine-tuned eval on val/test ----------------------------------------
log "STAGE 4: fine-tuned rollouts + judge (val/test)"
[ -f $D/rollouts.val.ft.jsonl ]  || uv run python cotctrl/qwen_rollouts.py --model "$MERGED" --pairs data/pairs.val.jsonl  --out $D/rollouts.val.ft.jsonl  --batch_size $RBS
[ -f $D/judged.val.ft.jsonl ]    || uv run python cotctrl/cot_judge.py --rollouts $D/rollouts.val.ft.jsonl  --out $D/judged.val.ft.jsonl
[ -f $D/rollouts.test.ft.jsonl ] || uv run python cotctrl/qwen_rollouts.py --model "$MERGED" --pairs data/pairs.test.jsonl --out $D/rollouts.test.ft.jsonl --batch_size $RBS
[ -f $D/judged.test.ft.jsonl ]   || uv run python cotctrl/cot_judge.py --rollouts $D/rollouts.test.ft.jsonl --out $D/judged.test.ft.jsonl

# ---- 5. held-out constraint-type generalization (base + ft) ----------------
log "STAGE 5: held-out constraint types (base + ft)"
[ -f $D/rollouts.heldout.base.jsonl ] || uv run python cotctrl/qwen_rollouts.py --model "$BASE_MODEL" --pairs data/pairs.heldout.jsonl --out $D/rollouts.heldout.base.jsonl --batch_size $RBS
[ -f $D/judged.heldout.base.jsonl ]   || uv run python cotctrl/cot_judge.py --rollouts $D/rollouts.heldout.base.jsonl --out $D/judged.heldout.base.jsonl
[ -f $D/rollouts.heldout.ft.jsonl ]   || uv run python cotctrl/qwen_rollouts.py --model "$MERGED" --pairs data/pairs.heldout.jsonl --out $D/rollouts.heldout.ft.jsonl --batch_size $RBS
[ -f $D/judged.heldout.ft.jsonl ]     || uv run python cotctrl/cot_judge.py --rollouts $D/rollouts.heldout.ft.jsonl --out $D/judged.heldout.ft.jsonl

# ---- 6. D2 faithfulness + monitorability (spontaneous hint) -----------------
log "STAGE 6: faithfulness + monitorability (hint)"
[ -f $D/hint.base.jsonl ] || uv run python cotctrl/eval/hint_rollouts.py --model "$BASE_MODEL" --out $D/hint.base.jsonl --limit 150
[ -f $D/hint.ft.jsonl ]   || uv run python cotctrl/eval/hint_rollouts.py --model "$MERGED" --out $D/hint.ft.jsonl --limit 150
uv run python cotctrl/eval/hint_score.py $D/hint.base.jsonl $D/hint.ft.jsonl

# ---- 7. D2b instructed evasion ---------------------------------------------
log "STAGE 7: instructed evasion (hint + --evade)"
[ -f $D/hint.base.evade.jsonl ] || uv run python cotctrl/eval/hint_rollouts.py --model "$BASE_MODEL" --out $D/hint.base.evade.jsonl --limit 150 --evade
[ -f $D/hint.ft.evade.jsonl ]   || uv run python cotctrl/eval/hint_rollouts.py --model "$MERGED" --out $D/hint.ft.evade.jsonl --limit 150 --evade
uv run python cotctrl/eval/hint_score.py $D/hint.base.evade.jsonl $D/hint.ft.evade.jsonl

# ---- 8. summary ------------------------------------------------------------
log "STAGE 8: SUMMARY ($TAG)"
echo "### VAL controllability (base -> ft) ###";      uv run python cotctrl/analyze.py $D/judged.val.jsonl  $D/judged.val.ft.jsonl
echo "### TEST controllability (base -> ft) ###";     uv run python cotctrl/analyze.py $D/judged.test.jsonl $D/judged.test.ft.jsonl
echo "### HELD-OUT types (base -> ft) ###";           uv run python cotctrl/analyze.py $D/judged.heldout.base.jsonl $D/judged.heldout.ft.jsonl
log "DONE. outputs in $D/  adapter=$ADAPTER  merged=$MERGED"
