#!/usr/bin/env bash
# End-to-end BASELINE experiment, designed to run detached under nohup:
#
#   cd /workspace/controlling-controllability
#   printf 'OPENAI_API_KEY=sk-...\n' > .env          # once
#   nohup bash run_experiment.sh > run.log 2>&1 &
#   tail -f run.log
#
# Idempotent: re-running re-uses the cloned benchmark and overwrites outputs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
log() { echo "[$(date '+%F %T')] $*"; }

# ---- persistence: send all caches/installs to /workspace ($HOME=/root is wiped) ----
source "$ROOT/workspace_env.sh"
export PYTHONUNBUFFERED=1   # live progress in nohup logs (Python stdout is block-buffered otherwise)

# ---- environment -----------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env" || true
if [ -f .env ]; then set -a; source .env; set +a; fi
: "${OPENAI_API_KEY:?set OPENAI_API_KEY in .env}"

# always stop any vLLM server we (or a previous run) left behind
cleanup() { pkill -f "vllm serve" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# Which CoT-Control QA datasets to benchmark (space-separated). Default: the
# smallest (mmlu_pro_mini, 300q) for fast iteration. Override via env, e.g.:
#   BENCH_DATASETS="gpqa_w_keyword.csv hle_w_keyword.csv mmlu_pro_mini_w_keyword.csv"
BENCH_DATASETS="${BENCH_DATASETS:-mmlu_pro_mini_w_keyword.csv}"

# ---- 0. deps ---------------------------------------------------------------
log "STEP 0: uv sync (core + gen/vLLM)"
uv sync --extra gen

# ---- 1. pairs (skip if already built) --------------------------------------
log "STEP 1: build train/val/test pairs"
[ -f data/pairs.test.jsonl ] || uv run python build_pairs.py --n_train 1000 --n_val 150 --n_test 250

# ---- 2. VAL baseline (skip steps whose output exists) ----------------------
log "STEP 2: VAL rollouts + judge"
[ -f data/rollouts.val.jsonl ] || uv run python qwen_rollouts.py --pairs data/pairs.val.jsonl  --out data/rollouts.val.jsonl
[ -f data/judged.val.jsonl ]   || uv run python cot_judge.py     --rollouts data/rollouts.val.jsonl --out data/judged.val.jsonl

# ---- 3. TEST baseline ------------------------------------------------------
log "STEP 3: TEST rollouts + judge"
[ -f data/rollouts.test.jsonl ] || uv run python qwen_rollouts.py --pairs data/pairs.test.jsonl --out data/rollouts.test.jsonl
[ -f data/judged.test.jsonl ]   || uv run python cot_judge.py     --rollouts data/rollouts.test.jsonl --out data/judged.test.jsonl

# ---- 4. exact CoT-Control benchmark ----------------------------------------
log "STEP 4: CoT-Control benchmark setup"
[ -d CoTControl ] || git clone https://github.com/YuehHanChen/CoTControl.git
# force gpt-4.1-mini judge (idempotent)
if grep -q 'gpt-5-mini' CoTControl/CoT-Control-QA/grade_compliance_csv.py; then
  sed -i 's/gpt-5-mini/gpt-4.1-mini/g' CoTControl/CoT-Control-QA/grade_compliance_csv.py
  log "  patched grader judge -> gpt-4.1-mini"
fi

log "STEP 4a: serve model (vLLM, background)"
uv run vllm serve Qwen/Qwen3-4B-Thinking-2507 \
  --served-model-name local --reasoning-parser qwen3 \
  --port 8000 --max-model-len 32768 > vllm.log 2>&1 &
SERVE_PID=$!
log "  vLLM pid=$SERVE_PID; waiting for /health (tail vllm.log for progress)"
READY=0
for _ in $(seq 1 180); do            # up to ~15 min (first run downloads weights)
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then READY=1; break; fi
  kill -0 "$SERVE_PID" 2>/dev/null || { log "  vLLM exited early — see vllm.log"; exit 1; }
  sleep 5
done
[ "$READY" = 1 ] || { log "  vLLM not ready (timeout) — see vllm.log"; exit 1; }
log "  vLLM ready"

log "STEP 4b: run 10 modes on [$BENCH_DATASETS] + grade (gpt-4.1-mini)"
uv run python eval/cotcontrol_local.py \
  --cotcontrol_dir CoTControl/CoT-Control-QA \
  --base_url http://localhost:8000/v1 --model local --log_dir base --grade \
  --datasets $BENCH_DATASETS

log "STEP 4c: proxy<->exact grader agreement"
uv run python eval/check_grader_agreement.py \
  --graded "CoTControl/CoT-Control-QA/logs/base/graded_results/*_graded.csv" || true

log "STEP 4d: stop vLLM"
cleanup

# ---- 5. aggregate ----------------------------------------------------------
log "STEP 5: benchmark summary by mode"
uv run python - <<'PY'
import glob, pandas as pd
g = glob.glob("CoTControl/CoT-Control-QA/logs/base/graded_results/*_graded.csv")
if g:
    df = pd.concat(pd.read_csv(p) for p in g)
    print(df.groupby("mode")[["compliance", "meta_discussion"]].mean())
else:
    print("no graded CSVs found")
PY

log "DONE."
log "  val:       data/judged.val.jsonl   (cot_judge summary printed above in STEP 2)"
log "  test:      data/judged.test.jsonl  (STEP 3)"
log "  benchmark: CoTControl/CoT-Control-QA/logs/base/graded_results/"
