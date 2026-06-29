# Setup & first experiment (baseline)

SSH server, single H100 (80GB), `uv` for deps. Do everything under `/workspace`
(only that persists). First experiment = **baseline** rollouts + judging of the
base Qwen3-4B-Thinking model on val, test, and the real CoT-Control benchmark.
All judging/rewriting uses **gpt-4.1-mini**.

```bash
# 0. clone + env -----------------------------------------------------------
cd /workspace
git clone https://github.com/AanshSamyani/controlling-controllability.git
cd controlling-controllability

# add your key (.env is gitignored)
printf 'OPENAI_API_KEY=sk-REPLACE_ME\n' > .env

# IMPORTANT: $HOME=/root is NOT persistent on this server; only /workspace survives.
# Redirect uv + uv-python + HuggingFace caches into /workspace so weights/wheels persist.
source workspace_env.sh

# install uv (into /workspace) if not present, then create the env (core + vLLM)
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra gen

# export the key for this shell (the benchmark grader reads it from the env)
set -a && source .env && set +a

# 1. build train/val/test pairs -------------------------------------------
uv run python build_pairs.py --n_train 1000 --n_val 150 --n_test 250

# 2. baseline rollouts + judge on VAL --------------------------------------
uv run python qwen_rollouts.py --pairs data/pairs.val.jsonl  --out data/rollouts.val.jsonl
uv run python cot_judge.py     --rollouts data/rollouts.val.jsonl --out data/judged.val.jsonl

# 3. baseline rollouts + judge on TEST -------------------------------------
uv run python qwen_rollouts.py --pairs data/pairs.test.jsonl --out data/rollouts.test.jsonl
uv run python cot_judge.py     --rollouts data/rollouts.test.jsonl --out data/judged.test.jsonl

# 4. the REAL CoT-Control benchmark (gpt-4.1-mini judge) -------------------
git clone https://github.com/YuehHanChen/CoTControl.git
# force the benchmark grader to use gpt-4.1-mini (NOT gpt-5-mini)
sed -i 's/gpt-5-mini/gpt-4.1-mini/g' CoTControl/CoT-Control-QA/grade_compliance_csv.py

# serve the base model (background). 'local' name avoids the harness's proprietary reasoning field.
nohup uv run vllm serve Qwen/Qwen3-4B-Thinking-2507 \
  --served-model-name local --reasoning-parser qwen3 \
  --port 8000 --max-model-len 32768 > vllm.log 2>&1 &
# wait until ready (if this hangs, check vllm.log — older vLLM uses
#   --enable-reasoning --reasoning-parser deepseek_r1 )
until curl -sf http://localhost:8000/health >/dev/null; do echo "waiting for vllm..."; sleep 5; done

# run all 10 modes x 3 QA datasets and grade with gpt-4.1-mini
uv run python eval/cotcontrol_local.py \
  --cotcontrol_dir CoTControl/CoT-Control-QA \
  --base_url http://localhost:8000/v1 --model local \
  --log_dir base --grade

# confirm our offline proxy verifiers match the real grader on identical traces
uv run python eval/check_grader_agreement.py \
  --graded "CoTControl/CoT-Control-QA/logs/base/graded_results/*_graded.csv"

# stop the server
pkill -f "vllm serve" || true

# 5. read benchmark results ------------------------------------------------
uv run python - <<'PY'
import glob, pandas as pd
g = glob.glob("CoTControl/CoT-Control-QA/logs/base/graded_results/*_graded.csv")
df = pd.concat(pd.read_csv(p) for p in g)
print(df.groupby("mode")[["compliance", "meta_discussion"]].mean())
PY
```

`cot_judge.py` already prints val/test controllability, no-meta, and accuracy per
family and per constraint. Save these baseline numbers — they are the pre-finetune
reference for the SFT experiment.
