# Exact CoT-Control evaluation

Run the **real** CoT-Control benchmark (its questions + its grader) on your
base and fine-tuned Qwen models — the headline metric. The offline
`heldout_benchmark` proxy in `cot_judge.py` stays for cheap per-checkpoint
iteration; `check_grader_agreement.py` proves the two agree.

## Why a thin adapter (not env vars)
The harness's `OpenRouterLLM` hard-codes `base_url` and uses `requests`, so
`OPENAI_BASE_URL` etc. do nothing. But `run_evaluation(..., llm=...)` accepts an
injected `BaseLLM`, and `_parse_response` reads `message.reasoning_content` first
— which is exactly what vLLM's `--reasoning-parser` emits. `cotcontrol_local.py`
subclasses `OpenRouterLLM`, overrides `base_url`, and injects it. Zero edits to
the benchmark repo.

## Steps (on the SSH server)
```bash
# 1. get the benchmark + deps
git clone https://github.com/YuehHanChen/CoTControl
pip install -r CoTControl/CoT-Control-QA/requirements.txt   # or: cd CoTControl && uv sync

# 2. serve a model (separate terminal). Base first to get the pre-finetune baseline:
./eval/serve_qwen.sh Qwen/Qwen3-4B-Thinking-2507 8000
# (fine-tuned: merge LoRA via merge_lora.py, then ./eval/serve_qwen.sh out/...-merged 8000)

# 3. grading uses gpt-5-mini (ignore_question + meta-discussion judge)
export OPENAI_API_KEY=sk-...

# 4. run all 10 modes x 3 QA datasets, and grade. Use a distinct --log_dir per model.
python eval/cotcontrol_local.py \
  --cotcontrol_dir CoTControl/CoT-Control-QA \
  --base_url http://localhost:8000/v1 --model local \
  --log_dir base --grade
# ...then re-serve the fine-tuned model and run with --log_dir finetuned
```

## Reading the results
Grading prints per-file controllability + meta-discussion. To aggregate per mode
across the graded CSVs:
```python
import glob, pandas as pd
df = pd.concat(pd.read_csv(p) for p in glob.glob("CoTControl/CoT-Control-QA/logs/base/graded_results/*_graded.csv"))
print(df.groupby("mode")[["compliance", "meta_discussion"]].mean())
```
"Controllability without meta-discussion" = mean of `compliance == 1 & meta_discussion == 1`.

## Confirm the proxy tracks the real grader
```bash
python eval/check_grader_agreement.py \
  --graded "CoTControl/CoT-Control-QA/logs/base/graded_results/*_graded.csv"
```
Expect ~100% per deterministic mode; large gaps mean a verifier in `constraints.py`
drifted from the benchmark's `grading.py` and should be reconciled.

## Caveats (from reading the harness source)
- `run_evaluation` hard-wires `temperature=0.0`. Fine for deterministic
  controllability scoring; note Qwen3-Thinking docs suggest higher temp for raw
  capability — not changeable without editing the harness.
- `--reasoning-parser` flag/name varies by vLLM version; verify
  `reasoning_content` is populated (curl snippet in `serve_qwen.sh`). If it's
  empty, the `<think>` block lands in `content` and grading breaks.
- The grader's judge model is hard-coded to `gpt-5-mini`; if your key lacks it,
  `ignore_question` + meta-discussion grading skip (other modes grade offline).
- Agent settings (SWE-bench/BFCL) need Docker and are intentionally skipped; the
  3 QA datasets are the controllability core.
- Respect the benchmark canary: this is eval-only, never training data.
