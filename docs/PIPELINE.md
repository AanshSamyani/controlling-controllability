# SFT data pipeline

Goal: a ~1k-example SFT set teaching a small Qwen model to follow arbitrary
verifiable CoT constraints, that generalizes to the held-out CoT-Control
benchmark. On-policy recipe: rewrite the model's **own** rollouts to be compliant.

## Files
| file | role |
|---|---|
| `cotctrl/constraints.py` | constraint library (28 constraints, verifiers, transforms) |
| `cotctrl/seed_datasets.py` | load + normalize 5 seed datasets; question formatting + answer checking |
| `cotctrl/build_pairs.py` | sample (question, constraint) pairs; train/val split |
| `cotctrl/qwen_rollouts.py` | run base Qwen on prompts → `<think>` reasoning + answer |
| `cotctrl/cot_judge.py` | score CoT-controllability (det. verify + GPT-4.1-mini judges) |
| `cotctrl/make_sft_data.py` | GPT-4.1-mini rewrites rollouts → compliant SFT targets |
| `cotctrl/openai_utils.py` | GPT-4.1-mini chat wrapper (`OPENAI_API_KEY`) |

## Seed datasets (5, diverse, disjoint from the benchmark)
gsm8k (math) · arc (science MCQ) · commonsenseqa (commonsense MCQ) ·
logiqa (logic MCQ) · strategyqa (multi-hop yes/no). None overlap CoT-Control's
GPQA/HLE/MMLU-Pro/SWE-bench/BFCL eval sets.

## Train / val
Questions are split disjointly. **Val uses the same `train` constraints** (new
questions only) — it measures generalization to unseen instances, not unseen
constraint types. The benchmark mirrors (`heldout_benchmark`) and `structural`
family (`heldout_transfer`) are never sampled here — they are evaluation-only.

## Order (run val first, as planned)
```bash
# 0. deps: pip install datasets transformers vllm openai ; export OPENAI_API_KEY=...

# 1. build pairs (writes data/pairs.train.jsonl, data/pairs.val.jsonl)
python cotctrl/build_pairs.py --n_train 1000 --n_val 120

# 2. Qwen rollouts on VAL first (baseline material + sanity check)
python cotctrl/qwen_rollouts.py --pairs data/pairs.val.jsonl --out data/rollouts.val.jsonl

# 3. judge the val baseline (how good is Qwen at controllability BEFORE training?)
python cotctrl/cot_judge.py --rollouts data/rollouts.val.jsonl --out data/judged.val.jsonl

# 4. once happy, rollouts on train, then rewrite both splits into SFT data
python cotctrl/qwen_rollouts.py --pairs data/pairs.train.jsonl --out data/rollouts.train.jsonl
python cotctrl/make_sft_data.py --rollouts data/rollouts.train.jsonl --out data/sft.train.jsonl
python cotctrl/make_sft_data.py --rollouts data/rollouts.val.jsonl   --out data/sft.val.jsonl
```

## Design notes
- **On-policy rewrite:** the SFT target is Qwen's own reasoning, rewritten by
  GPT-4.1-mini to satisfy the constraint and drop any meta-commentary, then
  **gated** on `verify()`/judge + a meta-mention check. Deterministic constraints
  get a `make_compliant` "repair" if the LLM rewrite isn't exactly compliant.
- **Correct-only targets:** by default only rollouts whose base answer was
  correct become SFT rows (the answer is fixed to gold), so every target is
  correct. `--allow_incorrect` overrides.
- **Baseline (~10%) examples** carry no constraint and keep the natural trace, so
  the model doesn't forget to just solve problems.
- **Offline proxy:** evaluate trained checkpoints on `heldout_benchmark`
  constraints (deterministic, no API cost) before spending on the real
  CoT-Control suite.
- **Training:** `cotctrl/train_sft.py` (TRL `SFTTrainer`, Qwen chat format, LoRA/QLoRA/full),
  `cotctrl/merge_lora.py`, `train.sh`, `requirements-train.txt`.
- **Exact eval:** `eval/` — `serve_qwen.sh` (vLLM + reasoning parser),
  `cotcontrol_local.py` (injects a local-server `BaseLLM` into the real harness),
  `check_grader_agreement.py` (proves the offline proxy matches the real grader),
  `eval/README_eval.md`.

## Full order
build_pairs → qwen_rollouts (val) → cot_judge (baseline) → qwen_rollouts (train)
→ make_sft_data (train+val) → `train.sh` (SFT + merge LoRA) → `eval/` (serve the
base and the fine-tuned model, run the exact CoT-Control suite, check agreement).
