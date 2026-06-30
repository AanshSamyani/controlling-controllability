#!/usr/bin/env bash
# Standalone SFT example (run_finetune.sh wraps this end-to-end). Edit as needed.
set -euo pipefail

MODEL="Qwen/Qwen3-4B-Thinking-2507"
OUT="out/qwen3-4b-cotctrl-lora"

# 0. sanity-check ONE rendered+tokenized example before committing to a run
uv run python cotctrl/train_sft.py --model "$MODEL" --train_file data/sft.train.jsonl --dry_run

# 1. LoRA SFT (single H100). Add --load_in_4bit only if VRAM-constrained.
uv run python cotctrl/train_sft.py \
  --model "$MODEL" \
  --train_file data/sft.train.jsonl \
  --output_dir "$OUT" \
  --lora --lora_r 32 --lora_alpha 64 \
  --epochs 3 --lr 1e-4 --batch_size 8 --grad_accum 4 --max_seq_length 8192

# 2. merge the adapter so vLLM can serve it for evaluation
uv run python cotctrl/merge_lora.py --base "$MODEL" --adapter "$OUT" --out "${OUT%-lora}-merged"

# --- full fine-tune variant (multi-GPU) ---
# accelerate launch cotctrl/train_sft.py --model "$MODEL" \
#   --train_file data/sft.train.jsonl --output_dir out/qwen3-4b-cotctrl-full \
#   --lr 1e-5 --epochs 3 --batch_size 1 --grad_accum 16
