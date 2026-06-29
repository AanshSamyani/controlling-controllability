#!/usr/bin/env bash
# Example training commands for the SSH server. Edit paths/hparams as needed.
set -euo pipefail

MODEL="Qwen/Qwen3-4B-Thinking-2507"
OUT="out/qwen3-4b-cotctrl-lora"

# 0. sanity-check ONE rendered+tokenized example before committing to a run
python train_sft.py --model "$MODEL" --train_file data/sft.train.jsonl --dry_run

# 1. LoRA SFT (single GPU). Add --load_in_4bit for QLoRA on small VRAM.
python train_sft.py \
  --model "$MODEL" \
  --train_file data/sft.train.jsonl \
  --eval_file  data/sft.val.jsonl \
  --output_dir "$OUT" \
  --lora --lora_r 32 --lora_alpha 64 \
  --epochs 3 --lr 1e-4 --batch_size 8 --grad_accum 4 --max_seq_length 4096

# 2. merge the adapter so vLLM can serve it for the exact CoT-Control eval
python merge_lora.py --base "$MODEL" --adapter "$OUT" --out "${OUT}-merged"

# --- full fine-tune variant (multi-GPU) ---
# accelerate launch train_sft.py --model "$MODEL" \
#   --train_file data/sft.train.jsonl --output_dir out/qwen3-4b-cotctrl-full \
#   --lr 1e-5 --epochs 3 --batch_size 1 --grad_accum 16
