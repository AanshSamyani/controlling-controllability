"""Merge a trained LoRA adapter into the base model for easy vLLM serving.

  python merge_lora.py --base Qwen/Qwen3-4B-Thinking-2507 \
      --adapter out/qwen3-4b-cotctrl-lora --out out/qwen3-4b-cotctrl-merged

The merged dir can then be served with vLLM (see serve_qwen.sh). vLLM can also
serve a LoRA adapter directly via --lora-modules, but a merged model is simplest.
Nothing runs on import.
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Thinking-2507")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        a.base, torch_dtype=torch.bfloat16, trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    merged.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.base, trust_remote_code=True).save_pretrained(a.out)
    print(f"merged model -> {a.out}")


if __name__ == "__main__":
    main()
