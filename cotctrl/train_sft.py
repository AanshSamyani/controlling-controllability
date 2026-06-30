"""SFT training for CoT-controllability (Qwen3 + TRL SFTTrainer).

Trains on chat-format JSONL produced by make_sft_data.py: each row is
  {"messages": [{"role":"user",...},{"role":"assistant","content":"<think>...</think>\n\nANSWER: X"}], "meta": {...}}

Single-GPU LoRA (recommended start):
  python train_sft.py \
      --train_file data/sft.train.jsonl --eval_file data/sft.val.jsonl \
      --model Qwen/Qwen3-4B-Thinking-2507 --lora \
      --output_dir out/qwen3-4b-cotctrl-lora --epochs 3 --lr 1e-4

Full fine-tune (more VRAM; use accelerate/deepspeed for multi-GPU):
  accelerate launch train_sft.py --train_file data/sft.train.jsonl \
      --model Qwen/Qwen3-4B-Thinking-2507 --output_dir out/qwen3-4b-cotctrl-full --lr 1e-5

QLoRA on a small GPU:  add --load_in_4bit
Inspect one fully-rendered, tokenized example without training:  --dry_run

Pinned deps in requirements-train.txt. Nothing runs on import; run on your server.
"""
from __future__ import annotations

import argparse

QWEN_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    p.add_argument("--train_file", default="data/sft.train.jsonl")
    p.add_argument("--eval_file", default=None)
    p.add_argument("--output_dir", default="out/qwen3-4b-cotctrl")
    # training
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_seq_length", type=int, default=8192)  # fit long <think> targets
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    # LoRA / quantization
    p.add_argument("--lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--dry_run", action="store_true",
                   help="print one rendered+tokenized example and exit")
    return p.parse_args()


def main():
    args = parse_args()
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- data (conversational: TRL applies the chat template to "messages") ----
    files = {"train": args.train_file}
    if args.eval_file:
        files["eval"] = args.eval_file
    ds = load_dataset("json", data_files=files)
    for split in ds:
        keep = [c for c in ds[split].column_names if c == "messages"]
        ds[split] = ds[split].select_columns(keep)

    if args.dry_run:
        ex = ds["train"][0]["messages"]
        rendered = tok.apply_chat_template(ex, tokenize=False)
        print("=== rendered chat example ===\n" + rendered)
        print("\n=== token count ===", len(tok(rendered)["input_ids"]))
        return

    # ---- model (+ optional 4-bit) ----
    model_kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    peft_config = None
    if args.lora or args.load_in_4bit:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=QWEN_LORA_TARGETS)

    cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        eval_strategy="epoch" if args.eval_file else "no",
        bf16=True,
        gradient_checkpointing=True,
        max_seq_length=args.max_seq_length,
        packing=False,
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds.get("eval"),
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
