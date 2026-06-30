"""Produce Qwen rollouts on a set of pairs.

Step 3 of the pipeline: run the BASE model (Qwen3-4B-Thinking) on each
(question + constraint) prompt to get its natural <think> reasoning + answer.
These rollouts serve two purposes:
  (a) baseline CoT-controllability measurement (-> cot_judge.py), and
  (b) raw material the GPT-4.1-mini rewriter turns into compliant SFT targets
      (-> make_sft_data.py).

Recommended: run on the VAL split first to sanity-check baseline + cost.

Run:  python qwen_rollouts.py --pairs data/pairs.val.jsonl --out data/rollouts.val.jsonl
Backends: --backend vllm (default) or hf. Nothing runs on import.
"""
from __future__ import annotations

import argparse
import json
import re

from constraints import REGISTRY
from seed_datasets import check_answer

THINK_RE = re.compile(r"<think>(.*?)</think>(.*)", re.DOTALL)


def split_reasoning(text: str):
    """Return (reasoning, answer_text) from a Qwen thinking generation."""
    m = THINK_RE.search(text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    if "</think>" in text:  # opening tag consumed by the chat template
        reasoning, _, answer = text.partition("</think>")
        return reasoning.replace("<think>", "").strip(), answer.strip()
    return "", text.strip()  # non-thinking output


class QwenRunner:
    def __init__(self, model="Qwen/Qwen3-4B-Thinking-2507", backend="vllm",
                 max_new_tokens=4096, temperature=0.6):
        self.model_name, self.backend = model, backend
        self.max_new_tokens, self.temperature = max_new_tokens, temperature
        self._load()

    def _load(self):
        if self.backend == "vllm":
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(self.model_name)
            self.llm = LLM(model=self.model_name)
            self.sp = SamplingParams(temperature=self.temperature, top_p=0.95,
                                     top_k=20, max_tokens=self.max_new_tokens)
        else:  # hf
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype="auto", device_map="auto")
            self.torch = torch

    def _render(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True)
        except TypeError:  # template without the enable_thinking kwarg
            return self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

    def generate(self, prompts: list[str]) -> list[str]:
        texts = [self._render(p) for p in prompts]
        if self.backend == "vllm":
            outs = self.llm.generate(texts, self.sp)
            return [o.outputs[0].text for o in outs]
        results = []
        for t in texts:  # hf path: simple per-prompt loop
            ids = self.tok(t, return_tensors="pt").to(self.model.device)
            with self.torch.no_grad():
                out = self.model.generate(
                    **ids, max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature, top_p=0.95, top_k=20,
                    do_sample=self.temperature > 0)
            results.append(self.tok.decode(out[0][ids.input_ids.shape[1]:],
                                           skip_special_tokens=True))
        return results


def run(pairs_path, out_path, model, backend, batch_size, limit,
        max_new_tokens, temperature):
    pairs = [json.loads(l) for l in open(pairs_path, encoding="utf-8")]
    if limit:
        pairs = pairs[:limit]
    print(f"loading {model} ({backend} backend); {len(pairs)} prompts to generate "
          f"(first run downloads weights)...", flush=True)
    runner = QwenRunner(model=model, backend=backend,
                        max_new_tokens=max_new_tokens, temperature=temperature)
    print("model loaded; generating", flush=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            gens = runner.generate([p["prompt"] for p in batch])
            for p, g in zip(batch, gens):
                reasoning, answer_text = split_reasoning(g)
                rec = dict(p)
                rec["raw_output"] = g
                rec["reasoning"] = reasoning
                rec["answer_text"] = answer_text
                rec["answer_correct"] = check_answer(p, answer_text or g)
                # deterministic constraint check (None if LLM-judged / baseline)
                key = p["constraint_key"]
                c = REGISTRY.get(key) if key else None
                if c and c.verify and not c.needs_llm_judge:
                    rec["constraint_satisfied"] = bool(c.verify(reasoning, p["params"]))
                else:
                    rec["constraint_satisfied"] = None
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  rollouts {min(i + batch_size, len(pairs))}/{len(pairs)}", flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs.val.jsonl")
    ap.add_argument("--out", default="data/rollouts.val.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.6)
    a = ap.parse_args()
    run(a.pairs, a.out, a.model, a.backend, a.batch_size, a.limit or None,
        a.max_new_tokens, a.temperature)
