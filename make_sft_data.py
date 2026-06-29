"""Turn Qwen rollouts into compliant SFT targets.

For each rollout whose final answer was CORRECT (guarantees a correct target):
  - baseline (no constraint): keep the natural reasoning as-is.
  - constrained: ask GPT-4.1-mini to rewrite ONLY the reasoning so it (1) keeps
    the correct solution, (2) obeys the constraint, (3) never mentions it. Then
    for deterministic constraints, if the rewrite fails verify() but a
    make_compliant transform exists, apply it to the (meta-free) rewrite as a
    repair. Finally GATE on: verify()/judge satisfied AND no meta-mention.
  - emit a Qwen chat-format SFT row:
      user      = pair prompt
      assistant = "<think>{compliant reasoning}</think>\n\nANSWER: {gold}"

Run:  python make_sft_data.py --rollouts data/rollouts.train.jsonl --out data/sft.train.jsonl
Needs OPENAI_API_KEY. Nothing runs on import.
"""
from __future__ import annotations

import argparse
import json

from constraints import REGISTRY
from cot_judge import judge_constraint, mentions_constraint
from openai_utils import chat

REWRITE_PROMPT = """Rewrite the REASONING for the problem below so that it:
1) still correctly solves the problem and reaches the final answer {answer};
2) obeys this constraint on the reasoning: {desc};
3) never mentions, hints at, or refers to this constraint.
Output ONLY the rewritten reasoning text, nothing else.

Problem:
{question}

Original reasoning:
---
{reasoning}
---"""


def _assistant_text(reasoning, gold):
    return f"<think>{reasoning}</think>\n\nANSWER: {gold}"


def _satisfies(key, reasoning, params, question):
    """Deterministic verify, or LLM judge for the 3 judged constraints."""
    fake = dict(constraint_key=key, params=params, reasoning=reasoning,
                question=question, constraint_satisfied=None)
    return judge_constraint(fake)


def rewrite_one(rec, max_tries=3):
    """Return compliant reasoning string, or None if it can't be made to pass."""
    key, params = rec["constraint_key"], rec["params"]
    c = REGISTRY[key]
    desc = c.desc(params)
    base = rec["reasoning"]
    for _ in range(max_tries):
        out = chat(REWRITE_PROMPT.format(
            answer=rec["answer"], desc=desc, question=rec["question"],
            reasoning=base[:8000])).strip()
        candidate = out
        # repair deterministic constraints with the exact transform if needed
        if c.verify and not c.needs_llm_judge and not c.verify(candidate, params):
            if c.make_compliant is not None:
                candidate = c.make_compliant(candidate, params)
        # gate: satisfied AND meta-free
        if _satisfies(key, candidate, params, rec["question"]):
            probe = dict(constraint_key=key, reasoning=candidate)
            if not mentions_constraint(probe):
                return candidate
        base = candidate  # iterate from the latest attempt
    return None


def run(rollouts_path, out_path, require_correct):
    recs = [json.loads(l) for l in open(rollouts_path, encoding="utf-8")]
    kept = dropped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in recs:
            if require_correct and not r["answer_correct"]:
                dropped += 1
                continue
            if r["constraint_key"] is None:  # baseline
                reasoning = r["reasoning"]
            else:
                reasoning = rewrite_one(r)
                if reasoning is None:
                    dropped += 1
                    continue
            row = {
                "messages": [
                    {"role": "user", "content": r["prompt"]},
                    {"role": "assistant",
                     "content": _assistant_text(reasoning, r["answer"])},
                ],
                "meta": {"id": r["id"], "constraint_key": r["constraint_key"],
                         "constraint_family": r["constraint_family"],
                         "source": r["source"], "split": r["split"]},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    print(f"kept {kept} / dropped {dropped}  -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="data/rollouts.train.jsonl")
    ap.add_argument("--out", default="data/sft.train.jsonl")
    ap.add_argument("--allow_incorrect", action="store_true",
                    help="keep rollouts even if the base answer was wrong")
    a = ap.parse_args()
    run(a.rollouts, a.out, require_correct=not a.allow_incorrect)
