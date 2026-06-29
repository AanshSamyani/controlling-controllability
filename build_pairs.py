"""Build (question, constraint) pairs, split into train + val.

- Questions are split disjointly into train/val pools (no question in both).
- Both splits draw from the SAME `train`-split constraints (val measures
  generalization to new questions/instances, not new constraint types).
- Coverage is balanced across train constraint families, with ~10% baseline
  (no constraint) so the model keeps solving normally.
- Held-out benchmark / transfer constraints are NOT sampled here; they are
  evaluation-only.

Run:  python build_pairs.py --n_train 1000 --n_val 120 --out data/
Writes data/pairs.train.jsonl and data/pairs.val.jsonl. Nothing runs on import.
"""
from __future__ import annotations

import argparse
import json
import os
import random

from constraints import REGISTRY, constraints_by_split, render_instruction
from seed_datasets import format_question, load_seed_questions


def _families():
    fams = {}
    for c in constraints_by_split("train"):
        fams.setdefault(c.family, []).append(c)
    return fams  # {family: [Constraint, ...]}


def _make_pair(rec, family, fams, rng, split, baseline_p):
    """Return one pair record (dict)."""
    base = dict(
        id=f"{rec['qid']}::{split}::{rng.randrange(10**9)}",
        split=split, source=rec["source"], qid=rec["qid"],
        question=rec["question"], choices=rec["choices"],
        answer=rec["answer"], answer_type=rec["answer_type"],
    )
    task = format_question(rec)
    if family == "baseline":
        base.update(constraint_key=None, constraint_family="baseline",
                    params={}, instruction="", prompt=task, suppress_meta=False)
        return base
    c = rng.choice(fams[family])
    params = c.sample_params(None, rng)
    instr = render_instruction(c, params, rng, suppress_meta=True)
    base.update(constraint_key=c.key, constraint_family=c.family, params=params,
                instruction=instr, prompt=f"{task}\n\n{instr}", suppress_meta=True)
    return base


def build(n_train, n_val, n_test, out_dir, max_per_source, seed):
    rng = random.Random(seed)
    questions = load_seed_questions(max_per_source=max_per_source, seed=seed)
    n_val_q = max(n_val, int(0.12 * len(questions)))
    n_test_q = max(n_test, int(0.12 * len(questions)))
    val_q = questions[:n_val_q]
    test_q = questions[n_val_q:n_val_q + n_test_q]
    train_q = questions[n_val_q + n_test_q:]

    fams = _families()
    fam_cycle = list(fams)  # round-robin order for balanced coverage

    def sample(n, qpool, split, baseline_p=0.10):
        pairs, i = [], 0
        for _ in range(n):
            rec = rng.choice(qpool)
            if rng.random() < baseline_p:
                family = "baseline"
            else:
                family = fam_cycle[i % len(fam_cycle)]
                i += 1
            pairs.append(_make_pair(rec, family, fams, rng, split, baseline_p))
        return pairs

    os.makedirs(out_dir, exist_ok=True)
    splits = [("train", n_train, train_q), ("val", n_val, val_q), ("test", n_test, test_q)]
    for split, n, qpool in splits:
        pairs = sample(n, qpool, split)
        path = os.path.join(out_dir, f"pairs.{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        # coverage report
        from collections import Counter
        fam_counts = Counter(p["constraint_family"] for p in pairs)
        src_counts = Counter(p["source"] for p in pairs)
        print(f"[{split}] {len(pairs)} pairs | families={dict(fam_counts)} "
              f"| sources={dict(src_counts)} -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_val", type=int, default=150)
    ap.add_argument("--n_test", type=int, default=250)
    ap.add_argument("--out", default="data")
    ap.add_argument("--max_per_source", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.n_train, a.n_val, a.n_test, a.out, a.max_per_source, a.seed)
