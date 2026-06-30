"""Validate that our OFFLINE proxy verifiers agree with the EXACT CoT-Control
grader on the SAME reasoning traces.

This is the concrete answer to "do the proxy and the real eval agree?": point it
at a graded CSV produced by the benchmark
(CoT-Control-QA/logs/<log_dir>/graded_results/*_graded.csv), and for each
deterministic mode it recomputes compliance with our constraints.py verifiers and
compares to the benchmark's own `compliance` column. ~100% agreement means our
heldout_benchmark proxy faithfully tracks the real grader.

  python eval/check_grader_agreement.py --graded "/path/to/graded_results/*_graded.csv"

Excludes ignore_question (LLM-judged) and baseline. Nothing runs on import.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

# make the project-root constraints.py importable regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constraints import (  # noqa: E402
    BM_END_WORD, BM_INSERT_TOKEN, BM_MARKER,
    v_alternating, v_boundary, v_end_sentence, v_forbid_words,
    v_insert_between, v_lowercase, v_uppercase,
)


def _keywords(row, multi):
    raw = row.get("keywords_with_synonyms")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    entries = data if multi else data[:1]
    words = []
    for e in entries:
        words.append(e["keyword"])
        words += e.get("synonyms", [])
    return words or None


def our_compliance(row):
    """Our reimplemented verifier's verdict (1/0), or None to skip this mode."""
    mode = row.get("mode")
    r = row.get("reasoning")
    r = r if isinstance(r, str) else ""
    fixed = {
        "uppercase_thinking": lambda: v_uppercase(r, {}),
        "lowercase_thinking": lambda: v_lowercase(r, {}),
        "alternating_case":   lambda: v_alternating(r, {}),
        "repeat_sentences":   lambda: v_boundary(r, {"marker": BM_MARKER}),
        "end_of_sentence":    lambda: v_end_sentence(r, {"word": BM_END_WORD}),
        "meow_between_words": lambda: v_insert_between(r, {"token": BM_INSERT_TOKEN}),
    }
    if mode in fixed:
        return int(fixed[mode]())
    if mode in ("word_suppression", "multiple_word_suppression"):
        w = _keywords(row, mode == "multiple_word_suppression")
        return None if w is None else int(v_forbid_words(r, {"words": w}))
    return None  # baseline / ignore_question / unknown


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--graded", nargs="+", required=True,
                    help="graded CSV path(s) or glob(s)")
    a = ap.parse_args()

    paths = [p for g in a.graded for p in glob.glob(g)]
    if not paths:
        print("no files matched", a.graded)
        return

    agree = defaultdict(lambda: [0, 0])  # mode -> [n_agree, n_total]
    for path in paths:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            ours = our_compliance(row)
            theirs = row.get("compliance")
            if ours is None or theirs not in (0, 1, 0.0, 1.0):
                continue
            ag, n = agree[row["mode"]]
            agree[row["mode"]] = [ag + int(ours == int(theirs)), n + 1]

    print(f"{'mode':24s} {'agree':>8s} {'n':>6s}")
    tot_a = tot_n = 0
    for mode in sorted(agree):
        ag, n = agree[mode]
        tot_a += ag
        tot_n += n
        print(f"{mode:24s} {ag / n:8.1%} {n:6d}")
    if tot_n:
        print(f"{'OVERALL':24s} {tot_a / tot_n:8.1%} {tot_n:6d}")


if __name__ == "__main__":
    main()
