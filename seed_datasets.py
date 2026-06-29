"""Seed question datasets for SFT data generation.

5 datasets spanning distinct reasoning types, all open and DISJOINT from the
CoT-Control benchmark's eval sets (GPQA / HLE / MMLU-Pro / SWE-bench / BFCL) to
avoid any train/test contamination:

  gsm8k         grade-school math word problems        answer: number
  arc           ARC-Challenge science MCQ              answer: letter
  commonsenseqa commonsense reasoning MCQ              answer: letter
  logiqa        logical reading-comprehension MCQ      answer: letter
  strategyqa    multi-hop implicit yes/no reasoning    answer: yes/no

Each loader normalizes to a common record:
  {qid, source, question, choices (list[str] | None), answer (canonical), answer_type}
  answer_type in {"number", "mcq", "yesno"}; for "mcq", answer is a letter and
  choices[i] corresponds to letter chr(65+i).

NOTE: HuggingFace dataset field names occasionally change between versions.
The loaders use defensive .get access; verify against the live cards if a load
fails. Nothing here runs on import.
"""
from __future__ import annotations

import hashlib
import random
import re

# source -> (hf_path, hf_config, split). Edit splits/sizes freely.
DATASETS = {
    "gsm8k":         ("openai/gsm8k",          "main",          "train"),
    "arc":           ("allenai/ai2_arc",       "ARC-Challenge", "train"),
    "commonsenseqa": ("tau/commonsense_qa",    None,            "train"),
    "openbookqa":    ("allenai/openbookqa",    "main",          "train"),
    "strategyqa":    ("ChilleD/StrategyQA",    None,            "train"),
}

_LETTERS = [chr(65 + i) for i in range(26)]


def _qid(source: str, question: str) -> str:
    return source + "-" + hashlib.sha1(question.strip().encode()).hexdigest()[:12]


# ---- per-source normalizers: raw example -> normalized record (or None) ---- #
def _norm_gsm8k(ex):
    q = ex.get("question", "").strip()
    raw = ex.get("answer", "")
    m = re.search(r"####\s*([-\d,\.]+)", raw)
    if not q or not m:
        return None
    num = m.group(1).replace(",", "").rstrip(".")
    return dict(question=q, choices=None, answer=num, answer_type="number")


def _norm_mcq(question, texts, gold_idx):
    if not question or not texts or gold_idx is None or gold_idx >= len(texts):
        return None
    return dict(question=question.strip(), choices=[t.strip() for t in texts],
                answer=_LETTERS[gold_idx], answer_type="mcq")


def _norm_arc(ex):
    ch = ex.get("choices", {}) or {}
    texts, labels = ch.get("text", []), ch.get("label", [])
    key = ex.get("answerKey")
    if key not in labels:
        return None
    return _norm_mcq(ex.get("question", ""), texts, labels.index(key))


def _norm_csqa(ex):
    ch = ex.get("choices", {}) or {}
    texts, labels = ch.get("text", []), ch.get("label", [])
    key = ex.get("answerKey")
    if key not in labels:
        return None
    return _norm_mcq(ex.get("question", ""), texts, labels.index(key))


def _norm_openbookqa(ex):
    ch = ex.get("choices", {}) or {}
    texts, labels = ch.get("text", []), ch.get("label", [])
    key = ex.get("answerKey")
    if key not in labels:
        return None
    return _norm_mcq(ex.get("question_stem", ""), texts, labels.index(key))


def _norm_strategyqa(ex):
    q = ex.get("question", "").strip()
    a = ex.get("answer")
    if q == "" or a is None:
        return None
    yn = "yes" if (a is True or str(a).lower() in ("true", "yes")) else "no"
    return dict(question=q, choices=None, answer=yn, answer_type="yesno")


_NORMALIZERS = {
    "gsm8k": _norm_gsm8k, "arc": _norm_arc, "commonsenseqa": _norm_csqa,
    "openbookqa": _norm_openbookqa, "strategyqa": _norm_strategyqa,
}


def load_seed_questions(sources=None, max_per_source=400, seed=0):
    """Return a shuffled, deduped list of normalized question records."""
    from datasets import load_dataset  # lazy import

    sources = sources or list(DATASETS)
    rng = random.Random(seed)
    records, seen = [], set()
    for src in sources:
        try:
            path, config, split = DATASETS[src]
            ds = load_dataset(path, config, split=split)
            idxs = list(range(len(ds)))
            rng.shuffle(idxs)
            kept = 0
            for i in idxs:
                if kept >= max_per_source:
                    break
                rec = _NORMALIZERS[src](ds[i])
                if rec is None:
                    continue
                qid = _qid(src, rec["question"])
                if qid in seen:
                    continue
                seen.add(qid)
                rec.update(qid=qid, source=src)
                records.append(rec)
                kept += 1
            print(f"[seed] {src}: kept {kept}")
        except Exception as e:  # skip unavailable / script-based / renamed datasets
            print(f"[seed] WARNING: skipping {src} ({type(e).__name__}: {e})")
    if not records:
        raise RuntimeError("no seed questions loaded from any source")
    rng.shuffle(records)
    return records


# --------------------------- prompt + scoring ------------------------------- #
def format_question(rec) -> str:
    """The task portion of the user prompt (constraint instruction appended later)."""
    q = f"Question: {rec['question']}"
    if rec["answer_type"] == "mcq":
        opts = "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(rec["choices"]))
        q += f"\n{opts}\n\nEnd your response with a line 'ANSWER: X' where X is the letter."
    elif rec["answer_type"] == "yesno":
        q += "\n\nEnd your response with a line 'ANSWER: yes' or 'ANSWER: no'."
    else:  # number
        q += "\n\nEnd your response with a line 'ANSWER: N' where N is the final number."
    return q


def extract_answer(text: str):
    m = re.findall(r"ANSWER:\s*(.+)", text, flags=re.IGNORECASE)
    return m[-1].strip() if m else None


def _norm_num(s: str):
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        m = re.search(r"-?\d+\.?\d*", s)
        return float(m.group()) if m else None


def check_answer(rec, model_output: str) -> bool:
    pred = extract_answer(model_output)
    if pred is None:
        return False
    if rec["answer_type"] == "number":
        a, b = _norm_num(pred), _norm_num(rec["answer"])
        return a is not None and b is not None and abs(a - b) < 1e-6
    if rec["answer_type"] == "yesno":
        return pred.lower().startswith(rec["answer"])
    m = re.search(r"[A-Za-z]", pred)  # mcq
    return bool(m) and m.group().upper() == rec["answer"]
