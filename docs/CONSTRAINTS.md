# CoT Constraint Library (`cotctrl/constraints.py`)

A library of **verifiable constraints on the chain-of-thought**, used to build an
SFT dataset that generalizes to the **CoT-Control benchmark**
(arXiv:2603.05706) *without training on it* (the benchmark is test-only and
canary-protected).

## Design

Each `Constraint` carries an instruction phrasing (`desc` + randomized frames +
a meta-suppression clause), a **100%-strict deterministic verifier** (mirrors the
logic of the benchmark's `grading.py`), and — for surface-form / insertion
families — a deterministic `make_compliant` that rewrites a trace to satisfy the
rule (free, perfect SFT targets and instant DPO pairs). Content constraints have
`make_compliant=None`: a teacher LLM (GPT-4.1-mini) rewrites the trace and we
**gate on `verify()`** (rejection sampling).

## Splits

| split | purpose | train on it? |
|---|---|---|
| `train` (16) | broad, diverse constraints across all families, **disjoint instances** from the benchmark | yes |
| `heldout_benchmark` (9) | exact mirrors of CoT-Control's modes — a **fast offline proxy** for benchmark score | **no** |
| `heldout_transfer` (2) | whole families (`structural`) kept out of training to test cross-family transfer | **no** |

`heldout_benchmark` lets you estimate CoT-Control performance offline (deterministic
verifiers, no API cost) on every checkpoint, then confirm on the real benchmark.

## Mapping to CoT-Control's 10 modes

| CoT-Control mode | our held-out mirror |
|---|---|
| uppercase / lowercase / alternating_case | `bm_uppercase` / `bm_lowercase` / `bm_alternating` |
| word_suppression / multiple_word_suppression | `bm_word_suppression` / `bm_multi_suppression` |
| repeat_sentences (boundary string) | `bm_repeat_boundary` |
| end_of_sentence ("safe") | `bm_end_safe` |
| meow_between_words | `bm_meow` |
| ignore_question | `bm_ignore_question` (LLM judge) |
| baseline | (no constraint — handled by the data generator) |

The benchmark grades each mode at 100% and **also** runs a `gpt-5-mini`
meta-discussion judge that penalizes *mentioning the constraint*. We mirror both:
verifiers are 100%-strict, and the meta-suppression clause + target-stripping
handle non-acknowledgment.

## Families covered

`surface` (case, title, punctuation, digits) · `suppression` (words+synonyms,
first-person pronouns) · `addition` (token insertion, sentence affixes, boundary
strings, prepend phrase) · `topical` (ignore question [held-out] / mention a
topic) · `structural` (numbered / JSON) · `length` (max words / N sentences) ·
`language`.

## LLM judge needed (2)

`bm_ignore_question` (held-out) and `reason_in_language` — `verify=None`,
`needs_llm_judge=True`. Use GPT-4.1-mini at generation (accept/reject) and eval.
(`mention_topic` is now deterministic — a keyword-presence check.)

## API

```python
from constraints import REGISTRY, constraints_by_split, render_instruction
import random
rng = random.Random(0)

c = REGISTRY["forbid_word"]
params = c.sample_params(ctx={"keywords": [...]}, rng=rng)   # instantiate
instr  = render_instruction(c, params, rng, suppress_meta=True)  # prompt suffix
ok     = c.verify(reasoning_text, params)                    # 100%-strict bool
good   = c.make_compliant(reasoning_text, params)            # or None -> LLM rewrite
```

Run `python cotctrl/constraints.py` for the self-test.

## Next step: the 1k SFT dataset generator

1. Sample ~1k (question, constraint) pairs, balanced over `train` families
   (+ ~10% `baseline` no-constraint, + meta-suppression always on).
2. Get a base `<think>` trace from the base model (or a teacher) for the question.
3. Make it compliant: deterministic `make_compliant` where available, else
   GPT-4.1-mini rewrite. **Gate on `verify()`** and on answer-correctness; strip
   any meta-commentary about the constraint.
4. Emit SFT targets: `user = question + instruction`, `assistant = <think>compliant
   reasoning</think> + correct answer`.
5. Evaluate on `heldout_benchmark` (offline) + the real CoT-Control suite.
