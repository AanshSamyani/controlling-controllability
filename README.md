# controlling-controllability

Fine-tuning small reasoning models (Qwen3) to follow **verifiable constraints on
their chain-of-thought**, and measuring the effect on CoT **controllability,
faithfulness, and monitorability**. Built around the CoT-Control benchmark
(arXiv:2603.05706), held out as the test set.

## Layout
| file | role |
|---|---|
| `constraints.py` | verifiable CoT-constraint library (train + held-out splits) |
| `seed_datasets.py` | 5 reasoning datasets (gsm8k/arc/commonsenseqa/logiqa/strategyqa), normalized |
| `build_pairs.py` | sample (question, constraint) pairs → train/val/test |
| `qwen_rollouts.py` | base-model rollouts via vLLM |
| `cot_judge.py` | controllability / meta-discussion / accuracy (gpt-4.1-mini) |
| `make_sft_data.py` | on-policy rewrite → compliant SFT targets (gpt-4.1-mini) |
| `train_sft.py`, `merge_lora.py` | SFT (TRL) + LoRA merge |
| `eval/` | exact CoT-Control benchmark wiring + proxy-agreement check |
| `PIPELINE.md`, `CONSTRAINTS.md`, `eval/README_eval.md` | docs |

## Quickstart
See **`SETUP.md`** for the full, ordered command list. Uses `uv`; needs an
H100-class GPU for generation and a `.env` containing `OPENAI_API_KEY` (for
gpt-4.1-mini judging/rewriting). All judging and rewriting use **gpt-4.1-mini**.
