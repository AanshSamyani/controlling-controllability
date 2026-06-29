"""Run the EXACT CoT-Control benchmark on a locally-served (vLLM) model.

The benchmark's OpenRouterLLM hard-codes its base_url and uses `requests`, so env
vars can't redirect it. But `run_evaluation(..., llm=...)` accepts an injected
BaseLLM, and `_parse_response` reads `message.reasoning_content` first (exactly
what vLLM's --reasoning-parser emits). So we subclass OpenRouterLLM, override
base_url, and inject it — reusing all of the harness's payload/parse/CSV logic.

Prereqs (on the server):
  1. git clone https://github.com/YuehHanChen/CoTControl
  2. pip install -r CoTControl/CoT-Control-QA/requirements.txt   (or `uv sync`)
  3. ./serve_qwen.sh <model> 8000          # vLLM, --served-model-name local
  4. export OPENAI_API_KEY=...             # only for grading (gpt-5-mini judge)

Run (loops the 10 modes x 3 QA datasets, then grades each):
  python cotcontrol_local.py \
      --cotcontrol_dir /path/to/CoTControl/CoT-Control-QA \
      --base_url http://localhost:8000/v1 --model local --grade

Compare base vs fine-tuned by serving each in turn and re-running with a different
--log_dir. The agent settings (SWE-bench/BFCL) need Docker and are skipped here.
Nothing runs on import.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

QA_DATASETS = ["gpqa_w_keyword.csv", "hle_w_keyword.csv", "mmlu_pro_mini_w_keyword.csv"]
MODES = ["baseline", "word_suppression", "multiple_word_suppression", "repeat_sentences",
         "end_of_sentence", "lowercase_thinking", "uppercase_thinking", "alternating_case",
         "meow_between_words", "ignore_question"]


def build_local_llm(OpenRouterLLM, base_url, model):
    class LocalServerLLM(OpenRouterLLM):
        def __init__(self):
            super().__init__(model=model, api_key=os.getenv("OPENROUTER_API_KEY", "EMPTY"))
            self.base_url = base_url  # override the hard-coded OpenRouter URL
    return LocalServerLLM()


async def run(args):
    cc_dir = os.path.abspath(args.cotcontrol_dir)  # resolve BEFORE chdir
    sys.path.insert(0, cc_dir)
    os.chdir(cc_dir)  # harness uses relative datasets/ and logs/ paths
    from llm import OpenRouterLLM            # noqa: E402
    from run_cceval import run_evaluation    # noqa: E402

    llm = build_local_llm(OpenRouterLLM, args.base_url, args.model)
    datasets = args.datasets or QA_DATASETS
    modes = args.modes or MODES

    produced = []
    for dsname in datasets:
        for mode in modes:
            print(f"[run] dataset={dsname} mode={mode}", flush=True)
            path = await run_evaluation(
                model_name=args.model,
                data_file=Path("datasets") / dsname,
                mode=mode,
                max_tokens=args.max_tokens,
                reasoning_tokens=0,            # keep the proprietary "reasoning" field off
                max_concurrency=args.max_concurrency,
                log_dir=args.log_dir,
                llm=llm,
            )
            produced.append(path)
            if args.grade and mode != "baseline":
                subprocess.run([sys.executable, "grade_compliance_csv.py",
                                "--input_file", str(path)], check=False)

    print("\nresults CSVs:")
    for p in produced:
        print(" ", p)
    print("\ngraded CSVs are under <log_dir>/graded_results/ (see check_grader_agreement.py)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cotcontrol_dir", required=True,
                    help="path to the cloned CoTControl/CoT-Control-QA directory")
    ap.add_argument("--model", default="local", help="must match vLLM --served-model-name")
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--log_dir", default="results", help="subfolder under logs/ for outputs")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--modes", nargs="*", default=None)
    ap.add_argument("--max_tokens", type=int, default=25000)
    ap.add_argument("--max_concurrency", type=int, default=16)
    ap.add_argument("--grade", action="store_true")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
