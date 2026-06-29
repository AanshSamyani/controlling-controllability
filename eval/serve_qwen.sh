#!/usr/bin/env bash
# Serve a Qwen3 model with vLLM's OpenAI-compatible server so the CoT-Control
# harness can read its chain-of-thought via `message.reasoning_content`.
#
#   ./serve_qwen.sh                                   # base model
#   ./serve_qwen.sh out/qwen3-4b-cotctrl-merged 8000  # your fine-tuned (merged) model
set -euo pipefail

MODEL="${1:-Qwen/Qwen3-4B-Thinking-2507}"
PORT="${2:-8000}"

# --served-model-name 'local' is deliberate: the harness adds a proprietary
# "reasoning" request field only when the model NAME contains qwen3/thinking/r1/oss,
# which a vLLM server may reject. Naming it 'local' avoids that entirely.
vllm serve "$MODEL" \
  --served-model-name local \
  --reasoning-parser qwen3 \
  --port "$PORT" \
  --max-model-len 32768

# NOTE: the reasoning-parser flag name varies by vLLM version. If the above fails:
#   older builds:  vllm serve "$MODEL" --served-model-name local \
#                    --enable-reasoning --reasoning-parser deepseek_r1 --port "$PORT"
# Sanity check after it's up (reasoning_content must be non-empty):
#   curl -s localhost:$PORT/v1/chat/completions -H 'content-type: application/json' \
#     -d '{"model":"local","messages":[{"role":"user","content":"2+2? think first"}]}' | python -m json.tool
