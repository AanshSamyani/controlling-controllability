#!/usr/bin/env bash
# Redirect ALL installs/caches into /workspace, because on this server $HOME=/root
# is NOT persistent and only /workspace survives. Source this before any uv / HF /
# torch work:  `source workspace_env.sh`  (run_experiment.sh sources it automatically).
#
# Without this, a server restart loses: the uv binary, the uv package cache, the
# uv-managed Python, and the ~8GB Qwen3 weights + datasets in the HF cache.
export WORKSPACE="${WORKSPACE:-/workspace}"

# generic XDG roots (catch torch/triton/vllm compile caches, etc.)
export XDG_CACHE_HOME="$WORKSPACE/.cache"
export XDG_DATA_HOME="$WORKSPACE/.local/share"

# uv: binary install dir, package cache, and managed-Python location
export UV_INSTALL_DIR="$WORKSPACE/.local/bin"
export UV_CACHE_DIR="$WORKSPACE/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$WORKSPACE/.local/share/uv/python"

# HuggingFace: model hub + datasets cache (the big one)
export HF_HOME="$WORKSPACE/.cache/huggingface"

# prefer a uv installed under /workspace if present
export PATH="$WORKSPACE/.local/bin:$PATH"

mkdir -p "$UV_INSTALL_DIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME"
