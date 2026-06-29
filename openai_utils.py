"""Thin OpenAI chat wrapper used by the judge and the rewriter.

Needs OPENAI_API_KEY in the environment. Default model gpt-4.1-mini.
Nothing runs on import.
"""
from __future__ import annotations

import os
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_MODEL = "gpt-4.1-mini"


def chat(user: str, system: str = "", model: str = DEFAULT_MODEL,
         temperature: float = 0.0, max_retries: int = 4) -> str:
    """Single-turn chat completion -> assistant text."""
    from openai import OpenAI  # lazy import
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": user}]
    last = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature)
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 - retry on transient API errors
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"OpenAI call failed after {max_retries} retries: {last}")
