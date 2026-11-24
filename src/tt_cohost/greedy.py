# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Greedy completions against an OpenAI-compatible server, and an exact comparison of two runs."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

#: The default prompt set: short, varied, and long enough at 64 tokens to show any drift.
PROMPTS = [
    "The capital of France is", "Write a haiku about silicon:", "def fibonacci(n):",
    "The three primary colors are", "In 1969, Neil Armstrong", "Explain photosynthesis in one sentence:",
    "The quick brown fox", "A list of five prime numbers:", "Translate 'good morning' to Spanish:",
    "The largest planet in the solar system is", "Once upon a time, in a small village,",
    "SELECT name FROM users WHERE", "The speed of light is approximately", "Water boils at",
    "The author of Pride and Prejudice is", "Step 1 of baking bread:",
]


def load_prompts(path: Optional[str]) -> List[str]:
    """A JSON list of strings, or one prompt per non-empty line; the default set without a path."""
    if not path:
        return list(PROMPTS)
    text = Path(path).read_text()
    if text.lstrip().startswith("["):
        return [str(p) for p in json.loads(text)]
    return [line for line in text.splitlines() if line.strip()]


def served_model(base: str, timeout: float = 30) -> str:
    """The id the server answers to (its --served-model-name), so callers need not know it."""
    with urllib.request.urlopen(f"{base}/v1/models", timeout=timeout) as r:
        return json.loads(r.read())["data"][0]["id"]


def complete(base: str, model: str, prompt: str, max_tokens: int = 64, timeout: float = 600) -> dict:
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
                       "seed": 0}).encode()
    req = urllib.request.Request(f"{base}/v1/completions", body, {"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return {"prompt": prompt, "text": out["choices"][0]["text"], "seconds": round(time.monotonic() - t0, 3),
            "completion_tokens": (out.get("usage") or {}).get("completion_tokens")}


def run(base: str, prompts: List[str], max_tokens: int = 64, concurrency: int = 1) -> List[dict]:
    model = served_model(base)
    with ThreadPoolExecutor(concurrency) as ex:
        return list(ex.map(lambda p: complete(base, model, p, max_tokens), prompts))


def keep_busy(base: str, prompts: List[str], stop: threading.Event, max_tokens: int = 64) -> int:
    """Send ``prompts`` round and round until ``stop``; returns how many completed."""
    model, n = served_model(base), 0
    while not stop.is_set():
        complete(base, model, prompts[n % len(prompts)], max_tokens)
        n += 1
    return n


def compare(reference: List[dict], other: List[dict]) -> Dict[str, object]:
    """Per-prompt exact comparison; a difference names the first character that differs."""
    diffs = []
    for i, (a, b) in enumerate(zip(reference, other)):
        if a["text"] != b["text"]:
            at = next((k for k, (x, y) in enumerate(zip(a["text"], b["text"])) if x != y),
                      min(len(a["text"]), len(b["text"])))
            diffs.append({"index": i, "prompt": a["prompt"], "at_char": at,
                          "reference": a["text"][at:at + 40], "got": b["text"][at:at + 40]})
    if len(reference) != len(other):
        diffs.append({"index": min(len(reference), len(other)), "prompt": None, "at_char": None,
                      "reference": f"{len(reference)} rows", "got": f"{len(other)} rows"})
    total = len(reference)
    return {"identical": total - sum(1 for d in diffs if d["prompt"] is not None), "total": total, "diffs": diffs}
