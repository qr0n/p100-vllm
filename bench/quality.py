#!/usr/bin/env python3
"""Quality gate for vLLM-on-P100 perf candidates.

Sends fixed real-text prompts (raw completions, temperature 0, one at a time)
and compares greedy tokens + logprobs against the stored reference.

  PORT=8000 NAME=<served-model-name> quality.py record|check

  quality.py record  -> writes reference.json (baseline build ONLY)
  quality.py check   -> compares; exit 1 on failure

A candidate passes only if every prompt's greedy continuation matches the
reference token-for-token for at least MIN_PREFIX tokens, and the mean absolute
logprob difference over the common prefix stays under MAX_MEAN_DLOGP.
fp16 reductions legitimately move with kernel changes, so exact bitwise equality
is not required, but a changed argmax early in a continuation is a failure --
unless the reference itself was a near-tie there (top-1 vs top-2 within
TIE_MARGIN nats), in which case either pick is legitimate and comparison stops.
The same build must pass against itself (prefix caching alone moves fp16 bits).
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

URL = f"http://localhost:{os.environ.get('PORT', '8000')}/v1/completions"
NAME = os.environ.get("NAME", "qwen3.8-27b-gptq")  # --served-model-name
REF = Path(__file__).with_name("reference.json")
MAX_TOKENS = 64
MIN_PREFIX = 32
MAX_MEAN_DLOGP = 0.02
TIE_MARGIN = 0.05

PROMPTS = [
    "The Treaty of Westphalia, signed in 1648, ended",
    "def quicksort(arr):\n    \"\"\"Sort a list using quicksort.\"\"\"\n",
    "Photosynthesis converts light energy into chemical energy. In the light-dependent reactions,",
    "The HBM2 memory on the Tesla P100 provides",
    "Q: A train leaves at 3:15 pm and travels 180 km at 72 km/h. When does it arrive?\nA:",
    "In Rust, the borrow checker enforces that",
    "La Révolution française a commencé en 1789 lorsque",
    "SELECT customer_id, SUM(amount) AS total\nFROM orders\nWHERE",
]


def complete(prompt: str) -> tuple[list[str], list[float], list[float]]:
    body = json.dumps({
        "model": NAME, "prompt": prompt, "max_tokens": MAX_TOKENS,
        "temperature": 0, "logprobs": 2, "seed": 0,
    }).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    lp = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["logprobs"]
    margins = []
    for top in lp["top_logprobs"]:
        v = sorted(top.values(), reverse=True)
        margins.append(v[0] - v[1] if len(v) > 1 else float("inf"))
    return lp["tokens"], lp["token_logprobs"], margins


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    got = [dict(zip(("tokens", "logprobs", "margins"), complete(p))) for p in PROMPTS]
    if mode == "record":
        if REF.exists():
            print(f"{REF} exists; refusing to overwrite the reference")
            return 1
        REF.write_text(json.dumps({"prompts": PROMPTS, "results": got}, indent=1))
        print(f"recorded {len(got)} references")
        return 0

    ref = json.loads(REF.read_text())
    assert ref["prompts"] == PROMPTS, "prompt set changed; the reference is invalid"
    ok = True
    for i, (r, g) in enumerate(zip(ref["results"], got)):
        n = 0
        while n < min(len(r["tokens"]), len(g["tokens"])) and r["tokens"][n] == g["tokens"][n]:
            n += 1
        d = [abs(a - b) for a, b in zip(r["logprobs"][:n], g["logprobs"][:n])]
        mean_d = sum(d) / len(d) if d else float("inf")
        full = n == len(r["tokens"])
        tie = not full and r["margins"][n] < TIE_MARGIN
        passed = (full or tie or n >= MIN_PREFIX) and mean_d <= MAX_MEAN_DLOGP
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] prompt {i}: match {n}/{len(r['tokens'])}"
              f" mean|dlogp| {mean_d:.4f}"
              + (f" (ref near-tie at {n}: margin {r['margins'][n]:.3f})" if tie else ""))
    print("QUALITY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
