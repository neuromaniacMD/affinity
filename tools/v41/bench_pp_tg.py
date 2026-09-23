#!/usr/bin/env python3
"""pp and tg for the affinity V4.1 lane, against a server that loaded the model once.

Prompt processing and token generation are two different questions and the usual mistake is to
answer only the first. pp is measured from the time to the FIRST token, which is what a prefill
actually costs a caller; tg from the tokens after it, so the prefill does not dilute the rate.

Each depth is run TWICE. The second pass is the interesting one for engram: its table is
demand-paged out of a 188.8 GiB mmap, so run one pays the cold page faults and run two reports
what a warm server does. Quote both — the gap IS the finding.

    URL=http://127.0.0.1:8080/v1/chat/completions bench_pp_tg.py <prompt-file> ...
"""
import json
import os
import sys
import time
import urllib.request

URL = os.environ.get("URL", "http://127.0.0.1:8080/v1/chat/completions")
MODEL = os.environ.get("MODEL", "affinity")
NGEN = int(os.environ.get("NGEN", "64"))


def run(prompt, max_tokens):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
    }
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    usage = {}
    n = 0
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            line = line.decode(errors="replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            try:
                d = json.loads(line[5:])
            except Exception:
                continue
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch and (ch[0].get("delta", {}).get("content") or
                       ch[0].get("delta", {}).get("reasoning_content")):
                n += 1
                if first is None:
                    first = time.time()
    end = time.time()
    ptok = usage.get("prompt_tokens") or 0
    ctok = usage.get("completion_tokens") or n
    ttft = (first or end) - t0
    gen = end - (first or end)
    return ptok, ctok, ttft, gen


def main():
    files = sys.argv[1:]
    run("hi", 4)                                   # warm the server's own first-call paths
    print(f"# {MODEL} @ {URL}   NGEN={NGEN}")
    print("# depth      prompt_tok |  pp tok/s  ttft s |  tg tok/s | "
          "pp2 tok/s  ttft2 s | tg2 tok/s")
    for f in files:
        prompt = open(f).read()
        rows = []
        for _ in range(2):
            rows.append(run(prompt, NGEN))
        (p1, c1, t1, g1), (p2, c2, t2, g2) = rows
        print(f"{os.path.basename(f):<10} {p1:11d} | {p1/max(t1,1e-9):9.1f} {t1:7.1f} | "
              f"{max(c1-1,1)/max(g1,1e-9):9.2f} | {p2/max(t2,1e-9):9.1f} {t2:8.1f} | "
              f"{max(c2-1,1)/max(g2,1e-9):9.2f}", flush=True)


if __name__ == "__main__":
    main()
