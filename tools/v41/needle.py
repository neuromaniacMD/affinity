#!/usr/bin/env python3
"""Does the model retrieve the planted fact? One request a fixture, greedy, chat mode.

The mkneedle.py fixtures plant `ZANTHIC-4471` in Case 3, far outside the sliding window. Before
d0b23f5 (V4.1's q was being RMS-normalised per head, which V4.1 does not do) every arm failed it and
the failure was blamed on the 2.875 bpw quant. This is the check that tells the two apart.

    URL=http://127.0.0.1:8099/v1/chat/completions needle.py <fixture>...
"""
import json, os, sys, time, urllib.request

URL = os.environ.get("URL", "http://127.0.0.1:8099/v1/chat/completions")
FACT = "ZANTHIC-4471"
ok = 0
for f in sys.argv[1:]:
    body = {"model": "affinity", "messages": [{"role": "user", "content": open(f).read()}],
            "max_tokens": 24, "temperature": 0.0, "thinking_mode": "chat"}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=7200))
    ans = (d["choices"][0]["message"].get("content") or "").strip()
    hit = FACT in ans
    ok += hit
    print(f"{os.path.basename(f):<10} {'PASS' if hit else 'FAIL'}  {time.time()-t0:6.1f}s  "
          f"{d.get('usage', {}).get('completion_tokens', '?')} tok  {ans[:80]!r}", flush=True)
print(f"needle: {ok}/{len(sys.argv) - 1}", flush=True)
