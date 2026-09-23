#!/usr/bin/env python3
"""Draft acceptance on free-form output. The server prints `slot 0: ... accepted (X%)` per request;
this only drives the requests. 256 tokens each, chat mode, temperature 0 and 1.0.

    URL=http://127.0.0.1:8099/v1/chat/completions draftacc.py
"""
import json, os, time, urllib.request

URL = os.environ.get("URL", "http://127.0.0.1:8099/v1/chat/completions")
PROMPTS = [
    "Explain in about 200 words how warfarin interacts with vitamin K and why INR monitoring matters.",
    "Write a Python function that parses an ISO 8601 date string without any library, with comments.",
    "Summarise the main causes of the First World War in five short paragraphs.",
]
for temp in (0.0, 1.0):
    for p in PROMPTS:
        body = {"model": "affinity", "messages": [{"role": "user", "content": p}],
                "max_tokens": 256, "temperature": temp, "thinking_mode": "chat"}
        req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
        t0 = time.time()
        d = json.load(urllib.request.urlopen(req, timeout=3600))
        n = d.get("usage", {}).get("completion_tokens", 0)
        txt = (d["choices"][0]["message"].get("content") or "")
        print(f"temp {temp}  {n:4d} tok  {n/max(time.time()-t0,1e-9):6.1f} tok/s  {txt[:70]!r}", flush=True)
