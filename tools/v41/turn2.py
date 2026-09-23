#!/usr/bin/env python3
"""Two chat turns against the affinity server: what a prefix cache is for, measured.

Turn 1 sends the prompt; turn 2 re-sends it with turn 1's reply and a short follow-up, which is the
shape an agent loop has. With --prefix-cache-dir on the server, turn 2 should restore turn 1's state
and prefill only the new suffix — its TTFT is the number. Without it, turn 2 re-prefills everything.

Greedy, chat mode, and the texts are printed with a hash so two servers (cache on, cache off) can be
compared: a restore that changed the state would change turn 2.

    URL=http://127.0.0.1:8099/v1/chat/completions turn2.py <prompt-file> [label]
"""
import hashlib
import json
import os
import sys
import time
import urllib.request

URL = os.environ.get("URL", "http://127.0.0.1:8099/v1/chat/completions")
NGEN = int(os.environ.get("NGEN", "64"))


def chat(messages):
    body = {"model": "affinity", "messages": messages, "max_tokens": NGEN, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 0.0, "thinking_mode": "chat"}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    first, text, usage = None, [], {}
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
            piece = ch[0].get("delta", {}).get("content") if ch else None
            if piece:
                text.append(piece)
                if first is None:
                    first = time.time()
    end = time.time()
    ctok = usage.get("completion_tokens") or len(text)
    if ctok > NGEN + 4:
        sys.exit(f"turn2: asked for {NGEN} tokens, got {ctok} -- serve with --max-tokens-floor 0")
    return usage.get("prompt_tokens") or 0, (first or end) - t0, "".join(text)


def main():
    f = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(f)
    prompt = open(f).read()
    m1 = [{"role": "user", "content": prompt}]
    p1, t1, a1 = chat(m1)
    m2 = m1 + [{"role": "assistant", "content": a1},
               {"role": "user", "content": "Answer again, in one short sentence."}]
    p2, t2, a2 = chat(m2)
    h = lambda s: hashlib.sha256(s.encode()).hexdigest()[:12]
    print(f"{label}: turn1 {p1} tok ttft {t1:.1f}s [{h(a1)}] | turn2 {p2} tok ttft {t2:.1f}s [{h(a2)}]",
          flush=True)
    print(f"  turn2 text: {a2[:200]!r}", flush=True)


if __name__ == "__main__":
    main()
