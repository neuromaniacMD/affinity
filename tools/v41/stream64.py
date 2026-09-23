import json, sys, time, urllib.request
URL = sys.argv[1]; f = sys.argv[2]; mt = int(sys.argv[3])
body = {"model": "x", "messages": [{"role": "user", "content": open(f).read()}], "max_tokens": mt,
        "temperature": 0.0, "thinking_mode": "chat", "stream": True}
req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
t0 = time.time(); n = 0; out = []
with urllib.request.urlopen(req, timeout=7200) as r:
    for line in r:
        line = line.decode(errors="replace").strip()
        if not line.startswith("data:") or line.endswith("[DONE]"): continue
        try: d = json.loads(line[5:])
        except Exception: continue
        ch = d.get("choices") or []
        if ch:
            p = ch[0].get("delta", {}).get("content") or ""
            if p: n += 1; out.append(p)
            fr = ch[0].get("finish_reason")
            if fr: print("finish", fr)
        if n and n % 50 == 0: print(f"{n} tok {time.time()-t0:.0f}s ...{''.join(out)[-80:]!r}", flush=True)
print(f"DONE {n} tok {time.time()-t0:.0f}s: {''.join(out)[:300]!r}", flush=True)
