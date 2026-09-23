#!/usr/bin/env python3
"""mklong.py — the long-answer fixtures l{4,8,16,32,48,64}k.txt, from mkneedle.py's n*k.txt.

Same text at the same depths, but the question asks for a long answer, so a pp/tg bench measures decode over real
tokens: the needle question is answered in ~8 tokens, which made every "tg" in the first 09-23 runs a start-up artefact.
    cd <dir with n*k.txt> && python3 mklong.py
"""
Q = ("\n\nQuestion: write a detailed summary of each of the first twelve cases, one paragraph per case, "
     "covering age, city, conditions, medications and the working diagnosis.\nAnswer:")
for k in ("4k", "8k", "16k", "32k", "48k", "64k"):
    body = open(f"n{k}.txt").read().split("\n\nQuestion:")[0]
    open(f"l{k}.txt", "w").write(body + Q)
    print(f"l{k}.txt", len(body + Q), "chars")
