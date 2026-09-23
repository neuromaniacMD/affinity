#!/usr/bin/env python3
"""The reference side of enc_check.cpp: the same conversations through DeepSeek-V4.1's encoding.py."""
import importlib.util, sys
p = sys.argv[1] if len(sys.argv) > 1 else "/path/to/hf/DeepSeek-V4.1-Flash/encoding/encoding.py"
s = importlib.util.spec_from_file_location("e41", p); e = importlib.util.module_from_spec(s); s.loader.exec_module(e)
u = {"role": "user", "content": "What is the capital of France?"}
sy = {"role": "system", "content": "You are terse."}
a = {"role": "assistant", "content": "Paris.", "reasoning_content": "Easy."}
u2 = {"role": "user", "content": "And Spain?"}
for msgs, mode, eff in [([u], "chat", 100), ([u], "thinking", 100), ([u], "thinking", 75), ([sy, u], "chat", 100),
                        ([sy, u], "thinking", 50), ([u, a, u2], "thinking", 100), ([u, a, sy, u2], "chat", 100)]:
    print(e.encode_messages(msgs, thinking_mode=mode, reasoning_effort=eff).replace("\n", "\\n"))
