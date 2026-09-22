#!/usr/bin/env python3
"""aff-fork-patch9.py — affinity fork patch 9 (run inside ~/src/affinity-fork on amdnas): keep the slab's reserve honest.
Probe 16 (2026-09-22 01:37, --gpu-headroom-mib sweep): free VRAM after the slabs decides everything — 62 MiB → GPU memory
fault in expert_gemm_wmma_kernel, 122 MiB (the DEFAULT) → silent permanent garbage after the first long request, ≥162 MiB →
clean. static_placement.hip:868 keeps 0.8 % of the card (262 MiB on a 32 GiB R9700) as "allocator slack … the engine's own
allocations after this point, which are small" — but the slab is then allocated as (per_gpu + shadow_slots_) slots + 4096,
i.e. the 32 shadow slots (32 × 4.32 MiB = 138 MiB) AND the codebook are taken on top of the budgeted slots, out of that
reserve, leaving 122 MiB where 262 were intended. This patch subtracts the engine's own post-sizing allocations (shadow
slots, codebook, pad) from the capacity before dividing by the slot stride, so the reserve the author measured is the reserve
the card actually keeps. Default flags then leave ~262 MiB free (= the always-clean static arm's figure); an explicit
--gpu-headroom-mib now means what it says. Costs 32 resident experts of ~4,953 per card (0.6 %)."""
import sys
MARK = "amdnas-fixes honest-reserve"
p = "src/gpu/static_placement.hip"; s = open(p).read()
if MARK in s: print("already patched"); sys.exit(0)
old = '''    const uint64_t cap = std::min<uint64_t>(want, freeb > keep ? freeb - keep : 0);
    per_gpu[i] = cap / slot_stride;
'''
new = '''    // amdnas-fixes honest-reserve: the slab below is allocated as (per_gpu + shadow_slots_) slots + 4096, and the codebook
    // follows it — all taken AFTER this sizing, out of `keep`. On a 32 GiB card that turned the intended 262 MiB into
    // 122 MiB, which is inside the band where the runtime silently computes garbage (probe 16). Reserve them here.
    const uint64_t engine_after = (uint64_t)shadow_slots_ * slot_stride + (uint64_t)cb_.v.size() * 4 + 4096;
    const uint64_t avail = freeb > keep + engine_after ? freeb - keep - engine_after : 0;
    const uint64_t cap = std::min<uint64_t>(want, avail);
    per_gpu[i] = cap / slot_stride;
    aff::ui::err("placement sizing: card %zu free %.0f MiB, reserve %.0f MiB + engine-after %.0f MiB (shadow %u slots, "
                 "codebook %.1f MiB) -> %llu slots\\n", i, (double)freeb / 1048576.0, (double)keep / 1048576.0,
                 (double)engine_after / 1048576.0, shadow_slots_, (double)cb_.v.size() * 4 / 1048576.0,
                 (unsigned long long)per_gpu[i]);
'''
n = s.count(old)
if n != 1: sys.exit(f"ABORT sizing anchor {n}x")
s = s.replace(old, new)
open(p, "w").write(s); print("patch 9 applied")
