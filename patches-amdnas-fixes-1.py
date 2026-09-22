#!/usr/bin/env python3
"""aff-fork-patch1.py — apply patch 1 to the affinity fork checkout (run inside ~/src/affinity-fork on amdnas).
(1) src/gpu/sampler_gpu.hip: the inverse-CDF draw sums keep_total STRIDED but the per-thread slices CONTIGUOUSLY, so with a
    flat distribution `want` can exceed every slice sum; the walk then falls to the LAST slice (top of the vocabulary = the
    special-token region, nothing kept), `last` stays 0xFFFFFFFF and the kernel returns an out-of-range token id.
    Fix: fall back to the highest kept id anywhere in the block.
(2) tools/affinity.cpp: in spec_decode, log loudly and clamp any sampled draw >= vocab so an out-of-range id can never reach
    the embedding lookup / tokenizer / KV. Diagnostic + hardening; whether this is THE amdnas poison is still open (23:35).
Idempotent: refuses if the markers are already present."""
import sys
def patch(path, old, new, tag):
    s = open(path).read()
    if tag in s: print(f"{path}: already patched"); return
    n = s.count(old)
    if n != 1: sys.exit(f"ABORT {path}: anchor occurs {n}x")
    open(path, "w").write(s.replace(old, new)); print(f"{path}: patched")

patch("src/gpu/sampler_gpu.hip",
'''    float mine = 0.0f;
    for (uint32_t v = lo; v < hi; ++v) {
      const float p = prob(v);
      if (!keep(p) || v == excl) continue;
      mine += p;
    }
    sh.red[tid] = mine;
    __syncthreads();''',
'''    float mine = 0.0f;
    // amdnas-fixes 2026-09-21: remember the highest kept id in THIS slice so a drift miss below can fall back to the
    // highest kept id in the whole block rather than to 0xFFFFFFFF (see the fallback after the slice walk).
    int32_t my_last = -1;
    for (uint32_t v = lo; v < hi; ++v) {
      const float p = prob(v);
      if (!keep(p) || v == excl) continue;
      mine += p;
      my_last = (int32_t)v;
    }
    sh.red[tid] = mine;
    __syncthreads();''', "amdnas-fixes 2026-09-21: remember")

patch("src/gpu/sampler_gpu.hip",
'''      if (sh.pick == 0xFFFFFFFFu) sh.pick = last;
    }
    __syncthreads();
    const uint32_t r = sh.pick;
    __syncthreads();
    return r;''',
'''      if (sh.pick == 0xFFFFFFFFu) sh.pick = last;
    }
    __syncthreads();
    // amdnas-fixes 2026-09-21 (BUG): `keep_total` is a STRIDED sum and the slices are CONTIGUOUS sums, so with a flat
    // distribution `want` can exceed the sum of every slice. The walk above then falls to slice kThr-1 — the top of the
    // vocabulary, the special-token region that top_p filters out — which holds no kept token, `last` stays 0xFFFFFFFF
    // and the kernel returned an OUT-OF-RANGE token id. Fall back to the highest kept id anywhere in the block.
    if (sh.pick == 0xFFFFFFFFu) {
      sh.red_i[tid] = (uint32_t)(my_last + 1);      // 0 = none, else id+1, so a plain max reduction works
      __syncthreads();
      for (uint32_t k = kThr / 2; k; k >>= 1) {
        if (tid < k) sh.red_i[tid] = max(sh.red_i[tid], sh.red_i[tid + k]);
        __syncthreads();
      }
      if (tid == 0) sh.pick = sh.red_i[0] ? sh.red_i[0] - 1u : 0u;
      __syncthreads();
    }
    const uint32_t r = sh.pick;
    __syncthreads();
    return r;''', "amdnas-fixes 2026-09-21 (BUG)")

patch("tools/affinity.cpp",
'''        aff::ui::fatal("fatal: the verify block refused %u tokens at %u\\n", nd + 1, P);
        std::abort();
      }''',
'''        aff::ui::fatal("fatal: the verify block refused %u tokens at %u\\n", nd + 1, P);
        std::abort();
      }
      // amdnas-fixes 2026-09-21: a sampled id outside the vocabulary must never reach the embedding lookup, the
      // tokenizer or the KV (the sampler kernel's drift fallback could return 0xFFFFFFFF — see sampler_gpu.hip).
      // Log loudly and clamp to the proposal (or 0) so the stream stays in range.
      if (sc) {
        for (uint32_t j = 0; j <= nd && j < vb.draws.size(); ++j) {
          Model::PosDraw& d = vb.draws[j];
          if (d.tok >= c.vocab || d.tok_excl >= c.vocab) {
            aff::ui::err("SAMPLER OOB: slot %u P=%u j=%u tok=%u tok_excl=%u p_query=%.4f (vocab %u, temp %.2f top_p %.2f)\\n",
                         slot, P, j, d.tok, d.tok_excl, d.p_query, (unsigned)c.vocab, vb.temperature, vb.top_p);
            const uint32_t safe = (j < nd && fed[j + 1] < c.vocab) ? fed[j + 1] : 0u;
            if (d.tok >= c.vocab) d.tok = safe;
            if (d.tok_excl >= c.vocab) d.tok_excl = safe;
          }
        }
      }''', "SAMPLER OOB")
print("patch 1 done")
