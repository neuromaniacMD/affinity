#!/usr/bin/env python3
"""aff-fork-patch8.py — affinity fork patch 8 (run inside ~/src/affinity-fork on amdnas): log every failed device allocation.
Probe 15 (2026-09-22 01:17): dense weights, codebook, slab pad (AFF_WEIGHT_VERIFY) and the device placement table
(AFF_PLC_VERIFY) are all byte-identical after the poisoning request, expert copies and mappings were already proven exact,
and the process still emits garbage for life — but only when the slab leaves ~122 MiB free (static: 262 MiB, headroom
2048: 1908 MiB — both always clean). The one class of failure that is (a) gated on free VRAM at that scale, (b) silent
(no kernel launch fails), and (c) permanent, is a RUNTIME hipMalloc that fails inside one of dense_gpu.hip's
"capacity-guarded, idempotent" growth helpers (d_qrn_cap, c_cap_tok, q_cap, c_py_cap, tap_bytes_s, smp_cap_ …): a first
long request asks for a bigger buffer than any short one, the allocation fails at 122 MiB free, and if the helper records
the new capacity (or a null/stale pointer) anyway, every later request believes the buffer exists. This patch wraps
hipMalloc through hip_common.h (included by every allocating .hip): a failure logs HIPMALLOC-FAIL <file:line> with the
size and the card's free VRAM; AFF_MALLOC_TRACE=1 additionally logs every successful allocation >= 1 MiB, so the runtime
allocations between two requests are visible by file:line. Behaviour is otherwise unchanged (same return value, the
sticky HIP error is left as the runtime set it)."""
import sys
MARK = "amdnas-fixes malloc-trace"
p = "src/gpu/hip_common.h"; s = open(p).read()
if MARK in s: print("already patched"); sys.exit(0)
s = s.rstrip("\n") + '''

// ---- amdnas-fixes malloc-trace --------------------------------------------------------------------------------------
// Every device allocation in src/gpu goes through here (this header is included by all of them, and the macro below
// rewrites the call sites). A failed hipMalloc is the one silent, VRAM-gated, permanent failure mode left after probes
// 8-15; it is logged with its site. AFF_MALLOC_TRACE=1 also logs successes >= 1 MiB.
inline hipError_t aff_hipMalloc_dbg(void** p, size_t n, const char* file, int line) {
  const hipError_t e = ::hipMalloc(p, n);
  static const bool trace = std::getenv("AFF_MALLOC_TRACE") != nullptr;
  int dev = -1; (void)hipGetDevice(&dev);
  if (e != hipSuccess) {
    size_t fb = 0, tb = 0; (void)hipMemGetInfo(&fb, &tb);
    aff::ui::err("HIPMALLOC-FAIL %s:%d dev %d bytes %zu (%.1f MiB) free %.1f MiB: %s\\n", file, line, dev, n,
                 (double)n / 1048576.0, (double)fb / 1048576.0, hipGetErrorString(e));
  } else if (trace && n >= (1u << 20)) {
    aff::ui::err("hipmalloc %s:%d dev %d %.1f MiB -> %p\\n", file, line, dev, (double)n / 1048576.0, *p);
  }
  return e;
}
#define hipMalloc(p, n) aff_hipMalloc_dbg((void**)(p), (n), __FILE__, __LINE__)
'''
open(p, "w").write(s); print("patch 8 applied")
