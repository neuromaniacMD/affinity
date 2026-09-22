#!/usr/bin/env python3
"""aff-fork-patch10.py — affinity fork patch 10 (run inside ~/src/affinity-fork on amdnas): huge pages BEFORE the pool is registered.
Probes 16–18 (2026-09-22): the poison is a GPU read of a HOST-POOL page whose GPU page-table entry is invalid or stale — at
62 MiB free VRAM it is a GCVM permission fault (both cards, one address, inside the 51 GiB pool mapping), at 122 MiB it is
silent permanent garbage, at ≥162 MiB it never happens. The pool is mmap'ed with MAP_POPULATE and madvise(MADV_HUGEPAGE)d
only afterwards (host_pool.cpp:213), so it is registered on 4 KiB pages ("0 % on 2 MiB pages"): the GPU mapping of 51 GiB
then needs ≈ 8 B/page ≈ 100 MiB of page tables per card in VRAM, taken AFTER the slab sizing, and khugepaged keeps
collapsing pool pages during the run, each collapse invalidating GPU PTEs that the driver must rebuild in VRAM it no
longer has. This patch (default ON, AFF_POOL_THP=0 restores upstream behaviour) maps the pool WITHOUT MAP_POPULATE, applies
MADV_HUGEPAGE first, then populates (MADV_POPULATE_WRITE, else one touch per 2 MiB) before the pread pass, so the pool is
registered on 2 MiB pages: ~0.2 MiB of page tables per card and nothing left for khugepaged to collapse. The server already
prints the achieved coverage ("N% on 2 MiB pages"). Load-time cost is direct compaction on a fragmented box; on amdnas the
probes drop caches first and 100+ GB is free, so it should be small — the log line says what it was."""
import sys
MARK = "amdnas-fixes thp-first"
p = "src/engine/host_pool.cpp"; s = open(p).read()
if MARK in s: print("already patched"); sys.exit(0)
old = '''  void* p = ::mmap(nullptr, alloc_bytes_, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
'''
new = '''  // amdnas-fixes thp-first: populate AFTER the hugepage hint so the pool is registered with the GPU on 2 MiB pages.
  // A 4 KiB-page mapping of tens of GiB costs the card ~8 B/page of VRAM page tables (taken after the slab was sized) and
  // khugepaged then collapses the pages under a live GPU mapping; both ended in stale/invalid GPU PTEs = garbage or a
  // permission fault from the expert GEMM. AFF_POOL_THP=0 restores the upstream order.
  const char* thp_env = std::getenv("AFF_POOL_THP");
  const bool thp_first = !(thp_env && thp_env[0] == '0');
  void* p = ::mmap(nullptr, alloc_bytes_, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | (thp_first ? 0 : MAP_POPULATE), -1, 0);
'''
n = s.count(old)
if n != 1: sys.exit(f"ABORT mmap anchor {n}x")
s = s.replace(old, new)
old = '''  (void)::madvise(base_, alloc_bytes_, MADV_HUGEPAGE);

  const auto t0 = std::chrono::steady_clock::now();
'''
new = '''  (void)::madvise(base_, alloc_bytes_, MADV_HUGEPAGE);
  if (thp_first) {
    const auto tp0 = std::chrono::steady_clock::now();
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif
    if (::madvise(base_, alloc_bytes_, MADV_POPULATE_WRITE) != 0) {
      for (uint64_t o = 0; o < alloc_bytes_; o += (2u << 20)) base_[o] = 0;   // one touch a 2 MiB page
    }
    aff::ui::err("host pool: hugepage-first populate of %.1f GiB in %.1f s (amdnas-fixes thp-first; AFF_POOL_THP=0 to disable)\\n",
                 (double)alloc_bytes_ / (1 << 30),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count());
  }

  const auto t0 = std::chrono::steady_clock::now();
'''
n = s.count(old)
if n != 1: sys.exit(f"ABORT madvise anchor {n}x")
s = s.replace(old, new)
for inc in ("#include <cstdlib>",):
    if inc not in s: s = s.replace("#include <sys/mman.h>", "#include <sys/mman.h>\n" + inc, 1)
if "ui/log.h" not in s: s = s.replace("#include <sys/mman.h>", '#include <sys/mman.h>\n#include "ui/log.h"', 1)
open(p, "w").write(s); print("patch 10 applied")
