#!/usr/bin/env python3
"""aff-fork-patch3.py — affinity fork patch 3 (run inside ~/src/affinity-fork on amdnas): AFF_MOVE_VERIFY.
Probe 8 (2026-09-22 00:15): the poison reproduces on the fully launch-checked build with NO abort and a clean load-time
slab verify — so at run time either (a) the bytes in a slab slot change underneath the table (SDMA copy landed wrong,
or the kernel driver evicted/relocated a buffer near-full VRAM), or (b) the mover's bookkeeping maps two experts to one
slot. This adds AFF_MOVE_VERIFY=1: in engine_retire, when a PROMOTION has landed and BEFORE its pool slot is unbound
(both copies valid at that instant), read the slab slot back on every card and memcmp it against the pool source; log every
mismatch with expert/slot ids and keep a running count. Also asserts on publish that no other resident expert already
owns the slot being published (owner map) — that is (b) directly. Diagnostic; synchronous readback costs decode speed."""
import sys
MARK = "amdnas-fixes move-verify"
p = "src/gpu/static_placement.hip"; s = open(p).read()
if MARK in s: print("already patched"); sys.exit(0)

old_publish = '''      if (M.promote) {
        device_of_[M.idx] = 0;                        // sharded: "which card" is not a question
        offset_of_[M.idx] = M.slab_slot * slot_stride();
        pool_mut_->bind(l, e, -1);'''
new_publish = '''      if (M.promote) {
        // ---- amdnas-fixes move-verify (AFF_MOVE_VERIFY=1) ------------------------------------------------
        // Both copies of the bytes are valid at this instant (comment above), so this is the one moment a
        // promotion can be checked end to end: read the slab slot back from every card and compare it with the
        // pool shard it was copied from. A mismatch = wrong bytes IN VRAM (copy landed wrong, or something moved
        // the buffer underneath us); a match while the process still poisons = a mapping/bookkeeping race.
        static const bool kMoveVerify = std::getenv("AFF_MOVE_VERIFY") != nullptr;
        if (kMoveVerify) {
          const uint64_t ss_v = slot_stride();
          const uint8_t* src_v = pool_mut_->slot_base(M.pool_slot);
          static std::vector<uint8_t> rb;
          if (rb.size() < ss_v) rb.resize(ss_v);
          for (size_t dv = 0; dv < R && src_v; ++dv) {
            AFF_HIP_CHECK(hipSetDevice(devs_[dv]));
            AFF_HIP_CHECK(hipMemcpy(rb.data(), I.slab[dv] + M.slab_slot * ss_v, ss_v, hipMemcpyDeviceToHost));
            ++mv_verified_;
            if (std::memcmp(rb.data(), src_v + (size_t)dv * ss_v, ss_v) != 0) {
              ++mv_bad_;
              size_t first = 0; while (first < ss_v && rb[first] == src_v[(size_t)dv * ss_v + first]) ++first;
              aff::ui::err("MOVE-VERIFY MISMATCH #%llu: layer %u expert %u -> slab slot %llu card %zu, first bad byte at %zu of %llu "
                           "(verified %llu so far)\\n", (unsigned long long)mv_bad_, l, e, (unsigned long long)M.slab_slot, dv,
                           first, (unsigned long long)ss_v, (unsigned long long)mv_verified_);
            }
          }
        }
        // owner map: no OTHER resident expert may already point at this slot (double allocation = permanent poison)
        {
          const uint64_t off_new = M.slab_slot * slot_stride();
          for (size_t j = 0; j < device_of_.size(); ++j)
            if (j != M.idx && device_of_[j] >= 0 && offset_of_[j] == off_new)
              aff::ui::err("MOVE-VERIFY DOUBLE-MAP: slab slot %llu already owned by layer %u expert %u while publishing "
                           "layer %u expert %u\\n", (unsigned long long)M.slab_slot, (unsigned)(j / n_expert_),
                           (unsigned)(j % n_expert_), l, e);
        }
        device_of_[M.idx] = 0;                        // sharded: "which card" is not a question
        offset_of_[M.idx] = M.slab_slot * slot_stride();
        pool_mut_->bind(l, e, -1);'''
n = s.count(old_publish)
if n != 1: sys.exit(f"ABORT publish anchor {n}x")
s = s.replace(old_publish, new_publish)

# counters: add two members next to estats_ usage — declare in the .hip as statics on the class is invasive; use file-scope
old_inc = "bool StaticPlacement::engine_issue(uint32_t idx, bool promote, void* const* streams) {"
new_inc = ("// amdnas-fixes move-verify counters (file scope; one engine per process)\n"
           "static unsigned long long mv_verified_ = 0, mv_bad_ = 0;\n"
           "bool StaticPlacement::engine_issue(uint32_t idx, bool promote, void* const* streams) {")
if s.count(old_inc) != 1: sys.exit("ABORT engine_issue anchor")
s = s.replace(old_inc, new_inc)

# report at exit alongside the engine report
old_rep = 'void StaticPlacement::engine_report() const {'
new_rep = ('void StaticPlacement::engine_report() const {\n'
           '  if (std::getenv("AFF_MOVE_VERIFY"))\n'
           '    aff::ui::err("move-verify: %llu promoted shards read back, %llu mismatched\\n", mv_verified_, mv_bad_);')
if s.count(old_rep) != 1: sys.exit("ABORT engine_report anchor")
s = s.replace(old_rep, new_rep)
if "#include <cstring>" not in s: s = s.replace("#include <array>", "#include <array>\n#include <cstring>\n#include <cstdlib>", 1)
s += f"\n// {MARK}: AFF_MOVE_VERIFY readback + owner-map checks in engine_retire (2026-09-22).\n"
open(p, "w").write(s); print("patch 3 applied")
