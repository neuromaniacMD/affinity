#!/usr/bin/env python3
"""aff-fork-patch4.py — affinity fork patch 4 (run inside ~/src/affinity-fork on amdnas): verify DEMOTIONS too.
Probe 11 (2026-09-22 00:24): with AFF_MOVE_VERIFY every promotion landed byte-exact and no slab slot was double-mapped,
yet the process still poisoned. Promotions are clean at publish time. Remaining write paths in the mover: the DEMOTION D2H
copy (slab → pool slot; once the slab slot is freed the pool copy is the expert's ONLY copy, so a bad copy is permanent),
and pool-slot double allocation. This extends AFF_MOVE_VERIFY: at a demotion's publish (slab bytes still valid), read the
slab slot back and memcmp it against the freshly written pool slot; and assert no other pooled expert is bound to that
pool slot. Same env var, same counters, distinct message prefixes (DEMOTE-MISMATCH / POOL-DOUBLE-MAP)."""
import sys
MARK = "amdnas-fixes demote-verify"
p = "src/gpu/static_placement.hip"; s = open(p).read()
if MARK in s: print("already patched"); sys.exit(0)
old = '''      } else {
        pool_mut_->bind(l, e, M.pool_slot);
        device_of_[M.idx] = -1;
        heat_.set_tier(M.idx, ExpertTier::Pool);'''
new = '''      } else {
        // ---- amdnas-fixes demote-verify (AFF_MOVE_VERIFY=1) --------------------------------------------------
        // The slab slot's bytes are still valid here (it is freed only after quarantine), and the pool slot has just
        // received the D2H copy: compare them. A mismatch means the expert's soon-to-be-ONLY copy is wrong = permanent.
        static const bool kMoveVerifyD = std::getenv("AFF_MOVE_VERIFY") != nullptr;
        if (kMoveVerifyD) {
          const uint64_t ss_v = slot_stride();
          const uint8_t* dst_v = pool_mut_->slot_base(M.pool_slot);
          static std::vector<uint8_t> rb2;
          if (rb2.size() < ss_v) rb2.resize(ss_v);
          for (size_t dv = 0; dv < R && dst_v; ++dv) {
            AFF_HIP_CHECK(hipSetDevice(devs_[dv]));
            AFF_HIP_CHECK(hipMemcpy(rb2.data(), I.slab[dv] + M.slab_slot * ss_v, ss_v, hipMemcpyDeviceToHost));
            ++mv_verified_;
            if (std::memcmp(rb2.data(), dst_v + (size_t)dv * ss_v, ss_v) != 0) {
              ++mv_bad_;
              size_t first = 0; while (first < ss_v && rb2[first] == dst_v[(size_t)dv * ss_v + first]) ++first;
              aff::ui::err("DEMOTE-MISMATCH #%llu: layer %u expert %u slab slot %llu -> pool slot %lld card %zu, first bad byte %zu of %llu\\n",
                           (unsigned long long)mv_bad_, l, e, (unsigned long long)M.slab_slot, (long long)M.pool_slot, dv, first,
                           (unsigned long long)ss_v);
            }
          }
          // pool owner map: no other expert may currently be bound to this pool slot
          for (uint32_t ll = 0; ll < n_layer_; ++ll)
            for (uint32_t ee = 0; ee < n_expert_; ++ee)
              if (!(ll == l && ee == e) && pool_mut_->slot_of(ll, ee) == M.pool_slot)
                aff::ui::err("POOL-DOUBLE-MAP: pool slot %lld already bound to layer %u expert %u while demoting layer %u expert %u\\n",
                             (long long)M.pool_slot, ll, ee, l, e);
        }
        pool_mut_->bind(l, e, M.pool_slot);
        device_of_[M.idx] = -1;
        heat_.set_tier(M.idx, ExpertTier::Pool);'''
n = s.count(old)
if n != 1: sys.exit(f"ABORT demote anchor {n}x")
s = s.replace(old, new)
s += f"\n// {MARK}: AFF_MOVE_VERIFY also checks demotion D2H copies and pool-slot ownership (2026-09-22).\n"
open(p, "w").write(s); print("patch 4 applied")
