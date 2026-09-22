#!/usr/bin/env python3
"""aff-fork-patch6.py — affinity fork patch 6 (run inside ~/src/affinity-fork on amdnas): verify the DEVICE placement table.
Probe 13 (2026-09-22 00:56): expert bytes verified on every promotion AND demotion, no double-maps, launches all checked, GTT
flat, history rings cleared per request — and the process still poisons for life at default headroom, while static placement
or 2 GiB headroom never do. Every check so far compared HOST structures (device_of_/offset_of_/pool bindings) and expert
BYTES. The one placement fact the cards actually consume is the device table `plc_[dv]` (static_placement.h:344), and it is
written on a different path from `plc_host_`: host words by a plain store, device words by `plc_publish_kernel` launched on
the compute stream from a PlcPatch (16 entries, flushed when full — static_placement.hip:2371). If ANY device word disagrees
with the host word for the life of the process, a kernel-side tile build reads an expert from the wrong slot forever —
exactly the symptom, invisible to every host-side check, and only reachable when the slab is full enough that demotions
and promotions recycle slots (which 2 GiB headroom avoids and static never does).
This patch adds `StaticPlacement::verify_plc(tag)`: reads the fine-grained device table from the host (no sync, no copy —
the same host-visible memory publish_base stores into at load), compares every word with `plc_host_`, and for any
mismatch settles the card (hipDeviceSynchronize, so a merely-queued patch is not a false positive) and re-reads; persistent
disagreements are logged as PLC-MISMATCH with layer/expert and both words decoded (slab slot / pool / ssd). Gated by
AFF_PLC_VERIFY=N (check every N engine ticks) and reported from engine_report."""
import sys
MARK = "amdnas-fixes plc-verify"
h = "src/gpu/static_placement.h"; s = open(h).read()
if MARK in s: print("already patched"); sys.exit(0)
old = "  void engine_report() const;\n"
new = '''  void engine_report() const;
  // amdnas-fixes plc-verify: compare the device placement table with plc_host_ (AFF_PLC_VERIFY=N ticks).
  void verify_plc(const char* tag) const;
  mutable uint64_t plc_checks_ = 0, plc_bad_ = 0;
'''
n = s.count(old)
if n != 1: sys.exit(f"ABORT header anchor {n}x")
s = s.replace(old, new); open(h, "w").write(s)

p = "src/gpu/static_placement.hip"; s = open(p).read()
old = '''  // Whatever landed since the last layer, applied HERE and by this thread: the transfers are async,
  // the table edits are not.
  engine_retire(streams);
'''
new = '''  // Whatever landed since the last layer, applied HERE and by this thread: the transfers are async,
  // the table edits are not.
  engine_retire(streams);
  // amdnas-fixes plc-verify: every AFF_PLC_VERIFY ticks, read the device table back and compare it with the host words.
  {
    static const long kPlcVerifyEvery = std::getenv("AFF_PLC_VERIFY") ? std::atol(std::getenv("AFF_PLC_VERIFY")) : 0;
    if (kPlcVerifyEvery > 0 && (estats_.ticks % (uint64_t)kPlcVerifyEvery) == 0) verify_plc("tick");
  }
'''
n = s.count(old)
if n != 1: sys.exit(f"ABORT engine_tick anchor {n}x")
s = s.replace(old, new)
old = '''void StaticPlacement::engine_report() const {
  if (std::getenv("AFF_MOVE_VERIFY"))
    aff::ui::err("move-verify: %llu promoted shards read back, %llu mismatched\\n", mv_verified_, mv_bad_);
'''
new = '''void StaticPlacement::engine_report() const {
  if (std::getenv("AFF_MOVE_VERIFY"))
    aff::ui::err("move-verify: %llu promoted shards read back, %llu mismatched\\n", mv_verified_, mv_bad_);
  if (std::getenv("AFF_PLC_VERIFY")) {
    verify_plc("report");
    aff::ui::err("plc-verify: %llu table checks, %llu persistent device/host mismatches\\n",
                 (unsigned long long)plc_checks_, (unsigned long long)plc_bad_);
  }
'''
n = s.count(old)
if n != 1: sys.exit(f"ABORT engine_report anchor {n}x")
s = s.replace(old, new)
s += '''
// ---- amdnas-fixes plc-verify -----------------------------------------------------------------------------------------
// The cards consume `plc_[dv]`; every other check in this fork compared host bookkeeping and expert bytes. This reads the
// fine-grained device table from the host (the same host-visible memory publish_base stores into at load time — no sync,
// no staging), compares it word for word with `plc_host_`, and treats a disagreement as real only if it survives a
// device sync (a PlcPatch queued on the compute stream and not yet run is stale by design, not wrong).
void aff::StaticPlacement::verify_plc(const char* tag) const {
  if (plc_.empty() || plc_host_.empty() || !impl_) return;
  const size_t n = (size_t)n_layer_ * n_expert_;
  const uint64_t ss = slot_stride();
  for (size_t dv = 0; dv < plc_.size(); ++dv) {
    if (!plc_[dv]) continue;
    volatile const uint64_t* dtab = plc_[dv];
    const uint64_t* htab = plc_host_.data() + dv * n;
    std::vector<size_t> bad;
    for (size_t i = 0; i < n; ++i)
      if (dtab[i] != htab[i]) bad.push_back(i);
    ++plc_checks_;
    if (bad.empty()) continue;
    (void)hipSetDevice(devs_[dv]);
    (void)hipDeviceSynchronize();
    auto decode = [&](uint64_t v, char* out, size_t cap) {
      if (v == 0) { snprintf(out, cap, "ssd(0)"); return; }
      const uint64_t a = v & kPlcAddrMask;
      if (v & kPlcResident) {
        const int64_t off = (int64_t)((uintptr_t)a - (uintptr_t)impl_->slab[dv]);
        snprintf(out, cap, "slab+%lld(slot %lld%s)", (long long)off, (long long)(ss ? off / (int64_t)ss : -1),
                 (ss && (off % (int64_t)ss)) ? " UNALIGNED" : "");
      } else {
        snprintf(out, cap, "pool 0x%llx", (unsigned long long)a);
      }
    };
    size_t persistent = 0;
    for (size_t i : bad) {
      const uint64_t d = dtab[i], hw = htab[i];
      if (d == hw) continue;
      ++persistent; ++plc_bad_;
      if (persistent <= 12) {
        char sd[96], sh[96];
        decode(d, sd, sizeof sd); decode(hw, sh, sizeof sh);
        aff::ui::err("PLC-MISMATCH [%s] card %zu layer %u expert %u: device=%s host=%s (0x%llx vs 0x%llx)\\n", tag, dv,
                     (unsigned)(i / n_expert_), (unsigned)(i % n_expert_), sd, sh, (unsigned long long)d,
                     (unsigned long long)hw);
      }
    }
    if (persistent)
      aff::ui::err("PLC-MISMATCH [%s] card %zu: %zu of %zu entries differ after a device sync (check #%llu)\\n", tag, dv,
                   persistent, n, (unsigned long long)plc_checks_);
  }
}
'''
open(p, "w").write(s); print("patch 6 applied")
