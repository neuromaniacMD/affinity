#!/usr/bin/env python3
"""aff-fork-patch7.py — affinity fork patch 7 (run inside ~/src/affinity-fork on amdnas): checksum the read-only device state.
Probe 14 (2026-09-22 01:08): the DEVICE placement table never diverges from the host words (AFF_PLC_VERIFY, 0 mismatches)
and the process still poisons for life after one 3.8K-token request at default headroom. Together with probes 11–13 that
leaves the READ-ONLY device weights: the dense FP8 shards (attention, shared expert, norms, heads), the expert CODEBOOK
(static_placement.hip:960 — allocated IMMEDIATELY AFTER the expert slab, whose only guard is a 4096-byte pad), and the pad
itself. A stray write past the end of a ~full slab (a shadow-slot index one past the end, a demotion copy sized wrong, …)
lands in the codebook = every routed expert decodes to garbage = exactly the symptom; at 2 GiB headroom the same write
lands in free VRAM or the heap places the codebook elsewhere, and static has no shadow slots and no moves.
This patch adds AFF_WEIGHT_VERIFY=1: before every chat request the server hashes (a) every dense shard's data and scales,
(b) each card's codebook, (c) the last 4096 bytes of each card's slab (never legitimately written), remembers the first
result as the baseline, and logs WEIGHT-MISMATCH <name> lines when any hash changes. Also: verify_plc now logs its first
check so a clean run carries positive evidence that the check ran."""
import sys
MARK = "amdnas-fixes weight-verify"

# ---- dense_gpu.h: byte sizes on the shard + digest method -------------------------------------------------------------
h = "src/gpu/dense_gpu.h"; s = open(h).read()
if MARK in s: print("already patched"); sys.exit(0)
old = "    bool     ps = false;\n  };\n"
new = '''    bool     ps = false;
    uint64_t data_bytes = 0, scales_bytes = 0;   // amdnas-fixes weight-verify: sizes for the read-back digest
  };
'''
n = s.count(old)
if n != 1: sys.exit(f"ABORT Shard anchor {n}x")
s = s.replace(old, new)
old = "  bool init(uint64_t max_rows, uint64_t max_cols, std::string* err, size_t want_ranks = 0);\n"
new = '''  bool init(uint64_t max_rows, uint64_t max_cols, std::string* err, size_t want_ranks = 0);
  // amdnas-fixes weight-verify: hash every uploaded shard (data + scales) on every rank. Appends
  // ("ent<i> rank<d> data|scales", hash) pairs; returns the bytes hashed.
  uint64_t digest_weights(std::vector<std::pair<std::string, uint64_t>>* out) const;
'''
n = s.count(old)
if n != 1: sys.exit(f"ABORT DenseGpu::init anchor {n}x")
s = s.replace(old, new)
if "#include <utility>" not in s: s = s.replace("#include <vector>", "#include <vector>\n#include <utility>", 1)
open(h, "w").write(s)

# ---- dense_gpu.hip: record sizes at upload + digest definition ----------------------------------------------------------
p = "src/gpu/dense_gpu.hip"; s = open(p).read()
old = "    if (hipMalloc(&s.data, nbytes) != hipSuccess) return -1;\n"
new = "    if (hipMalloc(&s.data, nbytes) != hipSuccess) return -1;\n    s.data_bytes = nbytes;   // amdnas-fixes weight-verify\n"
n = s.count(old)
if n != 1: sys.exit(f"ABORT data hipMalloc anchor {n}x")
s = s.replace(old, new)
old = "      if (hipMalloc((void**)&s.scales, sc_bytes) != hipSuccess) return -1;\n"
new = "      if (hipMalloc((void**)&s.scales, sc_bytes) != hipSuccess) return -1;\n      s.scales_bytes = sc_bytes;   // amdnas-fixes weight-verify\n"
n = s.count(old)
if n != 1: sys.exit(f"ABORT scales hipMalloc anchor {n}x")
s = s.replace(old, new)
s += '''
// ---- amdnas-fixes weight-verify -------------------------------------------------------------------------------------
namespace {
uint64_t aff_wv_hash(const uint8_t* p, size_t n, uint64_t h) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    uint64_t w; std::memcpy(&w, p + i, 8);
    h ^= w; h *= 0x9E3779B97F4A7C15ull; h ^= h >> 29;
  }
  for (; i < n; ++i) { h ^= p[i]; h *= 0x100000001B3ull; }
  return h;
}
uint64_t aff_wv_digest_dev(const void* dptr, uint64_t bytes, std::vector<uint8_t>& stage) {
  uint64_t h = 0xCBF29CE484222325ull;
  const size_t chunk = 64u << 20;
  if (stage.size() < chunk) stage.resize(chunk);
  for (uint64_t off = 0; off < bytes; off += chunk) {
    const size_t n = (size_t)std::min<uint64_t>(chunk, bytes - off);
    if (hipMemcpy(stage.data(), (const uint8_t*)dptr + off, n, hipMemcpyDeviceToHost) != hipSuccess) return 0xDEADull;
    h = aff_wv_hash(stage.data(), n, h);
  }
  return h;
}
}  // namespace

uint64_t aff::DenseGpu::digest_weights(std::vector<std::pair<std::string, uint64_t>>* out) const {
  if (!impl_) return 0;
  int prev = 0; (void)hipGetDevice(&prev);
  static std::vector<uint8_t> stage;
  uint64_t total = 0;
  for (size_t i = 0; i < ent_.size(); ++i) {
    const Entry& e = ent_[i];
    for (size_t d = 0; d < impl_->dev.size() && d < kMaxRanks; ++d) {
      const Shard& sh = e.sh[d];
      if (!sh.data || !sh.data_bytes) continue;
      (void)hipSetDevice(impl_->dev[d].id);
      out->emplace_back("ent" + std::to_string(i) + " rank" + std::to_string(d) + " data",
                        aff_wv_digest_dev(sh.data, sh.data_bytes, stage));
      total += sh.data_bytes;
      if (sh.scales && sh.scales_bytes) {
        out->emplace_back("ent" + std::to_string(i) + " rank" + std::to_string(d) + " scales",
                          aff_wv_digest_dev(sh.scales, sh.scales_bytes, stage));
        total += sh.scales_bytes;
      }
    }
  }
  (void)hipSetDevice(prev);
  return total;
}
'''
open(p, "w").write(s)

# ---- static_placement.h / .hip: codebook + slab tail digest, first-check log for verify_plc ---------------------------
h = "src/gpu/static_placement.h"; s = open(h).read()
old = "  void verify_plc(const char* tag) const;\n"
new = '''  void verify_plc(const char* tag) const;
  // amdnas-fixes weight-verify: hash each card's codebook and the 4096-byte pad past the end of its slab.
  uint64_t digest_invariants(std::vector<std::pair<std::string, uint64_t>>* out) const;
  std::vector<uint64_t> slab_alloc_bytes_;   // per card, what hipMalloc was asked for (slots + pad)
  uint64_t cb_bytes_ = 0;
'''
n = s.count(old)
if n != 1: sys.exit(f"ABORT verify_plc decl anchor {n}x")
s = s.replace(old, new); open(h, "w").write(s)

p = "src/gpu/static_placement.hip"; s = open(p).read()
old = '''    AFF_HIP_CHECK(hipMalloc(&I.slab[i], (per_gpu[i] + shadow_slots_) * slot_stride + 4096));
    AFF_HIP_CHECK(hipMalloc(&I.d_cb[i], cb_bytes));
'''
new = '''    AFF_HIP_CHECK(hipMalloc(&I.slab[i], (per_gpu[i] + shadow_slots_) * slot_stride + 4096));
    AFF_HIP_CHECK(hipMalloc(&I.d_cb[i], cb_bytes));
    slab_alloc_bytes_.push_back((per_gpu[i] + shadow_slots_) * slot_stride + 4096);   // amdnas-fixes weight-verify
    cb_bytes_ = cb_bytes;
'''
n = s.count(old)
if n != 1: sys.exit(f"ABORT slab/codebook alloc anchor {n}x")
s = s.replace(old, new)
old = '''    ++plc_checks_;
    if (bad.empty()) continue;
'''
new = '''    ++plc_checks_;
    if (plc_checks_ <= plc_.size())
      aff::ui::err("plc-verify [%s]: card %zu first check, %zu words, %zu differ before sync\\n", tag, dv, n, bad.size());
    if (bad.empty()) continue;
'''
n = s.count(old)
if n != 1: sys.exit(f"ABORT plc first-check anchor {n}x")
s = s.replace(old, new)
s += '''
// ---- amdnas-fixes weight-verify -------------------------------------------------------------------------------------
namespace {
uint64_t aff_sp_hash(const uint8_t* p, size_t n, uint64_t h) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    uint64_t w; std::memcpy(&w, p + i, 8);
    h ^= w; h *= 0x9E3779B97F4A7C15ull; h ^= h >> 29;
  }
  for (; i < n; ++i) { h ^= p[i]; h *= 0x100000001B3ull; }
  return h;
}
}  // namespace

uint64_t aff::StaticPlacement::digest_invariants(std::vector<std::pair<std::string, uint64_t>>* out) const {
  if (!impl_) return 0;
  int prev = 0; (void)hipGetDevice(&prev);
  std::vector<uint8_t> buf;
  uint64_t total = 0;
  for (size_t dv = 0; dv < devs_.size(); ++dv) {
    (void)hipSetDevice(devs_[dv]);
    if (dv < impl_->d_cb.size() && impl_->d_cb[dv] && cb_bytes_) {
      buf.resize(cb_bytes_);
      uint64_t h = 0xCBF29CE484222325ull;
      if (hipMemcpy(buf.data(), impl_->d_cb[dv], cb_bytes_, hipMemcpyDeviceToHost) == hipSuccess)
        h = aff_sp_hash(buf.data(), cb_bytes_, h);
      else h = 0xDEADull;
      out->emplace_back("codebook card" + std::to_string(dv), h);
      total += cb_bytes_;
    }
    if (dv < impl_->slab.size() && impl_->slab[dv] && dv < slab_alloc_bytes_.size() && slab_alloc_bytes_[dv] >= 4096) {
      buf.resize(4096);
      uint64_t h = 0xCBF29CE484222325ull;
      if (hipMemcpy(buf.data(), impl_->slab[dv] + slab_alloc_bytes_[dv] - 4096, 4096, hipMemcpyDeviceToHost) == hipSuccess)
        h = aff_sp_hash(buf.data(), 4096, h);
      else h = 0xDEADull;
      out->emplace_back("slab-tail-pad card" + std::to_string(dv), h);
      total += 4096;
    }
  }
  (void)hipSetDevice(prev);
  return total;
}
'''
open(p, "w").write(s)

# ---- server: hash before every chat request, compare with the baseline --------------------------------------------------
p = "tools/affinity.cpp"; s = open(p).read()
old = "  const bool ok = srv.start(host, port, [&](const CompletionRequest& req, const TokenSink& sink,"
new = '''  // amdnas-fixes weight-verify (AFF_WEIGHT_VERIFY=1): hash the read-only device state before each chat request and
  // compare it with the first result. A WEIGHT-MISMATCH names what changed; a clean line is the positive evidence.
  std::vector<std::pair<std::string, uint64_t>> wv_base;
  std::mutex wv_mu;
  auto weight_verify = [&](const char* tag) {
    static const bool on = std::getenv("AFF_WEIGHT_VERIFY") != nullptr;
    if (!on) return;
    std::lock_guard<std::mutex> lk(wv_mu);
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::pair<std::string, uint64_t>> cur;
    uint64_t bytes = dense_gpu.digest_weights(&cur);
    bytes += placement.digest_invariants(&cur);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (wv_base.empty()) {
      wv_base = cur;
      aff::ui::err("weight-verify [%s]: baseline %zu regions, %.2f GiB hashed in %.0f ms\\n", tag, cur.size(),
                   (double)bytes / 1073741824.0, ms);
      return;
    }
    size_t bad = 0;
    for (size_t i = 0; i < cur.size() && i < wv_base.size(); ++i)
      if (cur[i].second != wv_base[i].second || cur[i].first != wv_base[i].first) {
        ++bad;
        aff::ui::err("WEIGHT-MISMATCH [%s]: %s hash 0x%016llx was 0x%016llx\\n", tag, cur[i].first.c_str(),
                     (unsigned long long)cur[i].second, (unsigned long long)wv_base[i].second);
      }
    if (cur.size() != wv_base.size()) ++bad;
    aff::ui::err("weight-verify [%s]: %zu regions, %zu changed, %.0f ms\\n", tag, cur.size(), bad, ms);
  };
  const bool ok = srv.start(host, port, [&](const CompletionRequest& req, const TokenSink& sink,'''
n = s.count(old)
if n != 1: sys.exit(f"ABORT srv.start anchor {n}x")
s = s.replace(old, new)
old = '''    // The budget covers the whole generation, so it is also exactly what the state has to hold.
    model.init_state(&st, ids.size() + total_budget + 8);
'''
new = '''    // The budget covers the whole generation, so it is also exactly what the state has to hold.
    weight_verify("pre-request");   // amdnas-fixes weight-verify
    model.init_state(&st, ids.size() + total_budget + 8);
'''
n = s.count(old)
if n != 1: sys.exit(f"ABORT init_state anchor {n}x")
s = s.replace(old, new)
for inc in ("#include <chrono>", "#include <mutex>"):
    if inc not in s: s = s.replace('#include "gpu/static_placement.h"', '#include "gpu/static_placement.h"\n' + inc, 1)
open(p, "w").write(s); print("patch 7 applied")
