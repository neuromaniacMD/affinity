#include "engine/instrument.h"

#include <cstdlib>
#include <cstring>

namespace aff {

std::atomic<uint64_t>& drain_ns() {
  static std::atomic<uint64_t> v{0};
  return v;
}

std::atomic<uint64_t>& drain_count() {
  static std::atomic<uint64_t> v{0};
  return v;
}

std::atomic<uint64_t>& drain_max_ns() {
  static std::atomic<uint64_t> v{0};
  return v;
}

std::atomic<uint64_t>& drain_slow() {
  static std::atomic<uint64_t> v{0};
  return v;
}

std::atomic<uint64_t>& drain_max_at() {
  static std::atomic<uint64_t> v{0};
  return v;
}

BlockStat* block_stats() {
  static BlockStat v[(size_t)BlockSite::kCount];
  return v;
}

// The primitive is in the string, not in a second enum, because every time the two were separate
// somebody read the site column and assumed a stream sync.
const char* block_site_name(BlockSite s) {
  switch (s) {
    case BlockSite::RouterTopk:     return "router_topk       (stream sync)";
    case BlockSite::RouterHash:     return "router_hash       (stream sync)";
    case BlockSite::Router:         return "router            (stream sync)";
    case BlockSite::NormFetch:      return "norm_fetch        (stream sync)";
    case BlockSite::HcPre:          return "hc_pre norm_host  (stream sync)";
    case BlockSite::HcRead:         return "hc_read           (stream sync)";
    case BlockSite::HcMean:         return "hc_tap            (unused: no host side)";
    case BlockSite::Begin:          return "begin             (stream sync)";
    case BlockSite::BlockAdd:       return "block_add         (stream sync)";
    case BlockSite::Head:           return "head              (stream sync)";
    case BlockSite::HeadSample:     return "head_sample       (stream sync)";
    case BlockSite::DraftProj:      return "draft_proj        (stream sync)";
    case BlockSite::DraftHead:      return "draft_head markov (stream sync)";
    case BlockSite::PinRing:        return "pinned tile ring  (event sync)";
    case BlockSite::SsdRing:        return "ssd staging ring  (event sync)";
    case BlockSite::SsdRead:        return "ssd pread+shard   (host i/o)";
    case BlockSite::MoverRetire:    return "engine_retire     (event sync)";
    case BlockSite::RouteRing:      return "route_ring        (event sync)";
    case BlockSite::NcompRing:      return "ncomp_ring        (event sync)";
    case BlockSite::ArenaGrow:      return "routed arena grow (device sync)";
    case BlockSite::PageableH2D:    return "pageable h2d      (driver staging)";
    case BlockSite::BlockingMemcpy: return "blocking memcpy   (sync copy)";
    case BlockSite::DeviceSync:     return "device sync       (device sync)";
    case BlockSite::LaunchStall:    return "launch            (aql ring full)";
    case BlockSite::Other:          return "other             (unattributed)";
    case BlockSite::kCount:         break;
  }
  return "?";
}

void block_record(BlockSite site, uint64_t ns) {
  const size_t i = (size_t)site < (size_t)BlockSite::kCount ? (size_t)site
                                                            : (size_t)BlockSite::Other;
  BlockStat& b = block_stats()[i];
  b.ns.fetch_add(ns, std::memory_order_relaxed);
  const uint64_t n = b.count.fetch_add(1, std::memory_order_relaxed) + 1;
  uint64_t m = b.max_ns.load(std::memory_order_relaxed);
  while (ns > m && !b.max_ns.compare_exchange_weak(m, ns, std::memory_order_relaxed)) {}
  if (ns > m) b.max_at.store(n, std::memory_order_relaxed);

  // The aggregate, so every existing reader widens automatically rather than being left measuring
  // the one primitive it happened to know about.
  drain_ns().fetch_add(ns, std::memory_order_relaxed);
  const uint64_t g = drain_count().fetch_add(1, std::memory_order_relaxed) + 1;
  if (ns > 10000000ull) drain_slow().fetch_add(1, std::memory_order_relaxed);
  uint64_t gm = drain_max_ns().load(std::memory_order_relaxed);
  while (ns > gm && !drain_max_ns().compare_exchange_weak(gm, ns, std::memory_order_relaxed)) {}
  if (ns > gm) drain_max_at().store(g, std::memory_order_relaxed);
}

// AFF_BLOCK=1 as well as AFF_PROFILE, and the extra switch is not a convenience.
//
// AFF_PROFILE turns on the hybrid census, which walks the host's copy of the router's selection —
// so with the device-built dispatch it FORCES the readback and the drain this table exists to
// measure. An instrument that cannot be enabled without disabling the thing it measures is not an
// instrument. AFF_BLOCK turns on the table alone.
bool block_timed() {
  static const bool v = [] {
    for (const char* n : {"AFF_BLOCK", "AFF_PROFILE"}) {
      const char* s = std::getenv(n);
      if (s && *s && *s != '0') return true;
    }
    return false;
  }();
  return v;
}


} // namespace aff
