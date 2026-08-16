// Cross-cutting counters that a phase timer needs and cannot see from where it lives.
//
// `Model`'s phase profile is host code and knows nothing about HIP; the thing it most needs to know
// is how much of a phase was spent blocked in `hipStreamSynchronize`, which only gpu/ can measure.
// So the counter lives here, in a translation unit both halves link, and AFF_DRAIN bumps it.
//
// A PHASE LABEL NAMES HOST WORK; THE CLOCK MEASURES A WAIT. Decode's largest phase used to read as
// "routing (host top-k)", which is a narrow argmax over a few tokens. What the stopwatch was timing
// is the drain inside `batch_router_topk`: the host cannot group a layer's tokens by expert until it
// knows the selection, so it drains the stream, and that drain waits for the whole layer's device
// chain. A phase that is 99% wait is not a phase to optimise; it is a phase whose CONTENTS to go and
// find, and this is the column that says which kind it is.
//
// ---- AND WHY THE TABLE IS PER SITE --------------------------------------------------------------
//
// A stream drain is one of at least eight ways the dispatch thread can stop. The others are not
// exotic: `hipEventSynchronize` on the pinned and SSD staging rings, a `pread` on the dispatch
// thread, `hipDeviceSynchronize`, a blocking `hipMemcpy` — and `hipMemcpyAsync` FROM PAGEABLE
// MEMORY, which the driver stages through its own bounce buffer and which is host-synchronous at
// these sizes while looking, in every profile the engine prints, like a free async copy.
//
// So "is the host off the hot path" cannot be answered by a total; it needs a count per SITE, with
// the target zero at each rather than "small" in sum. A per-site table is also the only way to tell
// a removed wait from a MOVED one — removing one drain can make another absorb the identical wait,
// which against a single counter reads as no change at all.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

namespace aff {

// Nanoseconds spent inside device stream drains, and how many there were. Monotonic for the life of
// the process; the phase timer reads deltas. Relaxed ordering throughout — this is a stopwatch, and
// a torn read costs a wrong nanosecond in a number printed to two decimal places of a millisecond.
std::atomic<uint64_t>& drain_ns();
std::atomic<uint64_t>& drain_count();

// ---- and the TAIL of that distribution, which a mean cannot show ---------------------------------
//
// A kernel trace of decode contains a handful of dispatches three orders of magnitude above the
// median, and they arrive simultaneously on every card whichever kernel each was running. That is
// the whole machine stopping, not a slow kernel.
//
// A mean cannot see them: spread over thousands of drains, even seconds of stall move the average by
// a fraction of a millisecond and hide inside the run-to-run spread. So the drain records its worst
// case and how many were pathological — the cheapest instrument for "did the machine stop".
std::atomic<uint64_t>& drain_max_ns();
std::atomic<uint64_t>& drain_slow();      // drains over 10 ms
// ...and WHICH drain the worst one was. Prefill's spread is a stall — the run's worst drain predicts
// its throughput almost exactly — but a magnitude alone does not say whether it is the first chunk
// paying a one-off (an arena growth, a first touch of the registered pool) or something recurring.
// The index does, and it costs one more relaxed store on the path that already lost the race.
std::atomic<uint64_t>& drain_max_at();

inline uint64_t drain_ns_now() { return drain_ns().load(std::memory_order_relaxed); }

// ---- the per-site table -------------------------------------------------------------------------
//
// One entry per place the dispatch thread can block, named after the CALL SITE rather than the
// primitive, because "which hipEventSynchronize" is the question a person actually has. The
// primitive is in the name string so the two are never confused.
//
// Order is the report's order. `kOther` is deliberately last and deliberately exists: a site that
// has not been given a name yet must still be counted, or widening the instrument would silently
// shrink the total it is meant to widen.
enum class BlockSite : uint8_t {
  RouterTopk = 0,   // dense_gpu.hip  batch_router_topk        — the per-layer round trip
  RouterHash,       // dense_gpu.hip  batch_router_hash        — layers 0-2
  Router,           // dense_gpu.hip  batch_router             — non-dev_route layers, per card
  NormFetch,        // dense_gpu.hip  batch_norm_fetch
  HcPre,            // dense_gpu.hip  batch_hc_pre with norm_host
  HcRead,           // dense_gpu.hip  hc_read                  — once a token
  HcMean,           // dense_gpu.hip  hc_tap                   — kept so a re-introduced wait is named
  Begin,            // dense_gpu.hip  batch_begin              — `emb` is a caller local
  BlockAdd,         // dense_gpu.hip  batch_block_add          — CPU-only arm
  Head,             // dense_gpu.hip  batch_head
  HeadSample,       // dense_gpu.hip  batch_head_sample
  DraftProj,        // dense_gpu.hip  draft_proj
  DraftHead,        // dense_gpu.hip  draft_head               — twice a markov step, per card
  PinRing,          // static_placement.hip  the pinned tile ring slot (hipEventSynchronize)
  SsdRing,          // static_placement.hip  ssd_stage waiting for a staging slot
  SsdRead,          // static_placement.hip  ssd_stage's pread + transpose, on this thread
  MoverRetire,      // static_placement.hip  engine_retire
  RouteRing,        // dense_gpu.hip         route_publish, the selection mirror wrapping
  NcompRing,        // dense_gpu.hip         batch_attend, the per-token compressed-row counts
  ArenaGrow,        // static_placement.hip  a routed-arena buffer had to be reallocated
  PageableH2D,      // any hipMemcpyAsync whose source the engine did not pin
  BlockingMemcpy,   // a synchronous hipMemcpy anywhere on a per-token path
  DeviceSync,       // hipDeviceSynchronize
  LaunchStall,      // hipLaunchKernel itself blocking on a full AQL ring
  Other,
  kCount
};

struct BlockStat {
  std::atomic<uint64_t> ns{0};
  std::atomic<uint64_t> count{0};
  std::atomic<uint64_t> max_ns{0};
  std::atomic<uint64_t> max_at{0};      // the value of `count` when the worst one landed
};

// The table itself. `kCount` entries, function-local so there is no static init order to get wrong.
BlockStat* block_stats();
const char* block_site_name(BlockSite);

// Record one blocking event. Bumps BOTH the per-site entry and the aggregate `drain_*` counters, so
// every existing reader — AFF_PH's wait column, dump_profile's drain line — keeps working unchanged
// and automatically widens to cover every site the table names.
void block_record(BlockSite site, uint64_t ns);

// Whether to run the clock pair at all. Latched once from AFF_PROFILE: a getenv on a path that runs
// thousands of times a token is itself a measurable cost.
bool block_timed();

} // namespace aff

// A blocking call that is NOT a stream drain — an event sync, a synchronous copy, a pread. Scoped as
// a macro rather than an RAII object so the primitive it wraps stays visible at the call site; the
// point of the table is that a reader can see WHICH `hipEventSynchronize` a row refers to.
//
// Lives here rather than in gpu/ because static_placement and dense_gpu both need it and neither
// owns the other, and because nothing in it touches HIP.
#define AFF_BLOCK_AT(SITE, STMT) do { \
    if (aff::block_timed()) { \
      const auto _b0 = std::chrono::steady_clock::now(); \
      STMT; \
      aff::block_record((SITE), (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>( \
          std::chrono::steady_clock::now() - _b0).count()); \
    } else { \
      STMT; \
    } } while (0)
