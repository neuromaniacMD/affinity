// The host tier of the hybrid: an explicitly owned RAM pool for the experts that did not fit in
// VRAM, with SSD kept as a managed fallback rather than as the kernel's idea of paging.
//
// WHY THIS EXISTS, and why it is not just mmap.
//
// Reading straight out of the container's mapping is elegant until the machine is actually full, at
// which point the kernel — not this engine — decides what stays resident, and it decides badly for
// this workload: the same sweep runs several times slower inside a real decode run than in
// isolation, because almost nothing is free and every fault has to evict something first.
// MADV_WILLNEED makes it worse, which is the tell — the faults were never too small, the pages keep
// being taken away, and a hint cannot fix a policy disagreement.
//
// So: allocate the memory, copy the bytes in once, and never give the kernel the opportunity. The
// pool is owned here, it is MAP_POPULATE'd at load, and its residency does not change while
// decoding.
//
// Its PAGE SIZE does not matter: driving huge-page coverage from most of the pool to nearly all of
// it moves decode by nothing and costs seconds of load. `thp_bytes` in the stats reports the
// coverage so that claim stays falsifiable.
//
// SSD IS A TIER, DELIBERATELY — AND UNDER THE PLACEMENT ENGINE IT BECOMES A CHOSEN ONE.
//
// Today it holds whatever did not fit the RAM budget, which varies run to run with how much RAM is
// free at load; on a machine where VRAM's share plus the pool's covers the whole container, the
// tier is very nearly empty by accident. Do not read that as "disk is not really a tier here". The
// placement engine assigns it on purpose — the experts it
// predicts are least likely to be routed to go here, and giving the cold tail to disk is how it
// buys a RAM pool big enough to hold a working set that can actually move.
//
// ON ACTIVATION AN SSD-TIER EXPERT IS PURE STREAMING: read it, use it, discard it. No admission and
// no eviction, so this stays a `fetch()` and never grows cache semantics. Being routed to an expert
// on disk is the price of a bet the strategy made, paid in full each time; what changes in response
// is the strategy, not this. Promotion OUT of the tier is a placement decision made on the engine's
// own schedule and never a reaction to an activation.
//
// When it is used it is used EXPLICITLY — one pread of one expert into a caller-owned staging
// buffer, at whatever the device does on that access pattern, with a counter saying it happened. It
// is never a silent page fault.
//
// The budget is a hard cap, not a target: the host may be running other work, and a pool that grew
// until the OOM killer noticed would be worse than one that spills a few hundred experts to disk.

#pragma once

#include "format/aff_reader.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace aff {

struct HostPoolStats {
  uint64_t pooled = 0;             // experts resident in the RAM pool
  uint64_t ssd = 0;                // experts that must come from disk on every use
  uint64_t bytes = 0;              // pool size actually allocated
  uint64_t ssd_reads = 0;          // demand preads served
  uint64_t thp_bytes = 0;          // of `bytes`, how much came back on 2 MiB pages
  double   load_seconds = 0.0;
  double   load_gbps = 0.0;
};

class HostExpertPool {
public:
  ~HostExpertPool();
  HostExpertPool(const HostExpertPool&) = delete;
  HostExpertPool& operator=(const HostExpertPool&) = delete;
  HostExpertPool() = default;

  // Admits experts for which `wanted(layer, expert)` is true, hottest first by the container's
  // popularity profile, until `budget_bytes` is exhausted. The rest fall to the SSD tier.
  //
  // Call AFTER placement has decided the VRAM set, and pass a predicate that excludes it: an expert
  // held in VRAM has no reason to occupy host RAM as well, and the tiers are disjoint by design.
  //
  // DISJOINT IS A STATIC-PLACEMENT INVARIANT, NOT A LAW. It holds because nothing moves at
  // runtime. A placement engine that demotes an expert out of VRAM copies its bytes back D2H into a
  // pool slot, so the pool's membership changes after load, which this init() cannot express — it
  // takes a predicate once and is done. The mover will need slot-granular admit/evict beside it.
  using Wanted = bool (*)(void* ctx, uint32_t layer, uint32_t expert);

  // ---- what the pool STORES, and why it is not the container's bytes ---------------------------
  //
  // The hybrid expert GEMM reads a missing expert straight out of this pool over PCIe — the tile's
  // three pointers name host memory and the kernel streams the weights as it computes, with no
  // staging ring and no copy stream. That only
  // works if what is here is byte-for-byte what a VRAM slab would hold: colgroup-major planes, and
  // sharded when placement is sharding. Row-major would be correct-looking and several times slower
  // to read even in VRAM, which over the bus is not a tier at all.
  //
  // So `init` takes the same permutation the residency loader applies and runs it on the reader
  // threads, and the pool's slot stride becomes `out_stride` rather than the container's. `src` is
  // the container's `stride()` bytes as they came off disk, `dst` is `out_stride` bytes, and the
  // transform owns whatever scratch it needs (it runs on eight threads, so that scratch has to be
  // thread-local inside it).
  //
  // Null `tf` keeps the container layout verbatim, which is what a CPU reader would want.
  using Transform = void (*)(void* ctx, uint32_t layer, const uint8_t* src, uint8_t* dst);
  bool init(const AffReader* aff, uint64_t budget_bytes, Wanted wanted, void* wanted_ctx,
            uint32_t n_layer, uint32_t n_expert, std::string* err,
            Transform tf = nullptr, void* tf_ctx = nullptr, uint64_t out_stride = 0,
            uint64_t n_stage = 0, uint64_t reserve_bytes = 0);

  bool enabled() const { return base_ != nullptr; }
  // The CONTAINER's expert stride — what `fetch` writes and what the transform reads.
  uint64_t stride() const { return stride_; }
  // The POOL's slot stride, which is the transform's output size and differs from the above
  // whenever the pool holds shards. `resident()` is spaced by this.
  uint64_t out_stride() const { return out_stride_; }
  // The expert's bytes in RAM, or null when it is not pooled. Whatever `tf` produced — which under
  // the shipped placement is a VRAM slab's layout, not the container's, so the container's own
  // descriptors do NOT address it. Same contract as the slab, deliberately, because that is what
  // makes a pooled expert dispatchable without a copy.
  const uint8_t* resident(uint32_t layer, uint32_t expert) const noexcept {
    if (!base_) return nullptr;
    const int64_t s = slot_[(size_t)layer * n_expert_ + expert];
    return s < 0 ? nullptr : base_ + (uint64_t)s * out_stride_;
  }

  // ---- the placement engine's seam: membership after load ---------------------------------------
  //
  // init() takes a predicate once and is done, which was the whole story while nothing moved. A
  // demotion has to give an expert a slot here and a promotion has to take one back, so these three
  // exist and nothing else writes `slot_`.
  //
  // CALLED ON THE DISPATCH THREAD, at a layer boundary, and that is not a style preference.
  // `resident()` above is read by StaticPlacement::pool_shard_dev inside the dispatch loop; a mover
  // thread writing `slot_` underneath it would be a data race on the one table that decides where a
  // GEMM reads its weights. The engine issues transfers asynchronously and applies the table edit
  // itself once they land.
  int64_t slot_of(uint32_t layer, uint32_t expert) const noexcept {
    return base_ ? slot_[(size_t)layer * n_expert_ + expert] : -1;
  }
  // Writable, because a demotion DMAs into it. Null for an out-of-range slot rather than an address
  // just past the pool, which is the shape of bug that reads as a corrupted expert three layers on.
  uint8_t* slot_base(int64_t slot) const noexcept {
    if (!base_ || slot < 0 || (uint64_t)slot >= slots()) return nullptr;
    return base_ + (uint64_t)slot * out_stride_;
  }
  uint64_t slots() const noexcept { return out_stride_ ? alloc_bytes_ / out_stride_ : 0; }
  // `slot < 0` clears. The caller owns the invariant that a slot holds at most one expert; this
  // does not check, because the engine's free list is what enforces it and a second opinion here
  // would only be a second thing to keep in sync.
  void bind(uint32_t layer, uint32_t expert, int64_t slot) noexcept {
    if (base_) slot_[(size_t)layer * n_expert_ + expert] = slot;
  }

  // The SSD tier: read one expert into `dst`, which must hold stride() bytes. Explicit, counted,
  // and synchronous — the caller knows it paid for a disk read.
  bool fetch(uint32_t layer, uint32_t expert, uint8_t* dst) const;

  // ---- the staging ring, which is what makes the SSD tier dispatchable -------------------------
  //
  // These slots hold no expert permanently. A dispatch that selects an SSD-tier expert claims one,
  // reads the container into it through `fetch` and the same transform the pool was loaded with,
  // and points the tile at it — after which the GEMM streams from it over PCIe exactly as it would
  // from a pooled expert, with no kernel change. The slots are inside the registered allocation for
  // that reason; a private malloc would not be DMA-able and would cost a bounce copy.
  //
  // A claimed slot must stay intact until the kernel that reads it has completed, so the caller
  // owns the lifetime (StaticPlacement gates them on a per-card event).
  uint64_t stage_slots() const noexcept { return n_stage_; }
  uint8_t* stage_base(uint64_t i) const noexcept {
    return (base_ && i < n_stage_) ? base_ + (stage0_ + i) * out_stride_ : nullptr;
  }

  // The whole allocation, so a GPU driver can register it and DMA out of it directly. One
  // anonymous range populated in a single pass, so it registers as a contiguous span quickly, and
  // the cards then read it at the same rate as memory allocated pinned. Without registration an H2D
  // from here goes through the driver's bounce buffer and is a host memcpy in disguise, which is the
  // entire difference between this being useful and not.
  const uint8_t* base() const noexcept { return base_; }
  uint64_t bytes() const noexcept { return alloc_bytes_; }

  const HostPoolStats& stats() const {
    st_.ssd_reads = ssd_reads_.load(std::memory_order_relaxed);
    return st_;
  }

private:
  const AffReader* aff_ = nullptr;
  uint8_t* base_ = nullptr;
  uint64_t alloc_bytes_ = 0, stride_ = 0, out_stride_ = 0;
  uint32_t n_layer_ = 0, n_expert_ = 0;
  uint64_t n_stage_ = 0, stage0_ = 0;         // staging slots, and the slot index they start at
  std::vector<int64_t> slot_;                 // (layer*n_expert + e) -> pool slot, or -1
  mutable std::atomic<uint64_t> ssd_reads_{0};
  mutable HostPoolStats st_;
};

} // namespace aff
