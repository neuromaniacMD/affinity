#include "gpu/static_placement.h"
#include "gpu/expert_kernel_hip.h"
#include "gpu/batch_kernels.h"
#include "gpu/dense_gpu.h"
#include <array>

#include "gpu/digest.h"
#include "gpu/hip_common.h"
#include "gpu/moe_plan.h"
#include "gpu/trace.h"
#include "engine/expert_kernel.h"
#include "engine/instrument.h"
#include "engine/ops.h"
#include "engine/route_trace.h"
#include "engine/threadpool.h"

#include "ui/log.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>
#include <mutex>
#include <unistd.h>

namespace aff {

struct StaticPlacement::Impl {
  // One big slab per card rather than 7000 small allocations: the expert stride is uniform, so a
  // slot index is arithmetic, and hipMalloc is slow enough that per-expert calls dominate load.
  std::vector<uint8_t*> slab;
  std::vector<float*>   d_cb;        // codebook, per card
  std::vector<hipStream_t> stream;
  // ---- the hybrid stream ------------------------------------------------------------------------
  //
  // One per card, and it exists so that the experts this card does NOT hold can be read over PCIe
  // at the same time as the ones it does are read out of VRAM. Two different pieces of hardware; a
  // single stream would serialise them and there would be no hybrid tier worth the name.
  //
  // Created when the pool is registered, because that is the moment the tier becomes possible, and
  // never used when a layer misses nothing — a stream that carries no work still costs two event
  // records and two waits.
  std::vector<hipStream_t> hyb;
  // ---- THE FOUR CROSS-STREAM EVENTS, AS RINGS ---------------------------------------------------
  //
  // One event per card was correct only while the router's drain bounded the host to a layer ahead.
  // With the drain gone the host queues dozens of layers, and an event is a single signal: layer
  // L+1's record resets it, and a wait from layer L that has not been serviced yet is satisfied by
  // whichever record happens to have set it. The compute stream then reads the hybrid stream's gate
  // and up before they have landed — fluent output, and a different `ids-md5` every run.
  //
  // kEvSlots > the 43 layers of a block, so a slot cannot be reused inside one.
  static constexpr uint32_t kEvSlots = 64;
  std::vector<std::array<hipEvent_t, kEvSlots>> ev_q, ev_gu, ev_sw, ev_dn;
  std::vector<uint64_t> ev_seq;
  uint64_t stride = 0;               // == the container's expert stride; experts are verbatim

  // ---- the mover: everything the placement engine needs that has a HIP type in it ---------------
  //
  // One stream a card, so a promotion's H2D and a demotion's D2H run on the copy engine beside the
  // compute stream rather than in front of it. SDMA copies barely inflate a resident GEMM running
  // next to them, where zero-copy tiles inflate it several times over — the copy engine is the cheap
  // way to move bytes here, and it is the one the hybrid tier does not use.
  std::vector<hipStream_t> mv;
  // Recorded on each card's COMPUTE stream at the top of a tick that is about to issue, and waited
  // on by every transfer that tick. This is the whole of §3.3's answer: a slot the mover writes was
  // last read by a kernel launched before this record, so ordering behind it is ordering behind
  // every reader. It is recorded lazily — a tick that moves nothing pays nothing.
  std::vector<hipEvent_t> ev_layer;
  // The lead gate's event ring, `kLeadRing` blocks deep, R events a block.
  std::vector<hipEvent_t> lead_ev;
  uint64_t lead_blk = 0;
  uint64_t gate_tick = UINT64_MAX;   // the tick `ev_layer` was last recorded for

  // A transfer in flight and its afterlife. The same event set is used twice, in sequence: once for
  // the copy, and once — re-recorded on the compute stream at publish time — as the quarantine
  // barrier that says no kernel can still be reading the slot being freed.
  struct Move {
    // `Published` is a within-tick state and never survives `engine_retire`: the edit has gone into
    // this tick's patch, and the move is waiting for the patch to reach the stream so its quarantine
    // event can be recorded downstream of it.
    enum class State : uint8_t { Free, Copying, Published, Quarantine };
    State state = State::Free;
    bool promote = false;
    uint32_t idx = 0;                       // layer * n_expert + expert
    uint64_t slab_slot = 0;                 // promote: written. demote: read, then freed.
    int64_t  pool_slot = -1;                // promote: read, then freed. demote: written.
    hipEvent_t ev[8] = {};
  };
  std::vector<Move> moves;
  // Slots that hold nothing. `free_slab` starts with the shadow slots and nothing else; `free_pool`
  // starts EMPTY, because at load the pool holds exactly the complement of VRAM. The two feed each
  // other — a landed promotion frees the pool slot a demotion then writes, and a landed demotion
  // frees the slab slot the next promotion writes — so the engine primes itself off the shadows and
  // never needs a spare byte of either tier beyond them.
  std::vector<uint64_t> free_slab;
  std::vector<int64_t>  free_pool;

  // ---- the SSD tier's staging ring --------------------------------------------------------------
  //
  // An expert with neither a slab offset nor a pool slot has its bytes only on disk. A dispatch that
  // selects one claims a slot here, reads the container into it and runs the pool's own transform,
  // and from that point the tile is indistinguishable from a pooled expert's — same registered
  // memory, same zero-copy stream, no kernel change.
  //
  // The slot has to outlive the kernel that reads it, which is why each carries a per-card event
  // recorded after that layer's launches. A slot is reusable once every card's event has fired, so
  // the ring self-paces: with enough slots nothing waits, and with too few the claim blocks rather
  // than handing out memory a running GEMM is still streaming.
  //
  // A correctness path, not a fast one: a staged expert costs a pread plus the transform before its
  // layer can launch, an order of magnitude more than bytes already in RAM. Every one is reported so
  // that landing here reads as a misconfigured budget.
  //
  // TRANSPORT, NOT PLACEMENT. Retaining bindings and evicting by LRU was tried here and the reuse is
  // real and large — but it is a second placement
  // policy with its own eviction order running beside the engine's heat plane, and two policies
  // over one set of tiers is how they disagree. That finding belongs in the engine: an SSD expert
  // the heat plane sees selected repeatedly should be PROMOTED into the pool, once.
  //
  //   idx      the expert this slot holds for the dispatch being built
  //   claimed  a tile points here and its launches are not yet issued; never reuse
  //   armed    launches issued; free once every card's event has fired
  struct StageSlot {
    uint64_t idx = UINT64_MAX;
    bool claimed = false;
    bool armed = false;
    hipEvent_t ev[8] = {};
  };
  std::vector<StageSlot> stage;
  std::vector<uint32_t> stage_claimed;        // slots claimed by the dispatch being built
  std::vector<uint8_t> stage_raw;             // container bytes, reused across fetches

};

StaticPlacement::~StaticPlacement() {
  if (!impl_) return;
  for (size_t i = 0; i < impl_->mv.size(); ++i) {
    if (i < devs_.size()) (void)hipSetDevice(devs_[i]);
    if (impl_->mv[i]) (void)hipStreamDestroy(impl_->mv[i]);
    if (i < impl_->ev_layer.size() && impl_->ev_layer[i])
      (void)hipEventDestroy(impl_->ev_layer[i]);
  }
  for (auto& m : impl_->moves)
    for (size_t i = 0; i < devs_.size(); ++i)
      if (m.ev[i]) (void)hipEventDestroy(m.ev[i]);
  for (size_t i = 0; i < impl_->slab.size(); ++i) {
    (void)hipSetDevice(devs_[i]);
    (void)hipFree(impl_->slab[i]);
    (void)hipFree(impl_->d_cb[i]);
    (void)hipStreamDestroy(impl_->stream[i]);
  }
  for (size_t i = 0; i < plc_.size(); ++i)
    if (plc_[i]) { (void)hipSetDevice(devs_[i]); (void)hipFree(plc_[i]); }
  plc_.clear();
  for (size_t i = 0; i < devs_.size() && i < 8; ++i) {
    (void)hipSetDevice(devs_[i]);
    for (void* p : {(void*)plan_col_[i], (void*)plan_cnt_[i], (void*)plan_st_[i]})
      if (p) (void)hipFree(p);
  }
  if (pool_registered_ && pool_ && pool_->base())
    (void)hipHostUnregister(const_cast<uint8_t*>(pool_->base()));
  delete impl_;
  impl_ = nullptr;
}

// The residency loader's permutation, run on the pool's reader threads. Row-major bytes in,
// colgroup-major shards out — see the note in static_placement.h on why the pool cannot hold what
// the container holds, and expert_kernel_hip.h on why colgroup-major is so much faster to read.
//
// The scratch is function-local and thread-local because this runs on every reader at once, once per
// pooled expert; a vector per call would be the load cost.
void StaticPlacement::pool_tf(void* ctx, uint32_t layer, const uint8_t* src, uint8_t* dst) {
  StaticPlacement* self = (StaticPlacement*)ctx;
  if (!self || !self->aff_ || layer >= self->n_layer_) return;
  const AffLayerDesc& L = self->aff_->layer(layer);
  static thread_local std::vector<uint8_t> whole, scratch;
  whole.assign(src, src + L.stride);
  expert_transpose_planes_host(whole.data(), L, scratch);
  if (self->tp_parts_ > 1) {
    for (uint32_t p = 0; p < self->tp_parts_; ++p)
      expert_shard_host(whole.data(), L, self->shard_, self->tp_parts_, p,
                        dst + (size_t)p * self->shard_.stride);
  } else {
    std::memcpy(dst, whole.data(), L.stride);
  }
}

uint64_t StaticPlacement::slot_stride() const {
  if (tp_parts_ > 1) return shard_.stride;
  return impl_ ? impl_->stride : 0;
}

uint64_t StaticPlacement::pool_out_stride() const {
  if (tp_parts_ > 1) return (uint64_t)tp_parts_ * shard_.stride;
  return aff_ ? aff_->layer(0).stride : 0;
}

// ---- WHICH BUILD RUNS, AND WHY THAT IS NOT A SETTING --------------------------------------------
//
// The dispatch is built ON THE CARD for every layer that can be, always. There is no flag for it and
// no host-built alternative to fall back to: the two were compared field by field over ~1500
// dispatches and by the emitted text over whole runs, and they agree exactly (moe_plan.h names the
// four things that fix the bits and the kernel reproduces all four).
//
// The one layer it cannot claim is one holding an expert on the SSD TIER, which has no address until
// the host reads it off disk — and a kernel cannot read a disk. That is not a fallback either, it is
// the only code that can serve those bytes, and which layers they are is a fact about PLACEMENT that
// the host knows without asking the card anything (`ssd_layer_`, maintained in publish_base).
//
// The tier is not a misconfiguration to be tuned away. host_pool.cpp sizes the staging ring from the
// worst layer's spill and records why growing the pool cannot empty the tier: every slot the ring
// takes displaces a pooled expert and so creates spill, and the fixed point is always the ceiling.
// ---- AND THE ROUTER NO LONGER READS THE SELECTION BACK -------------------------------------------
//
// Which was the point of the whole exercise: it removes more than half the host blocking events a
// token. Removing the drain also exposed every buffer that drain had been guarding: the emitted text
// went NON-DETERMINISTIC, the same binary and seed giving a different `ids-md5` most runs, even under
// `--placement-strategy static` where the mover never runs at all.
//
// The cause was not this file. A per-stage fingerprint put the first divergence between the query and
// the attention output — the flash kernel producing a different result from an IDENTICAL query — and
// it was `batch_attend` staging its per-token compressed-row counts through a single pinned buffer.
// Pinned means the H2D is genuinely asynchronous, so the DMA read it later; the drain was the only
// thing stopping the next layer's memcpy from overwriting it first, and `b_ncomp` is how many KV rows
// each token attends to. It is a ring now, and the two arms are bit-identical.
//
// THE LESSON GENERALISES AND THE NEXT ONE IS ALREADY MARKED. Every host buffer feeding an
// asynchronous H2D on the layer path needs a ring or an event, not just a pin. `batch_router_hash`
// has the identical shape and is safe only because of its own drain; there is a comment there
// saying so.

const uint8_t* StaticPlacement::pool_shard_dev(uint32_t layer, uint32_t expert, size_t dv) const {
  if (!pool_dev_ || !pool_ || !pool_->base()) return nullptr;
  const uint8_t* h = pool_->resident(layer, expert);
  if (!h) return nullptr;
  return pool_dev_ + (size_t)(h - pool_->base()) +
         (tp_parts_ > 1 ? (size_t)dv * shard_.stride : 0);
}

// ---- the placement table ------------------------------------------------------------------------
//
// See static_placement.h for what it is and why it is not the device-side residency mirror that was
// deleted. These two functions are the whole of it: one allocation, and one publish that every site
// which changes placement calls.

bool StaticPlacement::plc_alloc(std::string* err) {
  const size_t n = (size_t)n_layer_ * n_expert_;
  if (!n || devs_.empty()) return true;
  plc_.assign(devs_.size(), nullptr);
  plc_host_.assign(devs_.size() * n, 0ull);
  ssd_layer_.assign(n_layer_, 0u);
  for (size_t dv = 0; dv < devs_.size(); ++dv) {
    if (hipSetDevice(devs_[dv]) != hipSuccess) { if (err) *err = "plc: hipSetDevice"; return false; }
    void* p = nullptr;
    // Fine-grained so the CPU can store into it directly — same allocation the all-reduce uses for
    // its flags. A plain hipMalloc would need a copy per publish and a stream to order it on.
    if (hipExtMallocWithFlags(&p, n * sizeof(uint64_t), hipDeviceMallocFinegrained) != hipSuccess) {
      if (err) *err = "plc: fine-grained alloc of " + std::to_string(n * 8) + " bytes failed";
      return false;
    }
    plc_[dv] = (uint64_t*)p;
    for (size_t i = 0; i < n; ++i) plc_[dv][i] = 0;
  }
  return true;
}

// Recompute one entry on every card. Called from the two places placement changes — the load fill,
// and the engine's publish — so the table cannot fall behind `device_of_`/`offset_of_` without one of
// them being edited to drop the call.
//
// `sink` decides where the DEVICE-side word goes; see the declaration. The host mirror is stored
// directly either way: its reader is this thread.
void StaticPlacement::publish_base(size_t idx, PlcPatch* sink) {
  if (plc_.empty()) return;
  const size_t n = (size_t)n_layer_ * n_expert_;
  const uint32_t layer = (uint32_t)(idx / n_expert_), e = (uint32_t)(idx % n_expert_);
  for (size_t dv = 0; dv < plc_.size(); ++dv) {
    if (!plc_[dv]) continue;
    // The tier rides in BIT 0. A shard base is 256-aligned on both tiers — `shard_.stride` is
    // 256-aligned per plane and the pool's slots are strided by it — so the low bits are free, and a
    // consumer needs the tier as well as the address: it decides which stream the tile goes on, and
    // it fixes `uniq`'s resident-before-pooled order, which fixes `start[e]` and therefore the bits.
    // One word rather than an address table plus a tier table, so there is one thing to publish and
    // one thing that can be stale.
    uint64_t a = 0;
    if (device_of_[idx] >= 0 && (tp_parts_ > 1 || device_of_[idx] == (int)dv))
      a = (uint64_t)(uintptr_t)(impl_->slab[dv] + offset_of_[idx]) | kPlcResident;
    else if (const uint8_t* p = pool_shard_dev(layer, e, dv))
      a = (uint64_t)(uintptr_t)p;
    if (sink) {
      // Queued, to land on this card's compute stream where the host published it from. Overflow is
      // a caller bug — the flush runs whenever a tick retires anything and the tick's move budget is
      // well under the patch's capacity — so it is fatal rather than dropped: a dropped publish is a
      // table that has silently fallen behind placement, which is the one thing this design must not
      // be able to do.
      PlcPatch& P = sink[dv];
      if (P.n >= kPlcPatchMax) {
        aff::ui::fatal("fatal: more than %u placement publishes in one tick on card %zu\n",
                     kPlcPatchMax, dv);
        std::abort();
      }
      P.idx[P.n] = (uint32_t)idx;
      P.val[P.n] = a;
      ++P.n;
    } else {
      plc_[dv][idx] = a;                   // load-time fill: nothing is in flight, nothing to order
    }
    plc_host_[dv * n + idx] = a;           // and the copy the host reads, from the same value
  }
  // And the layer's addressless count, maintained here because here is the only place an address
  // appears or disappears. Recomputed across the cards rather than incremented: a promote and a
  // demote both pass through this function and a delta would have to know which it was.
  if (layer < ssd_layer_.size()) {
    bool none = false;
    for (size_t dv = 0; dv < plc_.size(); ++dv) none |= (plc_host_[dv * n + idx] == 0ull);
    const uint32_t was = ssd_layer_[layer];
    // One expert, one unit, whichever cards lack it: the dispatch runs on all of them and one card
    // without an address is enough to make the layer unclaimable.
    static thread_local std::vector<uint8_t> mark;
    if (mark.size() != (size_t)n_layer_ * n_expert_) mark.assign((size_t)n_layer_ * n_expert_, 0u);
    const uint8_t now = none ? 1u : 0u;
    if (mark[idx] != now) {
      ssd_layer_[layer] += now ? 1u : (uint32_t)-1;
      mark[idx] = now;
    }
    if ((was == 0u) != (ssd_layer_[layer] == 0u))
      ssd_layers_hot_ += ssd_layer_[layer] ? 1u : (uint32_t)-1;
  }
}

// The plan kernel's per-card scratch. Grown with slack for the same reason `grow_w8a8` grows with
// slack: a realloc a layer would serialise the stream on the free.
bool StaticPlacement::plan_scratch(size_t dv, uint32_t total) {
  if (dv >= 8) return false;
  if (!plan_st_[dv]) {
    if (hipMalloc((void**)&plan_st_[dv], sizeof(MoePlanStatus)) != hipSuccess) return false;
    if (hipMalloc((void**)&plan_cnt_[dv], 8 * sizeof(uint32_t)) != hipSuccess) return false;
    // Zeroed once and never again: `bad` and `over` are monotone counters over the whole run, so
    // clearing them per call would turn a fault the host has not looked at yet into no fault at all.
    AFF_HIP_CHECK(hipMemset(plan_st_[dv], 0, sizeof(MoePlanStatus)));
  }
  if (plan_col_n_[dv] < total) {
    const uint32_t cap = total + (total >> 1) + 64u;
    if (plan_col_[dv]) {
      // Same argument as grow_w8a8's: the plan kernel of a dispatch still in flight is writing it.
      AFF_BLOCK_AT(aff::BlockSite::ArenaGrow, (void)hipDeviceSynchronize());
      AFF_HIP_CHECK(hipFree(plan_col_[dv]));
    }
    plan_col_[dv] = nullptr;
    if (hipMalloc((void**)&plan_col_[dv], (size_t)cap * sizeof(uint32_t)) != hipSuccess) return false;
    plan_col_n_[dv] = cap;
  }
  return true;
}

bool StaticPlacement::act_alloc() {
  if (act_ring_) return true;
  if (devs_.empty()) return false;
  AFF_HIP_CHECK(hipSetDevice(devs_[0]));
  void* h = nullptr;
  if (hipHostMalloc(&h, sizeof(ActRing), hipHostMallocMapped) != hipSuccess) return false;
  std::memset(h, 0, sizeof(ActRing));
  void* d = nullptr;
  if (hipHostGetDevicePointer(&d, h, 0) != hipSuccess) { (void)hipHostFree(h); return false; }
  act_ring_ = (ActRing*)h;
  act_dev_ = (ActRing*)d;
  return true;
}

// How many blocks of events the lead gate keeps. Deeper than any lead worth allowing.
static constexpr uint32_t kLeadRing = 128;

// Hold the host within `max_lead_blocks` LAYER DISPATCHES of the cards.
//
// WHY LAYERS AND NOT BLOCKS. A freed slab or pool slot returns to the free list only after its
// quarantine event signals, and that event is on the compute stream — so the recycle latency is
// however deep that stream is queued, and the move rate is `shadow slots / recycle latency`. The
// counters say it plainly: as shipped, 3026 of 3141 rounds find no free slot to move into and the
// engine manages 1017 promotions; with a per-layer drain restored it is 87 of 3181 and 4131. A
// block-granular gate does nothing about this, because one block IS 43 layers of queued work — it
// was built and measured first, and it moved the clock by 0.3%.
//
// One event a layer a card, waited on `max_lead_blocks` later. Not a drain: the host may still run
// that many dispatches ahead, which is far more lookahead than the issue path needs.
void StaticPlacement::lead_gate(void* const* streams, uint32_t nb_tok, size_t R) {
  if (!ecfg_.max_lead_blocks || !impl_) return;
  // DECODE ONLY. A prefill chunk is hundreds of tokens through one dispatch, so there are few of
  // them and each is enormous; holding the host two dispatches behind the cards throttles the issue
  // path for no gain, because the mover is not what bounds prefill. Measured at 8.5% of prefill for
  // nothing. The same nb_tok test the route trace uses to tell the two apart.
  if (nb_tok > 64) return;
  Impl& I = *impl_;
  if (I.lead_ev.empty()) {
    I.lead_ev.assign(kLeadRing * R, nullptr);
    for (size_t i = 0; i < I.lead_ev.size(); ++i) {
      if (hipSetDevice(devs_[i % R]) != hipSuccess) return;
      if (hipEventCreateWithFlags(&I.lead_ev[i], hipEventDisableTiming) != hipSuccess) return;
    }
  }
  const uint64_t blk = I.lead_blk++;
  const uint32_t d = ecfg_.max_lead_blocks < kLeadRing ? ecfg_.max_lead_blocks : kLeadRing - 1;
  if (blk >= d) {
    const size_t old = (size_t)((blk - d) % kLeadRing) * R;
    for (size_t dv = 0; dv < R; ++dv)
      if (I.lead_ev[old + dv]) (void)hipEventSynchronize(I.lead_ev[old + dv]);
  }
  const size_t cur = (size_t)(blk % kLeadRing) * R;
  for (size_t dv = 0; dv < R; ++dv) {
    if (!streams[dv] || !I.lead_ev[cur + dv]) continue;
    if (hipSetDevice(devs_[dv]) != hipSuccess) continue;
    (void)hipEventRecord(I.lead_ev[cur + dv], (hipStream_t)streams[dv]);
  }
}

// Take every record published since the last call. Deliberately non-blocking: whatever has landed is
// what heat sees, and what has not will be there next time.
uint32_t StaticPlacement::act_drain() {
  if (!act_ring_) return 0;
  const uint64_t h = __atomic_load_n(&act_ring_->head, __ATOMIC_ACQUIRE);
  if (h <= act_tail_) return 0;
  // Overrun. The writer never checks for space, so a reader this far behind is reading records the
  // card has already overwritten — skip to the oldest one still intact and SAY how many were lost.
  if (h - act_tail_ > kActSlots) {
    act_drop_ += h - act_tail_ - kActSlots;
    act_tail_ = h - kActSlots;
  }
  static thread_local std::vector<uint32_t> distinct, cnts;
  uint32_t took = 0;
  for (; act_tail_ < h; ++act_tail_, ++took) {
    const ActRecord& R = act_ring_->slot[act_tail_ % kActSlots];
    const uint32_t layer = R.layer;
    if (layer >= n_layer_) continue;
    distinct.clear(); cnts.clear();
    // ASCENDING by expert id, which is what note_missing's comment says the list is. The host loop
    // this replaces built it in order of first appearance in `sel` and the comment was simply
    // wrong; `observe` credits each expert once and is order-free, so heat is unaffected either way,
    // and `Demand` now drains its admissions in the order it documents.
    for (uint32_t e = 0; e < n_expert_ && e < kActExperts; ++e)
      if (R.counts[e]) { distinct.push_back(e); cnts.push_back(R.counts[e]); }
    heat_.observe(layer, distinct.data(), (uint32_t)distinct.size(), cnts.data());
    if (ecfg_.strategy == PlacementEngineConfig::Strategy::Demand) {
      static thread_local std::vector<uint32_t> missing;
      missing.clear();
      for (uint32_t e : distinct)
        if (device_of_[(size_t)layer * n_expert_ + e] < 0) missing.push_back(e);
      heat_.note_missing(layer, missing.data(), (uint32_t)missing.size());
    }
    ++act_seen_;
  }
  return took;
}

void StaticPlacement::publish_all() {
  if (plc_.empty()) return;
  const size_t n = (size_t)n_layer_ * n_expert_;
  // Directly: this runs at load, before a dispatch exists, so there is nothing queued for a store to
  // land in the middle of and no stream to order it on.
  for (size_t i = 0; i < n; ++i) publish_base(i, nullptr);
}

// ---- the SSD tier, made dispatchable ------------------------------------------------------------
//
// Return the address a tile carries for an expert whose bytes are only on disk: a pread plus the
// pool's own transform into a staging slot, after which the expert is byte-for-byte a pooled one
// and every stage downstream treats it as such.
//
// Blocking, deliberately. The alternative is issuing a layer whose weights have not arrived, and
// there is no correct version of that.
//
// `pool_`, not `pool_mut_`: staging only reads the pool (fetch and stage_base are both const), and
// pool_mut_ is set by placement_engine_init, which returns early under the static strategy. Gating
// on it made the SSD tier dispatchable under `heat` and fatal under `static`.
const uint8_t* StaticPlacement::ssd_stage(uint32_t layer, uint32_t expert, size_t dv) {
  if (!pool_ || !pool_dev_ || !impl_) return nullptr;
  Impl& I = *impl_;
  if (I.stage.empty()) return nullptr;
  const uint64_t idx = (uint64_t)layer * n_expert_ + expert;
  auto addr = [&](uint64_t s) {
    return pool_dev_ + (size_t)(pool_->stage_base(s) - pool_->base()) +
           (tp_parts_ > 1 ? (size_t)dv * shard_.stride : 0);
  };

  // Within ONE dispatch the same expert is asked for by both cards — the slot holds both shards and
  // only the offset differs. Answering from the claim list is deduplication, not caching: it never
  // outlives the dispatch and it never decides where an expert lives.
  for (uint32_t s : I.stage_claimed)
    if (I.stage[s].idx == idx) return addr(s);

  // `claimed` must never be reused: a tile in the dispatch being built points at it, its launches
  // have not been issued, so there is no event to wait on and no instant at which overwriting it
  // would be safe. Treating it as free produces fluent output with a different md5 every run.
  int64_t pick = -1;
  for (size_t s = 0; s < I.stage.size() && pick < 0; ++s) {
    if (I.stage[s].claimed) continue;
    if (!I.stage[s].armed) { pick = (int64_t)s; continue; }
    bool done = true;
    for (size_t i = 0; i < devs_.size() && done; ++i)
      if (I.stage[s].ev[i] && hipEventQuery(I.stage[s].ev[i]) != hipSuccess) done = false;
    if (done) { I.stage[s].armed = false; pick = (int64_t)s; }
  }
  // Nothing retired yet: wait on the first slot an earlier dispatch still owns.
  if (pick < 0)
    for (size_t s = 0; s < I.stage.size() && pick < 0; ++s) {
      if (I.stage[s].claimed || !I.stage[s].armed) continue;
      for (size_t i = 0; i < devs_.size(); ++i)
        if (I.stage[s].ev[i])
          AFF_BLOCK_AT(aff::BlockSite::SsdRing, (void)hipEventSynchronize(I.stage[s].ev[i]));
      I.stage[s].armed = false;
      pick = (int64_t)s;
    }
  if (pick < 0) {
    // Every slot belongs to the layer being built. Waiting cannot help, because the launches that
    // would release them are behind this call. Decline; the caller leaves the pair unhandled and
    // the model reports it by name rather than reading bytes that are being overwritten.
    aff::ui::err("ssd tier: layer %u needs more than %zu staged experts at once. Raise the staging "
                 "ring or the host pool so fewer experts live on disk.\n",
                 layer, I.stage.size());
    return nullptr;
  }

  uint8_t* dst = pool_->stage_base((uint64_t)pick);
  if (!dst) return nullptr;
  if (I.stage_raw.size() < pool_->stride()) I.stage_raw.resize(pool_->stride());
  I.stage[pick].idx = UINT64_MAX;
  // The pread AND the colgroup transform, both on the dispatch thread. Counted as one event because
  // that is how the caller experiences it: the layer does not proceed until both are done.
  {
    bool ok = true;
    AFF_BLOCK_AT(aff::BlockSite::SsdRead, {
      ok = pool_->fetch(layer, expert, I.stage_raw.data());
      if (ok) pool_tf(this, layer, I.stage_raw.data(), dst);
    });
    if (!ok) return nullptr;
  }

  I.stage[pick].idx = idx;
  I.stage[pick].claimed = true;
  I.stage[pick].armed = false;
  I.stage_claimed.push_back((uint32_t)pick);
  ++estats_.ssd_staged;
  return addr((uint64_t)pick);
}

// Every slot this dispatch claimed is now behind the layer's launches. Recording here rather than
// freeing outright is what lets the GEMM keep streaming from a slot after the host has moved on.
void StaticPlacement::ssd_stage_arm(void* const* streams) {
  if (!impl_) return;
  Impl& I = *impl_;
  for (uint32_t s : I.stage_claimed) {
    for (size_t i = 0; i < devs_.size(); ++i) {
      if (!I.stage[s].ev[i] || !streams[i]) continue;
      if (hipSetDevice(devs_[i]) != hipSuccess) continue;
      (void)hipEventRecord(I.stage[s].ev[i], (hipStream_t)streams[i]);
    }
    I.stage[s].armed = true;
    I.stage[s].claimed = false;
  }
  I.stage_claimed.clear();
}

// How much of the card a hybrid dispatch may occupy, as a FRACTION of whatever the device reports.
// See the comment at the stream's creation for why it is not the whole card: the zero-copy tiles
// are bus-bound, so every CU past what keeps the link busy is taken from the resident GEMM beside
// them.
static constexpr double kHybridCuFrac = 0.25;

// How many times the mover issues each shard copy. 1 always, unless AFF_MOVER_DUP says otherwise —
// a measurement knob for what the mover's link traffic actually costs, read once. See its use.
static const uint32_t kMoverDup = [] {
  const char* v = std::getenv("AFF_MOVER_DUP");
  const uint32_t n = v ? (uint32_t)std::strtoul(v, nullptr, 10) : 1u;
  return n ? n : 1u;
}();

// ...and WHICH DIRECTION it duplicates, because the two are not interchangeable and the difference
// decides a design. A promotion reads host RAM (H2D) on the same half of the link the hybrid tier
// streams misses on; a demotion writes it (D2H) on the half that tier leaves idle. If PCIe is
// full-duplex here then the demotion is nearly free and there is nothing to win by removing it; if
// the two share one budget, demotions are half the mover's cost. Duplicating both at once — which is
// what this knob did before — measures their sum and cannot tell the two cases apart.
//
// AFF_MOVER_DUP_DIR = both (default) | h2d | d2h
enum class MoverDupDir { Both, H2D, D2H };
static const MoverDupDir kMoverDupDir = [] {
  const char* v = std::getenv("AFF_MOVER_DUP_DIR");
  if (!v) return MoverDupDir::Both;
  if (!std::strcmp(v, "h2d")) return MoverDupDir::H2D;
  if (!std::strcmp(v, "d2h")) return MoverDupDir::D2H;
  return MoverDupDir::Both;
}();
static inline uint32_t mover_dup(bool h2d) {
  if (kMoverDupDir == MoverDupDir::Both) return kMoverDup;
  return ((kMoverDupDir == MoverDupDir::H2D) == h2d) ? kMoverDup : 1u;
}

static uint32_t hybrid_cus(int dev) {
  const uint32_t n = (uint32_t)(device_cus(dev) * kHybridCuFrac);
  return n < 1u ? 1u : n;
}

bool StaticPlacement::set_host_pool(const HostExpertPool* p) {
  pool_ = p;
  pool_registered_ = false;
  pool_dev_ = nullptr;
  if (!enabled_ || !impl_ || !p || !p->enabled() || !p->base()) return false;

  // Register the pool so the cards DMA out of it instead of through the driver's bounce buffer.
  // Registered, the cards read it at the same rate as memory allocated pinned, which is the wire
  // speed; unregistered it is a host memcpy wearing a costume and the whole tier is pointless. So a
  // failure here is fatal to the caller rather than a quietly slower mode.
  //
  // THERE IS A BAND WHERE THIS CALL LIES. Past a size that depends on what else the process has
  // mapped, hipHostRegister RETURNS SUCCESS and leaves the process unable to run: load finishes,
  // the banner prints, and the first forward either never returns or dies several stages inside
  // libamdhip64 on a refcount decrement through a null handle. Above that band it fails cleanly
  // with "invalid argument". A standalone probe cannot find the edge — bench/pinlimit.hip registers
  // 52 GiB with a stream, 64 events and a copy and reports everything fine.
  //
  // There is no size ceiling here: HostExpertPool::init clamps the pool against MemAvailable minus
  // what this process already holds, which is what a hardcoded limit was standing in for. What is
  // left is to let hipHostRegister answer for itself, and to fail here loudly rather than three
  // stages later.

  const auto t0 = std::chrono::steady_clock::now();
  const hipError_t err = hipHostRegister(const_cast<uint8_t*>(p->base()), p->bytes(),
                                         hipHostRegisterDefault);
  if (err != hipSuccess) {
    aff::ui::err("expert streaming off: hipHostRegister(%.1f GiB) failed: %s\n",
                 (double)p->bytes() / (1 << 30), hipGetErrorString(err));
    return false;
  }
  const double reg_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  // The hybrid stream and its four events, one set per card. Non-blocking so a miss can never be
  // ordered behind the legacy default stream by accident.
  //
  // AND CU-MASKED — but not for the reason a stalled-block argument would give. A zero-copy tile
  // alongside a resident GEMM inflates it several times over, and narrowing the mask does not change
  // that at all: four CUs cost what sixty-four do. So it is the memory system rather than occupancy.
  // Sustaining the link needs tens of KB of reads in flight, and a PCIe miss holds its L2
  // miss-status entry for microseconds where a VRAM miss holds it for nanoseconds — exhausting a
  // resource every other miss on the card needs, wherever the requesting blocks live. The copy
  // engine does not do this.
  //
  // What the mask buys is the TAIL. The two arms differ in VARIANCE far more than in centre: a
  // bounded grid is schedulable, an unbounded one occasionally is not. Compare them on the median,
  // never on the best of a few runs, which calls them tied.
  {
    Impl& I2 = *impl_;
    I2.hyb.assign(devs_.size(), nullptr);
    const std::array<hipEvent_t, Impl::kEvSlots> zero{};
    I2.ev_q.assign(devs_.size(), zero);
    I2.ev_gu.assign(devs_.size(), zero);
    I2.ev_sw.assign(devs_.size(), zero);
    I2.ev_dn.assign(devs_.size(), zero);
    I2.ev_seq.assign(devs_.size(), 0ull);
    for (size_t i = 0; i < devs_.size(); ++i) {
      if (hipSetDevice(devs_[i]) != hipSuccess) { I2.hyb[i] = nullptr; continue; }
      // The low `cus` of however many the device has. The mask is sized from the CU count rather
      // than fixed at two words, so a part with more than 64 does not silently get a mask covering
      // only its first 64. Falls back to an unmasked stream where the runtime has no masking — that
      // is slower, not wrong, and saying so beats guessing later.
      const uint32_t total_cu = device_cus(devs_[i]);
      const uint32_t cus = hybrid_cus(devs_[i]);
      std::vector<uint32_t> mask((total_cu + 31u) / 32u, 0u);
      for (uint32_t b = 0; b < cus; ++b) mask[b >> 5] |= 1u << (b & 31u);
      // SAY WHICH ONE HAPPENED. Two traces at the same nominal budget disagreed by a factor of two
      // on the hybrid stream's rate, which is the signature of an A/B whose flag never fired: a
      // silent fallback to an unmasked stream is indistinguishable from a mask that did nothing.
      const hipError_t merr =
          hipExtStreamCreateWithCUMask(&I2.hyb[i], (uint32_t)mask.size(), mask.data());
      if (merr != hipSuccess &&
          hipStreamCreateWithFlags(&I2.hyb[i], hipStreamNonBlocking) != hipSuccess) {
        I2.hyb[i] = nullptr;
        continue;
      }
      char cumsg[64];
      std::snprintf(cumsg, sizeof cumsg, "CU-masked to the low %u of %u", cus, total_cu);
      aff::ui::err("hybrid stream card %d: %s\n", devs_[i],
                   merr == hipSuccess ? cumsg
                                      : hipGetErrorString(merr));
      std::array<hipEvent_t, Impl::kEvSlots>* rings[4] = {&I2.ev_q[i], &I2.ev_gu[i], &I2.ev_sw[i],
                                                          &I2.ev_dn[i]};
      hipEvent_t* evs[4 * Impl::kEvSlots];
      for (uint32_t a = 0; a < 4; ++a)
        for (uint32_t b = 0; b < Impl::kEvSlots; ++b) evs[a * Impl::kEvSlots + b] = &(*rings[a])[b];
      for (hipEvent_t* e : evs)
        if (hipEventCreateWithFlags(e, hipEventDisableTiming) != hipSuccess) {
          (void)hipStreamDestroy(I2.hyb[i]);
          I2.hyb[i] = nullptr;
          break;
        }
    }
  }

  // The address the CARDS use for the same bytes. On ROCm it comes back equal to the host pointer,
  // and asking anyway is what makes that a checked fact — a hybrid tile carries this pointer into a
  // kernel, and an address that is merely plausible would fault or, worse, read something.
  {
    void* dp = nullptr;
    if (hipHostGetDevicePointer(&dp, const_cast<uint8_t*>(p->base()), 0) == hipSuccess && dp) {
      pool_dev_ = (uint8_t*)dp;
    } else {
      aff::ui::err("hybrid tier off: hipHostGetDevicePointer on the pool failed\n");
    }
  }

  // First touch is not the cause of prefill's first-chunk stall: a kernel touching one byte of every
  // pool page from each card, placed here, costs nothing and leaves the stalls unchanged, so
  // hipHostRegister leaves nothing lazy behind. Also ruled out, each by its own control: the mover,
  // the routed arena, and huge pages.

  // The SSD staging ring, one event set a card. Sized by the pool; zero slots simply means an SSD
  // expert has nowhere to land and the dispatch reports it rather than reading uninitialised bytes.
  {
    Impl& I3 = *impl_;
    I3.stage.assign(p->stage_slots(), Impl::StageSlot{});
    I3.stage_claimed.clear();
    I3.stage_raw.resize(p->stride());
    for (auto& s : I3.stage)
      for (size_t i = 0; i < devs_.size(); ++i) {
        if (hipSetDevice(devs_[i]) != hipSuccess) continue;
        (void)hipEventCreateWithFlags(&s.ev[i], hipEventDisableTiming);
      }
    if (!I3.stage.empty())
      aff::ui::err("ssd tier: %llu staging slots (%.1f MiB) for demand reads\n",
                   (unsigned long long)I3.stage.size(),
                   (double)I3.stage.size() * p->out_stride() / (1 << 20));
  }

  pool_registered_ = true;
  // Both tiers exist now, so every expert has an address. Idempotent, and the load path calls it too
  // for the case where no pool is attached at all.
  publish_all();
  aff::ui::err("expert streaming: %.1f GiB pool registered in %.2f s, read in place by the "
               "GEMM over PCIe\n", (double)p->bytes() / (1 << 30), reg_s);
  return true;
}

bool StaticPlacement::init(const AffReader* aff, const Codebook& cb, double budget_gib,
                       uint32_t n_layer, uint32_t n_expert, uint32_t batch_tok, uint32_t k,
                       std::string* err) {
  aff_ = aff;
  cb_ = cb;
  n_layer_ = n_layer;
  n_expert_ = n_expert;
  device_of_.assign((size_t)n_layer * n_expert, -1);
  offset_of_.assign((size_t)n_layer * n_expert, 0);
  stats_.total_experts = (uint64_t)n_layer * n_expert;

  const auto t_start = std::chrono::steady_clock::now();
  // Already filled in when the caller handed over the dense path's rank plan, which is the only way
  // to be sure a slab lands on the card whose rank will read it. Enumerating here is the fallback
  // for a standalone caller.
  if (devs_.empty()) devs_ = gfx12_devices();
  if (devs_.empty()) {
    if (err) *err = "no gfx12 device — expert cache disabled, CPU only";
    return false;
  }
  // The per-rank tables are kMaxRanks wide (see dense_gpu.h), so iterating over more devices than
  // that reads off the end of them. The extra cards are dropped here rather than half-used further
  // down.
  if (devs_.size() > kMaxRanks) {
    aff::ui::err("found %zu supported GPUs; this build's tables hold %zu. Ignoring the rest — set "
                 "HIP_VISIBLE_DEVICES to choose which.\n", devs_.size(), kMaxRanks);
    devs_.resize(kMaxRanks);
  }

  // The event rings are indexed by layer within a block, so a model with more layers than slots
  // would have layer L+kEvSlots reset the record layer L is still waiting on. That is not a crash:
  // it is the compute stream reading the hybrid stream's weights before they land, which reads as
  // fluent output with a different hash every run. Refuse instead.
  if (n_layer > Impl::kEvSlots) {
    if (err) *err = "model has " + std::to_string(n_layer) + " layers; the hybrid stream's event "
                    "rings hold " + std::to_string(Impl::kEvSlots) + ". Raise kEvSlots.";
    return false;
  }

  const AffLayerDesc& L0 = aff->layer(0);

  impl_ = new Impl();
  Impl& I = *impl_;
  // The slab holds each expert VERBATIM, at the container's own stride: an expert is already
  // contiguous on disk, so it is one pread and one H2D and the sub-offsets come from the
  // container's descriptors. A separate device layout meant 58,500 synchronous nine-plane copies
  // for a full load, each faulting its source in, with no disk read overlapping a PCIe transfer.
  I.stride = L0.stride;

  const uint32_t hidden = (uint32_t)L0.gate.cols;
  const uint32_t inter = (uint32_t)L0.gate.rows;

  // ---- the batched routed arena comes FIRST, and it is allocated rather than estimated ---------
  //
  // Placement is greedy: it takes free VRAM minus a constant, so everything allocated after it lives
  // on that constant — and the routed arena, the only such thing, scales with the prefill chunk.
  // Three estimates failed before this: a flat constant, the constant plus a measured split share
  // (how the cards split is a property of where placement put each expert, not of the model), and
  // the constant plus the provable bound (grow_w8a8 reallocates per layer as `taken` climbs, so the
  // peak is the final size plus what the freed predecessors fragmented). All three surfaced as
  // hipErrorOutOfMemory from an unrelated launch much later, because a failed hipMalloc only sets
  // the sticky error and this path declines the work.
  //
  // So: allocate the worst case here, once, before anyone measures free VRAM. `taken <= nb_tok*k`
  // because a token has k picks and each lands on one card. It costs roughly twice the typical
  // arena, against never reallocating and failing HERE rather than mid-forward.
  if (batch_tok && k) {
    const uint64_t bound = (uint64_t)batch_tok * k;
    for (size_t i = 0; i < devs_.size(); ++i) {
      AFF_HIP_CHECK(hipSetDevice(devs_[i]));
      // n_tiles is bounded by the slot count, and its arena is ~0.1% of the rest; no need to be
      // clever about how few experts 12288 slots can actually spread over.
      if (!grow_w8a8(i, (uint32_t)bound, (uint32_t)bound, batch_tok, hidden, inter, batch_tok)) {
        if (err) *err = "no VRAM for the batched routed arena at this prefill chunk";
        delete impl_; impl_ = nullptr;
        return false;
      }
    }
  }

  // How many experts fit per card, after leaving room for activations and scratch.
  //
  // PER CARD, not the minimum across cards. The cards are not symmetric: with `--tp-dense 1` the
  // whole dense set lives on card 0 and card 1 holds nothing but experts, so taking the minimum
  // sizes both slabs to the smaller card and leaves gigabytes of card 1 — hundreds of experts —
  // allocated and unused.
  // ---- hold a SHARD of every resident expert on every card, not the whole of a fraction ----
  //
  // Decided here, before the slab is sized, because it decides the slab's byte layout. It does NOT
  // change how many experts are resident: a shard is 1/N of an expert plus `down`'s row scales, the
  // one plane every card must keep whole, so `min(cap/shard)` and `sum(cap/whole)` come out within
  // 0.1% of each other. What changes is that every card then does 1/N of every routed expert
  // instead of all of some of them, which is worth hundreds of microseconds a token at the MoE
  // all-reduce — that costs the max over cards rather than the mean.
  //
  // Residency is unaffected in kind as well as in count — an expert is resident or it is not, and
  // a non-resident one still streams whole to one card on the path it always used.
  //
  // The rank count is offered to expert_shard_layout and it decides: gate/up need `rows % parts`
  // and down needs whole column groups AND a shard still 512 wide, which the model's 2048-wide
  // intermediate satisfies up to four ways and not eight. A count it rejects leaves whole experts,
  // which is correct and only less balanced — and, at more than one card, costs the placement
  // engine, which needs a slot to mean the same thing on every card.
  if (devs_.size() > 1 &&
      expert_shard_layout(aff->layer(0), (uint32_t)devs_.size(), &shard_)) {
    tp_parts_ = (uint32_t)devs_.size();
  } else {
    shard_ = ExpertShardLayout{};
  }
  const uint64_t slot_stride = this->slot_stride();

  std::vector<uint64_t> per_gpu(devs_.size(), 0);
  uint64_t total_slots = 0;
  for (size_t i = 0; i < devs_.size(); ++i) {
    AFF_HIP_CHECK(hipSetDevice(devs_[i]));
    size_t freeb = 0, totalb = 0;
    AFF_HIP_CHECK(hipMemGetInfo(&freeb, &totalb));
    const uint64_t want = (uint64_t)(budget_gib * 1073741824.0);
    // What the slab must not take. Two things: the engine's own allocations after this point,
    // which are small, and allocator slack — free VRAM that hipMemGetInfo reports but that cannot
    // come back as one contiguous block this size. The second dominates, and it is a property of
    // the driver, so this is measured per machine rather than derived.
    //
    // A FRACTION of the card rather than a byte count: the slack scales with the size of the block
    // being requested, and this binary should not assume which card it is on.
    constexpr double kDriverHeadroomFrac = 0.008;
    const uint64_t keep = headroom_mib_ > 0.0 ? (uint64_t)(headroom_mib_ * 1048576.0)
                                              : (uint64_t)((double)totalb * kDriverHeadroomFrac);
    const uint64_t cap = std::min<uint64_t>(want, freeb > keep ? freeb - keep : 0);
    per_gpu[i] = cap / slot_stride;
    // A shard slot is claimed on EVERY card at once, so the resident count is the smallest card's,
    // not the sum. With one card it is the same statement.
    if (tp_parts_ > 1) total_slots = i ? std::min(total_slots, per_gpu[i]) : per_gpu[i];
    else                total_slots += per_gpu[i];
  }
  if (tp_parts_ > 1) {
    for (size_t i = 0; i < devs_.size(); ++i) per_gpu[i] = total_slots;
    // The part count BY NUMBER, not "TP=2": it is the rank count now, and a banner that names the
    // wrong split is how a measurement gets attributed to a layout that did not run.
    aff::ui::err("routed experts: %u-way shards, gate/up %u of %u rows, down %u of %u cols, "
                 "%.2f MiB a shard, %llu experts a card\n",
                 tp_parts_, shard_.rows[0], (unsigned)aff->layer(0).gate.rows, shard_.cols[2],
                 (unsigned)aff->layer(0).down.cols, shard_.stride / 1048576.0,
                 (unsigned long long)total_slots);
  }

  if (total_slots == 0) {
    if (err) *err = "no VRAM available for the expert cache";
    delete impl_; impl_ = nullptr;
    return false;
  }

  // Rank experts by the container's popularity profile, hottest first. With no profile the order
  // is arbitrary but the mechanism is identical — measurement says the profile is barely better
  // than arbitrary anyway.
  const float* prof = aff->profile();
  std::vector<uint32_t> order((size_t)n_layer * n_expert);
  std::iota(order.begin(), order.end(), 0u);
  if (prof)
    std::stable_sort(order.begin(), order.end(),
                     [&](uint32_t a, uint32_t b) { return prof[a] > prof[b]; });

  const uint64_t n_resident = std::min<uint64_t>(total_slots, order.size());

  // ---- the wire format, latched from the container before any expert kernel runs ---------------
  //
  // Every layer shares one format by construction (aff-quantize writes the same geom for all of
  // them), so layer 0's gate answers for the file. A 4-wide codebook goes to the device ALREADY
  // PACKED as E4M3 quads — the GEMM cannot afford to build 4096 entries from floats in every block
  // — and `cb_` is exactly those bytes decoded, so re-packing here is the identity and cannot lose
  // anything. The pair formats want their floats and get them unchanged.
  const uint32_t wire = aff::expert_wire_format_for_quant((uint32_t)aff->layer(0).gate.quant);
  aff::expert_set_wire_format(wire);
  std::vector<uint32_t> cb_packed;
  if (cb_.pair == 4) cb_packed = pack_codebook_e4m3_quads(cb_);
  const void* cb_src = cb_.pair == 4 ? (const void*)cb_packed.data() : (const void*)cb_.v.data();
  const size_t cb_bytes = cb_.pair == 4 ? cb_packed.size() * 4 : cb_.v.size() * 4;
  if (!cb_bytes) { if (err) *err = "container codebook is empty"; return false; }

  // The decode kernel's precondition, checked once, loudly. e4m3_decode_norm drops the subnormal
  // branch because project_codebook_e4m3 clamps every entry to [2^-6, 1.875] — stored exponent
  // 1..7 — which is also what lets the block scale be an ADD to that field. A table that violated
  // it would decode to 2^-7 * 1.MMM instead of MMM * 2^-9: a wrong number, not a NaN, and nothing
  // downstream would notice. 8192 bytes, once, at load.
  if (cb_.pair == 4) {
    for (size_t i = 0; i < cb_packed.size(); ++i) {
      const uint32_t q = cb_packed[i];
      for (int k = 0; k < 4; ++k) {
        const uint32_t ex = ((q >> (8 * k)) >> 3) & 0x0Fu;
        if (ex == 0u || ex > 7u) {
          if (err) *err = "codebook entry " + std::to_string(i) + " byte " + std::to_string(k) +
                          " has E4M3 exponent " + std::to_string(ex) +
                          ", outside the 1..7 the expert decode assumes";
          return false;
        }
      }
    }
  }

  I.slab.resize(devs_.size());
  I.d_cb.resize(devs_.size());
  I.stream.resize(devs_.size());

  for (size_t i = 0; i < devs_.size(); ++i) {
    AFF_HIP_CHECK(hipSetDevice(devs_[i]));
    // Tail padding. The expert kernel assembles each 40-bit code window from a 64-bit read, so the
    // final window of the final expert legitimately reads a few bytes past its own data. Within an
    // expert that lands in the container's own alignment slack; at the END of the slab it ran off
    // the allocation and faulted the GPU outright.
    //
    // Plus the shadow slots, which hold no expert and exist so a promotion never DMAs over a live
    // one. They are extra ALLOCATION, not fewer residents: `total_slots` above is untouched, so
    // turning the engine on does not cost a single expert of hit rate. What it costs is a slice of
    // the free VRAM the load banner reports, which is why that banner is printed on every run.
    AFF_HIP_CHECK(hipMalloc(&I.slab[i], (per_gpu[i] + shadow_slots_) * slot_stride + 4096));
    AFF_HIP_CHECK(hipMalloc(&I.d_cb[i], cb_bytes));
    AFF_HIP_CHECK(hipMemcpy(I.d_cb[i], cb_src, cb_bytes, hipMemcpyHostToDevice));
    AFF_HIP_CHECK(hipStreamCreate(&I.stream[i]));
  }
  {
    // The slabs exist, so a resident expert's address is now computable. Pooled ones are still 0
    // until set_host_pool runs; publish_all() there fills them in.
    std::string perr;
    if (!plc_alloc(&perr)) {
      aff::ui::err("fatal: %s\n", perr.c_str());
      return false;
    }
  }

  // Upload, pipelined. Reader threads pread an expert into a pinned staging buffer and hand it to an
  // async H2D on their own stream, so the disk stays at depth and PCIe overlaps it. The floor is the
  // disk, which is an order of magnitude slower than the bus.
  //
  // Alternating cards keeps both slabs equally full, so a token's six experts tend to split across
  // both rather than piling onto one.
  {
    const int kReaders = 8;
    // Colgroup-major weight planes: the device kernels read the codes several times faster this way
    // than row-major — see expert_gemm_wmma_kernel's TR parameter. The reorder happens HERE, once, on
    // the pinned buffer between the pread and the H2D, so nothing on the device ever sees the
    // row-major form and the container on disk is untouched.
    //
    // The host cost is a strided gather over one plane — it stays in cache, and the load is
    // disk-bound anyway. Doing it on the device instead would need a staging buffer
    // per reader per card, and that VRAM is already committed to the slab by this point.
    const bool tr = aff::expert_planes_are_colgroup_major();
    std::vector<uint64_t> next_slot(devs_.size(), 0);
    std::vector<std::pair<uint32_t, std::pair<size_t, uint64_t>>> jobs;   // idx -> (dev, slot)
    jobs.reserve(n_resident);
    if (tp_parts_ > 1) {
      // Every expert to EVERY card, at the same slot on each, so `offset_of_` stays one number.
      // `dev` is meaningless here and is left at 0; the reader shards to all cards in one pass.
      for (uint64_t k = 0; k < n_resident; ++k) jobs.push_back({order[k], {0, k}});
    } else {
    size_t rr = 0;
    for (uint64_t k = 0; k < n_resident; ++k) {
      // Alternate, but skip a card that is already full instead of dropping the expert. The old
      // `continue` silently discarded every job whose turn landed on the full card, so the larger
      // card stopped filling the moment the smaller one did.
      size_t tries = 0;
      while (next_slot[rr] >= per_gpu[rr] && tries < devs_.size()) {
        rr = (rr + 1) % devs_.size(); ++tries;
      }
      if (tries == devs_.size()) break;
      jobs.push_back({order[k], {rr, next_slot[rr]++}});
      rr = (rr + 1) % devs_.size();
    }
    }

    std::atomic<size_t> cursor{0};
    const size_t tp2_check = 0;
    std::atomic<size_t> tp2_bad{0}, tp2_seen{0};
    // The load loop used to (void)-cast every HIP return. That is the exact shape of the failure
    // that cost this project the most time: with the slab sized to nearly all of the card, the
    // FIRST thing to fail is not the slab's own hipMalloc but one of these — a reader's stream,
    // its pinned buffer, or a copy — and a swallowed failure here leaves the slot holding whatever
    // the allocator had there before. No crash: fluent token soup, wrong expert weights, DSpark
    // acceptance pinned at 0.0%. So every return is counted, the first message is kept, and any
    // failure is FATAL after the join. AFF_SLAB_VERIFY=N additionally reads every Nth expert back
    // off the card and memcmps it against the pinned source it was copied from — that is the
    // decisive test between "the bytes never arrived" and "the bytes are there but read wrong".
    std::atomic<size_t> hip_errs{0};
    std::atomic<size_t> vrf_n{0}, vrf_bad{0};
    std::mutex err_mu;
    std::string first_err;
    auto note = [&](hipError_t e, const char* what, size_t dev, uint64_t slot) {
      if (e == hipSuccess) return;
      if (!hip_errs.fetch_add(1)) {
        std::lock_guard<std::mutex> g(err_mu);
        first_err = std::string(what) + ": " + hipGetErrorString(e) + " (dev " +
                    std::to_string(dev) + ", slot " + std::to_string(slot) + ")";
      }
    };
    size_t verify_every = 0;
    if (const char* v = getenv("AFF_SLAB_VERIFY")) verify_every = (size_t)atoll(v);
    std::vector<std::thread> th;
    for (int t = 0; t < kReaders; ++t) {
      th.emplace_back([&, t] {
        // One pinned buffer and one stream per reader. Pinned is what lets the H2D be a real DMA
        // rather than a staged copy through a bounce buffer.
        std::vector<void*> pin(devs_.size(), nullptr);
        std::vector<hipStream_t> st(devs_.size());
        std::vector<uint8_t> scratch;      // one plane's worth, reused; grows once per reader
        for (size_t d = 0; d < devs_.size(); ++d) {
          note(hipSetDevice(devs_[d]), "hipSetDevice", d, 0);
          note(hipHostMalloc(&pin[d], I.stride, hipHostMallocDefault), "hipHostMalloc(pin)", d, 0);
          note(hipStreamCreate(&st[d]), "hipStreamCreate", d, 0);
        }
        std::vector<uint8_t> vrf;    // readback target, grown on first use
        // Under TP=2 the pread lands in pin[0] and each card's shard is cut out of it into its own
        // pinned buffer, so the H2D is still a DMA from pinned memory and still one per card.
        std::vector<void*> pin_sh(tp_parts_ > 1 ? devs_.size() : 0, nullptr);
        for (size_t d = 0; d < pin_sh.size(); ++d) {
          note(hipSetDevice(devs_[d]), "hipSetDevice", d, 0);
          note(hipHostMalloc(&pin_sh[d], shard_.stride, hipHostMallocDefault),
               "hipHostMalloc(pin_sh)", d, 0);
        }
        for (;;) {
          const size_t j = cursor.fetch_add(1);
          if (j >= jobs.size()) break;
          const uint32_t idx = jobs[j].first;
          const size_t d = jobs[j].second.first;
          const uint64_t slot = jobs[j].second.second;
          const uint32_t layer = idx / n_expert, e = idx % n_expert;
          const uint64_t off = aff->expert_file_offset(layer, e);

          uint64_t got = 0;
          while (got < I.stride) {
            const ssize_t r = ::pread(aff->fd(), (uint8_t*)pin[d] + got, I.stride - got,
                                      (off_t)(off + got));
            if (r <= 0) break;
            got += (uint64_t)r;
          }
          // AFF_EXPERT_TP2_CHECK=N: verify the first N experts a reader shards, against the bytes
          // as they came off disk. Costs a copy of the raw expert, so it is off by default.
          std::vector<uint8_t> raw;
          if (tp2_check && tp2_seen.fetch_add(1) < tp2_check)
            raw.assign((const uint8_t*)pin[d], (const uint8_t*)pin[d] + I.stride);
          if (tr) aff::expert_transpose_planes_host((uint8_t*)pin[d], aff->layer(layer), scratch);
          if (tp_parts_ > 1) {
            for (size_t p = 0; p < devs_.size(); ++p) {
              aff::expert_shard_host((const uint8_t*)pin[d], aff->layer(layer), shard_, tp_parts_,
                                     (uint32_t)p, (uint8_t*)pin_sh[p]);
              if (!raw.empty()) {
                const size_t bad = aff::expert_shard_check_host(
                    raw.data(), aff->layer(layer), shard_, tp_parts_, (uint32_t)p,
                    (const uint8_t*)pin_sh[p]);
                if (bad) tp2_bad.fetch_add(bad);
              }
              note(hipSetDevice(devs_[p]), "hipSetDevice", p, slot);
              note(hipMemcpyAsync(I.slab[p] + slot * shard_.stride, pin_sh[p], shard_.stride,
                                  hipMemcpyHostToDevice, st[p]), "hipMemcpyAsync(shard)", p, slot);
              note(hipStreamSynchronize(st[p]), "hipStreamSynchronize(shard)", p, slot);
              if (verify_every && j % verify_every == 0) {
                if (vrf.size() < shard_.stride) vrf.resize(shard_.stride);
                note(hipMemcpy(vrf.data(), I.slab[p] + slot * shard_.stride, shard_.stride,
                               hipMemcpyDeviceToHost), "hipMemcpy(verify)", p, slot);
                vrf_n.fetch_add(1);
                if (memcmp(vrf.data(), pin_sh[p], shard_.stride)) vrf_bad.fetch_add(1);
              }
            }
            device_of_[idx] = 0;                    // resident; "which card" no longer applies
            offset_of_[idx] = slot * shard_.stride;
          } else {
            note(hipSetDevice(devs_[d]), "hipSetDevice", d, slot);
            note(hipMemcpyAsync(I.slab[d] + slot * I.stride, pin[d], I.stride,
                                hipMemcpyHostToDevice, st[d]), "hipMemcpyAsync", d, slot);
            note(hipStreamSynchronize(st[d]), "hipStreamSynchronize", d, slot);
            if (verify_every && j % verify_every == 0) {
              if (vrf.size() < I.stride) vrf.resize(I.stride);
              note(hipMemcpy(vrf.data(), I.slab[d] + slot * I.stride, I.stride,
                             hipMemcpyDeviceToHost), "hipMemcpy(verify)", d, slot);
              vrf_n.fetch_add(1);
              if (memcmp(vrf.data(), pin[d], I.stride)) vrf_bad.fetch_add(1);
            }
            device_of_[idx] = (int)d;
            offset_of_[idx] = slot * I.stride;
          }
        }
        for (size_t d = 0; d < devs_.size(); ++d) {
          (void)hipSetDevice(devs_[d]);
          (void)hipHostFree(pin[d]);
          (void)hipStreamDestroy(st[d]);
        }
        for (size_t d = 0; d < pin_sh.size(); ++d) {
          (void)hipSetDevice(devs_[d]);
          (void)hipHostFree(pin_sh[d]);
        }
      });
    }
    for (auto& x : th) x.join();
    if (verify_every)
      aff::ui::err("slab verify: %zu experts read back, %zu mismatched\n",
                   vrf_n.load(), vrf_bad.load());
    if (hip_errs.load() || vrf_bad.load()) {
      // Refuse to run degraded — the alternative is a server that answers fluently and wrongly.
      aff::ui::err("fatal: slab load: %zu swallowed HIP failures, %zu verify mismatches — first: "
                   "%s. The slab is not what the file holds; refusing to serve. Free VRAM was "
                   "probably exhausted by the slab itself — raise --gpu-headroom-mib.\n",
                   hip_errs.load(), vrf_bad.load(),
                   first_err.empty() ? "(verify only)" : first_err.c_str());
      return false;
    }
    if (tp2_check)
      aff::ui::err("TP=2 shard check: %llu experts, %llu mismatching spans\n",
                   (unsigned long long)std::min<size_t>(tp2_check, tp2_seen.load()),
                   (unsigned long long)tp2_bad.load());
    stats_.resident_experts = jobs.size();
    // Every resident expert now has a slab address. set_host_pool calls this again once the pooled
    // ones do too; running it twice costs 11008 stores and removes an ordering dependency between
    // the two, which is the better trade.
    publish_all();
    // Those bytes now live in VRAM; drop the host copies so the page cache can hold the complement.
    for (const auto& j : jobs)
      aff->advise_expert(j.first / n_expert, j.first % n_expert, /*want=*/false);
  }

  // The complement is NOT warmed into the page cache. Nothing reads those experts from the mapping:
  // the pool preads straight into its own registered pages and drops the cache copy behind it
  // (HostExpertPool::init), and an SSD-tier expert preads on demand. Warming them would make the
  // kernel hold a second copy of the pool's bytes during the exact window the pool is being
  // populated — gigabytes of peak RAM the pool then could not have.

  {
    size_t fb = 0, tb = 0;
    (void)hipSetDevice(devs_[0]);
    (void)hipMemGetInfo(&fb, &tb);
    // Printed on every run on purpose. Three separate bugs have been "the engine ran out of VRAM
    // somewhere after placement", and each time the first question was this number.
    aff::ui::err("placement headroom: %.0f MiB free on card 0 after the slabs\n",
                 fb / 1048576.0);
  }
  stats_.vram_used_gib = (double)(total_slots * I.stride) / 1073741824.0;
  stats_.load_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
  enabled_ = true;
  return true;
}

// Build the DeviceExpert triple for one resident expert. The slab holds it verbatim at the
// container's own stride, so the container's descriptors address it directly.
static inline DeviceExpert make_expert(uint8_t* slab_base, const AffLayerDesc& L) {
  auto mk = [&](const AffMatrixDesc& d) {
    DeviceMatrix m;
    m.data = slab_base + d.data_off;
    m.scales = slab_base + d.scale_off;
    m.rscales = (const uint16_t*)(slab_base + d.rscale_off);
    m.rows = (uint32_t)d.rows; m.cols = (uint32_t)d.cols;
    return m;
  };
  DeviceExpert e;
  e.gate = mk(L.gate); e.up = mk(L.up); e.down = mk(L.down);
  return e;
}

// The same, for a slab that holds SHARDS. The offsets and shapes come from the shard layout rather
// than the container's, and nothing else about a descriptor changes — a shard is just a smaller
// matrix in the identical colgroup-major form.
static inline DeviceExpert make_shard(uint8_t* base, const ExpertShardLayout& S) {
  DeviceExpert e;
  DeviceMatrix* m[3] = {&e.gate, &e.up, &e.down};
  for (int i = 0; i < 3; ++i) {
    m[i]->data = base + S.data_off[i];
    m[i]->scales = base + S.scale_off[i];
    m[i]->rscales = (const uint16_t*)(base + S.rscale_off[i]);
    m[i]->rows = S.rows[i];
    m[i]->cols = S.cols[i];
  }
  return e;
}

// EVERY CARD RUNS ITS OWN RESIDENT SUBSET, INTO ITS OWN BLOCK. There is no home card here any
// more, and no peer copy: the caller holds a block on each card and reduces them once per sublayer
// anyway (DenseOps::block_reduce), so an expert that lives on card 1 can accumulate into card 1's
// block and be summed in for free. What that deleted, in order: the peer copy of the activation,
// the double-buffered partial, the two-phase join events, and the entire class of ordering bug they
// existed to manage.
//
// It also fixes the batched path's worst limitation without touching it. Taking only the experts
// resident on a HOME card leaves every other pair to the per-token fallback, and with residency
// split across both cards that is most of the routed pairs on the slow path.
// Byte offsets of the five uploads inside one pinned slot, in the order they are filled. Sized
// from the arena caps rather than from `taken`, so a slot laid out once holds any layer.
namespace {
struct PinLayout { size_t tok, wt, ord, run, tiles, total; };
inline PinLayout pin_layout(uint32_t cap_tok, uint32_t cap_slot, uint32_t cap_nb,
                            uint32_t cap_tile) {
  auto a8 = [](size_t n) { return (n + 7u) & ~(size_t)7u; };
  PinLayout L{};
  L.tok   = 0;
  L.wt    = a8(L.tok + (size_t)cap_tok * sizeof(uint32_t));
  L.ord   = a8(L.wt + (size_t)cap_tok * sizeof(float));
  L.run   = a8(L.ord + (size_t)cap_slot * sizeof(uint32_t));
  L.tiles = a8(L.run + ((size_t)cap_nb + 1) * sizeof(uint32_t));
  L.total = a8(L.tiles + (size_t)cap_tile * 3 * sizeof(ExpertTile));
  return L;
}
}  // namespace

bool StaticPlacement::grow_w8a8(size_t dv, uint32_t taken, uint32_t n_tiles, uint32_t nb_tok,
                                uint32_t hidden, uint32_t inter, uint32_t nb_stride) {
  // ---- A GROWTH FREES MEMORY THE CARDS ARE STILL READING -----------------------------------------
  //
  // Every buffer below is named by tiles and launch arguments of dispatches already in flight, and
  // `hipFree` does not wait for them. That was safe only while the router's drain held the host to a
  // layer of lookahead — it is the exact shape of "removing a sync exposes every buffer it guarded",
  // and it is deterministic rather than flaky because a growth happens at a fixed point in the run.
  //
  // A growth is RARE: the caps are set by the first prefill chunk and are monotone, so this costs a
  // handful of synchronises for the life of the process. Draining here is not on the hot path in any
  // sense — reaching it at all means the shape changed.
  bool grew = false;
  auto grow = [&](void** p, size_t bytes) {
    if (*p) {
      if (!grew) {
        grew = true;
        for (size_t i = 0; i < devs_.size(); ++i) {
          if (hipSetDevice(devs_[i]) != hipSuccess) continue;
          AFF_BLOCK_AT(aff::BlockSite::ArenaGrow, (void)hipDeviceSynchronize());
        }
        AFF_HIP_CHECK(hipSetDevice(devs_[dv]));
      }
      AFF_HIP_CHECK(hipFree(*p));
    }
    *p = nullptr;
    return hipMalloc(p, bytes) == hipSuccess;
  };
  // Slack on every growth: layers differ in how many slots land on a card, and a realloc a layer
  // would serialise the stream on the free.
  if (bcap_slot_[dv] < taken) {
    const uint32_t cap = taken + (taken >> 1) + 64;
    // bf16: gate/up are read once by the SwiGLU, which requantises to E4M3 immediately, and `down`
    // once by the scatter, which sums into an f32 block. Both consumers run close to their DRAM
    // roofline over hundreds of MB a call, so the element width is most of what they cost.
    if (!grow(&bg_[dv], (size_t)inter * cap * sizeof(uint16_t))) return false;
    if (!grow(&bu_[dv], (size_t)inter * cap * sizeof(uint16_t))) return false;
    if (!grow(&ba8_[dv], (size_t)cap * inter)) return false;
    if (!grow(&bas_[dv], (size_t)cap * (inter / 128u) * sizeof(float))) return false;
    if (!grow(&bydn_[dv], (size_t)hidden * cap * sizeof(uint16_t))) return false;
    bcap_slot_[dv] = cap;
  }
  if (bcap_tok_[dv] < taken * 2u) {
    const uint32_t cap = taken * 2u + taken + 64;
    bcap_tok_[dv] = cap;
  }
  if (bcap_tile_[dv] < n_tiles) {
    const uint32_t cap = n_tiles + (n_tiles >> 1) + 16;
    bcap_tile_[dv] = cap;
  }
  if (bcap_nb_[dv] < nb_stride) {
    if (!grow(&bq8_[dv], (size_t)nb_stride * hidden)) return false;
    if (!grow(&bqs_[dv], (size_t)nb_stride * (hidden / 128u) * sizeof(float))) return false;
    bcap_nb_[dv] = nb_stride;
  }
  // ONE device buffer for the five uploads, laid out exactly like the pinned slot, so the layer's
  // whole argument block is a single copy. ROCm runs a copyBuffer KERNEL for small host-to-device
  // copies whether the source is pinned or not, and at five a layer those blits and the dispatch
  // gaps behind them are a real share of a cycle. Five become one.
  {
    const PinLayout PL =
        pin_layout(bcap_tok_[dv], bcap_slot_[dv], bcap_nb_[dv], bcap_tile_[dv]);
    if (PL.total > bpack_n_[dv]) {
      if (!grow(&bpack_[dv], PL.total)) return false;
      bpack_n_[dv] = PL.total;
    }
    uint8_t* const P = (uint8_t*)bpack_[dv];
    btok_[dv] = P + PL.tok; bwt_[dv] = P + PL.wt; bord_[dv] = P + PL.ord;
    brun_[dv] = P + PL.run; btiles_[dv] = P + PL.tiles;
  }
  // The pinned ring, sized from the caps just settled so a slot always holds a whole layer.
  {
    const size_t need =
        pin_layout(bcap_tok_[dv], bcap_slot_[dv], bcap_nb_[dv], bcap_tile_[dv]).total;
    if (need > bpin_stride_[dv]) {
      if (bpin_[dv]) {
        AFF_BLOCK_AT(aff::BlockSite::ArenaGrow, (void)hipDeviceSynchronize());
        AFF_HIP_CHECK(hipHostFree(bpin_[dv]));
      }
      bpin_[dv] = nullptr;
      if (hipHostMalloc(&bpin_[dv], need * kPinSlots, hipHostMallocDefault) != hipSuccess)
        return false;
      bpin_stride_[dv] = need;
      for (uint32_t s = 0; s < kPinSlots; ++s)
        if (!bpin_ev_[dv][s] &&
            hipEventCreateWithFlags((hipEvent_t*)&bpin_ev_[dv][s], hipEventDisableTiming) !=
                hipSuccess)
          return false;
    }
  }
  (void)nb_tok;
  return true;
}

uint32_t StaticPlacement::run_batch_device(uint32_t layer, const uint32_t* sel, const float* w,
                                           uint32_t nb_tok, uint32_t k, const float* const* d_x,
                                           float* const* d_block, uint32_t nb_stride,
                                           uint32_t hidden, uint32_t inter, float limit,
                                           void* const* streams, uint8_t* handled,
                                           const RouteDev* route) {
  if (!enabled_ || !nb_tok || !k || !d_x || !d_block || !handled || !streams) return 0;
  if (layer >= n_layer_) return 0;
  Impl& I = *impl_;
  const size_t R = devs_.size();
  if (R > 8) return 0;
  lead_gate(streams, nb_tok, R);
  inter = shard_inter(inter);

  const size_t total = (size_t)nb_tok * k;
  // ---- scratch, allocated ONCE ----------------------------------------------------------------
  //
  // These were locals, so a call heap-allocated a dozen vectors and zeroed three 256-entry planes,
  // 43 layers a block on two cards. A speculative block routes at most nb_tok*k = 36 (token,
  // expert) pairs, so everything here is tiny and the bookkeeping was the cost rather than the work.
  // `uniq` is what makes the n_expert-wide sweeps go away: the DISTINCT experts this card takes,
  // ascending, which is the order `start`, the tile list and every prefix sum below already walk.
  // counts/start/fill are touched only at those indices and cleared through the same list, so
  // nothing here reads an entry a previous layer wrote.
  // `mark` is a DEDICATED membership plane and deliberately not one of the three below. `fill` was
  // the obvious candidate — it is already per-expert scratch and the prefix loop zeroes it — but it
  // is zeroed there and then used as the per-expert FILL CURSOR, so it ends every call holding
  // counts[e] and is not clean on entry to the next one. Reusing it made every expert any previous
  // layer had touched read as already-claimed, which took the dispatch down to a handful of slots
  // and pushed the rest onto the host path, where it was merely very slow rather than wrong.
  static thread_local std::vector<uint32_t> counts, start, fill, uniq, mark;
  static thread_local std::vector<uint32_t> grp, h_tok, h_ord, h_run;
  static thread_local std::vector<float> gwt;
  static thread_local std::vector<ExpertTile> h_tiles;
  if (counts.size() != n_expert_) {
    counts.assign(n_expert_, 0u); start.assign(n_expert_, 0u); fill.assign(n_expert_, 0u);
    mark.assign(n_expert_, 0u);
  }
  uint32_t taken_all = 0;

  // ---- which build runs -----------------------------------------------------------------------
  //
  // Every condition here is a real impossibility rather than a preference, and each is checked once
  // per call because each can change between calls: the mirror is built on the first router call,
  // the placement table on the first dispatch, and the route trace and the census are per run but
  // ask for host-side facts the device build does not produce.
  //
  // The run-level half, latched once. Everything in it is either constant for the run or a
  // misconfiguration; nothing here may change between layers, because `device_dispatch` is what the
  // ROUTER consulted to decide whether to read the selection back at all.
  //
  // ---- TWO KINDS OF "THE DEVICE CANNOT BUILD THIS", AND THEY GO IN DIFFERENT PLACES -------------
  //
  // A MISCONFIGURATION is a run that cannot proceed correctly either way: no placement table, tp=1,
  // more experts or a wider k than the plan kernel was compiled for. Those belong in `ok`, and `ok`
  // is fatal, because a silent fallback is how an A/B measures one arm twice.
  //
  // A DIAGNOSTIC OPT-OUT is a run that has deliberately asked for something the device build does
  // not produce, and for which the host build is the correct answer rather than a fallback. Those
  // belong in the LATCH GUARD, so `plan_ready_` simply never sets and every layer takes the host
  // path.
  //
  // `--route-trace` is one, and it was in the wrong place. It records the router's selection and
  // the experts placement had to stream, both of which are host-side facts the device build does not
  // produce — so it must keep the host build, exactly as the plan check does. Sitting inside `ok` it
  // could only ever ABORT, and the abort was unreachable for as long as the SSD tier was non-empty:
  // an addressless expert forced the host build for its layer anyway, so `route[0].sel` was
  // populated by a layer that never consulted the latch. Emptying the tier made every layer
  // device-built and turned `--route-trace` from working into an immediate abort — which is how a
  // flag that had been fine for months broke without being touched.
  //
  // That matters beyond the flag: the route trace is the INPUT to `aff-predict`, so the entire
  // offline placement-design loop went with it. A diagnostic that cannot run is not a diagnostic,
  // and this one is the only instrument that can rank a placement policy without the card.
  if (!plan_ready_ && !route_trace().on() && route && route[0].sel &&
      (R < 2 || route[R - 1].sel)) {
    const bool ok = !plc_.empty() && tp_parts_ > 1 &&
                    n_expert_ <= kMoePlanMaxExperts && k <= kMoePlanMaxK;
    if (!ok) {
      aff::ui::fatal("fatal: the dispatch cannot be built on the card (table %d, tp %u, "
                   "experts %u, k %u) and there is no host-built alternative to run "
                   "instead. Refusing rather than falling back: a silent fallback is how an A/B "
                   "measures one arm twice.\n",
                   plc_.empty() ? 0 : 1, tp_parts_, n_expert_, k);
      std::abort();
    }
    plan_ready_ = true;
  }
  // The LDS bound is the one run-level condition that depends on a per-call number, so it is checked
  // per call and it is fatal too. At n_expert 256 it does not bind below ~6100 tokens a chunk, which
  // is three times the widest prefill chunk the engine takes.
  if (plan_ready_ && moe_plan_lds_bytes(n_expert_, nb_tok) > 64u * 1024u) {
    aff::ui::fatal("fatal: the plan kernel needs %zu bytes of LDS for a %u-token chunk and "
                 "the limit is 65536.\n", moe_plan_lds_bytes(n_expert_, nb_tok), nb_tok);
    std::abort();
  }
  const bool plan_on = plan_layer(layer);
  // A refusal that is never announced is how a configuration that never engaged becomes the noise
  // floor. Said once, with the reason, and separately for the two kinds: the SSD layers are an
  // expected, priced share of the run, and everything else is a misconfiguration.
  //
  // Counted as well as said, because "it printed no banner" is not evidence that it ran — this
  // project has twice been fooled by a flag whose absence of complaint meant it had never fired.
  // `aff-info`-style totals go out at exit beside the placement engine's.
  if (plan_on) ++plan_dispatch_; else ++host_dispatch_;
  {
    const bool ssd_here = layer < ssd_layer_.size() && ssd_layer_[layer];
    if (!plan_on && !ssd_here && !said_host_) {
      said_host_ = true;
      // No device-readable selection reached this placement, so the dispatch keeps the host build.
      // Not fatal, because the host build is still correct and still bit-identical — but
      // `host_dispatch_` counts every one and the exit line prints it, so a run that quietly fell
      // back says so with a number.
      aff::ui::err("moe dispatch: no device-readable selection reached this placement "
                   "(tag %c, route %s), so it keeps the host build and its drain.\n",
                   route_tag_, !route ? "absent" : route[0].sel ? "present" : "null");
    }
    if (ssd_here && !said_ssd_) {
      said_ssd_ = true;
      aff::ui::err("moe dispatch: %u of %u layers hold an expert with no address and keep "
                   "the host build and its drain — a kernel cannot read one off disk. This is a "
                   "property of how full both tiers are, not of the routing.\n",
                   ssd_layers_hot_, n_layer_);
    }
  }
  // What the launches read, whichever build filled it in.
  struct Groups { ExpertTile* base[8][3] = {}; uint32_t n[8] = {}, max_nt[8] = {}; };

  for (size_t dv = 0; dv < R; ++dv) {
    if (!d_x[dv] || !d_block[dv] || !streams[dv]) continue;
    // Opened here rather than at the top of the function so a card that is handed nothing does not
    // register a call. Stopped by hand just before the uploads.
    trace::XScope _xbucket((int)dv, 0, nullptr);
    // Bucket by expert. This is the entire "permutation" — an index list per expert, with the
    // activation left exactly where it is. vLLM's fused MoE reaches the same conclusion from the
    // other direction: never move the tokens, only the indices that name them.
    // ---- H6..H0: the layer's experts, split by WHERE THE BYTES ARE ----------------------------
    //
    // Two classes, and the only difference between them is the base pointer a tile carries. A
    // resident expert's shard is in this card's slab; a missing one's is in the registered host
    // pool, and the tile names it directly — the GEMM streams those 4.32 MiB over PCIe as it
    // computes rather than waiting for a copy to land. Everything downstream (the arena, the
    // SwiGLU, the requantise, the scatter) sees one undifferentiated slot list.
    //
    // `uniq` stays ascending WITHIN each class rather than across both, because the classes are
    // dispatched as two separate grids on two streams and the ordering that matters is the one
    // `start` prefix-sums in. Resident first: they are the ones that must not wait.
    // Pass 2 is the SSD tier: an expert with neither a slab offset nor a pool slot gets read off
    // disk into a staging slot, after which it is a pooled expert in every respect. It is last
    // because it is the only pass that blocks, so nothing that could have been claimed cheaply is
    // left behind it.
    uniq.clear();
    uint32_t taken = 0, n_res = 0;
    const bool hybrid = hybrid_ready();
    // Where the two builds meet. `base[group][plane]` and `n`/`max_nt` are everything the launches
    // need, and they are the only thing either build produces that the launches read.
    Groups GRP{};
    if (plan_on) {
      // The device build takes every VALID slot, which is every slot: the router pads a short token
      // with an out-of-range expert, and the engine's own invariant says no live dispatch reaches
      // the SSD tier. `status.taken` records what it actually took and the check below reads it.
      taken = (uint32_t)total;
    }
    if (!plan_on)
    for (int pass = 0; pass < 3; ++pass) {
      const size_t first = uniq.size();
      for (size_t i = 0; i < total; ++i) {
        const uint32_t e = sel[i];
        if (e >= n_expert_ || mark[e]) continue;
        // Sharded, "which card" is not a question for a resident expert: every card holds a piece
        // of every one of them and must run all of them, which is exactly what balances the two.
        // For a MISSING one the same is true for a different reason — the pool holds both shards,
        // so each card streams its own half and the two links work in parallel.
        // FROM THE PLACEMENT TABLE'S HOST MIRROR, not from `mine_for`/`pool_shard_dev`. Both of
        // those walk host structures per slot — `pool_->resident()` is a lookup into the pool's map
        // — and the card that walks them first takes the cache misses the other then finds warm, so
        // this scope costs one card several times what it costs the other. The mirror answers both
        // questions with one indexed load of a word this card will read again a few lines below.
        const uint64_t pv = plc_at(dv, layer, e);
        const bool res = (pv & kPlcResident) != 0;
        bool want;
        if (pass == 0) want = res;
        else if (pass == 1) want = hybrid && !res && pv != 0;
        // Only the SSD tier still asks a function, because a staged expert has no address until the
        // staging ring gives it one.
        else want = hybrid && !res && pv == 0 && ssd_stage(layer, e, dv) != nullptr;
        if (!want) continue;
        mark[e] = 1u;
        uniq.push_back(e);
      }
      std::sort(uniq.begin() + first, uniq.end());
      if (pass == 0) n_res = (uint32_t)uniq.size();
    }
    // ---- the routing trace, card 0 only ---------------------------------------------------------
    //
    // Card 0 and card 1 select the same experts — a resident expert has a shard on every card at
    // the same slab offset, and a missing one has both shards in the same pool slot — so recording
    // per card would write every dispatch twice with identical contents. The assert that keeps that
    // true is the tile count: the routed trace prints it per card and they match.
    //
    // Placed HERE, after the two membership passes and before anything is launched, because both
    // halves of the record have to describe the same instant: `sel` is what the router chose and
    // `uniq[n_res..]` is what placement had failed to have ready for it.
    if (!plan_on) {
      if (dv == 0 && route_trace().on())
        route_trace().dispatch(route_tag_, layer, nb_tok, sel, (uint32_t)total, uniq.data() + n_res,
                               (uint32_t)(uniq.size() - n_res));
      if (uniq.empty()) continue;
      for (size_t i = 0; i < total; ++i) {
        const uint32_t e = sel[i];
        if (e < n_expert_ && mark[e]) { ++counts[e]; ++taken; }
      }
      if (!taken) { for (uint32_t e : uniq) { mark[e] = 0u; counts[e] = 0u; } continue; }
    }

    AFF_HIP_CHECK(hipSetDevice(devs_[dv]));
    hipStream_t st = (hipStream_t)streams[dv];
    // The dispatch must not read the selection before it exists on this card. Rank 0's own stream
    // already orders it; every other rank waits on the event rank 0 records after the peer push.
    if (plan_on && route[dv].ready)
      AFF_HIP_CHECK(hipStreamWaitEvent(st, (hipEvent_t)route[dv].ready, 0));

    // One host-side build and ONE upload for the whole layer, rather than a copy per expert: at 256
    // experts x 43 layers a per-expert copy would be ~11k synchronous transfers a chunk.
    if (!plan_on) {
      grp.resize(taken);
      gwt.resize(taken);
      { uint32_t acc = 0; for (uint32_t e : uniq) { start[e] = acc; fill[e] = 0u; acc += counts[e]; } }
      for (size_t i = 0; i < total; ++i) {
        const uint32_t e = sel[i];
        // `counts[e]`, not `mine_for`: membership now covers both classes, and `fill` has just been
        // reset to serve as the per-expert fill cursor so it can no longer answer the question.
        if (e >= n_expert_ || !counts[e]) continue;
        const uint32_t at = start[e] + fill[e]++;
        grp[at] = (uint32_t)(i / k);                // which token column of the chunk
        gwt[at] = w[i];
        handled[i] = 1;
      }
    } else {
      // The device build takes every slot the router named, so `handled` is answered here rather
      // than per slot. The caller's only use of it is the `any_host` abort, and an unhandled slot is
      // now reported by `status.bad` instead — which is the same fact read off the card.
      for (size_t i = 0; i < total; ++i) handled[i] = 1;
    }

    // ---- the whole layer in seven launches ---------------------------------------------------
    //
    // What this replaced issued four kernels per token slice per expert — hundreds of launches a
    // layer a card, tens of thousands for a prompt — on a GEMM running at a few percent of the
    // matrix core. `expert_gemm_wmma_hip` takes every expert in the layer as one grid, so both
    // problems had the same answer.
    //
    // Every stage works on ONE arena whose columns are the card's routed (token, expert) slots,
    // already grouped by expert in `grp`/`gwt` above. A tile's `ld` is the arena width, so a tile
    // writes its own columns of a shared buffer and the SwiGLU, the requantise and the scatter are
    // single whole-arena launches rather than one per tile.
    //
    //   quantise x        [hidden][nb_stride] f32  ->  [nb_stride][hidden] E4M3 + scales
    //   gate, up          E4M3 x 2.875-bpw weights ->  [inter][taken] f32   (one launch a bucket)
    //   swiglu+requant    ->  [taken][inter] E4M3 + scales, router weight folded in
    //   down              ->  [hidden][taken] f32
    //   scatter           ->  += into the chunk's [hidden][nb_stride] accumulator
    //
    // Only the first quantise is a separate pass over memory, and it is there only until the norm
    // that produces `d_x` emits E4M3 itself — at which point the activation is fp8 end to end and
    // the dense GEMMs read the same buffer.
    {
      uint32_t n_miss_tiles = 0;
      if (plan_on) {
        // ---- THE DEVICE BUILD ---------------------------------------------------------------------
        //
        // One kernel replaces everything the `else` below does. It reads the selection off this
        // card and the placement table off this card, and writes the slot arena's token list, the
        // router weights, ord/run and the whole ExpertTile array — so nothing about this layer's
        // shape reaches the host, and nothing about it has to come back.
        //
        // The tile array is laid out [plane][class][tile_cap] rather than packed, because a host
        // that has not seen the routing cannot know where a packed class would start. `tile_cap` is
        // a function of nb_tok, k and n_expert, all of which it does know.
        const uint32_t cap = moe_plan_tile_cap(nb_tok, k, n_expert_);
        if (!grow_w8a8(dv, taken, 8u * cap, nb_tok, hidden, inter, nb_stride)) {
          aff::ui::fatal("fatal: the routed arena would not grow to %u slots and %u tiles on "
                       "card %zu\n", taken, 8u * cap, dv);
          std::abort();
        }
        if (!plan_scratch(dv, (uint32_t)total)) {
          aff::ui::fatal("fatal: the plan scratch would not allocate on card %zu\n", dv);
          std::abort();
        }
        // The placement engine's only input, and the last thing the drain was still feeding.
        if (engine_on_ && !act_ring_ && !act_alloc()) {
          aff::ui::fatal("fatal: the activation ring would not allocate — the placement "
                       "engine cannot be fed without it and running it off a stale selection is "
                       "worse than not running it.\n");
          std::abort();
        }
        auto* const Tg = (ExpertTile*)btiles_[dv];
        MoePlanArgs A;
        A.sel = route[dv].sel; A.wt = route[dv].wt; A.plc = plc_[dv];
        A.layer = layer; A.nb_tok = nb_tok; A.k = k; A.n_expert = n_expert_;
        A.tok = (uint32_t*)btok_[dv]; A.wt_out = (float*)bwt_[dv];
        A.ord = (uint32_t*)bord_[dv]; A.run = (uint32_t*)brun_[dv];
        A.slot_col = plan_col_[dv]; A.tiles = Tg; A.counts = plan_cnt_[dv];
        A.status = plan_st_[dv];
        // One card writes the ring. Both select the same experts — a resident expert has a shard on
        // every card at the same slab offset, and a missing one has both shards in the same pool
        // slot — so a ring per card would carry every dispatch twice, identically.
        A.act = (dv == 0) ? act_dev_ : nullptr;
        A.shard = shard_;
        A.bg = (uint16_t*)bg_[dv]; A.bu = (uint16_t*)bu_[dv]; A.bydn = (uint16_t*)bydn_[dv];
        A.tile_cap = cap;
        if (!moe_plan_hip(A, st)) {
          aff::ui::fatal("fatal: the plan kernel refused nb_tok %u k %u experts %u on card "
                       "%zu\n", nb_tok, k, n_expert_, dv);
          std::abort();
        }
        // The grid the host can size: the worst case per REGION, with the surplus padded to nt = 0.
        // An over-launch of several times costs nothing here — see the note on the early return in
        // expert_gemm_wmma_kernel.
        //
        // Only the regions `nb_tok` makes reachable, and each with its own bucket's template. At
        // decode nothing is wider than nb_tok <= 16, so exactly one bucket a class is launched and
        // the layer costs the same six launches it does today. Launching all four unconditionally
        // would take that to twenty-four.
        const uint32_t last = moe_plan_last_group(nb_tok);
        for (uint32_t cls = 0; cls < 2u; ++cls)
          for (uint32_t b = 0; b <= last; ++b) {
            const uint32_t g = cls * 4u + b;
            GRP.n[g] = cap;
            GRP.max_nt[g] = moe_plan_group_max_nt(g);
            for (uint32_t pl = 0; pl < 3u; ++pl)
              GRP.base[g][pl] = Tg + (size_t)(pl * 8u + g) * cap;
          }
        n_miss_tiles = cap;
        _xbucket.stop();
        trace::g_xt.taken[dv] += taken;
        trace::g_xt.tiles[dv] += (uint64_t)(last + 1u) * 2u * cap;
      } else {
      // Tiles, split at the kernel's widest token block and ordered by the NT template they need.
      // A tile of 8 tokens launched under NT=8 would stage and multiply 128 columns to use 8, so
      // the buckets exist to keep each launch's template matched to its widest member.
      struct Tile { uint32_t e, off, nt; uint32_t miss; };
      static thread_local std::vector<Tile> tl;
      tl.clear();
      for (uint32_t j = 0; j < uniq.size(); ++j) {
        const uint32_t e = uniq[j], m = j < n_res ? 0u : 1u;
        for (uint32_t off = 0; off < counts[e]; off += kExpertWmmaTok)
          tl.push_back({e, start[e] + off, std::min<uint32_t>(kExpertWmmaTok, counts[e] - off), m});
      }
      // `counts` and `mark` are scratch shared across cards and calls, and `uniq` is the only list
      // of what was touched. Cleared HERE, before anything below can return early.
      for (uint32_t e : uniq) { counts[e] = 0u; mark[e] = 0u; }
      auto bucket = [](uint32_t nt) { return nt <= 16u ? 0u : nt <= 32u ? 1u : nt <= 64u ? 2u : 3u; };
      // EIGHT groups, not four: the NT template a launch is instantiated for, crossed with which
      // stream it belongs on. Sorting by (miss, bucket) keeps the two classes contiguous, so the
      // resident groups can go out on the compute stream and the missing ones on the hybrid stream
      // without either list being a scatter.
      auto grp8 = [&](const Tile& t) { return t.miss * 4u + bucket(t.nt); };
      std::stable_sort(tl.begin(), tl.end(),
                       [&](const Tile& a, const Tile& b) { return grp8(a) < grp8(b); });
      uint32_t bbeg[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, bmax[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      { uint32_t b = 0;
        for (uint32_t i = 0; i < tl.size(); ++i) {
          while (b < grp8(tl[i])) bbeg[++b] = i;
          bmax[b] = std::max(bmax[b], tl[i].nt);
        }
        while (b < 8) bbeg[++b] = (uint32_t)tl.size(); }
      const uint32_t nt_tiles = (uint32_t)tl.size();
      n_miss_tiles = nt_tiles - bbeg[4];

      // Slots sorted by token, with an nb_tok+1 prefix, so the scatter reduces in a FIXED order
      // rather than racing atomics into the accumulator.
      h_ord.resize(taken);
      h_run.assign((size_t)nb_tok + 1, 0u);
      for (uint32_t s = 0; s < taken; ++s) ++h_run[grp[s] + 1];
      for (uint32_t t = 0; t < nb_tok; ++t) h_run[t + 1] += h_run[t];
      { static thread_local std::vector<uint32_t> fillt;
        fillt.assign(h_run.begin(), h_run.end() - 1);
        for (uint32_t s = 0; s < taken; ++s) h_ord[fillt[grp[s]]++] = s; }

      // Token ids for the gate/up tiles, then 0..taken-1 for the down tiles, whose activation
      // arena is the SwiGLU output and so is indexed by slot rather than by token.
      h_tok.resize((size_t)taken * 2);
      std::memcpy(h_tok.data(), grp.data(), (size_t)taken * sizeof(uint32_t));
      for (uint32_t s = 0; s < taken; ++s) h_tok[taken + s] = s;

      // Should now be a no-op: init() sized these for nb_tok*k, which bounds `taken`. It stays
      // because the decode path reaches run_batch_device with chunks init() never saw. SAY SO if it
      // ever does allocate and fail — falling back silently is what turned an under-reserved
      // placement into `hipErrorOutOfMemory` reported from an unrelated matvec launch several
      // kernels later: a failed hipMalloc leaves the sticky error, this path just declines the
      // work, and the next AFF_HIP_CHECK anywhere in the engine is the one that dies.
      //
      if (!grow_w8a8(dv, taken, nt_tiles, nb_tok, hidden, inter, nb_stride)) {
        static bool said = false;
        if (!said) {
          said = true;
          size_t freeb = 0, totb = 0;
          (void)hipMemGetInfo(&freeb, &totb);
          aff::ui::err("batched routed arena FAILED to grow on card %zu (%u slots, %u tiles, chunk "
                       "%u, %.0f MiB free): the routed FFN is falling back to the host path.\n",
                       dv, taken, nt_tiles, nb_tok, freeb / 1048576.0);
        }
        return taken_all;
      }
      auto* Tg = (ExpertTile*)btiles_[dv];
      h_tiles.resize((size_t)nt_tiles * 3);
      for (uint32_t i = 0; i < nt_tiles; ++i) {
        // The ONLY thing that distinguishes a missing expert from a resident one: where its shard
        // is. Both `pool_shard_dev` and `ssd_stage` hand back a device-visible address inside the
        // registered host pool, laid out byte for byte as the slab would hold it, so `make_shard`
        // does not know or care which tier it came from. `ssd_stage` is a lookup here rather than a
        // read — pass 2 above already claimed the slot and filled it.
        // FROM THE PLACEMENT TABLE. `plc_[dv][idx]` already IS this address — the table is written
        // at the three points placement changes, so reading it here removes the last host-side
        // residency lookup from the dispatch and is the step that lets a kernel do this instead.
        //
        // The SSD tier is the one thing the table cannot answer: a staged expert's address is
        // allocated by the staging ring at the moment of the miss, so it does not exist until asked
        // for. It stays a call, and it stays the only one.
        const uint64_t pv = plc_at(dv, layer, tl[i].e);
        uint8_t* base = (uint8_t*)(uintptr_t)(pv & kPlcAddrMask);
        if (!base) base = const_cast<uint8_t*>(ssd_stage(layer, tl[i].e, dv));
        const DeviceExpert de =
            tp_parts_ > 1 ? make_shard(base, shard_) : make_expert(base, aff_->layer(layer));
        const uint32_t off = tl[i].off, nt = tl[i].nt;
        auto set = [&](ExpertTile& t, const DeviceMatrix& m, const uint32_t* tok, void* y) {
          t.data = m.data; t.fields = m.scales; t.rscales = m.rscales;
          t.tok = tok; t.y = y; t.nt = nt; t.ld = taken;
        };
        set(h_tiles[i], de.gate, (const uint32_t*)btok_[dv] + off, (uint16_t*)bg_[dv] + off);
        set(h_tiles[nt_tiles + i], de.up, (const uint32_t*)btok_[dv] + off, (uint16_t*)bu_[dv] + off);
        set(h_tiles[2 * nt_tiles + i], de.down,
            (const uint32_t*)btok_[dv] + taken + off, (uint16_t*)bydn_[dv] + off);
      }


      // The host build's groups: tiles are packed tight, each plane is `nt_tiles` apart, and a
      // group is a slice named by the prefix.
      for (uint32_t b = 0; b < 8; ++b) {
        GRP.n[b] = bbeg[b + 1] - bbeg[b];
        GRP.max_nt[b] = bmax[b];
        for (uint32_t pl = 0; pl < 3; ++pl) GRP.base[b][pl] = Tg + pl * nt_tiles + bbeg[b];
      }

      _xbucket.stop();
      trace::g_xt.taken[dv] += taken;
      trace::g_xt.tiles[dv] += nt_tiles;
      for (uint32_t b = 0; b < 4; ++b)
        trace::g_xt.padded[dv] += (uint64_t)(bbeg[b + 1] - bbeg[b]) * bmax[b];

      { AFF_XTRACE((int)dv, 1, st);
      // Through the pinned ring, and with NO stream synchronise afterwards. Out of the vectors
      // directly these were pageable, which means the driver stages each one itself and runs a
      // copyBuffer KERNEL on the compute stream to do it, and the call blocks the host — so the
      // code had to drain before it could reuse the vectors on the next card — hundreds of blits a
      // cycle a rank, plus the gaps between them, plus a second full drain per layer per card beside
      // the router's. From pinned memory it is a DMA that never touches the compute stream's kernel
      // slots.
      const uint32_t ps = bpin_next_[dv];
      bpin_next_[dv] = (ps + 1) % kPinSlots;
      AFF_BLOCK_AT(aff::BlockSite::PinRing,
                   AFF_HIP_CHECK(hipEventSynchronize((hipEvent_t)bpin_ev_[dv][ps])));
      const PinLayout PL = pin_layout(bcap_tok_[dv], bcap_slot_[dv], bcap_nb_[dv], bcap_tile_[dv]);
      uint8_t* const slot = (uint8_t*)bpin_[dv] + (size_t)ps * bpin_stride_[dv];
      const size_t n_tok = (size_t)taken * 2 * sizeof(uint32_t);
      const size_t n_wt = (size_t)taken * sizeof(float);
      const size_t n_ord = (size_t)taken * sizeof(uint32_t);
      const size_t n_run = ((size_t)nb_tok + 1) * sizeof(uint32_t);
      const size_t n_tiles_b = h_tiles.size() * sizeof(ExpertTile);
      std::memcpy(slot + PL.tok, h_tok.data(), n_tok);
      std::memcpy(slot + PL.wt, gwt.data(), n_wt);
      std::memcpy(slot + PL.ord, h_ord.data(), n_ord);
      std::memcpy(slot + PL.run, h_run.data(), n_run);
      std::memcpy(slot + PL.tiles, h_tiles.data(), n_tiles_b);
      // ONE copy of the whole slot, padding included: it is ~8 KiB against five dispatches.
      AFF_HIP_CHECK(hipMemcpyAsync(bpack_[dv], slot, PL.tiles + n_tiles_b, hipMemcpyHostToDevice,
                                   st));
      (void)n_tok; (void)n_wt; (void)n_ord; (void)n_run; (void)Tg;
      AFF_HIP_CHECK(hipEventRecord((hipEvent_t)bpin_ev_[dv][ps], st)); }
      }   // the host build ends here; both paths have filled GRP and stopped the bucket scope

      // ---- the layer, on two streams -----------------------------------------------------------
      //
      // The resident tiles read VRAM; the missing ones read the host pool over PCIe, which is more
      // than an order of magnitude slower. Those are different pieces of hardware, so the only way a
      // miss costs nothing is if the two run at the same time — which is what `hyb` is for. The
      // compute stream never waits on the bus until
      // it has to, and "has to" is exactly twice: before the SwiGLU, which reads every slot's gate
      // and up, and before the scatter, which reads every slot's down.
      //
      // Three events, all one-directional:
      //   ev_q   the quantised activation and the tile descriptors are up   (comp -> hyb)
      //   ev_gu  the missing experts' gate and up have landed               (hyb  -> comp)
      //   ev_sw  the SwiGLU output is requantised for every slot            (comp -> hyb)
      //   ev_dn  the missing experts' down has landed                       (hyb  -> comp)
      // and nothing else crosses. A miss count small enough to fit under the resident chain makes
      // every one of these already-satisfied on arrival, which is the whole design.
      const bool split = n_miss_tiles && I.hyb.size() > dv && I.hyb[dv];
      hipStream_t hyb = split ? I.hyb[dv] : st;
      // This dispatch's slot in each of the four rings. Advanced once per (card, layer) so a record
      // and the wait that reads it are the same object, and no other layer touches it.
      //
      // Gated on `split` because that is what says the rings EXIST: a placement with no hybrid
      // stream — the DSpark draft's, whose experts are all resident — skips the block in init() that
      // sizes them, and indexing an empty vector here segfaulted inside the draft.
      const uint32_t es = split ? (uint32_t)(I.ev_seq[dv]++ % Impl::kEvSlots) : 0u;
      { AFF_XTRACE((int)dv, 2, st);
      batch_quant_fp8_hip(d_x[dv], (uint8_t*)bq8_[dv], (float*)bqs_[dv], hidden, nb_stride, 0, st); }
      if (split) {
        AFF_HIP_CHECK(hipEventRecord(I.ev_q[dv][es], st));
        AFF_HIP_CHECK(hipStreamWaitEvent(hyb, I.ev_q[dv][es], 0));
      }
      // The missing tiles go out FIRST. They are the long pole by a factor of twenty, and a stream
      // that is issued second starts second.
      //
      // Both the host build and the device build reach this through `GRP`, which names each group's
      // three plane bases and its launch width. The two lay their tiles out differently — the host
      // packs them tight and slices with a prefix, the device puts each class in a fixed region so
      // the host can name the base without seeing the routing — and this is the ONE place where
      // that has to stop mattering, because this is where the bits get computed.
      { AFF_XTRACE((int)dv, 8, hyb);
      for (uint32_t b = 4; b < 8; ++b) {
        if (!GRP.n[b]) continue;
        expert_gemm_wmma_hip(GRP.base[b][0], GRP.n[b], GRP.max_nt[b], (const uint8_t*)bq8_[dv],
                             (const float*)bqs_[dv], I.d_cb[dv], inter, hidden, 1.0f, hyb, /*y_bf16=*/true);
        expert_gemm_wmma_hip(GRP.base[b][1], GRP.n[b], GRP.max_nt[b], (const uint8_t*)bq8_[dv],
                             (const float*)bqs_[dv], I.d_cb[dv], inter, hidden, 1.0f, hyb, /*y_bf16=*/true);
      } }
      { AFF_XTRACE((int)dv, 3, st);
      for (uint32_t b = 0; b < 4; ++b) {
        if (!GRP.n[b]) continue;
        expert_gemm_wmma_hip(GRP.base[b][0], GRP.n[b], GRP.max_nt[b], (const uint8_t*)bq8_[dv],
                             (const float*)bqs_[dv], I.d_cb[dv], inter, hidden, 1.0f, st, /*y_bf16=*/true);
        expert_gemm_wmma_hip(GRP.base[b][1], GRP.n[b], GRP.max_nt[b], (const uint8_t*)bq8_[dv],
                             (const float*)bqs_[dv], I.d_cb[dv], inter, hidden, 1.0f, st, /*y_bf16=*/true);
      } }
      if (split) {
        AFF_HIP_CHECK(hipEventRecord(I.ev_gu[dv][es], hyb));
        AFF_HIP_CHECK(hipStreamWaitEvent(st, I.ev_gu[dv][es], 0));
      }
      { AFF_XTRACE((int)dv, 4, st);
      expert_swiglu_quant_hip((const uint16_t*)bg_[dv], (const uint16_t*)bu_[dv],
                              (const float*)bwt_[dv], (uint8_t*)ba8_[dv], (float*)bas_[dv], inter,
                              taken, limit, st); }
      if (split) {
        AFF_HIP_CHECK(hipEventRecord(I.ev_sw[dv][es], st));
        AFF_HIP_CHECK(hipStreamWaitEvent(hyb, I.ev_sw[dv][es], 0));
      }
      { AFF_XTRACE((int)dv, 9, hyb);
      for (uint32_t b = 4; b < 8; ++b) {
        if (!GRP.n[b]) continue;
        expert_gemm_wmma_hip(GRP.base[b][2], GRP.n[b], GRP.max_nt[b], (const uint8_t*)ba8_[dv],
                             (const float*)bas_[dv], I.d_cb[dv], hidden, inter, 1.0f, hyb, /*y_bf16=*/true);
      } }
      { AFF_XTRACE((int)dv, 5, st);
      for (uint32_t b = 0; b < 4; ++b) {
        if (!GRP.n[b]) continue;
        expert_gemm_wmma_hip(GRP.base[b][2], GRP.n[b], GRP.max_nt[b], (const uint8_t*)ba8_[dv],
                             (const float*)bas_[dv], I.d_cb[dv], hidden, inter, 1.0f, st, /*y_bf16=*/true);
      } }
      if (split) {
        AFF_HIP_CHECK(hipEventRecord(I.ev_dn[dv][es], hyb));
        AFF_HIP_CHECK(hipStreamWaitEvent(st, I.ev_dn[dv][es], 0));
      }
      { AFF_XTRACE((int)dv, 6, st);
      expert_scatter_slots_hip((const uint16_t*)bydn_[dv], d_block[dv], (const uint32_t*)bord_[dv],
                               (const uint32_t*)brun_[dv], hidden, nb_tok, nb_stride, taken, st); }
      trace::g_xt.hyb_tiles[dv] += n_miss_tiles;

      taken_all += taken;
    }
  }

  // THE LAYER BOUNDARY, in the only sense that matters: every launch this layer will make has been
  // issued, so an event recorded on the compute stream from here is downstream of every kernel that
  // could still be reading a slab slot or a pool slot the mover wants to write. The staging ring
  // needs exactly that guarantee, for the same reason.
  ssd_stage_arm(streams);
  if (engine_on_) engine_tick(layer, sel, total, streams, plan_on);

  // ---- what the card says about the plan, read BEHIND the issue frontier -------------------------
  //
  // Two things are fatal and neither can be checked where it happens: a slot whose expert is on the
  // SSD tier, which the dispatch has no way to compute, and a class that overflowed `tile_cap`,
  // which silently drops tiles. Both are counters the kernel bumps and nobody reads on the hot path.
  //
  // Read every 512th call and WITHOUT a synchronise: whatever has landed by then is what gets seen,
  // and one stale read costs at most another 512 layers before the next. The point is that the
  // process stops rather than emitting wrong text for a whole run — not that it stops on the exact
  // layer. Checking per layer is what the drain was for.
  if (plan_on && (++plan_calls_ & 511u) == 0u) {
    for (size_t dv = 0; dv < R; ++dv) {
      if (!plan_st_[dv]) continue;
      MoePlanStatus S{};
      if (hipMemcpy(&S, plan_st_[dv], sizeof S, hipMemcpyDeviceToHost) != hipSuccess) continue;
      if (S.bad > plan_bad_ || S.over > plan_over_) {
        aff::ui::err("fatal: the device dispatch could not place %u routed (token, expert) slots "
                     "on card %zu (%u tile overflows). A slot with no address is on the SSD tier "
                     "and nothing can compute it — raise %s.\n", S.bad, dv, S.over,
                     // Which model, and therefore which knob: the draft's tier is sized at load and
                     // refuses to start with anything on disk, so `bad` on tag 'S' means the pool
                     // lost slots after that check rather than never having had them.
                     route_tag_ == 'S' ? "--host-pool-reserve-mib, or --dspark-placement vram"
                                       : "--host-pool-mib");
        std::abort();
      }
      if (S.taken != (uint32_t)total) {
        aff::ui::fatal("fatal: the device dispatch took %u of %zu slots on card %zu; the "
                     "arena width the host launched with is wrong.\n", S.taken, total, dv);
        std::abort();
      }
    }
  }

  return taken_all;
}

// ================================================================================================
// The placement engine
// ================================================================================================

bool StaticPlacement::placement_engine_init(HostExpertPool* pool,
                                            const PlacementEngineConfig& cfg, std::string* err) {
  auto fail = [&](const char* m) { if (err) *err = m; return false; };
  if (!enabled_ || !impl_) return fail("placement is not enabled");
  if (cfg.strategy == PlacementEngineConfig::Strategy::Static) {
    // Not an error and not an engine. The static arm is the control every heat measurement is
    // against, and it is exactly today's code path with nothing added to it — no shadow slots
    // touched, no stream created, no per-tick work. That is what makes the A/B a fair one.
    engine_on_ = false;
    return true;
  }
  // ---- the refusals, and each of them is a real impossibility rather than a preference ----------
  // A slab slot has to be layer- AND card-agnostic for the mover's bookkeeping to hold, which
  // needs every card to hold a piece of every resident expert. The TP split gives that at two
  // cards; ONE card gives it trivially, because the only card holds the whole of every resident
  // expert and "which card" was never a question. Anything else — three or more cards with whole
  // experts spread across them — genuinely does not, and is refused.
  if (tp_parts_ <= 1 && devs_.size() != 1)
    return fail("the placement engine needs either the TP=2 expert shard split or a single card: "
                "a slab slot is only layer- and card-agnostic when every card holds a piece of "
                "every resident expert");
  if (!pool || !pool->enabled() || !pool_registered_)
    return fail("the placement engine needs the registered host pool — a demoted expert's bytes "
                "have nowhere else to go");
  if (!shadow_slots_)
    return fail("no shadow slab slots reserved; call reserve_shadow_slots() before init()");
  // Read directly rather than through Model::force_resident_routing(), to keep this file off
  // model.h. It is the same switch — model.h owns what it MEANS, and what it means is the reason
  // this is fatal: the flag rewrites the router's top-k to whatever is resident, so with a
  // placement that moves, the generated text becomes a function of when a transfer happened to
  // land. Two runs of the same binary would produce different text and neither would be wrong.
  if (const char* fr = std::getenv("AFF_FORCE_RESIDENT"); fr && *fr && *fr != '0')
    return fail("AFF_FORCE_RESIDENT confines routing to whatever is resident, and the engine moves "
                "what is resident — together they make the generated text a function of transfer "
                "timing. Run one or the other");

  pool_mut_ = pool;
  ecfg_ = cfg;
  estats_ = {};
  Impl& I = *impl_;
  const size_t R = devs_.size();
  const uint64_t ss = shard_.stride;

  // The strategy's tier map starts as a mirror of what load produced, and `Ssd` is the default in
  // init() rather than an afterthought: an expert that is in neither table is on disk, and the
  // strategy has to know not to pick it.
  heat_.init(n_layer_, n_expert_, cfg.heat);
  resident_now_ = 0;
  uint64_t pooled = 0, ssd = 0;
  for (uint32_t l = 0; l < n_layer_; ++l)
    for (uint32_t e = 0; e < n_expert_; ++e) {
      const uint32_t idx = l * n_expert_ + e;
      if (device_of_[idx] >= 0) { heat_.set_tier(idx, ExpertTier::Vram); ++resident_now_; }
      else if (pool->slot_of(l, e) >= 0) { heat_.set_tier(idx, ExpertTier::Pool); ++pooled; }
      else { heat_.set_tier(idx, ExpertTier::Ssd); ++ssd; }
    }
  resident_target_ = resident_now_;
  heat_.resident_per_layer(&resident_at_load_);
  ever_moved_.assign((size_t)n_layer_ * n_expert_, 0u);

  // The shadow slots sit past the resident ones, which is exactly where init() allocated them.
  I.free_slab.clear();
  for (uint32_t s = 0; s < shadow_slots_; ++s) I.free_slab.push_back(resident_target_ + s);
  I.free_pool.clear();

  I.mv.assign(R, nullptr);
  I.ev_layer.assign(R, nullptr);
  for (size_t i = 0; i < R; ++i) {
    if (hipSetDevice(devs_[i]) != hipSuccess) return fail("hipSetDevice failed for the mover");
    if (hipStreamCreateWithFlags(&I.mv[i], hipStreamNonBlocking) != hipSuccess)
      return fail("could not create the mover stream");
    if (hipEventCreateWithFlags(&I.ev_layer[i], hipEventDisableTiming) != hipSuccess)
      return fail("could not create the mover's layer-boundary event");
  }

  // One record per shadow slot is the natural bound: a promotion holds a slab slot for its whole
  // life, so there can never be more in flight than there are shadows, and a demotion is gated by
  // the pool slots those promotions free.
  I.moves.assign(shadow_slots_, Impl::Move{});

  for (uint32_t m = 0; m < shadow_slots_; ++m) {
    for (size_t i = 0; i < R; ++i) {
      if (hipSetDevice(devs_[i]) != hipSuccess) return fail("hipSetDevice failed for a move event");
      if (hipEventCreateWithFlags(&I.moves[m].ev[i], hipEventDisableTiming) != hipSuccess)
        return fail("could not create a move event");
    }
  }


  engine_on_ = true;
  aff::ui::err(// The strategy BY NAME: a banner that says "heat" whatever is running is how a
               // measurement ends up attributed to the arm that did not produce it.
               "placement engine: %s, %u shadow slots, %u move%s a layer\n"
               "                  at load %llu in VRAM / %llu pooled / %llu on ssd\n",
               cfg.strategy == PlacementEngineConfig::Strategy::Demand ? "demand" : "heat",
               shadow_slots_, cfg.moves_per_tick, cfg.moves_per_tick == 1 ? "" : "s",
               (unsigned long long)resident_now_, (unsigned long long)pooled,
               (unsigned long long)ssd);
  return true;
}

void StaticPlacement::engine_tick(uint32_t layer, const uint32_t* sel, size_t total,
                                  void* const* streams, bool from_ring) {
  Impl& I = *impl_;
  ++estats_.ticks;

  // Whatever landed since the last layer, applied HERE and by this thread: the transfers are async,
  // the table edits are not.
  engine_retire(streams);

  // The DISTINCT experts, not the slots. A batched dispatch reads an expert's weights once however
  // many of the block's tokens chose it, so a repeat inside one block costs nothing and must not be
  // counted twice — the heat table has to rank what the engine PAYS for.
  //
  // ...but it is credited by HOW MANY of the block's tokens agreed, which is a different question
  // and is free here: `seen` was already a per-expert cell, so counting costs the same pass that
  // deduplicating did. See ExpertHeat::observe for what it buys.
  // FROM THE RING WHEN THERE IS ONE. The record the cards publish carries the same two arrays this
  // loop builds, so heat's input is unchanged — what changes is that it arrives without the host
  // ever having asked for the selection, which is the whole of why the dispatch used to drain.
  //
  // `act_drain` returns 0 whenever nothing new has landed, and then this tick observes nothing and
  // issues nothing. That is the correct behaviour and not a stall: there is no new information, so
  // there is no new decision to take.
  // WHICHEVER BUILT THIS DISPATCH IS WHERE THE OBSERVATION COMES FROM, and the choice has to be per
  // dispatch rather than per run: a layer holding an addressless expert keeps the host build, and
  // the plan kernel that would have written its record never ran. Keying on `act_ring_` existing
  // instead made those layers invisible to heat — a large fraction of the run, silently unobserved,
  // which reads as an engine that has simply stopped promoting.
  //
  // Interleaving the two sources is safe because `observe` touches one LAYER's row and decays only
  // that row, so a device-built layer arriving late cannot disturb a host-built layer's history.
  uint32_t rounds = 1;
  if (from_ring) {
    rounds = act_drain();
    if (!rounds) return;
  } else {
  static thread_local std::vector<uint32_t> seen, distinct, counts;
  if (seen.size() != n_expert_) seen.assign(n_expert_, 0u);
  distinct.clear();
  counts.clear();
  for (size_t i = 0; i < total; ++i) {
    const uint32_t e = sel[i];
    if (e >= n_expert_) continue;
    if (!seen[e]) distinct.push_back(e);
    ++seen[e];
  }
  for (uint32_t e : distinct) { counts.push_back(seen[e]); seen[e] = 0u; }
  heat_.observe(layer, distinct.data(), (uint32_t)distinct.size(), counts.data());

  // The MISSES, which is what `Demand` admits on. Recomputed here from `device_of_` rather than
  // handed down from the dispatch loop above: that loop runs per card and this must run once, and
  // residency is a property of (layer, expert) rather than of a card — every card holds a shard of
  // every resident expert at the same slab offset.
  if (ecfg_.strategy == PlacementEngineConfig::Strategy::Demand) {
    static thread_local std::vector<uint32_t> missing;
    missing.clear();
    for (uint32_t e : distinct)
      if (device_of_[(size_t)layer * n_expert_ + e] < 0) missing.push_back(e);
    heat_.note_missing(layer, missing.data(), (uint32_t)missing.size());
  }
  }

  // ---- issue -----------------------------------------------------------------------------------
  //
  // A promotion needs a free slab slot and a demotion needs a free pool slot, and each move creates
  // the other's supply. The engine primes off the shadow slots, runs ahead by at most that many,
  // and settles into alternating pairs.
  //
  // One round PER RECORD CONSUMED, not per call. A tick that took two records has two dispatches'
  // worth of new evidence and gets two rounds of budget, so `moves_per_tick` keeps meaning moves a
  // dispatch whether the ring runs level or catches up. That is what makes the ring-fed engine the
  // same policy rather than a slower one.
  estats_.rounds_total += rounds;
  if (!rounds) ++estats_.dry;
  for (uint32_t r = 0; r < rounds; ++r)
  for (uint32_t m = 0; m < ecfg_.moves_per_tick; ++m) {
    if (I.free_slab.empty() && I.free_pool.empty()) { ++estats_.no_supply; break; }
    uint32_t up = 0, down = 0;
    const bool got = ecfg_.strategy == PlacementEngineConfig::Strategy::Demand
                         ? heat_.pick_demand(&up, &down)
                         : heat_.pick(&up, &down);
    if (!got) { ++estats_.refused; break; }
    bool moved = false;
    if (!I.free_slab.empty()) moved |= engine_issue(up, /*promote=*/true, streams);
    else ++estats_.starved_slab;
    if (!I.free_pool.empty()) moved |= engine_issue(down, /*promote=*/false, streams);
    else ++estats_.starved_pool;
    if (!moved) break;
  }
}

bool StaticPlacement::engine_issue(uint32_t idx, bool promote, void* const* streams) {
  Impl& I = *impl_;
  const size_t R = devs_.size();
  const uint64_t ss = slot_stride();
  const uint32_t l = idx / n_expert_, e = idx % n_expert_;

  uint32_t slot = UINT32_MAX;
  for (uint32_t m = 0; m < I.moves.size(); ++m)
    if (I.moves[m].state == Impl::Move::State::Free) { slot = m; break; }
  if (slot == UINT32_MAX) { ++estats_.no_slot; return false; }
  Impl::Move& M = I.moves[slot];

  // The gate, recorded ONCE PER TICK and only when a move is actually going out. Everything the
  // mover issues this tick waits on it, and it is downstream of every launch made up to now —
  // including the hybrid stream's, which run_batch_device joined back into the compute stream
  // before the scatter, so one event on the compute stream covers both.
  //
  // Once per tick, not once per move: re-recording it per move is a dozen-odd event records a layer
  // for one event's worth of information, all of them restating the same ordering constraint.
  if (I.gate_tick != estats_.ticks) {
    for (size_t dv = 0; dv < R; ++dv) {
      if (!streams[dv]) return false;
      AFF_HIP_CHECK(hipSetDevice(devs_[dv]));
      AFF_HIP_CHECK(hipEventRecord(I.ev_layer[dv], (hipStream_t)streams[dv]));
    }
    I.gate_tick = estats_.ticks;
  }

  M.promote = promote;
  M.idx = idx;

  if (promote) {
    const int64_t ps = pool_mut_->slot_of(l, e);
    uint8_t* src = pool_mut_->slot_base(ps);
    if (!src) return false;                       // on the ssd tier: not ours to move, see pick()
    M.pool_slot = ps;
    M.slab_slot = I.free_slab.back();
    I.free_slab.pop_back();
    for (size_t dv = 0; dv < R; ++dv) {
      AFF_HIP_CHECK(hipSetDevice(devs_[dv]));
      AFF_HIP_CHECK(hipStreamWaitEvent(I.mv[dv], I.ev_layer[dv], 0));
      // AFF_MOVER_DUP re-issues this copy, which prices the mover's bytes without touching the
      // policy: the repeats write the same bytes to the same slot, so residency, routing and the
      // emitted text are all identical and the only thing that moves is link traffic. A slope of
      // zero means the mover is already hidden behind compute and there is nothing to win by
      // issuing it earlier.
      for (uint32_t rep = 0; rep < mover_dup(true); ++rep)
        AFF_HIP_CHECK(hipMemcpyAsync(I.slab[dv] + M.slab_slot * ss, src + (size_t)dv * ss, ss,
                                     hipMemcpyHostToDevice, I.mv[dv]));
      AFF_HIP_CHECK(hipEventRecord(M.ev[dv], I.mv[dv]));
    }
    estats_.h2d_bytes += (uint64_t)R * ss;
  } else {
    const int64_t g = I.free_pool.back();
    uint8_t* dst = pool_mut_->slot_base(g);
    if (!dst) return false;
    I.free_pool.pop_back();
    M.pool_slot = g;
    M.slab_slot = offset_of_[idx] / ss;
    // D2H out of the slab. The slab layout IS the pool layout, so this is a straight copy with no
    // CPU transform.
    //
    // IT IS NOT FREE, AND THE LINK IS NOT FULL-DUPLEX HERE. It is tempting to read this as running
    // in the direction the hybrid tier leaves idle — that tier streams misses H2D and this writes
    // D2H — and so as very nearly free. Duplicating one direction at a time says otherwise: the two
    // costs are additive, so the directions share one budget and a demotion costs about what a
    // promotion does.
    //
    // The consequence is the exchange rate the whole strategy is tuned against: a swap moves two
    // shards at ~0.12 ms each against a miss's ~0.17, so it has to repay 1.4 future hits, not the
    // 0.74 a half-price demotion would imply.
    for (size_t dv = 0; dv < R; ++dv) {
      AFF_HIP_CHECK(hipSetDevice(devs_[dv]));
      AFF_HIP_CHECK(hipStreamWaitEvent(I.mv[dv], I.ev_layer[dv], 0));
      for (uint32_t rep = 0; rep < mover_dup(false); ++rep)
        AFF_HIP_CHECK(hipMemcpyAsync(dst + (size_t)dv * ss, I.slab[dv] + offset_of_[idx], ss,
                                     hipMemcpyDeviceToHost, I.mv[dv]));
      AFF_HIP_CHECK(hipEventRecord(M.ev[dv], I.mv[dv]));
    }
    estats_.d2h_bytes += (uint64_t)R * ss;
  }
  heat_.set_busy(idx, true);
  M.state = Impl::Move::State::Copying;
  return true;
}

void StaticPlacement::engine_retire(void* const* streams) {
  Impl& I = *impl_;
  const size_t R = devs_.size();
  auto landed = [&](Impl::Move& M) {
    for (size_t dv = 0; dv < R; ++dv)
      if (M.ev[dv] && hipEventQuery(M.ev[dv]) != hipSuccess) return false;
    return true;
  };

  // This tick's edits, one patch a card, applied on the compute streams below. Collected rather than
  // stored because the CARDS read the table when they RUN: a plain store lands in the middle of
  // dispatches the host has issued and the device has not reached. See moe_plan.h.
  PlcPatch patch[8];
  // A card with no stream cannot be ordered against, so the table would have to be stored blind.
  // That cannot happen on the shipped path — `engine_issue` already refuses a tick when a stream is
  // missing — and guessing here is how a correctness property quietly becomes a conditional one.
  auto flush = [&] {
    for (size_t dv = 0; dv < R; ++dv) {
      if (!patch[dv].n) continue;
      if (!streams[dv]) {
        aff::ui::fatal("fatal: the placement engine has no compute stream for card %zu and "
                     "cannot publish %u table edits in order.\n", dv, patch[dv].n);
        std::abort();
      }
      AFF_HIP_CHECK(hipSetDevice(devs_[dv]));
      if (!plc_publish_hip(plc_[dv], patch[dv], (hipStream_t)streams[dv])) std::abort();
      patch[dv] = PlcPatch{};
    }
  };

  for (Impl::Move& M : I.moves) {
    if (M.state == Impl::Move::State::Free) continue;
    const uint32_t l = M.idx / n_expert_, e = M.idx % n_expert_;

    if (M.state == Impl::Move::State::Copying) {
      if (!landed(M)) continue;

      // ---- PUBLISH. Both copies of the bytes are valid at this instant, which is what makes the
      // edit safe to make from this thread with nothing synchronised: the dispatch reads one table,
      // and whichever side of the write it lands on names bytes that are correct. That argument is
      // about BYTES; WHEN the card reads the word is the separate question the patch below answers.
      if (M.promote) {
        device_of_[M.idx] = 0;                        // sharded: "which card" is not a question
        offset_of_[M.idx] = M.slab_slot * slot_stride();
        pool_mut_->bind(l, e, -1);
        heat_.set_tier(M.idx, ExpertTier::Vram);
        ++resident_now_;
        ++estats_.promotions;
        if (ever_moved_[M.idx] & 2u) ++estats_.readmits;
        if (!(ever_moved_[M.idx] & 1u)) { ++estats_.distinct_promoted; ever_moved_[M.idx] |= 1u; }
      } else {
        pool_mut_->bind(l, e, M.pool_slot);
        device_of_[M.idx] = -1;
        heat_.set_tier(M.idx, ExpertTier::Pool);
        --resident_now_;
        ++estats_.demotions;
        ever_moved_[M.idx] |= 2u;
      }
      heat_.set_busy(M.idx, false);
      // The table's own publish, at the instant the fact changes and in the same thread that changes
      // it. Deliberately AFTER both arms rather than inside them: `device_of_`, `offset_of_` and the
      // pool binding are all inputs to the address, and a promote edits two of them while a demote
      // edits the other two, so one call after the branch is the only spelling that cannot be half
      // done. The store's safety is the argument three comments above — at this instant both copies
      // of the bytes are valid.
      // The patch holds a tick's worth; more retirements than that in one pass just take two
      // launches. Flushed BEFORE the append that would overflow so the abort in `publish_base` stays
      // what it says it is — a caller bug, not a capacity the loop is expected to hit.
      // Any card, not card 0: `publish_base` skips a card whose table failed to allocate, so the
      // counts are not guaranteed to move in lockstep.
      for (size_t dv = 0; dv < R; ++dv)
        if (patch[dv].n == kPlcPatchMax) { flush(); break; }
      publish_base(M.idx, patch);

      M.state = Impl::Move::State::Published;
      continue;
    }

    if (M.state == Impl::Move::State::Published) continue;   // this tick's; handled below

    if (!landed(M)) continue;
    if (M.promote) I.free_pool.push_back(M.pool_slot);
    else           I.free_slab.push_back(M.slab_slot);
    M.state = Impl::Move::State::Free;
  }

  flush();

  // ---- QUARANTINE. The slot being freed was named by tiles built in earlier layers, and those
  // kernels may still be running. Re-recording the move's own events on the COMPUTE stream is the
  // barrier: when they signal, every launch made before the publish has retired and the slot is safe
  // to write. This is §3.3, and it is the bug class that has cost this project the most time — a slot
  // reused too early is not a crash, it is fluent text with a new md5.
  //
  // AFTER the flush, and that is the whole reason this is a second pass. The claim the quarantine
  // makes is "every kernel that can still name the old address has retired"; a kernel issued between
  // the event and the publish would name it, so the event has to sit downstream of the publish in
  // stream order for the sentence to be true.
  for (Impl::Move& M : I.moves) {
    if (M.state != Impl::Move::State::Published) continue;
    for (size_t dv = 0; dv < R; ++dv) {
      // A null stream would record on the LEGACY DEFAULT stream, and the quarantine would then look
      // satisfied for a reason that has nothing to do with the kernels it is guarding. A card the
      // caller did not hand a stream for cannot be reasoned about; this cannot happen on the shipped
      // path — `engine_issue` refuses a tick without one — so it says so rather than being clever.
      if (!streams[dv]) {
        aff::ui::fatal("fatal: the placement engine has no compute stream for card %zu and "
                     "cannot quarantine the slot it just freed.\n", dv);
        std::abort();
      }
      AFF_HIP_CHECK(hipSetDevice(devs_[dv]));
      AFF_HIP_CHECK(hipEventRecord(M.ev[dv], (hipStream_t)streams[dv]));
    }
    M.state = Impl::Move::State::Quarantine;
  }
}

void StaticPlacement::snapshot_plane(std::vector<float>* heat, std::vector<uint8_t>* tier,
                                     std::vector<uint32_t>* per_layer) const {
  heat->clear();
  tier->clear();
  per_layer->clear();
  if (!enabled_ || !n_layer_ || !n_expert_) return;
  const size_t n = (size_t)n_layer_ * (size_t)n_expert_;
  tier->assign(n, (uint8_t)ExpertTier::Ssd);
  per_layer->assign(n_layer_, 0);
  if (engine_on_) heat->assign(n, 0.0f);

  for (uint32_t l = 0; l < n_layer_; ++l) {
    uint32_t res = 0;
    for (uint32_t e = 0; e < n_expert_; ++e) {
      const size_t i = (size_t)l * n_expert_ + e;
      // The engine's own tier map when it is running, because it is the one the strategy is acting
      // on; derived from the dispatch's own predicates otherwise, so the two never disagree about
      // an expert that moved between them.
      ExpertTier t;
      if (engine_on_) {
        t = heat_.tier((uint32_t)i);
        (*heat)[i] = heat_.heat((uint32_t)i);
      } else {
        t = resident(l, e) ? ExpertTier::Vram
                           : (pool_shard_dev(l, e, 0) ? ExpertTier::Pool : ExpertTier::Ssd);
      }
      (*tier)[i] = (uint8_t)t;
      if (t == ExpertTier::Vram) ++res;
    }
    (*per_layer)[l] = res;
  }
}

void StaticPlacement::engine_report() const {
  // Before the engine guard: the SSD tier is reachable under every strategy, including static, and
  // a run that touched it is a run whose throughput has an explanation the other counters cannot
  // give. Silence here is the useful case.
  if (estats_.ssd_staged)
    aff::ui::err("\nssd tier: %llu demand reads, each a pread plus a transform that blocks dispatch."
                 " The RAM budget does not cover the non-resident experts — raise --host-pool-mib "
                 "or lower --kv-size.\n",
                 (unsigned long long)estats_.ssd_staged);
  // Which build actually ran, and how the ring behind it fared. Printed whenever the device build
  // was asked for, even when it never engaged — especially then.
  // Tagged, because two placements report: 'D' is the target's 43 layers and 'S' the draft's three
  // stages. Untagged, the draft's line read as a second opinion about the target's.
  if (plan_dispatch_ || host_dispatch_) {
    const uint64_t tot = plan_dispatch_ + host_dispatch_;
    aff::ui::err("\nmoe dispatch [%c]: %llu of %llu built on the device (%.1f%%), %llu on the host; "
                 "%u of %u layers hold an addressless expert\n",
                 route_tag_, (unsigned long long)plan_dispatch_, (unsigned long long)tot,
                 tot ? 100.0 * (double)plan_dispatch_ / (double)tot : 0.0,
                 (unsigned long long)host_dispatch_, ssd_layers_hot_, n_layer_);
    if (act_seen_ || act_drop_)
      aff::ui::err("  activation ring: %llu records read, %llu dropped (%.2f%%)\n",
                   (unsigned long long)act_seen_, (unsigned long long)act_drop_,
                   (act_seen_ + act_drop_)
                       ? 100.0 * (double)act_drop_ / (double)(act_seen_ + act_drop_) : 0.0);
  }
  if (!engine_on_) return;
  const double gib = 1.0 / 1073741824.0;
  aff::ui::err("\nplacement engine: %llu ticks, %llu promotions, %llu demotions "
              "(%.2f GiB h2d, %.2f GiB d2h)\n",
              (unsigned long long)estats_.ticks, (unsigned long long)estats_.promotions,
              (unsigned long long)estats_.demotions, (double)estats_.h2d_bytes * gib,
              (double)estats_.d2h_bytes * gib);
  aff::ui::err("  %llu ticks declined on hysteresis, %llu starved of a slab slot, %llu of a pool "
              "slot; %llu resident now against %llu at load\n",
              (unsigned long long)estats_.refused, (unsigned long long)estats_.starved_slab,
              (unsigned long long)estats_.starved_pool, (unsigned long long)resident_now_,
              (unsigned long long)resident_target_);
  // The two ways a tick issues nothing WITHOUT the guard declining, which the line above cannot
  // distinguish and which is the difference between a mover at its budget and one that never got
  // to spend it.
  aff::ui::err("  %llu ticks got no ring record, %llu issues found every in-flight slot busy, "
               "%llu rounds found no free slot to move into; %llu rounds over %llu ticks\n",
               (unsigned long long)estats_.dry, (unsigned long long)estats_.no_slot,
               (unsigned long long)estats_.no_supply,
               (unsigned long long)estats_.rounds_total, (unsigned long long)estats_.ticks);
  // The one line that says whether the traffic above was worth paying. A readmit is an expert this
  // engine demoted and then fetched back: both directions of the bus spent to return to where it
  // started, on the link the hybrid tier needs. High here means the guards are too loose.
  aff::ui::err("  %llu distinct experts promoted, %llu readmits (%.0f%% of promotions were "
              "of something it had already demoted)\n",
              (unsigned long long)estats_.distinct_promoted, (unsigned long long)estats_.readmits,
              estats_.promotions ? 100.0 * (double)estats_.readmits / (double)estats_.promotions
                                 : 0.0);

  // ---- the histogram this engine exists for ----------------------------------------------------
  //
  // Static placement ranks every expert globally by the container's profile, which leaves the
  // per-layer resident count wherever that ranking happens to put it — in practice badly skewed. A
  // layer starved of slots misses on nearly every token no matter how good the ranking inside it is,
  // which is how a hit rate can land below what picking at random would give. Whether the engine
  // repaired that is a question about THIS distribution, not about the mean.
  std::vector<uint32_t> now;
  heat_.resident_per_layer(&now);
  auto spread = [&](const std::vector<uint32_t>& v, uint32_t* lo, uint32_t* hi, double* sd) {
    if (v.empty()) { *lo = *hi = 0; *sd = 0; return; }
    double sum = 0;
    *lo = v[0]; *hi = v[0];
    for (uint32_t x : v) { sum += x; *lo = std::min(*lo, x); *hi = std::max(*hi, x); }
    const double mean = sum / (double)v.size();
    double var = 0;
    for (uint32_t x : v) var += (x - mean) * (x - mean);
    *sd = std::sqrt(var / (double)v.size());
  };
  uint32_t lo0 = 0, hi0 = 0, lo1 = 0, hi1 = 0;
  double sd0 = 0, sd1 = 0;
  spread(resident_at_load_, &lo0, &hi0, &sd0);
  spread(now, &lo1, &hi1, &sd1);
  aff::ui::err("  resident experts a layer (of %u): at load min %u max %u sd %.1f"
              " -> now min %u max %u sd %.1f\n",
              n_expert_, lo0, hi0, sd0, lo1, hi1, sd1);
}


} // namespace aff
