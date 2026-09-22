// STATIC expert placement plus the hybrid batched dispatch — the whole routed-expert tier.
//
// Nothing moves at runtime. Residency is decided once at load and never revised: `device_of_` is
// written inside init() and only ever read at dispatch. There is no eviction and no admission.
// This is the substrate the placement engine sits on. Read init() before adding a strategy here,
// because most of what constrains one is a load-time fact and lives there.
//
// At load, the hottest experts by the container's profile are uploaded to VRAM. Every routed layer
// then splits its experts two ways, and ONLY two:
//
//   resident   -> the expert's shard is in this card's slab
//   missing    -> the expert's shard is in the registered host pool, and the tile names it
//                 directly, so the GEMM streams those 4.32 MiB over PCIe as it computes
//
// TWO here, THREE under the placement engine. The model has a third tier — experts on the SSD,
// which static placement reaches only as a budget leftover and which the engine populates on purpose.
// An SSD-tier expert is streamed in place from a staging buffer and discarded, never admitted; the
// dispatch path for it does not exist yet.
//
// There is no third ENGINE, which is a different statement: three tiers, one compute path. A CPU
// arm was built and validated and never paid — the host decodes an expert several times slower than
// the two links read one, so it has to hide completely, and the only thread that could wait for it
// is the one keeping the cards' queue full. Removing it also buys determinism: a missing expert is
// bit-identical to a resident one, so a token no longer depends on which fit.
//
// ---- ONE DISPATCH PATH ------------------------------------------------------------------------
//
// `run_batch_device` is it. `Model::forward_token` cannot reach this tier at all — it aborts in the
// dense matvec, because the fp8 shards are preshuffled for the W8A8 GEMM — so the variants only it
// could reach are gone rather than left dead: a runtime placement engine would otherwise have to
// keep each one's device-side residency mirror in sync on every move.
//
// Decode reaches this through DSpark's block verify and prefill through the chunk loop; both are
// `Model::forward_prefill`, so there is one caller shape, not two.

#pragma once

#include "engine/expert_heat.h"
#include "engine/host_pool.h"
#include "format/aff_reader.h"
#include "gpu/expert_kernel_hip.h"
#include "quant/blockquant.h"

#include <cstdint>
#include <string>
#include <vector>

namespace aff {

// ---- the placement engine's configuration ------------------------------------------------------
struct PlacementEngineConfig {
  // `Static` is today: nothing moves, and it is the control arm every heat measurement is against.
  //
  // `Heat` and `Demand` share the whole mechanism and one heat plane; the ONLY difference is what
  // triggers an admission. `Heat` promotes the globally hottest pooled expert, `Demand` promotes an
  // expert this dispatch just had to stream. See expert_heat.h — what separates them is that
  // unbounding `Heat`'s move budget barely helps, while changing the trigger is worth a great deal.
  enum class Strategy : uint8_t { Static, Heat, Demand };

  Strategy strategy = Strategy::Static;
  HeatParams heat;
  // Swaps STARTED per layer dispatch. A promotion and a streamed miss are the same 4.32 MiB over
  // the same link; the difference is that a promotion pays once and a miss pays every block that
  // selects the expert. So the fear is that a high budget makes the mover compete with the tier it
  // is trying to shrink.
  //
  // It is misplaced at this scale: raising the budget pays even though it moves considerably more
  // bytes, because converging sooner is worth more than the traffic costs. It then saturates, and
  // this sits at the point where the knob stops mattering rather than at the largest value that
  // still helped. Past that the GUARD declines the extra opportunities, not the budget, so raising
  // this alone is worth nothing until `min_gain` moves with it.
  uint32_t moves_per_tick = 8;
  // ---- HOW FAR THE HOST MAY RUN AHEAD OF THE CARDS, IN LAYER DISPATCHES ---------------------------
  //
  // 0 is unbounded, which is what the device-built dispatch produced, and it starves the mover.
  //
  // A freed slab or pool slot returns to the free list only after its quarantine event signals, and
  // that event is on the COMPUTE stream. So the recycle latency is however deep that stream happens
  // to be queued, and the move rate is `shadow slots / recycle latency`: let the host run and the
  // stream carries a whole block of work, and almost every round finds no free slot to move into.
  // The guard is not what binds — it declines the same handful of times either way — the engine
  // simply never gets the supply to spend.
  //
  // Holding the host near the cards is therefore not a latency measure but a SUPPLY one. 2 is the
  // smallest depth that recovers the whole effect while leaving the issue path a dispatch of
  // margin; it is worth a few percent of decode on a prompt whose working set fits and a quarter of
  // it on one whose does not.
  //
  // Two nearby changes look like this and are not. A BLOCK-granular gate does nothing, because one
  // block is already 43 layers of queued work — the granularity has to be the layer. And more
  // shadow slots are the wrong end: they raise the move rate and measure SLOWER, because the slots
  // come out of residency and more copies in flight contend with the hybrid tier.
  //
  // One event a dispatch a card, and a host wait on the one from `max_lead_blocks` ago.
  uint32_t max_lead_blocks = 2;
};

struct PlacementEngineStats {
  uint64_t ticks = 0;
  uint64_t promotions = 0, demotions = 0;
  uint64_t h2d_bytes = 0, d2h_bytes = 0;
  // Why a tick did NOT move anything, which is the diagnostic that matters once the engine looks
  // idle. `refused` is the strategy declining on hysteresis — the healthy steady state, and the
  // number that should dominate once the hot set has settled. The two `starved` counts mean the
  // pipeline is the limit rather than the policy.
  uint64_t refused = 0, starved_slab = 0, starved_pool = 0;
  // Why a tick issued NOTHING, which the three above cannot say. `no_slot` is every in-flight Move
  // record busy — the 32 shadow slots, still waiting on their copies. `dry` is a tick that consumed
  // no ring record and therefore got no budget at all: the cards had published nothing new. Without
  // these two a starved mover is indistinguishable from a satisfied one, because the guard shows 4
  // declines either way.
  uint64_t no_slot = 0, dry = 0, rounds_total = 0;
  // ...and the exit that had no counter at all: BOTH free lists empty, so the round cannot even ask
  // the strategy for a move. Supply, not policy. Slots return to those lists only after a move's
  // quarantine event signals, and that event is on the compute stream — so the recycle latency is
  // however deep that stream is queued, and the move rate is `shadow slots / recycle latency`.
  uint64_t no_supply = 0;
  // ---- is the churn PROGRESS or THRASH? --------------------------------------------------------
  //
  // The same total move count means two opposite things. If every promotion is of an expert never
  // promoted before, the engine is still discovering the working set and the traffic is an
  // investment. If it keeps re-admitting experts it demoted a moment ago, the traffic is pure loss:
  // both directions of the bus spent to arrive back where it started, on the same link the hybrid
  // tier is trying to use. `readmits` is the second case counted directly, and it is the number
  // that decides whether the hysteresis guards are set right.
  uint64_t distinct_promoted = 0, readmits = 0;
  // Demand reads served out of the SSD tier. Non-zero means the RAM budget did not cover the
  // complement and some dispatches blocked on a pread plus a transform before they could launch. It
  // is a correctness path; a run that leans on it is a run to re-budget.
  uint64_t ssd_staged = 0;
};

// What placement DID, not what dispatch is doing. Every field here is written once, by init().
//
// The per-dispatch counters live elsewhere: `HybridCensus` in engine/model.h, which is asked of
// PLACEMENT rather than of the dispatch, because the H-index printed at the end of a profiled run
// is the number a placement strategy is trying to move.
struct PlacementStats {
  uint64_t resident_experts = 0;
  uint64_t total_experts = 0;
  double   vram_used_gib = 0;
  double   load_seconds = 0;
};

class StaticPlacement {
public:
  StaticPlacement() = default;
  ~StaticPlacement();
  StaticPlacement(const StaticPlacement&) = delete;
  StaticPlacement& operator=(const StaticPlacement&) = delete;

  // Uploads as many experts as `vram_budget_gib` per card allows, hottest first by the container's
  // popularity profile. Falls back to CPU-only (and reports so) when no gfx12 device is present.
  //
  // `batch_tok` is the prefill chunk (0 for none) and `k` the router's top-k. Placement allocates
  // the batched routed arena for that worst case BEFORE it sizes its own slabs, because it is
  // greedy — it takes free VRAM minus a constant — and the arena is the only thing the engine
  // allocates afterwards. A caller that raises the chunk without telling this does not get fewer
  // experts, it gets `hipErrorOutOfMemory` out of the first launch after the arena fails to grow.
  bool init(const AffReader* aff, const Codebook& cb, double vram_budget_gib_per_gpu,
            uint32_t n_layer, uint32_t n_expert, uint32_t batch_tok, uint32_t k, std::string* err);

  // ---- SHADOW SLOTS: call before init(), and only when the engine will run ----------------------
  //
  // Slab slots that are allocated but hold no expert. A promotion writes into one of these and only
  // then publishes the new `offset_of_`, so a live slot is never DMA'd over while a kernel might
  // still be reading it. Getting that ordering wrong is the failure mode that has cost this project
  // the most time, because it does not crash — it gives fluent output and a different md5 every run.
  //
  // They come out of the VRAM the load banner reports free after the slabs, NOT out of residency, so
  // the engine costs nothing in hit rate to make room for — and `placement headroom` on the next run
  // is the number that says whether that is still true.
  void reserve_shadow_slots(uint32_t n) { shadow_slots_ = n; }

  // VRAM the slab must not take, MiB a card; 0 keeps the value derived from the card. Also before
  // init(). See the note at its use for what it covers.
  void set_driver_headroom_mib(double m) { headroom_mib_ = m; }

  bool enabled() const { return enabled_; }

  // The host tier for everything that did not fit in VRAM. Also registers the pool with the driver
  // so the cards can DMA out of it — without that, every missing expert is a bounce-buffer memcpy
  // and there is no hybrid tier at all. Must outlive this.
  //
  // Returns false if registration failed, and that is FATAL to the caller rather than a degraded
  // mode: with no pool there is nothing for a missing expert's tile to point at.
  bool set_host_pool(const HostExpertPool* p);

  // ---- the hybrid tier: what the host pool has to hold, and where it is ------------------------
  //
  // A missing expert is not copied anywhere. Its tile's three pointers name the registered host
  // pool and the GEMM streams the weights over PCIe as it computes — no staging ring, no copy
  // stream, no event, and no waiting for a whole expert to land before its first block can start.
  // End to end it reaches most of what the copy engine gets on its own, and no read geometry beats
  // the wire (bench/hostread).
  //
  // The price is that the pool must hold what a VRAM slab holds — colgroup-major, and sharded when
  // placement is sharding — so these two hand the residency loader's own permutation to
  // HostExpertPool::init. Call them after init() and before the pool is built.
  static void pool_tf(void* ctx, uint32_t layer, const uint8_t* src, uint8_t* dst);
  uint64_t pool_out_stride() const;

  // Card `dv`'s shard of a NON-resident expert as a DEVICE-VISIBLE address, which is the pointer a
  // zero-copy tile carries. Null when the expert is not pooled (the SSD tier) or the pool was never
  // registered — and a null is the caller's signal to leave the slot unhandled, not to guess.
  const uint8_t* pool_shard_dev(uint32_t layer, uint32_t expert, size_t dv) const;
  // The SSD tier's dispatch path: read an expert off disk into a staging slot in the registered
  // pool and return the address a tile carries, or null when there is no ring. Blocking, counted,
  // and the slot stays valid until ssd_stage_arm's events fire. See the definition.
  const uint8_t* ssd_stage(uint32_t layer, uint32_t expert, size_t dv);
  void ssd_stage_arm(void* const* streams);
  bool hybrid_ready() const { return pool_dev_ != nullptr; }

  // True when (layer, expert) is in VRAM. The host pool's admission predicate is the complement of
  // this, so the two tiers stay disjoint.
  //
  // Disjoint is an invariant of STATIC placement, held because nothing moves. A demotion copies the
  // expert's bytes back D2H out of the slab — the slab layout IS the pool layout, so it is a
  // straight copy.
  bool resident(uint32_t layer, uint32_t expert) const {
    if (!enabled_) return false;
    return device_of_[(size_t)layer * n_expert_ + expert] >= 0;
  }

  const PlacementStats& stats() const { return stats_; }

  // ---- the placement engine ---------------------------------------------------------------------
  //
  // Call after init() and after set_host_pool(): it needs the slab, the pool and both to be final.
  // The pool is taken NON-const because a demotion changes its membership, which is the one thing
  // load-time `init(wanted, ...)` cannot express.
  //
  // Everything runs on the DISPATCH THREAD. The transfers are asynchronous — their own stream,
  // gated behind an event recorded after the layer's launches — but every table edit is applied by
  // this thread at a layer boundary, draining a queue of transfers that have landed. No mutex on
  // the read side, because that side is the critical path of every layer, and no background thread
  // mutating `device_of_` mid-layer, because that would make an A/B unreproducible.
  //
  // Fails loudly rather than degrading: no TP=2 shards, no host pool, or AFF_FORCE_RESIDENT set
  // (which pins routing to whatever is resident, so a moving placement makes the output a lottery)
  // are each an error, not a quieter mode.
  bool placement_engine_init(HostExpertPool* pool, const PlacementEngineConfig& cfg,
                             std::string* err);
  // The exit banner: what moved, what it cost, and the per-layer residency histogram — which is the
  // one number that says whether the engine did the thing it exists for. Static placement ranks all
  // 11008 experts globally by the container's profile, and that leaves layers with wildly different
  // resident counts; a layer starved of slots misses on nearly every token however good the ranking
  // inside it is. The histogram before and after is the whole story.
  void engine_report() const;
  // amdnas-fixes plc-verify: compare the device placement table with plc_host_ (AFF_PLC_VERIFY=N ticks).
  void verify_plc(const char* tag) const;
  mutable uint64_t plc_checks_ = 0, plc_bad_ = 0;

  // The cards this placement actually uses. Every one of them needs a keepalive heartbeat, and this
  // is the only place that knows which they are.
  const std::vector<int>& devices() const { return devs_; }

  // The rank plan the DENSE path chose, so a slab lands on the same card as the rank that reads it,
  // in the same order. Call before init(). Without it placement enumerates for itself — right for a
  // standalone caller, and wrong the moment the two enumerations can differ, which they do under
  // AFF_LOGICAL_RANKS and whenever the dense path was capped below the card count.
  void set_rank_plan(const int* devs, size_t n) { devs_.assign(devs, devs + n); }

  // PREFILL AND DSPARK VERIFY: a whole chunk's routed experts for one layer, grouped by expert so
  // each one's 8.64 MiB is read once for every token that chose it rather than once per token.
  // `d_x[r]` and `d_block[r]` are the chunk's dim-major [hidden][nb_stride] buffers on rank r;
  // nothing is copied and no host contribution comes back.
  //
  // This is the ONLY dispatch path. Residency is read here, per call, through `mine_for` and
  // `pool_shard_dev` — so a placement engine that mutates `device_of_`/`offset_of_` is seen by the
  // very next layer without any device-side table to invalidate.
  //
  // `handled[b*k + j] = 1` for every pair taken. Returns how many.
  //
  // Anything left unhandled is an expert that is in neither VRAM nor the pool — i.e. on the SSD
  // tier — and there is no code that can serve one, so the caller treats it as fatal. That is right
  // for static placement, where the tier holds only what did not fit. It stops being right
  // the moment the placement engine puts a cold tail there deliberately, at which point this needs
  // a third membership pass reading a staging ring.

  // The selection as this card can read it, one per rank. When every entry is populated, the
  // dispatch is built by a kernel and the host `sel`/`w` are used only for the diagnostics that ask
  // for them. See moe_plan.h.
  struct RouteDev { const uint32_t* sel = nullptr; const float* wt = nullptr; void* ready = nullptr; };

  uint32_t run_batch_device(uint32_t layer, const uint32_t* sel, const float* w, uint32_t nb_tok,
                            uint32_t k, const float* const* d_x, float* const* d_block,
                            uint32_t nb_stride, uint32_t hidden, uint32_t inter, float limit,
                            void* const* streams, uint8_t* handled,
                            const RouteDev* route = nullptr);

  // ---- THE ONE DECISION ---------------------------------------------------------------------
  //
  // Will layer `l`'s dispatch be built on the device? Two callers ask, and they must never disagree:
  // the router asks so it can skip reading the selection back to the host, and the dispatch asks so
  // it knows which build to run. A router that skips the readback for a layer the dispatch then
  // builds on the host would hand it a null selection.
  //
  // Everything about it is settled except one thing, and that thing is per LAYER: a layer holding an
  // expert with no address anywhere keeps the host build, because a kernel cannot read one off disk.
  // The run-level conditions are latched on the first dispatch and are fatal rather than soft, so
  // they cannot flip underneath the two callers.
  bool plan_layer(uint32_t layer) const {
    return plan_ready_ && (layer >= ssd_layer_.size() || ssd_layer_[layer] == 0u);
  }
  // The question the ROUTER asks: may it skip reading the selection back? Exactly when the layer is
  // device-built, because then nothing on the fast path wants a host copy of it. Kept as its own
  // name because the caller's question is genuinely a different one from "who builds the dispatch",
  // and because the diagnostics that DO want the selection turn it off at their own site.
  bool device_dispatch(uint32_t layer) const { return plan_layer(layer); }

  uint32_t n_expert() const { return n_expert_; }
  uint32_t n_layer() const { return n_layer_; }

  // ---- what a live display reads, in one call ---------------------------------------------------
  //
  // `tier` is where every (layer, expert)'s bytes are, layer-major, in ExpertTier's encoding; it is
  // derived from the same `device_of_` and pool membership the dispatch reads, so it cannot disagree
  // with what a kernel would find. `heat` is the placement engine's decayed activation count, and it
  // is left EMPTY under static placement: there is no plane, and filling it from the container's
  // popularity profile would draw a ranking while claiming to draw what the router did.
  //
  // DISPATCH THREAD ONLY, the same rule as every other read of the placement tables. Copying rather
  // than exposing the vectors is the point — the caller hands the copy to a thread that must never
  // touch the live ones.
  void snapshot_plane(std::vector<float>* heat, std::vector<uint8_t>* tier,
                      std::vector<uint32_t>* per_layer) const;

  const PlacementEngineStats& engine_stats() const { return estats_; }
  bool engine_running() const { return engine_on_; }
  // Which model this placement belongs to, for the routing trace. 'D' (the target) unless the
  // DSpark draft's own placement sets 'S'; the draft's three stages are layers 0-2 of a different
  // container, and an untagged trace merges them with the target's layers 0-2. See route_trace.h.
  void set_route_trace_tag(char t) { route_tag_ = t; }

private:
  const AffReader* aff_ = nullptr;
  Codebook cb_;
  bool enabled_ = false;
  uint32_t n_layer_ = 0, n_expert_ = 0;
  char route_tag_ = 'D';
  std::vector<int> device_of_;                            // [layer*n_expert + e], -1 = not resident
  std::vector<uint64_t> offset_of_;

  // ---- THE PLACEMENT TABLE: where each expert's bytes are, as a card can read it ------------------
  //
  // `plc_[dv][layer*n_expert + e]` is the device-visible base address of THIS card's shard of that
  // expert — the slab address when it is resident, the registered pool's address when it is not, and
  // 0 when it is on neither (the SSD tier). One 64-bit word, 43 x 256 x 8 = 88 KiB a card.
  //
  // It carries an ADDRESS and not a tier because that is all a consumer needs: `run_batch_device`'s
  // tile build already reduces residency to exactly this question (static_placement.hip, "The ONLY
  // thing that distinguishes a missing expert from a resident one: where its shard is"), and a
  // consumer that cannot ask `device_of_` — a kernel — needs the answer rather than the reason.
  //
  // WHY THIS IS NOT THE DEVICE-SIDE MIRROR THAT WAS DELETED (see the note at the top of this file).
  // Those were CACHES: a device copy of a host-owned fact, which had to be invalidated and could
  // drift. This is a second ENCODING of one fact, written by the same thread at the same instants
  // `device_of_`/`offset_of_` are. `publish_base` is the one point where that happens, and all three
  // writers go through it, so a fourth cannot be added without noticing.
  //
  // Fine-grained device memory, the same allocation the all-reduce uses for its flags, so the CPU
  // stores into it directly and the publish needs no copy, no staging buffer and no stream.
  //
  // A PLAIN STORE IS NOT ENOUGH, because a kernel reads the table when it RUNS — an unbounded
  // distance after the host issued it. A host store therefore re-classifies an expert on a layer the
  // host never chose, which moves `start[e]`, which fixes the summation order, which moves the text.
  // The device-side word is published on the compute stream instead (`plc_publish_hip`); moe_plan.h
  // carries the argument and the measurement.
  // Bit 0 of an entry: set means the address is in this card's slab, clear means the registered host
  // pool. 0 as a whole word means neither — the SSD tier, which has no address until it is staged.
  static constexpr uint64_t kPlcResident = 1ull;
  static constexpr uint64_t kPlcAddrMask = ~1ull;
  std::vector<uint64_t*> plc_;                            // one table a card, on the card
  // The same words in ordinary host memory, laid out card-major. Fine-grained device memory is
  // readable by the CPU, which is what makes the publish a plain store — but only the STORE is
  // cheap. A host READ of it is an uncached PCIe round trip with no prefetching, and the dispatch
  // reads one word a routed slot: pointing the host pass at `plc_` adds a fixed cost per ENTRY READ
  // on both cards, proportional to entries rather than to work done.
  // So the host reads this and the cards read theirs, and `publish_base` writes both from one
  // computation — which is the same argument that makes the device table not a cache: two encodings
  // written at one instant by one thread, not a copy kept in step with an original.
  std::vector<uint64_t> plc_host_;                        // [card][layer*n_expert + e]
  // ---- WHICH LAYERS THE DEVICE BUILD CANNOT CLAIM ------------------------------------------------
  //
  // An expert on the SSD tier has no address until the staging ring reads it off disk, and reading
  // off disk is a host operation — so a kernel that meets one can only drop it, which is a wrong
  // answer rather than a slow one. At the shipped reserve the tier is empty and this claims every
  // layer, but that is a property of the machine's RAM and not something to rely on: a box that
  // spills even a handful of experts, spread thin, loses the device build on every layer they land
  // in. See host_pool.cpp, which prints the shortfall and the reserve that would close it.
  //
  // WHICH LAYER an expert is in is a fact about PLACEMENT and not about routing, so the host knows
  // it without asking the card anything. `ssd_layer_[l]` counts the layer's addressless experts, and
  // a layer with any keeps the host build and its drain. That is the whole handling: no prediction,
  // no speculative staging, and no third tile class.
  std::vector<uint32_t> ssd_layer_;                       // [layer], experts with no address
  uint32_t ssd_layers_hot_ = 0;                           // how many layers that is
  bool plan_ready_ = false;                               // the run-level half of plan_layer
  // Per PLACEMENT and not a function-level static, because two of them run: the target's and the
  // draft's. Shared, whichever fell back first spoke and the other's silence read as agreement.
  bool said_host_ = false, said_ssd_ = false;
  bool plc_alloc(std::string* err);
  // Recompute one entry on every card. `sink` is R patches, one a card, and decides where the
  // DEVICE-side word goes:
  //   nullptr  store it into the card's table now. Load-time fill only — nothing is in flight, and
  //            there is no stream to order it on yet.
  //   a patch  append it, for `plc_publish_hip` to apply on that card's compute stream.
  // The host mirror is written either way, because the host reads it from this thread.
  void publish_base(size_t idx, struct PlcPatch* sink);
  // What the dispatch reads: the host mirror, or 0 when there is no table yet.
  uint64_t plc_at(size_t dv, uint32_t layer, uint32_t e) const {
    const size_t i = dv * ((size_t)n_layer_ * n_expert_) + (size_t)layer * n_expert_ + e;
    return i < plc_host_.size() ? plc_host_[i] : 0ull;
  }
  // The load fill runs on worker threads and finishes BEFORE the host pool is registered, so a
  // per-index publish there would record 0 for every pooled expert. One sweep once both tiers exist
  // is simpler than ordering the two, and it is idempotent — call it from either.
  void publish_all();
  // --- the tensor-parallel expert split ---------------------------------------------------
  //
  // When every expert fits in VRAM, each card holds a SHARD of every expert instead of the whole of
  // half of them — same bytes, same VRAM, but both cards then do exactly half of a token's routed
  // work no matter which experts it picked. Removing that imbalance at the MoE all-reduce, which
  // costs the max of the two cards rather than the mean, is the only thing this exists for.
  //
  // All-or-nothing across the model, deliberately. A mix of sharded and whole experts inside one
  // layer would put two intermediate widths in one batched launch. `tp_parts_ == 1` is byte-for-byte
  // the old layout.
  //
  // With the split on, `device_of_[idx] >= 0` still means resident and `offset_of_[idx]` is the
  // shard's offset in EVERY card's slab (the layout is uniform), so `slab[d] + offset_of_[idx]` is
  // valid for any d. Which card a given expert is "on" stops being a question.
  uint32_t tp_parts_ = 1;

  ExpertShardLayout shard_{};
  // "Card `dv` should evaluate expert `e` of this layer." Sharded, every card evaluates every
  // resident expert; otherwise only its owner does.
  bool mine_for(uint32_t layer, uint32_t e, size_t dv) const {
    const int d = device_of_[(size_t)layer * n_expert_ + e];
    return tp_parts_ > 1 ? d >= 0 : d == (int)dv;
  }
  // The intermediate width one card sees. Every caller passes the model's `moe_inter`; the split
  // is StaticPlacement's own business and stops here rather than being threaded through the engine.
  uint32_t shard_inter(uint32_t inter) const { return inter / tp_parts_; }
  // Bytes one slab slot spans: a shard with the TP split on, a whole expert without it. ONE
  // definition, because the mover's copy size, the offset it reads the pool at and the
  // `offset_of_` it publishes must agree, and they are written in three different functions.
  uint64_t slot_stride() const;
  std::vector<int> devs_;
  const HostExpertPool* pool_ = nullptr;
  // The pool's base as the CARDS see it. On ROCm this is the host address itself, but going through
  // hipHostGetDevicePointer is what makes that a checked fact rather than an assumption.
  uint8_t* pool_dev_ = nullptr;
  bool pool_registered_ = false;            // so the destructor knows whether to unregister
  PlacementStats stats_;

  // ---- placement engine state ------------------------------------------------------------------
  //
  // The strategy is a plain host object and lives here rather than in Impl on purpose: it has no
  // HIP in it, so tests/test_expert_heat.cpp exercises the whole decision half on a machine with no
  // card. Everything that needs a stream or an event is in Impl.
  HostExpertPool* pool_mut_ = nullptr;
  PlacementEngineConfig ecfg_;
  PlacementEngineStats estats_;
  ExpertHeat heat_;
  bool engine_on_ = false;
  uint32_t shadow_slots_ = 0;
  double headroom_mib_ = 0.0;                 // 0 = the derived fraction; see set_driver_headroom_mib
  uint64_t resident_now_ = 0, resident_target_ = 0;
  std::vector<uint32_t> resident_at_load_;    // per layer, for the before/after histogram
  // Bit 1: this engine has promoted it before. Bit 2: it has demoted it before. Two bits over 11008
  // experts, so the thrash question costs 11 KiB and no time on the dispatch path.
  std::vector<uint8_t> ever_moved_;
  // One layer dispatch: fold the selection into the heat table, retire whatever has landed, and
  // start at most `moves_per_tick` swaps. Called at the very END of run_batch_device, which is the
  // layer boundary in every sense that matters — the layer's launches are all issued, so an event
  // recorded here covers every kernel that could still be reading a slot the mover wants.
  void engine_tick(uint32_t layer, const uint32_t* sel, size_t total, void* const* streams,
                   bool from_ring);
  void engine_retire(void* const* streams);
  bool engine_issue(uint32_t idx, bool promote, void* const* streams);

  // Lazily allocated per card, for run_batch_device: one layer's worth of token ids and router
  // weights uploaded in a single copy.
  void* btok_[8] = {};
  void* bwt_[8] = {};
  uint32_t bcap_tok_[8] = {};

  // W8A8 batched path. One arena per stage for the WHOLE layer rather than a four-kernel chain per
  // 16-token slice, so a layer is 7 launches instead of ~512.
  //   bq8/bqs   the chunk activation, E4M3 token-major + per-(token, 128-col) scale
  //   bg/bu     gate and up outputs, [inter][total] with total = this card's routed slots
  //   ba8/bas   the SwiGLU output requantised, E4M3 token-major, router weight already folded in
  //   bydn      the down output, [hidden][total]
  //   btiles    3 * n_tiles ExpertTile descriptors, gate then up then down
  //   bord/brun the slot list sorted by token and its nb+1 prefix, for a deterministic reduction
  void* bq8_[8] = {}; void* bqs_[8] = {};
  void* bg_[8] = {};  void* bu_[8] = {};
  void* ba8_[8] = {}; void* bas_[8] = {};
  void* bydn_[8] = {};
  void* btiles_[8] = {};
  // Slices of `bpack_`, not allocations: one buffer laid out like the pinned slot so the layer's
  // five uploads are a single copy. See the note on bpin_ for what that is worth.
  void* bord_[8] = {}; void* brun_[8] = {};
  void* bpack_[8] = {}; size_t bpack_n_[8] = {};
  // ---- the device build's own scratch ------------------------------------------------------------
  //
  // `plan_col_` is the arena column of every slot, which the plan kernel writes in one phase and
  // reads back in the next to sort each token's slots. `plan_cnt_` and `plan_st_` are what the card
  // says ABOUT the plan — tile counts per group, and the two things that are fatal: a slot whose
  // expert has no address, and a class that needed more tiles than the bound allowed. Neither is on
  // the dispatch path; the host reads them when it next has a reason to look.
  uint32_t* plan_col_[8] = {}; uint32_t plan_col_n_[8] = {};
  uint32_t* plan_cnt_[8] = {};
  struct MoePlanStatus* plan_st_[8] = {};
  uint64_t plan_calls_ = 0, plan_bad_ = 0, plan_over_ = 0;
  uint64_t plan_dispatch_ = 0, host_dispatch_ = 0;
public:
private:
  bool plan_scratch(size_t dv, uint32_t total);
  void lead_gate(void* const* streams, uint32_t nb_tok, size_t R);
  // ---- the activation ring ----------------------------------------------------------------------
  //
  // Pinned host memory the card writes and this thread reads, ~67 KiB. Stated because past a
  // driver-dependent total hipHostRegister returns success and leaves the process broken, so every
  // new pinned allocation is budgeted rather than assumed small.
  struct ActRing* act_ring_ = nullptr;       // host address
  struct ActRing* act_dev_ = nullptr;        // the same memory as the card addresses it
  uint64_t act_tail_ = 0, act_drop_ = 0, act_seen_ = 0;
  bool act_alloc();
  // Consume everything the cards have published and give it to heat. Returns how many records it
  // took, which is 0 whenever the kernel has not run yet — that is the normal case at the start of
  // a run and is not an error.
  uint32_t act_drain();
  uint32_t bcap_slot_[8] = {}, bcap_tile_[8] = {}, bcap_nb_[8] = {};

  // PINNED staging for the five per-layer uploads above, kPinSlots deep.
  //
  // Issuing them straight out of std::vector is pageable: the driver stages every one through its
  // own buffer and runs a copyBuffer KERNEL on the compute stream to do it, and the call is
  // synchronous with respect to the host, so the vectors cannot be reused without a
  // hipStreamSynchronize. That is hundreds of blits a cycle a rank, plus the dispatch gaps behind
  // them, plus a full stream drain per layer per card beside the router's.
  //
  // A ring rather than one buffer, because without the drain a slot is still being read by a DMA
  // when the next layer wants to write it; `bpin_ev_` is recorded after the copies and waited on
  // before the slot comes round again. Eight deep is far more than the host ever runs ahead, so the
  // wait is satisfied on arrival every time — it is there to be correct if that ever changes.
  static constexpr uint32_t kPinSlots = 8;
  void* bpin_[8] = {};                       // one contiguous block: kPinSlots * bpin_stride_
  void* bpin_ev_[8][kPinSlots] = {};
  size_t bpin_stride_[8] = {};
  uint32_t bpin_next_[8] = {};
  // Grows the arenas above to hold one layer's slots. Returns false if any allocation fails, in
  // which case the caller must not launch — the tiles point into buffers that do not exist.
  bool grow_w8a8(size_t dv, uint32_t taken, uint32_t n_tiles, uint32_t nb_tok, uint32_t hidden,
                 uint32_t inter, uint32_t nb_stride);

  struct Impl;
  Impl* impl_ = nullptr;                                  // HIP handles, kept out of this header
};

} // namespace aff
