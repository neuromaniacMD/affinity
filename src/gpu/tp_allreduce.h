// N-rank all-reduce across gfx12 cards, for tensor-parallel decode.
//
// TP is worth a large multiple at B=1: it divides the dominant cost — reading the dense weights —
// and lets an expert miss stream over every PCIe link at once. What it costs is an activation
// exchange twice per layer, and that cost is LATENCY rather than bandwidth: the crossing is flat
// across every size the engine exchanges, so the total over all the layers is small against what is
// saved.
//
// ---- why this stayed a one-shot push at N ranks --------------------------------------------------
//
// The push is O(N^2) in crossings — every rank sends its whole message to every peer — where
// reduce-scatter + allgather is 2(N-1)/N and wins on bytes from N=3 up. It is still the right shape
// here, because the exchange this engine actually issues is SMALL and its cost is the handshake, not
// the transfer: the decode exchange is 16 KiB, well under kTpFp8MinBytes, and reduce-scatter turns
// one round trip into two. Trading a latency doubling for a bandwidth saving is the wrong direction
// at that size.
//
// It is the wrong shape for the FP8 prefill exchange, which is tens of MiB and genuinely
// bandwidth-bound. That is where reduce-scatter belongs, and it is unbuilt because no box here has
// more than two cards to measure it on — at N=2 the two algorithms move identical bytes and the push
// wins on latency, so there is nothing to compare.
//
// Adapted from vllm-radiance's radiance_ar_ext; tp_allreduce.hip records what carried over and what
// a single-process engine does not need.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

// hipStream_t without dragging the HIP headers into CPU translation units.
struct ihipStream_t;

namespace aff {

// Reduction precision for the CROSSING.
//
// Both modes leave the two ranks holding the SAME BITS, which is what makes the result a replicated
// tensor rather than two similar ones. Exact gets it from commutativity; Fp8 gets it by summing the
// quantised form of both halves on both cards, rather than keeping the local half exact. Keeping it
// exact is more accurate per rank and was what this did, but it left the cards disagreeing by more
// than either one's own error — and the KV rows each card writes from that tensor inherit it.
//
//   Exact — f32 across the wire.
//   Fp8   — block-scaled e4m3, ~3.9x fewer bytes (1 B/elem + one f32 scale per 128).
//   Auto  — Fp8 only above kTpFp8MinBytes, Exact below.
//
// Small exchanges are LATENCY-bound — the time barely moves with size against a fixed floor — so
// shrinking the payload has almost nothing to shrink. FP8 pays from `kTpFp8MinBytes` up and
// strongly above that; vllm-radiance gates at the same threshold.
//
// The decode-path exchange is below it, so Auto leaves it Exact. FP8 is for prefill and for any
// batch above one.
enum class TpReduce { Exact, Fp8, Auto };
constexpr size_t kTpFp8MinBytes = 128 * 1024;
constexpr int kTpFp8Group = 128;            // elements sharing one f32 scale

constexpr int kTpMaxBlocks = 64;
// 16-byte words per block. Below this a message is latency-bound and extra blocks only add flag
// traffic; the whole point at 16 KiB is to use ONE block and pay the handshake once.
constexpr int kTpWordsPerBlock = 1400;

// Per-rank table bound. Deliberately a separate constant from dense_gpu.h's kMaxRanks so this header
// stays free of the engine's includes — tp_allreduce.hip asserts the two are equal.
constexpr size_t kTpMaxRanks = 8;
constexpr size_t kTpMaxPeers = kTpMaxRanks - 1;

// Where rank `p`'s contribution sits in rank `r`'s scratch, and which of `r`'s flags `p` releases.
// A rank keeps no sub-slot for itself, so the R-1 peers pack down to [0, R-1) with the rank's own
// index removed — and because the map is ascending in `p` on every rank, the reduce below can walk
// sub-slots in storage order and still sum peers in ascending rank order on all of them.
constexpr size_t tp_peer_slot(size_t p, size_t r) { return p < r ? p : p - 1; }

// Spin bound for EVERY kernel that waits on a peer's flag, here so there is one definition of it.
// Giving up means reducing against whatever is in the scratch, so a caller that breaks out must
// record it — see `timeout` below.
constexpr unsigned long long kTpSpinMax = 1ull << 32;

struct TpContext {
  size_t ranks = 0;                           // live rank count; every loop here runs to this
  int dev[kTpMaxRanks] = {};
  ihipStream_t* stream[kTpMaxRanks] = {};
  bool own_stream = true;                     // false when the caller supplied its own
  // ---- HOW MANY SLOTS, AND WHY TWO IS ENOUGH AT ANY LOOKAHEAD ------------------------------------
  //
  // Slots are not free: one holds a whole message, `n_embd * nb` floats, which at a prefill chunk
  // is tens of MiB EACH, on every card and once per peer. Deepening the ring costs that per slot and
  // buys nothing.
  //
  // THE BOUND IS THE PROTOCOL, NOT THE HOST. Per block b, a rank finishes collective s only after
  // EVERY one of its peers' flags for b has reached s, and nothing sets those except those peers'
  // pushes of s. So a rank cannot begin pushing s+1 until every peer has pushed s, and for any two
  // ranks |s_A - s_B| <= 1 at every instant no matter how many collectives the host has queued.
  // With that:
  //
  //   s_A == s_B      A's push into B's slot is exactly the one B is waiting for
  //   s_A == s_B +- 1 the two slots differ, for TWO slots already
  //
  // and a third case does not exist. The queue depth never enters the argument, and neither does the
  // rank count: the bound is pairwise and holds over all N(N-1) ordered pairs for the same reason it
  // held over the one.
  static constexpr uint32_t kTpSlots = 2;
  // kTpSlots * (ranks-1) sub-slots each, laid out [slot][peer][max_floats], fine-grained (uncached).
  // A rank keeps no sub-slot for its own contribution — that one is read from `in` directly.
  float* scratch[kTpMaxRanks] = {};
  // [peer][block], fine-grained. Peer p releases rank r's flag at tp_peer_slot(p, r).
  unsigned* flags[kTpMaxRanks] = {};
  unsigned* seq[kTpMaxRanks] = {};            // device-resident per-block counters
  size_t max_floats = 0;                      // per sub-slot, i.e. one whole message
  bool ready = false;

  // ---- a SECOND, disjoint set of the same three buffers -------------------------------
  //
  // For callers that run the exchange inside their own kernel rather than through
  // tp_all_reduce — see hc_expand_ar_kernel. They cannot share the buffers above, and the reason
  // is the per-block protocol rather than capacity.
  //
  // `seq[b]` advances only on calls whose grid reaches block b, and the double buffer is
  // `seq[b] & 1`, so two callers with DIFFERENT block counts drift out of phase with each other:
  // decode's 1-block reduce covers the whole message from block 0, prefill's covers 1/40 of a
  // much longer one, and a folded 16-block call covers 1/16. Once the parities disagree, one
  // caller's push can land in the slot a straggler on the other card is still draining, and the
  // corruption is silent. Same-geometry callers are safe because both ranks step in lockstep;
  // mixed-geometry ones are only safe if they never overlap, which is not a thing this file can
  // promise. A disjoint set makes the question not arise.
  //
  // TWO RANKS ONLY. The kernels that use these embed the single-peer protocol — one peer pointer,
  // one flag, `1 - r` for who to push to — inside a kernel that is doing something else, and there
  // is no N-peer form of them. tp_fold_prepare refuses above two ranks and the callers fall back to
  // the standalone tp_all_reduce, which costs the launch the fold exists to save. See
  // dense_gpu.hip's hc_expand_ar_kernel.
  float* fscratch[2] = {nullptr, nullptr};
  unsigned* fflags[2] = {nullptr, nullptr};
  unsigned* fseq[2] = {nullptr, nullptr};
  size_t fold_floats = 0;                     // per slot; 0 until tp_fold_prepare succeeds

  // ---- WHAT HAPPENS WHEN THE PEER NEVER ARRIVES ---------------------------------------------------
  //
  // The spin has a bound, because a rank that waits forever wedges the card and takes a reboot to
  // clear. But breaking out of it and reducing anyway means summing whatever is in the scratch —
  // fluent output with a new md5, which is the failure mode this project has spent the most time on
  // and the one it can least afford to have SILENTLY.
  //
  // So the break records itself here, in pinned host memory the kernel reaches with a system-scope
  // store, and the host reads it with a plain load — no copy, no stream and no synchronise, the same
  // discipline the plan's status word uses. It is checked periodically and at teardown, and it is
  // fatal: the point is that the process stops rather than emitting a whole run of wrong text, not
  // that it stops on the exact collective.
  //
  // The bound is ~4 billion iterations of an uncached peer load, which is hours. Nothing legitimate
  // reaches it, so a non-zero value here is always a bug and never a slow peer.
  unsigned* timeout = nullptr;                // pinned, host-readable; 0 = the spin never gave up
  unsigned* timeout_dev = nullptr;            // the same word as the CARDS address it
};

// Sets up peer access, streams and the shared scratch. `max_floats` bounds a single all-reduce.
//
// `ext` supplies the streams to reduce on instead of creating them. The engine passes its own
// per-rank streams: the reduction consumes what the previous kernel on that stream produced, and on
// a stream of its own that ordering would need an explicit event pair per call — 86 of them a
// token, to state something the stream already knew. Standalone callers (the benchmark, the test)
// pass null and get their own.
//
// `n_ranks`/`devs` state the rank plan. Null/0 means "enumerate gfx12 devices and use all of them",
// which is what the standalone callers want. The engine passes its own plan, because under
// AFF_LOGICAL_RANKS the plan is not the device list — the same card can back several ranks, and two
// ranks that share a device must not try to enable peer access to themselves.
bool tp_init(TpContext* ctx, size_t max_floats, std::string* err,
             ihipStream_t* const* ext = nullptr,
             size_t n_ranks = 0, const int* devs = nullptr);
void tp_free(TpContext* ctx);
// Grow the staging scratch to hold a bigger message. Cheap no-op when it already fits. Both ranks
// must be idle: the scratch is what an in-flight reduction is writing into.
bool tp_grow(TpContext* ctx, size_t max_floats, std::string* err);

// Allocate the disjoint fold buffers for a message of `floats` elements. Idempotent and cheap once
// they fit; the first call is the only one that allocates. Both ranks must be idle, for the same
// reason tp_grow says. A caller that gets `true` may read ctx.fscratch/fflags/fseq and hand them to
// its own kernel; the protocol it must implement is the one in tp_ar_kernel, block for block.
//
// FALSE at any rank count but two — see the fold buffers above. A caller that gets false must issue
// the collective through tp_all_reduce instead; it is not an error and not a reason to refuse to
// run.
bool tp_fold_prepare(TpContext* ctx, size_t floats, std::string* err);

// in[r] and out[r] must live on device ctx->dev[r], for r in [0, ctx.ranks). Every rank ends with
// the elementwise sum. May alias (in[r] == out[r]).
//
// At one rank this is the identity: a no-op when the buffers alias, a copy when they do not.
//
// Does NOT synchronise. The kernels are ordered on their own streams, and a decode step issues
// many of these back to back with real work between them — synchronising per call would pay a
// round trip 86 times a token for no reason. Call tp_sync() when the result must be on the host.
void tp_all_reduce(TpContext& ctx, float* const* in, float* const* out, size_t n,
                   TpReduce mode = TpReduce::Auto);

// Blocks until both ranks' streams are drained.
void tp_sync(TpContext& ctx);

// Did a rank ever give up waiting for its peer? Reads pinned memory — no copy, no synchronise — so
// a caller may ask as often as it likes. Aborts with the offending sequence number rather than
// returning it: there is no correct way to continue from a collective that summed uninitialised
// scratch, and a caller that could handle it would have to reproduce this argument.
void tp_check(const TpContext& ctx);

} // namespace aff
