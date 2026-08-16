// The MoE dispatch plan, built on the device.
//
// WHAT THIS REPLACES. Per layer per card, `run_batch_device` does a host-side pass that decides
// which experts this dispatch touches, where each one's bytes are, how the routed (token, expert)
// slots are grouped, and what the expert GEMM's tile list looks like. That pass needs the router's
// selection, the router's selection lives on the device, and the only way the host gets it is a
// `hipStreamSynchronize` — which waits for the whole previous layer's chain including the hybrid
// tier's PCIe reads, once a layer.
//
// The host pass is not free either: it is tens of microseconds a call on the card that pays the cold
// misses, on top of the drain.
//
// WHAT MAKES IT POSSIBLE. Two things the engine already has:
//
//   * the placement table (static_placement.h) — one 64-bit word per (layer, expert) per card
//     holding that card's shard base address with the tier in bit 0, written by the CPU at the one
//     point placement changes. That is the whole of what the host pass was computing from
//     `device_of_`/`offset_of_`/`pool_shard_dev`.
//   * `ExpertTile` was ALREADY a device array read by `expert_gemm_wmma_kernel` through
//     `tiles[blockIdx.y]`. The host was only filling it in.
//
// WHAT FIXES THE BITS, and it is narrower than it looks. The GEMM's epilogue STORES rather than
// accumulates and tiles cover disjoint arena columns, so permuting the tile list cannot change a
// bit. Exactly four things can:
//
//   1. `uniq` order — resident ascending, then pooled ascending — which fixes `start[e]`;
//   2. `counts[e]` and the prefix over that order;
//   3. within an expert, slots taken in ascending slot index, which fixes each slot's column;
//   4. `ord`/`run` — per token, its slots in ascending arena column — which fixes the scatter's
//      reduction order.
//
// This kernel reproduces all four exactly, and reproduces the tile order too, so the differential
// check against the host pass is a `memcmp` rather than a set comparison.
//
// THE ONE SIMPLIFICATION WORTH KNOWING. A token's top-k experts are distinct, so an expert appears
// at most once per token. Ordering the slots of one expert by slot index is therefore the same as
// ordering them by token, which is what lets rank-within-an-expert be a popcount over a per-token
// bitmask instead of a sequential cursor.

#pragma once

#include <cstddef>
#include <cstdint>

#include "gpu/expert_kernel_hip.h"

namespace aff {

// Written by the kernel, read by the host only when it wants to know — never on the dispatch path.
struct MoePlanStatus {
  uint32_t taken;        // routed (token, expert) slots this card took
  uint32_t n_res;        // distinct resident experts; the rest of `uniq` is pooled
  uint32_t n_uniq;       // distinct experts touched
  uint32_t n_tiles;      // tiles emitted, both classes
  uint32_t bad;          // slots whose expert has no address: the SSD tier. Fatal, read lazily.
  uint32_t over;         // a class needed more tiles than `tile_cap`. Fatal: tiles were dropped.
};

// ---- THE ACTIVATION RING: what the cards tell the CPU about routing ------------------------------
//
// The placement engine's only input is which experts a dispatch touched and how many of the block's
// tokens agreed — `ExpertHeat::observe` takes exactly that. Reading it off the host's copy of the
// selection is what forces the selection back, which is what makes the dispatch drain.
//
// The plan kernel already has the answer: `counts[e]` is a by-product of its histogram. It appends
// one record here and bumps a monotone head; the CPU reads it whenever it next wakes, one or more
// dispatches behind, and never on the critical path in either direction.
//
// Indexed by expert rather than a (list, count) pair, which is the shape `observe` takes. The
// conversion costs one 256-iteration pass on the host and buys a fixed-size record — and it makes
// the distinct list ASCENDING, which is what `ExpertHeat::note_missing`'s own comment already claims
// it is and the host loop it replaces was not.
//
// DELIBERATELY LOSSY AND COUNTED. The writer never blocks and never checks for space; a reader that
// falls more than `kActSlots` behind loses records and `head - tail` says exactly how many. Heat's
// ranking is insensitive to a small fraction of dropped records — but a ring that silently drops is
// a lie, so the harness asserts the drop rate rather than trusting it.
constexpr uint32_t kActExperts = 256;
constexpr uint32_t kActSlots = 64;

struct ActRecord {
  uint32_t layer;
  uint32_t pad[3];                    // keeps `counts` 16-byte aligned
  uint32_t counts[kActExperts];       // slots per expert, 0 = not routed to
};

// `head` is on its own line and published last: the record's bytes are written, then a system-scope
// fence, then this. A reader that sees head > n has seen record n complete.
struct ActRing {
  ActRecord slot[kActSlots];
  alignas(128) unsigned long long head;
};

// ---- PUBLISHING A PLACEMENT EDIT, IN STREAM ORDER ------------------------------------------------
//
// The placement table is written by the CPU and read by the cards. While the HOST built the
// dispatch, those two happened at the same instant: the mover's store and the membership pass that
// read it were the same thread, so a layer's dispatch saw exactly the placement that existed when
// that layer was issued.
//
// A KERNEL reads the table when it RUNS, which is an unbounded distance after the host issued it —
// and how far is set by how much lookahead the host has. So a plain host store publishes into the
// middle of a queue of dispatches that have not executed yet, and a promotion lands on some layer
// the host never chose. Nothing reads bad bytes (the mover's copy has landed and the quarantine
// still holds the old slot), but the expert's CLASS changes, `uniq` is ordered resident-before-
// pooled, that order fixes `start[e]`, and `start[e]` fixes the summation order. Different text,
// from the same binary and the same seed.
//
// It is not a race: it is one host store read at a distance the drain used to pin to one layer, so
// each drain interval gives its own reproducible answer.
//
// So the publish goes on the compute stream instead, as a handful of 64-bit stores in one launch.
// Then "the table a layer sees" is again "the table as of the moment that layer was issued", which
// is the property the host build had for free, and the device build is bit-identical to it at ANY
// lookahead depth — which is also what unblocks issuing more than one block ahead.
//
// A tick's worth, in one launch: the mover retires up to `moves_per_tick` moves at a time, and one
// launch for all of them is the difference between paying per tick and paying per move.
// The patch travels as a kernel ARGUMENT — 196 bytes, well inside the 4 KiB limit — so there is no
// staging buffer to allocate, no copy to order, and nothing to keep alive after the launch.
constexpr uint32_t kPlcPatchMax = 16;

struct PlcPatch {
  uint32_t n = 0;
  uint32_t idx[kPlcPatchMax] = {};
  uint64_t val[kPlcPatchMax] = {};
};

// `plc` is one card's table. Returns false only if the patch is over-full, which is a caller bug.
bool plc_publish_hip(uint64_t* plc, const PlcPatch& p, void* stream);

// The tile array is [3 planes][8 groups][tile_cap], a group being (class, NT bucket) — so the host
// can name the base of any region without knowing the routing, which is the whole point.
//
// EIGHT groups and not two, and that is not a convenience. `expert_gemm_wmma_hip` picks its
// (NT, MT) instantiation from the `max_nt` it is handed, and the instantiations are NOT bit-identical
// to one another: MT decides `kASz`, `kASz` decides `kSingleA`, and the single-buffered arm keeps its
// accumulators local to a j-group where the double-buffered one carries them across. Putting a whole
// class in one region forces one `max_nt` on every tile in it — a 6-wide decode tile computed on the
// arm a 64-wide one wants — and the summation changes, so identical plans emit different text.
//
// `counts[cls*4 + bucket]` is how many tiles each region actually holds; the host launches the
// regions its `nb_tok` makes reachable and pads the rest.
struct MoePlanArgs {
  // ---- in, all device-resident -------------------------------------------------------------------
  const uint32_t* sel = nullptr;      // [nb_tok][k] the router's choice, padded with e >= n_expert
  const float*    wt = nullptr;       // [nb_tok][k]
  const uint64_t* plc = nullptr;      // [n_layer][n_expert] this card's placement table
  uint32_t layer = 0, nb_tok = 0, k = 0, n_expert = 0;

  // ---- out, into the arena the dispatch already owns ---------------------------------------------
  uint32_t* tok = nullptr;            // [2*taken]: token id per slot, then identity for `down`
  float*    wt_out = nullptr;         // [taken] the router weight per slot
  uint32_t* ord = nullptr;            // [taken] slots sorted by token
  uint32_t* run = nullptr;            // [nb_tok+1] prefix of per-token slot counts
  uint32_t* slot_col = nullptr;       // [nb_tok*k] scratch: the arena column of each slot
  ExpertTile* tiles = nullptr;        // [3][8][tile_cap]
  uint32_t* counts = nullptr;         // [8] tiles per (class, bucket)
  MoePlanStatus* status = nullptr;
  // The activation ring, or null. One card writes it — the two select the same experts, so a ring
  // per card would carry every dispatch twice with identical contents.
  ActRing* act = nullptr;

  // ---- the tile records' constants ---------------------------------------------------------------
  ExpertShardLayout shard{};
  uint16_t* bg = nullptr;             // gate/up/down output arenas, column-strided by `taken`
  uint16_t* bu = nullptr;
  uint16_t* bydn = nullptr;
  uint32_t tile_cap = 0;              // per-class capacity; see moe_plan_tile_cap
};

// One workgroup. `nb_tok` is handled in groups of 64 tokens, so decode is one group and a 1408-token
// prefill chunk is 22 — the same code either way.
//
// Returns false when the shape is outside what the kernel serves (n_expert or k above its bound, or
// the LDS block over what the device gives a workgroup), in which case the caller keeps the host
// pass. Refusing is deliberate: a silent fallback is how an A/B measures one arm twice.
bool moe_plan_hip(const MoePlanArgs& a, void* stream);

// The dynamic LDS the kernel needs, so the caller can refuse before launching rather than after.
size_t moe_plan_lds_bytes(uint32_t n_expert, uint32_t nb_tok);

// The largest `n_expert` and `k` the single-workgroup form serves. 256 and 6 are this model's.
constexpr uint32_t kMoePlanMaxExperts = 512;
constexpr uint32_t kMoePlanMaxK = 16;

// TILES ONE CLASS CAN NEED, from `nb_tok` and `k` alone — which is what lets the host size the grid
// without seeing the routing.
//
// An expert routed `c` slots emits `c/64` full tiles and, when `c % 64` is non-zero, one narrow one.
// Summing over the experts: the full tiles are bounded by `taken/64` because they are 64 slots each,
// and the narrow ones by the number of distinct experts. Either class could be the whole dispatch,
// so the bound holds for each separately.
constexpr uint32_t moe_plan_tile_cap(uint32_t nb_tok, uint32_t k, uint32_t n_expert) {
  const uint32_t taken = nb_tok * k;
  const uint32_t distinct = taken < n_expert ? taken : n_expert;
  return distinct + taken / kExpertWmmaTok;
}

// The `max_nt` region `b` must be launched with. Tiles split at kExpertWmmaTok = 64, so bucket b
// holds widths in (16b, 16b+16], and `expert_gemm_wmma_hip` selects on `(max_nt + 15) / 16` — which
// is b+1 for every width in that range. So this names the instantiation the true per-bucket maximum
// would have named, exactly, for every reachable bucket.
constexpr uint32_t moe_plan_group_max_nt(uint32_t group) { return 16u * ((group & 3u) + 1u); }

// The widest tile `nb_tok` can produce, and therefore the last bucket that can be non-empty. An
// expert cannot be routed the same token twice, so a tile is at most `min(nb_tok, 64)` wide.
// Launching only these keeps decode at the six launches a layer it makes today.
constexpr uint32_t moe_plan_last_group(uint32_t nb_tok) {
  const uint32_t w = nb_tok < kExpertWmmaTok ? nb_tok : kExpertWmmaTok;
  return w <= 16u ? 0u : w <= 32u ? 1u : 2u;
}

} // namespace aff
