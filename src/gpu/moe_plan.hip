// The MoE dispatch plan, built on the device. See moe_plan.h for what this replaces and why.
//
// ONE WORKGROUP, because every phase here is a scan or a histogram over at most a few thousand
// values and the whole thing is bounded by its barriers rather than its arithmetic. A multi-CTA
// form would need a grid-wide barrier between each phase, and a cooperative launch's barrier costs
// more than the entire host pass it is replacing.
//
// THE FOUR THINGS THAT FIX THE BITS, restated because every phase below exists to reproduce one of
// them exactly:
//
//   1. `uniq` = resident experts ascending, then pooled experts ascending      -> phase B
//   2. `counts[e]`, and `start[e]` as the prefix over that order               -> phase A, B
//   3. within an expert, slots taken in ascending slot index                   -> phase C
//   4. `ord`: per token, its slots in ascending arena column                   -> phase E
//
// Anything else — the tile order, how an expert's slots split across tiles — cannot change a bit:
// the expert GEMM's epilogue STORES to disjoint arena columns and its K-loop order is fixed.

#include "gpu/moe_plan.h"

#include <hip/hip_runtime.h>

#include "gpu/hip_common.h"

namespace aff {
namespace {

constexpr uint32_t kPlanThreads = 1024;

// LDS a workgroup may claim on the current device, cached per device because this is consulted on
// the dispatch path and hipGetDeviceProperties is a driver call. 64 KiB is RDNA's figure and the
// fallback when the query fails.
size_t device_lds_per_block() {
  static size_t cached[kMaxGpuDevices] = {};
  int d = 0;
  if (hipGetDevice(&d) != hipSuccess || d < 0 || d >= kMaxGpuDevices) return 64u * 1024u;
  if (!cached[d]) {
    hipDeviceProp_t p{};
    cached[d] = (hipGetDeviceProperties(&p, d) == hipSuccess && p.sharedMemPerBlock > 0)
                    ? (size_t)p.sharedMemPerBlock
                    : (size_t)64u * 1024u;
  }
  return cached[d];
}

// Tokens per pass of phase C. A token's top-k experts are DISTINCT, so within a group of 64 tokens
// an expert is named by at most one slot per token and a 64-bit mask per expert says exactly which.
// That is what turns rank-within-an-expert from a sequential cursor into a popcount.
constexpr uint32_t kPlanGroup = 64;
static_assert(kPlanGroup <= 64, "phase C's per-expert group mask is 64 bits");

// `nt <= 16 ? 0 : nt <= 32 ? 1 : nt <= 64 ? 2 : 3`, byte for byte the host's lambda. Bucket 3 is
// unreachable while tiles split at kExpertWmmaTok = 64; it is kept so the two agree by construction
// rather than by an argument about reachability.
__device__ inline uint32_t plan_bucket(uint32_t nt) {
  return nt <= 16u ? 0u : nt <= 32u ? 1u : nt <= 64u ? 2u : 3u;
}

// Exclusive prefix sum over `nrow` independent rows of `n` values, laid out row-major with stride
// `n`. Hillis-Steele: O(n log n) work, but n is at most a few thousand and the alternative is a
// tree with twice the barriers. `tmp` must hold nrow*n.
__device__ void wg_excl_scan(uint32_t* __restrict__ a, uint32_t* __restrict__ tmp, uint32_t n,
                             uint32_t nrow) {
  const uint32_t tot = n * nrow;
  for (uint32_t off = 1; off < n; off <<= 1) {
    for (uint32_t i = threadIdx.x; i < tot; i += kPlanThreads) {
      const uint32_t c = i % n;
      tmp[i] = a[i] + (c >= off ? a[i - off] : 0u);
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < tot; i += kPlanThreads) a[i] = tmp[i];
    __syncthreads();
  }
  for (uint32_t i = threadIdx.x; i < tot; i += kPlanThreads)
    tmp[i] = (i % n) ? a[i - 1] : 0u;                      // inclusive -> exclusive
  __syncthreads();
  for (uint32_t i = threadIdx.x; i < tot; i += kPlanThreads) a[i] = tmp[i];
  __syncthreads();
}

// The dynamic LDS block, carved in one place so the host's size calculation and the kernel's
// addressing cannot drift.
struct PlanLds {
  uint32_t* counts;      // [n_expert] slots routed to each expert
  uint32_t* start;       // [n_expert] that expert's first arena column
  uint32_t* rank;        // [n_expert] its index in `uniq`, or kNone
  uint32_t* fill;        // [n_expert] running slot cursor across phase C's groups
  uint32_t* uniq;        // [n_expert] the expert at each index, residents first
  uint32_t* tc;          // [8][n_expert] tiles each uniq entry puts in each (class, bucket)
  uint32_t* run;         // [nb_tok+1]
  uint32_t* tmp;         // scan scratch, the largest of the above
  unsigned long long* gm;  // [n_expert] phase C's per-group token mask
};

__device__ inline PlanLds plan_lds(char* raw, uint32_t n_expert, uint32_t nb_tok) {
  PlanLds L;
  uint32_t* p = (uint32_t*)raw;
  L.counts = p; p += n_expert;
  L.start = p; p += n_expert;
  L.rank = p; p += n_expert;
  L.fill = p; p += n_expert;
  L.uniq = p; p += n_expert;
  L.tc = p; p += 8u * n_expert;
  L.run = p; p += nb_tok + 1u;
  const uint32_t tmpn = (8u * n_expert > nb_tok + 1u) ? 8u * n_expert : nb_tok + 1u;
  L.tmp = p; p += tmpn;
  // 8-byte aligned by construction: every block above is a whole number of uint32 and the base of a
  // dynamic LDS allocation is 16-byte aligned.
  p += ((uintptr_t)p & 7u) ? 1u : 0u;
  L.gm = (unsigned long long*)p;
  return L;
}

constexpr uint32_t kNone = 0xffffffffu;

__global__ __launch_bounds__(kPlanThreads) void moe_plan_kernel(MoePlanArgs a) {
  extern __shared__ char raw[];
  const uint32_t NE = a.n_expert, NB = a.nb_tok, K = a.k, TOTAL = NB * K;
  PlanLds L = plan_lds(raw, NE, NB);
  const uint64_t* __restrict__ plc = a.plc + (size_t)a.layer * NE;

  // ---- A. histogram ----------------------------------------------------------------------------
  for (uint32_t e = threadIdx.x; e < NE; e += kPlanThreads) {
    L.counts[e] = 0u; L.fill[e] = 0u; L.gm[e] = 0ull; L.rank[e] = kNone;
  }
  __syncthreads();
  for (uint32_t i = threadIdx.x; i < TOTAL; i += kPlanThreads) {
    const uint32_t e = a.sel[i];
    if (e < NE) atomicAdd(&L.counts[e], 1u);
  }
  __syncthreads();

  // ---- B. uniq, and the prefix that fixes start[e] ----------------------------------------------
  //
  // The host builds `uniq` by scanning the slot list once per class and sorting each run. Walking
  // the experts in order instead produces the identical list with no sort, because "ascending
  // within the class" is the whole of the ordering.
  //
  // `tc` is borrowed as the scan input: row 0 marks residents, row 1 marks pooled.
  for (uint32_t e = threadIdx.x; e < NE; e += kPlanThreads) {
    const uint64_t pv = plc[e];
    const uint32_t live = L.counts[e] ? 1u : 0u;
    const uint32_t res = (pv & 1ull) ? 1u : 0u;
    L.tc[e] = live & res;                                  // row 0: resident and routed to
    L.tc[NE + e] = live & (res ^ 1u) & (pv ? 1u : 0u);     // row 1: pooled and routed to
    // NEITHER TIER. The engine's invariant is that no expert reachable by a live dispatch is on
    // disk, and the host pass aborts when one is. A kernel cannot abort, so it records and the host
    // reads this whenever it next looks — which is not on the dispatch path. `counts` is cleared
    // with it so every phase downstream behaves as though the expert were not routed to at all,
    // rather than handing it an arena column and a null base pointer.
    if (live && !pv) { atomicAdd(&a.status->bad, 1u); L.counts[e] = 0u; }
  }
  __syncthreads();
  wg_excl_scan(L.tc, L.tmp, NE, 2u);
  // The scan is exclusive, so the totals have to be recovered from the last entry plus its own flag,
  // and that flag is gone. Recompute both counts directly; NE is 256 and this is one pass.
  __shared__ uint32_t s_nres, s_npool, s_taken, s_ntile[2];
  if (!threadIdx.x) { s_nres = 0u; s_npool = 0u; s_taken = 0u; s_ntile[0] = 0u; s_ntile[1] = 0u; }
  __syncthreads();
  for (uint32_t e = threadIdx.x; e < NE; e += kPlanThreads) {
    const uint64_t pv = plc[e];
    if (!L.counts[e]) continue;
    atomicAdd(&s_taken, L.counts[e]);
    if (pv & 1ull) atomicAdd(&s_nres, 1u);
    else if (pv) atomicAdd(&s_npool, 1u);
  }
  __syncthreads();
  const uint32_t n_res = s_nres, n_uniq = s_nres + s_npool;
  for (uint32_t e = threadIdx.x; e < NE; e += kPlanThreads) {
    const uint64_t pv = plc[e];
    if (!L.counts[e] || !pv) continue;
    const uint32_t j = (pv & 1ull) ? L.tc[e] : n_res + L.tc[NE + e];
    L.rank[e] = j;
    L.uniq[j] = e;
  }
  __syncthreads();
  // start[e] = prefix of counts in uniq order. Permute into `tmp`, scan, permute back.
  for (uint32_t j = threadIdx.x; j < n_uniq; j += kPlanThreads) L.start[j] = L.counts[L.uniq[j]];
  for (uint32_t j = n_uniq + threadIdx.x; j < NE; j += kPlanThreads) L.start[j] = 0u;
  __syncthreads();
  wg_excl_scan(L.start, L.tmp, NE, 1u);
  // `start` is indexed by uniq position at this point; fold it back onto the expert id.
  for (uint32_t j = threadIdx.x; j < n_uniq; j += kPlanThreads) L.tmp[L.uniq[j]] = L.start[j];
  __syncthreads();
  for (uint32_t e = threadIdx.x; e < NE; e += kPlanThreads) L.start[e] = L.tmp[e];
  __syncthreads();

  // ---- C. the arena column of every slot --------------------------------------------------------
  //
  // Column = start[e] + (how many earlier slots chose e). The host keeps a per-expert cursor and
  // walks the slots in order; a kernel cannot, so the tokens are taken kPlanGroup at a time and the
  // rank inside a group comes from a popcount. That is exact rather than approximate because a
  // token's top-k experts are distinct: expert e is named by at most one slot per token, so one bit
  // per token is a complete record of the group, and ascending token order IS ascending slot order.
  for (uint32_t g0 = 0; g0 < NB; g0 += kPlanGroup) {
    const uint32_t gn = (NB - g0 < kPlanGroup) ? NB - g0 : kPlanGroup;
    for (uint32_t i = threadIdx.x; i < gn * K; i += kPlanThreads) {
      const uint32_t e = a.sel[(size_t)(g0 + i / K) * K + i % K];
      if (e < NE && L.counts[e]) atomicOr(&L.gm[e], 1ull << (i / K));
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < gn * K; i += kPlanThreads) {
      const uint32_t b = i / K, t = g0 + b;
      const size_t s = (size_t)t * K + i % K;
      const uint32_t e = a.sel[s];
      if (e >= NE || !L.counts[e]) { a.slot_col[s] = kNone; continue; }
      const unsigned long long below = L.gm[e] & ((1ull << b) - 1ull);
      const uint32_t col = L.start[e] + L.fill[e] + (uint32_t)__popcll(below);
      a.slot_col[s] = col;
      a.tok[col] = t;
      a.wt_out[col] = a.wt[s];
    }
    __syncthreads();
    for (uint32_t e = threadIdx.x; e < NE; e += kPlanThreads) {
      L.fill[e] += (uint32_t)__popcll(L.gm[e]);
      L.gm[e] = 0ull;
    }
    __syncthreads();
  }
  const uint32_t taken = s_taken;
  // The down tiles read the SwiGLU arena, which is indexed by slot rather than by token, so their
  // token list is the identity.
  for (uint32_t s = threadIdx.x; s < taken; s += kPlanThreads) a.tok[taken + s] = s;

  // ---- D. run: the per-token slot prefix ---------------------------------------------------------
  // The scan input is the per-token COUNT at index t — not at t+1, which is where the host
  // accumulates it before summing in place. An exclusive scan of the counts gives
  // run[t] = sum of counts below t directly, and run[NB] = taken, which is what run[] means:
  // token t owns ord[run[t] .. run[t+1]).
  for (uint32_t t = threadIdx.x; t <= NB; t += kPlanThreads) L.run[t] = 0u;
  __syncthreads();
  for (uint32_t t = threadIdx.x; t < NB; t += kPlanThreads) {
    uint32_t n = 0;
    for (uint32_t j = 0; j < K; ++j)
      if (a.slot_col[(size_t)t * K + j] != kNone) ++n;
    L.run[t] = n;
  }
  __syncthreads();
  wg_excl_scan(L.run, L.tmp, NB + 1u, 1u);
  for (uint32_t t = threadIdx.x; t <= NB; t += kPlanThreads) a.run[t] = L.run[t];

  // ---- E. ord: each token's slots in ascending column --------------------------------------------
  //
  // At most K per token, so an insertion sort in registers by one thread per token. K is 6 here and
  // capped by the router; the loop below is 36 comparisons at that width.
  for (uint32_t t = threadIdx.x; t < NB; t += kPlanThreads) {
    uint32_t c[kMoePlanMaxK], n = 0;
    for (uint32_t j = 0; j < K; ++j) {
      const uint32_t v = a.slot_col[(size_t)t * K + j];
      if (v == kNone) continue;
      uint32_t m = n++;
      while (m && c[m - 1] > v) { c[m] = c[m - 1]; --m; }
      c[m] = v;
    }
    const uint32_t base = L.run[t];
    for (uint32_t m = 0; m < n; ++m) a.ord[base + m] = c[m];
  }

  // ---- F. tiles ----------------------------------------------------------------------------------
  //
  // Expert e with counts[e] = q*64 + r puts q full tiles in bucket 2 and, when r is non-zero, one
  // tile of width r in bucket(r). `tc[cls*4 + b][j]` is what uniq entry j contributes to each, and
  // the scan over j turns that into the offset of j's first tile in that group — which reproduces
  // the host's stable sort, whose order within a group is exactly (uniq index, then offset).
  __syncthreads();
  for (uint32_t i = threadIdx.x; i < 8u * NE; i += kPlanThreads) L.tc[i] = 0u;
  __syncthreads();
  for (uint32_t j = threadIdx.x; j < n_uniq; j += kPlanThreads) {
    const uint32_t e = L.uniq[j], c = L.counts[e], cls = (j < n_res) ? 0u : 1u;
    const uint32_t q = c / kExpertWmmaTok, r = c % kExpertWmmaTok;
    // One thread owns j, so these accumulate rather than race — and they DO have to accumulate:
    // when the remainder is 33..64 wide it is a bucket-2 tile too, and lands in the same row as the
    // full ones.
    L.tc[(cls * 4u + 2u) * NE + j] += q;
    if (r) L.tc[(cls * 4u + plan_bucket(r)) * NE + j] += 1u;
  }
  __syncthreads();
  // The per-group totals, needed both for `bbeg` and for the pad. Recomputed rather than recovered
  // from the scan, for the same reason as in phase B.
  __shared__ uint32_t s_gtot[8];
  for (uint32_t b = threadIdx.x; b < 8u; b += kPlanThreads) s_gtot[b] = 0u;
  __syncthreads();
  for (uint32_t i = threadIdx.x; i < 8u * NE; i += kPlanThreads)
    if (L.tc[i]) atomicAdd(&s_gtot[i / NE], L.tc[i]);
  __syncthreads();
  wg_excl_scan(L.tc, L.tmp, NE, 8u);
  if (!threadIdx.x)
    for (uint32_t cls = 0; cls < 2; ++cls)
      s_ntile[cls] = s_gtot[cls * 4] + s_gtot[cls * 4 + 1] + s_gtot[cls * 4 + 2] +
                     s_gtot[cls * 4 + 3];
  __syncthreads();

  const uint32_t cap = a.tile_cap;
  for (uint32_t j = threadIdx.x; j < n_uniq; j += kPlanThreads) {
    const uint32_t e = L.uniq[j], c = L.counts[e], cls = (j < n_res) ? 0u : 1u;
    const uint32_t q = c / kExpertWmmaTok, r = c % kExpertWmmaTok;
    const uint64_t pv = plc[e];
    uint8_t* const base = (uint8_t*)(uintptr_t)(pv & ~1ull);
    // gate, up, down for one tile, at the same index in each of the three planes. `grp` is the
    // (class, NT bucket) region — EIGHT of them, not two, because the bucket decides the template
    // the launch is instantiated for and the templates are not bit-identical to each other. A single
    // region per class forces one `max_nt` on every tile in it, which puts a 6-wide decode tile on
    // the arm a 64-wide one wants and changes the summation.
    auto emit = [&](uint32_t grp, uint32_t slot, uint32_t off, uint32_t nt) {
      const uint32_t at = grp * cap + slot;
      for (uint32_t pl = 0; pl < 3; ++pl) {
        ExpertTile& T = a.tiles[(size_t)pl * 8u * cap + at];
        T.data = base + a.shard.data_off[pl];
        T.fields = base + a.shard.scale_off[pl];
        T.rscales = (const uint16_t*)(base + a.shard.rscale_off[pl]);
        // gate and up read the chunk's token columns; down reads the slot arena, so its token list
        // is the identity half of `tok`.
        T.tok = a.tok + (pl == 2u ? taken : 0u) + off;
        T.y = (pl == 0u ? a.bg : pl == 1u ? a.bu : a.bydn) + off;
        T.nt = nt;
        T.ld = taken;
      }
    };
    // Within its own region a group starts at 0, so the scan offset IS the slot.
    const uint32_t s2 = L.tc[(cls * 4u + 2u) * NE + j];
    for (uint32_t i = 0; i < q; ++i)
      emit(cls * 4u + 2u, s2 + i, L.start[e] + i * kExpertWmmaTok, kExpertWmmaTok);
    if (r) {
      const uint32_t b = plan_bucket(r);
      // `+ q` when the remainder shares bucket 2 with this expert's own full tiles: the scan offset
      // counts all of j's tiles in the bucket, and the host emits them in ascending `off`, so the
      // remainder is the last of them and not the first. Without this the remainder overwrites the
      // expert's first full tile and one 64-wide tile of the dispatch is silently dropped.
      const uint32_t at = L.tc[(cls * 4u + b) * NE + j] + (b == 2u ? q : 0u);
      emit(cls * 4u + b, at, L.start[e] + q * kExpertWmmaTok, r);
    }
  }
  __syncthreads();

  // ---- G. the pad, and the status ----------------------------------------------------------------
  //
  // The host sizes the grid from `tile_cap`, which it can compute without the routing, and the
  // surplus is `nt = 0` — a tile the GEMM returns on before it builds its codebook.
  for (uint32_t g = 0; g < 8u; ++g) {
    for (uint32_t i = s_gtot[g] + threadIdx.x; i < cap; i += kPlanThreads)
      for (uint32_t pl = 0; pl < 3u; ++pl)
        a.tiles[(size_t)pl * 8u * cap + g * cap + i].nt = 0u;
  }
  // ---- the activation record ---------------------------------------------------------------------
  //
  // Written after the tiles, so a reader that sees the head has seen a dispatch that was fully
  // planned. The slot is chosen from the head the kernel is ABOUT to publish, which is safe because
  // one workgroup writes this and dispatches are ordered on the stream.
  if (a.act) {
    // Read ONCE. `head` is in pinned host memory, so every thread reading it is a separate uncached
    // PCIe round trip for the same word — the same trap the placement table's host mirror exists to
    // avoid, in the other direction.
    __shared__ unsigned long long s_head;
    if (!threadIdx.x) s_head = a.act->head;
    __syncthreads();
    const unsigned long long h = s_head;
    ActRecord* const rec = &a.act->slot[h % kActSlots];
    for (uint32_t e = threadIdx.x; e < kActExperts; e += kPlanThreads)
      rec->counts[e] = e < NE ? L.counts[e] : 0u;
    __syncthreads();
    if (!threadIdx.x) {
      rec->layer = a.layer;
      // System scope: the reader is the CPU, and a device-scope fence would leave the record in a
      // cache the host cannot see while the head says it is there.
      __threadfence_system();
      a.act->head = h + 1ull;
    }
  }
  if (!threadIdx.x) {
    a.status->taken = taken;
    a.status->n_res = n_res;
    a.status->n_uniq = n_uniq;
    a.status->n_tiles = s_ntile[0] + s_ntile[1];
    for (uint32_t b = 0; b < 8u; ++b) a.counts[b] = s_gtot[b];
    // An overflowing class is a bug in the host's bound, and it has already silently dropped tiles
    // by the time anyone can read this — so it is recorded beside `bad` rather than clamped, and
    // the host treats a non-zero value as fatal.
    for (uint32_t g = 0; g < 8u; ++g) if (s_gtot[g] > cap) a.status->over = 1u;
  }
}

// One wave, one store each. See the note in moe_plan.h: this exists so that a placement edit lands
// at the point in the QUEUE the host published it from, rather than at the point in TIME.
__global__ void plc_publish_kernel(uint64_t* __restrict__ plc, PlcPatch p) {
  const uint32_t i = threadIdx.x;
  if (i < p.n) plc[p.idx[i]] = p.val[i];
}

} // namespace

bool plc_publish_hip(uint64_t* plc, const PlcPatch& p, void* stream) {
  if (!plc || p.n > kPlcPatchMax) return false;
  if (!p.n) return true;
  hipLaunchKernelGGL(plc_publish_kernel, dim3(1), dim3(kPlcPatchMax), 0, (hipStream_t)stream, plc, p);
  return true;
}

size_t moe_plan_lds_bytes(uint32_t n_expert, uint32_t nb_tok) {
  const uint32_t tmpn = (8u * n_expert > nb_tok + 1u) ? 8u * n_expert : nb_tok + 1u;
  return (size_t)(5u * n_expert + 8u * n_expert + nb_tok + 1u + tmpn) * sizeof(uint32_t) + 8u +
         (size_t)n_expert * sizeof(unsigned long long);
}

bool moe_plan_hip(const MoePlanArgs& a, void* stream) {
  // Refusals, not fallbacks. A silent fallback here is how an A/B measures one arm twice.
  if (a.n_expert > kMoePlanMaxExperts || a.k > kMoePlanMaxK || !a.nb_tok || !a.k) return false;
  // A ring that cannot hold the model's experts would silently starve the placement engine, which
  // presents as an engine that has stopped promoting rather than as a refusal.
  if (a.act && a.n_expert > kActExperts) return false;
  if (!a.shard.stride) return false;      // the tile records are built from the shard layout
  // Against what the card will actually give a workgroup, not a literal. Queried once a device:
  // this is on the dispatch path and hipGetDeviceProperties is a driver call.
  const size_t lds = moe_plan_lds_bytes(a.n_expert, a.nb_tok);
  if (lds > device_lds_per_block()) return false;
  hipLaunchKernelGGL(moe_plan_kernel, dim3(1), dim3(kPlanThreads), lds, (hipStream_t)stream, a);
  return true;
}

} // namespace aff
