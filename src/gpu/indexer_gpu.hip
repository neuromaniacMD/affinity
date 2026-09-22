// The lightning indexer on device. See indexer_gpu.h for the shape and why the scales are per row.

#include "gpu/indexer_gpu.h"
#include "gpu/device_math.h"
#include "gpu/hip_common.h"
#include "gpu/wmma.h"

#include <hip/hip_runtime.h>

#include <cmath>

namespace aff {
namespace {

constexpr uint32_t kD = kIdxDim;             // 128
constexpr uint32_t kWaves = 4;
constexpr uint32_t kThreads = kWaves * 32;   // 128
constexpr uint32_t kKT = kIdxKeys / kWaves / 16u;   // 2 key tiles a wave
constexpr uint32_t kHG = 8;                  // heads staged in LDS at once
// 136 is 34 dwords, and 34c mod 32 = 2c is a bijection over 16 lanes, so a fragment load hits 32
// distinct banks. A bare 128 would put every lane on bank 0.
constexpr uint32_t kQLd = kD + 8;            // 136

// ONE block: 16 tokens x 128 keys. The key fragments do not depend on the head, so they are loaded
// once into registers and reused across all 64 heads — which is what makes this worth doing as a
// matrix op at all. Q is staged in LDS a head-group at a time so all four waves read it once,
// rather than each wave pulling its own copy from global.
__global__ __launch_bounds__(kThreads) void indexer_kernel(
    const uint8_t* __restrict__ q8, const float* __restrict__ qs, const uint8_t* __restrict__ k8,
    const float* __restrict__ ks, const float* __restrict__ hw, float* __restrict__ out,
    uint32_t nb, uint32_t n_keys, uint32_t n_head) {

  __shared__ __align__(16) uint8_t s_q[kHG * 16u * kQLd];
  // The head weight is per (token, head) -- it is the indexer's own projection of this token -- so
  // unlike the two scales it cannot leave the head loop. Staged [head][token] so the eight values a
  // lane needs, one per accumulator row, are contiguous.
  __shared__ float s_w[kHG * 16u];

  const uint32_t tid = threadIdx.x, lane = tid & 31u, wave = tid >> 5;
  const uint32_t c = lane & 15u, hlf = lane >> 4;
  const uint32_t t0 = blockIdx.x * kIdxTokTile;
  const uint32_t kb = blockIdx.y * kIdxKeys + wave * (kKT * 16u);

  // ---- B: the keys. Loaded once, held for the whole head loop.
  wmma::i2v bf[kKT][kD / 16u];
#pragma unroll
  for (uint32_t n = 0; n < kKT; ++n) {
    const uint32_t key = kb + n * 16u + c;
    const uint8_t* kp = k8 + (size_t)(key < n_keys ? key : 0u) * kD + hlf * 8u;
#pragma unroll
    for (uint32_t d = 0; d < kD / 16u; ++d) bf[n][d] = *(const wmma::i2v*)(kp + d * 16u);
  }

  wmma::f8v acc_out[kKT];
#pragma unroll
  for (uint32_t n = 0; n < kKT; ++n) acc_out[n] = wmma::acc_zero();

  for (uint32_t hg = 0; hg < n_head; hg += kHG) {
    // Stage kHG heads of this token tile. 128 threads, kHG*16 rows of kD bytes: each thread copies
    // one 32-byte run, so the global side is contiguous per row and the LDS side is one b128 pair.
    __syncthreads();
#pragma unroll
    for (uint32_t i = 0; i < kHG * 16u * (kD / 32u) / kThreads; ++i) {
      const uint32_t idx = tid + i * kThreads;
      const uint32_t row = idx / (kD / 32u), dq = (idx % (kD / 32u)) * 32u;
      const uint32_t hh = row / 16u, tt = row % 16u;
      const uint32_t t = t0 + tt;
      const uint4* src = (const uint4*)(q8 + ((size_t)t * n_head + hg + hh) * kD + dq);
      uint4 a = make_uint4(0, 0, 0, 0), b = make_uint4(0, 0, 0, 0);
      if (t < nb) { a = src[0]; b = src[1]; }
      uint8_t* dst = s_q + (size_t)row * kQLd + dq;
      *(uint4*)dst = a;
      *(uint4*)(dst + 16) = b;
    }
    if (tid < kHG * 16u) {
      const uint32_t hh = tid / 16u, tt = tid % 16u, t = t0 + tt;
      s_w[hh * 16u + tt] = (t < nb) ? hw[(size_t)t * n_head + hg + hh] : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (uint32_t hh = 0; hh < kHG; ++hh) {
      float wv[8];
#pragma unroll
      for (int q = 0; q < 8; ++q) wv[q] = s_w[hh * 16u + hlf * 8u + (uint32_t)q];
      const uint8_t* qp = s_q + (size_t)(hh * 16u + c) * kQLd + hlf * 8u;
      wmma::i2v af[kD / 16u];
#pragma unroll
      for (uint32_t d = 0; d < kD / 16u; ++d) af[d] = *(const wmma::i2v*)(qp + d * 16u);
#pragma unroll
      for (uint32_t n = 0; n < kKT; ++n) {
        wmma::f8v a = wmma::acc_zero();
#pragma unroll
        for (uint32_t d = 0; d < kD / 16u; ++d) a = wmma::mma_fp8(af[d], bf[n][d], a);
        // ReLU then weighted sum. No softmax anywhere — the indexer produces a ranking signal, not
        // a distribution — and this is exactly where the head loop cannot be one accumulator chain.
#pragma unroll
        for (int q = 0; q < 8; ++q) acc_out[n][q] += wv[q] * fmaxf(a[q], 0.0f);
      }
    }
  }

  // ---- both scales ride out here: ReLU commutes with a positive factor, so neither had to be
  // inside the head loop. The row scale is indexed by the accumulator row, the column scale by the
  // lane's own column.
#pragma unroll
  for (uint32_t n = 0; n < kKT; ++n) {
    const uint32_t key = kb + n * 16u + c;
    if (key >= n_keys) continue;
    const float kscale = ks[key];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const uint32_t t = t0 + wmma::acc_row(lane, (uint32_t)q);
      if (t < nb) out[(size_t)t * n_keys + key] = acc_out[n][q] * qs[t] * kscale;
    }
  }
}


// ---- decode's shape: ONE query token ------------------------------------------------------------
//
// The kernel above tiles M = TOKENS, sixteen to a block, because prefill presents 1536 of them.
// Decode presents one, so fifteen sixteenths of every matrix op multiplies a zero-padded row. The
// kernel then runs at the matrix core's ceiling with almost all of it waste, while moving key bytes
// at a small fraction of the card's bandwidth — so the SHAPE is wrong, not the kernel.
//
// Tile M = HEADS instead. There are 64, exactly four 16-row tiles, and the accumulator then holds
// D[head][key] — eight heads a lane, the key in the lane's own column — which is the layout the sum
// over heads wants. The key fragment is loaded once into sixteen registers and feeds all four head
// tiles, so a key's 128 bytes are read exactly once.
//
// THIS REASSOCIATES THE SUM OVER HEADS, and therefore moves the answer. The host reference and
// the batched kernel both add the 64 heads in order; here a lane adds the eight it holds and the
// two halves of the wave are combined at the end, so a tile contributes (h0..h7) + (h8..h15) rather
// than h0, then h1, then h2. Exact order would need one lane to hold all sixteen, which the
// fragment layout does not offer, and a cross-lane chain that preserved it would cost fifteen
// dependent shuffles a tile — more than the matrix ops it is protecting.
//
// What that is worth: these scores are a RANKING signal, and only the top `index_topk` of them
// matter, so a last-ulp difference changes nothing except where two rows sit within an ulp of the
// admission threshold. Nothing else may move: this path is reached only by a decode stream deep
// enough to accumulate index_topk compressed rows, so every other hash must come back unchanged.
constexpr uint32_t kIdxDecHeads = 64;

__global__ __launch_bounds__(kThreads) void indexer_dec_kernel(
    const uint8_t* __restrict__ q8, const float* __restrict__ qs, const uint8_t* __restrict__ k8,
    const float* __restrict__ ks, const float* __restrict__ hw, float* __restrict__ out,
    uint32_t n_keys, uint32_t n_head) {
  __shared__ __align__(16) uint8_t s_q[kIdxDecHeads * kQLd];
  __shared__ float s_w[kIdxDecHeads];

  const uint32_t tid = threadIdx.x, lane = tid & 31u, wave = tid >> 5;
  const uint32_t c = lane & 15u, hlf = lane >> 4;

  // Q is n_head rows of 128 bytes and every block reads all of it, so it is staged once: 8 KB at
  // 16 bytes a thread a step, contiguous on both sides.
  for (uint32_t i = tid; i < n_head * (kD / 16u); i += kThreads) {
    const uint32_t row = i / (kD / 16u), dq = (i % (kD / 16u)) * 16u;
    *(uint4*)(s_q + (size_t)row * kQLd + dq) = *(const uint4*)(q8 + (size_t)row * kD + dq);
  }
  if (tid < n_head) s_w[tid] = hw[tid];
  __syncthreads();

  const uint32_t key = blockIdx.x * (kWaves * 16u) + wave * 16u + c;
  // An out-of-range key reads row 0 and is dropped on store rather than branched around: every lane
  // of the wave has to supply a defined operand for the fragment to be well formed.
  const uint8_t* kp = k8 + (size_t)(key < n_keys ? key : 0u) * kD + hlf * 8u;
  wmma::i2v bf[kD / 16u];
#pragma unroll
  for (uint32_t d = 0; d < kD / 16u; ++d) bf[d] = *(const wmma::i2v*)(kp + d * 16u);

  float sum = 0.0f;
  for (uint32_t mt = 0; mt < n_head; mt += 16u) {
    wmma::f8v acc = wmma::acc_zero();
    const uint8_t* qp = s_q + (size_t)(mt + c) * kQLd + hlf * 8u;
#pragma unroll
    for (uint32_t d = 0; d < kD / 16u; ++d)
      acc = wmma::mma_fp8(*(const wmma::i2v*)(qp + d * 16u), bf[d], acc);
    // ReLU then the head weight, exactly as above; the accumulator row IS the head here, where in
    // the batched kernel it is the token.
#pragma unroll
    for (int j = 0; j < 8; ++j) sum += s_w[mt + hlf * 8u + (uint32_t)j] * fmaxf(acc[j], 0.0f);
  }
  sum += __shfl_xor(sum, 16);                  // the wave's other eight heads a tile
  if (hlf == 0u && key < n_keys) out[key] = sum * qs[0] * ks[key];
}

// ---- f32 rows -> E4M3 + one scale a row, on device ----------------------------------------------
//
// One block a row. `dim` is 8192 for a query token (64 heads x 128) and 128 for a key.
__global__ void idx_quant_kernel(const float* __restrict__ src, uint8_t* __restrict__ dst,
                                 float* __restrict__ scale, uint32_t dim) {
  __shared__ float s_part[256];
  const uint32_t r = blockIdx.x, tid = threadIdx.x;
  const float* p = src + (size_t)r * dim;
  float m = 0.0f;
  for (uint32_t i = tid; i < dim; i += 256u) m = fmaxf(m, fabsf(p[i]));
  s_part[tid] = m;
  __syncthreads();
  for (uint32_t o = 128; o; o >>= 1) {
    if (tid < o) s_part[tid] = fmaxf(s_part[tid], s_part[tid + o]);
    __syncthreads();
  }
  const float sc = s_part[0] > 0.0f ? s_part[0] * (1.0f / 448.0f) : 1.0f;
  if (tid == 0) scale[r] = sc;
  const float inv = 1.0f / sc;
  uint8_t* d = dst + (size_t)r * dim;
  for (uint32_t i = tid * 4u; i < dim; i += 256u * 4u) {
    // Clamp before the convert: v_cvt_pk_fp8_f32 answers an out-of-range input with NaN rather than
    // saturating, and max*(1/(max/448)) can land an ulp above 448.
    float v[4];
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) v[j] = fminf(448.0f, fmaxf(-448.0f, p[i + j] * inv));
    int pk = 0;
    pk = __builtin_amdgcn_cvt_pk_fp8_f32(v[0], v[1], pk, false);
    pk = __builtin_amdgcn_cvt_pk_fp8_f32(v[2], v[3], pk, true);
    *(uint32_t*)(d + i) = (uint32_t)pk;
  }
}

// ---- top-k -> the admission mask, one block a token ---------------------------------------------
//
// Must match `top_k_select` exactly, which orders by score DESCENDING and breaks ties by the LOWER
// index — the golden fixture reproduces a greedy continuation and a different tie rule is a
// different set of admitted rows.
//
// Four 8-bit radix passes over a monotone map of the float bits give the k-th largest key exactly.
// Everything strictly above it is admitted outright; the ties at it are then filled in INDEX order
// by a single wave, which is where the determinism comes from — an atomic would admit whichever
// block got there first.
__device__ __forceinline__ uint32_t mono(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// A THOUSAND AND TWENTY-FOUR THREADS, not 256. At decode the grid is one token, so this kernel is
// one workgroup on the whole card whatever it does; the only parallelism available to it is that
// workgroup's own width, and a narrower block leaves most of the workgroup processor idle. At long
// context a narrow block costs several times the attention it feeds.
constexpr uint32_t kTopkThr = 1024;

// One LDS atomic per DISTINCT bucket in a wave, rather than one per lane.
//
// The first radix pass buckets on `mono(f) >> 24`, which for a positive float is the sign bit and
// the top seven bits of the exponent — indexer scores are all of a similar magnitude, so a whole
// wave routinely lands in ONE bucket and `atomicAdd` costs 32 serialised LDS slots instead of 1.
// Electing a leader per distinct value and adding a popcount collapses that. The loop is
// wave-uniform (its condition is a ballot), so lanes that are already done still take part in the
// ballots the survivors need.
__device__ __forceinline__ void hist_add(uint32_t* h, uint32_t bin, bool active) {
  const uint32_t lane = threadIdx.x & 31u;
  while (true) {
    const uint64_t act = __ballot(active);
    if (!act) break;
    const uint32_t first = (uint32_t)__ffsll((unsigned long long)act) - 1u;
    // readlane, not __shfl: the index is uniform (it came out of s_ff1 on an SGPR), and __shfl
    // does not always prove that to itself — it compiles to ds_bpermute, an LDS round trip, on the
    // one instruction this loop repeats per distinct bucket.
    const uint32_t v = (uint32_t)__builtin_amdgcn_readlane((int)bin, (int)first);
    const uint64_t same = __ballot(active && bin == v);
    if (active && bin == v) {
      if (lane == first) atomicAdd(&h[v], (uint32_t)__popcll(same));
      active = false;
    }
  }
}

// The descending walk over the 256 buckets, by the whole block instead of by thread 0.
//
// `h` becomes its own inclusive SUFFIX sum — h[j] = sum_{i>=j} h[i], which is non-increasing — and
// the bucket the k-th largest lives in is then the largest j with h[j] >= need, one atomicMax away.
// A serial walk by thread 0 is 256 LDS-latency-bound iterations with a carried dependency, four
// times a call; in the multi-block kernel below the same walk would read GLOBAL memory 256 times,
// and would have to read it coherently.
//
// `b` comes back as that bucket and `acc` as the count strictly above it.
__device__ __forceinline__ void hist_scan(uint32_t* h, uint32_t* s_b, uint32_t* s_any,
                                          uint32_t need, uint32_t tid,
                                          uint32_t* b, uint32_t* acc) {
  for (uint32_t off = 1; off < 256u; off <<= 1) {
    uint32_t add = 0;
    if (tid + off < 256u) add = h[tid + off];
    __syncthreads();
    if (tid < 256u) h[tid] += add;
    __syncthreads();
  }
  if (tid == 0) { *s_b = 0; *s_any = 0; }
  __syncthreads();
  if (tid < 256u && h[tid] >= need) { atomicMax(s_b, tid); atomicExch(s_any, 1u); }
  __syncthreads();
  const uint32_t bb = *s_b;
  *b = bb;
  // No bucket reaching `need` cannot happen — the total is at least k at the first pass and at
  // least `need` at every later one — but the serial walk fell out with the whole sum, so match it.
  *acc = *s_any ? (bb + 1u < 256u ? h[bb + 1u] : 0u) : h[0];
}

// Admitting the ties at T, in INDEX order, `need` of them — the rule the CPU's top_k_select uses
// and the one the golden fixture depends on.
//
// The obvious implementation walks the row from index 0 and stops once the quota is filled: ONE
// WAVE, on a serial dependent chain, over however far into the row the k-th largest happens to sit.
// Exact float equality is rare, so `need` is normally 1 and the expected walk is HALF THE ROW, which
// at long context dominates the whole select once the histogram passes are spread across the card
// and stop hiding it.
//
// The marking pass reads every score against T anyway, so it collects the ties as it goes and this
// only has to rank them. `nt` above `kTieCap` cannot happen without that many bit-identical scores;
// it is still handled, by the old walk, rather than silently admitting the wrong rows.
constexpr uint32_t kTieCap = 1024;
constexpr uint32_t kTopkWaves = kTopkThr / 32u;

// ---- the mask, compacted into the ascending LIST of rows it admits ------------------------------
//
// The mask says something about every candidate row; the list says only which ones survived, and
// there are at most `index_topk` of those out of a context's worth. Attention is what cares: it
// walked every candidate row at deep context to give almost all of them a weight of zero, which is
// megabytes of key bytes read to use a few hundred KB. See attn_flash.h.
//
// Ascending, and by construction rather than by sorting: threads take the row in index order, a
// ballot ranks the admissions inside each wave, and the wave counts prefix across the block. That
// makes the list a deterministic function of the mask, which an atomic append would not.
//
// One block a token, which is what both callers already are — so this costs no extra launch.
__device__ __forceinline__ void compact_row(const uint8_t* __restrict__ row, uint32_t n,
                                            uint32_t tid, uint32_t* __restrict__ adm, uint32_t cap,
                                            uint32_t* s_wc, uint32_t* s_base) {
  const uint32_t lane = tid & 31u, wave = tid >> 5;
  if (tid == 0) *s_base = 0;
  __syncthreads();
  for (uint32_t i0 = 0; i0 < n; i0 += kTopkThr) {
    const uint32_t i = i0 + tid;
    const bool a = (i < n) && row[i] != 0;
    const uint64_t bm = __ballot(a);
    const uint32_t wr = (uint32_t)__popcll(bm & ((1ull << lane) - 1ull));
    if (lane == 0) s_wc[wave] = (uint32_t)__popcll(bm);
    __syncthreads();
    uint32_t off = 0;
    for (uint32_t w = 0; w < wave; ++w) off += s_wc[w];
    const uint32_t p = *s_base + off + wr;
    if (a && p < cap) adm[p] = i;
    __syncthreads();
    if (tid == 0) {
      uint32_t s = 0;
      for (uint32_t w = 0; w < kTopkWaves; ++w) s += s_wc[w];
      *s_base += s;
    }
    __syncthreads();
  }
}

__device__ __forceinline__ void tie_admit(const uint32_t* tie, uint32_t nt, uint32_t need,
                                          uint32_t tid, const float* sc, uint8_t* row,
                                          uint32_t n_keys, uint32_t T) {
  if (nt <= kTieCap) {
    // Rank by value, not by position in the list — the list's order is whatever the atomics gave
    // it, and the admission rule is by index.
    for (uint32_t a = tid; a < nt; a += kTopkThr) {
      const uint32_t v = tie[a];
      uint32_t rank = 0;
      for (uint32_t b = 0; b < nt; ++b) rank += (tie[b] < v) ? 1u : 0u;
      if (rank < need) row[v] = 1;
    }
    return;
  }
  if (tid < 32) {
    uint32_t quota = need;
    for (uint32_t i0 = 0; i0 < n_keys && quota; i0 += 32u) {
      const uint32_t i = i0 + tid;
      const bool eq = i < n_keys && mono(sc[i]) == T;
      const uint64_t bm = __ballot(eq);
      const uint32_t before = (uint32_t)__popcll(bm & ((1ull << tid) - 1ull));
      if (eq && before < quota) row[i] = 1;
      quota -= min(quota, (uint32_t)__popcll(bm));
    }
  }
}

__global__ __launch_bounds__(kTopkThr) void idx_topk_mask_kernel(
    const float* __restrict__ scores, uint8_t* __restrict__ mask,
    const uint32_t* __restrict__ nkeys, uint32_t stride, uint32_t mstride, uint32_t mlen,
    uint32_t k, uint32_t* __restrict__ adm, uint32_t* __restrict__ admn, uint32_t adm_stride) {
  __shared__ uint32_t s_hist[256];
  __shared__ uint32_t s_tie[kTieCap];
  __shared__ uint32_t s_wc[kTopkWaves];
  __shared__ uint32_t s_prefix, s_need, s_gt, s_b, s_any, s_nt, s_base;
  const uint32_t t = blockIdx.x, tid = threadIdx.x;
  const float* sc = scores + (size_t)t * stride;
  uint8_t* row = mask + (size_t)t * mstride;
  // The compressed cache GROWS inside a chunk — the compressor emits a row every `ratio` positions
  // — so each token sees its own prefix of it. Scoring past that is scoring against rows derived
  // from tokens that come later. Clamped to `stride` because that is how many scores exist: the two
  // compressors have separate capacities and the attention's count can be the larger, in which case
  // the rows past it have no score and must come back unadmitted rather than as whatever was there.
  const uint32_t nk0 = nkeys ? nkeys[t] : stride;
  const uint32_t n_keys = nk0 < stride ? nk0 : stride;

  // ---- clear ONLY the tail no later pass writes.
  //
  // The row pitch is sized from the cache's CAPACITY, so clearing all of it costs a full pitch a
  // call at every context however short. [0, n_keys) is written unconditionally by the marking pass
  // below, so all that is left here is the gap between the indexer's key count and the attention's,
  // which is normally empty — and it is cleared 16 bytes a lane rather than one.
  const uint32_t clr = mlen > n_keys ? mlen : n_keys;
  {
    uint32_t b = n_keys;
    const uint32_t head = (uint32_t)((16u - ((uintptr_t)(row + b) & 15u)) & 15u);
    const uint32_t hb = b + (head < clr - b ? head : clr - b);
    for (uint32_t i = b + tid; i < hb; i += kTopkThr) row[i] = 0;
    const uint32_t v4 = (clr - hb) >> 4;
    uint4* p4 = (uint4*)(row + hb);
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    for (uint32_t i = tid; i < v4; i += kTopkThr) p4[i] = z;
    for (uint32_t i = hb + (v4 << 4) + tid; i < clr; i += kTopkThr) row[i] = 0;
  }
  // Fewer keys than the budget: admit all of them, which is what the host path's "the indexer did
  // not run for this token" branch does.
  if (n_keys <= k) {
    for (uint32_t i = tid; i < n_keys; i += kTopkThr) row[i] = 1;
    // Every candidate survives, so the list is the identity and attention sees exactly the rows it
    // saw before. It is still WRITTEN, because the caller cannot tell the two cases apart.
    if (adm) {
      uint32_t* a = adm + (size_t)t * adm_stride;
      for (uint32_t i = tid; i < n_keys && i < adm_stride; i += kTopkThr) a[i] = i;
      if (tid == 0) admn[t] = n_keys < adm_stride ? n_keys : adm_stride;
    }
    return;
  }
  if (tid == 0) { s_prefix = 0; s_need = k; s_gt = 0; }
  __syncthreads();

  // Narrow the key eight bits at a time. `s_prefix` holds the bits already fixed, `s_need` how many
  // of the top k are still to be found inside it.
  for (int sh = 24; sh >= 0; sh -= 8) {
    if (tid < 256u) s_hist[tid] = 0;
    __syncthreads();
    const uint32_t pre = s_prefix, msk = (sh == 24) ? 0u : (0xFFFFFFFFu << (sh + 8));
    for (uint32_t i = tid; i < n_keys; i += kTopkThr) {
      const uint32_t key = mono(sc[i]);
      hist_add(s_hist, (key >> sh) & 0xFFu, (key & msk) == pre);
    }
    __syncthreads();
    uint32_t b = 0, acc = 0;
    hist_scan(s_hist, &s_b, &s_any, s_need, tid, &b, &acc);
    if (tid == 0) {
      s_gt += acc;                               // strictly above the bucket we descended into
      s_need -= acc;
      s_prefix = pre | (b << sh);
    }
    __syncthreads();
  }

  const uint32_t T = s_prefix, need = s_need;
  if (tid == 0) s_nt = 0;
  __syncthreads();
  // 0 as well as 1, which is what lets the clear above skip this range entirely. The pass also
  // COLLECTS the ties, because it is already reading every score and testing it against T.
  for (uint32_t i = tid; i < n_keys; i += kTopkThr) {
    const uint32_t key = mono(sc[i]);
    row[i] = key > T ? 1u : 0u;
    if (key == T) {
      const uint32_t p = atomicAdd(&s_nt, 1u);
      if (p < kTieCap) s_tie[p] = i;
    }
  }
  __syncthreads();
  tie_admit(s_tie, s_nt, need, tid, sc, row, n_keys, T);
  if (adm) {
    __syncthreads();                                 // the mask is only final after the ties
    // `adm` and `admn[0]`, WITHOUT the token offset, is a race: this kernel is one block a token, so
    // in prefill every block would write the same slot. It costs nothing and produces fluent text;
    // only three identical runs hashing three different ways catches it.
    compact_row(row, n_keys, tid, adm + (size_t)t * adm_stride, adm_stride, s_wc, &s_base);
    if (tid == 0) admn[t] = s_base < adm_stride ? s_base : adm_stride;
  }
}

// ---- the same select, spread over the card ------------------------------------------------------
//
// Everything above is one workgroup, and at decode the grid is ONE TOKEN, so it is one workgroup on
// the whole card no matter how wide the block is. Widening the block helps and then stops helping:
// the remaining factor is not inside it at all, it is the rest of the card. Prefill does not have
// this problem — it presents a block per token — so the single-block kernel stays, and this path is
// taken only when the token count is too small to fill the machine by itself.
//
// The select is unchanged: the SAME four 8-bit radix passes over the SAME monotone map, giving the
// same threshold and the same tie rule. What changes is who runs them. Each pass is a launch whose
// blocks histogram disjoint slices into one global 256-bin array, and the LAST block to arrive —
// the standard threadfence-and-count, no grid barrier and no cooperative launch, which costs more
// than the launches it would save — does the 256-entry scan and leaves the prefix
// for the next launch. It also zeroes the histogram behind itself, so the buffer needs clearing
// once at allocation and never again.
constexpr uint32_t kTopkHist = 256;

// Blocks the spread select launches: one a workgroup processor, cached per device.
uint32_t topk_blocks() {
  static uint32_t cached[kMaxGpuDevices] = {};
  int d = 0;
  if (hipGetDevice(&d) != hipSuccess || d < 0 || d >= kMaxGpuDevices) return 32u;
  if (!cached[d]) cached[d] = device_cus(d) / 2u;
  return cached[d] ? cached[d] : 32u;
}

// state[t]: prefix, need, gt, arrivals.
__global__ __launch_bounds__(kTopkThr) void idx_topk_hist_kernel(
    const float* __restrict__ scores, const uint32_t* __restrict__ nkeys, uint32_t stride,
    uint32_t k, int sh, uint32_t* __restrict__ hist, uint32_t* __restrict__ state) {
  __shared__ uint32_t s_h[kTopkHist];
  __shared__ uint32_t s_last, s_b, s_any;
  const uint32_t t = blockIdx.y, tid = threadIdx.x;
  const float* sc = scores + (size_t)t * stride;
  const uint32_t nk0 = nkeys ? nkeys[t] : stride;
  const uint32_t n_keys = nk0 < stride ? nk0 : stride;
  uint32_t* g_h = hist + (size_t)t * kTopkHist;
  uint32_t* g_s = state + (size_t)t * 4u;
  // Fewer keys than the budget admits all of them and never consults the threshold, so no pass runs
  // and nothing here may touch the counter — it has to come out of this call as it went in.
  if (n_keys <= k) return;

  if (tid < kTopkHist) s_h[tid] = 0;
  __syncthreads();
  const uint32_t pre = (sh == 24) ? 0u : g_s[0];
  const uint32_t msk = (sh == 24) ? 0u : (0xFFFFFFFFu << (sh + 8));
  const uint32_t per = (n_keys + gridDim.x - 1u) / gridDim.x;
  const uint32_t lo = blockIdx.x * per;
  const uint32_t hi = (lo + per < n_keys) ? lo + per : n_keys;
  for (uint32_t i = lo + tid; i < hi; i += kTopkThr) {
    const uint32_t key = mono(sc[i]);
    hist_add(s_h, (key >> sh) & 0xFFu, (key & msk) == pre);
  }
  __syncthreads();
  if (tid < kTopkHist && s_h[tid]) atomicAdd(&g_h[tid], s_h[tid]);
  // Every thread fences its own stores, then the barrier orders all of them before the arrival
  // count below — which is what makes "I am the last" mean "every block's counts are visible".
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = (atomicAdd(&g_s[3], 1u) == gridDim.x - 1u) ? 1u : 0u;
  __syncthreads();
  if (!s_last) return;

  // The acquire half of the same fence: everyone else's counts are in L2, and this block's own view
  // of those lines has to be thrown away before it reads them back as plain loads.
  __threadfence();
  const uint32_t need = (sh == 24) ? k : g_s[1];
  if (tid < kTopkHist) { s_h[tid] = g_h[tid]; g_h[tid] = 0; }   // zeroed for the next pass
  __syncthreads();
  uint32_t b = 0, acc = 0;
  hist_scan(s_h, &s_b, &s_any, need, tid, &b, &acc);
  if (tid == 0) {
    g_s[0] = pre | (b << sh);
    g_s[1] = need - acc;
    g_s[2] = ((sh == 24) ? 0u : g_s[2]) + acc;        // strictly above the bucket descended into
    g_s[3] = 0;
  }
}

// The marking pass, which also COLLECTS the ties — it is testing every score against T already.
// Admitting them is a separate launch because the rule is by global index and the blocks of one
// launch have no order between them.
__global__ __launch_bounds__(kTopkThr) void idx_topk_mark_kernel(
    const float* __restrict__ scores, uint8_t* __restrict__ mask,
    const uint32_t* __restrict__ nkeys, uint32_t stride, uint32_t mstride, uint32_t mlen,
    uint32_t k, const uint32_t* __restrict__ state, uint32_t* __restrict__ tie) {
  const uint32_t t = blockIdx.y, tid = threadIdx.x;
  const float* sc = scores + (size_t)t * stride;
  uint8_t* row = mask + (size_t)t * mstride;
  const uint32_t nk0 = nkeys ? nkeys[t] : stride;
  const uint32_t n_keys = nk0 < stride ? nk0 : stride;
  const uint32_t clr = mlen > n_keys ? mlen : n_keys;
  const uint32_t str = kTopkThr * gridDim.x;
  const uint32_t g0 = blockIdx.x * kTopkThr + tid;

  for (uint32_t i = n_keys + g0; i < clr; i += str) row[i] = 0;
  if (n_keys <= k) {
    for (uint32_t i = g0; i < n_keys; i += str) row[i] = 1;
    return;
  }
  const uint32_t T = state[(size_t)t * 4u];
  uint32_t* g_t = tie + (size_t)t * (kTieCap + 1u);
  for (uint32_t i = g0; i < n_keys; i += str) {
    const uint32_t key = mono(sc[i]);
    row[i] = key > T ? 1u : 0u;
    if (key == T) {
      const uint32_t p = atomicAdd(&g_t[0], 1u);
      if (p < kTieCap) g_t[1u + p] = i;
    }
  }
}

// The ties, ranked by index. One block: the list is normally one entry long, because two indexer
// scores are equal only when their float bits are.
__global__ __launch_bounds__(kTopkThr) void idx_topk_tie_kernel(
    const float* __restrict__ scores, uint8_t* __restrict__ mask,
    const uint32_t* __restrict__ nkeys, uint32_t stride, uint32_t mstride, uint32_t k,
    const uint32_t* __restrict__ state, uint32_t* __restrict__ tie, uint32_t* __restrict__ adm,
    uint32_t* __restrict__ admn, uint32_t adm_stride) {
  __shared__ uint32_t s_tie[kTieCap];
  __shared__ uint32_t s_wc[kTopkWaves];
  __shared__ uint32_t s_base;
  const uint32_t t = blockIdx.x, tid = threadIdx.x;
  const float* sc = scores + (size_t)t * stride;
  uint8_t* row = mask + (size_t)t * mstride;
  const uint32_t nk0 = nkeys ? nkeys[t] : stride;
  const uint32_t n_keys = nk0 < stride ? nk0 : stride;
  if (n_keys <= k) {
    if (adm) {
      uint32_t* a = adm + (size_t)t * adm_stride;
      for (uint32_t i = tid; i < n_keys && i < adm_stride; i += kTopkThr) a[i] = i;
      if (tid == 0) admn[t] = n_keys < adm_stride ? n_keys : adm_stride;
    }
    return;
  }
  uint32_t* g_t = tie + (size_t)t * (kTieCap + 1u);
  const uint32_t nt = g_t[0];
  const uint32_t nc = nt < kTieCap ? nt : kTieCap;
  for (uint32_t i = tid; i < nc; i += kTopkThr) s_tie[i] = g_t[1u + i];
  __syncthreads();
  tie_admit(s_tie, nt, state[(size_t)t * 4u + 1u], tid, sc, row, n_keys,
            state[(size_t)t * 4u]);
  __syncthreads();
  if (tid == 0) g_t[0] = 0;                          // back to empty for the next call
  if (adm) {
    compact_row(row, n_keys, tid, adm + (size_t)t * adm_stride, adm_stride, s_wc, &s_base);
    if (tid == 0) admn[t] = s_base < adm_stride ? s_base : adm_stride;
  }
}


// ---- RoPE tail + the indexer's QAT, per (token, head) --------------------------------------------
//
// On the host these two were most of a long-prompt prefill, and it was never the SCORING — that is
// the O(n_comp) part, and the WMMA kernel above replaced it. This is the O(1)-per-token part, and it
// dominated anyway because `rope_tail` recomputes theta INSIDE the head loop while theta depends
// only on (pos, i), so every head paid for the same transcendentals again. On device each thread
// computes its own theta once and the point is moot.
//
// One block a (token, head) row of 128: the Hadamard is a 7-stage butterfly through LDS, and the
// E2M1 groups are 32 wide, which is exactly a wave, so the group amax is a wave reduction.
__device__ __forceinline__ float yarn_ramp_dev(float low, float high, int i) {
  const float y = ((float)i / 2.0f - low) / fmaxf(0.001f, high - low);
  return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

template <bool HADAMARD>
__global__ __launch_bounds__(128) void idx_rope_qat_kernel(float* __restrict__ q, uint32_t n_head,
                                                           uint32_t n_rot, uint32_t pos0,
                                                           float freq_scale, float theta_scale,
                                                           float ext_factor, float lo, float hi) {
  __shared__ float r[kIdxDim];
  const uint32_t row = blockIdx.x, i = threadIdx.x;
  const uint32_t t = row / n_head;
  float* g = q + (size_t)row * kIdxDim;
  float v = g[i];

  const uint32_t n_nope = kIdxDim - n_rot;
  if (i >= n_nope) {
    const uint32_t j = i - n_nope;
    // theta_extrap after k doublings is pos * theta_scale^k, and k is (j/2) for this lane. The
    // host walks it multiplicatively per head; here it is one powf and no loop.
    const float theta_extrap = (float)(pos0 + t) * powf(theta_scale, (float)(j >> 1));
    float theta = freq_scale * theta_extrap;
    if (ext_factor != 0.0f) {
      const float mix = yarn_ramp_dev(lo, hi, (int)(j & ~1u)) * ext_factor;
      theta = theta * (1.0f - mix) + theta_extrap * mix;
    }
    const float c = __cosf(theta), sn = __sinf(theta);
    // The pair (j, j+1) is two adjacent lanes of the same wave, so the partner is a shuffle.
    const float partner = __shfl_xor(v, 1);
    v = (j & 1u) ? (partner * sn + v * c) : (v * c - partner * sn);
  }

  // HADAMARD is V4's. V4.1's `fp4_act_quant` is a plain per-32 round trip with no rotation at all,
  // and the queries have to be quantised in the SAME basis as the keys or the scores are a product
  // of two different approximations. Uniform across the block, as the __syncthreads() below needs.
  if (HADAMARD) {
    r[i] = v;
    __syncthreads();
    // 128-wide Hadamard: 64 butterflies a stage, one per thread below 64.
    for (uint32_t stride = 1; stride < kIdxDim; stride <<= 1) {
      float a = 0.0f, b = 0.0f;
      uint32_t ia = 0, ib = 0;
      if (i < kIdxDim / 2u) {
        const uint32_t base = (i / stride) * 2u * stride, off = i % stride;
        ia = base + off; ib = base + stride + off;
        a = r[ia]; b = r[ib];
      }
      __syncthreads();
      if (i < kIdxDim / 2u) { r[ia] = a + b; r[ib] = a - b; }
      __syncthreads();
    }
    v = r[i] * 0.08838834764831845f;            // 1/sqrt(128)
  }

  float amax = fabsf(v);
#pragma unroll
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o));
  amax = fmaxf(amax, 7.052966104933725e-38f);
  const float sc = exp2f(ceilf(log2f(amax / 6.0f)));
  g[i] = e2m1_round_d(fminf(6.0f, fmaxf(-6.0f, v / sc))) * sc;
}

}  // namespace

void indexer_scores_hip(const uint8_t* q8, const float* qs, const uint8_t* k8, const float* ks,
                        const float* hw, float* out, uint32_t nb, uint32_t n_keys,
                        uint32_t n_head, void* stream) {
  if (!nb || !n_keys || !n_head) return;
  if (nb == 1u && n_head <= kIdxDecHeads && (n_head % 16u) == 0u) {
    const uint32_t kpb = kWaves * 16u;                // 64 keys a block
    hipLaunchKernelGGL(indexer_dec_kernel, dim3((n_keys + kpb - 1) / kpb), dim3(kThreads), 0,
                       (hipStream_t)stream, q8, qs, k8, ks, hw, out, n_keys, n_head);
    AFF_HIP_CHECK(hipGetLastError());
    return;
  }
  const dim3 g((nb + kIdxTokTile - 1) / kIdxTokTile, (n_keys + kIdxKeys - 1) / kIdxKeys);
  hipLaunchKernelGGL(indexer_kernel, g, dim3(kThreads), 0, (hipStream_t)stream, q8, qs, k8, ks, hw,
                     out, nb, n_keys, n_head);
  AFF_HIP_CHECK(hipGetLastError());
}

void idx_rope_qat_hip(float* q, uint32_t n, uint32_t n_head, uint32_t n_rot, uint32_t pos0,
                      float freq_scale, float theta_scale, float ext_factor, float lo, float hi,
                      void* stream, bool hadamard) {
  if (!n || !n_head) return;
  if (hadamard)
    hipLaunchKernelGGL(idx_rope_qat_kernel<true>, dim3(n * n_head), dim3(kIdxDim), 0,
                       (hipStream_t)stream, q, n_head, n_rot, pos0, freq_scale, theta_scale,
                       ext_factor, lo, hi);
  else
    hipLaunchKernelGGL(idx_rope_qat_kernel<false>, dim3(n * n_head), dim3(kIdxDim), 0,
                       (hipStream_t)stream, q, n_head, n_rot, pos0, freq_scale, theta_scale,
                       ext_factor, lo, hi);
  AFF_HIP_CHECK(hipGetLastError());
}

void idx_quant_rows_hip(const float* src, uint8_t* dst, float* scale, uint32_t n, uint32_t dim,
                        void* stream) {
  if (!n || !dim) return;
  hipLaunchKernelGGL(idx_quant_kernel, dim3(n), dim3(256), 0, (hipStream_t)stream, src, dst, scale,
                     dim);
  AFF_HIP_CHECK(hipGetLastError());
}

uint64_t idx_topk_scratch_bytes(uint32_t tok, uint64_t* state_bytes, uint64_t* tie_bytes) {
  if (state_bytes) *state_bytes = (uint64_t)tok * 4u * sizeof(uint32_t);
  if (tie_bytes) *tie_bytes = (uint64_t)tok * (kTieCap + 1u) * sizeof(uint32_t);
  return (uint64_t)tok * kTopkHist * sizeof(uint32_t);
}

void idx_topk_mask_hip(const float* scores, uint8_t* mask, const uint32_t* nkeys, uint32_t nb,
                       uint32_t stride, uint32_t mstride, uint32_t mlen, uint32_t k,
                       void* stream, const TopkScratch* sp, uint32_t* adm, uint32_t* admn,
                       uint32_t adm_stride) {
  if (!nb || !stride) return;
  if (adm && (!admn || adm_stride < k)) { adm = nullptr; admn = nullptr; }
  hipStream_t st = (hipStream_t)stream;
  // One block a workgroup processor, which is what `multiProcessorCount` reports on RDNA and what
  // `device_cus` doubles. Queried once a device: this is on the dispatch path and
  // hipGetDeviceProperties is a driver call.
  const uint32_t blocks = topk_blocks();
  // Spread it only when the token count cannot fill the card by itself. Six launches for one select
  // is a real cost — an added kernel is never free here — so it has to buy back more than that,
  // which needs both a small grid and enough keys for the passes to dominate.
  if (sp && sp->hist && sp->tie && nb <= sp->tok_cap && nb < blocks && stride >= 4096u) {
    const dim3 g(blocks, nb), b(kTopkThr);
    for (int sh = 24; sh >= 0; sh -= 8)
      hipLaunchKernelGGL(idx_topk_hist_kernel, g, b, 0, st, scores, nkeys, stride, k, sh, sp->hist,
                         sp->state);
      AFF_HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(idx_topk_mark_kernel, g, b, 0, st, scores, mask, nkeys, stride, mstride,
                       mlen, k, sp->state, sp->tie);
    hipLaunchKernelGGL(idx_topk_tie_kernel, dim3(nb), dim3(kTopkThr), 0, st, scores, mask, nkeys,
                       stride, mstride, k, sp->state, sp->tie, adm, admn, adm_stride);
    AFF_HIP_CHECK(hipGetLastError());
    return;
  }
  hipLaunchKernelGGL(idx_topk_mask_kernel, dim3(nb), dim3(kTopkThr), 0, st, scores,
                     mask, nkeys, stride, mstride, mlen, k, adm, admn, adm_stride);
  AFF_HIP_CHECK(hipGetLastError());
}

void idx_encode_rows(const float* src, uint8_t* dst, float* scale, uint32_t n, uint32_t dim) {
  for (uint32_t r = 0; r < n; ++r) {
    const float* s = src + (size_t)r * dim;
    float amax = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) amax = std::fmax(amax, std::fabs(s[i]));
    const float sc = (amax > 0.0f) ? amax / 448.0f : 1.0f;
    scale[r] = sc;
    const float inv = 1.0f / sc;
    for (uint32_t i = 0; i < dim; ++i) {
      float a = std::fmin(448.0f, std::fmax(-448.0f, s[i] * inv));
      uint32_t u;
      __builtin_memcpy(&u, &a, 4);
      const uint32_t sign = (u >> 24) & 0x80u;
      float m = std::fabs(a);
      uint8_t b;
      if (!(m > 0.0f)) {
        b = (uint8_t)sign;
      } else if (m >= 0x1p-6f) {
        __builtin_memcpy(&u, &m, 4);
        int e = (int)((u >> 23) & 0xFFu) - 127;
        uint32_t mm = (u & 0x7FFFFFu) >> 20;
        const uint32_t rem = u & 0xFFFFFu, half = 1u << 19;
        if (rem > half || (rem == half && (mm & 1u))) { if (++mm == 8u) { mm = 0; ++e; } }
        if (e > 8) { e = 8; mm = 7; }
        b = (uint8_t)(sign | ((uint32_t)(e + 7) << 3) | mm);
      } else {
        uint32_t mm = (uint32_t)std::lrintf(m * 512.0f);
        if (mm > 7u) mm = 7u;
        b = (uint8_t)(sign | mm);
      }
      dst[(size_t)r * dim + i] = b;
    }
  }
}

}  // namespace aff

// amdnas-fixes launch-check: every hipLaunchKernelGGL in this file is followed by a launch-error check (2026-09-22).
