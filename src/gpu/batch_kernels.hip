#include "gpu/batch_kernels.h"
#include "gpu/device_math.h"
#include "gpu/hip_common.h"
#include "gpu/wmma.h"
#include "format/aff_format.h"

#include "ui/log.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace aff {
namespace {

// ---- the WMMA GEMM ------------------------------------------------------------------------------
//
// Every batched matmul goes through this. A scalar f32 loop over the same shape is limited by the
// f32 VALU rate; WMMA moves that ceiling by most of an order of magnitude before any tuning.
//
// Geometry, and where each number comes from:
//
//   kWM/kWN 128    128x128 output tile per block, so a typical chunk reads the weight matrix only a
//                  couple of times. This GEMM sits near the roofline crossover, and a smaller N tile
//                  would re-read the weights more often and push it bandwidth-bound.
//   kWWaves 8      256 threads and 8 accumulator tiles a wave, which §e.5 puts right at the VGPR
//                  cliff. It only fits because RDNA4 halved the A/B fragments.
//   kWK 64         An LDS row is then 64 halves = 8 granules of 16 B, exactly the shape the §e.1 XOR
//                  swizzle makes conflict-free. kWK=32 would leave the fragment read 8-way
//                  conflicted.
//
// A and B are stored in LDS the SAME way — [row][k], row-major, XOR-swizzled — because B's fragment
// wants B[8h+j][c], which for a [n][k] tile is lds[c][8h+j]: the identical addressing to A. That is
// why there is one `load_frag` and no transpose anywhere.
constexpr uint32_t kWM = 128, kWN = 128, kWK = 64;
constexpr uint32_t kWWaves = 8;
constexpr uint32_t kWThreads = kWWaves * kWave;
constexpr uint32_t kWMTiles = 2, kWNTiles = 4;   // 16x16 tiles a wave owns; 2*4 = 8 accumulators
// The N extent of the narrow instantiation, and the batch width at or below which it is used. One
// wmma tile: a speculative block is 6 live tokens in a 16-wide arena, so this is the whole arena.
constexpr uint32_t kNarrowN = 16;
constexpr uint32_t kWLdsRow = kWK;               // halves per LDS row

// What this kernel is NOT limited by, each tried and reverted rather than argued: the LDS row width
// against the cacheline, occupancy, the launch count, and split-K.
//
// What HAS moved it is removing the LDS round trip: `Wp -> registers -> LDS -> barrier -> LDS ->
// fragment` is a dependency chain the matvec does not have, which is why the matvec reads the same
// weights faster. The preshuffled path skips it (see load_frag8_ps), and hoisting its fragment loads
// a whole K tile ahead is worth a large fraction again.

// The LDS address carries a 16-byte granule XOR swizzle. §e.1: a fragment read is 16 lanes x 16 B at
// rows m..m+15 of one granule column; without the swizzle they hit 4 banks and conflict 16 ways,
// with it they cover all 32 banks exactly once and hit the 4-cycle floor.
__device__ __forceinline__ wmma::s8v load_frag(const uint16_t* __restrict__ lds, uint32_t row0,
                                               uint32_t k0, uint32_t lane) {
  const uint32_t row = row0 + wmma::frag_c(lane);
  const uint32_t k = k0 + wmma::frag_k0(lane);            // 8-aligned, so (k & 7) == 0
  const uint16_t* p = lds + row * kWLdsRow + ((((k >> 3) ^ (row & 7u))) << 3);
  wmma::s8v f;
#pragma unroll
  for (int j = 0; j < 8; ++j) f[j] = (short)p[j];
  return f;
}

// E4M3 -> bf16 is EXACT: E4M3 carries 3 mantissa bits and the UE8M0 tile scale is a power of two,
// so the product always fits bf16's 7 and truncating the top half of the f32 IS round-to-nearest.
// The only place this kernel loses precision is the activation, f32 -> bf16.
//
// The decode is still ~10 VALU ops an element, and that is what bound the kernel: neither
// compute- nor bandwidth-bound, but serial in front of the matrix work, because WMMA does not
// co-issue with VALU (§e.3).
//
// A 256-entry LDS table removes it, and it wins even though the LDS costs a block per WGP. The
// branchless register alternative — fold the UE8M0 exponent into bf16's exponent field and
// renormalise E4M3 subnormals with a clz — is exact and needs no LDS, and it is clearly slower. The
// dequant arithmetic really is the binding cost, so buying occupancy with it is a bad trade.
//
// The scale is ONE value for the whole 128x64 A tile — m0 is 128-aligned so the scale row cannot
// change, and a 64-wide K window starting at a multiple of 64 never crosses a 128-column scale
// boundary — so the table can have it baked in and staging becomes a single lookup. Building it is
// one entry per thread, and only when the scale index changes: with kWK=64, every second iteration.
__device__ __forceinline__ uint16_t e4m3_bf16_bits(uint32_t b, float ws) {
  return (uint16_t)(__float_as_uint(e4m3_decode(b) * ws) >> 16);
}


// ---- f32 dim-major -> E4M3 token-major, with a per-(token, 128-column) scale ---------------------
//
// The gate on every fp8 GEMM in the engine. A GEMM cannot compute its own output scale without a
// reduction across every M block, so it does not try; whatever produces an activation quantises it,
// and this is that step until it is fused into the producing kernel's epilogue.
//
// Two things it has to do at once, and they pull in opposite directions:
//
//   * the READ is dim-major ([dim][nb]) because that is what every f32 kernel in the engine writes,
//     so coalescing wants consecutive threads on consecutive TOKENS;
//   * the WRITE is token-major ([nb][dim]) because that is what makes the GEMM's staging a pure byte
//     copy, so coalescing wants consecutive threads on consecutive COLUMNS.
//
// So it is a transpose, and it goes through LDS. Read phase: 32 tokens x 128 columns staged with
// consecutive threads on consecutive tokens. Write phase: eight threads cover one token's 128 bytes
// contiguously. The `[128][33]` padding is one float per row, which is what stops the staged tile
// from putting all 32 tokens of a column in the same LDS bank.
//
// The scale is per 128 columns rather than per token because that is the block the WEIGHT scale
// uses, so the two are constant together over a K block and fold into the accumulator once.
constexpr uint32_t kQTok = 32, kQThreads = 256;

// ---- and what it reads is an accessor, so a producer folds in with one body --------------------
//
// `Src` answers "the f32 value at dim-major offset o" and nothing else. The tiling, the amax, the
// scale, the E4M3 conversion and the transpose are the kernel's and do not know which source they
// have. That is the whole extension mechanism: an elementwise pass whose output exists only to be
// quantised becomes a `Src` and stops writing to memory at all.
//
// Which activations can actually take this shape is a property of the PRODUCER, not of this kernel.
// An elementwise op over two dim-major arenas (`QSrcSwiGLU`) is a pure fold — same expression, same
// order, bit-identical, one launch and one f32 arena gone. A producer that needs a reduction across
// a token (`rms_norm`, `hc_pre`) is not: its threads are striped over `dim`, so no thread owns a
// whole 128-column scale block and the token-major store would be one byte per thread. Those are
// rewrites into this kernel's tile shape rather than folds.
struct QSrcF32 {
  const float* __restrict__ x;
  __device__ __forceinline__ float operator()(size_t o) const { return x[o]; }
};
// The dense SwiGLU, written out here rather than shared with `batch_swiglu_kernel` for the same
// reason the expert one is: the two must not drift by someone editing the other. `expf` and this
// exact order, so the fold is bit-identical to swiglu-then-quantise.
struct QSrcSwiGLU {
  const float* __restrict__ g;
  const float* __restrict__ u;
  float limit;
  __device__ __forceinline__ float operator()(size_t o) const {
    float gv = g[o], uv = u[o];
    if (limit > 0.0f) { gv = fminf(gv, limit); uv = fminf(limit, fmaxf(-limit, uv)); }
    return (gv / (1.0f + expf(-gv))) * uv;
  }
};

template <class Src>
__global__ __launch_bounds__(kQThreads) void batch_quant_fp8_kernel(
    Src src, uint8_t* __restrict__ out,
    float* __restrict__ scales, uint32_t dim, uint32_t nb, uint32_t x_off) {
  __shared__ float tile[128][kQTok + 1];
  __shared__ float part[8][kQTok];
  __shared__ float smax[kQTok];

  const uint32_t tid = threadIdx.x, tt = tid & 31u, kk = tid >> 5;
  const uint32_t t0 = blockIdx.x * kQTok;
  const uint32_t tok = t0 + tt;
  const uint32_t nkb = dim >> 7;

  // gridDim.y IS the 128-dim block index, not a loop. As a loop the grid is only nb/kQTok blocks,
  // which does not fill the card. Every (token, kb) pair is independent — the scale is per pair, and
  // `tile`/`part`/`smax` are re-derived each iteration — so the iteration space is the grid. Same
  // arithmetic in the same order per pair, so the output is bit-identical; only the schedule moved.
  for (uint32_t kb = blockIdx.y; kb < nkb; kb += gridDim.y) {
    __syncthreads();
    float m = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) {
      const uint32_t k = kk + i * 8;
      const size_t o = (size_t)(x_off + kb * 128 + k) * nb + tok;
      const float v = tok >= nb ? 0.0f : src(o);
      tile[k][tt] = v;
      m = fmaxf(m, fabsf(v));
    }
    part[kk][tt] = m;
    __syncthreads();
    if (kk == 0) {
      float mm = part[0][tt];
#pragma unroll
      for (uint32_t i = 1; i < 8; ++i) mm = fmaxf(mm, part[i][tt]);
      // 448 is E4M3's largest finite value. A zero block would divide by zero, so it gets a scale of
      // 1 and quantises to zeros.
      smax[tt] = mm > 0.0f ? mm * (1.0f / 448.0f) : 1.0f;
      if (tok < nb) scales[(size_t)tok * nkb + kb] = smax[tt];
    }
    __syncthreads();

    // Write phase: thread owns 16 contiguous columns of one token, so eight threads lay down that
    // token's 128 bytes back to back.
    const uint32_t wt = tid >> 3, wk = (tid & 7u) * 16;
    if (t0 + wt < nb) {
      const float inv = 1.0f / smax[wt];
      // Clamp, do not trust the round trip. smax is max/448, so max*(1/smax) is 448 in exact
      // arithmetic but can land an ulp ABOVE it in float — and v_cvt_pk_fp8_f32 answers an
      // out-of-range input with NaN rather than saturating, so without the clamp every block
      // containing its own maximum produces a NaN.
      uint32_t w[4];
#pragma unroll
      for (uint32_t q = 0; q < 4; ++q) {
        float v[4];
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j)
          v[j] = fminf(fmaxf(tile[wk + q * 4 + j][wt] * inv, -448.0f), 448.0f);
        int pk = 0;
        pk = __builtin_amdgcn_cvt_pk_fp8_f32(v[0], v[1], pk, false);
        pk = __builtin_amdgcn_cvt_pk_fp8_f32(v[2], v[3], pk, true);
        w[q] = (uint32_t)pk;
      }
      *(uint4*)(out + (size_t)(t0 + wt) * dim + kb * 128 + wk) =
          make_uint4(w[0], w[1], w[2], w[3]);
    }
  }
}

// Sum the split-K partials. The z loop is in a FIXED order and there is no atomic anywhere, because
// `long_code_audit` reproduces the hosted API's greedy continuation exactly and a non-deterministic
// summation would make that intermittent rather than wrong.
// It is LATENCY bound, not bandwidth bound, and that is why the loads are batched.
//
// The output decides the grid: `rows * nb / 256` blocks. The narrowest shape here is a couple of
// blocks, and each thread then walks `ksplit` planes whose addresses are `n` floats apart. Written
// as one `acc +=` per iteration the adds are a dependent chain and each load sits behind the
// previous add, so those few blocks issue one small round of requests at a time and wait — which can
// cost several times the GEMM the split was widening.
//
// Batching eight loads before the adds breaks the chain without touching the ORDER of the adds:
// they are still z ascending, one accumulator, no tree and no atomic. Bit-identical by
// construction, which is what the fixed order above requires.
__global__ void splitk_reduce_kernel(const float* __restrict__ ws, float* __restrict__ y,
                                     uint32_t rows, uint32_t nb, uint32_t ksplit) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t n = (size_t)rows * nb;
  if (i >= n) return;
  const float* __restrict__ p = ws + i;
  constexpr uint32_t U = 8;
  float acc = 0.0f;
  uint32_t z = 0;
  for (; z + U <= ksplit; z += U) {
    float v[U];
#pragma unroll
    for (uint32_t u = 0; u < U; ++u) v[u] = p[(size_t)(z + u) * n];
#pragma unroll
    for (uint32_t u = 0; u < U; ++u) acc += v[u];
  }
  for (; z < ksplit; ++z) acc += p[(size_t)z * n];
  y[i] = acc;
}

// Threads a block for the reduce, and it stays 256 for every width. Narrowing it to get more blocks
// looks like the same argument the batched loads won on, but eight loads in flight already supply
// the memory-level parallelism the extra blocks would. Once the dependent chain is gone, this shape
// is not block-starved.

// One workspace a device, grown on demand. Only ever used by shapes too narrow to fill the card,
// so it stays small: the hyper-connection control GEMM is 24 rows and 16 splits is 393 KB.
struct SplitKWs { float* p = nullptr; size_t n = 0; };
SplitKWs g_skws[kMaxGpuDevices];


// Blocks the split is aiming for. Both GEMMs use it: gemm_launch counts blocks, batch_gemm_fp8_hip
// counts WAVES (blocks * mw), because there a wave owns 16 rows and the block count alone hides how
// few of them there are.
//
// The split's WIDTH is not a lever: anything from 64 to 256 behaves alike, and past that it degrades
// and starts moving the emitted text. Setting it to 0 never splits, which is the arm that prices the
// mechanism rather than its tuning — not splitting costs a large share of decode, because K is the
// only dimension a decode-shaped GEMM has enough of to fill the card.
//
// A TUNING DEFAULT, swept on a 64-CU part at roughly two blocks a CU. It is deliberately NOT derived
// from the device's CU count: `ksplit` sets the order the K partials are summed in, so deriving it
// would make the emitted text a function of which card the binary is on. A different part wants this
// re-tuned by hand, inside the flat window above.
constexpr uint32_t kSplitTarget = 128u;

// ...and the cap on the split when the OUTPUT is thin, which is a different question: a shape with
// `rows * nb < 8192` makes a reduce that can be thinner than the GEMM the split was widening. The
// reduce keeps eight loads in flight rather than running as a dependent chain, which is what lets
// the cap sit this high.
//
// What it buys is the hyper-connection control GEMM, whose M and N are both a single tile — every
// block it will ever get has to come from K.
//
// NOT BIT-IDENTICAL, and it cannot be: ksplit repartitions K, so the accumulation chain differs and
// the emitted text can move. That is permitted — the engine's invariant is that speculative decode
// matches greedy decode in the SAME binary, which is unaffected — but it means an arm that changes
// this must be judged on ms a cycle, never on tok/s, because acceptance rate depends on which
// continuation is being decoded. For the same reason, two settings whose output hashes differ are
// decoding different sequences and cannot be compared on either.
constexpr uint32_t kSplitThinCap = 64u;

float* splitk_buf(size_t need) {
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= kMaxGpuDevices) return nullptr;
  SplitKWs& w = g_skws[dev];
  if (w.n < need) {
    if (w.p) (void)hipFree(w.p);
    w.p = nullptr;
    if (hipMalloc(&w.p, need * sizeof(float)) != hipSuccess) { w.n = 0; return nullptr; }
    w.n = need;
  }
  return w.p;
}

// ---- the W8A8 GEMM -------------------------------------------------------------------------------
//
// fp8 x fp8 -> f32. Nothing is converted anywhere in the loop: the weights are already E4M3 in the
// container and the activation is already E4M3 in the pipeline, so both go global -> LDS ->
// fragment as raw bytes. That is the signature of a correct fp8 GEMM on this card — zero 16-bit
// WMMA and zero in-loop dequant v_cvt in the disassembly.
//
// Over the bf16 kernel above: twice the matrix-core peak, no dequant (which was the binding cost
// there), half the LDS and half the fragment registers.
//
// ---- scales ----
//
// Both operands carry a scale per 128 along K, so the two are constant together over a 128-column
// block and can be applied ONCE per block to the partial sum rather than per element:
//
//   weights     UE8M0, one per 128x128 tile. m0 is 128-aligned so the row index is fixed per block.
//   activation  one per (token, 128-column block), which is the same block-fp8 shape.
//
// The activation is TOKEN-major here ([nb][cols]), not dim-major like the f32 buffers. That is what
// makes staging a pure byte copy: a thread wants 32 consecutive k for one token, which is 32
// consecutive bytes. Dim-major would make it 32 separate strided reads — the exact failure that
// cost the bf16 kernel a factor of three above. It is also vLLM's layout.
//
// A lane owns one output COLUMN, i.e. one token, so the activation scale is one scalar per lane and
// the flush is a single fma per accumulator element. That is why there are two accumulator sets:
// `raw` collects 128 columns of K unscaled, then folds into `out`. It costs 64 extra VGPRs and it
// is what keeps the inner loop free of any per-element arithmetic at all.
// kFN 128 and 8 waves. A block re-reads its whole row band of the weight once per N tile, so weight
// traffic is `rows*cols * ceil(nb/kFN)`, and the kernel reports close to the achievable streaming
// rate — which reads as a memory wall and is not one. A 256-wide N tile halves that traffic and is
// SLOWER on all but the tallest shape, because halving the N grid halves the block count, and those
// blocks are what generate the parallel misses that keep the memory system busy.
constexpr uint32_t kFM = 128, kFN = 128, kFK = 64;    // kFK is BYTES per LDS row
constexpr uint32_t kFKNarrow = 256;                   // one gfx1201 cacheline; see AFF_GEMM_F8
constexpr uint32_t kFWaves = 8;
constexpr uint32_t kFThreads = kFWaves * kWave;
// 1 M tile x 8 N tiles a wave, so the 8 waves cover 8 DISTINCT row tiles. With A coming straight
// from global there is no LDS copy to share, so any two waves on the same rows would each fetch it —
// which is what a 4(M)x2(N) grid does, and it costs real throughput. Distinct rows per wave makes
// every A byte read exactly once per block. B is still shared through LDS by all eight.
constexpr uint32_t kFMTiles = 1, kFNTiles = 8;

// ---- preshuffled weights ------------------------------------------------------------------------
//
// `shuffle_weight(w, (16,16)).reshape(N//16, K*16)`: the weight is permuted ONCE at load into
// exactly the order the fragment wants, so a wave reads its whole A fragment with one coalesced
// 8-byte-a-lane load that covers a full cacheline. A never touches LDS again: no staging, no
// swizzle, no barrier.
//
// Layout: for row tile `rt` and K tile `kt`, the 256 bytes are stored in LANE order, so the element
// at (lane, j) is W[rt*16 + lane%16][kt*16 + (lane/16)*8 + j] — the fragment mapping from wmma.h,
// written down. One copy serves prefill and decode both, so it costs no extra VRAM.
//
// The layout is the remaining lever, not the schedule: pipelining depth, split-K and occupancy have
// all been tried here and none of them moves this kernel much.
__device__ __forceinline__ wmma::i2v load_frag8_ps(const uint8_t* __restrict__ W, uint32_t rt,
                                                   uint32_t kt, uint32_t ktiles, uint32_t lane) {
  const uint2 v = *(const uint2*)(W + (((size_t)rt * ktiles + kt) * 32 + lane) * 8);
  wmma::i2v f;
  f[0] = (int)v.x;
  f[1] = (int)v.y;
  return f;
}

__global__ void preshuffle_fp8_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                                      uint32_t cols, uint32_t ktiles, uint32_t groups) {
  const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= groups) return;
  const uint32_t lane = g & 31u, kt = (g >> 5) % ktiles, rt = (g >> 5) / ktiles;
  const uint8_t* p = src + (size_t)(rt * 16 + (lane & 15u)) * cols + kt * 16 + (lane >> 4) * 8;
  *(uint2*)(dst + (size_t)g * 8) = *(const uint2*)p;
}

// An fp8 fragment is 8 bytes a lane, so the granule is 8 B and a 64 B row holds 8 of them — the
// same count the bf16 path had at 16 B, so the same XOR swizzle applies and for the same reason.
template <uint32_t FK>
__device__ __forceinline__ wmma::i2v load_frag8(const uint8_t* __restrict__ lds, uint32_t row0,
                                                uint32_t k0, uint32_t lane) {
  const uint32_t row = row0 + wmma::frag_c(lane);
  const uint32_t k = k0 + wmma::frag_k0(lane);           // 8-aligned
  const uint2 v = *(const uint2*)(lds + (size_t)row * FK + ((((k >> 3) ^ (row & 7u))) << 3));
  wmma::i2v f;
  f[0] = (int)v.x;
  f[1] = (int)v.y;
  return f;
}

// Stage 32 contiguous bytes a thread as four 8-byte granules — a pure copy, no arithmetic at all.
// Identical for A and B, which is the whole point of keeping the activation in fp8 AND token-major:
// both operands are [row][k] byte planes and both stage with the same code.
// The 32 bytes a thread stages, held in registers between the global load and the LDS store.
//
// They are split because of what sits between them. Issued as one `stage8`, the load is followed
// immediately by the barrier that publishes it, so a thread has exactly one 32-byte request in
// flight and then waits — and the K loop becomes a chain of memory latencies rather than a stream.
// The symptom is a block sweep that saturates well below the card's coalesced ceiling and then does
// not move when the block count doubles.
struct Stage8Reg { uint4 v, w; };

__device__ __forceinline__ Stage8Reg stage8_load(const uint8_t* __restrict__ src, bool live) {
  Stage8Reg r;
  r.v = make_uint4(0, 0, 0, 0);
  r.w = make_uint4(0, 0, 0, 0);
  if (live) {
    r.v = *(const uint4*)src;
    r.w = *(const uint4*)(src + 16);
  }
  return r;
}

template <uint32_t FK>
__device__ __forceinline__ void stage8_store(uint8_t* __restrict__ lds, uint32_t row, uint32_t k0,
                                             Stage8Reg r) {
  const uint4 v = r.v, w = r.w;
  const uint2 gr[4] = {make_uint2(v.x, v.y), make_uint2(v.z, v.w), make_uint2(w.x, w.y),
                       make_uint2(w.z, w.w)};
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    const uint32_t k = k0 + i * 8;
    *(uint2*)(lds + (size_t)row * FK + ((((k >> 3) ^ (row & 7u))) << 3)) = gr[i];
  }
}

// `PS` picks where A comes from. true: the preshuffled weight, read straight to registers with one
// coalesced 8-byte-a-lane load and no LDS at all. false: the ORDINARY row-major weight, staged
// through LDS by exactly the same `stage8` that stages B — which is possible only because both
// operands are [row][k] byte planes once the activation is fp8 and token-major.
//
// The false path exists because not every registered tensor is preshuffled — see `ps_for` in
// dense_gpu.hip for which are and why.
//
// ---- NT: the N tile is a floor, and at nb <= 16 the floor is the whole cost -------------------
//
// `NT` is how many 16-wide token tiles a wave carries. Eight is the prefill shape and every comment
// above is about it. One exists for the SPECULATIVE batch: DSPark drafts gamma=7 and the target
// verifies eight positions in a single pass, so decode's nb goes 1 -> 8 and never approaches 128.
//
// At the wide NT a speculative call computes a full N tile of which almost every column is zero
// padding. Three separate costs, and the arithmetic is the least of them:
//
//   * every WMMA instruction issued, nearly all of them multiplying by staged zeros;
//   * a full set of accumulator VGPRs (`raw` and `out` across every tile), which is what pins the
//     kernel to a couple of waves a SIMD — on a kernel whose only latency hiding is other waves;
//   * the whole `s_b` allocation for a fraction of it live.
//
// NT=1 pays one accumulator and issues one WMMA per K tile. Nothing else changes: same staging, same
// swizzle, same scale fold, same epilogue.
//
// ---- MW: and then the block is too WIDE in M ---------------------------------------------------
//
// With NT=1 the kernel is purely weight-bandwidth-bound, and throughput is then a clean monotone
// function of ONE variable: the block count. It keeps climbing well past the point where the card is
// nominally full.
//
// Eight waves x 16 rows is a 128-row block, and decode's narrowest shards are only a few such blocks
// on a card of this size. `MW` is the wave count and it is the M tile, so lowering it multiplies the
// block count. It costs nothing: A staging is 32 B a thread at every MW, because rows and threads
// halve together.
//
// This is NOT the prefill lever. There nb=1536 gives 12 N tiles on top of the M grid and the block
// count is already in the hundreds; kFWaves stays 8 and the wide instantiation is untouched.
struct GemmJobsArgs {
  const void* w[kGemmJobsMax];
  const uint8_t* s[kGemmJobsMax];
  float* y[kGemmJobsMax];
  uint32_t rows[kGemmJobsMax];
  uint32_t tile_end[kGemmJobsMax];      // cumulative M-tile count, so blockIdx.x can be resolved
  // Per job, for a group that slices ONE weight and ONE activation rather than holding several:
  // the o_lora projection is n_groups windows of the heads buffer against n_groups row bands of
  // one shard, which is a launch parameter difference and nothing else.
  uint32_t xoff[kGemmJobsMax];
  uint32_t woff[kGemmJobsMax];
  uint32_t n;
};

// The body, with `m0` and the K range handed in — same split the WMMA kernel took, and for the
// same reason: a grouped wrapper maps blockIdx.x through a job table so several weight matrices
// sharing an activation are one launch, and this is far too much staging code to keep two of.
template <bool Full, bool PS, uint32_t NT, uint32_t MW, uint32_t FK>
__device__ __forceinline__ void gemm_fp8_body(
    const uint8_t* __restrict__ Wp, const uint8_t* __restrict__ Sw, const uint8_t* __restrict__ Xp,
    const float* __restrict__ Sx, float* __restrict__ Y, uint32_t rows, uint32_t cols, uint32_t nb,
    uint32_t sc_cols, uint32_t x_off, uint32_t w_row_off, uint32_t x_ld, uint32_t sx_ld,
    int accum, uint32_t m0, uint32_t kbeg, uint32_t kend) {
  constexpr uint32_t FN = NT * 16;                       // tokens a block, kFN when NT == kFNTiles
  constexpr uint32_t FM = MW * kFMTiles * 16;            // rows a block, kFM when MW == kFWaves
  // FK BYTES a row a K tile, and for the narrow instantiation that is a whole 256-byte cacheline
  // — this card's line size, not 128. At FK = 64 a thread reads a quarter of a line and the other
  // three quarters are only paid for if the line survives in L0 for three more K tiles.
  constexpr uint32_t kNG = FK / 64;                      // 32-byte granule pairs a thread
  __shared__ uint8_t s_b[FN * FK];
  // Only instantiated for PS == false; an empty array is not legal, so it is one byte otherwise.
  __shared__ uint8_t s_a[PS ? 1 : FM * FK];

  const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
  const uint32_t n0 = blockIdx.y * FN;
  // Wave grid 8(M) x 1(N): a wave owns 16 rows and all FN tokens.
  const uint32_t wm = wave * (kFMTiles * 16), wn = 0;

  wmma::f8v raw[kFMTiles][NT], out[kFMTiles][NT];
#pragma unroll
  for (uint32_t i = 0; i < kFMTiles; ++i)
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) {
      raw[i][j] = wmma::acc_zero();
      out[i][j] = wmma::acc_zero();
    }

  // B: FN tokens x 64 B, 32 B a thread. At FN=128 that is every thread; at FN=16 it is the first
  // 32, and the other 224 sit out a 1 KB copy rather than being given a share of a smaller one.
  const uint32_t sn = tid >> 1, sk = (tid & 1u) * 32;
  const uint32_t ktiles = cols >> 4;
  const uint32_t rt0 = (w_row_off + m0 + wm) >> 4;       // this wave's first row tile
  const uint32_t srow = (w_row_off + m0) >> 7;
  // The tokens this lane owns, one per N tile — a lane owns column `frag_c` of each of its NT
  // accumulators, and those are 16 columns apart. Folding the scale with only the first one is
  // wrong for the others, which reads as a ~2.0 relative error rather than as noise.
  const uint32_t tok0 = n0 + wn + wmma::frag_c(lane);

  // Row-major A staging assignment, identical to B's: 128 rows x 64 B, 32 B a thread.
  const uint32_t am = tid >> 1;

  // x_ld, not cols: the quantised activation is ONE buffer per sublayer and a grouped GEMM reads a
  // `cols`-wide window of it at `x_off`. Using cols as the row stride silently reads the wrong
  // token once the two differ, which is a wrong activation that still looks fluent.
  const uint8_t* bsrc = Xp + (size_t)(n0 + sn) * x_ld + x_off + sk;
  const uint8_t* asrc = Wp + (size_t)(w_row_off + m0 + am) * cols + sk;
  const bool b_row = sn < FN && (Full || (n0 + sn) < nb);
  const bool a_row = !PS && (Full || (m0 + am) < rows);
  // One K tile ahead, in registers. The load for tile c0+kFK is issued before the WMMA for tile c0
  // and consumed after it, so the memory latency sits under the arithmetic instead of in front of a
  // barrier. Nothing about what is computed changes -- same operands, same order, bit-identical.
  Stage8Reg pb[kNG], pa[kNG];
#pragma unroll
  for (uint32_t g = 0; g < kNG; ++g) {
    const uint32_t o = kbeg + g * 64;
    pb[g] = stage8_load(bsrc + o, b_row && (Full || o + sk + 32 <= cols));
    pa[g] = stage8_load(asrc + o, a_row && (Full || o + sk + 32 <= cols));
  }

  for (uint32_t c0 = kbeg; c0 < kend; c0 += FK) {
    __syncthreads();
#pragma unroll
    for (uint32_t g = 0; g < kNG; ++g) {
      if (sn < FN) stage8_store<FK>(s_b, sn, sk + g * 64, pb[g]);
      if (!PS) stage8_store<FK>(s_a, am, sk + g * 64, pa[g]);
    }
    __syncthreads();
    const uint32_t c1 = c0 + FK;
    if (c1 < kend) {
#pragma unroll
      for (uint32_t g = 0; g < kNG; ++g) {
        const uint32_t o = c1 + g * 64;
        const bool live = Full || (o + sk + 32 <= cols);
        if (sn < FN) pb[g] = stage8_load(bsrc + o, b_row && live);
        if (!PS) pa[g] = stage8_load(asrc + o, a_row && live);
      }
    }

    // PRESHUFFLED A: every fragment for this whole K tile, issued before any of them is used. Loaded
    // one at a time inside the loop below, each mma waits on its own fetch, and the scale fold stops
    // the compiler hoisting past it — which is enough to make the preshuffled path SLOWER than
    // row-major despite reading the same bytes with no LDS round trip and no barrier.
    constexpr uint32_t kKT = FK / wmma::kK;
    wmma::i2v apf[PS ? kFMTiles : 1][PS ? kKT : 1];
    if (PS) {
#pragma unroll
      for (uint32_t t = 0; t < kKT; ++t)
#pragma unroll
        for (uint32_t i = 0; i < kFMTiles; ++i)
          apf[i][t] = load_frag8_ps(Wp, rt0 + i, (c0 + t * wmma::kK) >> 4, ktiles, lane);
    }
#pragma unroll
    for (uint32_t ks = 0; ks < FK; ks += wmma::kK) {
      wmma::i2v af[kFMTiles], bf[NT];
#pragma unroll
      for (uint32_t i = 0; i < kFMTiles; ++i)
        af[i] = PS ? apf[i][ks / wmma::kK] : load_frag8<FK>(s_a, wm + i * 16, ks, lane);
#pragma unroll
      for (uint32_t j = 0; j < NT; ++j) bf[j] = load_frag8<FK>(s_b, wn + j * 16, ks, lane);
#pragma unroll
      for (uint32_t i = 0; i < kFMTiles; ++i)
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j) raw[i][j] = wmma::mma_fp8(af[i], bf[j], raw[i][j]);

    // End of a 128-column scale block: fold both scales in and reset. INSIDE the ks loop, because
    // at FK = 256 one staged tile spans two scale blocks and folding once would apply the second
    // block's scale to the first block's products. `c0` is uniform and `ks` is a constant after
    // unrolling, so the test is uniform and mostly compile-time.
    if ((c0 + ks + wmma::kK) % 128u == 0 || c0 + ks + wmma::kK >= kend) {
      const uint32_t kb = (c0 + ks) >> 7;
      const uint32_t e = Sw[(size_t)srow * sc_cols + kb];
      const float ws = __uint_as_float(e ? (e << 23) : 0u);
#pragma unroll
      for (uint32_t j = 0; j < NT; ++j) {
        const uint32_t tok = tok0 + j * 16;
        const float s = ws * ((Full || tok < nb) ? Sx[(size_t)tok * sx_ld + (x_off >> 7) + kb]
                                                 : 0.0f);
#pragma unroll
        for (uint32_t i = 0; i < kFMTiles; ++i)
#pragma unroll
          for (uint32_t t = 0; t < 8; ++t) {
            out[i][j][t] += raw[i][j][t] * s;
            raw[i][j][t] = 0.0f;
          }
      }
    }
    }
  }

  const uint32_t c = wmma::frag_c(lane);
#pragma unroll
  for (uint32_t i = 0; i < kFMTiles; ++i)
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) {
      const uint32_t col = n0 + wn + j * 16 + c;
      if (!Full && col >= nb) continue;
      const uint32_t rb = m0 + wm + i * 16;
#pragma unroll
      for (uint32_t t = 0; t < 8; ++t) {
        const uint32_t r = rb + wmma::acc_row(lane, t);
        if (!Full && r >= rows) continue;
        // `accum` is launch-uniform and this is the epilogue, so the test is scalar and runs once a
        // store — nothing of it reaches the K loop.
        if (accum) Y[(size_t)r * nb + col] += out[i][j][t];
        else       Y[(size_t)r * nb + col]  = out[i][j][t];
      }
    }
}

template <bool Full, bool PS, uint32_t NT, uint32_t MW, uint32_t FK>
__global__ __launch_bounds__(MW * kWave) void batch_gemm_fp8_kernel(
    const uint8_t* __restrict__ Wp, const uint8_t* __restrict__ Sw, const uint8_t* __restrict__ Xp,
    const float* __restrict__ Sx, float* __restrict__ Y, uint32_t rows, uint32_t cols, uint32_t nb,
    uint32_t sc_cols, uint32_t x_off, uint32_t w_row_off, uint32_t x_ld, uint32_t sx_ld,
    int accum, uint32_t ck) {
  // Split-K. `ck` is a multiple of both 128 (the scale block) and FK (the staged tile), and ksplit
  // is derived FROM it at the launcher, so every slice has work and starts on a scale boundary.
  const uint32_t kbeg = ck ? blockIdx.z * ck : 0u;
  const uint32_t kend = ck ? min(cols, kbeg + ck) : cols;
  gemm_fp8_body<Full, PS, NT, MW, FK>(Wp, Sw, Xp, Sx, Y + (size_t)blockIdx.z * rows * nb, rows,
                                      cols, nb, sc_cols, x_off, w_row_off, x_ld, sx_ld, accum,
                                      blockIdx.x * (MW * kFMTiles * 16), kbeg, kend);
}

// Several fp8 weight matrices against one activation, one launch. See batch_gemm_multi_kernel.
template <bool Full, bool PS, uint32_t NT, uint32_t MW, uint32_t FK>
__global__ __launch_bounds__(MW * kWave) void batch_gemm_fp8_multi_kernel(
    GemmJobsArgs J, const uint8_t* __restrict__ Xp, const float* __restrict__ Sx, uint32_t cols,
    uint32_t nb, uint32_t sc_cols, uint32_t x_ld, uint32_t sx_ld, uint32_t ck, uint32_t total) {
  uint32_t j = 0;
  while (j + 1u < J.n && blockIdx.x >= J.tile_end[j]) ++j;
  const uint32_t tbeg = j ? J.tile_end[j - 1] : 0u;
  const uint32_t kbeg = ck ? blockIdx.z * ck : 0u;
  const uint32_t kend = ck ? min(cols, kbeg + ck) : cols;
  gemm_fp8_body<Full, PS, NT, MW, FK>(
      (const uint8_t*)J.w[j], J.s[j], Xp, Sx, J.y[j] + (size_t)blockIdx.z * total * nb, J.rows[j],
      cols, nb, sc_cols, J.xoff[j], J.woff[j], x_ld, sx_ld, 0,
      (blockIdx.x - tbeg) * (MW * kFMTiles * 16), kbeg, kend);
}

// `Full` = every dimension is an exact multiple of its tile, which every dense tensor in this model
// satisfies (rows and nb are 128-aligned, cols is 64-aligned). The checked variant compiled to 107
// s_and_saveexec_b32 and 140 s_cbranch_execz — a predicated branch around every staged element and
// every epilogue store, in the inner loop, for tails that never occur. The host picks.
// MT/NT/NW are the wave geometry: NW waves along N owning NT tiles each, the remaining
// kWWaves/NW along M owning MT. The N extent is NW*NT*16 and the M extent is (kWWaves/NW)*MT*16.
//
//   <2, 4, 2>   128 x 128, the prefill tile the constants above are named for
//   <1, 1, 1>   128 x  16, one accumulator a wave, for a SPECULATIVE BLOCK
//
// The narrow one exists because the comment above — "a smaller N tile would re-read the weights more
// times and make it bandwidth-bound" — inverts at a narrow batch. At nb=16 there is exactly one N
// tile either way, so a narrower tile re-reads nothing; all it does is stop computing the columns
// that do not exist. At that width the kernel is tile-bound, not bandwidth-bound, and the wide tile
// reads dense weight at a fraction of what the matvec path gets on the same bytes.
//
// M stays 128 in both, which keeps the A staging, `m0`, `srow` and the scale-tile LUT untouched —
// the narrow variant drops MT to 1 and spends the freed waves on M instead. It also costs one
// accumulator a wave instead of eight, freeing the VGPR budget §e.5 spends on 8 tiles for occupancy.

// The body, with `m0` and the K range handed in. Two __global__ wrappers call it: the ordinary one
// below, and the GROUPED one, which maps blockIdx.x through a job table so several weight matrices
// sharing an activation are one launch instead of several. Extracted rather than copied — the fusion
// is worth well over a third, and this is 120 lines of WMMA staging that must not drift between two
// versions of itself.
template <int Q, bool Full, bool XB, uint32_t MT, uint32_t NT, uint32_t NW>
__device__ __forceinline__ void gemm_wmma_body(
    const void* __restrict__ Wp, const uint8_t* __restrict__ S, const void* __restrict__ Xp,
    float* __restrict__ Y, uint32_t rows, uint32_t cols, uint32_t nb, uint32_t sc_cols,
    uint32_t x_off, uint32_t w_row_off, uint32_t m0, uint32_t kbeg, uint32_t kend) {
  constexpr uint32_t kMW = kWWaves / NW;                   // waves along M
  constexpr uint32_t kMExt = kMW * MT * 16;                // == kWM in both instantiations
  constexpr uint32_t kNExt = NW * NT * 16;
  __shared__ uint16_t s_a[kMExt * kWLdsRow];
  __shared__ uint16_t s_b[kNExt * kWLdsRow];

  __shared__ uint16_t s_lut[256];                  // E4M3 byte -> scaled bf16, fp8 only

  const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
  const uint32_t n0 = blockIdx.y * kNExt;
  // Wave grid kMW(M) x NW(N): at <2,4,2> a wave owns 32 rows and 64 tokens, at <1,1,1> 16 and 16.
  const uint32_t wm = (wave / NW) * (MT * 16), wn = (wave % NW) * (NT * 16);

  wmma::f8v acc[MT][NT];
#pragma unroll
  for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) acc[i][j] = wmma::acc_zero();

  // Staging assignments. Both stage 128 rows x 64 k as 4 granules of 8 halves per thread, so both
  // write 16 B at a time instead of 32 scalar halves.
  const uint32_t sm = tid >> 1, sh = (tid & 1u) * 32;          // A: row, k base
  // B: 4 consecutive tokens x 8 k a thread. The activation is dim-major ([cols][nb]), so "32 k for
  // one token" — the obvious assignment — is 32 separate scalar loads that cannot be vectorised, and
  // that staging alone dominated the kernel. Four tokens wide makes each read a float4, and the
  // [k][tok] -> [tok][k] transpose falls out in registers because both indices are compile-time
  // constant after unrolling.
  //
  // A narrow tile has less B than the block has threads: 16 tokens x 64 k is 32 threads' worth, so
  // the other 224 sit the staging out. They are still needed for A, which is the same 128 rows.
  constexpr uint32_t kBTG = kNExt / 4;                         // thread groups along the token axis
  constexpr uint32_t kBThreads = kBTG * (kWK / 8);
  const uint32_t bt = (tid % kBTG) * 4, bk = (tid / kBTG) * 8;
  const bool b_stager = kBThreads == kWThreads || tid < kBThreads;

  const uint32_t srow = (w_row_off + m0) >> 7;     // uniform: m0 and w_row_off are 128-aligned
  uint32_t lut_tile = 0xffffffffu;

  for (uint32_t c0 = kbeg; c0 < kend; c0 += kWK) {
    __syncthreads();

    // The dequant table, rebuilt only when the scale tile changes. `c0 >> 7` is uniform across the
    // block, so the branch is uniform and the barrier inside it is legal.
    if (Q != AFF_BF16 && (c0 >> 7) != lut_tile) {
      lut_tile = c0 >> 7;
      const uint32_t e = S[(size_t)srow * sc_cols + lut_tile];
      s_lut[tid] = e4m3_bf16_bits(tid, __uint_as_float(e ? (e << 23) : 0u));
      __syncthreads();
    }

    // ---- A: the weights. 32 contiguous elements a thread in two vector loads — cols, c0 and sh
    // are all multiples of 32, so the address is always 16-byte aligned. Everything stays in
    // registers: no local array is ever addressed through a pointer, because a spill to scratch
    // here would cost more than the dequant it replaces.
    // Each 16-byte granule is built and written before the next is started. Holding all 32 staged
    // elements live first cost 174 VGPRs — occupancy 6 waves/SIMD against the 16 that ≤96 buys, and
    // §e.3 says the only way to hide VALU behind WMMA on RDNA4 is to have other waves resident.
    {
      const uint32_t grow = w_row_off + m0 + sm;
      const bool live = Full || ((m0 + sm) < rows && c0 + sh + 32 <= cols);
      uint16_t* dst = s_a + sm * kWLdsRow;
      if (live && Q == AFF_BF16) {
        const uint4* src = (const uint4*)((const uint16_t*)Wp + (size_t)grow * cols + c0 + sh);
#pragma unroll
        for (uint32_t gi = 0; gi < 4; ++gi) {                    // one uint4 IS one granule
          const uint32_t k = sh + gi * 8;
          *(uint4*)(dst + ((((k >> 3) ^ (sm & 7u))) << 3)) = src[gi];
        }
      } else if (live) {
        const uint4* src = (const uint4*)((const uint8_t*)Wp + (size_t)grow * cols + c0 + sh);
#pragma unroll
        for (uint32_t h = 0; h < 2; ++h) {                       // 16 source bytes = 2 granules
          const uint4 v = src[h];
          const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (uint32_t gi = 0; gi < 2; ++gi) {
            uint32_t d[4];
#pragma unroll
            for (uint32_t q = 0; q < 4; ++q) {
              const uint32_t j = gi * 8 + q * 2;
              d[q] = (uint32_t)s_lut[(w[j >> 2] >> (8 * (j & 3))) & 0xffu] |
                     ((uint32_t)s_lut[(w[(j + 1) >> 2] >> (8 * ((j + 1) & 3))) & 0xffu] << 16);
            }
            const uint32_t k = sh + h * 16 + gi * 8;
            *(uint4*)(dst + ((((k >> 3) ^ (sm & 7u))) << 3)) = make_uint4(d[0], d[1], d[2], d[3]);
          }
        }
      } else {
        // Tail: the row is past `rows`, or fewer than 32 columns remain. Rare — every dense tensor
        // in this model has cols a multiple of 64 — so it needs to be right, not fast.
#pragma unroll
        for (uint32_t gi = 0; gi < 4; ++gi) {
          uint32_t d[4];
#pragma unroll
          for (uint32_t q = 0; q < 4; ++q) {
            uint32_t h2[2] = {0, 0};
#pragma unroll
            for (uint32_t e = 0; e < 2; ++e) {
              const uint32_t k = sh + gi * 8 + q * 2 + e;
              if ((m0 + sm) >= rows || c0 + k >= cols) continue;
              const size_t idx = (size_t)grow * cols + c0 + k;
              h2[e] = Q == AFF_BF16 ? ((const uint16_t*)Wp)[idx]
                                    : s_lut[((const uint8_t*)Wp)[idx]];
            }
            d[q] = h2[0] | (h2[1] << 16);
          }
          const uint32_t k = sh + gi * 8;
          *(uint4*)(dst + ((((k >> 3) ^ (sm & 7u))) << 3)) = make_uint4(d[0], d[1], d[2], d[3]);
        }
      }
    }

    // ---- B: the activations, f32 -> bf16. For a fixed k the 128 threads of a half-block read 128
    // consecutive floats, so every global read here is a full 512 B coalesced line.
    if (b_stager) {
      const size_t xo = (size_t)(x_off + c0 + bk) * nb + n0 + bt;
      const bool vec = Full || ((nb & 3u) == 0 && n0 + bt + 4 <= nb && c0 + bk + 8 <= cols);
      // The halves that go to LDS. XB means the activation is ALREADY bf16 in memory, which is
      // free rather than lossy: this staging has always truncated to bf16 right here, so a bf16
      // arena hands the matrix core exactly the operand it was going to get anyway — it just moves
      // half the bytes to do it, and skips 32 v_cvt per thread per stage.
      uint16_t h[8][4];
      if (XB) {
        const uint16_t* xb = (const uint16_t*)Xp + xo;
        if (vec) {
#pragma unroll
          for (uint32_t kk = 0; kk < 8; ++kk) {
            const ushort4 v = *(const ushort4*)(xb + (size_t)kk * nb);
            h[kk][0] = v.x; h[kk][1] = v.y; h[kk][2] = v.z; h[kk][3] = v.w;
          }
        } else {
#pragma unroll
          for (uint32_t kk = 0; kk < 8; ++kk)
#pragma unroll
            for (uint32_t tt = 0; tt < 4; ++tt) {
              const bool in = (n0 + bt + tt) < nb && (c0 + bk + kk) < cols;
              h[kk][tt] = in ? xb[(size_t)kk * nb + tt] : (uint16_t)0;
            }
        }
      } else {
        const float* xb = (const float*)Xp + xo;
        float f[8][4];
        if (vec) {
#pragma unroll
          for (uint32_t kk = 0; kk < 8; ++kk) {
            const float4 v = *(const float4*)(xb + (size_t)kk * nb);
            f[kk][0] = v.x; f[kk][1] = v.y; f[kk][2] = v.z; f[kk][3] = v.w;
          }
        } else {
#pragma unroll
          for (uint32_t kk = 0; kk < 8; ++kk)
#pragma unroll
            for (uint32_t tt = 0; tt < 4; ++tt) {
              const bool in = (n0 + bt + tt) < nb && (c0 + bk + kk) < cols;
              f[kk][tt] = in ? xb[(size_t)kk * nb + tt] : 0.0f;
            }
        }
#pragma unroll
        for (uint32_t kk = 0; kk < 8; ++kk)
#pragma unroll
          for (uint32_t tt = 0; tt < 4; ++tt) h[kk][tt] = wmma::f32_to_bf16(f[kk][tt]);
      }
      // One granule a token: k is 8-aligned by construction, so each store is a whole 16 B.
#pragma unroll
      for (uint32_t tt = 0; tt < 4; ++tt) {
        const uint32_t row = bt + tt;
        uint32_t d[4];
#pragma unroll
        for (uint32_t q = 0; q < 4; ++q)
          d[q] = (uint32_t)h[q * 2][tt] | ((uint32_t)h[q * 2 + 1][tt] << 16);
        *(uint4*)(s_b + row * kWLdsRow + ((((bk >> 3) ^ (row & 7u))) << 3)) =
            make_uint4(d[0], d[1], d[2], d[3]);
      }
    }
    __syncthreads();

    // One long accumulator chain of a single opcode: §e.2 row 2 says back-to-back same-type D->C is
    // the fast case, and mixing WMMA types in the inner loop is what stalls.
#pragma unroll
    for (uint32_t ks = 0; ks < kWK; ks += wmma::kK) {
      wmma::s8v af[MT], bf[NT];
#pragma unroll
      for (uint32_t i = 0; i < MT; ++i) af[i] = load_frag(s_a, wm + i * 16, ks, lane);
#pragma unroll
      for (uint32_t j = 0; j < NT; ++j) bf[j] = load_frag(s_b, wn + j * 16, ks, lane);
#pragma unroll
      for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j) acc[i][j] = wmma::mma_bf16(af[i], bf[j], acc[i][j]);
    }
  }

  const uint32_t c = wmma::frag_c(lane);
#pragma unroll
  for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) {
      const uint32_t col = n0 + wn + j * 16 + c;
      if (!Full && col >= nb) continue;
      const uint32_t rb = m0 + wm + i * 16;
#pragma unroll
      for (uint32_t t = 0; t < 8; ++t) {
        const uint32_t r = rb + wmma::acc_row(lane, t);
        if (Full || r < rows) Y[(size_t)r * nb + col] = acc[i][j][t];
      }
    }
}

template <int Q, bool Full, bool XB, uint32_t MT = kWMTiles, uint32_t NT = kWNTiles,
          uint32_t NW = 2>
__global__ __launch_bounds__(kWThreads) void batch_gemm_wmma_kernel(
    const void* __restrict__ Wp, const uint8_t* __restrict__ S, const void* __restrict__ Xp,
    float* __restrict__ Y, uint32_t rows, uint32_t cols, uint32_t nb, uint32_t sc_cols,
    uint32_t x_off, uint32_t w_row_off, uint32_t ck) {
  // Split-K. `ck` is a multiple of 128 so each split still starts on a scale-block boundary, which
  // is what lets the UE8M0 fold stay correct inside a partial K range. ck == 0 means no split.
  // No `kbeg >= kend` bail: ksplit is derived FROM ck at the launcher (ceil(cols/ck)), so every
  // slice has work. It used to be derived independently, which could leave the last one or two
  // empty — they returned without writing, and splitk_reduce summed their uninitialised partials.
  const uint32_t kbeg = ck ? blockIdx.z * ck : 0u;
  const uint32_t kend = ck ? min(cols, kbeg + ck) : cols;
  gemm_wmma_body<Q, Full, XB, MT, NT, NW>(Wp, S, Xp, Y + (size_t)blockIdx.z * rows * nb, rows, cols,
                                          nb, sc_cols, x_off, w_row_off,
                                          blockIdx.x * (kWWaves / NW * MT * 16), kbeg, kend);
}

// Several weight matrices against one activation, in one launch. blockIdx.x is a GLOBAL M-tile
// index; `tile_end` maps it to a job in at most kGemmJobsMax scalar compares, all uniform.
//
// No split-K here: the whole point is that the union of the jobs' row tiles already fills the grid.
template <int Q, bool Full, bool XB, uint32_t MT = 1, uint32_t NT = 1, uint32_t NW = 1>
__global__ __launch_bounds__(kWThreads) void batch_gemm_multi_kernel(GemmJobsArgs J,
                                                                    const void* __restrict__ Xp,
                                                                    uint32_t cols, uint32_t nb,
                                                                    uint32_t sc_cols, uint32_t ck,
                                                                    uint32_t total) {
  uint32_t j = 0;
  while (j + 1u < J.n && blockIdx.x >= J.tile_end[j]) ++j;
  const uint32_t tbeg = j ? J.tile_end[j - 1] : 0u;
  // Split-K as well, and it is not optional: the union of the jobs' row tiles is what fills the
  // grid, and for the compressor group that is only a handful of blocks — fusing without the split
  // is a net loss. The partial plane is one [z][total][nb] block, so `y[j]` already points at this
  // job's band of it and z strides by the whole thing.
  const uint32_t kbeg = ck ? blockIdx.z * ck : 0u;
  const uint32_t kend = ck ? min(cols, kbeg + ck) : cols;
  gemm_wmma_body<Q, Full, XB, MT, NT, NW>(J.w[j], J.s[j], Xp,
                                          J.y[j] + (size_t)blockIdx.z * total * nb, J.rows[j], cols,
                                          nb, sc_cols, J.xoff[j], J.woff[j],
                                          (blockIdx.x - tbeg) * (kWWaves / NW * MT * 16), kbeg,
                                          kend);
}

// ---- the f32 GEMM, as three bf16 WMMA passes ---------------------------------------------------
//
// The only f32 GEMM the batched path runs is the router gate, and it was the last scalar kernel in
// prefill — a percent-scale share of the whole phase, at a small fraction of the matrix core's peak.
// It stayed scalar because rounding the gate to bf16 is a REAL loss, and this GEMM feeds a discrete
// choice: top-k over near-tied scores, where a perturbation does not move a number, it swaps a whole
// expert and every byte of weight that goes with it.
//
// Both things are true at once, so split instead of round. With w = w_hi + w_lo and x = x_hi + x_lo,
// each term a bf16 carrying 8 mantissa bits,
//
//     w*x = w_hi*x_hi + w_hi*x_lo + w_lo*x_hi + w_lo*x_lo
//
// and the last term is negligible and dropped. Three WMMA passes over the same accumulator then
// carry about 16 mantissa bits against f32's 24, which puts the dot product's relative error at the
// same ORDER as the f32 kernel's own summation-order error over the same number of terms, and orders
// of magnitude tighter than a single bf16 pass. The accumulator is f32 in the WMMA either way.
//
// Geometry: 64 rows x 128 tokens a block, K in steps of 64. kWLdsRow is 64 halves, so `load_frag`
// above applies unchanged and the fragment reads stay conflict-free. A wave owns ONE
// 16-row tile and four 16-token tiles: 4 accumulators, and 10 live fragments feeding 12 mma per K
// step. LDS is 48 KiB — one block a WGP, which is what 8 waves of this shape want anyway.
constexpr uint32_t kGM = 64, kGN = 128, kGK = kWLdsRow;
constexpr uint32_t kGWaves = 8;
constexpr uint32_t kGThreads = kGWaves * kWave;
constexpr uint32_t kGNTiles = 4;

// v -> (bf16 hi, bf16 lo) with v ~= hi + lo. `v - hi` is exact in f32 (Sterbenz: hi is within a
// factor of two of v), so lo carries the next 8 mantissa bits and nothing is lost to the subtract.
__device__ __forceinline__ void bf16_split(float v, uint16_t* hi, uint16_t* lo) {
  const uint16_t h = wmma::f32_to_bf16(v);
  *hi = h;
  *lo = wmma::f32_to_bf16(v - wmma::bf16_to_f32(h));
}

template <bool Full>
__global__ __launch_bounds__(kGThreads) void batch_gemm_f32x3_kernel(
    const float* __restrict__ Wp, const float* __restrict__ X, float* __restrict__ Y,
    uint32_t rows, uint32_t cols, uint32_t nb) {
  __shared__ uint16_t s_ah[kGM * kWLdsRow];
  __shared__ uint16_t s_al[kGM * kWLdsRow];
  __shared__ uint16_t s_bh[kGN * kWLdsRow];
  __shared__ uint16_t s_bl[kGN * kWLdsRow];

  const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
  const uint32_t m0 = blockIdx.x * kGM, n0 = blockIdx.y * kGN;
  // Wave grid 4(M) x 2(N): a wave owns 16 rows and 64 tokens.
  const uint32_t wm = (wave & 3u) * 16, wn = (wave >> 2) * (kGNTiles * 16);

  wmma::f8v acc[kGNTiles];
#pragma unroll
  for (uint32_t j = 0; j < kGNTiles; ++j) acc[j] = wmma::acc_zero();

  // A: 16 consecutive weights a thread, so four threads of a row cover 256 B — one cacheline.
  const uint32_t am = tid >> 2, ak = (tid & 3u) * 16;
  // B: 4 consecutive tokens x 8 k a thread, exactly as the bf16 kernel above, so each read is a
  // float4 out of the dim-major activation rather than 32 unvectorisable scalars.
  const uint32_t bt = (tid & 31u) * 4, bk = (tid >> 5) * 8;

  for (uint32_t c0 = 0; c0 < cols; c0 += kGK) {
    __syncthreads();
    {
      const uint32_t r = m0 + am;
      const bool vec = Full || ((cols & 3u) == 0 && r < rows && c0 + ak + 16 <= cols);
      float f[16];
      if (vec) {
        const float4* p = (const float4*)(Wp + (size_t)r * cols + c0 + ak);
#pragma unroll
        for (uint32_t q = 0; q < 4; ++q) {
          const float4 v = p[q];
          f[q * 4 + 0] = v.x; f[q * 4 + 1] = v.y; f[q * 4 + 2] = v.z; f[q * 4 + 3] = v.w;
        }
      } else {
#pragma unroll
        for (uint32_t q = 0; q < 16; ++q) {
          const uint32_t c = c0 + ak + q;
          f[q] = (r < rows && c < cols) ? Wp[(size_t)r * cols + c] : 0.0f;
        }
      }
#pragma unroll
      for (uint32_t g = 0; g < 2; ++g) {          // two 16 B granules of 8 halves each
        uint32_t dh[4], dl[4];
#pragma unroll
        for (uint32_t q = 0; q < 4; ++q) {
          uint16_t h0, l0, h1, l1;
          bf16_split(f[g * 8 + q * 2], &h0, &l0);
          bf16_split(f[g * 8 + q * 2 + 1], &h1, &l1);
          dh[q] = (uint32_t)h0 | ((uint32_t)h1 << 16);
          dl[q] = (uint32_t)l0 | ((uint32_t)l1 << 16);
        }
        const uint32_t off = am * kWLdsRow + ((((ak + g * 8) >> 3) ^ (am & 7u)) << 3);
        *(uint4*)(s_ah + off) = make_uint4(dh[0], dh[1], dh[2], dh[3]);
        *(uint4*)(s_al + off) = make_uint4(dl[0], dl[1], dl[2], dl[3]);
      }
    }
    {
      const float* xb = X + (size_t)(c0 + bk) * nb + n0 + bt;
      const bool vec = Full || ((nb & 3u) == 0 && n0 + bt + 4 <= nb && c0 + bk + 8 <= cols);
      float f[8][4];
      if (vec) {
#pragma unroll
        for (uint32_t kk = 0; kk < 8; ++kk) {
          const float4 v = *(const float4*)(xb + (size_t)kk * nb);
          f[kk][0] = v.x; f[kk][1] = v.y; f[kk][2] = v.z; f[kk][3] = v.w;
        }
      } else {
#pragma unroll
        for (uint32_t kk = 0; kk < 8; ++kk)
#pragma unroll
          for (uint32_t tt = 0; tt < 4; ++tt) {
            const bool in = (n0 + bt + tt) < nb && (c0 + bk + kk) < cols;
            f[kk][tt] = in ? xb[(size_t)kk * nb + tt] : 0.0f;
          }
      }
#pragma unroll
      for (uint32_t tt = 0; tt < 4; ++tt) {
        const uint32_t row = bt + tt;
        uint32_t dh[4], dl[4];
#pragma unroll
        for (uint32_t q = 0; q < 4; ++q) {
          uint16_t h0, l0, h1, l1;
          bf16_split(f[q * 2][tt], &h0, &l0);
          bf16_split(f[q * 2 + 1][tt], &h1, &l1);
          dh[q] = (uint32_t)h0 | ((uint32_t)h1 << 16);
          dl[q] = (uint32_t)l0 | ((uint32_t)l1 << 16);
        }
        const uint32_t off = row * kWLdsRow + ((((bk >> 3) ^ (row & 7u))) << 3);
        *(uint4*)(s_bh + off) = make_uint4(dh[0], dh[1], dh[2], dh[3]);
        *(uint4*)(s_bl + off) = make_uint4(dl[0], dl[1], dl[2], dl[3]);
      }
    }
    __syncthreads();

#pragma unroll
    for (uint32_t ks = 0; ks < kGK; ks += wmma::kK) {
      const wmma::s8v ah = load_frag(s_ah, wm, ks, lane);
      const wmma::s8v al = load_frag(s_al, wm, ks, lane);
      wmma::s8v bh[kGNTiles], bl[kGNTiles];
#pragma unroll
      for (uint32_t j = 0; j < kGNTiles; ++j) bh[j] = load_frag(s_bh, wn + j * 16, ks, lane);
#pragma unroll
      for (uint32_t j = 0; j < kGNTiles; ++j) bl[j] = load_frag(s_bl, wn + j * 16, ks, lane);
#pragma unroll
      for (uint32_t j = 0; j < kGNTiles; ++j) {
        acc[j] = wmma::mma_bf16(ah, bh[j], acc[j]);
        acc[j] = wmma::mma_bf16(ah, bl[j], acc[j]);
        acc[j] = wmma::mma_bf16(al, bh[j], acc[j]);
      }
    }
  }

  const uint32_t c = wmma::frag_c(lane);
#pragma unroll
  for (uint32_t j = 0; j < kGNTiles; ++j) {
    const uint32_t col = n0 + wn + j * 16 + c;
    if (!Full && col >= nb) continue;
#pragma unroll
    for (uint32_t t = 0; t < 8; ++t) {
      const uint32_t r = m0 + wm + wmma::acc_row(lane, t);
      if (Full || r < rows) Y[(size_t)r * nb + col] = acc[j][t];
    }
  }
}

// ---- one thread per token, striped over the reduced dimension ---------------------------------
//
// kBlkTok tokens per block and kBlkStripe stripes over `dim`. A thread owns one token and walks its
// own stripe, so the per-token reductions that would otherwise need a cross-lane tree become a plain
// serial accumulation inside a lane -- which is the whole reason the batched buffers are dim-major.
//
// A WAVE IS kBlkTok TOKENS AT kWave/kBlkTok STRIPES, not 32 consecutive tokens: `bl` is the low bits
// of the thread id, so a wave issues several short segments far apart rather than one contiguous
// run. That looks like read amplification against the cacheline and is not, because the blocks
// covering a line's worth of tokens all read the same rows and L2 serves them. Widening kBlkTok to
// make each wave contiguous loses: it divides the block count by the same factor, and those blocks
// are what generate the parallel misses.
//
// AT A SPECULATIVE BLOCK the token axis is one tile, so this family is only a couple of workgroups.
// That is worth fixing where the kernel has work to spread, and each of those has a narrow sibling
// now. It is NOT worth fixing where the call is already short: a call whose cost is launch ramp does
// not get cheaper from more blocks.
//
// Total threads for the blocked family is nb * kBlkStripe, so kBlkStripe — not kBlkTok — is what
// sets parallelism. Too few stripes leaves one wave a SIMD with nothing to hide a dependent load
// behind.
constexpr uint32_t kBlkTok    = 8;
constexpr uint32_t kBlkStripe = 64;
constexpr uint32_t kBlkThreads = kBlkTok * kBlkStripe;

// EVERY thread of the block must call this, including ones whose token is past nb: it carries two
// barriers. The first is not redundant — `red` is reused by the second reduction in the same
// kernel, and without it that write races the first reduction's reads.
__device__ __forceinline__ float stripe_sum(float* red, uint32_t s, uint32_t bl, float v) {
  __syncthreads();
  red[s * kBlkTok + bl] = v;
  __syncthreads();
  float t = 0.0f;
  for (uint32_t k = 0; k < kBlkStripe; ++k) t += red[k * kBlkTok + bl];
  return t;
}

__global__ __launch_bounds__(kBlkThreads) void batch_rms_norm_kernel(
    const float* __restrict__ x, const float* __restrict__ w32, const uint16_t* __restrict__ wb,
    float* __restrict__ out, uint32_t n, uint32_t nb, float eps) {
  __shared__ float red[kBlkStripe * kBlkTok];
  const uint32_t bl = threadIdx.x & (kBlkTok - 1), s = threadIdx.x / kBlkTok;
  const uint32_t b = blockIdx.x * kBlkTok + bl;
  const bool live = b < nb;
  float acc = 0.0f;
  if (live)
    for (uint32_t i = s; i < n; i += kBlkStripe) {
      const float v = x[(size_t)i * nb + b];
      acc += v * v;
    }
  const float inv = rsqrtf(stripe_sum(red, s, bl, acc) / (float)n + eps);
  if (!live) return;
  for (uint32_t i = s; i < n; i += kBlkStripe) {
    const float w = w32 ? w32[i] : (wb ? __uint_as_float((uint32_t)wb[i] << 16) : 1.0f);
    out[(size_t)i * nb + b] = x[(size_t)i * nb + b] * inv * w;
  }
}

// The same RMS norm at a speculative block's width, plus the E4M3 quantisation its consumer wants.
//
// The kernel above puts kBlkTok tokens in a block and stripes `dim` across kBlkStripe threads, so
// at nb=16 it is TWO workgroups — the pathology hcn_* was written to fix. Here a block owns ONE
// token and all of `n`, which is 16 blocks, and because a block then holds every dim of its token
// it can compute the per-128-column maxima itself and emit the fp8 as well. That deletes the
// batch_quant_fp8_hip launch that always followed.
//
// BIT-IDENTICAL: the sum keeps the same kBlkStripe stripes accumulated in the same index order and
// summed k = 0..63 in order, exactly as stripe_sum does it, and the normalise writes x*inv*w
// whichever thread gets there. The quantise mirrors batch_quant_fp8_kernel expression for
// expression — `1/s` once and MULTIPLIED, the same clamp, the same intrinsic.
constexpr uint32_t kRmsQThr = 256;
constexpr uint32_t kRmsQWinMax = 64;                  // 128-column scale blocks, so n <= 8192

__global__ __launch_bounds__(kRmsQThr) void batch_rms_norm_q8_kernel(
    const float* __restrict__ x, const float* __restrict__ w32, const uint16_t* __restrict__ wb,
    float* __restrict__ out, uint8_t* __restrict__ q8, float* __restrict__ qs, uint32_t n,
    uint32_t nb, float eps) {
  __shared__ float red[kBlkStripe];
  __shared__ float s_inv;
  __shared__ float smx[kRmsQWinMax];
  const uint32_t tid = threadIdx.x, b = blockIdx.x;
  if (b >= nb) return;
  const uint32_t nkb = n >> 7;
  if (tid < kBlkStripe) {
    float acc = 0.0f;
    for (uint32_t i = tid; i < n; i += kBlkStripe) {
      const float v = x[(size_t)i * nb + b];
      acc += v * v;
    }
    red[tid] = acc;
  }
  if (q8 && tid < nkb) smx[tid] = 0.0f;
  __syncthreads();
  if (!tid) {
    float t = 0.0f;
    for (uint32_t k = 0; k < kBlkStripe; ++k) t += red[k];
    s_inv = rsqrtf(t / (float)n + eps);
  }
  __syncthreads();
  const float inv = s_inv;
  for (uint32_t i = tid; i < n; i += kRmsQThr) {
    const float w = w32 ? w32[i] : (wb ? __uint_as_float((uint32_t)wb[i] << 16) : 1.0f);
    const float v = x[(size_t)i * nb + b] * inv * w;
    out[(size_t)i * nb + b] = v;
    if (q8) atomicMax((unsigned int*)&smx[i >> 7], __float_as_uint(fabsf(v)));
  }
  if (!q8) return;
  __syncthreads();
  if (tid < nkb) {
    const float m = smx[tid];
    qs[(size_t)b * nkb + tid] = m > 0.0f ? m * (1.0f / 448.0f) : 1.0f;
  }
  __syncthreads();
  for (uint32_t i = tid; i < n; i += kRmsQThr) {
    const float m = smx[i >> 7];
    const float sc = m > 0.0f ? m * (1.0f / 448.0f) : 1.0f;
    const float t = fminf(fmaxf(out[(size_t)i * nb + b] * (1.0f / sc), -448.0f), 448.0f);
    int pk = 0;
    pk = __builtin_amdgcn_cvt_pk_fp8_f32(t, 0.0f, pk, false);
    q8[(size_t)b * n + i] = (uint8_t)((uint32_t)pk & 0xffu);
  }
}

__global__ __launch_bounds__(kBlkThreads) void batch_rms_norm_rope_kernel(
    const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out, uint32_t n,
    uint32_t nb, float eps, uint32_t n_rot, uint32_t pos0, float freq_scale, float theta_scale,
    float ext_factor, float lo, float hi) {
  __shared__ float red[kBlkStripe * kBlkTok];
  const uint32_t bl = threadIdx.x & (kBlkTok - 1), s = threadIdx.x / kBlkTok;
  const uint32_t b = blockIdx.x * kBlkTok + bl;
  const bool live = b < nb;
  float acc = 0.0f;
  if (live)
    for (uint32_t i = s; i < n; i += kBlkStripe) {
      const float v = x[(size_t)i * nb + b];
      acc += v * v;
    }
  const float inv = rsqrtf(stripe_sum(red, s, bl, acc) / (float)n + eps);
  if (live)
    for (uint32_t i = s; i < n; i += kBlkStripe)
      out[(size_t)i * nb + b] = x[(size_t)i * nb + b] * inv * (w ? w[i] : 1.0f);
  // Reached by every thread: the rotation below reads pairs another stripe just wrote, and a
  // barrier some lanes skip is undefined rather than merely wrong.
  __syncthreads();
  if (!live) return;
  float* p = out + (size_t)(n - n_rot) * nb + b;
  for (uint32_t k = s; k < n_rot / 2; k += kBlkStripe)
    rope_pair_strided(p, k, nb, pos0 + b, freq_scale, theta_scale, ext_factor, lo, hi, 1.0f);
}


// The same on a bf16 buffer. Rotating in f32 and rounding once is what the f32 form does too — it
// reads two floats that were themselves rounded on the way in — so the only difference is where the
// bytes live.
__global__ void batch_rope_tail_bf16_kernel(uint16_t* __restrict__ x, uint32_t n_head,
                                            uint32_t head_dim, uint32_t nb, uint32_t n_rot,
                                            uint32_t pos0, float freq_scale, float theta_scale,
                                            float ext_factor, float lo, float hi, float sin_sign) {
  const uint32_t pairs = n_rot / 2;
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t b = t % nb, rest = t / nb;
  if (rest >= n_head * pairs) return;
  const uint32_t h = rest / pairs, k = rest % pairs;
  uint16_t* p = x + (size_t)(h * head_dim + head_dim - n_rot) * nb + b;
  float c, sn;
  rope_cs(k, pos0 + b, freq_scale, theta_scale, ext_factor, lo, hi, sin_sign, &c, &sn);
  const size_t i0 = (size_t)(2 * k) * nb, i1 = (size_t)(2 * k + 1) * nb;
  const float x0 = wmma::bf16_to_f32(p[i0]), x1 = wmma::bf16_to_f32(p[i1]);
  p[i0] = wmma::f32_to_bf16(x0 * c - x1 * sn);
  p[i1] = wmma::f32_to_bf16(x0 * sn + x1 * c);
}

// gate is clamped on ONE side, up on BOTH — see engine/ops.cpp, this asymmetry is from the model.
__global__ void batch_swiglu_kernel(const float* __restrict__ g, const float* __restrict__ u,
                                    float* __restrict__ out, uint64_t n, float limit) {
  const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float gv = g[i], uv = u[i];
  if (limit > 0.0f) { gv = fminf(gv, limit); uv = fminf(limit, fmaxf(-limit, uv)); }
  out[i] = (gv / (1.0f + expf(-gv))) * uv;
}

// 64x64 LDS tile, padded to 65 so a column read hits 32 distinct banks twice rather than one bank
// 64 times. Both directions are coalesced, which the naive form is not in one of them.
//
// 64, not the 32 this used to be, and the reason is a line and not a preference: a transpose reads
// 32 (or 64) consecutive floats on one side and writes them on the other, and at 32 that is 128 B
// — HALF a line on both. The two halves of a destination line are then written by blocks one grid
// row apart, and whether that lands in cache depends on how many blocks the grid row holds. It
// showed up as an asymmetry the trace handed over: going from chunk 1536 to 3840, the query
// transpose (which reads along tokens) nearly doubled while this one, transposing the other way,
// was flat. 64 makes both sides a whole line and removes the dependence.
//
// 64x8 threads walking eight rows each rather than a 4096-thread square.
constexpr uint32_t kTrRows = 8, kTrTile = 64;
// `dstb` instead of `dst` when non-null: the heads buffer the attention output goes through is
// bf16, and rounding on the way out of the tile costs nothing the transpose was not already paying.
__global__ __launch_bounds__(kTrTile * kTrRows) void batch_transpose_kernel(
    const float* __restrict__ src, float* __restrict__ dst, uint16_t* __restrict__ dstb,
    uint32_t n, uint32_t nb, int to_token_major, float scale) {
  __shared__ float tile[kTrTile][kTrTile + 1];
  const uint32_t bx = blockIdx.x * kTrTile, by = blockIdx.y * kTrTile;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  // to_token_major: src is [n][nb], dst is [nb][n]. Otherwise the reverse.
  const uint32_t srow = to_token_major ? n : nb, scol = to_token_major ? nb : n;
  for (uint32_t k = 0; k < kTrTile; k += kTrRows)
    if (bx + tx < scol && by + ty + k < srow)
      tile[ty + k][tx] = src[(size_t)(by + ty + k) * scol + bx + tx];
  __syncthreads();
  for (uint32_t k = 0; k < kTrTile; k += kTrRows)
    if (by + tx < srow && bx + ty + k < scol) {
      const float v = tile[tx][ty + k] * scale;
      if (dstb) dstb[(size_t)(bx + ty + k) * srow + by + tx] = wmma::f32_to_bf16(v);
      else      dst[(size_t)(bx + ty + k) * srow + by + tx] = v;
    }
}

// Up to four dim-major -> token-major transposes in ONE launch, one per blockIdx.z. Same tile,
// same arithmetic; the compressor issues four of these back to back after its (already fused) GEMM
// and each was a dispatch of its own. The jobs differ only in `src`, `dst` and `n`, so the grid
// covers the widest and the bounds tests mask the rest.
constexpr uint32_t kTrJobsMax = 4;
struct TrJobs {
  const float* src[kTrJobsMax];
  float* dst[kTrJobsMax];
  uint32_t n[kTrJobsMax];
};

__global__ __launch_bounds__(kTrTile * kTrRows) void batch_transpose_multi_kernel(TrJobs J,
                                                                                 uint32_t nb) {
  __shared__ float tile[kTrTile][kTrTile + 1];
  const uint32_t j = blockIdx.z;
  const float* __restrict__ src = J.src[j];
  float* __restrict__ dst = J.dst[j];
  const uint32_t srow = J.n[j], scol = nb;          // to_token_major: src is [n][nb]
  const uint32_t bx = blockIdx.x * kTrTile, by = blockIdx.y * kTrTile;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  for (uint32_t k = 0; k < kTrTile; k += kTrRows)
    if (bx + tx < scol && by + ty + k < srow)
      tile[ty + k][tx] = src[(size_t)(by + ty + k) * scol + bx + tx];
  __syncthreads();
  for (uint32_t k = 0; k < kTrTile; k += kTrRows)
    if (by + tx < srow && bx + ty + k < scol)
      dst[(size_t)(bx + ty + k) * srow + by + tx] = tile[tx][ty + k];
}

// ---- the same thing in two passes instead of three, because the transpose was already there -----
//
// What this replaced walked the query buffer THREE times — once to accumulate the per-head sum of
// squares, once to scale in place, once more for the rope tail — and the transpose that follows it
// walks it twice more. At a prefill chunk that made it one of the largest device items in the phase
// with no GEMM anywhere in it.
//
// Only the first pass genuinely needs its own launch: a head's RMS scale is not known until all of
// its dims have been read. Everything after it is elementwise, and the transpose has to touch every
// element anyway — so the scale and the rope ride on the transpose. The one below produces the scale
// and nothing else.
__global__ void batch_head_rms_inv_kernel(const float* __restrict__ x, float* __restrict__ inv,
                                          uint32_t n_head, uint32_t head_dim, uint32_t nb,
                                          float eps) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t b = t % nb, h = t / nb;
  if (h >= n_head) return;
  const float* p = x + (size_t)h * head_dim * nb + b;
  float acc = 0.0f;
  // Same order as the kernel it replaces, so the sum is the same float.
  for (uint32_t i = 0; i < head_dim; ++i) {
    const float v = p[(size_t)i * nb];
    acc += v * v;
  }
  inv[(size_t)h * nb + b] = rsqrtf(acc / (float)head_dim + eps);
}

// The transpose, with the head scale folded into the load and the rope done in LDS.
//
// 64x64, not the 32x32 the generic transpose uses. This direction reads along tokens (row stride nb)
// and writes along dims (row stride n), so BOTH sides need 64 consecutive floats to fill a line. At
// 32 the two halves of each destination line are written by blocks nb/32 apart in dispatch order,
// which at a wide chunk is far enough that the line is evicted in between and the write becomes
// read-modify-write. The tell is that this kernel degrades with chunk width while the same tile
// transposing the OTHER way stays flat.
//
// Two alignment facts let the rope ride along, and the launcher checks both rather than assuming:
//
//  * `rope_pair_strided` rotates dims 2k and 2k+1 — ADJACENT, not k and k+n_rot/2 — so a pair never
//    crosses a row boundary of any even-height tile. This is what makes an LDS tile work at all;
//    the half-and-half convention would need the tile to be n_rot tall.
//  * head_dim % 64 == 0 and n_rot % 64 == 0, so a tile lies inside ONE head (the scale is uniform
//    over it) and is either entirely inside that head's rope tail or entirely outside it.
//
// Values go through LDS as f32 and are scaled before the rope reads them, which is exactly the
// order the three-pass version had — it stored the scaled value to global and the rope loaded it
// back. So this is bit-identical, not merely equivalent.
constexpr uint32_t kQTr = 64, kQTrRows = 8;
__global__ __launch_bounds__(kQTr * kQTrRows) void batch_head_scale_rope_tr_kernel(
    const float* __restrict__ src, uint16_t* __restrict__ dst, const float* __restrict__ inv,
    uint32_t n, uint32_t nb, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, float freq_scale,
    float theta_scale, float ext_factor, float lo, float hi) {
  __shared__ float tile[kQTr][kQTr + 1];
  const uint32_t bx = blockIdx.x * kQTr, by = blockIdx.y * kQTr;   // bx over tokens, by over dims
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t b = bx + tx;
  const float sc = (b < nb) ? inv[(size_t)(by / head_dim) * nb + b] : 0.0f;
  for (uint32_t k = 0; k < kQTr; k += kQTrRows)
    if (b < nb && by + ty + k < n)
      tile[ty + k][tx] = src[(size_t)(by + ty + k) * nb + b] * sc;
  __syncthreads();
  const uint32_t dh = by % head_dim;
  if (dh >= head_dim - n_rot && b < nb) {
    const uint32_t k0 = (dh - (head_dim - n_rot)) / 2;         // this tile's first pair index
    for (uint32_t p = ty; p < kQTr / 2; p += kQTrRows) {
      float c, sn;
      rope_cs(k0 + p, pos0 + b, freq_scale, theta_scale, ext_factor, lo, hi, 1.0f, &c, &sn);
      const float x0 = tile[2 * p][tx], x1 = tile[2 * p + 1][tx];
      tile[2 * p][tx]     = x0 * c - x1 * sn;
      tile[2 * p + 1][tx] = x0 * sn + x1 * c;
    }
  }
  __syncthreads();
  for (uint32_t k = 0; k < kQTr; k += kQTrRows)
    if (bx + ty + k < nb && by + tx < n)
      dst[(size_t)(bx + ty + k) * n + by + tx] = wmma::f32_to_bf16(tile[tx][ty + k]);
}

// ---- hyper-connections ------------------------------------------------------------------------
//
// The same arithmetic as hc_fused_kernel, restriped. That kernel is one block for the whole model
// and reduces 16384 elements across 1024 threads; here a block owns 32 tokens and each thread walks
// its own token's stripe, so a wave still reads 32 consecutive floats and the Sinkhorn — 20 sweeps
// over a 4x4 matrix — is done once per token by one thread instead of once per model by thread 0.
__device__ inline void hc_gates_lane(const float* __restrict__ mix, uint32_t nb, uint32_t b,
                                     float mix_scale, const float* __restrict__ scale,
                                     const float* __restrict__ base, float* pre, float* post,
                                     float* comb, uint32_t n_hc, uint32_t iters, float hc_eps) {
  for (uint32_t i = 0; i < n_hc; ++i) {
    const float z = mix[(size_t)i * nb + b] * mix_scale * scale[0] + base[i];
    pre[i] = 1.0f / (1.0f + __expf(-z)) + hc_eps;
  }
  for (uint32_t i = 0; i < n_hc; ++i) {
    const uint32_t o = n_hc + i;
    const float z = mix[(size_t)o * nb + b] * mix_scale * scale[1] + base[o];
    post[i] = 2.0f / (1.0f + __expf(-z));
  }
  // Softmax over src, then Sinkhorn. The first pass is NOT the loop body: it normalises rows via
  // the softmax and then columns once. Running a column sweep first converges somewhere else.
  for (uint32_t dst = 0; dst < n_hc; ++dst) {
    float rmax = -1e30f;
    for (uint32_t src = 0; src < n_hc; ++src) {
      const uint32_t idx = src + dst * n_hc;
      comb[idx] = mix[(size_t)(2 * n_hc + idx) * nb + b] * mix_scale * scale[2] + base[2 * n_hc + idx];
      rmax = fmaxf(rmax, comb[idx]);
    }
    float rsum = 0.0f;
    for (uint32_t src = 0; src < n_hc; ++src) {
      const uint32_t idx = src + dst * n_hc;
      comb[idx] = __expf(comb[idx] - rmax); rsum += comb[idx];
    }
    for (uint32_t src = 0; src < n_hc; ++src)
      comb[src + dst * n_hc] = comb[src + dst * n_hc] / rsum + hc_eps;
  }
  for (uint32_t src = 0; src < n_hc; ++src) {
    float sum = 0.0f;
    for (uint32_t dst = 0; dst < n_hc; ++dst) sum += comb[src + dst * n_hc];
    const float inv = 1.0f / (sum + hc_eps);
    for (uint32_t dst = 0; dst < n_hc; ++dst) comb[src + dst * n_hc] *= inv;
  }
  for (uint32_t it = 1; it < iters; ++it) {
    for (uint32_t dst = 0; dst < n_hc; ++dst) {
      float sum = 0.0f;
      for (uint32_t src = 0; src < n_hc; ++src) sum += comb[src + dst * n_hc];
      const float inv = 1.0f / (sum + hc_eps);
      for (uint32_t src = 0; src < n_hc; ++src) comb[src + dst * n_hc] *= inv;
    }
    for (uint32_t src = 0; src < n_hc; ++src) {
      float sum = 0.0f;
      for (uint32_t dst = 0; dst < n_hc; ++dst) sum += comb[src + dst * n_hc];
      const float inv = 1.0f / (sum + hc_eps);
      for (uint32_t dst = 0; dst < n_hc; ++dst) comb[src + dst * n_hc] *= inv;
    }
  }
}

__global__ __launch_bounds__(kBlkThreads) void batch_hc_fused_kernel(
    const uint16_t* __restrict__ hc, const float* __restrict__ mix, const float* __restrict__ scale,
    const float* __restrict__ base, const float* __restrict__ nw, float* __restrict__ pre_out,
    float* __restrict__ post_out, float* __restrict__ comb_out, float* __restrict__ cur,
    float* __restrict__ norm, uint32_t n_embd, uint32_t n_hc, uint32_t nb, uint32_t iters,
    float hc_eps, float rms_eps, int has_control, const float* __restrict__ pre_in) {
  __shared__ float red[kBlkStripe * kBlkTok];
  __shared__ float s_pre[kBlkTok * 8], s_post[kBlkTok * 8], s_comb[kBlkTok * 64];
  __shared__ float s_scale[kBlkTok];
  const uint32_t bl = threadIdx.x & (kBlkTok - 1), s = threadIdx.x / kBlkTok;
  const uint32_t b = blockIdx.x * kBlkTok + bl;
  const bool live = b < nb;
  const uint64_t hc_dim = (uint64_t)n_embd * n_hc;

  if (has_control) {
    float acc = 0.0f;
    if (live)
      for (uint64_t i = s; i < hc_dim; i += kBlkStripe) {
        const float v = wmma::bf16_to_f32(hc[i * nb + b]);
        acc += v * v;
      }
    const float tot = stripe_sum(red, s, bl, acc);
    if (s == 0) s_scale[bl] = rsqrtf(tot / (float)hc_dim + rms_eps);
    __syncthreads();
    // One thread per token does the whole 4x4 Sinkhorn; the matvec that produced `mix` already ran
    // as a batched GEMM, and the RMS scale it should have been applied to is folded in here by
    // linearity, exactly as in the single-token kernel.
    if (s == 0 && live)
      hc_gates_lane(mix, nb, b, s_scale[bl], scale, base, s_pre + bl * 8, s_post + bl * 8,
                    s_comb + bl * 64, n_hc, iters, hc_eps);
    __syncthreads();
    if (s == 0 && live) {
      for (uint32_t h = 0; h < n_hc; ++h) {
        pre_out[(size_t)h * nb + b]  = s_pre[bl * 8 + h];
        post_out[(size_t)h * nb + b] = s_post[bl * 8 + h];
      }
      for (uint32_t i = 0; i < n_hc * n_hc; ++i) comb_out[(size_t)i * nb + b] = s_comb[bl * 64 + i];
      // V4.1 collapses with the mix the PREVIOUS sublayer computed. The one computed here has just
      // been published for the next one; `s_pre` now becomes what this collapse actually uses.
      if (pre_in)
        for (uint32_t h = 0; h < n_hc; ++h) s_pre[bl * 8 + h] = pre_in[(size_t)h * nb + b];
    }
    __syncthreads();
  }

  float sq = 0.0f;
  if (live)
    for (uint32_t d = s; d < n_embd; d += kBlkStripe) {
      float acc = 0.0f;
      if (has_control)
        for (uint32_t h = 0; h < n_hc; ++h)
          acc += wmma::bf16_to_f32(hc[((size_t)h * n_embd + d) * nb + b]) * s_pre[bl * 8 + h];
      else
        acc = wmma::bf16_to_f32(hc[(size_t)d * nb + b]);
      cur[(size_t)d * nb + b] = acc;
      sq += acc * acc;
    }
  const float inv = rsqrtf(stripe_sum(red, s, bl, sq) / (float)n_embd + rms_eps);
  if (!live) return;
  for (uint32_t d = s; d < n_embd; d += kBlkStripe)
    norm[(size_t)d * nb + b] = cur[(size_t)d * nb + b] * inv * (nw ? nw[d] : 1.0f);
}

// ---- the same thing for a speculative block ----------------------------------------------------
//
// The kernel above parallelises over TOKENS, which is tens of thousands of threads at a prefill
// chunk and a couple of workgroups at a speculative block. At that width it reads a fraction of a
// percent of the card's bandwidth and is one of the largest kernels in a verify block.
//
// So parallelise over `n_embd` instead, which is constant whatever the block width. A thread owns one
// dim index and ALL NB tokens: the row it loads is NB bf16 = one or two 16-byte granules, adjacent
// threads load adjacent rows, and the per-token reduction that used to be a serial walk inside a
// lane becomes NB accumulators in registers plus one LDS tree.
//
// NB is a template argument, not `nb`. Every plane in the speculative arena is kNarrowTok columns
// wide by construction, so the trip count is a compile-time constant here and the accumulators can
// live in registers — a runtime NB would spill them to scratch and unroll nothing.
//
// THE REDUCTION ORDER IS NOT THE ONE ABOVE, so this is not bit-identical to prefill's kernel: 64
// serial stripes summed in index order becomes a pairwise tree over `kHcnThreads`. Same terms, and
// a tree is the better-conditioned of the two, but the last bits move. Prefill still runs the
// kernel above and its fingerprint must not move.
// hc_gates_lane with n_hc as a template argument. Same terms in the same order — the runtime form
// above is the reference and this is a transcription of it — but every trip count is a constant, so
// `pre`, `post` and `comb` are indexed by unrolled constants and stay in registers.
template <uint32_t NH>
__device__ __forceinline__ void hc_gates_lane_nh(const float* __restrict__ mix, uint32_t nb,
                                                 uint32_t b, float mix_scale,
                                                 const float* __restrict__ scale,
                                                 const float* __restrict__ base, float* pre,
                                                 float* post, float* comb, uint32_t iters,
                                                 float hc_eps) {
#pragma unroll
  for (uint32_t i = 0; i < NH; ++i) {
    const float z = mix[(size_t)i * nb + b] * mix_scale * scale[0] + base[i];
    pre[i] = 1.0f / (1.0f + __expf(-z)) + hc_eps;
  }
#pragma unroll
  for (uint32_t i = 0; i < NH; ++i) {
    const uint32_t o = NH + i;
    const float z = mix[(size_t)o * nb + b] * mix_scale * scale[1] + base[o];
    post[i] = 2.0f / (1.0f + __expf(-z));
  }
#pragma unroll
  for (uint32_t dst = 0; dst < NH; ++dst) {
    float rmax = -1e30f;
#pragma unroll
    for (uint32_t src = 0; src < NH; ++src) {
      const uint32_t idx = src + dst * NH;
      comb[idx] = mix[(size_t)(2 * NH + idx) * nb + b] * mix_scale * scale[2] + base[2 * NH + idx];
      rmax = fmaxf(rmax, comb[idx]);
    }
    float rsum = 0.0f;
#pragma unroll
    for (uint32_t src = 0; src < NH; ++src) {
      const uint32_t idx = src + dst * NH;
      comb[idx] = __expf(comb[idx] - rmax); rsum += comb[idx];
    }
#pragma unroll
    for (uint32_t src = 0; src < NH; ++src)
      comb[src + dst * NH] = comb[src + dst * NH] / rsum + hc_eps;
  }
#pragma unroll
  for (uint32_t src = 0; src < NH; ++src) {
    float sum = 0.0f;
#pragma unroll
    for (uint32_t dst = 0; dst < NH; ++dst) sum += comb[src + dst * NH];
    const float inv = 1.0f / (sum + hc_eps);
#pragma unroll
    for (uint32_t dst = 0; dst < NH; ++dst) comb[src + dst * NH] *= inv;
  }
  for (uint32_t it = 1; it < iters; ++it) {
#pragma unroll
    for (uint32_t dst = 0; dst < NH; ++dst) {
      float sum = 0.0f;
#pragma unroll
      for (uint32_t src = 0; src < NH; ++src) sum += comb[src + dst * NH];
      const float inv = 1.0f / (sum + hc_eps);
#pragma unroll
      for (uint32_t src = 0; src < NH; ++src) comb[src + dst * NH] *= inv;
    }
#pragma unroll
    for (uint32_t src = 0; src < NH; ++src) {
      float sum = 0.0f;
#pragma unroll
      for (uint32_t dst = 0; dst < NH; ++dst) sum += comb[src + dst * NH];
      const float inv = 1.0f / (sum + hc_eps);
#pragma unroll
      for (uint32_t dst = 0; dst < NH; ++dst) comb[src + dst * NH] *= inv;
    }
  }
}

// [NB][kHcnThreads] rather than [kHcnThreads][NB]: consecutive threads then land in consecutive
// banks. The transposed form strides by NB=16 floats and conflicts 16 ways on every step.
template <uint32_t NB>
__device__ __forceinline__ void hcn_block_sum(float* __restrict__ s, float* acc, uint32_t tid) {
#pragma unroll
  for (uint32_t b = 0; b < NB; ++b) s[b * kHcnThreads + tid] = acc[b];
  __syncthreads();
  for (uint32_t st = kHcnThreads / 2; st; st >>= 1) {
    if (tid < st)
#pragma unroll
      for (uint32_t b = 0; b < NB; ++b) s[b * kHcnThreads + tid] += s[b * kHcnThreads + tid + st];
    __syncthreads();
  }
}

// One dim-major row of NB bf16, as NB floats in registers. The cast to uint4 is what makes it a
// single global_load_b128 (two at NB=16) instead of NB separate ushort loads.
template <uint32_t NB>
__device__ __forceinline__ void hcn_row(const uint16_t* __restrict__ p, float* v) {
  static_assert(NB * sizeof(uint16_t) % 16 == 0, "a row must be a whole 16-byte granule");
  constexpr uint32_t kG = NB * sizeof(uint16_t) / 16;
#pragma unroll
  for (uint32_t g = 0; g < kG; ++g) {
    const uint4 r = ((const uint4*)p)[g];
    const uint32_t w[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      v[g * 8 + j * 2 + 0] = __uint_as_float(w[j] << 16);
      v[g * 8 + j * 2 + 1] = __uint_as_float(w[j] & 0xffff0000u);
    }
  }
}

// Sum of squares of the whole [n_embd * n_hc][NB] control plane, one partial row per block.
template <uint32_t NB>
__global__ __launch_bounds__(kHcnThreads) void hcn_sq_kernel(const uint16_t* __restrict__ hc,
                                                             float* __restrict__ part,
                                                             uint64_t hc_dim) {
  __shared__ float s[NB * kHcnThreads];
  const uint32_t tid = threadIdx.x;
  float acc[NB] = {};
  for (uint64_t i = (uint64_t)blockIdx.x * kHcnThreads + tid; i < hc_dim;
       i += (uint64_t)kHcnChunks * kHcnThreads) {
    float v[NB];
    hcn_row<NB>(hc + i * NB, v);
#pragma unroll
    for (uint32_t b = 0; b < NB; ++b) acc[b] += v[b] * v[b];
  }
  hcn_block_sum<NB>(s, acc, tid);
  if (tid < NB) part[(size_t)blockIdx.x * NB + tid] = s[tid * kHcnThreads];
}

// The 4x4 Sinkhorn, one thread per token. Unchanged arithmetic — it is `hc_gates_lane`, the same
// function prefill's kernel calls — but it needs its own launch because the scale it folds in is a
// reduction over a grid now instead of over a block.

// cur = the pre-gated lane sum, and the partial sums of squares the norm below needs. NH is a
// template argument for the same reason it is one in batch_hc_expand_reg_kernel: at a runtime trip
// count the four loads do not unroll and `pre` is re-read for every one of them.
template <uint32_t NB, uint32_t NH>
__global__ __launch_bounds__(kHcnThreads) void hcn_expand_kernel(
    const uint16_t* __restrict__ hc, const float* __restrict__ part_in,
    const float* __restrict__ mix, const float* __restrict__ scale,
    const float* __restrict__ base, float* __restrict__ pre_out, float* __restrict__ post_out,
    float* __restrict__ comb_out, float* __restrict__ cur, float* __restrict__ part,
    uint32_t n_embd, uint64_t hc_dim, uint32_t iters, float hc_eps, float rms_eps,
    const float* __restrict__ pre_in) {
  __shared__ float s[NB * kHcnThreads];
  // The gates, computed HERE and redundantly in every block rather than in a launch of their own.
  // The Sinkhorn is a long dependent chain in NB threads whatever runs it, so doing it in every
  // block costs the same latency as doing it once — and as its own launch it is a dispatch gap and a
  // kernel ramp for a fraction of the hc work.
  __shared__ float s_pre[NB * NH];
  const uint32_t tid = threadIdx.x;
  if (tid < NB) {
    float tot = 0.0f;
    for (uint32_t c = 0; c < kHcnChunks; ++c) tot += part_in[(size_t)c * NB + tid];
    float pre[NH], post[NH], comb[NH * NH];
    hc_gates_lane_nh<NH>(mix, NB, tid, rsqrtf(tot / (float)hc_dim + rms_eps), scale, base, pre,
                         post, comb, iters, hc_eps);
#pragma unroll
    for (uint32_t h = 0; h < NH; ++h) s_pre[h * NB + tid] = pre[h];
    // One block writes the planes the expand kernel downstream reads; the rest only need s_pre.
    if (blockIdx.x == 0) {
#pragma unroll
      for (uint32_t h = 0; h < NH; ++h) {
        pre_out[(size_t)h * NB + tid] = pre[h];
        post_out[(size_t)h * NB + tid] = post[h];
      }
#pragma unroll
      for (uint32_t i = 0; i < NH * NH; ++i) comb_out[(size_t)i * NB + tid] = comb[i];
    }
    // See the wide kernel: on V4.1 the collapse uses the previous sublayer's mix, and the one just
    // computed goes forward through pre_out.
    if (pre_in)
#pragma unroll
      for (uint32_t h = 0; h < NH; ++h) s_pre[h * NB + tid] = pre_in[(size_t)h * NB + tid];
  }
  __syncthreads();
  float g[NH][NB];
#pragma unroll
  for (uint32_t h = 0; h < NH; ++h)
#pragma unroll
    for (uint32_t b = 0; b < NB; ++b) g[h][b] = s_pre[h * NB + b];
  float acc[NB] = {};
  for (uint32_t d = blockIdx.x * kHcnThreads + tid; d < n_embd; d += kHcnChunks * kHcnThreads) {
    float o[NB] = {};
#pragma unroll
    for (uint32_t h = 0; h < NH; ++h) {
      float v[NB];
      hcn_row<NB>(hc + ((size_t)h * n_embd + d) * NB, v);
#pragma unroll
      for (uint32_t b = 0; b < NB; ++b) o[b] += v[b] * g[h][b];
    }
#pragma unroll
    for (uint32_t b = 0; b < NB; ++b) {
      cur[(size_t)d * NB + b] = o[b];
      acc[b] += o[b] * o[b];
    }
  }
  hcn_block_sum<NB>(s, acc, tid);
  if (tid < NB) part[(size_t)blockIdx.x * NB + tid] = s[tid * kHcnThreads];
}

// The HEAD's collapse, which is hcn_expand_kernel with three quarters of the gate deleted.
//
// A sublayer's hyper-connection needs `pre` to weight the lanes going in, `post` to weight what
// comes out, and a 4x4 Sinkhorn to say how the lanes mix. The head needs none of the last two: it is
// the end of the stream, so the four lanes collapse into one vector and nothing is expanded again.
// So `mix` here is [n_hc][nb] out of a [n_hc, n_hc*n_embd] matrix rather than [2*n_hc + n_hc^2], and
// the gate is one sigmoid — the first loop of hc_gates_lane_nh, character for character.
//
// This is the arithmetic the host was running per token behind a device-to-host copy of the whole
// [n_hc][n_embd] control plane. Same terms; NOT the same order — the sum of squares is a tree here
// and was a serial walk there, and the mix scale is applied to the four dots instead of to the
// 16384 inputs of them. It is the DRAFT's head, and a draft's arithmetic cannot reach the output:
// greedy speculation emits the target's argmax whatever the draft proposed. Only acceptance moves.
template <uint32_t NB, uint32_t NH>
__global__ __launch_bounds__(kHcnThreads) void hcn_head_gate_kernel(
    const uint16_t* __restrict__ hc, const float* __restrict__ part_in,
    const float* __restrict__ mix, const float* __restrict__ scale,
    const float* __restrict__ base, float* __restrict__ cur, float* __restrict__ part,
    uint32_t n_embd, uint64_t hc_dim, float hc_eps, float rms_eps,
    const float* __restrict__ pre_in) {
  __shared__ float s[NB * kHcnThreads];
  __shared__ float s_pre[NB * NH];
  const uint32_t tid = threadIdx.x;
  // The gates in every block rather than in a launch of their own, for the reason on the kernel
  // above: NH sigmoids in NB threads is latency whatever runs it, and a launch here is ~2 us.
  if (tid < NB) {
    // V4.1's head has no mixer of its own: `pre_in` is the mix the last layer's FFN already
    // produced, so the gates are read rather than recomputed and `mix`/`scale`/`base`/`part_in`
    // are not touched at all.
    if (pre_in) {
#pragma unroll
      for (uint32_t h = 0; h < NH; ++h) s_pre[h * NB + tid] = pre_in[(size_t)h * NB + tid];
    } else {
      float tot = 0.0f;
      for (uint32_t c = 0; c < kHcnChunks; ++c) tot += part_in[(size_t)c * NB + tid];
      const float mix_scale = rsqrtf(tot / (float)hc_dim + rms_eps);
#pragma unroll
      for (uint32_t h = 0; h < NH; ++h) {
        const float z = mix[(size_t)h * NB + tid] * mix_scale * scale[0] + base[h];
        s_pre[h * NB + tid] = 1.0f / (1.0f + __expf(-z)) + hc_eps;
      }
    }
  }
  __syncthreads();
  float g[NH][NB];
#pragma unroll
  for (uint32_t h = 0; h < NH; ++h)
#pragma unroll
    for (uint32_t b = 0; b < NB; ++b) g[h][b] = s_pre[h * NB + b];
  float acc[NB] = {};
  for (uint32_t d = blockIdx.x * kHcnThreads + tid; d < n_embd; d += kHcnChunks * kHcnThreads) {
    float o[NB] = {};
#pragma unroll
    for (uint32_t h = 0; h < NH; ++h) {
      float v[NB];
      hcn_row<NB>(hc + ((size_t)h * n_embd + d) * NB, v);
#pragma unroll
      for (uint32_t b = 0; b < NB; ++b) o[b] += v[b] * g[h][b];
    }
#pragma unroll
    for (uint32_t b = 0; b < NB; ++b) {
      cur[(size_t)d * NB + b] = o[b];
      acc[b] += o[b] * o[b];
    }
  }
  hcn_block_sum<NB>(s, acc, tid);
  if (tid < NB) part[(size_t)blockIdx.x * NB + tid] = s[tid * kHcnThreads];
}

// The has_control == 0 form: `cur` is lane 0 of `hc` and there is nothing to gate.
template <uint32_t NB>
__global__ __launch_bounds__(kHcnThreads) void hcn_copy_kernel(const uint16_t* __restrict__ hc,
                                                               float* __restrict__ cur,
                                                               float* __restrict__ part,
                                                               uint32_t n_embd) {
  __shared__ float s[NB * kHcnThreads];
  const uint32_t tid = threadIdx.x;
  float acc[NB] = {};
  for (uint32_t d = blockIdx.x * kHcnThreads + tid; d < n_embd; d += kHcnChunks * kHcnThreads) {
    float v[NB];
    hcn_row<NB>(hc + (size_t)d * NB, v);
#pragma unroll
    for (uint32_t b = 0; b < NB; ++b) {
      cur[(size_t)d * NB + b] = v[b];
      acc[b] += v[b] * v[b];
    }
  }
  hcn_block_sum<NB>(s, acc, tid);
  if (tid < NB) part[(size_t)blockIdx.x * NB + tid] = s[tid * kHcnThreads];
}

// The query's per-head RMS scale, with the grid on `head_dim` instead of on the token count.
//
// batch_head_rms_inv_kernel gives one thread each (head, token) pair and walks head_dim inside it.
// At a prefill chunk that is tens of thousands of threads; at a speculative block it is a couple of
// workgroups moving barely a megabyte. Here a thread owns one (head, dim) and all NB tokens — the NB
// floats at [h][i][*] are contiguous, so that is one granule — and a block owns one head, reducing
// over dim through the tree above.
//
// Not bit-identical: a serial walk over head_dim becomes pairwise. Prefill keeps the old kernel.
template <uint32_t NB>
__global__ __launch_bounds__(kHcnThreads) void hcn_head_rms_inv_kernel(
    const float* __restrict__ x, float* __restrict__ inv, uint32_t head_dim, float eps) {
  __shared__ float s[NB * kHcnThreads];
  const uint32_t tid = threadIdx.x, h = blockIdx.x;
  const float* p = x + (size_t)h * head_dim * NB;
  float acc[NB] = {};
  for (uint32_t i = tid; i < head_dim; i += kHcnThreads) {
    const float* q = p + (size_t)i * NB;
#pragma unroll
    for (uint32_t b = 0; b < NB; ++b) {
      const float v = q[b];
      acc[b] += v * v;
    }
  }
  hcn_block_sum<NB>(s, acc, tid);
  if (tid < NB) inv[(size_t)h * NB + tid] = rsqrtf(s[tid * kHcnThreads] / (float)head_dim + eps);
}

// Every block re-runs the kHcnChunks-term tail reduction rather than a fifth launch computing it
// once: a few hundred floats out of L2 is cheaper than a dispatch.
// ...and the E4M3 quantisation of what it just wrote, when the caller wants one.
//
// Every consumer of `norm` on the W8A8 path calls batch_quant_fp8_hip on it immediately, which is a
// launch that re-reads the whole plane to produce a per-(token, 128-column) scale and a token-major
// byte. This kernel already holds each dim's NB values in registers, so it can do both.
//
// Bit-identical, and by a stronger argument than usual: the scale is a MAX, which is exact and
// order-free, so reducing it across 128 threads instead of eight groups of sixteen cannot move it.
// Everything after the scale is per-element.
//
// The block owns 256 CONTIGUOUS dims for this — two whole 128-column scale blocks — where it used
// to grid-stride. The write stays coalesced: thread d writes out[tok*dim + d], so for one token the
// block lays down 256 contiguous bytes.
// `norm_tm` is the same values written TOKEN-major, for a consumer that wants `nb` contiguous
// vectors rather than a dim-major plane. matvec_multix_hip is the one: it reads each token's whole
// row and the head is a matvec, not a GEMM. The store pattern is the one q8 already uses — thread
// `d` writing [b][d] for each b — so it is coalesced across the block and costs a store, not a pass.
template <uint32_t NB>
__global__ __launch_bounds__(kHcnThreads) void hcn_norm_kernel(const float* __restrict__ cur,
                                                               const float* __restrict__ part,
                                                               const float* __restrict__ nw,
                                                               float* __restrict__ norm,
                                                               uint8_t* __restrict__ q8,
                                                               float* __restrict__ qs,
                                                               uint32_t n_embd, float rms_eps,
                                                               float* __restrict__ norm_tm) {
  __shared__ float s_inv[NB];
  __shared__ float s_max[2][NB];
  const uint32_t tid = threadIdx.x;
  if (tid < NB) {
    float tot = 0.0f;
    for (uint32_t c = 0; c < kHcnChunks; ++c) tot += part[(size_t)c * NB + tid];
    s_inv[tid] = rsqrtf(tot / (float)n_embd + rms_eps);
  }
  if (tid < 2 * NB) s_max[tid / NB][tid % NB] = 0.0f;
  __syncthreads();
  float inv[NB];
#pragma unroll
  for (uint32_t b = 0; b < NB; ++b) inv[b] = s_inv[b];

  const uint32_t d = blockIdx.x * kHcnThreads + tid;
  float v[NB];
  if (d < n_embd) {
    const float w = nw ? nw[d] : 1.0f;
#pragma unroll
    for (uint32_t b = 0; b < NB; ++b) {
      v[b] = cur[(size_t)d * NB + b] * inv[b] * w;
      norm[(size_t)d * NB + b] = v[b];
    }
    if (norm_tm)
#pragma unroll
      for (uint32_t b = 0; b < NB; ++b) norm_tm[(size_t)b * n_embd + d] = v[b];
  }
  if (!q8) return;
  const uint32_t win = tid >> 7;                   // which 128-column scale block of this pair
  if (d < n_embd)
#pragma unroll
    for (uint32_t b = 0; b < NB; ++b)
      atomicMax((unsigned int*)&s_max[win][b], __float_as_uint(fabsf(v[b])));
  __syncthreads();
  const uint32_t nkb = n_embd >> 7;
  const uint32_t kb = blockIdx.x * 2 + win;
  // Same expressions as batch_quant_fp8_kernel, to the letter. `1/s` computed once and MULTIPLIED,
  // not a divide: v * (1/s) and v / s differ in the last bit, and that bit reaches the E4M3 code.
  // Writing the divide instead moves the emitted text, and the acceptance rate with it — which
  // reads like the prompt lottery and is not, it is a different quantisation.
  float sc[NB];
#pragma unroll
  for (uint32_t b = 0; b < NB; ++b) {
    const float m = s_max[win][b];
    sc[b] = m > 0.0f ? m * (1.0f / 448.0f) : 1.0f;
  }
  if ((tid & 127u) < NB) qs[(size_t)(tid & 127u) * nkb + kb] = sc[tid & 127u];
  if (d >= n_embd) return;
#pragma unroll
  for (uint32_t b = 0; b < NB; ++b) {
    const float t = fminf(fmaxf(v[b] * (1.0f / sc[b]), -448.0f), 448.0f);
    int pk = 0;
    pk = __builtin_amdgcn_cvt_pk_fp8_f32(t, 0.0f, pk, false);
    q8[(size_t)b * n_embd + d] = (uint8_t)((uint32_t)pk & 0xffu);
  }
}

// The same kernel with `hc` read once instead of once per destination. At n_hc=4 the loop above
// loads the same four values sixteen times; the trip count is a runtime argument so neither loop
// unrolls and nothing gets hoisted. Same terms, same order, same `acc = bv * post[...]` seed, so it
// is BIT-IDENTICAL. `post` and `comb` are indexed by the batch lane here, so unlike the decode
// kernel they stay vector loads and only `hc` is redundant — but that is the one that scales with
// n_embd. See hc_expand_reg_kernel in dense_gpu.hip; this is the prefill half of the same fix.
// ---- AND IT LEAVES THE BLOCK ZEROED, which deletes a memset a layer a card -----------------------
//
// `ZB` says whether `block` is the sublayer's accumulator this call drains — in which case the next
// sublayer needs it zero and this kernel is already the one pass that visits every element of it —
// or a buffer somebody else owns. Exactly the distinction hc_add_kernel<ZB> makes on the decode
// side, for the same reason.
//
// The grid is `ceil(n_embd*nb / 256)` and the guard is `d >= n_embd`, so a surviving thread has
// `d*nb + b == t` with `t < n_embd*nb`: one thread per element of the arena, each visiting its own,
// which is precisely the property a memset of the same range has. The memset this replaced was
// justified by "no consumer's grid partitions the whole arena"; that was true of a kernel THIS one
// replaced, and the store is free here because the line is already in cache from the load above.
// 86 memsets a block a card at 256 KiB each, the largest single line in the blit census, gone.
// `zb` is a runtime argument and not a template one, unlike `NH`. NH has to be a template parameter
// because it is a trip count the compiler must see to unroll; `zb` is a block-uniform predicate over
// a single store, so an instantiation would buy nothing and the comma in a two-parameter template
// argument list does not survive the launch macro.
template <uint32_t NH>
__global__ void batch_hc_expand_reg_kernel(uint16_t* __restrict__ out_hc,
                                           float* __restrict__ block,
                                           const uint16_t* __restrict__ hc,
                                           const float* __restrict__ post,
                                           const float* __restrict__ comb, uint32_t n_embd,
                                           uint32_t nb, int zb) {
  const uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t b = (uint32_t)(t % nb), d = (uint32_t)(t / nb);
  if (d >= n_embd) return;
  const float bv = block[(size_t)d * nb + b];
  if (zb) block[(size_t)d * nb + b] = 0.0f;
  float v[NH];
#pragma unroll
  for (uint32_t src = 0; src < NH; ++src)
    v[src] = wmma::bf16_to_f32(hc[((size_t)src * n_embd + d) * nb + b]);
#pragma unroll
  for (uint32_t dst = 0; dst < NH; ++dst) {
    float acc = bv * post[(size_t)dst * nb + b];
#pragma unroll
    for (uint32_t src = 0; src < NH; ++src)
      acc += comb[(size_t)(dst + src * NH) * nb + b] * v[src];
    out_hc[((size_t)dst * n_embd + d) * nb + b] = wmma::f32_to_bf16(acc);
  }
}

__global__ void batch_hc_add_kernel(float* __restrict__ dst, const float* __restrict__ src,
                                    uint64_t n) {
  const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}

// The same, into the bf16 hyper-connection lane. Only a layer with NO control reaches this: it adds
// the sublayer's block into lane 0 and leaves the other lanes alone. `ZB` as above — one thread per
// element of `src`, so the zero rides the pass that was already reading it.
__global__ void batch_hc_add_bf16_kernel(uint16_t* __restrict__ dst, float* __restrict__ src,
                                         uint64_t n, int zb) {
  const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = wmma::f32_to_bf16(wmma::bf16_to_f32(dst[i]) + src[i]);
    if (zb) src[i] = 0.0f;
  }
}

__global__ void batch_hc_seed_kernel(const float* __restrict__ emb, uint16_t* __restrict__ hc,
                                     uint32_t n_embd, uint32_t n_hc, uint32_t nb) {
  const uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t b = (uint32_t)(t % nb), d = (uint32_t)(t / nb);
  if (d >= n_embd) return;
  const uint16_t v = wmma::f32_to_bf16(emb[(size_t)b * n_embd + d]);
  for (uint32_t h = 0; h < n_hc; ++h) hc[((size_t)h * n_embd + d) * nb + b] = v;
}


// The same gather out of a bf16 dim-major buffer — the hyper-connection lanes, which the host reads
// one token at a time at the end of a chunk.
__global__ void batch_column_bf16_kernel(const uint16_t* __restrict__ src, float* __restrict__ dst,
                                         uint32_t n, uint32_t nb, uint32_t b) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = wmma::bf16_to_f32(src[(size_t)i * nb + b]);
}

// `h.mean(dim=2)` straight into the plane the draft's projection reads, token-major, with no host
// copy on either side of it.
//
// What this replaces was the same mean into a dim-major scratch, a transpose, a D2H, a STREAM DRAIN
// because the host wanted the bytes, a memcpy into the tap plane, and — a few calls later — an H2D
// of that same plane back onto both cards. The arithmetic below is transcribed exactly: `h` ascends,
// the divide is by the same float, so the values are bit-identical to what the host used to receive.
// Only their address changed.
//
// Written by EVERY card rather than by rank 0 and pushed. The hc lanes are replicated — hc_post
// expands from a block both cards have already all-reduced — so each card recomputes an identical
// mean from identical inputs, which costs one small kernel and no crossing at all.
//
// `b_lo` is where this sub-batch's kept tokens start: a chunk that straddles the trailing window
// keeps only its tail, and the row a token lands on is `row0 + (b - b_lo)`.
__global__ void batch_hc_tap_kernel(const uint16_t* __restrict__ hc, float* __restrict__ tap,
                                    uint32_t n_embd, uint32_t n_hc, uint32_t nb, uint32_t b_lo,
                                    uint32_t n_keep, uint32_t row0, uint32_t rows_cap,
                                    uint32_t row_stride, uint32_t slot_off) {
  const uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t b = (uint32_t)(t % n_keep), d = (uint32_t)(t / n_keep);
  if (d >= n_embd) return;
  const uint32_t row = row0 + b;
  if (row >= rows_cap) return;
  float acc = 0.0f;
  for (uint32_t h = 0; h < n_hc; ++h)
    acc += wmma::bf16_to_f32(hc[((size_t)h * n_embd + d) * nb + (b_lo + b)]);
  tap[(size_t)row * row_stride + slot_off + d] = acc / (float)n_hc;
}

constexpr uint32_t kArgmaxThr = 256;

__global__ __launch_bounds__(kArgmaxThr) void batch_argmax_kernel(const float* __restrict__ y,
                                                                  float* __restrict__ out,
                                                                  uint32_t rows, uint32_t nb,
                                                                  uint32_t row0, uint32_t y_rs,
                                                                  uint32_t y_cs) {
  __shared__ float sv[kArgmaxThr];
  __shared__ uint32_t si[kArgmaxThr];
  const uint32_t b = blockIdx.x;
  float best = -INFINITY;
  uint32_t bi = 0;
  // Strictly greater, so a thread's own scan keeps the LOWEST row on a tie — which is what an
  // argmax over a contiguous host vector does, and the two have to agree or a greedy speculative
  // decode stops being bit-identical to a greedy one on exactly the ties nobody looks at.
  for (uint32_t r = threadIdx.x; r < rows; r += kArgmaxThr) {
    const float v = y[(size_t)r * y_rs + (size_t)b * y_cs];
    if (v > best) { best = v; bi = r; }
  }
  sv[threadIdx.x] = best;
  si[threadIdx.x] = bi;
  __syncthreads();
  for (uint32_t s = kArgmaxThr / 2; s; s >>= 1) {
    if (threadIdx.x < s) {
      const float ov = sv[threadIdx.x + s];
      const uint32_t oi = si[threadIdx.x + s];
      if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) {
        sv[threadIdx.x] = ov; si[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }
  if (!threadIdx.x) { out[b] = sv[0]; out[(size_t)nb + b] = (float)(row0 + si[0]); }
}

// One block per 256 rows of the band, so the whole thing is one grid-wide argmax done in two
// levels: each block reduces its own rows into `out`, and there is no second launch because the
// blocks combine through a spin-free trick — they do not. `out` is written by block 0 after an
// atomic max on a packed (value, row) key, which is exact for f32 logits because the ordering of
// IEEE floats above -inf agrees with the ordering of their bit patterns as sign-magnitude.
//
// `kMarkovRankMax` bounds the LDS copy of w1's row; the shipped rank is 256.
constexpr uint32_t kMarkovThr = 256;
constexpr uint32_t kMarkovRankMax = 1024;

// Pack (value, row) so a single atomicMax orders by value and then by LOWEST row — the same
// tie-break batch_argmax_kernel and a host argmax use. Values are logits, always finite here.
//
// `row` is ABSOLUTE — the row's index in the whole vocabulary, not in this card's band. Within one
// card that is a monotone shift and changes no ordering, so the packed maximum is the same one it
// always was. What it buys is that the two cards' keys are now comparable to each other, which is
// what lets the combine be an 8-byte peer copy and a max instead of two logit planes coming back
// to the host so it can run the same comparison there.
__device__ __forceinline__ uint64_t markov_key(float v, uint32_t row) {
  uint32_t u = __float_as_uint(v);
  u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);      // monotone map f32 -> uint32
  return ((uint64_t)u << 32) | (uint64_t)(~row);       // ~row: bigger key = smaller row
}

__global__ __launch_bounds__(kMarkovThr) void markov_step_kernel(
    const uint16_t* __restrict__ w1, const uint16_t* __restrict__ w2,
    const uint32_t* __restrict__ ids, const float* __restrict__ logits,
    unsigned long long* __restrict__ best, uint32_t rows, uint32_t row0, uint32_t rank,
    uint32_t nb, uint32_t col, uint32_t l_rs, uint32_t l_cs) {
  __shared__ float s_e[kMarkovRankMax];
  __shared__ unsigned long long s_best;
  // Step `col`'s input token is step `col-1`'s winner, and it is read HERE rather than passed as a
  // launch argument. That is the whole of what made the chain serial through the host: a kernel
  // argument has to be a host value, so every step had to wait for the previous one's argmax to
  // come back. `ids[0]` is the block's first token and `ids[col+1]` is what this step will win.
  const uint32_t token = ids[col];
  for (uint32_t i = threadIdx.x; i < rank; i += kMarkovThr)
    s_e[i] = wmma::bf16_to_f32(w1[(size_t)token * rank + i]);
  if (!threadIdx.x) s_best = 0ull;
  __syncthreads();

  // ONE WAVE per row, not one thread. A thread owning a whole `rank`-wide row reads 512 contiguous
  // bytes, but the 32 threads of a wave then sit on 32 rows 512 bytes apart, so every load
  // instruction touches 32 separate cachelines and uses a fraction of each. That is the same
  // signature every LDS-staged GEMM here has, arrived at from the opposite direction. A wave
  // sweeping one row reads it as whole cachelines instead.
  //
  // The dot product is a wave tree now rather than a serial walk, so the last bits move. That
  // cannot change the OUTPUT: greedy speculation only ever emits the target's argmax, and this is
  // the draft. It changes how often the draft is right, and nothing else.
  const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
  const uint32_t r = blockIdx.x * (kMarkovThr / kWave) + wave;
  if (r < rows) {
    const uint16_t* row = w2 + (size_t)(row0 + r) * rank;
    float acc = 0.0f;
    for (uint32_t i = lane * 8; i < rank; i += kWave * 8) {
      const uint4 v = *(const uint4*)(row + i);
      const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        acc = fmaf(__uint_as_float(w[j] << 16), s_e[i + j * 2], acc);
        acc = fmaf(__uint_as_float(w[j] & 0xffff0000u), s_e[i + j * 2 + 1], acc);
      }
    }
#pragma unroll
    for (uint32_t o = kWave / 2; o; o >>= 1) acc += __shfl_xor(acc, o, kWave);
    // (l_rs, l_cs) rather than (nb, 1): the head writes its logits dim-major when a GEMM produced
    // them and token-major when the multi-x matvec did, and token-major is also the layout that
    // reads one contiguous column here instead of one float out of every 64-byte group.
    if (!lane)
      atomicMax(&s_best, (unsigned long long)markov_key(
                             logits[(size_t)r * l_rs + (size_t)col * l_cs] + acc, row0 + r));
  }
  __syncthreads();
  if (!threadIdx.x && s_best) atomicMax(best, s_best);
}

// The two cards' packed keys combined into step `col`'s winner, which is step `col+1`'s input.
//
// This is the whole cross-card exchange. The alternative is for each card to unpack its band's best
// into a (value, id) pair and copy 2*nb floats to the host, which then DRAINS — twice a step, five
// steps a block — to run `v > best || (v == best && id < win)` itself. That comparison is
// exactly what `max` on the packed key does, because the key was built to order by value and then
// by lowest row, so the same tie-break is preserved by construction rather than by two copies of
// it staying in agreement.
//
// Every card runs this on its own copy, with `a` its own key and `peers` the others', staged
// contiguously. `max` is associative and commutative and the packed key is a total order, so the
// fold lands on the same winner on every card whatever order it visits them in — which is what
// makes the rank count free here, where a float sum would have needed a fixed order.
__global__ void markov_combine_kernel(const unsigned long long* __restrict__ a,
                                      const unsigned long long* __restrict__ peers, uint32_t npeer,
                                      uint32_t* __restrict__ ids, uint32_t col) {
  if (threadIdx.x) return;
  uint64_t k = *a;
  for (uint32_t j = 0; j < npeer; ++j) {
    const uint64_t kb = peers[j];
    k = k > kb ? k : kb;
  }
  ids[col + 1] = ~(uint32_t)(k & 0xffffffffu);
}

// The two cards' per-position argmaxes, combined without either plane coming back to the host.
//
// The vocabulary is Row-split, so a card knows only the best of its own band. Reading both `b_amx`
// planes back costs a stream drain per card per verify block, to run "higher value, then lower id"
// on the host. That is exactly what `markov_key` already packs into one word, and the argmax
// kernel already writes the id ABSOLUTE, so the two cards' keys are directly comparable: an 8-byte
// comparison per position instead of a round trip per card.
//
// `a` is this card's plane and `peers` the copies of the others', staged back to back. Each is
// [2][nb]: value at [t], id-as-float at [nb + t]. Ids are under 2^24 so the float holds them
// exactly. npeer == 0 is the identity, which is what "no other card had a band" means.
__global__ void batch_head_combine_kernel(const float* __restrict__ a,
                                          const float* __restrict__ peers, uint32_t npeer,
                                          uint32_t* __restrict__ ids, uint32_t n, uint32_t nb) {
  const uint32_t t = threadIdx.x;
  if (t >= n) return;
  uint64_t k = markov_key(a[t], (uint32_t)a[(size_t)nb + t]);
  for (uint32_t j = 0; j < npeer; ++j) {
    const float* b = peers + (size_t)j * 2u * nb;
    const uint64_t kb = markov_key(b[t], (uint32_t)b[(size_t)nb + t]);
    k = k > kb ? k : kb;
  }
  ids[t] = ~(uint32_t)(k & 0xffffffffu);
}

} // namespace

// Every launch here is checked with its own shape attached. A bare AFF_HIP_CHECK(hipGetLastError())
// after a launch names the FILE and LINE and nothing else, and "invalid resource handle" at a line
// number is indistinguishable from a sticky error left by an earlier kernel.
#define AFF_BLAUNCH(kern, grid, block, lds, st, ...)                                            \
  do {                                                                                          \
    const dim3 aff_g_ = (grid), aff_b_ = (block);                                               \
    hipLaunchKernelGGL(kern, aff_g_, aff_b_, (lds), (st), __VA_ARGS__);                          \
    const hipError_t aff_e_ = hipGetLastError();                                                \
    if (aff_e_ != hipSuccess) {                                                                 \
      aff::ui::fatal("%s: grid(%u,%u,%u) block(%u,%u,%u) lds=%zu -> %s\n", #kern,          \
                   aff_g_.x, aff_g_.y, aff_g_.z, aff_b_.x, aff_b_.y, aff_b_.z, (size_t)(lds),   \
                   hipGetErrorString(aff_e_));                                                  \
      std::abort();                                                                             \
    }                                                                                           \
  } while (0)

// ---- launchers --------------------------------------------------------------------------------

namespace {

// ---- AFF_PROFILE: which CALL SITE issued which GEMM shape, and the bytes it read ----------------
//
// A shape does not name a tensor. `[1024, 4096]` fp8 is `q_a_proj`, it is `attn_out_a`, and it is
// the draft's copies of both; `[24, 16384]` bf16 is the hyper-connection control matrix issued from
// two different places. rocprof cannot separate them either — it names kernels, not callers, which
// is the whole reason the `batch_gemm_wmma` bucket had never been attributed. So the caller's own
// line rides in on a `__builtin_LINE()` default argument declared in the header, which is evaluated
// AT THE CALL SITE.
//
// It must cover EVERY entry point, not just the single-matrix one: tallying `gemm_launch` alone
// leaves the two grouped launchers invisible, and the grouped fp8 path is where the W8A8 dense chain
// lives — so the total comes out short and every share is against the wrong denominator.
struct GemmSite {
  uint32_t line, quant, rows, cols, nb, xb;
  const char* file;
  bool operator<(const GemmSite& o) const {
    if (line != o.line) return line < o.line;
    if (quant != o.quant) return quant < o.quant;
    if (rows != o.rows) return rows < o.rows;
    if (cols != o.cols) return cols < o.cols;
    if (nb != o.nb) return nb < o.nb;
    if (xb != o.xb) return xb < o.xb;
    return file < o.file;
  }
};
struct GemmTally {
  std::map<GemmSite, uint64_t> n;
  std::mutex mu;
  ~GemmTally() {
    if (n.empty()) return;
    aff::ui::err("\ngemm call sites (weight bytes read, and the shape each one issues)\n"
                 "  q: 2=bf16 3=fp8-e4m3 4=f32. xb: bf16 activation. site: file:line of the CALLER.\n");
    double tot = 0;
    std::vector<std::pair<double, std::string>> lines;
    for (auto& kv : n) {
      const GemmSite& k = kv.first;
      const double bpw = k.quant == AFF_BF16 ? 2.0 : (k.quant == AFF_F32 ? 4.0 : 1.0);
      const double mib = (double)k.rows * k.cols * bpw / 1048576.0 * (double)kv.second;
      tot += mib;
      const char* base = std::strrchr(k.file, '/');
      char buf[200];
      std::snprintf(buf, sizeof buf,
                    "  %9.1f MiB  x%-6llu  q=%u xb=%u rows=%6u cols=%6u nb=%3u   %s:%u", mib,
                    (unsigned long long)kv.second, k.quant, k.xb, k.rows, k.cols, k.nb,
                    base ? base + 1 : k.file, k.line);
      lines.emplace_back(mib, buf);
    }
    std::sort(lines.rbegin(), lines.rend());
    for (auto& l : lines) aff::ui::err("%s\n", l.second.c_str());
    aff::ui::err("  TOTAL %.1f MiB over the run\n", tot);
  }
};
// `static const bool`, not a `getenv` per launch: this sits on the launch path of every GEMM in
// every run, profiled or not, and getenv is a linear scan of `environ`.
inline bool gemm_profiling() {
  static const bool on = getenv("AFF_PROFILE") != nullptr;
  return on;
}
inline void gemm_tally(uint32_t quant, uint32_t rows, uint32_t cols, uint32_t nb, bool xb,
                       const char* file, uint32_t line) {
  static GemmTally t;
  std::lock_guard<std::mutex> lk(t.mu);
  ++t.n[GemmSite{line, quant, rows, cols, nb, (uint32_t)xb, file}];
}

void gemm_launch(uint32_t quant, const void* W, const uint8_t* S, const void* X, float* Y,
                 uint32_t rows, uint32_t cols, uint32_t nb, uint32_t x_off, uint32_t w_row_off,
                 hipStream_t st, bool x_bf16 = false, const char* site_file = "batch_kernels.hip",
                 uint32_t site_line = 0) {
  const uint32_t sc_cols = (uint32_t)((cols + 127) / 128);
  if (gemm_profiling()) gemm_tally(quant, rows, cols, nb, x_bf16, site_file, site_line);
  // Everything goes to WMMA. For fp8 and bf16 the weight side of the conversion is exact (see
  // `weight_bf16`), so the only precision change is the activation. Rounding an f32 weight to bf16
  // WOULD be a real loss, so the f32 path does not round it — it splits it across three bf16 passes
  // (`batch_gemm_f32x3_kernel`) and keeps ~16 mantissa bits.
  if (quant == AFF_F32) {
    const dim3 grid((rows + kGM - 1) / kGM, (nb + kGN - 1) / kGN);
    const bool full = (rows % kGM == 0) && (nb % kGN == 0) && (cols % kGK == 0);
    const float* Wf = (const float*)W + (size_t)w_row_off * cols;
    const float* Xf = (const float*)X + (size_t)x_off * nb;
    if (full)
      AFF_BLAUNCH((batch_gemm_f32x3_kernel<true>), grid, dim3(kGThreads), 0, st, Wf, Xf, Y, rows,
                  cols, nb);
    else
      AFF_BLAUNCH((batch_gemm_f32x3_kernel<false>), grid, dim3(kGThreads), 0, st, Wf, Xf, Y, rows,
                  cols, nb);
    return;
  }
  // A speculative block is a handful of tokens in a 16-wide arena, and the wide N tile computes all
  // of it. Below one tile there is nothing to lose by narrowing — gy is 1 either way, so the weights
  // are still read exactly once — and most of the arithmetic to win. See the kernel's geometry.
  const uint32_t wn_eff = nb <= kNarrowN ? kNarrowN : kWN;
  const uint32_t gx = (rows + kWM - 1) / kWM, gy = (nb + wn_eff - 1) / wn_eff;
  // The narrow control GEMM is a couple of blocks on the whole card, reading tens of MB at a
  // two-block share of bandwidth. Split K until the grid can fill the machine. `ck` is rounded to
  // 128 so every split still begins on a scale-block boundary.
  //
  // "Fill the machine" is NOT one block a WGP: a block is 8 waves, which on four SIMDs is two waves
  // each — enough to occupy the WGPs and nowhere near enough to hide a global load. At one block a
  // WGP the fn control GEMM still runs at a fraction of the streaming rate.
  uint32_t ksplit = 1;
  float* ws = nullptr;
  if (gx * gy < kSplitTarget && cols >= 512u) {
    const uint32_t want = (kSplitTarget + gx * gy - 1) / (gx * gy);
    uint32_t ck = ((cols / want) + 127u) / 128u * 128u;
    if (ck < 128u) ck = 128u;
    uint32_t k2 = (cols + ck - 1) / ck;
    // The reduce below launches rows*nb/256 blocks and every thread sums `ksplit` partials in a
    // fixed order, so a small OUTPUT makes it as thin as the GEMM the split was widening. The hc
    // control GEMM is that case — a couple of workgroups — where at a wide split the reduce costs
    // several times the GEMM it is reducing. Below a floor of reduce blocks, cap the split and let
    // the GEMM keep the wider tail.
    if (k2 > kSplitThinCap && (uint64_t)rows * nb < 8192) k2 = kSplitThinCap;
    if (k2 > 1) {
      ws = splitk_buf((size_t)k2 * rows * nb);
      if (ws) ksplit = k2;
    }
  }
  uint32_t ck = ksplit > 1 ? ((cols + ksplit - 1) / ksplit + 127u) / 128u * 128u : 0u;
  // ksplit FROM ck, not the other way round: derived independently they disagree (cols 4096 with
  // k2 13 gives ck 384 and covers the matrix in 11), and the slices past the end wrote nothing
  // while the reduce still summed their uninitialised partials.
  if (ck) ksplit = (cols + ck - 1) / ck;
  if (ksplit < 2) { ck = 0; ksplit = 1; ws = nullptr; }
  const dim3 grid(gx, gy, ksplit);
  const bool full = (rows % kWM == 0) && (nb % wn_eff == 0) && (cols % kWK == 0);
  const bool narrow = wn_eff == kNarrowN;
  float* const dst = ksplit > 1 ? ws : Y;
#define AFF_GEMM_G(q, f, xb, mt, nt, nw)                                                         \
  AFF_BLAUNCH((batch_gemm_wmma_kernel<q, f, xb, mt, nt, nw>), grid, dim3(kWThreads), 0, st, W, S, \
              X, dst, rows, cols, nb, sc_cols, x_off, w_row_off, ck)
#define AFF_GEMM_W(q, f, xb)                                                                     \
  do {                                                                                           \
    if (narrow) AFF_GEMM_G(q, f, xb, 1, 1, 1); else AFF_GEMM_G(q, f, xb, kWMTiles, kWNTiles, 2); \
  } while (0)
#define AFF_GEMM_X(q)                                                                            \
  do {                                                                                           \
    if (x_bf16) { if (full) AFF_GEMM_W(q, true, true);  else AFF_GEMM_W(q, false, true); }        \
    else        { if (full) AFF_GEMM_W(q, true, false); else AFF_GEMM_W(q, false, false); }       \
  } while (0)
  if (quant == AFF_BF16) AFF_GEMM_X(AFF_BF16); else AFF_GEMM_X(AFF_FP8_E4M3);
#undef AFF_GEMM_X
#undef AFF_GEMM_W
  if (ksplit > 1) {
    const size_t n = (size_t)rows * nb;
    AFF_BLAUNCH(splitk_reduce_kernel, dim3((uint32_t)((n + 255) / 256)), dim3(256), 0, st, ws, Y,
                rows, nb, ksplit);
  }
}
}

bool batch_gemm_multi_hip(uint32_t quant, const BatchGemmJob* jobs, uint32_t n, const void* X,
                          uint32_t cols, uint32_t nb, bool x_bf16, void* stream,
                          const char* aff_file, uint32_t aff_line) {
  if (!n || n > kGemmJobsMax || quant == AFF_F32) return false;
  // Narrow only. The wide instantiation is prefill, where a single GEMM's grid is already in the
  // hundreds of blocks and there is nothing to fuse.
  if (nb > kNarrowN) return false;
  // Tallied per JOB, and after the two refusals above, so the table counts the weights this launch
  // actually reads and does not credit a group the caller had to issue one at a time anyway.
  if (gemm_profiling())
    for (uint32_t i = 0; i < n; ++i)
      gemm_tally(quant, jobs[i].rows, cols, nb, x_bf16, aff_file, aff_line);
  GemmJobsArgs J{};
  constexpr uint32_t kM = (kWWaves / 1u) * 1u * 16u;     // rows a block in the narrow geometry
  uint32_t tiles = 0, total = 0;
  bool full = (cols % kWK == 0) && (nb % kNarrowN == 0);
  for (uint32_t i = 0; i < n; ++i) {
    J.rows[i] = jobs[i].rows;
    tiles += (jobs[i].rows + kM - 1) / kM;
    J.tile_end[i] = tiles;
    total += jobs[i].rows;
    full = full && (jobs[i].rows % kM == 0);
  }
  J.n = n;
  const uint32_t gy = (nb + kNarrowN - 1) / kNarrowN;
  // The same split-K rule gemm_launch uses, on the UNION grid.
  uint32_t ksplit = 1;
  float* ws = nullptr;
  if (tiles * gy < kSplitTarget && cols >= 512u) {
    const uint32_t want = (kSplitTarget + tiles * gy - 1) / (tiles * gy);
    uint32_t ckw = ((cols / want) + 127u) / 128u * 128u;
    if (ckw < 128u) ckw = 128u;
    uint32_t k2 = (cols + ckw - 1) / ckw;
    if (k2 > kSplitThinCap && (uint64_t)total * nb < 8192) k2 = kSplitThinCap;
    if (k2 > 1) {
      ws = splitk_buf((size_t)k2 * total * nb);
      if (ws) ksplit = k2;
    }
  }
  uint32_t ck = ksplit > 1 ? ((cols + ksplit - 1) / ksplit + 127u) / 128u * 128u : 0u;
  if (ck) ksplit = (cols + ck - 1) / ck;
  if (ksplit < 2) { ck = 0; ksplit = 1; ws = nullptr; }
  // Jobs write into consecutive bands of one destination, so the partial plane is one [z][total][nb]
  // block and a single reduce covers every job.
  uint32_t off = 0;
  for (uint32_t i = 0; i < n; ++i) {
    J.w[i] = jobs[i].w; J.s[i] = jobs[i].s;
    J.xoff[i] = jobs[i].x_off; J.woff[i] = jobs[i].w_row_off;
    J.y[i] = (ws ? ws : jobs[0].y) + (size_t)off * nb;
    off += jobs[i].rows;
  }
  const dim3 grid(tiles, gy, ksplit);
  const uint32_t sc_cols = (uint32_t)((cols + 127) / 128);
  const hipStream_t st = (hipStream_t)stream;
#define AFF_GEMM_M(q, f, xb) \
  AFF_BLAUNCH((batch_gemm_multi_kernel<q, f, xb>), grid, dim3(kWThreads), 0, st, J, X, cols, nb, \
              sc_cols, ck, total)
  if (quant == AFF_BF16) {
    if (x_bf16) { if (full) AFF_GEMM_M(AFF_BF16, true, true);  else AFF_GEMM_M(AFF_BF16, false, true); }
    else        { if (full) AFF_GEMM_M(AFF_BF16, true, false); else AFF_GEMM_M(AFF_BF16, false, false); }
  } else {
    if (x_bf16) { if (full) AFF_GEMM_M(AFF_FP8_E4M3, true, true);  else AFF_GEMM_M(AFF_FP8_E4M3, false, true); }
    else        { if (full) AFF_GEMM_M(AFF_FP8_E4M3, true, false); else AFF_GEMM_M(AFF_FP8_E4M3, false, false); }
  }
#undef AFF_GEMM_M
  if (ksplit > 1) {
    const size_t nn = (size_t)total * nb;
    AFF_BLAUNCH(splitk_reduce_kernel, dim3((uint32_t)((nn + 255) / 256)), dim3(256), 0, st, ws,
                jobs[0].y, total, nb, ksplit);
  }
  return true;
}

void batch_gemm_hip(uint32_t quant, const void* W, const uint8_t* S, const float* X, float* Y,
                    uint32_t rows, uint32_t cols, uint32_t nb, void* stream, const char* aff_file,
                    uint32_t aff_line) {
  gemm_launch(quant, W, S, X, Y, rows, cols, nb, 0, 0, (hipStream_t)stream, false, aff_file,
              aff_line);
}

bool batch_gemm_bf16x_hip(uint32_t quant, const void* W, const uint8_t* S, const uint16_t* X,
                          float* Y, uint32_t rows, uint32_t cols, uint32_t nb, void* stream,
                          const char* aff_file, uint32_t aff_line) {
  // AFF_F32 weights take `batch_gemm_f32x3_kernel`, which splits the WEIGHT across three bf16
  // passes and reads the activation as f32. A bf16 activation there would be a silent layout
  // mismatch, so refuse rather than reinterpret — the caller returns false and model.cpp aborts.
  if (quant == AFF_F32) return false;
  gemm_launch(quant, W, S, X, Y, rows, cols, nb, 0, 0, (hipStream_t)stream, true, aff_file,
              aff_line);
  return true;
}

// The grid is (token groups, 128-dim blocks). Y is what makes it fill the card: the token axis alone
// is tens of blocks and the dim axis multiplies that by the block count. `tile` is ~18 KiB of LDS so
// only about three blocks are resident a CU regardless — the point is having enough of them that
// every CU has one at all.
static inline dim3 quant_grid(uint32_t dim, uint32_t nb) {
  const uint32_t nkb = dim >> 7;
  return dim3((nb + kQTok - 1) / kQTok, nkb ? nkb : 1u);
}

void batch_quant_fp8_hip(const float* x, uint8_t* out, float* scales, uint32_t dim, uint32_t nb,
                         uint32_t x_off, void* stream) {
  AFF_BLAUNCH(batch_quant_fp8_kernel<QSrcF32>, quant_grid(dim, nb), dim3(kQThreads), 0,
              (hipStream_t)stream, QSrcF32{x}, out, scales, dim, nb, x_off);
}

void batch_swiglu_quant_fp8_hip(const float* g, const float* u, uint8_t* out, float* scales,
                                uint32_t dim, uint32_t nb, float limit, void* stream) {
  AFF_BLAUNCH(batch_quant_fp8_kernel<QSrcSwiGLU>, quant_grid(dim, nb), dim3(kQThreads), 0,
              (hipStream_t)stream, QSrcSwiGLU{g, u, limit}, out, scales, dim, nb, 0u);
}

// Waves a block for the narrow (nb <= 16) instantiation, i.e. the M tile in units of 16 rows. Take
// the widest tile that still leaves kSplitTarget blocks — two a CU on the 64-CU part this was tuned
// against — and never fewer than one 16-row tile a wave.
uint32_t fp8_narrow_mw(uint32_t rows) {
  const uint32_t tiles = (rows + 15u) / 16u;             // 16-row tiles this shape has at all
  for (uint32_t mw : {8u, 4u, 2u}) if (tiles / mw >= kSplitTarget) return mw;
  return tiles >= 8u ? 1u : 8u;                          // too short to split; one block will do
}

bool batch_gemm_fp8_multi_hip(const BatchGemmJob* jobs, uint32_t n, const uint8_t* X,
                              const float* Sx, uint32_t cols, uint32_t nb, uint32_t x_ld,
                              uint32_t sx_ld, void* stream, bool preshuffled, const char* aff_file,
                              uint32_t aff_line) {
  if (!n || n > kGemmJobsMax || nb > wmma::kN) return false;
  if ((cols % kFKNarrow) != 0) return false;
  if (gemm_profiling())
    for (uint32_t i = 0; i < n; ++i)
      gemm_tally(AFF_FP8_E4M3, jobs[i].rows, cols, nb, false, aff_file, aff_line);
  const uint32_t sc_cols = (uint32_t)((cols + 127) / 128);
  if (!x_ld) x_ld = cols;
  if (!sx_ld) sx_ld = sc_cols;
  GemmJobsArgs J{};
  uint32_t tiles = 0, total = 0;
  // One mw for the whole group, from the largest job: a block's row tile has to be the same for
  // every job or blockIdx.x cannot be resolved to one.
  uint32_t big = 0;
  for (uint32_t i = 0; i < n; ++i) big = jobs[i].rows > big ? jobs[i].rows : big;
  const uint32_t mw = fp8_narrow_mw(big);
  const uint32_t fm = mw * kFMTiles * 16u;
  bool full = (nb % 16 == 0) && (cols % 128 == 0);
  for (uint32_t i = 0; i < n; ++i) {
    J.rows[i] = jobs[i].rows;
    tiles += (jobs[i].rows + fm - 1) / fm;
    J.tile_end[i] = tiles;
    total += jobs[i].rows;
    full = full && (jobs[i].rows % fm == 0);
  }
  J.n = n;
  const uint32_t gy = (nb + 15u) / 16u;
  uint32_t ksplit = 1;
  float* ws = nullptr;
  if (tiles * gy * mw < kSplitTarget && cols >= 512u) {
    const uint32_t want = (kSplitTarget + tiles * gy * mw - 1) / (tiles * gy * mw);
    uint32_t ckw = ((cols / want) + kFKNarrow - 1u) / kFKNarrow * kFKNarrow;
    if (ckw < kFKNarrow) ckw = kFKNarrow;
    uint32_t k2 = (cols + ckw - 1) / ckw;
    if (k2 > kSplitThinCap && (uint64_t)total * nb < 8192) k2 = kSplitThinCap;
    if (k2 > 1) { ws = splitk_buf((size_t)k2 * total * nb); if (ws) ksplit = k2; }
  }
  uint32_t ck = ksplit > 1 ? ((cols + ksplit - 1) / ksplit + kFKNarrow - 1u) / kFKNarrow * kFKNarrow
                           : 0u;
  if (ck) ksplit = (cols + ck - 1) / ck;
  if (ksplit < 2) { ck = 0; ksplit = 1; ws = nullptr; }
  uint32_t off = 0;
  for (uint32_t i = 0; i < n; ++i) {
    J.w[i] = jobs[i].w; J.s[i] = jobs[i].s;
    J.xoff[i] = jobs[i].x_off; J.woff[i] = jobs[i].w_row_off;
    J.y[i] = (ws ? ws : jobs[0].y) + (size_t)off * nb;
    off += jobs[i].rows;
  }
  const dim3 grid(tiles, gy, ksplit);
  const hipStream_t st = (hipStream_t)stream;
#define AFF_F8M(f, ps, mw_)                                                                       \
  AFF_BLAUNCH((batch_gemm_fp8_multi_kernel<f, ps, 1u, mw_, kFKNarrow>), grid, dim3(mw_ * kWave),   \
              0, st, J, X, Sx, cols, nb, sc_cols, x_ld, sx_ld, ck, total)
#define AFF_F8M_MW(f, ps)                                                                         \
  do {                                                                                            \
    if (mw == 1u)      AFF_F8M(f, ps, 1u);                                                        \
    else if (mw == 2u) AFF_F8M(f, ps, 2u);                                                        \
    else if (mw == 4u) AFF_F8M(f, ps, 4u);                                                        \
    else               AFF_F8M(f, ps, 8u);                                                        \
  } while (0)
  if (preshuffled) { if (full) AFF_F8M_MW(true, true);  else AFF_F8M_MW(false, true); }
  else             { if (full) AFF_F8M_MW(true, false); else AFF_F8M_MW(false, false); }
#undef AFF_F8M_MW
#undef AFF_F8M
  if (ksplit > 1) {
    const size_t nn = (size_t)total * nb;
    AFF_BLAUNCH(splitk_reduce_kernel, dim3((uint32_t)((nn + 255) / 256)), dim3(256), 0, st, ws,
                jobs[0].y, total, nb, ksplit);
  }
  return true;
}

void batch_gemm_fp8_hip(const uint8_t* W, const uint8_t* Sw, const uint8_t* X, const float* Sx,
                        float* Y, uint32_t rows, uint32_t cols, uint32_t nb, uint32_t x_off,
                        uint32_t w_row_off, void* stream, bool preshuffled, uint32_t x_ld,
                        uint32_t sx_ld, bool accum, uint32_t mw, const char* aff_file,
                        uint32_t aff_line) {
  if (gemm_profiling()) gemm_tally(AFF_FP8_E4M3, rows, cols, nb, false, aff_file, aff_line);
  const uint32_t sc_cols = (uint32_t)((cols + 127) / 128);
  if (!x_ld) x_ld = cols;
  if (!sx_ld) sx_ld = sc_cols;
  // One 16-wide token tile once nb fits in one, eight otherwise. There is nothing between: the
  // engine's two callers are a 1536-token prefill chunk and a <=8-token speculative batch, and
  // every intermediate instantiation is another copy of the K loop in the binary for a shape
  // nothing issues. See the NT comment on the kernel for what the narrow form is buying.
  // The narrow tile stages 256 columns and would run past the end of a matrix that is not a
  // multiple of that — the same hazard the split-K note below describes, and it decodes fluently
  // while being wrong. V4.1 produces such a shape the moment a 2304-wide intermediate is split two
  // ways (1152), so a K that does not fit the narrow tile takes the wide kernel, whose unit is 64.
  // Slower at small `nb`, correct at every width.
  const bool narrow = nb <= wmma::kN && (cols % kFKNarrow) == 0;
  const uint32_t nt = narrow ? 1u : kFNTiles;
  const uint32_t fn = nt * 16u;
  // The M tile. Wide calls keep the 8-wave block they were tuned with; narrow ones want as many
  // blocks as the row count can be cut into, up to ~4 a CU, because at nb<=16 the only thing that
  // moves is weight bandwidth and only resident blocks generate misses. `mw` overrides for the
  // sweep in bench/spec_batch.hip; 0 means use this rule.
  // fp8_narrow_mw, not kFWaves. A wider M tile would cut the activation re-read substantially, and
  // split-K can now give it the blocks it needs — but it measures flat in isolation and slightly
  // worse in the engine. The activation window is small and every block reads the same one, so it
  // lives in L2 and the re-read never reaches DRAM.
  if (!mw) mw = narrow ? fp8_narrow_mw(rows) : kFWaves;
  const uint32_t fm = mw * kFMTiles * 16u;
  const uint32_t gx = (rows + fm - 1) / fm, gy = (nb + fn - 1) / fn;
  // Split K until the WAVE count can cover the machine's memory latency — see the note in the
  // kernel. Not when `accum` is set: the epilogue would then add each K slice into Y separately.
  uint32_t ksplit = 1;
  float* ws = nullptr;
  if (narrow && !accum && gx * gy * mw < kSplitTarget && cols >= 512u) {
    const uint32_t want = (kSplitTarget + gx * gy * mw - 1) / (gx * gy * mw);
    uint32_t ckw = ((cols / want) + 127u) / 128u * 128u;
    if (ckw < 128u) ckw = 128u;
    uint32_t k2 = (cols + ckw - 1) / ckw;
    if (k2 > kSplitThinCap && (uint64_t)rows * nb < 8192) k2 = kSplitThinCap;      // see gemm_launch
    if (k2 > 1) {
      ws = splitk_buf((size_t)k2 * rows * nb);
      if (ws) ksplit = k2;
    }
  }
  // Rounded to the STAGED TILE, not to the 128-column scale block. A narrow tile is 256 wide, so a
  // slice whose length is only a multiple of 128 runs one tile past its own end and accumulates the
  // next slice's columns as well — a wrong answer that still decodes fluently.
  // max, not the tile alone: a slice must be a whole number of 128-column SCALE blocks as well,
  // or the UE8M0 fold reads the wrong exponent for part of it.
  const uint32_t kq = narrow ? (kFKNarrow > 128u ? kFKNarrow : 128u) : 128u;
  uint32_t ck = ksplit > 1 ? ((cols + ksplit - 1) / ksplit + kq - 1u) / kq * kq : 0u;
  if (ck) ksplit = (cols + ck - 1) / ck;                     // see gemm_launch
  if (ksplit < 2) { ck = 0; ksplit = 1; ws = nullptr; }
  const dim3 grid(gx, gy, ksplit);
  float* const dst = ksplit > 1 ? ws : Y;
  // `cols % 128` is required, not just preferred: the scale block is 128 wide and the fold at the
  // end of the K loop assumes a whole one.
  const bool full = (rows % fm == 0) && (nb % fn == 0) && (cols % 128 == 0);
  // A narrow tile stages 256 columns; below that there is nothing to widen and the tail test
  // inside the ks loop would be the only thing running. Every dense tensor here has cols a
  // multiple of 1024, so this is a guard rather than a path.
  if (narrow && (cols % kFKNarrow) != 0) { aff::ui::fatal("batch_gemm_fp8: cols=%u is not a multiple of %u\n", cols, kFKNarrow); std::abort(); }
// FK: a whole 256-byte cacheline a row for the narrow instantiation, 64 for the wide one. Wide is
// prefill, where the LDS tile is 128 rows and 256 bytes of it would be 32 KiB of A alone -- half
// the occupancy, on a path that is already at its roofline.
#define AFF_GEMM_F8(f, ps, nt_, mw_, fk_)                                                         \
  AFF_BLAUNCH((batch_gemm_fp8_kernel<f, ps, nt_, mw_, fk_>), grid, dim3(mw_ * kWave), 0,          \
              (hipStream_t)stream, W, Sw, X, Sx, dst, rows, cols, nb, sc_cols, x_off, w_row_off,  \
              x_ld, sx_ld, accum ? 1 : 0, ck)
#define AFF_GEMM_F8_MW(f, ps)                                                                     \
  do {                                                                                            \
    if (!narrow)        AFF_GEMM_F8(f, ps, kFNTiles, kFWaves, kFK);                               \
    else if (mw == 1u)  AFF_GEMM_F8(f, ps, 1u, 1u, kFKNarrow);                                    \
    else if (mw == 2u)  AFF_GEMM_F8(f, ps, 1u, 2u, kFKNarrow);                                    \
    else if (mw == 4u)  AFF_GEMM_F8(f, ps, 1u, 4u, kFKNarrow);                                    \
    else                AFF_GEMM_F8(f, ps, 1u, 8u, kFKNarrow);                                    \
  } while (0)
  if (preshuffled) { if (full) AFF_GEMM_F8_MW(true, true);  else AFF_GEMM_F8_MW(false, true); }
  else             { if (full) AFF_GEMM_F8_MW(true, false); else AFF_GEMM_F8_MW(false, false); }
#undef AFF_GEMM_F8_MW
#undef AFF_GEMM_F8
  if (ksplit > 1) {
    const size_t n = (size_t)rows * nb;
    AFF_BLAUNCH(splitk_reduce_kernel, dim3((uint32_t)((n + 255) / 256)), dim3(256), 0,
                (hipStream_t)stream, ws, Y, rows, nb, ksplit);
  }
}

void preshuffle_fp8_hip(const uint8_t* src, uint8_t* dst, uint32_t rows, uint32_t cols,
                        void* stream) {
  const uint32_t ktiles = cols / 16, groups = (rows / 16) * ktiles * 32;
  const dim3 grid((groups + 255) / 256);
  AFF_BLAUNCH(preshuffle_fp8_kernel, grid, dim3(256), 0, (hipStream_t)stream, src, dst, cols, ktiles,
              groups);
}

void batch_gemm_grouped_hip(uint32_t quant, const void* W, const uint8_t* S, const void* X,
                            float* Y, uint32_t n_groups, uint32_t group_dim, uint32_t rank,
                            uint32_t nb, void* stream, bool x_bf16, const char* aff_file,
                            uint32_t aff_line) {
  // One launch per group, and it stays that way after TWO attempts to fuse them.
  //
  // The first held the fused union grid to kSplitTarget blocks, and regressed with identical
  // arithmetic: each group on its own splits K until it reaches the target, so n_groups launches
  // carry n_groups times that many block-slots between them, and the union carried a fraction of it.
  // The second scaled the target by the job count, which makes the block-slots match — and measured
  // exactly flat while removing several GEMM launches and split-K reduces a layer. Whatever those
  // launches cost, the fused kernel gives back.
  //
  // Fusion still wins where the jobs are small and the union is thin: the compressor's four,
  // shexp's pair, attn_qkv's pair. It does not win where each job could have filled the card.
  for (uint32_t g = 0; g < n_groups; ++g)
    gemm_launch(quant, W, S, X, Y + (size_t)g * rank * nb, rank, group_dim, nb, g * group_dim,
                g * rank, (hipStream_t)stream, x_bf16, aff_file, aff_line);
}

void batch_rms_norm_hip(const float* x, const float* w32, const uint16_t* wbf16, float* out,
                        uint32_t n, uint32_t nb, float eps, void* stream) {
  AFF_BLAUNCH(batch_rms_norm_kernel, dim3((nb + kBlkTok - 1) / kBlkTok), dim3(kBlkThreads),
                     0, (hipStream_t)stream, x, w32, wbf16, out, n, nb, eps);
}

bool batch_rms_norm_q8_hip(const float* x, const float* w32, const uint16_t* wbf16, float* out,
                           uint8_t* q8, float* qs, uint32_t n, uint32_t nb, float eps,
                           void* stream) {
  if (nb > kNarrowTok || (n % 128) != 0 || (n >> 7) > kRmsQWinMax) return false;
  AFF_BLAUNCH(batch_rms_norm_q8_kernel, dim3(nb), dim3(kRmsQThr), 0, (hipStream_t)stream, x, w32,
              wbf16, out, q8, qs, n, nb, eps);
  return true;
}

void batch_rms_norm_rope_hip(const float* x, const float* w, float* out, uint32_t n, uint32_t n_rot,
                             uint32_t pos0, RopeDerived r, float eps, uint32_t nb, void* stream) {
  AFF_BLAUNCH(batch_rms_norm_rope_kernel, dim3((nb + kBlkTok - 1) / kBlkTok),
                     dim3(kBlkThreads), 0, (hipStream_t)stream, x, w, out, n, nb, eps, n_rot, pos0,
                     r.freq_scale, r.theta_scale, r.ext_factor, r.lo, r.hi);
}

bool batch_head_rms_rope_tr_hip(const float* x, uint16_t* dst, float* inv, uint32_t n_head,
                                uint32_t head_dim, uint32_t n_rot, uint32_t pos0, RopeDerived r,
                                float eps, uint32_t nb, void* stream) {
  // Both are alignment facts about the model, not tuning: violate either and the tile would have to
  // straddle a head boundary or a rope boundary, and the kernel would be silently wrong for the
  // rows that straddle. Refuse instead — the caller returns false and model.cpp aborts loudly,
  // which is what every other precondition on this path does.
  if (head_dim % kQTr || n_rot % kQTr || n_rot > head_dim) return false;
  if (nb == kNarrowTok) {
    AFF_BLAUNCH((hcn_head_rms_inv_kernel<kNarrowTok>), dim3(n_head), dim3(kHcnThreads), 0,
                (hipStream_t)stream, x, inv, head_dim, eps);
  } else {
    const uint32_t thr = 256, blk = (n_head * nb + thr - 1) / thr;
    AFF_BLAUNCH(batch_head_rms_inv_kernel, dim3(blk), dim3(thr), 0, (hipStream_t)stream,
                x, inv, n_head, head_dim, nb, eps);
  }
  const uint32_t n = n_head * head_dim;
  AFF_BLAUNCH(batch_head_scale_rope_tr_kernel, dim3((nb + kQTr - 1) / kQTr, (n + kQTr - 1) / kQTr),
              dim3(kQTr, kQTrRows), 0, (hipStream_t)stream, x, dst, inv, n, nb, head_dim, n_rot,
              pos0, r.freq_scale, r.theta_scale, r.ext_factor, r.lo, r.hi);
  return true;
}


void batch_rope_tail_bf16_hip(uint16_t* x, uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
                              uint32_t pos0, RopeDerived r, float sin_sign, uint32_t nb,
                              void* stream) {
  const uint32_t thr = 256, pairs = n_rot / 2;
  const uint32_t blk = (n_head * pairs * nb + thr - 1) / thr;
  AFF_BLAUNCH(batch_rope_tail_bf16_kernel, dim3(blk), dim3(thr), 0, (hipStream_t)stream,
                     x, n_head, head_dim, nb, n_rot, pos0, r.freq_scale, r.theta_scale,
                     r.ext_factor, r.lo, r.hi, sin_sign);
}

// dst[dim][nb] <- src[nb][dim], rounding to bf16. The heads path only: attn_out's transpose is the
// producer of a buffer two GEMMs and a quantiser then read, and all three take bf16 without loss —
// the WMMA one truncates to bf16 anyway and the quantiser is going to E4M3.
void batch_transpose_bf16_hip(const float* src, uint16_t* dst, uint32_t n, uint32_t nb,
                              int to_token_major, void* stream) {
  const dim3 grid(((to_token_major ? nb : n) + kTrTile - 1) / kTrTile,
                  ((to_token_major ? n : nb) + kTrTile - 1) / kTrTile);
  AFF_BLAUNCH(batch_transpose_kernel, grid, dim3(kTrTile, kTrRows), 0, (hipStream_t)stream,
                     src, nullptr, dst, n, nb, to_token_major, 1.0f);
}

void batch_swiglu_hip(const float* g, const float* u, float* out, uint32_t n, uint32_t nb,
                      float limit, void* stream) {
  const uint64_t total = (uint64_t)n * nb;
  const uint32_t thr = 256;
  AFF_BLAUNCH(batch_swiglu_kernel, dim3((uint32_t)((total + thr - 1) / thr)), dim3(thr), 0,
                     (hipStream_t)stream, g, u, out, total, limit);
}

void batch_transpose_multi_hip(const float* const* src, float* const* dst, const uint32_t* n,
                               uint32_t n_jobs, uint32_t nb, void* stream) {
  if (!n_jobs || n_jobs > kTrJobsMax) return;
  TrJobs J{};
  uint32_t nmax = 0;
  for (uint32_t i = 0; i < n_jobs; ++i) {
    J.src[i] = src[i]; J.dst[i] = dst[i]; J.n[i] = n[i];
    nmax = n[i] > nmax ? n[i] : nmax;
  }
  const dim3 grid((nb + kTrTile - 1) / kTrTile, (nmax + kTrTile - 1) / kTrTile, n_jobs);
  AFF_BLAUNCH(batch_transpose_multi_kernel, grid, dim3(kTrTile, kTrRows), 0, (hipStream_t)stream, J,
              nb);
}

void batch_transpose_hip(const float* src, float* dst, uint32_t n, uint32_t nb, int to_token_major,
                         void* stream) {
  batch_scale_transpose_hip(src, dst, n, nb, to_token_major, 1.0f, stream);
}

namespace {
// Reads are contiguous; writes are strided by `cap`. Deliberately that way round: this runs once
// per token per layer and the transposed alternative would need a second pass over the block at
// flush, which is more traffic than the scattered stores of 16 KiB.
__global__ void stage_col_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                 uint32_t n, uint32_t cap, uint32_t k) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[(size_t)i * cap + k] = src[i];
}
}  // namespace

void batch_stage_col_hip(const float* src, float* dst, uint32_t n, uint32_t cap, uint32_t k,
                         void* stream) {
  if (!n || k >= cap) return;
  AFF_BLAUNCH(stage_col_kernel, dim3((n + 255) / 256), dim3(256), 0, (hipStream_t)stream, src, dst,
              n, cap, k);
}

void batch_scale_transpose_hip(const float* src, float* dst, uint32_t n, uint32_t nb,
                               int to_token_major, float scale, void* stream) {
  const dim3 grid(((to_token_major ? nb : n) + kTrTile - 1) / kTrTile,
                  ((to_token_major ? n : nb) + kTrTile - 1) / kTrTile);
  AFF_BLAUNCH(batch_transpose_kernel, grid, dim3(kTrTile, kTrRows), 0, (hipStream_t)stream,
                     src, dst, nullptr, n, nb, to_token_major, scale);
}

void batch_hc_fused_hip(const uint16_t* hc, const float* mix, const float* scale, const float* base,
                        const float* nw, float* pre, float* post, float* comb, float* cur,
                        float* norm, uint32_t n_embd, uint32_t n_hc, uint32_t nb, uint32_t iters,
                        float hc_eps, float rms_eps, int has_control, void* stream, const float* pre_in) {
  AFF_BLAUNCH(batch_hc_fused_kernel, dim3((nb + kBlkTok - 1) / kBlkTok), dim3(kBlkThreads),
                     0, (hipStream_t)stream, hc, mix, scale, base, nw, pre, post, comb, cur, norm,
                     n_embd, n_hc, nb, iters, hc_eps, rms_eps, has_control, pre_in);
}

void batch_hc_fused_narrow_hip(const uint16_t* hc, const float* mix, const float* scale,
                               const float* base, const float* nw, float* pre, float* post,
                               float* comb, float* cur, float* norm, float* part, uint8_t* q8,
                               float* qs, uint32_t n_embd, uint32_t n_hc, uint32_t nb,
                               uint32_t iters, float hc_eps, float rms_eps, int has_control,
                               void* stream, const float* pre_in) {
  // Not a size the caller may vary: the arena is kNarrowTok columns wide and NB below is that
  // constant. A different width here would read past the plane it was told to reduce.
  if (nb != kNarrowTok || !part) {
    aff::ui::fatal("batch_hc_fused_narrow: nb=%u is not the %u-column arena\n", nb,
                 kNarrowTok);
    std::abort();
  }
  constexpr uint32_t NB = kNarrowTok;
  const hipStream_t st = (hipStream_t)stream;
  const dim3 g(kHcnChunks), t(kHcnThreads);
  // TWO partial planes. The expand kernel reads the first (every block re-derives the gates from
  // it) and writes the second, and one plane would let block 0's write race block 5's read.
  float* const part2 = part + (size_t)kHcnChunks * NB;
  if (has_control) {
    AFF_BLAUNCH((hcn_sq_kernel<NB>), g, t, 0, st, hc, part, (uint64_t)n_embd * n_hc);
#define AFF_HCN_X(NH)                                                                             \
  AFF_BLAUNCH((hcn_expand_kernel<NB, NH>), g, t, 0, st, hc, part, mix, scale, base, pre, post,     \
              comb, cur, part2, n_embd, (uint64_t)n_embd * n_hc, iters, hc_eps, rms_eps, pre_in)
    if (n_hc == 1)      { AFF_HCN_X(1); }
    else if (n_hc == 2) { AFF_HCN_X(2); }
    else if (n_hc == 4) { AFF_HCN_X(4); }
    else if (n_hc == 8) { AFF_HCN_X(8); }
    else { aff::ui::fatal("batch_hc_fused_narrow: n_hc=%u\n", n_hc); std::abort(); }
#undef AFF_HCN_X
  } else {
    AFF_BLAUNCH((hcn_copy_kernel<NB>), g, t, 0, st, hc, cur, part2, n_embd);
  }
  // n_embd/kHcnThreads blocks, each owning 256 contiguous dims — see the note on the kernel.
  AFF_BLAUNCH((hcn_norm_kernel<NB>), dim3((n_embd + kHcnThreads - 1) / kHcnThreads), t, 0, st, cur,
              part2, nw, norm, q8, qs, n_embd, rms_eps, (float*)nullptr);
}

// The head's collapse: sum of squares, gate, weighted lane sum, rms_norm. Three launches and no
// host in the loop, where this was `nb` device-to-host copies of a [n_hc][n_embd] plane with a
// stream drain behind each one and 4 x 16384 bf16 of matvec on the CPU between them.
void batch_hc_head_narrow_hip(const uint16_t* hc, const float* mix, const float* scale,
                              const float* base, const float* nw, float* cur, float* norm,
                              float* norm_tm, float* part, uint32_t n_embd, uint32_t n_hc,
                              uint32_t nb, float hc_eps, float rms_eps, void* stream, const float* pre_in) {
  if (nb != kNarrowTok || !part) {
    aff::ui::fatal("batch_hc_head_narrow: nb=%u is not the %u-column arena\n", nb,
                 kNarrowTok);
    std::abort();
  }
  constexpr uint32_t NB = kNarrowTok;
  const hipStream_t st = (hipStream_t)stream;
  const dim3 g(kHcnChunks), t(kHcnThreads);
  float* const part2 = part + (size_t)kHcnChunks * NB;   // two planes, for the reason above
  AFF_BLAUNCH((hcn_sq_kernel<NB>), g, t, 0, st, hc, part, (uint64_t)n_embd * n_hc);
#define AFF_HCH_X(NH)                                                                             \
  AFF_BLAUNCH((hcn_head_gate_kernel<NB, NH>), g, t, 0, st, hc, part, mix, scale, base, cur, part2, \
              n_embd, (uint64_t)n_embd * n_hc, hc_eps, rms_eps, pre_in)
  if (n_hc == 1)      { AFF_HCH_X(1); }
  else if (n_hc == 2) { AFF_HCH_X(2); }
  else if (n_hc == 4) { AFF_HCH_X(4); }
  else if (n_hc == 8) { AFF_HCH_X(8); }
  else { aff::ui::fatal("batch_hc_head_narrow: n_hc=%u\n", n_hc); std::abort(); }
#undef AFF_HCH_X
  AFF_BLAUNCH((hcn_norm_kernel<NB>), dim3((n_embd + kHcnThreads - 1) / kHcnThreads), t, 0, st, cur,
              part2, nw, norm, (uint8_t*)nullptr, (float*)nullptr, n_embd, rms_eps, norm_tm);
}

void batch_hc_expand_hip(uint16_t* out_hc, float* block, const uint16_t* hc,
                         const float* post,
                         const float* comb, uint32_t n_embd, uint32_t n_hc, uint32_t nb,
                         bool zero_block, void* stream) {
  const uint64_t total = (uint64_t)n_embd * nb;
  const uint32_t thr = 256;
  // n_hc is a TEMPLATE argument: the runtime-n_hc kernel it replaced re-read hc once per
  // destination and could not unroll either loop. Bit-identical, so the switch is the difference.
  const dim3 g((uint32_t)((total + thr - 1) / thr)), t3(thr);
  const int zb = zero_block ? 1 : 0;
#define AFF_BXP(NH)                                                                              \
  AFF_BLAUNCH(HIP_KERNEL_NAME(batch_hc_expand_reg_kernel<NH>), g, t3, 0, (hipStream_t)stream,    \
              out_hc, block, hc, post, comb, n_embd, nb, zb)
  if (n_hc == 1)      { AFF_BXP(1); }
  else if (n_hc == 2) { AFF_BXP(2); }
  else if (n_hc == 8) { AFF_BXP(8); }
  else                { AFF_BXP(4); }
#undef AFF_BXP
}

void batch_hc_add_hip(float* dst, const float* src, uint64_t n, void* stream) {
  const uint32_t thr = 256;
  AFF_BLAUNCH(batch_hc_add_kernel, dim3((uint32_t)((n + thr - 1) / thr)), dim3(thr), 0,
                     (hipStream_t)stream, dst, src, n);
}

void batch_hc_add_bf16_hip(uint16_t* dst, float* src, uint64_t n, bool zero_src, void* stream) {
  const uint32_t thr = 256;
  AFF_BLAUNCH(batch_hc_add_bf16_kernel, dim3((uint32_t)((n + thr - 1) / thr)), dim3(thr), 0,
              (hipStream_t)stream, dst, src, n, zero_src ? 1 : 0);
}

void batch_hc_seed_hip(const float* emb, uint16_t* hc, uint32_t n_embd, uint32_t n_hc, uint32_t nb,
                       void* stream) {
  const uint64_t total = (uint64_t)n_embd * nb;
  const uint32_t thr = 256;
  AFF_BLAUNCH(batch_hc_seed_kernel, dim3((uint32_t)((total + thr - 1) / thr)), dim3(thr), 0,
                     (hipStream_t)stream, emb, hc, n_embd, n_hc, nb);
}

// ---- the router's top-k ------------------------------------------------------------------------
//
// One block per token, one thread per expert. Six rounds of a block-wide argmax rather than a sort:
// k is 6 of 256 and a partial sort would need LDS for the whole permutation.
constexpr uint32_t kTopKMax = 8;

__device__ __forceinline__ bool topk_better(float va, uint32_t ia, float vb, uint32_t ib) {
  // route_topk's comparator exactly: value descending, then expert id ASCENDING. The tie-break is
  // not cosmetic — with 256 near-tied scores it decides which 9 MB of weights the token reads.
  if (va != vb) return va > vb;
  return ia < ib;
}

__global__ __launch_bounds__(256) void router_topk_kernel(
    const float* __restrict__ logits, const float* __restrict__ bias,
    const uint8_t* __restrict__ resident, uint32_t* __restrict__ sel, float* __restrict__ wt,
    uint32_t* __restrict__ nsel, uint32_t n_expert, uint32_t nb, uint32_t k, int norm_topk_prob,
    float routed_scaling) {

  extern __shared__ float s_rt[];                 // [n_expert] score, [n_expert] selection value
  float* s_score = s_rt;
  float* s_sel   = s_rt + n_expert;
  __shared__ float r_val[256];
  __shared__ uint32_t r_idx[256];
  __shared__ uint32_t s_best[kTopKMax];
  __shared__ float s_bscore[kTopKMax];

  const uint32_t b = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const float NEG = -3.0e38f;

  for (uint32_t e = tid; e < n_expert; e += blockDim.x) {
    const float x = logits[(size_t)e * nb + b];
    // sqrtsoftplus, with the same large-x guard the host uses. log1pf/expf, not the fast intrinsic:
    // this feeds a discrete choice.
    const float sp = (x > 20.0f) ? x : log1pf(expf(x));
    const float sc = sqrtf(sp);
    s_score[e] = sc;
    const bool live = !resident || resident[e];
    s_sel[e] = live ? sc + (bias ? bias[e] : 0.0f) : NEG;
  }
  __syncthreads();

  uint32_t got = 0;
  for (uint32_t r = 0; r < k; ++r) {
    float bv = NEG; uint32_t bi = 0xFFFFFFFFu;
    for (uint32_t e = tid; e < n_expert; e += blockDim.x)
      if (s_sel[e] != NEG && topk_better(s_sel[e], e, bv, bi)) { bv = s_sel[e]; bi = e; }
    r_val[tid] = bv; r_idx[tid] = bi;
    __syncthreads();
    for (uint32_t o = blockDim.x >> 1; o; o >>= 1) {
      if (tid < o && r_idx[tid + o] != 0xFFFFFFFFu &&
          topk_better(r_val[tid + o], r_idx[tid + o], r_val[tid], r_idx[tid]))
        { r_val[tid] = r_val[tid + o]; r_idx[tid] = r_idx[tid + o]; }
      __syncthreads();
    }
    if (r_idx[0] == 0xFFFFFFFFu) break;           // fewer resident experts than the layer selects
    if (tid == 0) {
      s_best[r] = r_idx[0];
      s_bscore[r] = s_score[r_idx[0]];
      s_sel[r_idx[0]] = NEG;                      // taken; do not win again
    }
    ++got;
    __syncthreads();
  }

  if (tid == 0) {
    double sum = 0.0;
    for (uint32_t i = 0; i < got; ++i) sum += s_bscore[i];
    if (sum < 6.103515625e-5) sum = 6.103515625e-5;   // fp16 min-normal floor, as in the reference
    const double inv = norm_topk_prob ? 1.0 / sum : 1.0;
    for (uint32_t i = 0; i < got; ++i) {
      sel[(size_t)b * k + i] = s_best[i];
      wt[(size_t)b * k + i] = (float)(s_bscore[i] * inv * routed_scaling);
    }
    nsel[b] = got;
  }
}

// One block a token, one thread doing the arithmetic: k is at most 8 and the logits are already
// computed, so this is a handful of loads and a divide. It exists to delete a host loop, not to be
// fast.
//
// TRANSCRIBED FROM THE HOST HASH PATH, not from router_topk_kernel's tail. The two differ: the
// top-k sums its scores in DOUBLE and multiplies by a reciprocal, where the hash path sums in
// FLOAT and DIVIDES. Following the wrong one would put a needless ulp between this and the
// reference for no reason at all — the dot product is already the one thing that has to change.
__global__ __launch_bounds__(64) void router_hash_wt_kernel(
    const float* __restrict__ logits, const uint32_t* __restrict__ sel,
    const uint32_t* __restrict__ nsel, float* __restrict__ wt, uint32_t nb, uint32_t k,
    int norm_topk_prob, float routed_scaling) {
  const uint32_t b = blockIdx.x;
  if (threadIdx.x != 0) return;
  uint32_t got = nsel[b];
  if (got > k) got = k;
  float sc[kTopKMax];
  float sum = 0.0f;
  for (uint32_t i = 0; i < got; ++i) {
    const float x = logits[(size_t)sel[(size_t)b * k + i] * nb + b];
    const float sp = (x > 20.0f) ? x : log1pf(expf(x));
    sc[i] = sqrtf(sp);
    sum += sc[i];
  }
  if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;   // fp16 min-normal floor, as in the reference
  for (uint32_t i = 0; i < got; ++i) {
    float v = sc[i];
    if (norm_topk_prob) v /= sum;
    wt[(size_t)b * k + i] = v * routed_scaling;
  }
}

void router_hash_wt_hip(const float* logits, const uint32_t* sel, const uint32_t* nsel, float* wt,
                        uint32_t nb, uint32_t live, uint32_t k, int norm_topk_prob,
                        float routed_scaling, void* stream) {
  if (!live || !k || k > kTopKMax) return;
  hipLaunchKernelGGL(router_hash_wt_kernel, dim3(live), dim3(64), 0, (hipStream_t)stream, logits,
                     sel, nsel, wt, nb, k, norm_topk_prob, routed_scaling);
  AFF_HIP_CHECK(hipGetLastError());
}

void router_topk_hip(const float* logits, const float* bias, const uint8_t* resident,
                     uint32_t* sel, float* wt, uint32_t* nsel, uint32_t n_expert, uint32_t nb,
                     uint32_t live, uint32_t k, int norm_topk_prob, float routed_scaling,
                     void* stream) {
  if (!live || !n_expert || !k || k > kTopKMax) return;
  const size_t lds = (size_t)n_expert * 2 * sizeof(float);
  hipLaunchKernelGGL(router_topk_kernel, dim3(live), dim3(256), lds, (hipStream_t)stream, logits,
                     bias, resident, sel, wt, nsel, n_expert, nb, k, norm_topk_prob,
                     routed_scaling);
  AFF_HIP_CHECK(hipGetLastError());
}


void batch_column_bf16_hip(const uint16_t* src, float* dst, uint32_t n, uint32_t nb, uint32_t b,
                           void* stream) {
  const uint32_t thr = 256;
  AFF_BLAUNCH(batch_column_bf16_kernel, dim3((n + thr - 1) / thr), dim3(thr), 0,
                     (hipStream_t)stream, src, dst, n, nb, b);
}

void batch_hc_tap_hip(const uint16_t* hc, float* tap, uint32_t n_embd, uint32_t n_hc, uint32_t nb,
                      uint32_t b_lo, uint32_t n_keep, uint32_t row0, uint32_t rows_cap,
                      uint32_t row_stride, uint32_t slot_off, void* stream) {
  if (!n_keep || row0 >= rows_cap) return;
  const uint64_t total = (uint64_t)n_embd * n_keep;
  const uint32_t thr = 256;
  AFF_BLAUNCH(batch_hc_tap_kernel, dim3((uint32_t)((total + thr - 1) / thr)), dim3(thr), 0,
              (hipStream_t)stream, hc, tap, n_embd, n_hc, nb, b_lo, n_keep, row0, rows_cap,
              row_stride, slot_off);
}

void batch_argmax_hip(const float* y, float* out, uint32_t rows, uint32_t nb, uint32_t live,
                      uint32_t row0, void* stream, int y_token_major) {
  if (!rows || !live) return;
  AFF_BLAUNCH(batch_argmax_kernel, dim3(live), dim3(kArgmaxThr), 0, (hipStream_t)stream, y, out,
              rows, nb, row0, y_token_major ? 1u : nb, y_token_major ? rows : 1u);
}

void markov_step_hip(const uint16_t* w1, const uint16_t* w2, const uint32_t* ids,
                     const float* logits, uint64_t* best, uint32_t rows, uint32_t row0,
                     uint32_t rank, uint32_t nb, uint32_t col, void* stream,
                     int logits_token_major) {
  if (!rows || !best || rank > kMarkovRankMax) return;
  const uint32_t l_rs = logits_token_major ? 1u : nb;
  const uint32_t l_cs = logits_token_major ? rows : 1u;
  // The grid-wide maximum lands here; it has to start at "nothing", which is 0 because markov_key
  // maps every finite logit to a strictly positive key.
  auto* b64 = (unsigned long long*)best;
  AFF_HIP_CHECK(hipMemsetAsync(b64, 0, sizeof(unsigned long long), (hipStream_t)stream));
  AFF_BLAUNCH(markov_step_kernel,
              dim3((rows + (kMarkovThr / kWave) - 1) / (kMarkovThr / kWave)), dim3(kMarkovThr), 0,
              (hipStream_t)stream, w1, w2, ids, logits, b64, rows, row0, rank, nb, col, l_rs,
              l_cs);
}

void markov_combine_hip(const uint64_t* own, const uint64_t* peers, uint32_t npeer, uint32_t* ids,
                        uint32_t col, void* stream) {
  AFF_BLAUNCH(markov_combine_kernel, dim3(1), dim3(64), 0, (hipStream_t)stream,
              (const unsigned long long*)own, (const unsigned long long*)peers, npeer, ids, col);
}

void batch_head_combine_hip(const float* own, const float* peers, uint32_t npeer, uint32_t* ids,
                            uint32_t n, uint32_t nb, void* stream) {
  AFF_BLAUNCH(batch_head_combine_kernel, dim3(1), dim3(kNarrowTok), 0, (hipStream_t)stream, own,
              peers, npeer, ids, n, nb);
}


} // namespace aff
