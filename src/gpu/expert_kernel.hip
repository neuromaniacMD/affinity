#include "gpu/expert_kernel_hip.h"
#include "gpu/hip_common.h"
#include "gpu/device_math.h"
#include "gpu/wmma.h"
#include "format/aff_format.h"
#include "ui/log.h"
#include <hip/hip_cooperative_groups.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace aff {

// Defined near expert_planes_are_colgroup_major, at the bottom with the other host entry points.
static inline void aff_tr_guard(const char* who);

namespace {

// A pointer loaded out of a struct in device memory is generic, so it compiles to `flat_*`: no
// SGPR-base addressing, and a wait on the LDS counter as well as the memory one. Casting to
// address_space(1) asserts what is already true — these never point at LDS. `__restrict__` cannot
// do this; it promises non-aliasing, not an address space.
//
// Device pass only: on the host `uint4` is a class whose assignment cannot bind an addrspace(1)
// object, and the qualifier means nothing there anyway.
#ifdef __HIP_DEVICE_COMPILE__
#define AFF_GLOBAL __attribute__((address_space(1)))
#define AFF_CONST  __attribute__((address_space(4)))
#else
#define AFF_GLOBAL
#define AFF_CONST
#endif
template <class T>
__device__ __forceinline__ const AFF_GLOBAL T* as_global(const T* p) {
  return (const AFF_GLOBAL T*)p;
}

// address_space(4) is CONSTANT; on AMDGPU it shares the flat aperture with global, so the cast is
// free. It promises the load is invariant for the dispatch, which lets a wave-uniform address become
// one `s_load_*` through the scalar cache instead of 32 lanes of `global_load_*`.
//
// Only sound when nothing in the same dispatch writes those bytes. Every use here reads an
// activation a previous kernel produced, and a kernel boundary invalidates K$.
template <class T>
__device__ __forceinline__ const AFF_CONST T* as_const(const T* p) {
  return (const AFF_CONST T*)p;
}

// `threadIdx.x / kWave` is constant across a wave, but LLVM's divergence analysis does not know the
// wave size and calls the quotient divergent, so everything derived from it lands in VGPRs.
// readfirstlane asserts what is true for one instruction, after which the loop bounds and the
// activation address are scalar.
__device__ __forceinline__ uint32_t wave_uniform(uint32_t v) {
#ifdef __HIP_DEVICE_COMPILE__
  return (uint32_t)__builtin_amdgcn_readfirstlane((int)v);
#else
  return v;
#endif
}

// Work decomposition: one wave per output row, each lane owning a contiguous run of 16-weight
// blocks. Splitting by block rather than by column keeps every lane's start byte-aligned, since a
// lane's codes span bpl*40 bits (= bpl*5 bytes) and its scales bpl*6 bits. Both cols=4096 and
// cols=2048 land on byte boundaries, so no lane pays a cross-byte shift chain to find its data, and
// the wave's 32 lanes sweep one row in a single contiguous span.

// Pointer tables travel as kernel arguments rather than through device memory: a table in VRAM would
// need an allocation, a synchronous copy and a sync on the critical path, which the decode path
// cannot afford at fourteen small launches an FFN. A full batch is a few hundred bytes, comfortably
// inside the kernel argument budget.
constexpr int kMaxBatch = 32;
struct BatchPtrs {
  const uint8_t*  data[kMaxBatch];
  const uint8_t*  scales[kMaxBatch];
  const uint16_t* rscales[kMaxBatch];
  const float*    xs[kMaxBatch];
  float           w[kMaxBatch];
  // Only the SPLIT-output form uses this: one destination a matrix, rather than every matrix
  // accumulating into one y. See expert_gemv_split_hip.
  float*          ys[kMaxBatch];
};

constexpr int kWavesPerBlock = 8;                       // 8 waves x 32 lanes = 256 threads
constexpr int kThreads = kWavesPerBlock * kWave;
constexpr int kMaxCols = 4096;                          // LDS staging for the activation

// Truncation, which for a value that came out of an f32 accumulator is what every other bf16 store
// in this engine does — see wmma::f32_to_bf16.
__device__ inline uint16_t f32_to_bf16_dev(float f) {
  uint32_t u; __builtin_memcpy(&u, &f, 4);
  return (uint16_t)(u >> 16);
}

__device__ inline float bf16_to_f32_dev(uint16_t h) {
  const uint32_t u = (uint32_t)h << 16;
  return __uint_as_float(u);
}

// Sums `v` across a wave32. Explicit width — a width-less shuffle silently means "whatever the
// wave is", and this kernel is only correct at 32.
__device__ inline float wave_reduce_add(float v) {
  #pragma unroll
  for (int off = kWave / 2; off > 0; off >>= 1) v += __shfl_xor(v, off, kWave);
  return v;
}

// SwiGLU over EVERY expert's intermediate in one launch. The scratch layout is
// [expert][gate|up][inter], so gate for expert k sits at 2*k*inter and up at (2*k+1)*inter — a
// single flat index maps onto it, and n_experts launches collapse to one.
__global__ void swiglu_all_kernel(float* __restrict__ scratch, uint32_t n_experts,
                                  uint32_t inter, float limit) {
  const uint32_t total = n_experts * inter;
  for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < total;
       t += gridDim.x * blockDim.x) {
    const uint32_t k = t / inter, i = t % inter;
    float* g = scratch + (size_t)(2 * k) * inter + i;
    const float* u = scratch + (size_t)(2 * k + 1) * inter + i;
    float gv = *g, uv = *u;
    if (limit > 0.0f) {
      gv = fminf(gv, limit);
      uv = fminf(limit, fmaxf(-limit, uv));
    }
    *g = (gv / (1.0f + expf(-gv))) * uv;
  }
}

// A lane's run is contiguous and byte-aligned by construction, so it is pulled in whole rather than
// assembled from byte loads, and extraction happens in registers. The 40-bit code window never
// straddles three dwords: bit offset b*40 is 0, 8, 16 or 24 mod 32, and 64 - 24 = 40.
constexpr int kMaxCodeWords = 10;               // 40 bytes, the cols=4096 case
constexpr int kMaxFieldBytes = 6;

// Four independent partial sums rather than one: a single accumulator makes every FMA wait on the
// previous, and this kernel is far from both the bandwidth and the fp32 wall, so that serial
// dependency is the limit. Costs a ~1e-7 change in summation order.

// Coalesced staging. Lane l owns blocks [l*bpl, (l+1)*bpl), so reading them directly gives 32 lanes
// a 40-byte stride and the wave touches a whole row to consume a tenth of it. Separating how the row
// is READ from who OWNS which block fixes that: lane l takes dword l, l+32, l+64 into LDS, then
// reads its own blocks back out. `x` moves out of LDS to pay for it — every wave reads it, so it
// stays hot in L1.

// The batched form: one expert, T token columns, one pass over the weights. A GEMV called once per
// token re-reads each selected expert tens of times over a prefill chunk; this reads it once.
//
// The lane mapping stays the GEMV's, because that is what makes the code loads coalesce: lane =
// column slice, but each lane carries T accumulators and each decoded weight multiplies into all T.
// It amortises the DECODE, not the load.
//
// `x` is [cols][T] dim-major and compact, so the caller gathers the expert's token columns out of
// the chunk arena first; that gather is small against the weight re-reads it removes. `y` is
// [rows][T] and accumulated into, so several experts can share an output tile. blockIdx.y selects
// one of up to two matrices sharing `x` — gate and up are exactly that pair.
struct GemmJob {
  const uint8_t* data;
  const uint8_t* fields;
  const uint16_t* rscales;
  float* y;
};

template <int T>
__global__ __launch_bounds__(kThreads) void expert_gemm_tokens_kernel(
    GemmJob j0, GemmJob j1, const float* __restrict__ cb,
    const float* __restrict__ x, float wscale, uint32_t rows, uint32_t cols) {

  const GemmJob jb = blockIdx.y ? j1 : j0;
  const uint8_t* __restrict__ data = jb.data;
  const uint8_t* __restrict__ fields = jb.fields;
  const uint16_t* __restrict__ rscales = jb.rscales;
  float* __restrict__ y = jb.y;

  __shared__ float s_cb[4 * 32 * 2];
  __shared__ uint32_t s_code[kWavesPerBlock][320];
  __shared__ uint32_t s_field[kWavesPerBlock][48];

  const int tid  = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;

  for (int i = tid; i < 4 * 32 * 2; i += kThreads) s_cb[i] = cb[i];
  __syncthreads();

  const uint32_t nblk = cols / 16;
  const uint32_t bpl  = nblk / kWave;
  const uint32_t code_words_row  = cols / 2 * 5 / 8 / 4;
  const uint32_t field_words_row = nblk * 6 / 8 / 4;
  const uint32_t code_bytes_lane = bpl * 5;
  const uint32_t field_bytes_lane = bpl * 6 / 8;

  for (uint32_t row = blockIdx.x * kWavesPerBlock + wave; row < rows;
       row += gridDim.x * kWavesPerBlock) {

    float acc[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0.0f;

    const float rs = bf16_to_f32_dev(rscales[row]) * wscale;
    if (rs != 0.0f) {
      const uint32_t* crow = (const uint32_t*)data + (size_t)row * code_words_row;
      const uint32_t* frow = (const uint32_t*)fields + (size_t)row * field_words_row;
      for (uint32_t i = lane; i < code_words_row; i += kWave) s_code[wave][i] = crow[i];
      for (uint32_t i = lane; i < field_words_row; i += kWave) s_field[wave][i] = frow[i];
      __builtin_amdgcn_s_waitcnt(0);
      __builtin_amdgcn_wave_barrier();

      const uint8_t* cb_lane = (const uint8_t*)s_code[wave] + lane * code_bytes_lane;
      const uint8_t* fb_lane = (const uint8_t*)s_field[wave] + lane * field_bytes_lane;

      for (uint32_t b = 0; b < bpl; ++b) {
        const uint32_t fbit = b * 6;
        const uint32_t fw = (uint32_t)fb_lane[fbit >> 3] | ((uint32_t)fb_lane[(fbit >> 3) + 1] << 8);
        const uint32_t field = (fw >> (fbit & 7)) & 63u;
        const float bs = rs * (float)(field & 15u);
        if (bs == 0.0f) continue;
        const float* lut = s_cb + (size_t)(field >> 4) * 64;

        const uint32_t cbit = b * 40;
        uint64_t win = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) win |= (uint64_t)cb_lane[(cbit >> 3) + i] << (8 * i);
        win >>= (cbit & 7);

        // The 16 columns this block covers, as a base row index into the [cols][T] tile.
        const uint32_t c0col = (lane * bpl + b) * 16;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
          const uint32_t code = (uint32_t)((win >> (5 * i)) & 31u);
          const float w0 = lut[code * 2 + 0] * bs;
          const float w1 = lut[code * 2 + 1] * bs;
          const float* xa = x + (size_t)(c0col + i * 2 + 0) * T;
          const float* xb2 = x + (size_t)(c0col + i * 2 + 1) * T;
          #pragma unroll
          for (int t = 0; t < T; ++t) acc[t] = fmaf(w0, xa[t], acc[t]);
          #pragma unroll
          for (int t = 0; t < T; ++t) acc[t] = fmaf(w1, xb2[t], acc[t]);
        }
      }
      __builtin_amdgcn_wave_barrier();
    }

    // One reduction per token instead of one per row. T of them, all wave-wide -- this is the cost
    // the "lane = token" mapping in batch_kernels.hip avoids, and it is affordable here only
    // because T columns share the whole decode above.
    //
    // STORE, not accumulate. Every tile belongs to exactly one expert for the length of one call,
    // so there is nothing to accumulate onto -- and the tiles then need no zeroing pass, which is
    // three launches and 768 KB of writes an expert saved.
    #pragma unroll
    for (int t = 0; t < T; ++t) {
      const float total = wave_reduce_add(acc[t]);
      if (lane == 0) y[(size_t)row * T + t] = total;
    }
  }
}


// ---- the W8A8 expert GEMM ---------------------------------------------------------------------
//
// The scalar kernel was activation bound, not weight bound: one wave owned one output row and its
// lanes re-read the whole [cols][T] tile per row. A 16x16 WMMA tile reuses each activation sixteen
// times.
//
// Three scales sit on a weight: a 4-bit scale per 16-column block, a codebook variant per the same
// block, and a bf16 rscale per row. Left as they are, the block scale varies per row and every 16
// columns, forcing an accumulator flush at every WMMA K-step. Expanding the codebook to absorb the
// 4-bit scale removes that:
//
//   s_lut[(variant*16 + scale4)*32 + code]  ->  two E4M3 bytes, the pair that code addresses
//
// Decoding a pair is then one LDS lookup and no arithmetic. The row rscale is constant along K so it
// applies once in the epilogue, and only the activation scale folds at the 128-column boundary.
// E4M3 carries 4 significant bits, so the table adds error on top of the code; bench/expert_wmma.hip
// measures it against an exact CPU reference rather than assuming it is dominated.
//
// MT (M tiles a wave owns) is a template parameter because the accumulators cost 2*MT*NT*8 VGPRs.
// More waves a SIMD is not automatically better: halving MT doubles the block count, and every block
// rebuilds the LUT and re-stages the activation tile. That per-block overhead can outweigh the
// occupancy. MT=2 also gives a wave two independent WMMA chains, which matters most at nt=16 where
// there is only one N tile.
//
// RDNA4 does not co-issue VALU with WMMA, so the code extraction and LUT loads are serial with the
// matrix ops and dominate the K loop. Staging decoded granules through `d[][]` before storing is
// deliberate — it lets all the LUT loads issue before any is consumed. Everything except the decode
// is shared between the formats below, so an A/B measures the decode and nothing else.
enum : uint32_t {
  kFmtPair16 = 0,   // shipped: 5-bit pair codes, 4 variants x 16 integer scales folded in (4 KB)
  kFmtQuad4  = 4,   // 10-bit quad codes, 4 variants (16 KB)
  kFmtQuad8  = 8,   // 10-bit quad codes, 8 variants (32 KB)
  // 10-bit quad codes, 4 variants, and the field's sixth bit negates the block. Three scale bits
  // plus two variant bits is five of six; spending the spare bit on a sign buys an extra bit of
  // codebook, since a symmetric fit otherwise stores the negative of half its entries. Costs one
  // `^` on the dword the scale already adds to. kFmtQuad4 is its no-sign control.
  kFmtQuad4S = 5,
  // The LUT competes with the activation staging for the WGP's 128 KB of LDS, so a wider table costs
  // blocks per WGP. Two variants is the largest table that still fits four blocks, which is why it
  // is the one that gets the decode saving without paying occupancy for it. The field bit the second
  // variant no longer needs goes to ROT2 below.
  kFmtQuad2S = 6,   // 10-bit quads, 2 variants, sign, and a 2-byte rotate. 8 KB.
};
constexpr bool     fmt_is_quad(uint32_t f) {
  return f == kFmtQuad4 || f == kFmtQuad8 || f == kFmtQuad4S || f == kFmtQuad2S;
}
constexpr uint32_t fmt_variants(uint32_t f) {
  return f == kFmtQuad4S ? 4u : f == kFmtQuad2S ? 2u : f;
}
constexpr bool     fmt_has_sign(uint32_t f) { return f == kFmtQuad4S || f == kFmtQuad2S; }
// The sixth field bit, only where the variant does not need it: it rotates the decoded quad by two
// bytes, so (a,b,c,d) also spells (c,d,a,b). Weights within a row have no positional structure, so a
// codebook fitted to them holds both orderings anyway; this bit reclaims one of the two copies for a
// single v_alignbit and no extra LDS.
constexpr bool     fmt_has_rot2(uint32_t f) { return f == kFmtQuad2S; }
constexpr uint32_t kEWaves = 4, kEThreads = kEWaves * kWave;
constexpr uint32_t kEK = 64;           // columns staged per iteration = bytes per LDS row

// Single-buffer the A stage to cross an occupancy step. Compile-time so the two arms differ in the
// kernel and nowhere else; see the note at the LDS declarations for why it is legal and what it
// costs. -DAFF_ESINGLE_A=0 to measure the other arm.
#ifndef AFF_ESINGLE_A
#define AFF_ESINGLE_A 1
#endif
constexpr bool kSingleAWanted = AFF_ESINGLE_A != 0;
// LDS a WGP holds, so blocks a WGP = kWgpLds / bytes a block. 128 KiB is RDNA's, and this has to be
// a literal because the occupancy predicate below it is constexpr — a device query cannot pick a
// template arm. An architecture with a different LDS budget needs this changed by hand.
constexpr uint32_t kWgpLds = 128u * 1024u;

__device__ __forceinline__ uint32_t e4m3_pack2(float a, float b) {
  int p = 0;
  p = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, p, false);
  return (uint32_t)p & 0xffffu;
}

// NOT `__restrict__` on `lds`, and that is load-bearing under kSingleA. A noalias scope on the LDS
// pointer licenses hoisting a later store to s_a above these loads, and under kSingleA the next
// half's staging is exactly such a store to the rows these loads read. Dropping the qualifier denies
// that reorder while still letting the compiler hoist stage_a's global loads above the MMA — a
// different address space, so no aliasing question — which is the prefetch the single-buffered arm
// depends on. A memory-clobber asm would block both.
__device__ __forceinline__ wmma::i2v load_efrag(const uint8_t* lds, uint32_t row0,
                                                uint32_t k0, uint32_t lane) {
  const uint32_t row = row0 + wmma::frag_c(lane);
  const uint32_t k = k0 + wmma::frag_k0(lane);
  const uint2 v = *(const uint2*)(lds + row * kEK + ((((k >> 3) ^ (row & 7u))) << 3));
  wmma::i2v f;
  f[0] = (int)v.x;
  f[1] = (int)v.y;
  return f;
}

// One block per (row tile, expert). One expert at kEM=128 is only a handful of blocks, which leaves
// most of the card idle, so batching the experts into one launch is not a launch-count optimisation
// — it is what fills the grid.
//
// TR — the weight planes are colgroup-major, `[c0/kEK][row][bytes]` instead of row-major
// `[row][block]`. Same bytes, only the order differs. Row-major puts a thread's code bytes a whole
// row stride from its neighbour's, so a wave asks for 32 scattered partial cache lines and gets a
// fraction of streaming bandwidth. Colgroup-major makes those spans adjacent, and the read then
// lands on the plain-stream ceiling. Widening the scattered read instead does not close the gap:
// the request size was never the problem, the addressing was.
template <uint32_t NT, uint32_t MT, uint32_t FMT = kFmtPair16, bool TR = false>
__global__ __launch_bounds__(kEThreads) void expert_gemm_wmma_kernel(
    const ExpertTile* __restrict__ tiles, const uint8_t* __restrict__ x8,
    const float* __restrict__ xs, const float* __restrict__ cb,
    float wscale, uint32_t rows, uint32_t cols, int y_bf16) {
  constexpr uint32_t kEM = kEWaves * MT * 16;     // weight rows a block owns
  // Threads that cooperate on ONE row of the weight, and how many 16-column blocks each decodes.
  // At MT=2 the block owns 128 rows and there is one thread per row decoding all four; at MT=1 it
  // owns 64 and two threads split them. Both cover the same kEK columns.
  constexpr uint32_t kETPR = kEThreads / kEM;
  constexpr uint32_t kEBPT = (kEK / 16) / kETPR;
  static_assert(kEThreads % kEM == 0 && (kEK / 16) % kETPR == 0, "A staging does not divide");
  const ExpertTile tile = tiles[blockIdx.y];
  // AN EMPTY TILE IS A PAD, AND IT COSTS A RETURN.
  //
  // The grid's y extent is the number of tiles, which is a fact about ROUTING — how many distinct
  // experts a dispatch touched. A host that has not seen the routing cannot know it, so the
  // device-built dispatch launches the worst case and fills the surplus with `nt = 0`. Cheaper than
  // it looks: falling through would rebuild the whole LDS codebook and walk every column to store
  // nothing, since the epilogue writes `t < nt` and there is no such `t`. So this is not a new
  // behaviour, only an early one — the padded tile was always a no-op, and this makes it a fast one.
  if (!tile.nt) return;
  const uint8_t* __restrict__ data = tile.data;
  const uint8_t* __restrict__ fields = tile.fields;
  const uint16_t* __restrict__ rscales = tile.rscales;
  const uint32_t* __restrict__ tok = tile.tok;
  void* __restrict__ y = tile.y;
  const uint32_t nt = tile.nt;
  const uint32_t yld = tile.ld ? tile.ld : nt;   // one shared arena, or this tile's own block
  // 2048 packed pairs, 256 packed pairs, or FMT tables of 1024 packed quads. Declared as dwords
  // whichever it is, so one array serves all three; the pair paths read it through a uint16_t*.
  constexpr uint32_t kLutW =
      FMT == kFmtPair16 ? 1024u : fmt_variants(FMT) * 1024u;
  // The codebook stays in LDS. RDNA4 cannot co-issue VALU and WMMA within a wave, so waves-per-SIMD
  // is the only thing that overlaps the decode with the matrix work, and LDS is what caps it. The
  // decode's instruction count is not the lever — across formats at identical traffic, throughput
  // tracks occupancy and not VALU slots.
  __shared__ uint32_t s_lut[kLutW];
  // s_b is double-buffered: with a single buffer the two __syncthreads() an iteration force every
  // wave to finish staging before any starts WMMA and vice versa, so the dequant and the matrix work
  // are phase-separated across the block. The only overlap available is between waves, and one
  // buffer forbids exactly that.
  //
  // That argument does not apply to s_a, which is a within-wave transpose in every instantiation:
  // thread `tid` decodes row `tid / kETPR`, so wave w writes rows [w*kWave/kETPR, +kWave/kETPR) and
  // reads [w*MT*16, +MT*16) — the same range at every NT/MT pair here. A wave never touches another
  // wave's A rows, so the barriers protect s_b alone and an s_a hazard needs only the compiler
  // keeping the store below the load. That is what `kSingleA` relies on.
  //
  // Occupancy probe, ships at 0: moves the LDS allocation and nothing else, so one instantiation can
  // be measured either side of an occupancy step. It hangs off a live array on purpose — as its own
  // `__shared__` written under an unreachable guard the store is dead and the array is elided, and
  // the allocation never moves. Verify with `.group_segment_fixed_size`, not the -Rpass remark,
  // which interleaves across instantiations.
  constexpr uint32_t kEPad = 0;
  constexpr uint32_t kASz = kEM * kEK, kBSz = NT * 16 * kEK;
  // kSingleA. Overwriting s_a is a write-after-read against this wave's own reads, so it needs only
  // the lgkmcnt the compiler already emits between may-aliasing LDS ops, not a barrier. It costs
  // ordering: stage_a for the next half moves after the mma_half that reads the current one, so that
  // half's weight loads issue behind the MMA rather than in front of it.
  //
  // Occupancy is a floor division, so the saved bytes are worth nothing unless they move the
  // quotient — while the ordering cost is paid either way. The predicate below enables the single
  // buffer only where it actually crosses a step.
  constexpr uint32_t kLdsRest = kLutW * 4u + 2u * kBSz + kEPad;
  constexpr bool kSingleA = kSingleAWanted &&
      kWgpLds / (kLdsRest + kASz) > kWgpLds / (kLdsRest + 2 * kASz);
  __shared__ uint8_t s_a_all[(kSingleA ? 1 : 2) * kASz];
  __shared__ uint8_t s_b_all[2 * kBSz + kEPad];

  const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
  const uint32_t m0 = blockIdx.x * kEM;
  const uint32_t code_bytes_row = cols / 2 * 5 / 8;
  const uint32_t field_bytes_row = cols / 16 * 6 / 8;
  const uint32_t sc_cols = cols >> 7;

  // The expanded codebook, strided so it does not depend on the block size. A fixed trip count here
  // was sized for 256 threads and silently built only half the table when the block became 128.
  if constexpr (fmt_is_quad(FMT)) {
    // Nothing to compute: the 4-wide codebook is stored as packed E4M3 quads, so this is a dword
    // copy of coalesced L2 reads and no VALU. Building the entries from floats instead would be
    // thousands of codebook loads and converts in every block, and per-block table cost is exactly
    // what the MT choice above turns on.
    const auto* cb4 = (const AFF_GLOBAL uint32_t*)cb;
    for (uint32_t e = tid; e < kLutW; e += kEThreads) s_lut[e] = cb4[e];
  } else {
    auto* lut2 = (uint16_t*)s_lut;
    for (uint32_t e = tid; e < 2048; e += kEThreads) {
      const uint32_t code = e & 31u, sc = (e >> 5) & 15u, var = e >> 9;
      const float f = (float)sc;
      lut2[e] = (uint16_t)e4m3_pack2(cb[var * 64 + code * 2 + 0] * f,
                                     cb[var * 64 + code * 2 + 1] * f);
    }
  }

  // `raw` is written before it is read on every iteration of the K loop below — the first WMMA of a
  // 128-column block produces it — so it needs no initialiser and gets none.
  wmma::f8v out[MT][NT];
  // Only the double-buffered arm carries a persistent `raw`; the kSingleA arm keeps its
  // accumulators local to a j-group inside mma_half, which is the whole point of the split.
  [[maybe_unused]] wmma::f8v raw[kSingleA ? 1 : MT][kSingleA ? 1 : NT];
#pragma unroll
  for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) out[i][j] = wmma::acc_zero();

  const uint32_t sm = tid / kETPR;                      // A: the row this thread decodes into
  const uint32_t sblk = (tid % kETPR) * kEBPT;          // its first 16-column block
  const uint32_t tok0 = wmma::frag_c(lane);

  // The activation scale row for each of this lane's NT token columns, hoisted out of the K loop.
  // `tok[t]` is loop-invariant, and reloading it inside the fold makes a dependent pair of global
  // loads — tok[t], then xs[tok[t]*sc_cols + kb] — so the fold stalls on a pointer chase every 128
  // columns. Dead columns (t >= nt) point at row 0 rather than branching: `stage` zeroes the B
  // fragment past `nt` so their `raw` is zero, and the epilogue does not store them.
  const AFF_GLOBAL float* xrow[NT];
#pragma unroll
  for (uint32_t j = 0; j < NT; ++j) {
    const uint32_t t = tok0 + j * 16;
    xrow[j] = as_global(xs) +
              (t < nt ? (size_t)(tok ? as_global(tok)[t] : t) * sc_cols : (size_t)0);
  }

  // Everything that fills one LDS buffer, so the loop can call it for the NEXT column block while
  // the WMMA below still reads the current one. A and B are SEPARATE lambdas because under kSingleA
  // they no longer belong at the same point in the loop: B still stages a block ahead, A cannot.
  // ---- THE WEIGHT PLANES ARE READ ONCE AND NEVER AGAIN --------------------------------------------
  //
  // grid.x partitions the rows and grid.y the tiles, so within a launch every (row, column group) of
  // every expert is staged by exactly one block exactly once. Nothing re-reads it — the next launch
  // is a different expert set, and by the time this one comes round again 43 layers have gone past.
  // So caching it is pure eviction: 86 MB a layer a card flowing through an 8 MB L2, displacing the
  // activation and the codebook, which ARE reused (every tile of the layer reads the same `x8`).
  //
  // Non-temporal on the CODE and FIELD planes only, and not on `x8`/`xs`/`cb`. Caching is not
  // arithmetic, so the output hash must not move across this change.
  auto ld_nt = [](const auto* p) { return __builtin_nontemporal_load(p); };
  auto stage_a = [&](uint32_t buf, uint32_t c0) {
    uint8_t* const s_a = s_a_all + (size_t)(kSingleA ? 0u : buf) * kASz;

    // ---- A: decode 32 weights (two 16-column blocks) a thread, straight into LDS as E4M3.
    {
      const uint32_t row = m0 + sm;
      uint32_t d[kEBPT * 2][2];
      if (row < rows) {
        const uint32_t b0 = (c0 >> 4) + sblk;
        // This thread's kEBPT*5 code bytes, and the field byte-group covering them.
        //
        // Under TR the whole column group's kEK/16 fields are ONE aligned 3-byte unit shared by the
        // kETPR threads that split the group, so `fsh0` picks out this thread's share rather than
        // the address doing it. That retires the straddle branch at MT=1 too: at kETPR=2 a thread's
        // two 6-bit fields start 12 bits into the group, which is not a byte, but it is a shift.
        const AFF_GLOBAL uint8_t* cp0;
        const AFF_GLOBAL uint8_t* fp0;
        uint32_t fsh0;
        if constexpr (TR) {
          const size_t g = (size_t)(c0 / kEK) * rows + row;
          cp0 = as_global(data + g * (kEK / 16 * 5) + (size_t)sblk * 5);
          fp0 = as_global(fields + g * (kEK / 16 * 6 / 8));
          fsh0 = sblk * 6;
        } else {
          cp0 = as_global(data + (size_t)row * code_bytes_row) + (size_t)b0 * 5;
          fp0 = as_global(fields + (size_t)row * field_bytes_row) + (size_t)b0 * 6 / 8;
          fsh0 = 0;
        }
        const auto* crow = as_global(data + (size_t)row * code_bytes_row);
        const auto* frow = as_global(fields + (size_t)row * field_bytes_row);

        // The two loads, widened. A thread's kEBPT blocks are consecutive and both planes are packed
        // with no per-block padding, so its share of each is one contiguous span rather than kEBPT
        // separate reads.
        //
        // Codes: 5*kEBPT contiguous bytes, dword-aligned — `crow` is 16-aligned and 5*b0 is a
        // multiple of 20 — so five dword loads. A block's 5 bytes never span more than two of them,
        // so its window is one 64-bit shift of an adjacent pair.
        //
        // Fields: b0 a multiple of 4 makes 6*b0 a multiple of 24 bits, so the group starts on a byte
        // and is exactly 3 bytes wide, which retires both the straddle branch and the one-byte plane
        // overrun it existed to avoid.
        //
        // NAMED SCALARS, NOT AN ARRAY. `uint32_t cw[5]` indexed by an unrolled `bb` looks like it
        // must fold and does not — it lands in scratch and costs a double-digit percentage of the
        // kernel. Check ScratchSize in `-Rpass-analysis=kernel-resource-usage` before changing this.
        // Under TR the same five dwords are still one aligned span, so the widened load survives the
        // reorder untouched.
        constexpr bool kWide = kETPR == 1 && kEBPT == 4;
        uint32_t w0 = 0, w1 = 0, w2 = 0, w3 = 0, w4 = 0, f24 = 0;
        if (kWide) {
          const auto* cp32 = (const AFF_GLOBAL uint32_t*)cp0;
          w0 = ld_nt(&cp32[0]); w1 = ld_nt(&cp32[1]); w2 = ld_nt(&cp32[2]);
          w3 = ld_nt(&cp32[3]); w4 = ld_nt(&cp32[4]);
        }
        // Row-major reaches this only when kWide made 6*b0 a multiple of 24; TR always can, because
        // the group's three bytes are a unit by construction.
        if (kWide || TR)
          f24 = (uint32_t)ld_nt(&fp0[0]) | ((uint32_t)ld_nt(&fp0[1]) << 8) |
                ((uint32_t)ld_nt(&fp0[2]) << 16);

#pragma unroll
        for (uint32_t bb = 0; bb < kEBPT; ++bb) {
          const uint32_t b = b0 + bb;
          uint32_t field;
          uint64_t win;                                 // 8 pairs = 40 bits at byte b*5
          if (kWide) {
            field = (f24 >> (fsh0 + bb * 6)) & 63u;
            // Bytes [5bb, 5bb+5) never span more than two dwords, and `bb` is a literal here.
            const uint32_t lo = bb == 0 ? w0 : bb == 1 ? w1 : bb == 2 ? w2 : w3;
            const uint32_t hi = bb == 0 ? w1 : bb == 1 ? w2 : bb == 2 ? w3 : w4;
            win = ((((uint64_t)hi) << 32) | lo) >> (((bb * 5) & 3u) * 8);
          } else if (TR) {
            // MT=1: a thread owns half a group, so its span starts 10 bytes in and is only 2-byte
            // aligned. Bytes, not dwords — but the field still comes out of the group's f24.
            field = (f24 >> (fsh0 + bb * 6)) & 63u;
            const auto* cp = cp0 + (size_t)bb * 5;
            win = 0;
#pragma unroll
            for (uint32_t i = 0; i < 5; ++i) win |= (uint64_t)ld_nt(&cp[i]) << (8 * i);
          } else {
            const uint32_t fb = b * 6;
            // The second byte is only needed when the 6-bit field straddles a byte, and reading it
            // unconditionally overruns the plane by one on the final block of the final row.
            const uint32_t fi = fb >> 3;
            const uint32_t fw = (uint32_t)frow[fi] |
                                ((fb & 7u) > 2u ? ((uint32_t)frow[fi + 1] << 8) : 0u);
            field = (fw >> (fb & 7)) & 63u;
            const auto* cp = crow + (size_t)b * 5;
            win = 0;
#pragma unroll
            for (uint32_t i = 0; i < 5; ++i) win |= (uint64_t)cp[i] << (8 * i);
          }
          // 3 variant bits pick the table, 3 scale bits become an exponent add. `sc * 0x08080808`
          // is `sc << 3` in each byte and cannot carry out of one: the largest is 7*8 = 56. It
          // lands on the PACKED dword, so it scales four weights at once whichever format built
          // them, and both pow2 paths pay exactly one v_add per four weights.
          const uint32_t add = (field & 7u) * 0x08080808u;
          if (fmt_is_quad(FMT)) {
            // Where the variant starts depends on whether the sign bit is between it and the scale.
            // The mask folds away at 8 variants, where a 6-bit field already makes `field >> 3`
            // three bits; it is here for the 4-variant instantiations.
            constexpr uint32_t kVSh = 3u + (fmt_has_sign(FMT) ? 1u : 0u) +
                                            (fmt_has_rot2(FMT) ? 1u : 0u);
            const uint32_t* tab = s_lut + ((field >> kVSh) & (fmt_variants(FMT) - 1u)) * 1024u;
            // The sign rides the same dword as the scale and cannot collide with it: the add lands
            // in bits 3-6 of each byte (exponent <= 7 by construction of the table, plus at most 7)
            // and this lands in bit 7. `(field & 8) * 0x10101010` is exactly 0x80808080 or nothing,
            // with no carry to reason about, and the compiler folds it to one v_mul_lo_u32.
            const uint32_t neg = fmt_has_sign(FMT) ? (field & 8u) * 0x10101010u : 0u;
            // A 2-byte rotate is a funnel shift of the dword with itself, and it commutes with both
            // the exponent add and the sign xor because all three act on whole bytes.
            const uint32_t rot = fmt_has_rot2(FMT) ? (field & 16u) : 0u;
#pragma unroll
            for (uint32_t i = 0; i < 4; ++i) {
              uint32_t q = (tab[(uint32_t)((win >> (10 * i)) & 1023u)] + add) ^ neg;
              if (fmt_has_rot2(FMT) && rot) q = (q >> 16) | (q << 16);
              d[bb * 2 + (i >> 1)][i & 1] = q;
            }
          } else {
            const uint16_t* tab = (const uint16_t*)s_lut + ((field >> 4) * 16u + (field & 15u)) * 32u;
#pragma unroll
            for (uint32_t i = 0; i < 4; ++i) {
              const uint32_t lo = tab[(uint32_t)((win >> (10 * i)) & 31u)];
              const uint32_t hi = tab[(uint32_t)((win >> (10 * i + 5)) & 31u)];
              d[bb * 2 + (i >> 1)][i & 1] = lo | (hi << 16);
            }
          }
        }
      } else {
#pragma unroll
        for (uint32_t i = 0; i < kEBPT * 2; ++i) { d[i][0] = 0; d[i][1] = 0; }
      }
#pragma unroll
      for (uint32_t g = 0; g < kEBPT * 2; ++g) {
        const uint32_t k = sblk * 16 + g * 8;
        *(uint2*)(s_a + sm * kEK + ((((k >> 3) ^ (sm & 7u))) << 3)) =
            make_uint2(d[g][0], d[g][1]);
      }
    }
  };

  auto stage_b = [&](uint32_t buf, uint32_t c0) {
    uint8_t* const s_b = s_b_all + (size_t)buf * kBSz;
    // ---- B: the activation, already E4M3 and token-major, so a pure copy. Strided over
    // (token, half) for the same reason the table build is: NT*16 tokens can exceed the block.
    for (uint32_t idx = tid; idx < NT * 16 * 2; idx += kEThreads) {
      const uint32_t sn = idx >> 1, sk = (idx & 1u) * 32;
      const bool live = sn < nt && c0 + sk + 32 <= cols;
      uint4 v = make_uint4(0, 0, 0, 0), w = make_uint4(0, 0, 0, 0);
      if (live) {
        // One indirection: an expert owns an arbitrary subset of the chunk's tokens.
        const auto* p = as_global(x8 + (size_t)(tok ? as_global(tok)[sn] : sn) * cols + c0 + sk);
        v = *(const AFF_GLOBAL uint4*)p;
        w = *(const AFF_GLOBAL uint4*)(p + 16);
      }
      const uint2 gr[4] = {make_uint2(v.x, v.y), make_uint2(v.z, v.w), make_uint2(w.x, w.y),
                           make_uint2(w.z, w.w)};
#pragma unroll
      for (uint32_t g = 0; g < 4; ++g) {
        const uint32_t k = sk + g * 8;
        *(uint2*)(s_b + sn * kEK + ((((k >> 3) ^ (sn & 7u))) << 3)) = gr[g];
      }
    }
  };

  // The LUT is read by `stage`, so it has to be complete before the first call; the loop's own
  // barrier comes too late to cover it.
  __syncthreads();
  stage_a(0, 0);
  stage_b(0, 0);

  // One 128-column block an iteration, two staged buffers, hand-unrolled. The point is not the
  // branch: it puts the activation-scale fold at a static point, so `raw` is dead at the top of
  // every iteration and the first WMMA of a block produces it rather than accumulating into a
  // freshly zeroed copy. The half index also becomes a literal, which makes the LDS base a constant
  // offset.
  //
  // One barrier a staged buffer, not two. Past it, buffer `buf` is fully staged and `buf^1` is dead,
  // so a wave may stage the next column block and run WMMA on this one in either order, and waves
  // may sit in different halves. That is the whole point — WMMA and VALU do not co-issue within a
  // wave, so the dequant can only hide behind another wave's matrix work, and a second barrier would
  // make that impossible.
  const wmma::f8v zero8 = wmma::acc_zero();
  // Two bodies, because single-buffering s_a changes what stays live. `raw[MT][NT]` and `out[MT][NT]`
  // are eight f32 each and already pin half the wave32 register file; overlapping the staging with
  // the MMA adds enough on top to spill, which costs more than the extra wave is worth.
  //
  // The fix is where the fold sits, not how wide the groups are: with the fold after the second
  // half, `raw` spans both halves and every accumulator stays live across the stage_a that
  // overwrites s_a. Folding inside mma_half ends that span. kJG only buys headroom beyond it, and is
  // set to 2 for the register margin rather than for measured time.
  //
  // NOT BIT-IDENTICAL to the double-buffered arm, and cannot be: folding per half computes
  // `out += raw_h0*s + raw_h1*s` where the other shape computes `out += (raw_h0 + raw_h1)*s`. That
  // is a legitimate f32 reassociation, and it moves the output hash. The double-buffered body is
  // kept verbatim so arm 0 stays a fixed numeric reference.
  constexpr uint32_t kJG = (NT >= 2) ? 2u : 1u;
  auto mma_half = [&](uint32_t buf, bool init, uint32_t kb) {
    // The same fold as stage_a's, and it has to be: with one A buffer, `s_a_all + buf * kASz` at
    // buf=1 walks off the end of the array and into s_b_all. Half 1 would then read activations as
    // weights — a wrong answer, and a faster one, since the LDS really is smaller. Only the
    // reference residual catches that.
    const uint8_t* const s_a = s_a_all + (size_t)(kSingleA ? 0u : buf) * kASz;
    const uint8_t* const s_b = s_b_all + (size_t)buf * kBSz;
    if constexpr (kSingleA) {
      (void)init;
#pragma unroll
      for (uint32_t j0 = 0; j0 < NT; j0 += kJG) {
        wmma::f8v acc[MT][kJG];
#pragma unroll
        for (uint32_t ks = 0; ks < kEK; ks += wmma::kK) {
          wmma::i2v af[MT];
#pragma unroll
          for (uint32_t i = 0; i < MT; ++i)
            af[i] = load_efrag(s_a, (wave * MT + i) * 16, ks, lane);
#pragma unroll
          for (uint32_t jj = 0; jj < kJG; ++jj) {
            const wmma::i2v bf = load_efrag(s_b, (j0 + jj) * 16, ks, lane);
#pragma unroll
            for (uint32_t i = 0; i < MT; ++i)
              acc[i][jj] = wmma::mma_fp8(af[i], bf, ks == 0 ? zero8 : acc[i][jj]);
          }
        }
        // Folded HERE, while this group's accumulators are still the only ones live.
#pragma unroll
        for (uint32_t jj = 0; jj < kJG; ++jj) {
          const float s = xrow[j0 + jj][kb];
#pragma unroll
          for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
            for (uint32_t q = 0; q < 8; ++q)
              out[i][j0 + jj][q] = __builtin_fmaf(acc[i][jj][q], s, out[i][j0 + jj][q]);
        }
      }
    } else {
      (void)kb;
#pragma unroll
      for (uint32_t ks = 0; ks < kEK; ks += wmma::kK) {
        wmma::i2v af[MT];
#pragma unroll
        for (uint32_t i = 0; i < MT; ++i)
          af[i] = load_efrag(s_a, (wave * MT + i) * 16, ks, lane);
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j) {
          const wmma::i2v bf = load_efrag(s_b, j * 16, ks, lane);
#pragma unroll
          for (uint32_t i = 0; i < MT; ++i)
            raw[i][j] = wmma::mma_fp8(af[i], bf, (init && ks == 0) ? zero8 : raw[i][j]);
        }
      }
    }
  };

  // Under kSingleA the A stage moves to after the mma_half that reads it. That is legal without a
  // barrier only because s_a is wave-private (see the declarations); s_b keeps its position, so the
  // barrier structure is the same in both arms.
  //
  // THE WRITE-AFTER-READ ON s_a IS ORDERED BY THE COMPILER, NOT BY A BARRIER. A wave's DS ops issue
  // in order and the rows are wave-private, so the only thing that has to hold is that the compiler
  // keeps the store below the loads — which is why `load_efrag` does not take its LDS pointer
  // `__restrict__`.
  for (uint32_t c0 = 0; c0 < cols; c0 += 128) {
    __syncthreads();
    if (c0 + kEK < cols) {
      if constexpr (!kSingleA) stage_a(1, c0 + kEK);
      stage_b(1, c0 + kEK);
    }
    mma_half(0, /*init=*/true, c0 >> 7);
    // A tail half-block only happens if `cols` is not a multiple of 128, which no shipped shape is
    // — `sc_cols = cols >> 7` above already assumes it. The guard is a guard, not a path.
    if (c0 + kEK < cols) {
      if constexpr (kSingleA) stage_a(1, c0 + kEK);
      __syncthreads();
      if (c0 + 128 < cols) {
        if constexpr (!kSingleA) stage_a(0, c0 + 128);
        stage_b(0, c0 + 128);
      }
      mma_half(1, /*init=*/false, c0 >> 7);
      if constexpr (kSingleA) {
        if (c0 + 128 < cols) stage_a(0, c0 + 128);
      }
    }

    if constexpr (!kSingleA) {
      const uint32_t kb = c0 >> 7;
#pragma unroll
      for (uint32_t j = 0; j < NT; ++j) {
        const float s = xrow[j][kb];
#pragma unroll
        for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
          for (uint32_t q = 0; q < 8; ++q) {
            // __builtin_fmaf, not `out += raw * s`. The build carries -ffp-contract=off so a GPU
            // result stays comparable with the CPU reference, and that flag would split this into a
            // separate multiply and add. An explicit fma is not contraction, so the flag does not
            // forbid it, and it is the more accurate of the two.
            out[i][j][q] = __builtin_fmaf(raw[i][j][q], s, out[i][j][q]);
          }
      }
    }
  }

  // The row rscale is constant along K, so it lands here and nowhere in the loop.
#pragma unroll
  for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
    for (uint32_t q = 0; q < 8; ++q) {
      const uint32_t row = m0 + (wave * MT + i) * 16 + wmma::acc_row(lane, q);
      if (row >= rows) continue;
      const float rs = bf16_to_f32_dev(as_global(rscales)[row]) * wscale;
#pragma unroll
      for (uint32_t j = 0; j < NT; ++j) {
        const uint32_t t = tok0 + j * 16;
        if (t >= nt) continue;
        // bf16 for the engine, f32 for bench/expert_wmma. These intermediates — gate/up on the way
        // to the SwiGLU, `down` on the way to the scatter — are read exactly once each and are most
        // of what those two kernels move, so halving them is worth a branch. It is uniform across
        // the launch and sits in the epilogue, so it does not need to be a template.
        const float v = out[i][j][q] * rs;
        if (y_bf16) ((AFF_GLOBAL uint16_t*)y)[(size_t)row * yld + t] = f32_to_bf16_dev(v);
        else        ((AFF_GLOBAL float*)y)[(size_t)row * yld + t] = v;
      }
    }
}

// Gather T token columns out of a dim-major [dim][nb] arena into a compact [dim][T] tile.
__global__ void expert_gather_tokens_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                            const uint32_t* __restrict__ tok, uint32_t dim,
                                            uint32_t nb, uint32_t nt) {
  const uint32_t t = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nt || c >= dim) return;
  dst[(size_t)c * nt + t] = src[(size_t)c * nb + tok[t]];
}

// SwiGLU over a [n][nt] tile. Same asymmetric clamp as the dense path and as Expert.forward in
// the checkpoint's own inference/model.py — gate bounded above only, `up` on both sides. Written
// here rather than borrowed from batch_kernels.hip so the expert semantics cannot drift from the
// dense ones by someone editing the other file.
__global__ void expert_swiglu_tile_kernel(const float* __restrict__ g, const float* __restrict__ u,
                                          float* __restrict__ out, uint32_t n, float limit) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float gv = g[i], uv = u[i];
  if (limit > 0.0f) { gv = fminf(gv, limit); uv = fminf(limit, fmaxf(-limit, uv)); }
  out[i] = (gv / (1.0f + expf(-gv))) * uv;
}

// SwiGLU and the requantise in one pass, so the f32 intermediate never reaches memory. This is
// `batch_quant_fp8_kernel` with its load replaced by the SwiGLU of two arenas and the router weight;
// keeping them separate would write [inter][total] f32 and read it straight back for a value used
// once. Fused, the only thing that lands is the E4M3 the down GEMM consumes.
//
// The weight rides here rather than in the scatter for a reason worth keeping: the quantise picks a
// scale per (slot, 128-row block) from this very data, so a per-slot factor applied BEFORE the
// amax is absorbed into that scale exactly and costs no precision at all.
//
// The clamp is the asymmetric one from Expert.forward in the checkpoint's own inference/model.py --
// gate bounded above only, `up` on both sides -- and is written out here rather than shared with
// the dense SwiGLU so the two cannot drift by someone editing the other file.
constexpr uint32_t kSQTok = 32, kSQThreads = 256;

__global__ __launch_bounds__(kSQThreads) void expert_swiglu_quant_kernel(
    const uint16_t* __restrict__ g, const uint16_t* __restrict__ u, const float* __restrict__ w,
    uint8_t* __restrict__ out, float* __restrict__ scales, uint32_t dim, uint32_t total,
    float limit) {
  __shared__ float tile[128][kSQTok + 1];
  __shared__ float part[8][kSQTok];
  __shared__ float smax[kSQTok];

  const uint32_t tid = threadIdx.x, tt = tid & 31u, kk = tid >> 5;
  const uint32_t t0 = blockIdx.x * kSQTok;
  const uint32_t sl = t0 + tt;
  const uint32_t nkb = dim >> 7;
  const float wv = (sl < total && w) ? w[sl] : 1.0f;

  // blockIdx.y is the 128-column band. Everything here is independent per band — the scale is per
  // (slot, band) and the store is to that band alone — so bands are grid, not a loop. At a prefill
  // chunk `total` alone makes the grid wide enough that it does not matter; at a speculative block
  // it is only a couple of slots, and serialising the bands would leave the card nearly idle.
  for (uint32_t kb = blockIdx.y; kb < nkb; kb += gridDim.y) {
    __syncthreads();
    float m = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) {
      const uint32_t k = kk + i * 8;
      float v = 0.0f;
      if (sl < total) {
        const size_t o = (size_t)(kb * 128 + k) * total + sl;
        float gv = bf16_to_f32_dev(g[o]), uv = bf16_to_f32_dev(u[o]);
        if (limit > 0.0f) { gv = fminf(gv, limit); uv = fminf(limit, fmaxf(-limit, uv)); }
        v = (gv / (1.0f + expf(-gv))) * uv * wv;
      }
      tile[k][tt] = v;
      m = fmaxf(m, fabsf(v));
    }
    part[kk][tt] = m;
    __syncthreads();
    if (kk == 0) {
      float mm = part[0][tt];
#pragma unroll
      for (uint32_t i = 1; i < 8; ++i) mm = fmaxf(mm, part[i][tt]);
      smax[tt] = mm > 0.0f ? mm * (1.0f / 448.0f) : 1.0f;
      if (sl < total) scales[(size_t)sl * nkb + kb] = smax[tt];
    }
    __syncthreads();

    const uint32_t wt = tid >> 3, wk = (tid & 7u) * 16;
    if (t0 + wt < total) {
      const float inv = 1.0f / smax[wt];
      uint32_t wo[4];
#pragma unroll
      for (uint32_t q = 0; q < 4; ++q) {
        float v[4];
        // Clamp, do not trust the round trip: smax is max/448, so max*(1/smax) can land an ulp
        // above 448 in float, and v_cvt_pk_fp8_f32 answers an out-of-range input with NaN rather
        // than saturating -- so the block's own maximum is the element that breaks.
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j)
          v[j] = fminf(fmaxf(tile[wk + q * 4 + j][wt] * inv, -448.0f), 448.0f);
        int pk = 0;
        pk = __builtin_amdgcn_cvt_pk_fp8_f32(v[0], v[1], pk, false);
        pk = __builtin_amdgcn_cvt_pk_fp8_f32(v[2], v[3], pk, true);
        wo[q] = (uint32_t)pk;
      }
      *(uint4*)(out + (size_t)(t0 + wt) * dim + kb * 128 + wk) =
          make_uint4(wo[0], wo[1], wo[2], wo[3]);
    }
  }
}

// One block a dim. Reads the whole contiguous `total`-wide row -- scattered within it, but every
// byte of it is used by the block, so cache serves the order -- and writes consecutive tokens.
__global__ void expert_scatter_slots_kernel(const uint16_t* __restrict__ y, float* __restrict__ block,
                                            const uint32_t* __restrict__ ord,
                                            const uint32_t* __restrict__ run, uint32_t nb_tok,
                                            uint32_t nb_stride, uint32_t total) {
  const uint16_t* __restrict__ row = y + (size_t)blockIdx.x * total;
  for (uint32_t t = threadIdx.x; t < nb_tok; t += blockDim.x) {
    const uint32_t a = run[t], b = run[t + 1];
    if (a == b) continue;
    float acc = 0.0f;
    for (uint32_t i = a; i < b; ++i) acc += bf16_to_f32_dev(row[ord[i]]);
    block[(size_t)blockIdx.x * nb_stride + t] += acc;
  }
}

// Scatter a [dim][nt] result back into the chunk's [dim][nb] accumulator, applying each token's own
// router weight on the way. The weight cannot ride along in the GEMM's `wscale` because it differs
// per COLUMN and the GEMM's scale is per matrix.
//
// atomicAdd because a token appears in several experts' lists and two experts may be in flight on
// the same stream position; float atomics make the accumulation order non-deterministic, which is
// the same reassociation the batched dense path already accepts.
__global__ void expert_scatter_tokens_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                             const uint32_t* __restrict__ tok,
                                             const float* __restrict__ w, uint32_t dim,
                                             uint32_t nb, uint32_t nt) {
  const uint32_t t = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nt || d >= dim) return;
  atomicAdd(&dst[(size_t)d * nb + tok[t]], w[t] * src[(size_t)d * nt + t]);
}

// Each matrix carries its own activation. The down projections of six experts share an output vector
// but not an input, so batching them needs this form. `x` is read from global: it is per-matrix, so
// staging it in LDS would have to be redone for every matrix anyway.
__global__ __launch_bounds__(kThreads) void expert_gemv_multix_kernel(
    BatchPtrs p,
    const float* __restrict__ weights,
    uint32_t n_mats,
    const float* __restrict__ cb,
    float* __restrict__ y,
    uint32_t rows, uint32_t cols) {

  __shared__ float s_cb[4 * 32 * 2];
  __shared__ uint32_t s_code[kWavesPerBlock][320];
  __shared__ uint32_t s_field[kWavesPerBlock][48];

  const int tid  = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;
  for (int i = tid; i < 4 * 32 * 2; i += kThreads) s_cb[i] = cb[i];
  __syncthreads();

  const uint32_t nblk = cols / 16;
  const uint32_t bpl  = nblk / kWave;
  const uint32_t code_words_row  = cols / 2 * 5 / 8 / 4;
  const uint32_t field_words_row = nblk * 6 / 8 / 4;
  const uint32_t field_bytes_lane = bpl * 6 / 8;

  for (uint32_t row = blockIdx.x * kWavesPerBlock + wave; row < rows;
       row += gridDim.x * kWavesPerBlock) {
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (uint32_t m = 0; m < n_mats; ++m) {
      const float rs = bf16_to_f32_dev(p.rscales[m][row]) * weights[m];
      if (rs == 0.0f) continue;
      const float* xm = p.xs[m];
      const uint32_t* crow = (const uint32_t*)(p.data[m]) + (size_t)row * code_words_row;
      const uint32_t* frow = (const uint32_t*)(p.scales[m]) + (size_t)row * field_words_row;
      for (uint32_t i = lane; i < code_words_row; i += kWave) s_code[wave][i] = crow[i];
      for (uint32_t i = lane; i < field_words_row; i += kWave) s_field[wave][i] = frow[i];
      __builtin_amdgcn_s_waitcnt(0);
      __builtin_amdgcn_wave_barrier();

      const uint32_t nwords = bpl * 5 / 4;
      uint32_t cw[kMaxCodeWords + 1];
      const uint32_t base_w = lane * bpl * 5 / 4;
      #pragma unroll
      for (int i = 0; i < kMaxCodeWords; ++i)
        cw[i] = (i < (int)nwords) ? s_code[wave][base_w + i] : 0u;
      cw[kMaxCodeWords] = 0u;
      const uint32_t fbyte = lane * field_bytes_lane;
      const uint32_t fa = fbyte & ~3u;
      const uint64_t fwin =
          (((uint64_t)s_field[wave][fa / 4 + 1] << 32) | s_field[wave][fa / 4]) >> ((fbyte - fa) * 8);
      for (uint32_t b = 0; b < bpl; ++b) {
        const uint32_t field = (uint32_t)((fwin >> (b * 6)) & 63u);
        const float bs = rs * (float)(field & 15u);
        if (bs == 0.0f) continue;
        const float* lut = s_cb + (size_t)(field >> 4) * 64;
        const uint32_t cbit = b * 40;
        const uint32_t w = cbit >> 5;
        const uint64_t win = (((uint64_t)cw[w + 1] << 32) | cw[w]) >> (cbit & 31);
        const float* xb = xm + (size_t)(lane * bpl + b) * 16;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i += 4) {
          const uint32_t c0 = (uint32_t)((win >> (5 * (i + 0))) & 31u);
          const uint32_t c1 = (uint32_t)((win >> (5 * (i + 1))) & 31u);
          const uint32_t c2 = (uint32_t)((win >> (5 * (i + 2))) & 31u);
          const uint32_t c3 = (uint32_t)((win >> (5 * (i + 3))) & 31u);
          s0 = fmaf(lut[c0 * 2 + 0], xb[(i + 0) * 2 + 0], s0);
          s1 = fmaf(lut[c1 * 2 + 0], xb[(i + 1) * 2 + 0], s1);
          s2 = fmaf(lut[c2 * 2 + 0], xb[(i + 2) * 2 + 0], s2);
          s3 = fmaf(lut[c3 * 2 + 0], xb[(i + 3) * 2 + 0], s3);
          s0 = fmaf(lut[c0 * 2 + 1], xb[(i + 0) * 2 + 1], s0);
          s1 = fmaf(lut[c1 * 2 + 1], xb[(i + 1) * 2 + 1], s1);
          s2 = fmaf(lut[c2 * 2 + 1], xb[(i + 2) * 2 + 1], s2);
          s3 = fmaf(lut[c3 * 2 + 1], xb[(i + 3) * 2 + 1], s3);
        }
        acc0 = fmaf(bs, s0, acc0);
        acc1 = fmaf(bs, s1, acc1);
        acc2 = fmaf(bs, s2, acc2);
        acc3 = fmaf(bs, s3, acc3);
      }
      __builtin_amdgcn_wave_barrier();
    }
    const float total = wave_reduce_add((acc0 + acc1) + (acc2 + acc3));
    if (lane == 0) y[row] += total;
  }
}

// The colgroup-major decode matvec: lane = row instead of lane = column.
//
// The row-major kernel put one row per wave and split its columns across lanes, which coalesced the
// code loads. The batched GEMM cannot use that mapping — it needs 32 rows at once, and row-major
// puts those a row stride apart — so the layout changed for it and the two kernels share one buffer.
// Turning the mapping inside out brings the coalescing back: lane l owns row r0+l for the whole
// matrix, so at each column group the 32 lanes' spans are adjacent. Three things follow: the codes
// need no LDS staging (a lane's bytes go straight to registers as five dwords), each lane holds a
// complete row dot product so there is no wave reduction, and the store coalesces as well. The cost
// is that the `rs == 0` and `bs == 0` skips are per-lane now, and rare enough not to matter.
//
// This is decode's routed-expert matvec and the only reader of the packed planes decode reaches, so
// it must understand exactly the wire formats the batched WMMA kernel does. FMT is templated for the
// same reason it is there: an A/B changes the decode and nothing else, and a mismatch is a link
// error rather than silently wrong numbers. The quad table stays PACKED rather than expanded to
// floats so the block scale is an exponent add and the sign an xor — the same two integer ops the
// WMMA path uses, on the same bytes, so the two cannot drift.
//
// One matrix, RB rows a lane, this wave's slice of the column groups. Factored out because both the
// accumulating kernel and the split-output one call it, and two copies of a decode have drifted
// apart in this file before.
//
// RB blocks rows because the activation dominates the load issue: per column group a lane issues a
// couple of loads for codes and fields against sixteen for the activation, and every lane and every
// row in the block wants the same values. RB rows share one activation window, so blocking rows cuts
// loads per FMA without touching the activation count at all.
//
// `row[r]` are the RB rows this lane owns; they are kWave apart, so each row's 32 lanes still read
// 32 consecutive `idx` and the code loads stay coalesced.
template <uint32_t FMT, uint32_t RB>
__device__ __forceinline__ void tr_mat_dot_rb(const BatchPtrs& p, uint32_t m,
                                              const uint32_t* __restrict__ row,
                                              const bool* __restrict__ live,
                                              uint32_t g_lo, uint32_t g_hi, uint32_t rows,
                                              const uint32_t* __restrict__ s_lut,
                                              float* __restrict__ sacc) {
  const uint8_t* __restrict__ dm = p.data[m];
  const uint8_t* __restrict__ fm = p.scales[m];
  // The activation dominates this loop's VMEM instruction count while being a few percent of its
  // bytes, and it arrives at a wave-uniform address — every lane and every row wants the same
  // floats. As a generic pointer it compiles to flat_load_b128, which also waits on the LDS counter;
  // as constant the uniform address scalarises off the vector memory pipe entirely.
  const AFF_CONST float* __restrict__ xm = as_const(p.xs[m]);
  const float* const s_cb = (const float*)s_lut;
  // One column group's weights for the RB rows this lane owns.
  auto fetch = [&](uint32_t g, uint32_t* a0, uint32_t* a1, uint32_t* a2, uint32_t* a3, uint32_t* a4,
                   uint32_t* af) {
#pragma unroll
    for (uint32_t r = 0; r < RB; ++r) {
      a0[r] = a1[r] = a2[r] = a3[r] = a4[r] = af[r] = 0;
      if (live[r]) {
        const size_t idx = (size_t)g * rows + row[r];
        // idx*20 is a multiple of 4 for every idx, and the plane is exactly ng*rows*20 bytes, so
        // the fifth dword of the last lane of the last group is the last dword of the plane —
        // this addressing needs no tail padding, where row-major did.
        const auto* c32 = (const AFF_GLOBAL uint32_t*)(dm + idx * 20);
        a0[r] = c32[0]; a1[r] = c32[1]; a2[r] = c32[2]; a3[r] = c32[3]; a4[r] = c32[4];
        const auto* fp = as_global(fm + idx * 3);
        af[r] = (uint32_t)fp[0] | ((uint32_t)fp[1] << 8) | ((uint32_t)fp[2] << 16);
      }
    }
  };
#pragma unroll
  for (uint32_t r = 0; r < RB; ++r) sacc[r] = 0.0f;
  // No software pipelining here: fetching group g+1 while g decodes does not pay.
  // The other waves already hide that latency, and a prefetch still waits immediately on the field
  // bytes anyway, because `fetch` folds them into f24 before storing.
  for (uint32_t g = g_lo; g < g_hi; ++g) {
    uint32_t w0[RB], w1[RB], w2[RB], w3[RB], w4[RB], f24[RB];
    fetch(g, w0, w1, w2, w3, w4, f24);
    // Every lane in the wave wants the SAME 64 activations, so this is one request broadcast, not
    // 32 — and now it feeds RB rows rather than one.
    const auto* xg = xm + (size_t)g * kEK;
#pragma unroll
    for (uint32_t bb = 0; bb < 4; ++bb) {
      const auto* xb = xg + bb * 16;
      float xr[16];
#pragma unroll
      for (int k = 0; k < 16; ++k) xr[k] = xb[k];
#pragma unroll
      for (uint32_t r = 0; r < RB; ++r) {
        const uint32_t field = (f24[r] >> (bb * 6)) & 63u;
        const uint32_t lo = bb == 0 ? w0[r] : bb == 1 ? w1[r] : bb == 2 ? w2[r] : w3[r];
        const uint32_t hi = bb == 0 ? w1[r] : bb == 1 ? w2[r] : bb == 2 ? w3[r] : w4[r];
        const uint64_t win = ((((uint64_t)hi) << 32) | lo) >> (((bb * 5) & 3u) * 8);
        float s0 = 0.0f, s1 = 0.0f;
        if constexpr (fmt_is_quad(FMT)) {
          // The block scale is already inside the bytes: `+ (e << 3)` per byte is a multiply by
          // 2^e, and the row scale carries the bias (see block_scale_from_code). So there is no
          // `bs` factor here at all, which is one fewer FMA per 16 weights than the pair form.
          constexpr uint32_t kVSh = 3u + (fmt_has_sign(FMT) ? 1u : 0u) +
                                          (fmt_has_rot2(FMT) ? 1u : 0u);
          const uint32_t* lut = s_lut + ((field >> kVSh) & (fmt_variants(FMT) - 1u)) * 1024u;
          const uint32_t add = (field & 7u) * 0x08080808u;
          const uint32_t neg = fmt_has_sign(FMT) ? (field & 8u) * 0x10101010u : 0u;
          // The 2-byte rotate as a funnel shift by a variable amount rather than a rotate under a
          // select. `alignbit(q, q, s)` is `(((u64)q << 32) | q) >> s` truncated, so s=16 is exactly
          // (q >> 16) | (q << 16) and s=0 is q — one instruction either way, where the conditional
          // form needs an alignbit and a v_cndmask. `rsh` is per block field, so it hoists out of
          // the unrolled quad loop.
          const uint32_t rsh = (fmt_has_rot2(FMT) && (field & 16u)) ? 16u : 0u;
#pragma unroll
          for (uint32_t i = 0; i < 4; ++i) {
            uint32_t q = (lut[(uint32_t)((win >> (10 * i)) & 1023u)] + add) ^ neg;
            if constexpr (fmt_has_rot2(FMT)) q = __builtin_amdgcn_alignbit(q, q, rsh);
            // The whole quad in two instructions. This kernel is ALU-bound and the decode is the
            // ALU, so unpacking E4M3 a byte at a time dominated it; v_cvt_pk_f32_fp8 does two at a
            // time and gfx1201 has it. See e4m3_decode_pk.
            const aff_f32x2 d01 = e4m3_decode_pk<false>(q);
            const aff_f32x2 d23 = e4m3_decode_pk<true>(q);
            s0 = fmaf(d01.x, xr[i * 4 + 0], s0);
            s1 = fmaf(d01.y, xr[i * 4 + 1], s1);
            s0 = fmaf(d23.x, xr[i * 4 + 2], s0);
            s1 = fmaf(d23.y, xr[i * 4 + 3], s1);
          }
          sacc[r] += s0 + s1;
        } else {
          const float bs = (float)(field & 15u);
          const float* lut = s_cb + (size_t)(field >> 4) * 64;
#pragma unroll
          for (int i = 0; i < 8; i += 2) {
            const uint32_t c0 = (uint32_t)((win >> (5 * (i + 0))) & 31u);
            const uint32_t c1 = (uint32_t)((win >> (5 * (i + 1))) & 31u);
            s0 = fmaf(lut[c0 * 2 + 0], xr[(i + 0) * 2 + 0], s0);
            s1 = fmaf(lut[c1 * 2 + 0], xr[(i + 1) * 2 + 0], s1);
            s0 = fmaf(lut[c0 * 2 + 1], xr[(i + 0) * 2 + 1], s0);
            s1 = fmaf(lut[c1 * 2 + 1], xr[(i + 1) * 2 + 1], s1);
          }
          sacc[r] = fmaf(bs, s0 + s1, sacc[r]);
        }
      }
    }
  }
}

// One matrix, one row — the RB=1 case, kept as the name the accumulating kernel calls.
template <uint32_t FMT>
__device__ __forceinline__ float tr_mat_dot(const BatchPtrs& p, uint32_t m, uint32_t row, bool live,
                                            uint32_t g_lo, uint32_t g_hi, uint32_t rows,
                                            const uint32_t* __restrict__ s_lut) {
  float acc = 0.0f;
  tr_mat_dot_rb<FMT, 1>(p, m, &row, &live, g_lo, g_hi, rows, s_lut, &acc);
  return acc;
}

// Factored out of the kernel so the accumulating form and the split-output one cannot drift.
template <uint32_t FMT>
__device__ __forceinline__ void expert_gemv_tr_body(
    const BatchPtrs& p,
    const float* __restrict__ weights,
    uint32_t n_mats,
    const float* __restrict__ cb,
    float* __restrict__ y,
    uint32_t rows, uint32_t cols) {

  constexpr uint32_t kLutW = fmt_is_quad(FMT) ? fmt_variants(FMT) * 1024u : 4u * 32u * 2u;
  __shared__ uint32_t s_lut[kLutW];
  __shared__ float s_part[kWavesPerBlock][kWave];
  const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1), wave = wave_uniform(tid / kWave);
  float* const s_cb = (float*)s_lut;
  for (uint32_t i = tid; i < kLutW; i += kThreads) s_lut[i] = ((const uint32_t*)cb)[i];
  __syncthreads();

  const uint32_t ng = cols / kEK;              // column groups: 4 blocks, 20 code bytes, 3 field
  // THE WAVES OF A BLOCK SPLIT THE COLUMNS, NOT THE ROWS. Giving each wave its own 32 rows is the
  // obvious reading of lane-per-row and starves the grid — the whole matvec becomes rows/32 waves.
  // Instead a block owns 32 rows and all of the columns, and its eight waves take an eighth of the
  // column groups each. The eight partial sums per row then reduce through LDS in a FIXED order, so
  // the result is deterministic; an atomicAdd would be simpler and would make decode irreproducible
  // run to run.
  const uint32_t gpw = (ng + kWavesPerBlock - 1) / kWavesPerBlock;
  const uint32_t g_lo = wave * gpw, g_hi = g_lo + gpw < ng ? g_lo + gpw : ng;
  for (uint32_t r0 = blockIdx.x * kWave; r0 < rows; r0 += gridDim.x * kWave) {
    const uint32_t row = r0 + lane;
    const bool live = row < rows;
    float acc = 0.0f;
    for (uint32_t m = 0; m < n_mats; ++m) {
      const float rs = live ? bf16_to_f32_dev(p.rscales[m][row]) * weights[m] : 0.0f;
      acc = fmaf(rs, tr_mat_dot<FMT>(p, m, row, live, g_lo, g_hi, rows, s_lut), acc);
    }
    // The block's eight column-partials for this row, summed in wave order and stored once.
    __syncthreads();
    s_part[wave][lane] = acc;
    __syncthreads();
    if (wave == 0) {
      float t = 0.0f;
#pragma unroll
      for (uint32_t w = 0; w < kWavesPerBlock; ++w) t += s_part[w][lane];
      if (live) y[row] += t;      // 32 lanes, 32 consecutive floats — the store coalesces too
    }
  }
  (void)s_cb;
}

template <uint32_t FMT>
__global__ __launch_bounds__(kThreads) void expert_gemv_kernel_tr(
    BatchPtrs p,
    const float* __restrict__ weights,
    uint32_t n_mats,
    const float* __restrict__ cb,
    float* __restrict__ y,
    uint32_t rows, uint32_t cols) {
  expert_gemv_tr_body<FMT>(p, weights, n_mats, cb, y, rows, cols);
}

// Split output: n_mats matrices of identical shape, each writing its own vector.
//
// The accumulating kernel above is the `down` form, where every expert sums into one hidden vector.
// Gate and up each write their own intermediate, so they would otherwise be one launch per matrix —
// and a matvec that small reads well under what its bytes stream at.
//
// So the grid covers (matrix, row) instead of row, and the matrix loop disappears. `rows` is a
// multiple of kWave for every expert shape here, so a 32-row block never straddles two matrices and
// the LDS reduction is unchanged. The store is a store rather than an accumulate, which also retires
// the memset that had to zero the scratch buffer first.
template <uint32_t FMT, uint32_t RB>
__device__ __forceinline__ void expert_gemv_tr_split_body(
    const BatchPtrs& p,
    const float* __restrict__ weights,
    uint32_t n_mats,
    const float* __restrict__ cb,
    uint32_t rows, uint32_t cols) {

  constexpr uint32_t kLutW = fmt_is_quad(FMT) ? fmt_variants(FMT) * 1024u : 4u * 32u * 2u;
  __shared__ uint32_t s_lut[kLutW];
  __shared__ float s_part[kWavesPerBlock][RB * kWave];
  const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1), wave = wave_uniform(tid / kWave);
  for (uint32_t i = tid; i < kLutW; i += kThreads) s_lut[i] = ((const uint32_t*)cb)[i];
  __syncthreads();

  const uint32_t ng = cols / kEK;
  const uint32_t gpw = (ng + kWavesPerBlock - 1) / kWavesPerBlock;
  const uint32_t g_lo = wave * gpw, g_hi = g_lo + gpw < ng ? g_lo + gpw : ng;
  const uint32_t nb = (rows + RB * kWave - 1) / (RB * kWave);   // row-blocks a matrix
  for (uint32_t b = blockIdx.x; b < n_mats * nb; b += gridDim.x) {
    const uint32_t m = b / nb;
    const uint32_t r0 = (b - m * nb) * RB * kWave;
    uint32_t row[RB];
    bool live[RB];
    float rs[RB], sacc[RB];
#pragma unroll
    for (uint32_t r = 0; r < RB; ++r) {
      row[r] = r0 + r * kWave + lane;
      live[r] = row[r] < rows;
      rs[r] = live[r] ? bf16_to_f32_dev(p.rscales[m][row[r]]) * weights[m] : 0.0f;
    }
    tr_mat_dot_rb<FMT, RB>(p, m, row, live, g_lo, g_hi, rows, s_lut, sacc);
    __syncthreads();
#pragma unroll
    for (uint32_t r = 0; r < RB; ++r) s_part[wave][r * kWave + lane] = rs[r] * sacc[r];
    __syncthreads();
    if (wave == 0) {
#pragma unroll
      for (uint32_t r = 0; r < RB; ++r) {
        float t = 0.0f;
#pragma unroll
        for (uint32_t w = 0; w < kWavesPerBlock; ++w) t += s_part[w][r * kWave + lane];
        if (live[r]) p.ys[m][row[r]] = t;
      }
    }
  }
}

template <uint32_t FMT, uint32_t RB>
__global__ __launch_bounds__(kThreads) void expert_gemv_kernel_tr_split(
    BatchPtrs p,
    const float* __restrict__ weights,
    uint32_t n_mats,
    const float* __restrict__ cb,
    uint32_t rows, uint32_t cols) {
  expert_gemv_tr_split_body<FMT, RB>(p, weights, n_mats, cb, rows, cols);
}

// The row-major decode gemv. Only a test or a bench that builds its own planes reaches it — the
// engine's slab is colgroup-major and takes expert_gemv_kernel_tr.
__global__ __launch_bounds__(kThreads) void expert_gemv_kernel_v8(
    BatchPtrs p,
    const float* __restrict__ weights,
    uint32_t n_mats,
    const float* __restrict__ cb,
    float* __restrict__ y,
    uint32_t rows, uint32_t cols) {

  __shared__ float s_cb[4 * 32 * 2];
  __shared__ uint32_t s_code[kWavesPerBlock][320];
  __shared__ uint32_t s_field[kWavesPerBlock][48];

  const int tid  = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;

  for (int i = tid; i < 4 * 32 * 2; i += kThreads) s_cb[i] = cb[i];
  __syncthreads();

  const uint32_t nblk = cols / 16;
  const uint32_t bpl  = nblk / kWave;
  const uint32_t code_words_row  = cols / 2 * 5 / 8 / 4;
  const uint32_t field_words_row = nblk * 6 / 8 / 4;
  const uint32_t nwords = bpl * 5 / 4;                 // 10 at cols 4096, 5 at 2048

  for (uint32_t row = blockIdx.x * kWavesPerBlock + wave; row < rows;
       row += gridDim.x * kWavesPerBlock) {

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (uint32_t m = 0; m < n_mats; ++m) {
      const float rs = bf16_to_f32_dev(p.rscales[m][row]) * weights[m];
      const uint32_t* crow = (const uint32_t*)(p.data[m]) + (size_t)row * code_words_row;
      const uint32_t* frow = (const uint32_t*)(p.scales[m]) + (size_t)row * field_words_row;
      for (uint32_t i = lane; i < code_words_row; i += kWave) s_code[wave][i] = crow[i];
      for (uint32_t i = lane; i < field_words_row; i += kWave) s_field[wave][i] = frow[i];
      __builtin_amdgcn_s_waitcnt(0);
      __builtin_amdgcn_wave_barrier();
      if (rs == 0.0f) continue;

      // Ten dword reads, then everything else happens in registers.
      uint32_t cw[kMaxCodeWords + 1];
      const uint32_t base_w = lane * bpl * 5 / 4;
      #pragma unroll
      for (int i = 0; i < kMaxCodeWords; ++i)
        cw[i] = (i < (int)nwords) ? s_code[wave][base_w + i] : 0u;
      cw[kMaxCodeWords] = 0u;
      // Fields: bpl*6 bits per lane, starting at a 2-byte boundary, so two dwords always cover it.
      const uint32_t fbyte = lane * bpl * 6 / 8;
      const uint32_t fa = fbyte & ~3u;
      const uint32_t fsh = (fbyte - fa) * 8;
      const uint64_t fwin = (((uint64_t)s_field[wave][fa / 4 + 1] << 32) | s_field[wave][fa / 4]) >> fsh;

      for (uint32_t b = 0; b < bpl; ++b) {
        const uint32_t field = (uint32_t)((fwin >> (b * 6)) & 63u);
        const float bs = rs * (float)(field & 15u);
        if (bs == 0.0f) continue;
        const float* lut = s_cb + (size_t)(field >> 4) * 64;

        // b*40 mod 32 is always in {0,8,16,24}, so two dwords always hold the 40-bit window.
        const uint32_t cbit = b * 40;
        const uint32_t w = cbit >> 5;
        const uint64_t win = (((uint64_t)cw[w + 1] << 32) | cw[w]) >> (cbit & 31);

        const float* xb = p.xs[m] + (size_t)(lane * bpl + b) * 16;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i += 4) {
          const uint32_t c0 = (uint32_t)((win >> (5 * (i + 0))) & 31u);
          const uint32_t c1 = (uint32_t)((win >> (5 * (i + 1))) & 31u);
          const uint32_t c2 = (uint32_t)((win >> (5 * (i + 2))) & 31u);
          const uint32_t c3 = (uint32_t)((win >> (5 * (i + 3))) & 31u);
          s0 = fmaf(lut[c0 * 2 + 0], xb[(i + 0) * 2 + 0], s0);
          s1 = fmaf(lut[c1 * 2 + 0], xb[(i + 1) * 2 + 0], s1);
          s2 = fmaf(lut[c2 * 2 + 0], xb[(i + 2) * 2 + 0], s2);
          s3 = fmaf(lut[c3 * 2 + 0], xb[(i + 3) * 2 + 0], s3);
          s0 = fmaf(lut[c0 * 2 + 1], xb[(i + 0) * 2 + 1], s0);
          s1 = fmaf(lut[c1 * 2 + 1], xb[(i + 1) * 2 + 1], s1);
          s2 = fmaf(lut[c2 * 2 + 1], xb[(i + 2) * 2 + 1], s2);
          s3 = fmaf(lut[c3 * 2 + 1], xb[(i + 3) * 2 + 1], s3);
        }
        acc0 = fmaf(bs, s0, acc0);
        acc1 = fmaf(bs, s1, acc1);
        acc2 = fmaf(bs, s2, acc2);
        acc3 = fmaf(bs, s3, acc3);
      }
      __builtin_amdgcn_wave_barrier();
    }
    const float total = wave_reduce_add((acc0 + acc1) + (acc2 + acc3));
    if (lane == 0) y[row] += total;
  }
}

// Monolithic FFN: the whole routed-expert block in one cooperative launch, which removes thirteen
// of the fourteen launches the staged form needs.
//
// It has to be cooperative rather than merely bigger: down(a_k) needs every row of gate_k and up_k,
// which is a grid-wide dependency, and __syncthreads() only covers a workgroup. The alternatives are
// worse — one workgroup per expert leaves most of the card idle, and recomputing gate/up per
// workgroup adds most of the arithmetic back. The grid is sized from queried occupancy, not assumed,
// because the whole grid must be resident for grid.sync().

// One matrix row: coalesced staging into LDS, lane-major read-back, four independent FMA chains.
// Returns this lane's partial sum; the caller reduces across the wave.
__device__ inline float gemv_row_staged(
    const uint8_t* __restrict__ data, const uint8_t* __restrict__ scales,
    const uint16_t* __restrict__ rscales, uint32_t row, uint32_t cols,
    const float* __restrict__ xv, const float* __restrict__ s_cb,
    uint32_t* __restrict__ s_code, uint32_t* __restrict__ s_field, int lane, float wmul) {

  const uint32_t nblk = cols / 16;
  const uint32_t bpl  = nblk / kWave;
  const uint32_t code_words_row  = cols / 2 * 5 / 8 / 4;
  const uint32_t field_words_row = nblk * 6 / 8 / 4;

  const float rs = bf16_to_f32_dev(rscales[row]) * wmul;
  const uint32_t* crow = (const uint32_t*)data + (size_t)row * code_words_row;
  const uint32_t* frow = (const uint32_t*)scales + (size_t)row * field_words_row;
  for (uint32_t i = lane; i < code_words_row; i += kWave) s_code[i] = crow[i];
  for (uint32_t i = lane; i < field_words_row; i += kWave) s_field[i] = frow[i];
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_wave_barrier();
  if (rs == 0.0f) return 0.0f;

  // Bulk-read this lane's slice into registers: ten dword reads instead of ~80 byte reads.
  const uint32_t nwords = bpl * 5 / 4;
  uint32_t cw[kMaxCodeWords + 1];
  const uint32_t base_w = lane * bpl * 5 / 4;
  #pragma unroll
  for (int i = 0; i < kMaxCodeWords; ++i) cw[i] = (i < (int)nwords) ? s_code[base_w + i] : 0u;
  cw[kMaxCodeWords] = 0u;
  const uint32_t fbyte = lane * bpl * 6 / 8;
  const uint32_t fa = fbyte & ~3u;
  const uint32_t fsh = (fbyte - fa) * 8;
  const uint64_t fwin = (((uint64_t)s_field[fa / 4 + 1] << 32) | s_field[fa / 4]) >> fsh;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  for (uint32_t b = 0; b < bpl; ++b) {
    const uint32_t field = (uint32_t)((fwin >> (b * 6)) & 63u);
    const float bs = rs * (float)(field & 15u);
    if (bs == 0.0f) continue;
    const float* lut = s_cb + (size_t)(field >> 4) * 64;
    const uint32_t cbit = b * 40;
    const uint32_t w = cbit >> 5;
    const uint64_t win = (((uint64_t)cw[w + 1] << 32) | cw[w]) >> (cbit & 31);
    const float* xb = xv + (size_t)(lane * bpl + b) * 16;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i += 4) {
      const uint32_t c0 = (uint32_t)((win >> (5 * (i + 0))) & 31u);
      const uint32_t c1 = (uint32_t)((win >> (5 * (i + 1))) & 31u);
      const uint32_t c2 = (uint32_t)((win >> (5 * (i + 2))) & 31u);
      const uint32_t c3 = (uint32_t)((win >> (5 * (i + 3))) & 31u);
      s0 = fmaf(lut[c0 * 2 + 0], xb[(i + 0) * 2 + 0], s0);
      s1 = fmaf(lut[c1 * 2 + 0], xb[(i + 1) * 2 + 0], s1);
      s2 = fmaf(lut[c2 * 2 + 0], xb[(i + 2) * 2 + 0], s2);
      s3 = fmaf(lut[c3 * 2 + 0], xb[(i + 3) * 2 + 0], s3);
      s0 = fmaf(lut[c0 * 2 + 1], xb[(i + 0) * 2 + 1], s0);
      s1 = fmaf(lut[c1 * 2 + 1], xb[(i + 1) * 2 + 1], s1);
      s2 = fmaf(lut[c2 * 2 + 1], xb[(i + 2) * 2 + 1], s2);
      s3 = fmaf(lut[c3 * 2 + 1], xb[(i + 3) * 2 + 1], s3);
    }
    acc0 = fmaf(bs, s0, acc0);
    acc1 = fmaf(bs, s1, acc1);
    acc2 = fmaf(bs, s2, acc2);
    acc3 = fmaf(bs, s3, acc3);
  }
  return (acc0 + acc1) + (acc2 + acc3);
}

struct MonoArgs {
  BatchPtrs gate, up, down;
  const float* weights;
  const float* cb;
  const float* x;
  float* scratch;
  float* out;
  uint32_t n_experts, inter, hidden;
  float limit;
};

__global__ __launch_bounds__(kThreads) void expert_ffn_mono_kernel(MonoArgs a) {
  namespace cg = cooperative_groups;
  cg::grid_group grid = cg::this_grid();

  // x is NOT staged in LDS: reading it from global costs nothing measurable, and the LDS it would
  // take halves resident blocks per CU — which for a cooperative kernel also halves the grid, since
  // the whole grid must be resident for grid.sync().
  __shared__ float s_cb[4 * 32 * 2];
  __shared__ uint32_t s_code[kWavesPerBlock][320];
  __shared__ uint32_t s_field[kWavesPerBlock][48];

  const int tid = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;
  const uint32_t waves_per_grid = gridDim.x * kWavesPerBlock;
  const uint32_t gwave = blockIdx.x * kWavesPerBlock + wave;

  for (int i = tid; i < 4 * 32 * 2; i += kThreads) s_cb[i] = a.cb[i];
  __syncthreads();

  // --- phase 1: gate and up for every expert, straight into scratch ---------------------------
  // One flat work list over (expert, gate|up, row) so the whole grid stays busy regardless of how
  // the expert count divides into the block count.
  const uint32_t n_gu = a.n_experts * 2 * a.inter;
  for (uint32_t item = gwave; item < n_gu; item += waves_per_grid) {
    const uint32_t e = item / (2 * a.inter);
    const uint32_t rest = item - e * 2 * a.inter;
    const uint32_t which = rest / a.inter;
    const uint32_t row = rest - which * a.inter;
    const BatchPtrs& src = which == 0 ? a.gate : a.up;
    const float v = gemv_row_staged(src.data[e], src.scales[e], src.rscales[e], row, a.hidden,
                                    a.x, s_cb, s_code[wave], s_field[wave], lane, 1.0f);
    const float total = wave_reduce_add(v);
    if (lane == 0) a.scratch[(size_t)(2 * e + which) * a.inter + row] = total;
  }
  grid.sync();

  // --- phase 2: SwiGLU over every expert's intermediate ----------------------------------------
  const uint32_t n_act = a.n_experts * a.inter;
  for (uint32_t t = blockIdx.x * kThreads + tid; t < n_act; t += gridDim.x * kThreads) {
    const uint32_t k = t / a.inter, i = t - k * a.inter;
    float* g = a.scratch + (size_t)(2 * k) * a.inter + i;
    const float* u = a.scratch + (size_t)(2 * k + 1) * a.inter + i;
    float gv = *g, uv = *u;
    if (a.limit > 0.0f) { gv = fminf(gv, a.limit); uv = fminf(a.limit, fmaxf(-a.limit, uv)); }
    *g = (gv / (1.0f + expf(-gv))) * uv;
  }
  grid.sync();

  // --- phase 3: down, all experts summed into out ----------------------------------------------
  // One wave per OUTPUT row, looping the experts, so the sum happens in registers and `out` is
  // written once — no atomics and no second reduction pass.
  for (uint32_t row = gwave; row < a.hidden; row += waves_per_grid) {
    float sum = 0.0f;
    for (uint32_t e = 0; e < a.n_experts; ++e) {
      const float* act = a.scratch + (size_t)(2 * e) * a.inter;
      const float v = gemv_row_staged(a.down.data[e], a.down.scales[e], a.down.rscales[e], row,
                                      a.inter, act, s_cb, s_code[wave], s_field[wave], lane,
                                      a.weights[e]);
      sum += wave_reduce_add(v);
    }
    if (lane == 0) a.out[row] += sum;
  }
}

// Dense bf16 matvec. The dense weights are most of the per-token traffic and, unlike the routed
// experts, they fit in VRAM — so this reads at device bandwidth rather than over the host bus.
//
// One wave per output row, each lane loading a uint4 (8 bf16), so a wave covers 512 contiguous bytes
// and every read is fully coalesced.

constexpr int kDenseWaves = 8;
constexpr int kDenseThreads = kDenseWaves * kWave;

// The same for bf16.
__device__ __forceinline__ float bf16_row_dot(const uint16_t* __restrict__ W,
                                              const float* __restrict__ xv, uint32_t row,
                                              uint32_t cols, int lane) {
  const uint32_t stride = kWave * 8u;
  const uint32_t full = (cols / stride) * stride;
  const uint16_t* w = W + (size_t)row * cols;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  for (uint32_t c = (uint32_t)lane * 8u; c < full; c += stride) {
    const uint4 packed = *(const uint4*)(w + c);
    const uint32_t* pw = (const uint32_t*)&packed;
    a0 = fmaf(__uint_as_float((pw[0] & 0xFFFFu) << 16), xv[c + 0], a0);
    a0 = fmaf(__uint_as_float(pw[0] & 0xFFFF0000u),     xv[c + 1], a0);
    a1 = fmaf(__uint_as_float((pw[1] & 0xFFFFu) << 16), xv[c + 2], a1);
    a1 = fmaf(__uint_as_float(pw[1] & 0xFFFF0000u),     xv[c + 3], a1);
    a2 = fmaf(__uint_as_float((pw[2] & 0xFFFFu) << 16), xv[c + 4], a2);
    a2 = fmaf(__uint_as_float(pw[2] & 0xFFFF0000u),     xv[c + 5], a2);
    a3 = fmaf(__uint_as_float((pw[3] & 0xFFFFu) << 16), xv[c + 6], a3);
    a3 = fmaf(__uint_as_float(pw[3] & 0xFFFF0000u),     xv[c + 7], a3);
  }
  for (uint32_t t = full + (uint32_t)lane; t < cols; t += kWave)
    a0 = fmaf(__uint_as_float((uint32_t)w[t] << 16), xv[t], a0);
  return wave_reduce_add((a0 + a1) + (a2 + a3));
}



__global__ __launch_bounds__(kDenseThreads) void matvec_bf16_tall_kernel(
    const uint16_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ out,
    uint32_t cols) {
  __shared__ float red[kDenseWaves];
  const int tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
  const uint16_t* w = W + (size_t)blockIdx.x * cols;

  // Eight halves a thread, so each read is a uint4 and the block covers kDenseThreads*8 consecutive
  // columns a step — one fully coalesced sweep rather than kDenseWaves interleaved ones.
  const uint32_t stride = kDenseThreads * 8u;
  const uint32_t full = (cols / stride) * stride;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  for (uint32_t c = (uint32_t)tid * 8u; c < full; c += stride) {
    const uint4 packed = *(const uint4*)(w + c);
    const uint32_t* pw = (const uint32_t*)&packed;
    a0 = fmaf(__uint_as_float((pw[0] & 0xFFFFu) << 16), x[c + 0], a0);
    a0 = fmaf(__uint_as_float(pw[0] & 0xFFFF0000u),     x[c + 1], a0);
    a1 = fmaf(__uint_as_float((pw[1] & 0xFFFFu) << 16), x[c + 2], a1);
    a1 = fmaf(__uint_as_float(pw[1] & 0xFFFF0000u),     x[c + 3], a1);
    a2 = fmaf(__uint_as_float((pw[2] & 0xFFFFu) << 16), x[c + 4], a2);
    a2 = fmaf(__uint_as_float(pw[2] & 0xFFFF0000u),     x[c + 5], a2);
    a3 = fmaf(__uint_as_float((pw[3] & 0xFFFFu) << 16), x[c + 6], a3);
    a3 = fmaf(__uint_as_float(pw[3] & 0xFFFF0000u),     x[c + 7], a3);
  }
  for (uint32_t t = full + (uint32_t)tid; t < cols; t += kDenseThreads)
    a0 = fmaf(__uint_as_float((uint32_t)w[t] << 16), x[t], a0);

  const float v = wave_reduce_add((a0 + a1) + (a2 + a3));
  if (lane == 0) red[wave] = v;
  __syncthreads();
  if (tid == 0) {
    float t = 0.0f;
    for (int k = 0; k < kDenseWaves; ++k) t += red[k];
    out[blockIdx.x] = t;
  }
}

__global__ __launch_bounds__(kDenseThreads) void matvec_bf16_kernel(
    const uint16_t* __restrict__ W, const float* __restrict__ x,
    float* __restrict__ out, uint32_t rows, uint32_t cols, int stage_x) {

  extern __shared__ float s_x[];
  const int tid = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;

  if (stage_x) {
    for (uint32_t i = tid; i < cols; i += kDenseThreads) s_x[i] = x[i];
    __syncthreads();
  }
  const float* xv = stage_x ? s_x : x;

  for (uint32_t row = blockIdx.x * kDenseWaves + wave; row < rows;
       row += gridDim.x * kDenseWaves) {
    const float total = bf16_row_dot(W, xv, row, cols, lane);
    if (lane == 0) out[row] = total;
  }
}


} // namespace

void expert_gather_tokens_hip(const float* src, float* dst, const uint32_t* tok, uint32_t dim,
                              uint32_t nb, uint32_t nt, void* stream) {
  if (!nt || !dim) return;
  const dim3 blk(64, 4);
  const dim3 grd((dim + blk.x - 1) / blk.x, (nt + blk.y - 1) / blk.y);
  hipLaunchKernelGGL(expert_gather_tokens_kernel, grd, blk, 0, (hipStream_t)stream, src, dst, tok,
                     dim, nb, nt);
  AFF_HIP_CHECK(hipGetLastError());
}

bool expert_gemm_tokens_hip(const DeviceMatrix& m, const float* cb, const float* x, float* y,
                            uint32_t nt, float wscale, void* stream) {
  return expert_gemm_tokens_pair_hip(m, y, nullptr, nullptr, cb, x, nt, wscale, stream);
}

bool expert_gemm_tokens_pair_hip(const DeviceMatrix& a, float* ya, const DeviceMatrix* b, float* yb,
                                 const float* cb, const float* x, uint32_t nt, float wscale,
                                 void* stream) {
  // This one can decline rather than abort: `bool` already means "did it run", and every caller
  // handles false. It is the tile path, which the engine never reaches while its planes are
  // colgroup-major.
  if (expert_planes_are_colgroup_major()) return false;
  const DeviceMatrix& m = a;
  if (b && (b->rows != a.rows || b->cols != a.cols)) return false;
  // cols % 2048, not % 512. `field_bytes_lane = bpl * 6 / 8` is an integer division, so a lane's
  // fields only start on a byte boundary when bpl % 4 == 0, i.e. cols % 2048. Below that, odd lanes
  // read their fields half a byte early and the result is uncorrelated rather than merely imprecise.
  // Both real expert shapes satisfy it.
  if (!nt || m.cols % 2048 != 0) return false;
  const uint32_t blocks = std::min<uint32_t>(2048u, (m.rows + kWavesPerBlock - 1) / kWavesPerBlock);
  // T is a template parameter because it sizes a REGISTER array; spilling the accumulators to
  // scratch would give back exactly the reuse the kernel exists to get. Sizes between the tiers are
  // padded up by the caller: a few wasted columns, no correctness cost.
  const GemmJob j0{a.data, a.scales, a.rscales, ya};
  const GemmJob j1 = b ? GemmJob{b->data, b->scales, b->rscales, yb} : j0;
  #define AFF_GEMM_T(TT) \
    hipLaunchKernelGGL(expert_gemm_tokens_kernel<TT>, dim3(blocks, b ? 2u : 1u), dim3(kThreads), \
                       0, (hipStream_t)stream, j0, j1, cb, x, wscale, m.rows, m.cols); AFF_HIP_CHECK(hipGetLastError())
  switch (nt) {
    case 1:  AFF_GEMM_T(1);  break;
    case 2:  AFF_GEMM_T(2);  break;
    case 4:  AFF_GEMM_T(4);  break;
    case 8:  AFF_GEMM_T(8);  break;
    case 16: AFF_GEMM_T(16); break;
    case 32: AFF_GEMM_T(32); break;
    default: return false;
  }
  #undef AFF_GEMM_T
  AFF_HIP_CHECK(hipGetLastError());
  return true;
}

bool expert_ffn_tokens_hip(const DeviceExpert& e, const float* cb, const float* x_arena,
                           const uint32_t* d_tok, const float* d_w, uint32_t nt, uint32_t nb,
                           uint32_t hidden, uint32_t inter, float limit, float* scratch,
                           float* d_block, void* stream) {
  if (!nt || nt > kExpertTokMax) return false;
  uint32_t T = 1;                        // the kernel's register array is a power of two
  while (T < nt) T <<= 1;

  hipStream_t st = (hipStream_t)stream;
  // scratch layout, all dim-major [dim][T]: x | g | u(then a, in place) | y
  float* x_tile = scratch;
  float* g_tile = x_tile + (size_t)hidden * T;
  float* u_tile = g_tile + (size_t)inter * T;
  float* y_tile = u_tile + (size_t)inter * T;

  // Padded columns nt..T carry token 0 and weight 0: they compute a real result that is then
  // scattered with a zero weight. Cheaper than a bounds test on every FMA in the inner loop, and
  // they are at most T/2 of the work only when nt is just past a power of two. No memset any more —
  // the GEMM stores rather than accumulates, so there is nothing to zero.
  expert_gather_tokens_hip(x_arena, x_tile, d_tok, hidden, nb, T, stream);
  if (!expert_gemm_tokens_pair_hip(e.gate, g_tile, &e.up, u_tile, cb, x_tile, T, 1.0f, stream))
    return false;
  {
    const uint32_t n = inter * T, thr = 256;
    hipLaunchKernelGGL(expert_swiglu_tile_kernel, dim3((n + thr - 1) / thr), dim3(thr), 0, st,
                       g_tile, u_tile, u_tile, n, limit);
    AFF_HIP_CHECK(hipGetLastError());
  }
  if (!expert_gemm_tokens_hip(e.down, cb, u_tile, y_tile, T, 1.0f, stream)) return false;
  {
    const dim3 blk(64, 4);
    const dim3 grd((hidden + blk.x - 1) / blk.x, (T + blk.y - 1) / blk.y);
    hipLaunchKernelGGL(expert_scatter_tokens_kernel, grd, blk, 0, st, y_tile, d_block, d_tok, d_w,
                       hidden, nb, T);
    AFF_HIP_CHECK(hipGetLastError());
  }
  return true;
}

void matvec_bf16_hip(const uint16_t* W, const float* x, uint32_t rows, uint32_t cols,
                     float* out, void* stream, bool want_tall) {
  const size_t lds = (size_t)cols * sizeof(float);
  // Staged, unlike the FP8 kernel below. Both hit LDS bank conflicts on the staged activation, but
  // at half the factor here — 8 floats a lane rather than 16 — and at that width the conflicts cost
  // nothing measurable, so this keeps the staging and the L1 pressure it relieves.
  const int stage = lds <= 32768 ? 1 : 0;      // 32 KiB ceiling keeps two blocks resident per CU
  const uint32_t blocks = std::min<uint32_t>(2048u, (rows + kDenseWaves - 1) / kDenseWaves);
  // A row per wave leaves the card idle when the matrix is short, so below the crossover each row
  // gets a whole block instead — same bytes, 8x the threads, still one launch. `cols` has to be wide
  // enough that a block-wide sweep is worth its cross-wave reduction.
  //
  // `want_tall` IS A NUMERIC REQUEST, NOT A PERFORMANCE HINT. The two kernels reduce a row
  // differently — one wave reduction against eight waves through LDS — so they agree to about the
  // last bit and no further. That is invisible almost everywhere and decisive in the router, where
  // near-tied selection values are compared against each other and a row landing one ulp lower swaps
  // an expert. A caller that computes different row counts on different paths must pin this, or the
  // two will route differently.
  if ((want_tall || blocks < 32u) && cols >= (uint32_t)kDenseThreads * 8u) {
    hipLaunchKernelGGL(matvec_bf16_tall_kernel, dim3(rows), dim3(kDenseThreads), 0,
                       (hipStream_t)stream, W, x, out, cols);
    AFF_HIP_CHECK(hipGetLastError());
    return;
  }
  hipLaunchKernelGGL(matvec_bf16_kernel, dim3(blocks), dim3(kDenseThreads),
                     stage ? lds : 0, (hipStream_t)stream, W, x, out, rows, cols, stage);
  AFF_HIP_CHECK(hipGetLastError());
}

// FP8-E4M3 dense matvec, one UE8M0 scale per 128x128 tile — the checkpoint's own dense format.
//
// Same shape as the bf16 kernel: one wave per row, LDS-staged activation when it fits. The scale is
// constant across a 128-wide run, so the row splits into tiles, each tile accumulates in its own
// register chain and is folded in with a single multiply. That keeps the scale off the inner FMA.
//
// E4M3 -> f32 decode: gpu/device_math.h, shared with the attention and batched paths.

// One output row of an FP8 matrix, reduced across the wave. Factored out so the single-matrix
// kernel and the batched one below are literally the same arithmetic rather than two copies that
// drift apart.
//
// 32 lanes x uint4 = 512 contiguous bytes a pass, fully coalesced. That width also fixes the
// reduction: a 512 B pass spans four scale tiles and 128/16 = 8, so lanes 0-7 land entirely inside
// the first tile, 8-15 the second, and so on. Every lane's 16 columns therefore share one scale, so
// a lane applies its own scale locally and the row needs a single wave reduction at the end rather
// than one per tile. Summation order changes — per-lane then across, rather than across then
// per-tile — but the value does not.
//
// `XV` is anything with `operator[](uint32_t) -> float`. A raw `const float*` satisfies it, and so
// does `RmsView` below, which is how the RMSNorm folds in without a second copy of this loop.
template <class XV>
__device__ __forceinline__ float fp8_row_dot(const uint8_t* __restrict__ W,
                                             const uint8_t* __restrict__ S,
                                             const XV& xv, uint32_t row,
                                             uint32_t cols, uint32_t sc_cols, int lane) {
  const uint32_t stride = kWave * 16u;
  const uint32_t full = (cols / stride) * stride;
  const uint8_t* w = W + (size_t)row * cols;
  const uint8_t* srow = S + (size_t)(row >> 7) * sc_cols;
  float acc = 0.0f;
  for (uint32_t c = (uint32_t)lane * 16u; c < full; c += stride) {
    const uint4 packed = *(const uint4*)(w + c);
    const uint32_t* pw = (const uint32_t*)&packed;
    // UE8M0: value = 2^(e-127). Built directly rather than through exp2f.
    const uint32_t e = srow[c >> 7];
    const float scale = __uint_as_float(e ? (e << 23) : 0u);
    float a0 = 0.0f, a1 = 0.0f;
    for (int k = 0; k < 4; ++k) {
      const uint32_t p = pw[k];
      const uint32_t c4 = c + (uint32_t)k * 4u;
      // Two instructions for four bytes: v_cvt_pk_f32_fp8. No precondition here, unlike the
      // expert codebook -- these are arbitrary model weights and can be subnormal, which the
      // HARDWARE handles correctly where the software fast path does not.
      const aff_f32x2 e01 = e4m3_decode_pk<false>(p);
      const aff_f32x2 e23 = e4m3_decode_pk<true>(p);
      a0 = fmaf(e01.x, xv[c4 + 0], a0);
      a1 = fmaf(e01.y, xv[c4 + 1], a1);
      a0 = fmaf(e23.x, xv[c4 + 2], a0);
      a1 = fmaf(e23.y, xv[c4 + 3], a1);
    }
    acc = fmaf(a0 + a1, scale, acc);
  }
  // Tail, when cols is not a multiple of 512. Same per-lane-scale rule, one column at a time.
  for (uint32_t c = full + (uint32_t)lane; c < cols; c += kWave) {
    const uint32_t e = srow[c >> 7];
    acc = fmaf(e4m3_decode(w[c]) * xv[c], __uint_as_float(e ? (e << 23) : 0u), acc);
  }
  return wave_reduce_add(acc);
}

// Rows a wave in the dense FP8 matvec, picked by shape. Blocking trades waves for memory-level
// parallelism, so where it lands depends on whether the kernel is waiting on loads or on the ALU.
// That makes this a property of the shape AND the inner loop, fitted to the engine's own per-rank
// shapes — re-measure it whenever that loop changes.
static uint32_t mv_rb(uint32_t rows, uint32_t cols) {
  // Wide rows already have enough loads in flight, so blocking would only remove waves.
  if (cols > 2048u) return 1u;
  // Narrow rows cannot hide a load behind two inner iterations, so block — harder the fewer rows
  // there are to make waves from.
  return rows >= 8192u ? 2u : 4u;
}

// The same dot, RB rows at a time, so one activation load feeds RB weight loads. The one-row form
// spends six memory instructions per 16 weight bytes and only one of them is the weight, which sits
// alone on the critical path; a narrow matrix runs too few inner iterations to hide that latency
// behind.
//
// RB consecutive rows share the activation window and the UE8M0 scale — `r0` is RB-aligned and RB
// divides 128, so `row>>7` cannot change inside the block. What multiplies is the weight load, which
// is the point: RB of them issue back to back and are in flight together.
//
// Summation order is untouched: each row still accumulates per lane and reduces once at the end.
template <uint32_t RB, class XV>
__device__ __forceinline__ void fp8_rows_dot(const uint8_t* __restrict__ W,
                                             const uint8_t* __restrict__ S,
                                             const XV& xv, uint32_t row0,
                                             uint32_t nr, uint32_t cols, uint32_t sc_cols,
                                             int lane, float* __restrict__ acc) {
  const uint32_t stride = kWave * 16u;
  const uint32_t full = (cols / stride) * stride;
  const uint8_t* w[RB];
  // Rows past `rows` re-read row0 rather than branching: the result is discarded by the caller, and
  // a guarded load inside the inner loop would cost more than the duplicate traffic on the one
  // ragged wave of the grid.
#pragma unroll
  for (uint32_t r = 0; r < RB; ++r) w[r] = W + (size_t)(row0 + (r < nr ? r : 0)) * cols;
  const uint8_t* srow = S + (size_t)(row0 >> 7) * sc_cols;
#pragma unroll
  for (uint32_t r = 0; r < RB; ++r) acc[r] = 0.0f;

  for (uint32_t c = (uint32_t)lane * 16u; c < full; c += stride) {
    uint4 packed[RB];
#pragma unroll
    for (uint32_t r = 0; r < RB; ++r) packed[r] = *(const uint4*)(w[r] + c);
    float xr[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) xr[k] = xv[c + (uint32_t)k];
    const uint32_t e = srow[c >> 7];
    const float scale = __uint_as_float(e ? (e << 23) : 0u);
#pragma unroll
    for (uint32_t r = 0; r < RB; ++r) {
      const uint32_t* pw = (const uint32_t*)&packed[r];
      float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const uint32_t p = pw[k];
        // Two instructions for four bytes: v_cvt_pk_f32_fp8. No precondition here, unlike the
        // expert codebook -- these are arbitrary model weights and can be subnormal, which the
        // HARDWARE handles correctly where the software fast path does not.
        const aff_f32x2 e01 = e4m3_decode_pk<false>(p);
        const aff_f32x2 e23 = e4m3_decode_pk<true>(p);
        a0 = fmaf(e01.x, xr[k * 4 + 0], a0);
        a1 = fmaf(e01.y, xr[k * 4 + 1], a1);
        a0 = fmaf(e23.x, xr[k * 4 + 2], a0);
        a1 = fmaf(e23.y, xr[k * 4 + 3], a1);
      }
      acc[r] = fmaf(a0 + a1, scale, acc[r]);
    }
  }
  for (uint32_t c = full + (uint32_t)lane; c < cols; c += kWave) {
    const uint32_t e = srow[c >> 7];
    const float sc = __uint_as_float(e ? (e << 23) : 0u);
    const float xc = xv[c];
#pragma unroll
    for (uint32_t r = 0; r < RB; ++r) acc[r] = fmaf(e4m3_decode(w[r][c]) * xc, sc, acc[r]);
  }
#pragma unroll
  for (uint32_t r = 0; r < RB; ++r) acc[r] = wave_reduce_add(acc[r]);
}

template <uint32_t RB>
__global__ __launch_bounds__(kDenseThreads) void matvec_fp8_rb_kernel(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ x,
    float* __restrict__ out, uint32_t rows, uint32_t cols, uint32_t sc_cols) {
  const int tid = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;
  const uint32_t step = gridDim.x * kDenseWaves * RB;
  for (uint32_t r0 = ((uint32_t)blockIdx.x * kDenseWaves + (uint32_t)wave) * RB; r0 < rows;
       r0 += step) {
    const uint32_t nr = min(RB, rows - r0);
    float acc[RB];
    fp8_rows_dot<RB>(W, S, x, r0, nr, cols, sc_cols, lane, acc);
    if (lane == 0) {
#pragma unroll
      for (uint32_t r = 0; r < RB; ++r)
        if (r < nr) out[r0 + r] = acc[r];
    }
  }
}

// RMSNorm folded into the matvec that reads its output. A standalone norm over one hidden vector is
// a single-block launch — one CU of the card — and the matvec then reads its result hundreds of
// times over. Every block recomputing the whole reduction instead is the cheap side of that trade:
// the re-read is all L1 after the first block, and the global round trip disappears with the launch.
//
// BIT-IDENTICAL to the standalone kernel on purpose, so the A/B is one flag and not a quality
// argument: the same thread count and strided accumulation order, the same shuffle reduction and
// serial add of the warp partials, and `x[i] * inv * w[i]` associating left exactly as it did.
struct RmsView {
  const float* __restrict__ x;
  const float* __restrict__ w;
  float inv;
  __device__ __forceinline__ float operator[](uint32_t i) const { return x[i] * inv * w[i]; }
};

// Lifted verbatim out of rms_norm_f32_kernel, which is the only thing that keeps the two from
// drifting apart in the one way that would matter. `red` is the caller's __shared__ scratch.
__device__ __forceinline__ float rms_inv_256(const float* __restrict__ x, uint32_t n, float eps,
                                             float* red) {
  const uint32_t tid = threadIdx.x, nt = blockDim.x;
  float acc = 0.0f;
  for (uint32_t i = tid; i < n; i += nt) { const float v = x[i]; acc += v * v; }
  const uint32_t lane = tid & 31, warp = tid >> 5;
  for (int o = 16; o; o >>= 1) acc += __shfl_down(acc, o);
  if (lane == 0) red[warp] = acc;
  __syncthreads();
  if (tid == 0) {
    float t = 0.0f;
    for (uint32_t k = 0; k < (nt + 31) / 32; ++k) t += red[k];
    red[0] = rsqrtf(t / (float)n + eps);
  }
  __syncthreads();
  return red[0];
}

template <uint32_t RB>
__global__ __launch_bounds__(kDenseThreads) void matvec_fp8_rn_kernel(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ x,
    const float* __restrict__ nw, float* __restrict__ out, float* __restrict__ xn, uint32_t rows,
    uint32_t cols, uint32_t sc_cols, float eps) {
  __shared__ float red[32];
  const RmsView xv{x, nw, rms_inv_256(x, cols, eps, red)};
  // The lightning indexer reads the normalised vector, but only once the context is long enough to
  // run at all. One block writes it; the other 511 would be writing the same bytes again.
  if (xn && blockIdx.x == 0)
    for (uint32_t i = threadIdx.x; i < cols; i += kDenseThreads) xn[i] = xv[i];

  const int tid = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;
  const uint32_t step = gridDim.x * kDenseWaves * RB;
  for (uint32_t r0 = ((uint32_t)blockIdx.x * kDenseWaves + (uint32_t)wave) * RB; r0 < rows;
       r0 += step) {
    const uint32_t nr = min(RB, rows - r0);
    float acc[RB];
    fp8_rows_dot<RB>(W, S, xv, r0, nr, cols, sc_cols, lane, acc);
    if (lane == 0) {
#pragma unroll
      for (uint32_t r = 0; r < RB; ++r)
        if (r < nr) out[r0 + r] = acc[r];
    }
  }
}

__global__ __launch_bounds__(kDenseThreads) void matvec_fp8_kernel(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ x,
    float* __restrict__ out, uint32_t rows, uint32_t cols, uint32_t sc_cols, int stage_x) {

  extern __shared__ float s_x[];
  const int tid = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;

  if (stage_x) {
    for (uint32_t i = tid; i < cols; i += kDenseThreads) s_x[i] = x[i];
    __syncthreads();
  }
  const float* xv = stage_x ? s_x : x;

  for (uint32_t row = blockIdx.x * kDenseWaves + wave; row < rows;
       row += gridDim.x * kDenseWaves) {
    const float total = fp8_row_dot(W, S, xv, row, cols, sc_cols, lane);
    if (lane == 0) out[row] = total;
  }
}

// Grouped variant: row i of the output reads x + (i / rank) * group_dim. The groups of the attention
// output projection are contiguous row blocks, so the whole thing is one launch over n_groups*rank
// rows instead of one launch per group.
//
// The inverse RoPE folds in here too. As its own kernel it is a handful of blocks undoing in place
// the rotation V carried — arithmetic too small to measure, in a launch that is not.
// Nothing reads a_out between that kernel and this one, so the rotation can happen on the way in.
//
// Rotated ONCE PER BLOCK into LDS, not once per row. Every row of a group reads the same activation
// vector, so rotating per row would repeat the transcendentals once for every row in the group; per
// block it is one pair a thread, once, behind a __syncthreads the block needed anyway.
//
// The BRANCH is the reason this is cheap rather than merely correct. A head is head_dim wide and
// only its last n_rot floats move, head_dim - n_rot is 16-aligned, and a lane owns 16 consecutive
// columns — so a lane's window is entirely inside the tail or entirely outside it and the test is
// per-window. At the shape that matters (cols 4096, head_dim 512) a 512-column pass is exactly one
// head, so lanes 28-31 take the LDS side and the other 28 never do.
struct RopeTail {
  uint32_t head_dim, n_rot, pos;
  uint32_t hd_shift, rot_shift, nope;     // shifts precomputed on the host; both are powers of two
  float freq_scale, theta_scale, ext_factor, lo, hi;
};
constexpr uint32_t kGroupTailMax = 1024;  // floats of LDS for one group's rotated tail

template <bool Rope>
__global__ __launch_bounds__(kDenseThreads) void matvec_fp8_grouped_kernel(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ x,
    float* __restrict__ out, uint32_t rows, uint32_t cols, uint32_t sc_cols, uint32_t rank,
    RopeTail rt) {
  const int tid = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;
  const uint32_t stride = kWave * 16u;                  // see matvec_fp8_kernel for why 16
  const uint32_t full = (cols / stride) * stride;

  __shared__ float s_tail[Rope ? kGroupTailMax : 1];
  if (Rope) {
    // One group per block: the launcher refuses unless rank % kDenseWaves == 0 and the grid covers
    // every row without striding, which together make row/rank constant inside the block.
    const float* xg = x + (size_t)(((uint32_t)blockIdx.x * kDenseWaves) / rank) * cols;
    const uint32_t hp = rt.n_rot >> 1;
    for (uint32_t t = (uint32_t)tid; t < (cols / rt.head_dim) * hp; t += kDenseThreads) {
      const uint32_t h = t / hp, k = t - h * hp;
      const float* p = xg + (size_t)h * rt.head_dim + rt.nope;
      float c, sn;
      // Sine negated: this is the INVERSE, exactly as rope_tail_kernel's -1.0f argument.
      rope_cs(k, rt.pos, rt.freq_scale, rt.theta_scale, rt.ext_factor, rt.lo, rt.hi, -1.0f, &c, &sn);
      const float x0 = p[2 * k], x1 = p[2 * k + 1];
      float* d = s_tail + (h << rt.rot_shift);
      d[2 * k] = x0 * c - x1 * sn;
      d[2 * k + 1] = x0 * sn + x1 * c;
    }
    __syncthreads();
  }

  for (uint32_t row = blockIdx.x * kDenseWaves + wave; row < rows;
       row += gridDim.x * kDenseWaves) {
    const uint8_t* w = W + (size_t)row * cols;
    const uint8_t* srow = S + (size_t)(row >> 7) * sc_cols;
    const float* xv = x + (size_t)(row / rank) * cols;
    float acc = 0.0f;
    for (uint32_t c = (uint32_t)lane * 16u; c < full; c += stride) {
      const uint4 packed = *(const uint4*)(w + c);
      const uint32_t* pw = (const uint32_t*)&packed;
      const uint32_t e = srow[c >> 7];
      const float scale = __uint_as_float(e ? (e << 23) : 0u);
      float xr[16];
      if (Rope && (c & (rt.head_dim - 1u)) >= rt.nope) {
        const float* s = s_tail + (((c >> rt.hd_shift) << rt.rot_shift) +
                                   ((c & (rt.head_dim - 1u)) - rt.nope));
#pragma unroll
        for (int k = 0; k < 16; ++k) xr[k] = s[k];
      } else {
#pragma unroll
        for (int k = 0; k < 16; ++k) xr[k] = xv[c + k];
      }
      float a0 = 0.0f, a1 = 0.0f;
      for (int k = 0; k < 4; ++k) {
        const uint32_t p = pw[k];
        // Two instructions for four bytes: v_cvt_pk_f32_fp8. No precondition here, unlike the
        // expert codebook -- these are arbitrary model weights and can be subnormal, which the
        // HARDWARE handles correctly where the software fast path does not.
        const aff_f32x2 e01 = e4m3_decode_pk<false>(p);
        const aff_f32x2 e23 = e4m3_decode_pk<true>(p);
        a0 = fmaf(e01.x, xr[k * 4 + 0], a0);
        a1 = fmaf(e01.y, xr[k * 4 + 1], a1);
        a0 = fmaf(e23.x, xr[k * 4 + 2], a0);
        a1 = fmaf(e23.y, xr[k * 4 + 3], a1);
      }
      acc = fmaf(a0 + a1, scale, acc);
    }
    // Rope never gets here: the launcher requires cols % 512 == 0, so `full` is `cols`.
    for (uint32_t c = full + (uint32_t)lane; c < cols; c += kWave) {
      const uint32_t e = srow[c >> 7];
      acc = fmaf(e4m3_decode(w[c]) * xv[c], __uint_as_float(e ? (e << 23) : 0u), acc);
    }
    const float total = wave_reduce_add(acc);
    if (lane == 0) out[row] = total;
  }
}

// Row-blocked grouped variant. RB consecutive rows share an activation window only if they are in
// the same group, which holds exactly when `rank % RB == 0` — r0 is RB-aligned, so the group index
// r0/rank cannot change inside the block. The launcher checks it rather than assuming.
template <uint32_t RB>
__global__ __launch_bounds__(kDenseThreads) void matvec_fp8_grouped_rb_kernel(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ x,
    float* __restrict__ out, uint32_t rows, uint32_t cols, uint32_t sc_cols, uint32_t rank) {
  const int tid = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;
  const uint32_t step = gridDim.x * kDenseWaves * RB;
  for (uint32_t r0 = ((uint32_t)blockIdx.x * kDenseWaves + (uint32_t)wave) * RB; r0 < rows;
       r0 += step) {
    const uint32_t nr = min(RB, rows - r0);
    float acc[RB];
    fp8_rows_dot<RB>(W, S, x + (size_t)(r0 / rank) * cols, r0, nr, cols, sc_cols, lane, acc);
    if (lane == 0) {
#pragma unroll
      for (uint32_t r = 0; r < RB; ++r)
        if (r < nr) out[r0 + r] = acc[r];
    }
  }
}

void matvec_fp8_grouped_hip(const uint8_t* W, const uint8_t* S, const float* x, uint32_t n_groups,
                            uint32_t group_dim, uint32_t rank, float* out, void* stream) {
  const uint32_t rows = n_groups * rank;
  const uint32_t rb = mv_rb(rows, group_dim);       // group_dim IS this kernel's row width
  if (rb > 1u && rank % rb == 0u) {
    const uint32_t per = kDenseWaves * rb;
    const uint32_t blocks = std::min<uint32_t>(2048u, (rows + per - 1) / per);
    const uint32_t sc = (group_dim + 127) / 128;
#define AFF_MVGRB(N)                                                                          \
  hipLaunchKernelGGL((matvec_fp8_grouped_rb_kernel<N>), dim3(blocks), dim3(kDenseThreads), 0,  \
                     (hipStream_t)stream, W, S, x, out, rows, group_dim, sc, rank)
    if (rb == 2u) { AFF_MVGRB(2); } else if (rb == 4u) { AFF_MVGRB(4); } else { AFF_MVGRB(8); }
#undef AFF_MVGRB
    AFF_HIP_CHECK(hipGetLastError());
    return;
  }
  const uint32_t blocks = std::min<uint32_t>(2048u, (rows + kDenseWaves - 1) / kDenseWaves);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(matvec_fp8_grouped_kernel<false>), dim3(blocks),
                     dim3(kDenseThreads), 0, (hipStream_t)stream, W, S, x, out, rows, group_dim,
                     (group_dim + 127) / 128, rank, RopeTail{});
  AFF_HIP_CHECK(hipGetLastError());
}

static bool pow2(uint32_t v) { return v && !(v & (v - 1)); }
static uint32_t log2u(uint32_t v) { uint32_t s = 0; while ((1u << s) < v) ++s; return s; }

bool matvec_fp8_grouped_rope_hip(const uint8_t* W, const uint8_t* S, const float* x,
                                 uint32_t n_groups, uint32_t group_dim, uint32_t rank,
                                 uint32_t head_dim, uint32_t n_rot, uint32_t pos,
                                 float freq_scale, float theta_scale, float ext_factor, float lo,
                                 float hi, float* out, void* stream) {
  const uint32_t rows = n_groups * rank;
  // Every one of these is what the kernel's LDS staging and its per-window branch ASSUME. Returning
  // false hands the caller back to rope_tail_kernel + the plain grouped matvec, which is the same
  // answer one launch slower; getting any of them wrong silently would be the same answer wrong.
  if (mv_rb(rows, group_dim) != 1u) return false;              // the RB kernel has no rope form
  if (!pow2(head_dim) || !pow2(n_rot) || n_rot > head_dim) return false;
  if (group_dim % head_dim || group_dim % (kWave * 16u)) return false;  // no ragged tail loop
  const uint32_t nope = head_dim - n_rot;
  if (nope % 16u) return false;                                // a lane's 16 columns must not split
  if ((group_dim / head_dim) * n_rot > kGroupTailMax) return false;
  if (rank % kDenseWaves) return false;                        // one group per block
  const uint32_t blocks = std::min<uint32_t>(2048u, (rows + kDenseWaves - 1) / kDenseWaves);
  if ((uint64_t)blocks * kDenseWaves < rows) return false;     // no grid stride across groups

  const RopeTail rt{head_dim, n_rot,           pos,        log2u(head_dim), log2u(n_rot), nope,
                    freq_scale, theta_scale,   ext_factor, lo,              hi};
  hipLaunchKernelGGL(HIP_KERNEL_NAME(matvec_fp8_grouped_kernel<true>), dim3(blocks),
                     dim3(kDenseThreads), 0, (hipStream_t)stream, W, S, x, out, rows, group_dim,
                     (group_dim + 127) / 128, rank, rt);
  AFF_HIP_CHECK(hipGetLastError());
  return true;
}

// Several dense matvecs sharing one x, in one launch. The grid is sized from the total rows and a
// wave finds its matrix by walking a prefix sum held in registers. A short matrix on its own asks
// for too few blocks to fill the card; batched with its siblings it contributes to a grid that does,
// and the launch count falls to one at the same time.
//
// Descriptors travel as a kernel argument rather than through a device buffer — a struct under the
// 4 KiB argument limit costs nothing, where an allocation and a synchronous copy per dispatch would
// dominate a kernel this size.
struct MvGuArgs {
  const void* wg; const uint8_t* sg; uint32_t quant_g;
  const void* wu; const uint8_t* su; uint32_t quant_u;
  float* out;
  uint32_t rows, cols, sc_cols;
  float limit;
};

struct MvBatchArgs {
  const void* data[kMvBatchMax];
  const uint8_t* scales[kMvBatchMax];
  float* out[kMvBatchMax];
  uint32_t row_end[kMvBatchMax];       // prefix sum, so a global row index locates its matrix
  uint32_t quant[kMvBatchMax];
  float scale[kMvBatchMax];            // folded into the store; 1.0f for every caller but idx_proj
  uint32_t n_jobs, cols, sc_cols, total_rows;
  // The optional side copy — see MvStage. Null dst is the common case and costs one uniform branch.
  const float* stage_src;
  float* stage_dst;
  uint32_t stage_n, stage_stride;
};

__global__ __launch_bounds__(kDenseThreads) void matvec_batch_kernel(MvBatchArgs a,
                                                                    const float* __restrict__ x) {
  const int tid = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;

  for (uint32_t g = blockIdx.x * kDenseWaves + wave; g < a.total_rows;
       g += gridDim.x * kDenseWaves) {
    // Which matrix owns global row g. At most eight jobs, so a scan beats anything cleverer, and
    // every lane of the wave takes the same branch — the whole wave is on one row.
    uint32_t j = 0;
    while (g >= a.row_end[j]) ++j;
    const uint32_t row = g - (j ? a.row_end[j - 1] : 0u);
    const float total = a.quant[j] == AFF_FP8_E4M3
        ? fp8_row_dot((const uint8_t*)a.data[j], a.scales[j], x, row, a.cols, a.sc_cols, lane)
        : bf16_row_dot((const uint16_t*)a.data[j], x, row, a.cols, lane);
    if (lane == 0) a.out[j][row] = total * a.scale[j];
  }

  // Block 0 also carries the side copy: a few iterations per thread against a row loop that reads
  // kilobytes a wave, so it disappears into the matvec it rides on.
  if (a.stage_dst && blockIdx.x == 0)
    for (uint32_t i = (uint32_t)tid; i < a.stage_n; i += (uint32_t)blockDim.x)
      a.stage_dst[(size_t)i * a.stage_stride] = a.stage_src[i];
}

// One weight row against T activation vectors, T accumulators, one pass over the weight.
//
// The point is what does NOT change: each of the T accumulators runs the same four partials in the
// same fold order as `bf16_row_dot`, so this is bit-identical to T separate calls. What moves is the
// weight — the unpack hoists out of the token loop, so the row is read once for the group rather
// than once per token. A WMMA GEMM cannot do that cheaply at small T, because its N tile is a floor:
// a few-token GEMM computes a full-width tile and throws most of the arithmetic away.
//
// The cost is registers — 4T partials plus the unpacked weights — which is what sets the T cap.
template <uint32_t T>
__device__ __forceinline__ void bf16_row_dot_multix(const uint16_t* __restrict__ W,
                                                    const float* __restrict__ x, uint32_t xs,
                                                    uint32_t row, uint32_t cols, int lane,
                                                    float* __restrict__ out) {
  const uint32_t stride = kWave * 8u;
  const uint32_t full = (cols / stride) * stride;
  const uint16_t* w = W + (size_t)row * cols;
  float a0[T], a1[T], a2[T], a3[T];
#pragma unroll
  for (uint32_t t = 0; t < T; ++t) { a0[t] = 0.0f; a1[t] = 0.0f; a2[t] = 0.0f; a3[t] = 0.0f; }
  for (uint32_t c = (uint32_t)lane * 8u; c < full; c += stride) {
    const uint4 packed = *(const uint4*)(w + c);
    const uint32_t* pw = (const uint32_t*)&packed;
    const float w0 = __uint_as_float((pw[0] & 0xFFFFu) << 16);
    const float w1 = __uint_as_float(pw[0] & 0xFFFF0000u);
    const float w2 = __uint_as_float((pw[1] & 0xFFFFu) << 16);
    const float w3 = __uint_as_float(pw[1] & 0xFFFF0000u);
    const float w4 = __uint_as_float((pw[2] & 0xFFFFu) << 16);
    const float w5 = __uint_as_float(pw[2] & 0xFFFF0000u);
    const float w6 = __uint_as_float((pw[3] & 0xFFFFu) << 16);
    const float w7 = __uint_as_float(pw[3] & 0xFFFF0000u);
#pragma unroll
    for (uint32_t t = 0; t < T; ++t) {
      // Two float4 loads, not eight scalar ones. Same eight values in the same eight fmas, so it is
      // bit-identical; `c` is a multiple of 8 and `xs` is n_embd, so both are 32-byte aligned.
      //
      // THE COST HERE IS x, NOT THE WEIGHT. A wave owns one row and re-reads all T activation
      // vectors for it, so the working set grows with T while the weight traffic does not. There is
      // a T above which that set no longer fits in cache, and crossing it costs far more than the
      // extra arithmetic is worth — which is what `kMvMultiXFast` caps, and where a caller chooses
      // between this and the GEMM.
      const float* xv = x + (size_t)t * xs + c;
      const float4 xa = *(const float4*)(xv);
      const float4 xb = *(const float4*)(xv + 4);
      a0[t] = fmaf(w0, xa.x, a0[t]);
      a0[t] = fmaf(w1, xa.y, a0[t]);
      a1[t] = fmaf(w2, xa.z, a1[t]);
      a1[t] = fmaf(w3, xa.w, a1[t]);
      a2[t] = fmaf(w4, xb.x, a2[t]);
      a2[t] = fmaf(w5, xb.y, a2[t]);
      a3[t] = fmaf(w6, xb.z, a3[t]);
      a3[t] = fmaf(w7, xb.w, a3[t]);
    }
  }
  for (uint32_t c = full + (uint32_t)lane; c < cols; c += kWave) {
    const float wv = __uint_as_float((uint32_t)w[c] << 16);
#pragma unroll
    for (uint32_t t = 0; t < T; ++t) a0[t] = fmaf(wv, x[(size_t)t * xs + c], a0[t]);
  }
#pragma unroll
  for (uint32_t t = 0; t < T; ++t) out[t] = wave_reduce_add((a0[t] + a1[t]) + (a2[t] + a3[t]));
}

template <uint32_t T>
__global__ __launch_bounds__(kDenseThreads) void matvec_multix_kernel(MvBatchArgs a,
                                                                     const float* __restrict__ x,
                                                                     uint32_t xs) {
  const int tid = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;
  for (uint32_t g = blockIdx.x * kDenseWaves + wave; g < a.total_rows;
       g += gridDim.x * kDenseWaves) {
    uint32_t j = 0;
    while (g >= a.row_end[j]) ++j;
    const uint32_t base = j ? a.row_end[j - 1] : 0u;
    const uint32_t row = g - base, rows_j = a.row_end[j] - base;
    float acc[T];
    bf16_row_dot_multix<T>((const uint16_t*)a.data[j], x, xs, row, a.cols, lane, acc);
    if (lane == 0) {
      // Token-major, so the pool reads token t's whole projection as one contiguous row and needs
      // no transpose behind this at all.
#pragma unroll
      for (uint32_t t = 0; t < T; ++t) a.out[j][(size_t)t * rows_j + row] = acc[t] * a.scale[j];
    }
  }
}

bool matvec_multix_hip(const MvJob* jobs, uint32_t n_jobs, const float* x, uint32_t cols,
                       uint32_t nt, uint32_t x_stride, void* stream) {
  if (!n_jobs || n_jobs > kMvBatchMax || !nt || nt > kMvMultiXMax) return false;
  for (uint32_t i = 0; i < n_jobs; ++i)
    if (jobs[i].quant != AFF_BF16) return false;
  MvBatchArgs a{};
  uint32_t acc = 0;
  for (uint32_t i = 0; i < n_jobs; ++i) {
    a.data[i] = jobs[i].data;
    a.scales[i] = jobs[i].scales;
    a.out[i] = jobs[i].out;
    a.quant[i] = jobs[i].quant;
    a.scale[i] = jobs[i].scale;
    acc += jobs[i].rows;
    a.row_end[i] = acc;
  }
  for (uint32_t i = n_jobs; i < kMvBatchMax; ++i) a.row_end[i] = 0xFFFFFFFFu;
  a.n_jobs = n_jobs;
  a.cols = cols;
  a.sc_cols = (cols + 127) / 128;
  a.total_rows = acc;
  const uint32_t blocks = std::min<uint32_t>(2048u, (acc + kDenseWaves - 1) / kDenseWaves);
#define AFF_MULTIX(T)                                                                             \
  case T:                                                                                         \
    hipLaunchKernelGGL((matvec_multix_kernel<T>), dim3(blocks), dim3(kDenseThreads), 0,            \
                       (hipStream_t)stream, a, x, x_stride); AFF_HIP_CHECK(hipGetLastError());                                      \
    break
  switch (nt) {
    AFF_MULTIX(1); AFF_MULTIX(2); AFF_MULTIX(3); AFF_MULTIX(4);
    AFF_MULTIX(5); AFF_MULTIX(6); AFF_MULTIX(7); AFF_MULTIX(8);
    default: return false;
  }
#undef AFF_MULTIX
  AFF_HIP_CHECK(hipGetLastError());
  return true;
}

// Gate and up share `x` and the row index, so one wave takes both dots and applies the SwiGLU on the
// spot. Separately they are a batched matvec followed by an elementwise kernel over a few thousand
// floats — far too little work to be worth a dependent launch.
//
// The two row dots cost twice the registers of one, which is fine here: the wave still owns one row
// and nothing crosses lanes until the reductions at the end.
__global__ __launch_bounds__(kDenseThreads) void mv_gate_up_swiglu_kernel(
    MvGuArgs a, const float* __restrict__ x) {
  const int tid = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;
  for (uint32_t row = blockIdx.x * kDenseWaves + wave; row < a.rows;
       row += gridDim.x * kDenseWaves) {
    const float g = a.quant_g == AFF_FP8_E4M3
        ? fp8_row_dot((const uint8_t*)a.wg, a.sg, x, row, a.cols, a.sc_cols, lane)
        : bf16_row_dot((const uint16_t*)a.wg, x, row, a.cols, lane);
    const float u = a.quant_u == AFF_FP8_E4M3
        ? fp8_row_dot((const uint8_t*)a.wu, a.su, x, row, a.cols, a.sc_cols, lane)
        : bf16_row_dot((const uint16_t*)a.wu, x, row, a.cols, lane);
    if (lane == 0) {
      float gv = g, uv = u;
      // Identical to the standalone SwiGLU above, deliberately: this is the same arithmetic moved,
      // not a new formulation, and the two must not drift.
      if (a.limit > 0.0f) { gv = fminf(gv, a.limit); uv = fminf(a.limit, fmaxf(-a.limit, uv)); }
      a.out[row] = (gv / (1.0f + expf(-gv))) * uv;
    }
  }
}

void mv_gate_up_swiglu_hip(const void* wg, const uint8_t* sg, uint32_t quant_g, const void* wu,
                           const uint8_t* su, uint32_t quant_u, const float* x, float* out,
                           uint32_t rows, uint32_t cols, float limit, void* stream) {
  MvGuArgs a{};
  a.wg = wg; a.sg = sg; a.quant_g = quant_g;
  a.wu = wu; a.su = su; a.quant_u = quant_u;
  a.out = out; a.rows = rows; a.cols = cols;
  a.sc_cols = (cols + 127) / 128;
  a.limit = limit;
  const uint32_t blocks = std::min<uint32_t>(2048u, (rows + kDenseWaves - 1) / kDenseWaves);
  hipLaunchKernelGGL(mv_gate_up_swiglu_kernel, dim3(blocks), dim3(kDenseThreads), 0,
                     (hipStream_t)stream, a, x);
  AFF_HIP_CHECK(hipGetLastError());
}

void matvec_batch_hip(const MvJob* jobs, uint32_t n_jobs, const float* x, uint32_t cols,
                      void* stream, const MvStage* stage) {
  if (!n_jobs || n_jobs > kMvBatchMax) return;
  MvBatchArgs a{};
  if (stage && stage->dst && stage->n) {
    a.stage_src = stage->src; a.stage_dst = stage->dst;
    a.stage_n = stage->n;     a.stage_stride = stage->stride;
  }
  uint32_t acc = 0;
  for (uint32_t i = 0; i < n_jobs; ++i) {
    a.data[i] = jobs[i].data;
    a.scales[i] = jobs[i].scales;
    a.out[i] = jobs[i].out;
    a.quant[i] = jobs[i].quant;
    a.scale[i] = jobs[i].scale;
    acc += jobs[i].rows;
    a.row_end[i] = acc;
  }
  // The scan in the kernel is unguarded, so every index below total_rows must terminate it. Pad the
  // unused slots past the end rather than bounds-checking on the hot path.
  for (uint32_t i = n_jobs; i < kMvBatchMax; ++i) a.row_end[i] = 0xFFFFFFFFu;
  a.n_jobs = n_jobs;
  a.cols = cols;
  a.sc_cols = (cols + 127) / 128;
  a.total_rows = acc;
  const uint32_t blocks = std::min<uint32_t>(2048u, (acc + kDenseWaves - 1) / kDenseWaves);
  hipLaunchKernelGGL(matvec_batch_kernel, dim3(blocks), dim3(kDenseThreads), 0,
                     (hipStream_t)stream, a, x);
  AFF_HIP_CHECK(hipGetLastError());
}


void matvec_fp8_hip(const uint8_t* W, const uint8_t* S, const float* x, uint32_t rows,
                    uint32_t cols, float* out, void* stream) {
  // NO LDS STAGING, deliberately. With uint4 loads a lane reads 16 consecutive activations, so 16
  // lanes land on the same LDS bank and the conflicts cost more than the staging saves. `x` is small
  // enough that every wave in the block keeps it in L1 regardless.
  const uint32_t rb = mv_rb(rows, cols);
  if (rb > 1u) {
    const uint32_t per = kDenseWaves * rb;
    const uint32_t blocks = std::min<uint32_t>(2048u, (rows + per - 1) / per);
    const uint32_t sc = (cols + 127) / 128;
#define AFF_MVRB(N)                                                                          \
  hipLaunchKernelGGL((matvec_fp8_rb_kernel<N>), dim3(blocks), dim3(kDenseThreads), 0,         \
                     (hipStream_t)stream, W, S, x, out, rows, cols, sc)
    if (rb == 2u) { AFF_MVRB(2); } else if (rb == 4u) { AFF_MVRB(4); } else { AFF_MVRB(8); }
#undef AFF_MVRB
    AFF_HIP_CHECK(hipGetLastError());
    return;
  }
  const uint32_t blocks = std::min<uint32_t>(2048u, (rows + kDenseWaves - 1) / kDenseWaves);
  hipLaunchKernelGGL(matvec_fp8_kernel, dim3(blocks), dim3(kDenseThreads),
                     0, (hipStream_t)stream, W, S, x, out, rows, cols,
                     (cols + 127) / 128, 0);
  AFF_HIP_CHECK(hipGetLastError());
}

// out = W * rmsnorm(x, nw), in one launch. `xn` may be null; when it is not, the normalised vector
// is also written there for whoever else wanted it.
//
// The grid comes out of the SAME mv_rb as the unfused path, so the fold cannot silently re-block
// the matvec underneath itself — the A/B stays a fold and not a fold plus a geometry change.
void matvec_fp8_rmsnorm_hip(const uint8_t* W, const uint8_t* S, const float* x, const float* nw,
                            float eps, uint32_t rows, uint32_t cols, float* out, float* xn,
                            void* stream) {
  const uint32_t rb = mv_rb(rows, cols);
  const uint32_t per = kDenseWaves * rb;
  const uint32_t blocks = std::min<uint32_t>(2048u, (rows + per - 1) / per);
  const uint32_t sc = (cols + 127) / 128;
#define AFF_MVRN(N)                                                                          \
  hipLaunchKernelGGL((matvec_fp8_rn_kernel<N>), dim3(blocks), dim3(kDenseThreads), 0,         \
                     (hipStream_t)stream, W, S, x, nw, out, xn, rows, cols, sc, eps); AFF_HIP_CHECK(hipGetLastError())
  if (rb == 1u) { AFF_MVRN(1); }
  else if (rb == 2u) { AFF_MVRN(2); }
  else if (rb == 4u) { AFF_MVRN(4); }
  else { AFF_MVRN(8); }
#undef AFF_MVRN
  AFF_HIP_CHECK(hipGetLastError());
}

void expert_ffn_mono_hip(const DeviceExpert* experts, const float* weights, uint32_t n_experts,
                         const float* cb, const float* x, float swiglu_limit,
                         float* scratch, float* out, void* stream) {
  aff_tr_guard("expert_ffn_mono_hip");
  if (!n_experts || n_experts > kMaxBatch) return;
  MonoArgs a{};
  for (uint32_t k = 0; k < n_experts; ++k) {
    a.gate.data[k] = experts[k].gate.data; a.gate.scales[k] = experts[k].gate.scales;
    a.gate.rscales[k] = experts[k].gate.rscales;
    a.up.data[k] = experts[k].up.data; a.up.scales[k] = experts[k].up.scales;
    a.up.rscales[k] = experts[k].up.rscales;
    a.down.data[k] = experts[k].down.data; a.down.scales[k] = experts[k].down.scales;
    a.down.rscales[k] = experts[k].down.rscales;
  }
  a.weights = weights; a.cb = cb; a.x = x; a.scratch = scratch; a.out = out;
  a.n_experts = n_experts;
  a.inter = experts[0].gate.rows;
  a.hidden = experts[0].gate.cols;
  a.limit = swiglu_limit;

  // The grid MUST fit on the device at once or grid.sync() deadlocks, so it is sized from the
  // occupancy this kernel actually achieves — never from a guess. Cached per device, and per
  // device is the point: HIP device 0 need not be one of ours (see gfx12_devices), and the two
  // cards do not have to be the same part.
  int dev = 0;
  AFF_HIP_CHECK(hipGetDevice(&dev));
  if (dev < 0 || dev >= kMaxGpuDevices) {
    aff::ui::fatal("device ordinal %d is outside kMaxGpuDevices (%d)\n", dev, kMaxGpuDevices);
    std::abort();
  }
  static int grid_blocks[kMaxGpuDevices] = {};
  if (!grid_blocks[dev]) {
    int per_wgp = 0;
    AFF_HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_wgp, (const void*)expert_ffn_mono_kernel, kThreads, 0));
    hipDeviceProp_t pr{};
    AFF_HIP_CHECK(hipGetDeviceProperties(&pr, dev));
    grid_blocks[dev] = per_wgp * pr.multiProcessorCount;
    if (grid_blocks[dev] < 1) grid_blocks[dev] = 1;
  }

  AFF_HIP_CHECK(hipMemsetAsync(scratch, 0, (size_t)2 * n_experts * a.inter * sizeof(float),
                               (hipStream_t)stream));
  void* args[] = {&a};
  AFF_HIP_CHECK(hipLaunchCooperativeKernel((const void*)expert_ffn_mono_kernel,
                                           dim3(grid_blocks[dev]), dim3(kThreads), args, 0,
                                           (hipStream_t)stream));
}

void expert_gemv_multix_hip(const DeviceMatrix* mats, const float* const* xs, const float* weights,
                            uint32_t n_mats, const float* cb, float* y, void* stream) {
  if (!n_mats) return;
  const uint32_t rows = mats[0].rows, cols = mats[0].cols;
  // expert_gemv_kernel_tr already reads p.xs[m] per matrix, so it is the multix form — a shared-x
  // caller just passes the same pointer n times. One kernel covers both cases here, where the
  // row-major path needs two that differ in one line.
  const bool tr = expert_planes_are_colgroup_major();
  const uint32_t rows_per_block = tr ? kWave : kWavesPerBlock;
  const uint32_t blocks = (rows + rows_per_block - 1) / rows_per_block;
  for (uint32_t off = 0; off < n_mats; off += kMaxBatch) {
    const uint32_t n = std::min<uint32_t>(kMaxBatch, n_mats - off);
    BatchPtrs p{};
    for (uint32_t i = 0; i < n; ++i) {
      p.data[i] = mats[off + i].data;
      p.scales[i] = mats[off + i].scales;
      p.rscales[i] = mats[off + i].rscales;
      p.xs[i] = xs[off + i];
    }
    if (tr) {
      // Same wire-format switch as expert_gemv_batch_hip; see the comment there.
      const uint32_t fmt = expert_wire_format();
#define AFF_GTRX(FV)                                                                              \
  hipLaunchKernelGGL((expert_gemv_kernel_tr<FV>), dim3(blocks), dim3(kThreads), 0,                 \
                     (hipStream_t)stream, p, weights + off, n, cb, y, rows, cols); AFF_HIP_CHECK(hipGetLastError())
      if (fmt == kFmtQuad2S) { AFF_GTRX(kFmtQuad2S); }
      else if (fmt == kFmtQuad4S) { AFF_GTRX(kFmtQuad4S); }
      else if (fmt == kFmtQuad4) { AFF_GTRX(kFmtQuad4); }
      else if (fmt == kFmtQuad8) { AFF_GTRX(kFmtQuad8); }
      else { AFF_GTRX(kFmtPair16); }
#undef AFF_GTRX
    } else
      hipLaunchKernelGGL(expert_gemv_multix_kernel, dim3(blocks), dim3(kThreads), 0,
                         (hipStream_t)stream, p, weights + off, n, cb, y, rows, cols);
    AFF_HIP_CHECK(hipGetLastError());
  }
}

// n_mats matrices of the SAME shape, each reading its own x and writing its own y. See
// expert_gemv_kernel_tr_split for why decode wants this and the accumulating form cannot serve it.
//
// Returns false when the planes are not colgroup-major — the row-major layout has no split kernel
// and the caller falls back to a launch per matrix, which is what it did for every layout before.
bool expert_gemv_split_hip(const DeviceMatrix* mats, const float* const* xs, float* const* ys,
                           const float* weights, uint32_t n_mats, const float* cb, void* stream) {
  if (!n_mats) return true;
  if (!expert_planes_are_colgroup_major()) return false;
  const uint32_t rows = mats[0].rows, cols = mats[0].cols;
  // RB=1: blocking rows measures worse here, unlike the dense FP8 matvec. The two kernels differ in
  // what the activation load is — there each lane reads its own 16 columns, so sharing the window
  // across rows removes real traffic, while here every lane wants the same values and L0 already
  // broadcasts them. Blocking would remove nothing and pay RB times the code registers.
  constexpr uint32_t rb = 1u;
  // A block owns rb*kWave rows; anything that would leave the last block ragged falls back, because
  // the ragged lanes re-read row0 and a STORE (unlike the accumulate) would then write it twice.
  const uint32_t use_rb = (rows % (rb * kWave) == 0) ? rb : 1u;
  const uint32_t nb = rows / (use_rb * kWave);
  for (uint32_t off = 0; off < n_mats; off += kMaxBatch) {
    const uint32_t n = std::min<uint32_t>(kMaxBatch, n_mats - off);
    BatchPtrs p{};
    for (uint32_t i = 0; i < n; ++i) {
      // Same shape for every member, or the (matrix, row) grid maps to the wrong plane. A mixed
      // batch here would read one expert's bytes with another's stride and still finish.
      if (mats[off + i].rows != rows || mats[off + i].cols != cols) return false;
      p.data[i] = mats[off + i].data;
      p.scales[i] = mats[off + i].scales;
      p.rscales[i] = mats[off + i].rscales;
      p.xs[i] = xs[off + i];
      p.ys[i] = ys[off + i];
    }
    const uint32_t blocks = n * nb;
    const uint32_t fmt = expert_wire_format();
#define AFF_GSPL(FV, RV)                                                                          \
  hipLaunchKernelGGL((expert_gemv_kernel_tr_split<FV, RV>), dim3(blocks), dim3(kThreads), 0,       \
                     (hipStream_t)stream, p, weights + off, n, cb, rows, cols); AFF_HIP_CHECK(hipGetLastError())
#define AFF_GSPR(FV)                                                                              \
  do { if (use_rb == 4u) { AFF_GSPL(FV, 4); }                                                     \
       else if (use_rb == 2u) { AFF_GSPL(FV, 2); }                                                \
       else { AFF_GSPL(FV, 1); } } while (0)
    if (fmt == kFmtQuad2S) { AFF_GSPR(kFmtQuad2S); }
    else if (fmt == kFmtQuad4S) { AFF_GSPR(kFmtQuad4S); }
    else if (fmt == kFmtQuad4) { AFF_GSPR(kFmtQuad4); }
    else if (fmt == kFmtQuad8) { AFF_GSPR(kFmtQuad8); }
    else { AFF_GSPR(kFmtPair16); }
#undef AFF_GSPR
#undef AFF_GSPL
    AFF_HIP_CHECK(hipGetLastError());
  }
  return true;
}

// ---------------------------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------------------------

bool hip_available() { return !gfx12_devices().empty(); }

std::string hip_device_summary() {
  const std::vector<int> devs = gfx12_devices();
  if (devs.empty()) return "no gfx12 devices";
  std::string out;
  for (int i : devs) {
    DeviceInfo d;
    std::string err;
    if (!hip_device_info(i, &d, &err)) { out += err + "\n"; continue; }
    char buf[256];
    std::snprintf(buf, sizeof(buf), "gpu%d %s  %d CUs  %.1f/%.1f GiB free  %d KiB LDS/block\n",
                  d.index, d.arch.c_str(), d.cus, d.vram_free / 1073741824.0,
                  d.vram_total / 1073741824.0, d.lds_per_block / 1024);
    out += buf;
  }
  return out;
}

void expert_gemv_batch_hip(const DeviceMatrix* mats, const float* weights, uint32_t n_mats,
                           const float* cb, const float* x, float* y, void* stream) {
  if (!n_mats) return;
  const uint32_t rows = mats[0].rows, cols = mats[0].cols;
  const bool tr = expert_planes_are_colgroup_major();
  // A block covers ONE row per wave row-major, and 32 rows (all waves splitting the columns)
  // colgroup-major. Getting this wrong is not a fault — it is a grid that silently covers a
  // fraction of the matrix.
  const uint32_t rows_per_block = tr ? kWave : kWavesPerBlock;
  const uint32_t blocks = (rows + rows_per_block - 1) / rows_per_block;

  for (uint32_t off = 0; off < n_mats; off += kMaxBatch) {
    const uint32_t n = std::min<uint32_t>(kMaxBatch, n_mats - off);
    BatchPtrs p{};
    for (uint32_t i = 0; i < n; ++i) {
      p.data[i] = mats[off + i].data;
      p.scales[i] = mats[off + i].scales;
      p.rscales[i] = mats[off + i].rscales;
      p.xs[i] = x;
    }
    // `weights` is a device pointer, so the kernel still reads it from memory; only the pointer
    // TABLES move into arguments. That keeps the router weights where the caller put them.
    // The layout wins over the variant knob: the older variants all read row-major and would
    // return silently wrong numbers against a colgroup-major slab.
    if (tr) {
      // Same wire-format switch as the batched GEMM, and for the same reason: decode reads the
      // very same planes through this kernel, so a container the GEMM understands and this does
      // not is silently wrong output rather than a failure.
      const uint32_t fmt = expert_wire_format();
#define AFF_GTR(FV)                                                                               \
  hipLaunchKernelGGL((expert_gemv_kernel_tr<FV>), dim3(blocks), dim3(kThreads), 0,                 \
                     (hipStream_t)stream, p, weights + off, n, cb, y, rows, cols); AFF_HIP_CHECK(hipGetLastError())
      if (fmt == kFmtQuad2S) { AFF_GTR(kFmtQuad2S); }
      else if (fmt == kFmtQuad4S) { AFF_GTR(kFmtQuad4S); }
      else if (fmt == kFmtQuad4) { AFF_GTR(kFmtQuad4); }
      else if (fmt == kFmtQuad8) { AFF_GTR(kFmtQuad8); }
      else { AFF_GTR(kFmtPair16); }
#undef AFF_GTR
      AFF_HIP_CHECK(hipGetLastError());
      continue;
    }
    // Row-major, which only a test or a bench that builds its own planes ever asks for.
    hipLaunchKernelGGL(expert_gemv_kernel_v8, dim3(blocks), dim3(kThreads), 0, (hipStream_t)stream,
                       p, weights + off, n, cb, y, rows, cols);
    AFF_HIP_CHECK(hipGetLastError());
  }
  // No synchronise here. Work is ordered on the stream, and the caller's download (or explicit
  // sync) is what needs to wait — not every intermediate stage of an FFN.
}

void expert_ffn_hip(const DeviceExpert* experts, const float* weights, uint32_t n_experts,
                    const float* cb, const float* x, float swiglu_limit,
                    float* scratch, float* out, void* stream) {
  if (!n_experts) return;
  // The staged path is the default, not the megakernel. The megakernel's overhead is fixed — two
  // grid.sync barriers and a grid sized to what fits resident rather than to the work — so it wins
  // only while the per-launch cost is a large share of the FFN. Faster row kernels shrank the work
  // it overlaps until that stopped being true. It is kept and still verified bit-identical, because
  // the balance tips back as soon as the FFN grows: more experts, or a batch above one.
  const uint32_t inter = experts[0].gate.rows;
  hipStream_t st = (hipStream_t)stream;

  // Unit weights for gate/up: the router weight belongs on the DOWN projection, where it scales an
  // expert's whole contribution exactly once. One persistent device float, not a per-call malloc.
  // Keyed by device. A single function-local static is a device-0 pointer that gets dereferenced
  // on device 1 the moment placement spreads a token's experts across both cards — a page fault
  // that no single-GPU test can reach, and whose address is identical every run because the
  // allocation is tiny and deterministic.
  static DeviceBuffer<float> ones[kMaxGpuDevices];
  int cur_dev = 0;
  AFF_HIP_CHECK(hipGetDevice(&cur_dev));
  DeviceBuffer<float>& ones_d = ones[cur_dev % kMaxGpuDevices];
  if (ones_d.size() < 2 * kMaxBatch) {
    ones_d.alloc(2 * kMaxBatch);
    const std::vector<float> o(2 * kMaxBatch, 1.0f);
    ones_d.upload(o.data(), o.size());
  }

  // Three of the four stages collapse to one launch each:
  //   * gate and up write their own intermediates, so they cannot accumulate into one vector the way
  //     `down` does — but they can share a launch, which is what expert_gemv_split_hip does. Without
  //     it this is two launches per expert, each on a matrix too small to reach streaming bandwidth.
  //   * swiglu is elementwise over every expert           -> one launch
  //   * down accumulates all experts into the same vector -> one launch
  //
  // The split kernel stores rather than accumulates, which also retires the memset that had to zero
  // the whole scratch buffer first.
  std::vector<DeviceMatrix> gu(2 * (size_t)n_experts);
  std::vector<const float*> gu_x(2 * (size_t)n_experts, x);
  std::vector<float*> gu_y(2 * (size_t)n_experts);
  for (uint32_t k = 0; k < n_experts; ++k) {
    float* g = scratch + (size_t)(2 * k) * inter;
    gu[2 * k] = experts[k].gate;     gu_y[2 * k] = g;
    gu[2 * k + 1] = experts[k].up;   gu_y[2 * k + 1] = g + inter;
  }
  if (!expert_gemv_split_hip(gu.data(), gu_x.data(), gu_y.data(), ones_d.get(),
                             (uint32_t)gu.size(), cb, stream)) {
    // Row-major planes have no split kernel. The old shape, memset included, because the
    // accumulating form it falls back to needs a zeroed destination.
    AFF_HIP_CHECK(hipMemsetAsync(scratch, 0, (size_t)2 * n_experts * inter * sizeof(float), st));
    for (uint32_t k = 0; k < n_experts; ++k) {
      float* g = scratch + (size_t)(2 * k) * inter;
      expert_gemv_batch_hip(&experts[k].gate, ones_d.get(), 1, cb, x, g, stream);
      expert_gemv_batch_hip(&experts[k].up, ones_d.get(), 1, cb, x, g + inter, stream);
    }
  }
  {
    const uint32_t threads = 256;
    const uint32_t total = n_experts * inter;
    const uint32_t blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(swiglu_all_kernel, dim3(blocks), dim3(threads), 0, st,
                       scratch, n_experts, inter, swiglu_limit);
    AFF_HIP_CHECK(hipGetLastError());
  }
  std::vector<DeviceMatrix> downs(n_experts);
  std::vector<const float*> acts(n_experts);
  for (uint32_t k = 0; k < n_experts; ++k) {
    downs[k] = experts[k].down;
    acts[k] = scratch + (size_t)(2 * k) * inter;
  }
  expert_gemv_multix_hip(downs.data(), acts.data(), weights, n_experts, cb, out, stream);
}

void expert_gemv_hip(const DeviceMatrix& m, const float* cb, const float* x, float w, float* y,
                     void* stream) {
  static DeviceBuffer<float> wbuf[kMaxGpuDevices];   // per device, see expert_ffn_hip
  int cur_dev = 0;
  AFF_HIP_CHECK(hipGetDevice(&cur_dev));
  DeviceBuffer<float>& wb = wbuf[cur_dev % kMaxGpuDevices];
  if (wb.size() < 1) wb.alloc(1);
  AFF_HIP_CHECK(hipMemcpy(wb.get(), &w, sizeof(float), hipMemcpyHostToDevice));
  expert_gemv_batch_hip(&m, wb.get(), 1, cb, x, y, stream);
  AFF_HIP_CHECK(hipStreamSynchronize((hipStream_t)stream));
}

void expert_swiglu_quant_hip(const uint16_t* g, const uint16_t* u, const float* w, uint8_t* out,
                             float* scales, uint32_t dim, uint32_t total, float limit,
                             void* stream) {
  if (!dim || !total) return;
  // Enough bands to keep every CU busy when the slot axis alone cannot: ~64 blocks, capped at the
  // band count. A prefill chunk already has more than that along the slot axis and takes gy = 1.
  const uint32_t slot_blocks = (total + kSQTok - 1) / kSQTok;
  const uint32_t bands = dim >> 7;
  const uint32_t gy = bands ? std::min(bands, std::max(1u, 64u / std::max(1u, slot_blocks))) : 1u;
  hipLaunchKernelGGL(expert_swiglu_quant_kernel, dim3(slot_blocks, gy),
                     dim3(kSQThreads), 0, (hipStream_t)stream, g, u, w, out, scales, dim, total,
                     limit);
  AFF_HIP_CHECK(hipGetLastError());
}

void expert_scatter_slots_hip(const uint16_t* y, float* block, const uint32_t* ord,
                              const uint32_t* run, uint32_t dim, uint32_t nb_tok,
                              uint32_t nb_stride, uint32_t total, void* stream) {
  if (!dim || !nb_tok || !total) return;
  hipLaunchKernelGGL(expert_scatter_slots_kernel, dim3(dim), dim3(256), 0, (hipStream_t)stream, y,
                     block, ord, run, nb_tok, nb_stride, total);
  AFF_HIP_CHECK(hipGetLastError());
}

// ---- the device weight layout ----------------------------------------------------------------
//
// One switch, consulted by the loader and by every kernel that touches the two planes. They MUST
// agree: a mismatch is not a crash, it is silently wrong numbers, because both layouts are the same
// size and every address stays in bounds. Colgroup-major is what the engine uses; the setter exists
// for tests and tools that build their own row-major planes.
static int g_colgroup_major = 1;

bool expert_planes_are_colgroup_major() { return g_colgroup_major != 0; }

void expert_set_plane_layout_colgroup(bool on) { g_colgroup_major = on ? 1 : 0; }

// ---- the wire format -------------------------------------------------------------------------
//
// Which of the five decode paths the packed codes are. Unlike the plane layout this is NOT a
// preference — it is a property of the container, and reading a 5-bit-pair model as 10-bit quads
// (or the reverse) is in-bounds, right-sized and completely wrong. The loader sets it from the
// quant id; a bench with no loader must set it for itself before it builds a plane.
static int g_wire_format = 0;

uint32_t expert_wire_format() { return (uint32_t)g_wire_format; }

uint32_t expert_wire_format_for_quant(uint32_t q) {
  // This must track whatever the writer emits for the quant id. Pointing it at the wrong variant
  // count does not fault: the variant comes out of the wrong field bits, the LUT copy runs past its
  // buffer, and the output is merely uncorrelated. `aff-gpucheck --dq` is what catches it.
  return q == (uint32_t)AFF_Q2P875_Q4 ? (uint32_t)kFmtQuad2S : (uint32_t)kFmtPair16;
}

void expert_set_wire_format(uint32_t fmt) { g_wire_format = (int)fmt; }

// Every other reader of the two planes is row-major. Against a colgroup-major slab they do not
// fault and do not read out of bounds — both layouts are the same size and every address stays
// inside it — they just return wrong numbers, quietly, in a model whose output is plausible either
// way. So they abort instead. If one of these fires it is a live path that still needs converting,
// and the message says which.
static inline void aff_tr_guard(const char* who) {
  if (!expert_planes_are_colgroup_major()) return;
  aff::ui::fatal("aff: %s reads the expert planes ROW-MAJOR, but the slab is colgroup-major. That "
               "combination is silently wrong, not merely slow.\n", who);
  std::abort();
}

void expert_transpose_plane_host(const uint8_t* src_codes, const uint8_t* src_fields,
                                 uint8_t* dst_codes, uint8_t* dst_fields,
                                 uint32_t rows, uint32_t cols) {
  // A column group is kEK=64 columns = 4 blocks = exactly 20 code bytes and exactly 3 field bytes,
  // both whole. That is what makes this a pure permutation of spans with no bit surgery.
  constexpr uint32_t kCG = kEK / 16 * 5, kFG = kEK / 16 * 6 / 8;   // 20 and 3
  const uint32_t ng = cols / kEK;
  const uint32_t cbr = cols / 2 * 5 / 8;                 // code bytes a row
  const uint32_t fbr = cols / 16 * 6 / 8;                // field bytes a row
  // Both planes in one pass over the groups, so their strided reads stay hot together rather than
  // sweeping the same rows twice.
  for (uint32_t g = 0; g < ng; ++g)
    for (uint32_t r = 0; r < rows; ++r) {
      std::memcpy(dst_codes + ((size_t)g * rows + r) * kCG,
                  src_codes + (size_t)r * cbr + (size_t)g * kCG, kCG);
      std::memcpy(dst_fields + ((size_t)g * rows + r) * kFG,
                  src_fields + (size_t)r * fbr + (size_t)g * kFG, kFG);
    }
}

namespace {
// Plane byte counts. Layout-independent — the colgroup permutation moves bytes, never counts.
inline size_t plane_codes(uint64_t rows, uint64_t cols)  { return (size_t)rows * (cols / 2 * 5 / 8); }
inline size_t plane_fields(uint64_t rows, uint64_t cols) { return (size_t)rows * (cols / 16 * 6 / 8); }
inline uint32_t align256(size_t v) { return (uint32_t)((v + 255) & ~(size_t)255); }
} // namespace

bool expert_shard_layout(const AffLayerDesc& L, uint32_t parts, ExpertShardLayout* out) {
  if (!out || parts < 1) return false;
  const AffMatrixDesc* ms[3] = {&L.gate, &L.up, &L.down};
  ExpertShardLayout S{};
  uint32_t off = 0;
  for (int i = 0; i < 3; ++i) {
    const uint64_t rows = ms[i]->rows, cols = ms[i]->cols;
    // gate and up shard by rows; down (i == 2) shards by columns.
    const uint64_t rs = (i == 2) ? rows : rows / parts;
    const uint64_t cs = (i == 2) ? cols / parts : cols;
    if (i == 2) {
      if (cols % ((uint64_t)kEK * parts)) return false;    // whole column GROUPS only
      // The shard's own width only has to be whole column groups: the colgroup-major kernels stage
      // kEK columns an iteration and address the plane by group, so kEK is the unit. The old rule
      // here was 512, which V4's 2048-wide intermediate met by accident and V4.1's 2304 does not
      // (1152 at two ranks) -- it cost the whole TP=2 path on V4.1 for no kernel reason. The 2048
      // rule in the TILE path is a different kernel, which the engine declines while its planes are
      // colgroup-major. AFF_SLAB_VERIFY checks every shard against the container's row-major bytes
      // on load, which is what makes relaxing this testable rather than hopeful.
      if (cs % kEK) return false;
    } else {
      if (rows % parts) return false;
    }
    S.rows[i] = (uint32_t)rs;
    S.cols[i] = (uint32_t)cs;
    S.data_off[i] = off;   off = align256(off + plane_codes(rs, cs));
    S.scale_off[i] = off;  off = align256(off + plane_fields(rs, cs));
    S.rscale_off[i] = off; off = align256(off + (size_t)rs * 2);
  }
  S.stride = off;
  *out = S;
  return true;
}

void expert_shard_host(const uint8_t* whole, const AffLayerDesc& L, const ExpertShardLayout& S,
                       uint32_t parts, uint32_t part, uint8_t* dst) {
  constexpr uint32_t kCG = kEK / 16 * 5, kFG = kEK / 16 * 6 / 8;   // 20 and 3
  const AffMatrixDesc* ms[3] = {&L.gate, &L.up, &L.down};
  for (int i = 0; i < 3; ++i) {
    const uint32_t rows = (uint32_t)ms[i]->rows, cols = (uint32_t)ms[i]->cols;
    const uint32_t ng = cols / kEK;
    const uint32_t rs = S.rows[i];
    const uint8_t* src_c = whole + ms[i]->data_off;
    const uint8_t* src_f = whole + ms[i]->scale_off;
    const uint8_t* src_r = whole + ms[i]->rscale_off;
    if (i == 2) {
      // Column shard: a contiguous run of groups in BOTH planes, and every row scale.
      const uint32_t gs = ng / parts, g0 = part * gs;
      std::memcpy(dst + S.data_off[i], src_c + (size_t)g0 * rows * kCG, (size_t)gs * rows * kCG);
      std::memcpy(dst + S.scale_off[i], src_f + (size_t)g0 * rows * kFG, (size_t)gs * rows * kFG);
      std::memcpy(dst + S.rscale_off[i], src_r, (size_t)rows * 2);
    } else {
      // Row shard: one span per group, since rows are the INNER index of a colgroup plane.
      const uint32_t r0 = part * rs;
      for (uint32_t g = 0; g < ng; ++g) {
        std::memcpy(dst + S.data_off[i] + (size_t)g * rs * kCG,
                    src_c + ((size_t)g * rows + r0) * kCG, (size_t)rs * kCG);
        std::memcpy(dst + S.scale_off[i] + (size_t)g * rs * kFG,
                    src_f + ((size_t)g * rows + r0) * kFG, (size_t)rs * kFG);
      }
      std::memcpy(dst + S.rscale_off[i], src_r + (size_t)r0 * 2, (size_t)rs * 2);
    }
  }
}

size_t expert_shard_check_host(const uint8_t* raw, const AffLayerDesc& L,
                               const ExpertShardLayout& S, uint32_t parts, uint32_t part,
                               const uint8_t* shard) {
  constexpr uint32_t kCG = kEK / 16 * 5, kFG = kEK / 16 * 6 / 8;
  const AffMatrixDesc* ms[3] = {&L.gate, &L.up, &L.down};
  size_t bad = 0;
  for (int i = 0; i < 3; ++i) {
    const uint32_t cols = (uint32_t)ms[i]->cols;
    const uint32_t cbr = cols / 2 * 5 / 8, fbr = cols / 16 * 6 / 8;   // row-major bytes a row
    const uint32_t rs = S.rows[i], ngs = S.cols[i] / kEK;
    // Where this shard's (group, row) sits in the ORIGINAL logical matrix.
    const uint32_t row0 = (i == 2) ? 0 : part * rs;
    const uint32_t grp0 = (i == 2) ? part * ngs : 0;
    for (uint32_t g = 0; g < ngs; ++g)
      for (uint32_t r = 0; r < rs; ++r) {
        const uint8_t* ec = raw + ms[i]->data_off + (size_t)(row0 + r) * cbr + (size_t)(grp0 + g) * kCG;
        const uint8_t* ef = raw + ms[i]->scale_off + (size_t)(row0 + r) * fbr + (size_t)(grp0 + g) * kFG;
        const uint8_t* ac = shard + S.data_off[i] + ((size_t)g * rs + r) * kCG;
        const uint8_t* af = shard + S.scale_off[i] + ((size_t)g * rs + r) * kFG;
        if (std::memcmp(ec, ac, kCG)) ++bad;
        if (std::memcmp(ef, af, kFG)) ++bad;
      }
    if (std::memcmp(raw + ms[i]->rscale_off + (size_t)row0 * 2, shard + S.rscale_off[i],
                    (size_t)rs * 2))
      ++bad;
  }
  return bad;
}

void expert_transpose_planes_host(uint8_t* buf, const AffLayerDesc& L,
                                  std::vector<uint8_t>& scratch) {
  const AffMatrixDesc* ms[3] = {&L.gate, &L.up, &L.down};
  for (const AffMatrixDesc* d : ms) {
    const uint32_t rows = (uint32_t)d->rows, cols = (uint32_t)d->cols;
    const size_t cbytes = (size_t)rows * (cols / 2 * 5 / 8);
    const size_t fbytes = (size_t)rows * (cols / 16 * 6 / 8);
    scratch.resize(cbytes + fbytes);
    expert_transpose_plane_host(buf + d->data_off, buf + d->scale_off, scratch.data(),
                                scratch.data() + cbytes, rows, cols);
    std::memcpy(buf + d->data_off, scratch.data(), cbytes);
    std::memcpy(buf + d->scale_off, scratch.data() + cbytes, fbytes);
  }
}

void expert_gemm_wmma_hip(const ExpertTile* tiles, uint32_t n_tiles, uint32_t max_nt,
                          const uint8_t* x8, const float* xs, const float* cb, uint32_t rows,
                          uint32_t cols, float wscale, void* stream, bool y_bf16) {
  if (!n_tiles) return;
  const uint32_t ntiles = (max_nt + 15) / 16;
  // The wire format follows the CONTAINER, not taste: each one needs its own codebook tensor. See
  // expert_wire_format_for_quant.
  const uint32_t fmt = expert_wire_format();
  const bool tr = expert_planes_are_colgroup_major();
#define AFF_EW1(NTV, MTV, FV, TRV)                                                                \
  do {                                                                                             \
  hipLaunchKernelGGL((expert_gemm_wmma_kernel<NTV, MTV, FV, TRV>),                                 \
                     dim3((rows + kEWaves * MTV * 16 - 1) / (kEWaves * MTV * 16), n_tiles),        \
                     dim3(kEThreads), 0, (hipStream_t)stream, tiles, x8, xs, cb, wscale, rows,     \
                     cols, y_bf16 ? 1 : 0); AFF_HIP_CHECK(hipGetLastError());                      \
  } while (0)
// The plane layout and the wire format are INDEPENDENT, and both `tr` branches carry every format
// on purpose: dispatching `tr` first and falling through to kFmtPair16 would read the shipped
// layout with the wrong decode — right size, in-bounds addresses, wrong weights.
#define AFF_EWF(NTV, MTV, TRV)                                                                    \
  do {                                                                                             \
    if (fmt == kFmtQuad8) AFF_EW1(NTV, MTV, kFmtQuad8, TRV);                                       \
    else if (fmt == kFmtQuad2S) AFF_EW1(NTV, MTV, kFmtQuad2S, TRV);                                \
    else if (fmt == kFmtQuad4S) AFF_EW1(NTV, MTV, kFmtQuad4S, TRV);                                \
    else if (fmt == kFmtQuad4) AFF_EW1(NTV, MTV, kFmtQuad4, TRV);                                  \
    else AFF_EW1(NTV, MTV, kFmtPair16, TRV);                                                       \
  } while (0)
#define AFF_EW(NTV, MTV)                                                                          \
  do {                                                                                             \
    if (tr) AFF_EWF(NTV, MTV, true); else AFF_EWF(NTV, MTV, false);                                \
  } while (0)
  // MT differs between the arms, and that is the point. At prefill's tile width the grid is already
  // hundreds of blocks, so MT=2 is right: fewer blocks, fewer per-block LUT rebuilds. Decode's arm
  // runs one narrow tile over a row shard, where the grid is small enough that it cannot supply the
  // waves the registers have paid for — there MT=1 doubles the block count and closes the gap. The
  // two are bit-identical by construction, since MT decides which rows a wave owns and not the order
  // of the K reduction.
  if (ntiles <= 1) AFF_EW(1, 1);
  else if (ntiles <= 2) AFF_EW(2, 2);
  else if (ntiles <= 4) AFF_EW(4, 2);
  else AFF_EW(8, 1);
#undef AFF_EW
#undef AFF_EWF
#undef AFF_EW1
}

} // namespace aff

// amdnas-fixes launch-check: every hipLaunchKernelGGL in this file is followed by a launch-error check (2026-09-22).
