// Flash-style fp8 WMMA attention. See attn_flash.h for the cache format and why the row scale is
// per row and not per 64 dims.
//
// ---- why neither operand needs an LDS tile -----------------------------------------------------
//
// Both WMMA operands want the REDUCTION index contiguous (`wmma.h`, "A is free, B is not"). The two
// halves of attention reduce over different axes, so they want the KV tile in opposite layouts:
//
//   S^T = K . Q^T   reduces over dim   -> K wants dim contiguous  == the cache's own layout
//   O   = P . V     reduces over key   -> V wants key contiguous  == the transpose of it
//
// One tile cannot be both, and at 512 dims a 64-key tile is 32 KB, so holding both layouts in LDS is
// impossible. Every flash implementation pays for this somewhere: a pre-transposed V in memory, an
// LDS round trip, or a hardware transpose.
//
// RDNA4 has the hardware transpose — `global_load_tr_b64`, 8-bit, column-major, wave32 — and
// a measured probe pinned the lane mapping AMD does not document. The crossbar is
//
//     out[l][j] = in[ 16*(l>>4) + 4*((l>>3)&1) + (j&3) + 8*(j>>2) ][ l & 7 ]
//
// which is a fixed lane permutation and therefore invertible. Feeding each lane the address
//
//     addr(L) = (8*b4 + 4*b3 + 2*b1 + b0) * ldb + 8*b2        (b4..b0 = bits of L)
//
// makes the instruction deliver exactly the fragment `wmma.h` describes — verified byte for byte
// against `load_b_fp8` at strides 16, 64, 520 and 528. So there is NO K relabelling to do and no
// cross-lane fixup: A keeps this header's labelling and B comes back in it.
//
// The consequence is the shape of this kernel: LDS holds Q, P and the softmax state and nothing
// else, ~21 KB, independent of context length. K comes from global as a plain 128-bit load, V from
// global as a transpose load, both 8 bytes a lane.
//
// `global_load_tr_b64` REQUIRES EXEC TO BE ALL ONES (§11.6.2 — "otherwise the operation is
// undefined"). The PV loop is therefore unconditional: out-of-range keys are given a valid row
// pointer and a zero weight rather than being branched around.

#include "gpu/attn_flash.h"
#include "gpu/hip_common.h"
#include "gpu/device_math.h"
#include "gpu/wmma.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>

namespace aff {
namespace {

constexpr uint32_t kW = kFlashWidth;         // 512
constexpr uint32_t kRowB = kFlashRowBytes;   // 528

// FOUR waves. LDS caps occupancy, so eight waves a block would buy one more wave a SIMD — and that
// does not pay: the key tile has to grow with the wave count (QK gives one 16-key M-tile to each
// wave, so kKT = 16 * kWaves), which halves the output dims a wave owns, and the dims a wave owns
// are exactly what lets ONE transposed V fragment feed many accumulators. Occupancy is not the
// binding constraint here; operand reuse is.
constexpr uint32_t kWaves = 4;
constexpr uint32_t kThreads = kWaves * 32;   // 128
constexpr uint32_t kKT = 16u * kWaves;       // 64 keys a tile
constexpr uint32_t kDW = kW / kWaves;        // 128 output dims a wave
constexpr uint32_t kOT = kDW / 16u;          // 8 accumulator tiles a wave a head tile

// Prologue split: 16 heads a pass, so the threads a head and the dims each of them owns both follow
// from the block size rather than being written out.
constexpr uint32_t kQTh = kThreads / 16u;    // threads a head
constexpr uint32_t kQD = kW / kQTh;          // dims a thread
constexpr uint32_t kQV = kQD / 4u;           // float4s a thread

// LDS row strides. A bare 512 puts every lane on bank 0 — a 16-way conflict.
//   Q: 528 is 132 dwords, so lane c starts at bank 4c and its four dwords cover four consecutive
//      banks; the wave's 512 bytes hit every bank exactly four times, which is the 128 B/cycle
//      bandwidth floor for a 128-bit read. It is also 16-byte aligned, which the paired load needs.
//   P: kKT alone would be 16 dwords -> banks 0/16 only; +8 fixes it, see the constant.
constexpr uint32_t kQLd = kW + 16;           // 528
constexpr uint32_t kPLd = kKT + 8;           // 72 = 18 dwords, and 18c mod 32 is a bijection

constexpr float kNeg = -3.0e38f;

typedef wmma::i2v __attribute__((address_space(1))) * i2v_gp;

// A pointer that has been through LDS is generic, and generic loads compile to flat_, which has no
// SGPR-base form and waits on the LDS counter as well as the memory one. s_ptr holds key rows —
// device memory, always — so say so. Empty on the host pass, where address spaces do not exist and
// uint4 is a class the qualifier cannot bind to.
// uint4 is a class, and dereferencing a qualified pointer to one casts straight back to generic to
// copy-construct — the load stays flat_. A native vector is a plain load and keeps the space.
#ifdef __HIP_DEVICE_COMPILE__
#define AFF_GLOBAL __attribute__((address_space(1)))
#else
#define AFF_GLOBAL
#endif
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

// The inverted transpose-load addressing, from the crossbar mapping above.
__device__ __forceinline__ uint32_t tr_row(uint32_t l) {
  return ((l >> 4) << 3) | (((l >> 3) & 1u) << 2) | (l & 3u);
}
__device__ __forceinline__ uint32_t tr_col(uint32_t l) { return ((l >> 2) & 1u) * 8u; }

__device__ __forceinline__ wmma::i2v load_b_tr(const uint8_t* p) {
#ifdef __HIP_DEVICE_COMPILE__
  return __builtin_amdgcn_global_load_tr_b64_v2i32((i2v_gp)p);
#else
  wmma::i2v z;
  z[0] = z[1] = 0;
  return z;
#endif
}

// v_cvt_pk_fp8_f32 emits NaN, not a saturated 448, for out-of-range input — and `max(x, NaN)`
// returns x, so an unclamped conversion reports zero error while poisoning the result. Clamp.
__device__ __forceinline__ float clamp448(float x) { return fminf(448.0f, fmaxf(-448.0f, x)); }

__device__ __forceinline__ uint32_t pack4(float a, float b, float c, float d) {
  uint32_t p = 0;
  p = __builtin_amdgcn_cvt_pk_fp8_f32(clamp448(a), clamp448(b), p, false);
  p = __builtin_amdgcn_cvt_pk_fp8_f32(clamp448(c), clamp448(d), p, true);
  return p;
}

// One block: NT * 16 query heads of ONE token, a contiguous RUN of keys.
// grid = (tokens, n_head / (NT*16), key slices).
//
//   wave w owns key sub-tile [16w, 16w+16) for the scores, and output dims [kDW*w, kDW*w+kDW) for
//   the weighted sum. The first split makes the per-head softmax a wave reduction (lane c and c+16
//   both hold head c); the second keeps the O accumulator to NT*kOT tiles a lane.
//
// gridDim.z slices the KEYS. At gridDim.z == 1 the kernel is exactly what it was — same loop
// bounds, same write, bit-identical — and that is the prefill path, which already has 1536 blocks.
// Above 1 each block walks tiles [lo, hi) of its own, seeds the softmax cold (the sink belongs to
// slice 0 alone, since it must be counted once), and writes its UNNORMALISED numerator O*S_last
// plus its (m, l) to the arena for `flash_combine_kernel` to merge. The slices are disjoint, so the
// bytes read are the same as one block would have read.
//
// NT is the arithmetic-intensity knob and the only one that matters at this point. A block reads the
// key tile twice — once as K, once as V — no matter how many heads it serves, so intensity scales
// with NT. At NT=1 both throughput and byte rate are flat across every context, which is a bandwidth
// ceiling and not a latency one: the fix is to spend more matrix ops per byte fetched, not to fetch
// faster. NT=2 also lets one K fragment feed two head tiles and one V fragment feed two
// accumulators, so the operand-load count per mma falls from 1.0 to 0.75 (QK) and 1.125 to 0.625
// (PV) as a side effect.
template <uint32_t NT, bool HAS_SINK, bool HAS_MASK>
__global__ __launch_bounds__(kThreads) void flash_kernel(
    const void* __restrict__ qp, const uint8_t* __restrict__ raw, const uint8_t* __restrict__ comp,
    const uint8_t* __restrict__ allowed, const uint32_t* __restrict__ n_comp_v, uint32_t n_comp1,
    const float* __restrict__ sinks, float* __restrict__ out, uint32_t b0, uint32_t pos0,
    uint32_t sliding, uint32_t ring, uint32_t max_comp, uint32_t n_head, float scale,
    int q_bf16, float* __restrict__ part_o, float* __restrict__ part_ml, uint32_t part_tok,
    const uint32_t* __restrict__ adm, uint32_t adm_stride) {

  constexpr uint32_t NH = NT * 16u;                   // query heads a block

  __shared__ __align__(16) uint8_t s_q[NH * kQLd];   // Q, E4M3, head-major
  __shared__ __align__(16) uint8_t s_p[NH * kPLd];   // P, E4M3, head-major, key contiguous
  __shared__ const uint8_t* s_ptr[kKT];              // this tile's key rows, ring/comp resolved
  __shared__ float s_rs[kKT];                        // their row scales; 0 marks absent or masked
  __shared__ float s_qs[NH];                         // Q dequant scale a head
  __shared__ float s_m[NH], s_l[NH], s_af[NH];       // running max, denominator, rescale factor
  __shared__ float s_red[kWaves * NH];               // cross-wave max / sum scratch
  __shared__ float s_amax[kThreads];
  __shared__ float s_scur, s_sprev;
  __shared__ int s_skip;

  const uint32_t tid = threadIdx.x, lane = tid & 31u, wave = tid >> 5;
  const uint32_t c = lane & 15u, hlf = lane >> 4;    // fragment column, lane half
  const uint32_t h0 = blockIdx.y * NH;
  const uint32_t tk = blockIdx.x;
  const uint32_t pos = pos0 + tk;
  const uint32_t n_raw = (pos + 1u < sliding) ? pos + 1u : sliding;
  const uint32_t raw0 = (pos + 1u - n_raw) % ring;
  const uint32_t n_comp = n_comp_v ? n_comp_v[tk] : n_comp1;
  const uint32_t n_keys = n_raw + n_comp;
  const size_t qoff = (size_t)(b0 + tk) * n_head * kW + (size_t)h0 * kW;

  // ---- this block's key run. Slice boundaries are multiples of the tile so every tile but the
  // very last is full and the resolve loop below needs no extra case. Tokens in the same launch
  // legitimately hold different key counts (n_comp_v), so the cut is computed per block from ITS
  // own n_keys rather than passed in — a token short enough to fit in fewer slices simply leaves
  // the tail ones empty, and an empty slice reports m = -inf, l = 0 and merges as nothing.
  const uint32_t nsp = gridDim.z, sp = blockIdx.z;
  uint32_t kt_lo = 0u, kt_hi = n_keys;
  if (nsp > 1u) {
    const uint32_t tiles = (n_keys + kKT - 1u) / kKT;
    const uint32_t tps = (tiles + nsp - 1u) / nsp;
    const uint32_t lo = sp * tps * kKT, hi = lo + tps * kKT;
    kt_lo = lo < n_keys ? lo : n_keys;
    kt_hi = hi < n_keys ? hi : n_keys;
  }
  // The sink SEEDS the softmax, so exactly one slice may hold it or the denominator gains a term
  // per block.
  const bool sink0 = HAS_SINK && sp == 0u;

  // ---- Q -> E4M3, one scale a head. 8 threads a head, 64 dims each, 16 heads a pass.
  //
  // The 64 floats are HELD, not re-read: reading twice costs a second full pass over Q, and the
  // prologue and epilogue together are a large share of a short-context call. The 16 float4s are
  // live only here, before the O accumulator exists, so they cost nothing at the kernel's register
  // peak. Passing over 16 heads at a time rather than NH at once is what keeps that true for NT > 1.
#pragma unroll
  for (uint32_t sb = 0; sb < NT; ++sb) {
    const uint32_t hh = sb * 16u + tid / kQTh, seg = (tid % kQTh) * kQD;
    const size_t qo = qoff + (size_t)hh * kW + seg;
    float4 v[kQV];
    float a = 0.0f;
    // q_bf16 is uniform across the launch. Q is 8.4 MB a launch at prefill's shape and this
    // prologue is a full pass over it, so halving the element halves the pass; nothing downstream
    // notices, because the very next thing that happens to these values is E4M3.
    if (q_bf16) {
      const uint16_t* src = (const uint16_t*)qp + qo;
#pragma unroll
      for (uint32_t i = 0; i < kQV; ++i) {
        const ushort4 h = *(const ushort4*)(src + i * 4);
        v[i] = make_float4(wmma::bf16_to_f32(h.x), wmma::bf16_to_f32(h.y),
                           wmma::bf16_to_f32(h.z), wmma::bf16_to_f32(h.w));
        a = fmaxf(fmaxf(fabsf(v[i].x), fabsf(v[i].y)), fmaxf(fmaxf(fabsf(v[i].z), fabsf(v[i].w)), a));
      }
    } else {
      const float* src = (const float*)qp + qo;
#pragma unroll
      for (uint32_t i = 0; i < kQV; ++i) {
        v[i] = *(const float4*)(src + i * 4);
        a = fmaxf(fmaxf(fabsf(v[i].x), fabsf(v[i].y)), fmaxf(fmaxf(fabsf(v[i].z), fabsf(v[i].w)), a));
      }
    }
    s_amax[tid] = a;
    __syncthreads();
    if (tid < 16u) {
      float m = 0.0f;
      for (uint32_t i = 0; i < kQTh; ++i) m = fmaxf(m, s_amax[tid * kQTh + i]);
      const uint32_t hd = sb * 16u + tid;
      s_qs[hd] = (m > 0.0f) ? m * (1.0f / 448.0f) : 1.0f;
      s_m[hd] = sink0 ? sinks[h0 + hd] : kNeg;
      // The sink contributes no value vector: the denominator starts at exp(sink - sink) = 1 and
      // the numerator at zero.
      s_l[hd] = sink0 ? 1.0f : 0.0f;
    }
    __syncthreads();
    const float inv = 1.0f / s_qs[hh];
    uint8_t* dst = s_q + hh * kQLd + seg;
#pragma unroll
    for (uint32_t i = 0; i < kQV; ++i)
      *(uint32_t*)(dst + i * 4) = pack4(v[i].x * inv, v[i].y * inv, v[i].z * inv, v[i].w * inv);
  }
  if (tid == 0) s_sprev = 1.0f;
  __syncthreads();

  wmma::f8v o[NT][kOT];
#pragma unroll
  for (uint32_t nt = 0; nt < NT; ++nt)
#pragma unroll
    for (uint32_t n = 0; n < kOT; ++n) o[nt][n] = wmma::acc_zero();

  const uint32_t kr = tr_row(lane), co = tr_col(lane);

  for (uint32_t kt0 = kt_lo; kt0 < kt_hi; kt0 += kKT) {
    // ---- resolve the tile's rows on wave 0 alone, two keys a lane, so the tile scale reduction
    // needs no second barrier.
    if (wave == 0) {
      float rs[kKT / 32u];
#pragma unroll
      for (uint32_t u = 0; u < kKT / 32u; ++u) {
        const uint32_t i = lane + u * 32u;
        const uint32_t k = kt0 + i;
        const uint8_t* p = raw;                      // valid fallback: absent keys get weight 0
        float r = 0.0f;
        if (k < kt_hi) {
          if (k < n_raw) {
            uint32_t sl = raw0 + k;
            if (sl >= ring) sl -= ring;
            p = raw + (size_t)sl * kRowB;
          } else {
            // GATHERED, when the indexer left a list: `n_comp` is then the number of rows it
            // ADMITTED and this walks those rows only. Without one it is the whole compressed
            // cache and the row index is the key's own offset, as it always was.
            const uint32_t cr = adm ? adm[(size_t)tk * adm_stride + (k - n_raw)] : (k - n_raw);
            p = comp + (size_t)cr * kRowB;
          }
          r = *(const AFF_GLOBAL float*)(p + kW);
          // `allowed[k - n_raw]` — WITHOUT the token stride — is a quiet bug. The mask is a PLANE,
          // one row of `max_comp` bytes a token, and the VALU kernel this replaced offsets by it
          // (`attention_gpu.hip`, mqa_attend_batch_kernel). Without the stride every token of a
          // batched launch reads the FIRST token's admissions — fluent text from the wrong keys, at
          // long-context prefill only, since decode is one token and never sees it.
          if (HAS_MASK && k >= n_raw && !allowed[(size_t)tk * max_comp + (k - n_raw)]) r = 0.0f;
        }
        s_ptr[i] = p;
        s_rs[i] = r;
        rs[u] = r;
      }
      float mx = rs[0];
#pragma unroll
      for (uint32_t u = 1; u < kKT / 32u; ++u) mx = fmaxf(mx, rs[u]);
      for (int off = 16; off; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off));
      if (lane == 0) s_scur = (mx > 0.0f) ? mx : s_sprev;
    }
    __syncthreads();

    // ---- S^T = K . Q^T over all 512 dims. A is K (dim contiguous, plain load), B is Q from LDS.
    // D[m][n] is S^T[key m][head n], so a lane owns ONE head and eight keys — which is what makes
    // the softmax below a wave reduction with no block barrier.
    //
    // TWO mmas per 128-bit load a side, not one per 64-bit load. A K-major fp8 fragment is only
    // 8 bytes, so the natural load leaves half the width on the table and QK spends two operand
    // fetches per matrix op. Reading 16 bytes at `hlf * 16` and splitting it relabels K as
    //     mma 0: dims d + 16h + 0..7      mma 1: dims d + 16h + 8..15
    // which is a permutation of the 32-dim super-group applied IDENTICALLY to K and Q, and the
    // product is invariant under exactly that (`wmma.h`, "the K labelling"). Both operands are read
    // that way here, which is the condition; mixing it with an unrelabelled load is silently wrong,
    // so neither pointer below is used anywhere else. The one K fragment feeds all NT head tiles.
    const uint8_t* kp = s_ptr[16u * wave + c] + hlf * 16u;
    wmma::f8v acc[NT];
#pragma unroll
    for (uint32_t nt = 0; nt < NT; ++nt) acc[nt] = wmma::acc_zero();
#pragma unroll 4
    for (uint32_t d = 0; d < kW; d += 32) {
      const u32x4 kk = *(const AFF_GLOBAL u32x4*)(kp + d);
      wmma::i2v a0, a1;
      a0[0] = (int)kk.x; a0[1] = (int)kk.y; a1[0] = (int)kk.z; a1[1] = (int)kk.w;
#pragma unroll
      for (uint32_t nt = 0; nt < NT; ++nt) {
        const uint4 qq = *(const uint4*)(s_q + (size_t)(nt * 16u + c) * kQLd + hlf * 16u + d);
        wmma::i2v b0, b1;
        b0[0] = (int)qq.x; b0[1] = (int)qq.y; b1[0] = (int)qq.z; b1[1] = (int)qq.w;
        acc[nt] = wmma::mma_fp8(a0, b0, acc[nt]);
        acc[nt] = wmma::mma_fp8(a1, b1, acc[nt]);
      }
    }

    // ---- online softmax. The key's row scale and the head's Q scale are both single multiplies
    // here: one indexes the accumulator ROW (8h+j, a key), the other the COLUMN (c, a head).
    // The eight row scales a lane needs are the same for every head tile and are wanted twice more
    // below, so they are read once. Left inline they were 16*NT LDS reads a tile for 8 values.
    float rw[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) rw[j] = s_rs[16u * wave + hlf * 8u + (uint32_t)j];

    float sv[NT][8], mnew[NT], alpha[NT], ps[NT];
#pragma unroll
    for (uint32_t nt = 0; nt < NT; ++nt) {
      const float qs = s_qs[nt * 16u + c] * scale;
      float mt = kNeg;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        sv[nt][j] = (rw[j] > 0.0f) ? acc[nt][j] * rw[j] * qs : kNeg;
        mt = fmaxf(mt, sv[nt][j]);
      }
      mt = fmaxf(mt, __shfl_xor(mt, 16));
      if (lane < 16) s_red[wave * NH + nt * 16u + c] = mt;
    }
    __syncthreads();
#pragma unroll
    for (uint32_t nt = 0; nt < NT; ++nt) {
      const uint32_t hd = nt * 16u + c;
      float mtile = s_red[hd];
#pragma unroll
      for (uint32_t w = 1; w < kWaves; ++w) mtile = fmaxf(mtile, s_red[w * NH + hd]);
      const float mold = s_m[hd];
      mnew[nt] = fmaxf(mold, mtile);
      alpha[nt] = (mold <= kNeg) ? 0.0f : __expf(mold - mnew[nt]);
      float psum = 0.0f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float e = (sv[nt][j] > kNeg) ? __expf(sv[nt][j] - mnew[nt]) : 0.0f;
        sv[nt][j] = e;                                // sv becomes P in place
        psum += e;
      }
      ps[nt] = psum + __shfl_xor(psum, 16);
    }
    __syncthreads();                                  // s_red is reused for the denominator
#pragma unroll
    for (uint32_t nt = 0; nt < NT; ++nt)
      if (lane < 16) s_red[wave * NH + nt * 16u + c] = ps[nt];
    __syncthreads();

    // ---- P -> E4M3, with the key's row scale folded in.
    //
    // O is kept in units of the tile's own scale S_cur. A row scale belongs to V, and V is the
    // reduction operand, so it can only ride on P; but P is a softmax weight in [0,1] and
    // multiplying it by a raw 2^-7 would push most of it into E4M3 subnormals. Dividing by S_cur
    // holds the ratio at or below 1, and the unit change is absorbed by the rescale the online
    // softmax already applies each tile: alpha * (S_prev / S_cur).
    const float sratio = s_sprev / s_scur;
    const float inv_scur = 1.0f / s_scur;
    int skip = 1;
#pragma unroll
    for (uint32_t nt = 0; nt < NT; ++nt) {
      const uint32_t hd = nt * 16u + c;
      const float fac = alpha[nt] * sratio;
      // Rescaling O is 64*NT multiplies a lane a tile — the largest non-matrix cost in the loop —
      // and it is the identity on most tiles, because the running max stops moving after the first
      // few and neighbouring rows share a scale. One ballot prices the check at nothing.
      if (wave == 0) {
        if (__ballot(fac == 1.0f) != 0xFFFFFFFFull) skip = 0;
        if (lane < 16) {
          float ltile = s_red[hd];
#pragma unroll
          for (uint32_t w = 1; w < kWaves; ++w) ltile += s_red[w * NH + hd];
          s_m[hd] = mnew[nt];
          s_l[hd] = s_l[hd] * alpha[nt] + ltile;
          s_af[hd] = fac;
        }
      }
      float pw[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) pw[j] = sv[nt][j] * (rw[j] * inv_scur);
      uint8_t* pdst = s_p + (size_t)hd * kPLd + 16u * wave + hlf * 8u;
      *(uint32_t*)(pdst) = pack4(pw[0], pw[1], pw[2], pw[3]);
      *(uint32_t*)(pdst + 4) = pack4(pw[4], pw[5], pw[6], pw[7]);
    }
    if (wave == 0 && lane == 0) s_skip = skip;
    __syncthreads();

    // ---- rescale O, then O += P . V. The accumulator row is the HEAD here, not the key, so the
    // factor is indexed by 8h+j and the eight values are contiguous in s_af.
    if (!s_skip) {
#pragma unroll
      for (uint32_t nt = 0; nt < NT; ++nt) {
        float f[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) f[j] = s_af[nt * 16u + hlf * 8u + (uint32_t)j];
#pragma unroll
        for (uint32_t n = 0; n < kOT; ++n)
#pragma unroll
          for (int j = 0; j < 8; ++j) o[nt][n][j] *= f[j];
      }
    }

#pragma unroll
    for (uint32_t t = 0; t < kKT / 16u; ++t) {
      wmma::i2v a[NT];
#pragma unroll
      for (uint32_t nt = 0; nt < NT; ++nt)
        a[nt] = *(const wmma::i2v*)(s_p + (size_t)(nt * 16u + c) * kPLd + 16u * t + hlf * 8u);
      const uint8_t* vp = s_ptr[16u * t + kr] + co + wave * kDW;
#pragma unroll
      for (uint32_t n = 0; n < kOT; ++n) {
        const wmma::i2v b = load_b_tr(vp + n * 16u);   // one V fragment feeds every head tile
#pragma unroll
        for (uint32_t nt = 0; nt < NT; ++nt) o[nt][n] = wmma::mma_fp8(a[nt], b, o[nt][n]);
      }
    }
    if (tid == 0) s_sprev = s_scur;
    __syncthreads();
  }

  // ---- out = O * S_last / denominator. O's row is the head, its column the dim.
  //
  // O carries the units of the LAST tile's row scale (see the P encode above), so O * S_last is the
  // true numerator. A slice writes exactly that, undivided, and lets the combine pass do the
  // division once it knows every slice's (m, l); an empty slice writes zeros over a max of -inf,
  // which merges as nothing.
  const float sfin = s_sprev;
  if (nsp > 1u) {
    float* po = part_o + (((size_t)sp * part_tok + tk) * n_head + h0) * kW;
#pragma unroll
    for (uint32_t nt = 0; nt < NT; ++nt)
#pragma unroll
      for (uint32_t n = 0; n < kOT; ++n) {
        const uint32_t dim = wave * kDW + n * 16u + c;
#pragma unroll
        for (int j = 0; j < 8; ++j)
          po[(size_t)(nt * 16u + hlf * 8u + (uint32_t)j) * kW + dim] = o[nt][n][j] * sfin;
      }
    if (wave == 0 && lane < 16u) {
      float* pm = part_ml + (((size_t)sp * part_tok + tk) * n_head + h0) * 2;
#pragma unroll
      for (uint32_t nt = 0; nt < NT; ++nt) {
        const uint32_t hd = nt * 16u + lane;
        pm[(size_t)hd * 2 + 0] = s_m[hd];
        pm[(size_t)hd * 2 + 1] = s_l[hd];
      }
    }
    return;
  }
#pragma unroll
  for (uint32_t nt = 0; nt < NT; ++nt)
#pragma unroll
    for (uint32_t n = 0; n < kOT; ++n) {
      const uint32_t dim = wave * kDW + n * 16u + c;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const uint32_t hd = nt * 16u + hlf * 8u + (uint32_t)j;
        const float den = s_l[hd];
        out[qoff + (size_t)hd * kW + dim] = o[nt][n][j] * sfin * ((den > 0.0f) ? 1.0f / den : 0.0f);
      }
    }
}

// ---- merge the slices. One block a (token, head): 128 threads, four dims each, one float4 a slice.
//
// This is the same online-softmax update the kernel applies tile to tile, done once across slices
// instead: take the maximum, rescale every numerator and denominator into it, divide at the end.
// The arithmetic is associative in exactly the way the tile loop already relies on, so the ONLY
// difference from the single-block answer is summation ORDER — which is not nothing in fp32, and
// is why this path is checked by hash-and-compare rather than by bit-identity.
constexpr uint32_t kCT = 128;                        // threads a combine block; kW / 4 exactly
constexpr uint32_t kMaxSlices = 64;

__global__ __launch_bounds__(kCT) void flash_combine_kernel(
    const float* __restrict__ part_o, const float* __restrict__ part_ml, float* __restrict__ out,
    uint32_t nsp, uint32_t part_tok, uint32_t n_head, uint32_t b0) {
  __shared__ float s_w[kMaxSlices];                  // exp(m_s - M), the slice's rescale
  __shared__ float s_inv;                            // 1 / sum_s l_s * w_s

  const uint32_t tk = blockIdx.x, hd = blockIdx.y;
  const size_t row = (size_t)tk * n_head + hd;       // within a slice
  const size_t plane = (size_t)part_tok * n_head;    // slice to slice

  // The (m, l) pairs are 2 * nsp floats a block, so one wave reads them and shares the result
  // rather than 128 threads each walking the same 512 bytes.
  if (threadIdx.x < 32u) {
    float m = kNeg;
    for (uint32_t s = threadIdx.x; s < nsp; s += 32u)
      m = fmaxf(m, part_ml[(row + (size_t)s * plane) * 2]);
    for (int off = 16; off; off >>= 1) m = fmaxf(m, __shfl_xor(m, off));
    float l = 0.0f;
    for (uint32_t s = threadIdx.x; s < nsp; s += 32u) {
      const float ms = part_ml[(row + (size_t)s * plane) * 2];
      const float w = (ms > kNeg) ? __expf(ms - m) : 0.0f;
      s_w[s] = w;
      l += part_ml[(row + (size_t)s * plane) * 2 + 1] * w;
    }
    for (int off = 16; off; off >>= 1) l += __shfl_xor(l, off);
    if (threadIdx.x == 0) s_inv = (l > 0.0f) ? 1.0f / l : 0.0f;
  }
  __syncthreads();

  const uint32_t d = threadIdx.x * 4u;
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (uint32_t s = 0; s < nsp; ++s) {
    const float w = s_w[s];
    if (w == 0.0f) continue;                         // an empty slice, and at depth most are not
    const float4 v = *(const float4*)(part_o + (row + (size_t)s * plane) * kW + d);
    acc.x += v.x * w; acc.y += v.y * w; acc.z += v.z * w; acc.w += v.w * w;
  }
  const float inv = s_inv;
  *(float4*)(out + ((size_t)(b0 + tk) * n_head + hd) * kW + d) =
      make_float4(acc.x * inv, acc.y * inv, acc.z * inv, acc.w * inv);
}

}  // namespace

// Head tiles a block, 16 heads each — the kernel's arithmetic-intensity knob.
//
// 4 IS NOT OFFERED, and the reason is arithmetic rather than tuning. The O accumulator is
// `o[NT][kOT]` of 8 floats, i.e. NT * kOT * 8 = NT * 64 VGPRs — 64 at NT=1, 128 at NT=2, and 256 at
// NT=4, which is the whole register file before a single other value exists. Compiled, NT=4 reports
// 1234 spilled VGPRs, 1268 bytes/lane of scratch and occupancy 3 against NT=2's 240 VGPRs, no
// scratch and occupancy 6. Any sweep that included it was pricing the spill, not the tile.
static uint32_t flash_head_tiles() { return 2u; }

// ---- host-side row encode ----------------------------------------------------------------------
//
// The scale is a power of two chosen exactly as fp8_kv_qat chooses it, so for a row the model has
// already quantised this reproduces its bytes rather than rounding them again.
void kv_encode_fp8r_row(const float* src, uint8_t* dst) {
  float amax = 0.0f;
  for (uint32_t i = 0; i < kFlashWidth; ++i) amax = std::fmax(amax, std::fabs(src[i]));
  if (amax < 1e-6f) amax = 1e-6f;
  const float s = std::ldexp(1.0f, (int)std::ceil(std::log2(amax / 448.0f)));
  const float inv = 1.0f / s;
  for (uint32_t i = 0; i < kFlashWidth; ++i) {
    float a = src[i] * inv;
    a = std::fmin(448.0f, std::fmax(-448.0f, a));
    // round-to-nearest-even E4M3, matching the device converter
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
      if (rem > half || (rem == half && (mm & 1u))) {
        if (++mm == 8u) { mm = 0; ++e; }
      }
      if (e > 8) { e = 8; mm = 7; }
      b = (uint8_t)(sign | ((uint32_t)(e + 7) << 3) | mm);
    } else {
      uint32_t mm = (uint32_t)std::lrintf(m * 512.0f);
      if (mm > 7u) mm = 7u;
      b = (uint8_t)(sign | mm);
    }
    dst[i] = b;
  }
  *(float*)(dst + kFlashWidth) = s;
}

uint64_t flash_split_bytes(uint32_t tok, uint32_t heads, uint32_t slots, uint64_t* ml_bytes) {
  const uint64_t o = (uint64_t)slots * tok * heads * kW * sizeof(float);
  const uint64_t m = (uint64_t)slots * tok * heads * 2 * sizeof(float);
  if (ml_bytes) *ml_bytes = m;
  return o;
}

// How many ways to cut the keys: two ceilings and a floor. The arena is preallocated, so `slots` is
// hard, and a slice wants enough tiles to amortise its own prologue (Q is re-quantised per block)
// and its share of the combine pass, so below `kMinTiles` the cut is not taken.
//
// kTargetBlocks is a TUNING DEFAULT, not an occupancy figure: it is the point past which a finer
// cut stops gaining, on a 64-CU part, and it is also the cheapest arena. A card with a very
// different CU count would want it re-tuned — but it is only a ceiling on a grid that is otherwise
// one workgroup, so a wrong value costs a coarse split and nothing worse.
static uint32_t flash_split_slices(uint32_t n_tok, uint32_t head_tiles, uint32_t n_keys,
                                   const FlashSplit* sp) {
  if (!sp || !sp->o || !sp->ml || !sp->slots || n_tok > sp->tok_cap) return 1u;
  constexpr uint32_t kTargetBlocks = 128u;
  constexpr uint32_t kMinTiles = 4u;                 // keys a slice, in units of kKT = 64
  const uint32_t nat = n_tok * head_tiles;
  if (!nat || nat >= kTargetBlocks) return 1u;
  const uint32_t tiles = (n_keys + kKT - 1u) / kKT;
  uint32_t s = std::min(kTargetBlocks / nat, sp->slots);
  s = std::min(s, kMaxSlices);
  s = std::min(s, tiles / kMinTiles);
  return s ? s : 1u;
}

void mqa_attend_flash_hip(const void* q, const void* raw, const void* comp,
                          const uint8_t* allowed, const uint32_t* n_comp_v, uint32_t n_comp1,
                          const float* sinks,
                          float* out, uint32_t b0, uint32_t n, uint32_t /*nb*/, uint32_t pos0,
                          uint32_t sliding, uint32_t ring, uint32_t max_comp, uint32_t n_head,
                          float scale, void* stream, bool q_bf16, const FlashSplit* split,
                          uint32_t n_comp_hint, const uint32_t* adm, uint32_t adm_stride) {
  if (!n || !n_head) return;
  hipStream_t st = (hipStream_t)stream;
  const uint8_t* r8 = (const uint8_t*)raw;
  const uint8_t* c8 = (const uint8_t*)comp;
  // Clamped below to what the head count can actually supply: a tile count that does not divide the
  // heads asks for a zero-height grid, which HIP reports as "invalid configuration argument" at
  // launch and not as anything that points here.
  uint32_t nt = flash_head_tiles();
  // DECODE HAS ONE QUERY TOKEN, so the grid is (1, n_head / (nt*16)) — which with a card's share of
  // the heads is exactly ONE workgroup: the whole attention sublayer on one CU, reading at a few
  // percent of the card's bandwidth. gridDim.z is the fix; see flash_split_slices for what sets it.
  //
  // Splitting the HEAD tile instead — `if (n == 1) nt = 1`, two blocks rather than one — is worse at
  // every context, and bit-identically so, which makes it a pure timing loss rather than a different
  // answer: both blocks then walk the SAME key rows, so the bytes read double while the CU count
  // only doubles, and the deeper the cache the more that costs. The keys are the axis to split.
  while (nt > 1u && n_head % (nt * 16u) != 0u) nt >>= 1;
  const uint32_t ntiles = n_head / (nt * 16u);
  // The key count the split is sized against: the sliding window of the FIRST token in the launch
  // plus the widest compressed run in it. Both are lower bounds on later tokens, and the kernel
  // recomputes its own cut from its own count anyway, so an underestimate costs a slightly coarse
  // grid and nothing else.
  const uint32_t nraw0 = (pos0 + 1u < sliding) ? pos0 + 1u : sliding;
  const uint32_t nkeys = nraw0 + (n_comp_v ? n_comp_hint : n_comp1);
  const uint32_t nsp = flash_split_slices(n, ntiles, nkeys, split);
  float* po = nsp > 1u ? split->o : nullptr;
  float* pm = nsp > 1u ? split->ml : nullptr;
  const uint32_t ptok = nsp > 1u ? split->tok_cap : 0u;
  const dim3 g(n, ntiles, nsp), b(kThreads);
#define AFF_LAUNCH(T, S, M)                                                                      \
  hipLaunchKernelGGL((flash_kernel<T, S, M>), g, b, 0, st, q, r8, c8, allowed, n_comp_v, n_comp1,  \
                     sinks, out, b0, pos0, sliding, ring, max_comp, n_head, scale, q_bf16 ? 1 : 0, \
                     po, pm, ptok, adm, adm_stride)
#define AFF_PICK(T)                                                    \
  do {                                                                 \
    if (sinks && allowed)  AFF_LAUNCH(T, true, true);                  \
    else if (sinks)        AFF_LAUNCH(T, true, false);                 \
    else if (allowed)      AFF_LAUNCH(T, false, true);                 \
    else                   AFF_LAUNCH(T, false, false);                \
  } while (0)
  if (nt == 2u) AFF_PICK(2u);
  else          AFF_PICK(1u);
#undef AFF_PICK
#undef AFF_LAUNCH
  if (nsp > 1u)
    hipLaunchKernelGGL(flash_combine_kernel, dim3(n, n_head), dim3(kCT), 0, st, po, pm, out, nsp,
                       ptok, n_head, b0);
  AFF_HIP_CHECK(hipGetLastError());
}

}  // namespace aff
