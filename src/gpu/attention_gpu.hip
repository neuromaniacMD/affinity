// Shared-KV MQA attention on device, with the KV cache resident in VRAM.
//
// On the host this was the single largest remaining CPU cost. It is COMPUTE
// bound, not bandwidth bound, which is why it is worth moving: the work is n_heads x n_keys x width
// twice per layer — scores, then the weighted sum — and the CPU sustains a small fraction of what
// one card does on it.
//
// THE SHAPE IS FIXED, WHICH IS WHY ONE BLOCK PER HEAD IS ENOUGH.
// n_keys is bounded no matter how long the context gets: the raw sliding window is 128 and the
// lightning indexer admits at most index_topk (512) compressed rows, so a layer never attends over
// more than ~640 keys. aiter's unified attention splits the KV sequence across programs and then
// reduces the partials (its 3D path) precisely because a decode step with one KV head otherwise
// leaves the device idle — worth doing when n_keys is unbounded. Here 64 heads already give 64
// blocks and each block has real work, so the split-K machinery would add a reduction pass and a
// second kernel launch to solve a problem this model's sparsity already solved. If index_topk ever
// grows, that is the change to make.
//
// SEMANTICS THAT ARE EASY TO GET WRONG, all of them load-bearing:
//  * The value is the FULL 512-wide latent, rope tail included — not the leading 448. The rotation
//    is undone afterwards by re-applying RoPE with the sine negated.
//  * The sink SEEDS the softmax: m starts at sinks[h] and the denominator starts at exp(sink - m).
//    It contributes NO value vector, so the weights may sum to less than one. Adding it as a
//    pseudo-key with a value silently changes every output.
//  * One shared latent serves all 64 heads. The score uses all 512 dims and the value is the same
//    512 dims of the same vector; per-head K/V is never materialised.
//  * A compressed row the indexer did not select is skipped ENTIRELY — it must not reach the max
//    reduction either, or it shifts the softmax.

#include "gpu/attention_gpu.h"
#include "gpu/attn_flash.h"
#include "gpu/hip_common.h"
#include "gpu/device_math.h"

#include "ui/log.h"
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace aff {

// ---- cache element codecs ----------------------------------------------------------------------
//
// A row is laid out data-then-scales: `width` elements, followed for FP8 by width/64 f32 scales.
// The scale is a power of two chosen exactly as fp8_kv_qat chooses it, so for the nope part the
// encode below reproduces bytes the model already committed to rather than introducing rounding.

// f32 -> E4M3 byte, round-to-nearest-even, clamped to the format's +-448. For the nope part the
// input has already been through fp8_kv_qat, so this reproduces the exact byte the model chose;
// the rounding below only does real work on the rope tail.
static inline uint8_t e4m3_encode(float f) {
  uint32_t u; __builtin_memcpy(&u, &f, 4);
  const uint32_t sign = (u >> 24) & 0x80u;
  float a = std::fabs(f);
  if (!(a > 0.0f)) return (uint8_t)sign;
  if (a > 448.0f) a = 448.0f;
  if (a >= 0x1p-6f) {                                   // normal
    __builtin_memcpy(&u, &a, 4);
    int e = (int)((u >> 23) & 0xFFu) - 127;
    const uint32_t m = u & 0x7FFFFFu;
    uint32_t mm = m >> 20;
    const uint32_t rem = m & 0xFFFFFu, half = 1u << 19;
    if (rem > half || (rem == half && (mm & 1u))) {
      if (++mm == 8u) { mm = 0; ++e; }
    }
    if (e > 8) { e = 8; mm = 7; }
    return (uint8_t)(sign | ((uint32_t)(e + 7) << 3) | mm);
  }
  uint32_t mm = (uint32_t)std::lrintf(a * 512.0f);       // subnormal: m * 2^-9
  if (mm > 7u) mm = 7u;
  return (uint8_t)(sign | mm);
}

// f32 -> IEEE binary16 bits, round-to-nearest-even. The input has already been rounded to fp16 on
// the way into the cache, so on the model's own data this is exact; the general path is here so the
// format is a format and not an assumption.
static inline uint16_t f16_encode(float f) {
  uint32_t u; __builtin_memcpy(&u, &f, 4);
  const uint16_t sign = (uint16_t)((u >> 16) & 0x8000u);
  int e = (int)((u >> 23) & 0xFFu) - 127;
  uint32_t m = u & 0x7FFFFFu;
  if (e == 128) return (uint16_t)(sign | 0x7C00u | (m ? 0x200u : 0u));   // inf / nan
  if (e > 15) return (uint16_t)(sign | 0x7C00u);                         // overflow -> inf
  if (e >= -14) {                                                        // normal
    uint32_t mm = m >> 13;
    const uint32_t rem = m & 0x1FFFu, half = 1u << 12;
    if (rem > half || (rem == half && (mm & 1u))) {
      if (++mm == 0x400u) { mm = 0; ++e; if (e > 15) return (uint16_t)(sign | 0x7C00u); }
    }
    return (uint16_t)(sign | ((uint32_t)(e + 15) << 10) | mm);
  }
  if (e < -25) return sign;                                              // underflow -> +-0
  m |= 0x800000u;                                                        // subnormal
  const int sh = -e - 14 + 13;
  uint32_t mm = m >> sh;
  const uint32_t rem = m & ((1u << sh) - 1u), half = 1u << (sh - 1);
  if (rem > half || (rem == half && (mm & 1u))) ++mm;
  return (uint16_t)(sign | mm);
}

// One 64-wide group of the nope part: the power-of-2 scale, then the bytes. Shared by FP8 and MIXED
// so there is exactly one place that decides what an E4M3 cache group looks like.
static inline float e4m3_group(const float* src, uint8_t* o) {
  float amax = 0.0f;
  for (uint32_t i = 0; i < kKvScaleGroup; ++i) amax = std::max(amax, std::fabs(src[i]));
  if (amax < 1e-4f) amax = 1e-4f;
  const float scale = std::ldexp(1.0f, (int)std::ceil(std::log2(amax / 448.0f)));
  for (uint32_t i = 0; i < kKvScaleGroup; ++i)
    o[i] = e4m3_encode(std::min(448.0f, std::max(-448.0f, src[i] / scale)));
  return scale;
}

void kv_encode_row(KvDtype d, const float* src, uint32_t width, uint32_t n_rot, void* dst) {
  if (d == KvDtype::F32) { __builtin_memcpy(dst, src, (size_t)width * 4); return; }
  // One encoder for FP8R, shared with the flash kernel's bench, so a change to the row layout
  // cannot land in one of them and not the other.
  if (d == KvDtype::FP8R) { kv_encode_fp8r_row(src, (uint8_t*)dst); return; }
  uint8_t* o = (uint8_t*)dst;
  // MIXED stops the E4M3 part at the rope tail and stores the tail at the precision the model kept
  // it in; FP8 runs E4M3 over the whole row, which re-rounds a tail that was never quantised.
  const uint32_t n_e4m3 = (d == KvDtype::MIXED) ? width - n_rot : width;
  float* sc = (float*)(o + ((d == KvDtype::MIXED) ? n_e4m3 + n_rot * 2 : width));
  for (uint32_t g = 0; g * kKvScaleGroup < n_e4m3; ++g) {
    const uint32_t off = g * kKvScaleGroup;
    sc[g] = e4m3_group(src + off, o + off);
  }
  if (d != KvDtype::MIXED) return;
  uint16_t* h = (uint16_t*)(o + n_e4m3);
  for (uint32_t i = 0; i < n_rot; ++i) h[i] = f16_encode(src[n_e4m3 + i]);
}

namespace {

constexpr int kAttnThreads = 256;
constexpr int kAttnWaves = kAttnThreads / kWave;


// One cached row, decoded on read. DT is a template parameter so the branch is resolved at compile
// time and the f32 path stays a plain load. The row stride comes in from the host rather than being
// recomputed here: two independent expressions for the same layout is how a cache silently reads
// the wrong row.
template <int DT>
struct KvRow {
  const uint8_t* p;
  const float* sc;
  const __half* h;
  uint32_t n_nope;
  __device__ __forceinline__ KvRow(const void* base, uint32_t row, uint32_t stride, uint32_t width,
                                   uint32_t nope)
      : p((const uint8_t*)base + (size_t)row * stride), sc(nullptr), h(nullptr), n_nope(nope) {
    if (DT == 2 || DT == 4) sc = (const float*)(p + width);
    if (DT == 3) {
      h  = (const __half*)(p + nope);
      sc = (const float*)(p + nope + (size_t)(width - nope) * 2);
    }
  }
  __device__ __forceinline__ float operator[](uint32_t i) const {
    if (DT == 0) return ((const float*)p)[i];
    if (DT == 4) return e4m3_decode(p[i]) * sc[0];   // one scale, the whole row
    if (DT == 3 && i >= n_nope) return __half2float(h[i - n_nope]);
    return e4m3_decode(p[i]) * sc[i >> 6];
  }
};

__device__ __forceinline__ float wave_sum(float v) {
  for (int o = kWave / 2; o; o >>= 1) v += __shfl_down(v, o);
  return __shfl(v, 0);
}

// One block per (query head, token). The body is a __device__ function and not a kernel because
// there are now two entry points into it — decode's one token and prefill's sub-batch — and a
// difference between them would not be a crash, it would be text that reads correctly and attends
// to the wrong keys.
//
//   phase 1  each wave cooperatively dots one key at a time, scores land in LDS
//   phase 2  block-wide max, seeded by the sink
//   phase 3  every thread owns width/threads output dims and walks the keys
//
// The raw keys are ring slots raw0, raw0+1, ... raw0+n_raw-1 modulo `ring`. raw0 + j < 2 * ring
// always (raw0 < ring and j < n_raw <= ring), so the wrap is one conditional subtract.
// ---- the attention body, over kHeadsPerBlk query heads at once -------------------------------------
//
// This model is MQA: ONE kv head shared by every query head. With a grid of (head, token), one block
// per pair, every block re-reads the whole K/V window for its token — once per head. That made the
// kernel L2-bandwidth-bound at several times its own VALU roofline. The bytes were the problem, not
// the arithmetic.
//
// So a block now owns kHeadsPerBlk heads and loads each K/V element ONCE for all of them:
//
//   scores  a wave loads k[j] into registers and dots it against kHeadsPerBlk q vectors
//   AV      a thread loads k[d] once and accumulates into kHeadsPerBlk outputs
//
// kHeadsPerBlk == kAttnWaves is not a coincidence: it lets wave `w` own head `w`'s softmax entirely,
// so the per-head max and denominator are wave reductions with no block barrier at all.
constexpr uint32_t kHeadsPerBlk = kAttnWaves;          // 8
// Elements a lane stages in registers during the score loop. The real trip count is `width / kWave`,
// so this is the widest head the kernel can hold — a compile-time bound on a runtime shape.
constexpr uint32_t kQPerLane = 16;
constexpr uint32_t kMaxWidth = kQPerLane * (uint32_t)kWave;   // 512

// A wider head would run off the end of that register array, which is not a crash: it is text that
// reads correctly and attended to whatever was in the neighbouring registers.
void require_width(uint32_t width, const char* who) {
  if (width <= kMaxWidth) return;
  aff::ui::fatal("fatal: %s was given a %u-wide head. This kernel stages a head in registers "
               "and is compiled for at most %u.\n", who, width, kMaxWidth);
  std::abort();
}

__device__ __forceinline__ float bf16_to_f32_att(uint16_t h) {
  return __uint_as_float((uint32_t)h << 16);
}

// QB: the query buffer is bf16 rather than f32. The batched prefill path feeds it from the arena,
// where the heads buffer is bf16 and the flash kernel beside it reads the same thing; the
// single-token path still passes f32.
template <int DT, bool QB = false>
__device__ __forceinline__ void attend_body(
    const void* __restrict__ qbase, const void* __restrict__ raw, const void* __restrict__ comp,
    const uint8_t* __restrict__ allowed, const float* __restrict__ sinks, bool has_sink,
    float* __restrict__ obase, float* __restrict__ s_score, uint32_t raw0, uint32_t n_raw,
    uint32_t ring, uint32_t n_comp, uint32_t width, uint32_t nope, uint32_t stride, float scale,
    uint32_t nh) {

  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWave - 1), wave = tid / kWave;
  const uint32_t n_keys = n_raw + n_comp;
  // Indexed by the block's OWN head count, not the constant. The launcher sizes LDS as
  // nhb * n_keys + nhb, so a fixed kHeadsPerBlk stride wrote past the end for any smaller nhb --
  // and the small-key cases still passed, because the stray address was at least consistent between
  // the write and the read. Only n_keys >= 128 pushed it far enough out to corrupt anything.
  float* s_inv = s_score + (size_t)nh * n_keys;               // nh floats

  const float NEG = -3.0e38f;
  const uint32_t per_lane = width / kWave;                  // 16 at width 512

  // ---- scores. One K row per wave iteration, reused across every head in the block.
  for (uint32_t j = wave; j < n_keys; j += kAttnWaves) {
    const bool skip = (j >= n_raw) && allowed && !allowed[j - n_raw];
    if (skip) {
      if (lane == 0)
        for (uint32_t h = 0; h < nh; ++h) s_score[(size_t)h * n_keys + j] = NEG;
      continue;
    }
    uint32_t rs = raw0 + j; if (rs >= ring) rs -= ring;
    const KvRow<DT> k = (j < n_raw) ? KvRow<DT>(raw, rs, stride, width, nope)
                                    : KvRow<DT>(comp, j - n_raw, stride, width, nope);
    float kv[kQPerLane];
    for (uint32_t t = 0; t < per_lane; ++t) kv[t] = k[lane + t * kWave];
    for (uint32_t h = 0; h < nh; ++h) {
      const float* qh = QB ? nullptr : (const float*)qbase + (size_t)h * width;
      const uint16_t* qhb = QB ? (const uint16_t*)qbase + (size_t)h * width : nullptr;
      float d = 0.0f;
      for (uint32_t t = 0; t < per_lane; ++t)
        d = fmaf(QB ? bf16_to_f32_att(qhb[lane + t * kWave]) : qh[lane + t * kWave], kv[t], d);
      d = wave_sum(d);
      if (lane == 0) s_score[(size_t)h * n_keys + j] = d * scale;
    }
  }
  __syncthreads();

  // ---- softmax, one head per wave. Purely wave-level, so no barrier until the scores are needed
  // by every thread again.
  if (wave < nh) {
    float* row = s_score + (size_t)wave * n_keys;
    const float sink = sinks ? sinks[wave] : 0.0f;
    float m = has_sink ? sink : NEG;
    for (uint32_t j = lane; j < n_keys; j += kWave) m = fmaxf(m, row[j]);
    for (int o = kWave / 2; o; o >>= 1) m = fmaxf(m, __shfl_xor(m, o));
    float den = 0.0f;
    for (uint32_t j = lane; j < n_keys; j += kWave) {
      const float sj = row[j];
      const float e = (sj != NEG) ? __expf(sj - m) : 0.0f;
      row[j] = e;
      den += e;
    }
    for (int o = kWave / 2; o; o >>= 1) den += __shfl_xor(den, o);
    if (has_sink) den += __expf(sink - m);
    if (lane == 0) s_inv[wave] = (den > 0.0f) ? 1.0f / den : 0.0f;
  }
  __syncthreads();

  // ---- weighted sum. k[d] is loaded once and feeds every head, which is the other half of the
  // per-head re-read this kernel used to pay.
  for (uint32_t d = tid; d < width; d += kAttnThreads) {
    float acc[kHeadsPerBlk];
    for (uint32_t h = 0; h < nh; ++h) acc[h] = 0.0f;
    for (uint32_t j = 0; j < n_keys; ++j) {
      uint32_t rs = raw0 + j; if (rs >= ring) rs -= ring;
      const KvRow<DT> k = (j < n_raw) ? KvRow<DT>(raw, rs, stride, width, nope)
                                      : KvRow<DT>(comp, j - n_raw, stride, width, nope);
      const float kd = k[d];
      for (uint32_t h = 0; h < nh; ++h) acc[h] = fmaf(s_score[(size_t)h * n_keys + j], kd, acc[h]);
    }
    for (uint32_t h = 0; h < nh; ++h) obase[(size_t)h * width + d] = acc[h] * s_inv[h];
  }
}

template <int DT>
__global__ __launch_bounds__(kAttnThreads) void mqa_attend_kernel(
    const float* __restrict__ q, const void* __restrict__ raw, const void* __restrict__ comp,
    const uint8_t* __restrict__ allowed, const float* __restrict__ sinks,
    float* __restrict__ out, uint32_t raw0, uint32_t n_raw, uint32_t ring, uint32_t n_comp,
    uint32_t width, uint32_t nope, uint32_t stride, float scale, uint32_t n_head, uint32_t nhb) {
  extern __shared__ float s_score[];              // nhb * n_keys, then nhb
  const uint32_t h0 = blockIdx.x * nhb;
  attend_body<DT>((const void*)(q + (size_t)h0 * width), raw, comp, allowed,
                  sinks ? sinks + h0 : nullptr,
                  sinks != nullptr, out + (size_t)h0 * width, s_score, raw0, n_raw, ring, n_comp,
                  width, nope, stride, scale, min(nhb, n_head - h0));
}

// grid(n_head, n). blockIdx.y names the token; everything else is the same attention.
//
// The window comes from the ABSOLUTE position and nothing else — no cursor is threaded through,
// which is what makes committing the whole sub-batch first safe. `n_comp_v` is per token because
// the compressor emits a row every `ratio` positions, so tokens inside one sub-batch legitimately
// see different numbers of compressed rows; taking the batch's max here would let an early token
// attend to a row derived from a later one.
template <int DT, bool QB>
__global__ __launch_bounds__(kAttnThreads) void mqa_attend_batch_kernel(
    const void* __restrict__ q, const void* __restrict__ raw, const void* __restrict__ comp,
    const uint8_t* __restrict__ allowed, const uint32_t* __restrict__ n_comp_v,
    const float* __restrict__ sinks, float* __restrict__ out, uint32_t b0, uint32_t pos0,
    uint32_t sliding, uint32_t ring, uint32_t max_comp, uint32_t n_head, uint32_t width,
    uint32_t nope, uint32_t stride, float scale, uint32_t nhb, uint32_t flat_n,
    uint32_t flat_raw0) {

  extern __shared__ float s_score[];
  const uint32_t h0 = blockIdx.x * nhb, t = blockIdx.y;
  const uint32_t pos = pos0 + t;
  // Causal by construction: the window IS [pos-sliding+1, pos], so there is no mask to apply. The
  // DSpark draft is the one caller that wants the same window for every token — see the header.
  const uint32_t n_raw = flat_n ? flat_n : ((pos + 1u < sliding) ? pos + 1u : sliding);
  const uint32_t raw0 = flat_n ? flat_raw0 : ((pos + 1u - n_raw) % ring);
  const uint32_t n_comp = n_comp_v ? n_comp_v[t] : 0u;
  const size_t tok = (size_t)(b0 + t) * n_head * width + (size_t)h0 * width;
  const void* qp = QB ? (const void*)((const uint16_t*)q + tok)
                      : (const void*)((const float*)q + tok);
  attend_body<DT, QB>(qp, raw, comp, allowed ? allowed + (size_t)t * max_comp : nullptr,
                      sinks ? sinks + h0 : nullptr, sinks != nullptr, out + tok, s_score, raw0,
                      n_raw, ring, n_comp, width, nope, stride, scale, min(nhb, n_head - h0));
}


// One pow2 scale for a whole `width`-wide row, reduced through LDS in two levels: `width/64` group
// maxima, then one thread over those. `s_abs` must already hold |v| for every element and `s_scale`
// must have width/64 + 1 slots; the chosen scale lands in s_scale[width/64] and in the row.
//
// The group level is not an optimisation, it is the same shape the per-64 encoders use — and the
// reason THEY use it is that `__shfl_xor(v, 32)` is a no-op at wave32, so an apparent 64-lane
// shuffle reduction silently scales each group from its first 32 elements only.
__device__ __forceinline__ void row_scale(float* s_abs, float* s_scale, uint32_t i, uint32_t width,
                                          uint8_t* slot) {
  const uint32_t n_grp = width / 64u;
  __syncthreads();
  if (i < n_grp) {
    float a = 0.0f;
    for (uint32_t j = 0; j < 64u; ++j) a = fmaxf(a, s_abs[i * 64u + j]);
    s_scale[i] = a;
  }
  __syncthreads();
  if (i == 0) {
    float a = 0.0f;
    for (uint32_t g = 0; g < n_grp; ++g) a = fmaxf(a, s_scale[g]);
    const float sc = exp2f(ceilf(log2f(fmaxf(a, 1e-6f) / 448.0f)));
    s_scale[n_grp] = sc;
    *(float*)(slot + width) = sc;
  }
  __syncthreads();
}

// One block, `width` threads, one cache row out.
//
// THE GROUP REDUCTION GOES THROUGH LDS, NOT SHUFFLES. It used to be
//     for (int o = 32; o; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o));
// which looks like a 64-lane reduction and is not: waves are 32 lanes here, so `__shfl_xor(v, 32)`
// returns v unchanged and the scale for each 64-group came from its FIRST 32 elements only. When
// the back half held a larger magnitude the scale was too small and those elements clamped at 448 —
// silently, on the device commit path only, at whatever rate the data happened to trigger it.
template <int DT>
__global__ void kv_store_kernel(const float* __restrict__ kv, uint8_t* __restrict__ slot,
                                uint32_t width, uint32_t nope) {
  extern __shared__ float s_abs[];                  // width floats, then width/64 scales
  const uint32_t i = threadIdx.x;
  // Every row the reference caches has been through np.float16 first. Round here so that what the
  // scale is chosen from, and what gets encoded, is the value the model committed to.
  const float v = (i < width) ? __half2float(__float2half(kv[i])) : 0.0f;

  if (DT == 0) {                                    // uniform over the block: DT is compile-time
    if (i < width) ((float*)slot)[i] = v;
    return;
  }
  float* s_scale = s_abs + width;
  if (DT == 4) {                                    // one scale over the whole row
    s_abs[i] = (i < width) ? fabsf(v) : 0.0f;
    row_scale(s_abs, s_scale, i, width, slot);
    if (i < width) slot[i] = e4m3_enc(fminf(448.0f, fmaxf(-448.0f, v / s_scale[width / 64u])));
    return;
  }
  const uint32_t n_e4m3 = (DT == 3) ? nope : width;
  const uint32_t n_grp  = n_e4m3 / 64u;
  float* g_scale = (float*)(slot + ((DT == 3) ? nope + (size_t)(width - nope) * 2 : width));

  if (i < n_e4m3) s_abs[i] = fabsf(v);
  __syncthreads();
  if (i < n_grp) {
    float amax = 0.0f;
    for (uint32_t j = 0; j < 64u; ++j) amax = fmaxf(amax, s_abs[i * 64u + j]);
    s_scale[i] = exp2f(ceilf(log2f(fmaxf(amax, 1e-4f) / 448.0f)));
    g_scale[i] = s_scale[i];
  }
  __syncthreads();

  if (i < n_e4m3) slot[i] = e4m3_enc(fminf(448.0f, fmaxf(-448.0f, v / s_scale[i >> 6])));
  else if (DT == 3 && i < width) ((__half*)(slot + nope))[i - nope] = __float2half(v);
}

// kv_qat + kv_store for a whole sub-batch, one block per token, from the chunk's dim-major latent.
//
// The two steps are fused rather than merely batched because they were never separable in the
// first place: the pair only existed as two launches so that the single-token path could reuse
// kv_store_hip for the host-supplied row. Both halves are transcribed from the kernels above and
// must stay identical to them — the QAT's per-64 scale is chosen from the RAW column and the
// store's from the QAT'd, f16-rounded value, and swapping those two silently widens the cache
// error by the factor that the "reduction goes through LDS, not shuffles" comment describes.
// The learned norm and the RoPE tail that used to run in front of this kernel, as a prologue.
// `attn_kv` wrote the latent to d_u with a single-block launch and this kernel read it straight
// back — two launches for a couple of KiB, with nothing between them. Norm=true takes `v`
// from the prologue instead of from memory and NOTHING ELSE CHANGES — the QAT, the f16 round and
// the store are the same code, not a transcription of it.
//
// Bit-identical to rms_norm_rope_kernel followed by this kernel. That one ran 256 threads, so the
// reduction here is restricted to threads 0-255 with the same strided partition: 256 is 8 whole
// waves, so the __shfl_down is never partially predicated. The rotation is the same rope_cs and the
// same two expressions; a tail thread reads its partner out of LDS instead of out of global.
struct KvNorm {
  const float* w;                                   // learned norm weight, may be null
  float eps;
  uint32_t n_rot;
  uint32_t pos;
  float freq_scale, theta_scale, ext_factor, lo, hi;
};

template <int DT, bool Norm = false>
__global__ void kv_commit_batch_kernel(const float* __restrict__ kv, uint32_t nb, uint32_t b0,
                                       uint8_t* __restrict__ raw, uint64_t rb, uint32_t pos0,
                                       uint32_t ring, uint32_t width, uint32_t nope,
                                       KvNorm nrm = KvNorm{}) {
  extern __shared__ float s_abs[];                  // width floats, then width/64 scales
  float* s_scale = s_abs + width;
  const uint32_t t = blockIdx.x;
  const uint32_t i = threadIdx.x;
  uint8_t* slot = raw + (size_t)((pos0 + t) % ring) * rb;

  float v;
  if (Norm) {
    float* red = s_scale + width / 64u + 1;         // 32 more floats; the launcher adds them
    constexpr uint32_t kNT = 256;                   // rms_norm_rope_kernel's blockDim, exactly
    if (i < kNT) {
      float acc = 0.0f;
      for (uint32_t j = i; j < width; j += kNT) { const float f = kv[j]; acc += f * f; }
      for (int o = 16; o; o >>= 1) acc += __shfl_down(acc, o);
      if ((i & 31u) == 0u) red[i >> 5] = acc;
    }
    __syncthreads();
    if (i == 0) {
      float s = 0.0f;
      for (uint32_t k = 0; k < kNT / 32u; ++k) s += red[k];
      red[0] = rsqrtf(s / (float)width + nrm.eps);
    }
    __syncthreads();
    const float inv = red[0];
    if (i < width) s_abs[i] = kv[i] * inv * (nrm.w ? nrm.w[i] : 1.0f);
    __syncthreads();
    if (i < nope) {
      v = s_abs[i];
    } else if (i < width) {
      // This thread's element is half of a rotated pair; the other half is its LDS neighbour.
      const uint32_t r = i - nope, k = r >> 1;
      float c, sn;
      rope_cs(k, nrm.pos, nrm.freq_scale, nrm.theta_scale, nrm.ext_factor, nrm.lo, nrm.hi, 1.0f,
              &c, &sn);
      const float x0 = s_abs[nope + 2u * k], x1 = s_abs[nope + 2u * k + 1u];
      v = (r & 1u) ? (x0 * sn + x1 * c) : (x0 * c - x1 * sn);
    } else {
      v = 0.0f;
    }
    __syncthreads();                                // s_abs is the QAT's scratch from here on
  } else {
    v = (i < width) ? kv[(size_t)i * nb + b0 + t] : 0.0f;
  }

  // ---- fp8_kv_qat, in place, NOPE part only ----
  const uint32_t n_qat = (nope / 64u) * 64u;
  s_abs[i] = (i < n_qat) ? fabsf(v) : 0.0f;
  __syncthreads();
  if (i < n_qat / 64u) {
    float amax = 0.0f;
    for (uint32_t j = 0; j < 64u; ++j) amax = fmaxf(amax, s_abs[i * 64u + j]);
    s_scale[i] = exp2f(ceilf(log2f(fmaxf(amax, 1e-4f) / 448.0f)));
  }
  __syncthreads();
  if (i < n_qat) {
    const float sc = s_scale[i >> 6];
    v = e4m3_decode(e4m3_enc(fminf(448.0f, fmaxf(-448.0f, v / sc)))) * sc;
  }
  __syncthreads();

  // ---- kv_store: f16-round first, then encode ----
  v = __half2float(__float2half(v));
  if (DT == 0) {
    if (i < width) ((float*)slot)[i] = v;
    return;
  }
  if (DT == 4) {
    s_abs[i] = (i < width) ? fabsf(v) : 0.0f;
    row_scale(s_abs, s_scale, i, width, slot);
    if (i < width) slot[i] = e4m3_enc(fminf(448.0f, fmaxf(-448.0f, v / s_scale[width / 64u])));
    return;
  }
  const uint32_t n_e4m3 = (DT == 3) ? nope : width;
  const uint32_t n_grp  = n_e4m3 / 64u;
  float* g_scale = (float*)(slot + ((DT == 3) ? nope + (size_t)(width - nope) * 2 : width));

  s_abs[i] = (i < n_e4m3) ? fabsf(v) : 0.0f;
  __syncthreads();
  if (i < n_grp) {
    float amax = 0.0f;
    for (uint32_t j = 0; j < 64u; ++j) amax = fmaxf(amax, s_abs[i * 64u + j]);
    s_scale[i] = exp2f(ceilf(log2f(fmaxf(amax, 1e-4f) / 448.0f)));
    g_scale[i] = s_scale[i];
  }
  __syncthreads();

  if (i < n_e4m3) slot[i] = e4m3_enc(fminf(448.0f, fmaxf(-448.0f, v / s_scale[i >> 6])));
  else if (DT == 3 && i < width) ((__half*)(slot + nope))[i - nope] = __float2half(v);
}

} // namespace


void kv_store_hip(KvDtype d, const float* kv, void* slot, uint32_t width, uint32_t n_rot,
                  void* stream) {
  const size_t lds = ((size_t)width + width / 64u + 1) * sizeof(float);
  const dim3 g(1), b(width);
  hipStream_t st = (hipStream_t)stream;
  const uint32_t nope = width - n_rot;
  switch (d) {
    case KvDtype::F32:
      hipLaunchKernelGGL((kv_store_kernel<0>), g, b, lds, st, kv, (uint8_t*)slot, width, nope);
      break;
    case KvDtype::FP8:
      hipLaunchKernelGGL((kv_store_kernel<2>), g, b, lds, st, kv, (uint8_t*)slot, width, nope);
      break;
    case KvDtype::MIXED:
      hipLaunchKernelGGL((kv_store_kernel<3>), g, b, lds, st, kv, (uint8_t*)slot, width, nope);
      break;
    case KvDtype::FP8R:
      hipLaunchKernelGGL((kv_store_kernel<4>), g, b, lds, st, kv, (uint8_t*)slot, width, nope);
      break;
  }
  AFF_HIP_CHECK(hipGetLastError());
}

void kv_commit_batch_hip(KvDtype d, const float* kv, uint32_t nb, uint32_t b0, uint32_t n,
                         uint8_t* raw, uint32_t pos0, uint32_t ring, uint32_t width,
                         uint32_t n_rot, void* stream) {
  if (!n || !width) return;
  const size_t lds = ((size_t)width + width / 64u + 1) * sizeof(float);
  const dim3 g(n), b(width);
  hipStream_t st = (hipStream_t)stream;
  const uint32_t nope = width - n_rot;
  const uint64_t rb = kv_row_bytes(d, width, n_rot);
  switch (d) {
    case KvDtype::F32:
      hipLaunchKernelGGL((kv_commit_batch_kernel<0>), g, b, lds, st, kv, nb, b0, raw, rb, pos0,
                         ring, width, nope, KvNorm{}); break;
    case KvDtype::FP8:
      hipLaunchKernelGGL((kv_commit_batch_kernel<2>), g, b, lds, st, kv, nb, b0, raw, rb, pos0,
                         ring, width, nope, KvNorm{}); break;
    case KvDtype::MIXED:
      hipLaunchKernelGGL((kv_commit_batch_kernel<3>), g, b, lds, st, kv, nb, b0, raw, rb, pos0,
                         ring, width, nope, KvNorm{}); break;
    case KvDtype::FP8R:
      hipLaunchKernelGGL((kv_commit_batch_kernel<4>), g, b, lds, st, kv, nb, b0, raw, rb, pos0,
                         ring, width, nope, KvNorm{}); break;
  }
  AFF_HIP_CHECK(hipGetLastError());
}

// The same commit with the learned norm and the RoPE tail folded in front of it, for the ONE-token
// decode shape. Returns false having launched nothing when the shape is not the one the prologue
// assumes, and the caller must then run rms_norm_rope_kernel itself.
bool kv_norm_commit_hip(KvDtype d, const float* latent, const float* nw, float eps, uint8_t* raw,
                        uint32_t pos, uint32_t ring, uint32_t width, uint32_t n_rot,
                        float freq_scale, float theta_scale, float ext_factor, float lo, float hi,
                        void* stream) {
  // 256 threads must be 8 whole waves inside a block of `width`, the rotated tail must be an even
  // count inside the block, and the reduction needs its 32 floats past the QAT scales.
  if (width < 256u || width % 64u || n_rot % 2u || n_rot > width) return false;
  const size_t lds = ((size_t)width + width / 64u + 1 + 32) * sizeof(float);
  const dim3 g(1), b(width);
  hipStream_t st = (hipStream_t)stream;
  const uint32_t nope = width - n_rot;
  const uint64_t rb = kv_row_bytes(d, width, n_rot);
  const KvNorm nrm{nw, eps, n_rot, pos, freq_scale, theta_scale, ext_factor, lo, hi};
  switch (d) {
    case KvDtype::F32:
      hipLaunchKernelGGL((kv_commit_batch_kernel<0, true>), g, b, lds, st, latent, 1u, 0u, raw, rb,
                         pos, ring, width, nope, nrm); break;
    case KvDtype::FP8:
      hipLaunchKernelGGL((kv_commit_batch_kernel<2, true>), g, b, lds, st, latent, 1u, 0u, raw, rb,
                         pos, ring, width, nope, nrm); break;
    case KvDtype::MIXED:
      hipLaunchKernelGGL((kv_commit_batch_kernel<3, true>), g, b, lds, st, latent, 1u, 0u, raw, rb,
                         pos, ring, width, nope, nrm); break;
    case KvDtype::FP8R:
      hipLaunchKernelGGL((kv_commit_batch_kernel<4, true>), g, b, lds, st, latent, 1u, 0u, raw, rb,
                         pos, ring, width, nope, nrm); break;
  }
  AFF_HIP_CHECK(hipGetLastError());
  return true;
}

void mqa_attend_hip(KvDtype dt, const float* q, const void* raw, const void* comp,
                    const uint8_t* allowed, const float* sinks, float* out, uint32_t n_head,
                    uint32_t raw0, uint32_t n_raw, uint32_t ring, uint32_t n_comp, uint32_t width,
                    uint32_t n_rot, float scale, void* stream) {
  const uint32_t n_keys = n_raw + n_comp;
  if (!n_head || !n_keys) return;
  require_width(width, "mqa_attend_hip");
  // ONE head a block here, unlike the batched path. Grouping heads trades grid width for K/V reuse,
  // and at a single token the grid IS the parallelism: grouping makes this kernel twice as slow
  // while making the batched one more than twice as fast. The reuse is worth nothing when there is
  // only one token's window to reuse and most of the card is left idle to get it.
  const uint32_t nhb = 1;
  const size_t lds = ((size_t)nhb * n_keys + nhb) * sizeof(float);
  const dim3 g((n_head + nhb - 1) / nhb), b(kAttnThreads);
  hipStream_t st = (hipStream_t)stream;
  const uint32_t sd = (uint32_t)kv_row_bytes(dt, width, n_rot);
  const uint32_t nope = width - n_rot;
  switch (dt) {
    case KvDtype::F32:
      hipLaunchKernelGGL((mqa_attend_kernel<0>), g, b, lds, st, q, raw, comp, allowed, sinks, out,
                         raw0, n_raw, ring, n_comp, width, nope, sd, scale, n_head, nhb); break;
    case KvDtype::FP8:
      hipLaunchKernelGGL((mqa_attend_kernel<2>), g, b, lds, st, q, raw, comp, allowed, sinks, out,
                         raw0, n_raw, ring, n_comp, width, nope, sd, scale, n_head, nhb); break;
    case KvDtype::MIXED:
      hipLaunchKernelGGL((mqa_attend_kernel<3>), g, b, lds, st, q, raw, comp, allowed, sinks, out,
                         raw0, n_raw, ring, n_comp, width, nope, sd, scale, n_head, nhb); break;
    case KvDtype::FP8R:
      hipLaunchKernelGGL((mqa_attend_kernel<4>), g, b, lds, st, q, raw, comp, allowed, sinks, out,
                         raw0, n_raw, ring, n_comp, width, nope, sd, scale, n_head, nhb); break;
  }
  AFF_HIP_CHECK(hipGetLastError());
}

void mqa_attend_batch_hip(KvDtype dt, const void* q, const void* raw, const void* comp,
                          const uint8_t* allowed, const uint32_t* n_comp_v, const float* sinks,
                          float* out, uint32_t b0, uint32_t n, uint32_t /*nb*/, uint32_t pos0,
                          uint32_t sliding, uint32_t ring, uint32_t n_comp_max, uint32_t max_comp,
                          uint32_t n_head, uint32_t width, uint32_t n_rot, float scale,
                          void* stream, bool q_bf16, uint32_t flat_n, uint32_t flat_raw0) {
  if (!n_head || !n) return;
  require_width(width, "mqa_attend_batch_hip");
  // Sized for the WORST token in the sub-batch. Every block allocates the same LDS, so this is the
  // batch max of n_raw + n_comp, not each token's own; a block simply leaves the tail unused.
  const uint32_t last = pos0 + n - 1;
  const uint32_t n_raw_max = flat_n ? flat_n : ((last + 1u < sliding) ? last + 1u : sliding);
  // Batched: n tokens already give the grid its width, so spend head-grouping on K/V reuse.
  const uint32_t nhb = kHeadsPerBlk;
  const size_t lds = ((size_t)nhb * (n_raw_max + n_comp_max) + nhb) * sizeof(float);
  const dim3 g((n_head + nhb - 1) / nhb, n), b(kAttnThreads);
  hipStream_t st = (hipStream_t)stream;
  const uint32_t sd = (uint32_t)kv_row_bytes(dt, width, n_rot);
  const uint32_t nope = width - n_rot;
#define AFF_ATTB(DTV, QBV)                                                                     \
  hipLaunchKernelGGL((mqa_attend_batch_kernel<DTV, QBV>), g, b, lds, st, q, raw, comp, allowed,   \
                     n_comp_v, sinks, out, b0, pos0, sliding, ring, max_comp, n_head, width,      \
                     nope, sd, scale, nhb, flat_n, flat_raw0)
#define AFF_ATTQ(DTV) do { if (q_bf16) AFF_ATTB(DTV, true); else AFF_ATTB(DTV, false); } while (0)
  switch (dt) {
    case KvDtype::F32:   AFF_ATTQ(0); break;
    case KvDtype::FP8:   AFF_ATTQ(2); break;
    case KvDtype::MIXED: AFF_ATTQ(3); break;
    case KvDtype::FP8R:  AFF_ATTQ(4); break;
  }
#undef AFF_ATTQ
#undef AFF_ATTB
  AFF_HIP_CHECK(hipGetLastError());
}

} // namespace aff
