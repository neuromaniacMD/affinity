// See compressor_gpu.h for why a state machine that looks sequential is a fixed window of reads.

#include "gpu/compressor_gpu.h"
#include "gpu/device_math.h"
#include "gpu/hip_common.h"

#include "ui/log.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdio>
#include <cstdlib>

namespace aff {
namespace {

// A position's projections, wherever they live. Inside the chunk they are token-major rows of the
// GEMM output; before it they are ring slots left by the previous chunk. A position BEFORE the
// sequence starts is neither, and comes back null so the caller can substitute the reset value —
// score -1e30, which the softmax turns into weight zero.
struct Window {
  const float* kv;
  const float* sc;
  const float* hkv;
  const float* hsc;
  uint32_t pos0, width, hist;

  __device__ __forceinline__ bool at(int64_t p, const float** k, const float** s) const {
    if (p < 0) return false;
    if (p >= (int64_t)pos0) {
      const size_t r = (size_t)(p - (int64_t)pos0);
      *k = kv + r * width; *s = sc + r * width;
    } else {
      const size_t r = (size_t)p & (size_t)(hist - 1u);
      *k = hkv + r * width; *s = hsc + r * width;
    }
    return true;
  }
};

// ---- the indexer's key row, from a rotated 128-wide vector ---------------------------------------
//
// Shared by V4's index compressor (mode 1 below, HADAMARD true) and V4.1's `wk` projection
// (idx_key_kernel, HADAMARD false). One encoder, because two encoders for one cache format is how a
// cache silently changes meaning — the scoring kernel cannot tell which wrote a row.
//
// `s` is head_dim floats and `s_sc` at least head_dim/32; both are the caller's LDS. HADAMARD is a
// template parameter and therefore uniform across the block, which the __syncthreads() inside the
// butterfly requires.
template <bool HADAMARD>
__device__ __forceinline__ void idx_key_store(float v, float* s, float* s_sc, uint32_t j,
                                              uint32_t hd, uint8_t* __restrict__ out,
                                              float* __restrict__ out_scale, uint64_t out_stride,
                                              uint32_t row) {
  if (HADAMARD) {
    s[j] = v;
    __syncthreads();
    for (uint32_t stride = 1; stride < hd; stride <<= 1) {
      float a = 0.0f, b = 0.0f;
      uint32_t ia = 0, ib = 0;
      if (j < hd / 2u) {
        const uint32_t base = (j / stride) * 2u * stride, off = j % stride;
        ia = base + off; ib = base + stride + off;
        a = s[ia]; b = s[ib];
      }
      __syncthreads();
      if (j < hd / 2u) { s[ia] = a + b; s[ib] = a - b; }
      __syncthreads();
    }
    // 1/sqrt(hd) as a literal, because hd is 128 here and no runtime rsqrt reproduces these bits.
    // The callers refuse any other width rather than letting the factor go quietly wrong.
    v = s[j] * 0.08838834764831845f;
    __syncthreads();
  }
  // ---- an E2M1 round trip per 32, with an E8M0 scale ---------------------------------------------
  s[j] = fabsf(v);
  __syncthreads();
  if (j < hd / 32u) {
    float amax = 0.0f;
    for (uint32_t i = 0; i < 32u; ++i) amax = fmaxf(amax, s[j * 32u + i]);
    s_sc[j] = exp2f(ceilf(log2f(fmaxf(amax, 7.052966104933725e-38f) / 6.0f)));
  }
  __syncthreads();
  {
    const float q = s_sc[j >> 5];
    v = e2m1_round_d(fminf(6.0f, fmaxf(-6.0f, v / q))) * q;
  }
  __syncthreads();

  // ---- the cache row: E4M3 with ONE scale, chosen the way idx_quant_kernel chooses it ------------
  v = __half2float(__float2half(v));
  s[j] = fabsf(v);
  __syncthreads();
  for (uint32_t o = hd >> 1; o; o >>= 1) {
    if (j < o) s[j] = fmaxf(s[j], s[j + o]);
    __syncthreads();
  }
  const float q = s[0] > 0.0f ? s[0] * (1.0f / 448.0f) : 1.0f;
  if (j == 0) out_scale[row] = q;
  // The hardware convert, not e4m3_enc: this row is read by the same kernel that used to quantise
  // it, and two encoders for one format is how a cache silently changes meaning.
  const float c = fminf(448.0f, fmaxf(-448.0f, v * (1.0f / q)));
  const int pk8 = __builtin_amdgcn_cvt_pk_fp8_f32(c, c, 0, false);
  out[(size_t)row * out_stride + j] = (uint8_t)(pk8 & 0xFF);
}

// One block per emitted row, `head_dim` threads. Shared: head_dim floats for the row, then
// head_dim/64 + 1 for the group scales.
template <int MODE>
__global__ void compress_kernel(Window win, const float* __restrict__ ape,
                                const float* __restrict__ nrm, uint8_t* __restrict__ out,
                                float* __restrict__ out_scale, uint64_t out_stride, uint32_t slot0,
                                uint32_t hd, uint32_t n_rot, uint32_t ratio, float freq_scale,
                                float theta_scale, float ext_factor, float lo, float hi,
                                float rms_eps, float* __restrict__ lat) {
  extern __shared__ float s[];
  float* s_sc = s + hd;
  const uint32_t j = threadIdx.x;
  const uint32_t width = win.width;

  // The chunk's first emitting position is its first p with p % ratio == ratio-1; they run every
  // `ratio` after that, one per block.
  const uint32_t pe0 = win.pos0 + (ratio - 1u - win.pos0 % ratio);
  const int64_t pe = (int64_t)pe0 + (int64_t)blockIdx.x * ratio;

  // ---- the pool ---------------------------------------------------------------------------------
  //
  // At ratio 4 each output dimension draws from 2*ratio candidates: lane r of the PREVIOUS group at
  // column j, and lane r of the current one at column hd+j. At any other ratio the projection is
  // half as wide and only the current group contributes. Two passes rather than an online rescale,
  // so the reduction order is the host's — max over all candidates, then the weighted sum.
  //
  // The ape row index is the position's own `pos % ratio`, which for both windows is exactly the
  // loop counter: pe ≡ ratio-1, so (pe - 2*ratio + 1 + r) and (pe - ratio + 1 + r) are both ≡ r.
  // V4.1 brings two shapes V4 never had: no `ape` at all (null here), and ratio 1, where a group is
  // ONE position and the layer has no gate projection. The softmax over a single candidate is 1
  // whatever the score, so `gated` false weights every candidate equally instead of reading a score
  // buffer that does not exist — which would be an uninitialised load, not a harmless wrong weight.
  const bool gated = win.sc != nullptr;
  const bool paired = ratio == 4u;
  const int64_t p0 = pe - (int64_t)(ratio - 1u);
  auto score = [&](const float* b, uint32_t c, uint32_t r) -> float {
    return (gated ? b[c] : 0.0f) + (ape ? ape[(size_t)r * width + c] : 0.0f);
  };
  float m = -1e30f;
  for (uint32_t r = 0; r < ratio; ++r) {
    const float *a = nullptr, *b = nullptr;
    if (paired && win.at(p0 - (int64_t)ratio + r, &a, &b))
      m = fmaxf(m, score(b, j, r));
    if (win.at(p0 + r, &a, &b))
      m = fmaxf(m, score(b, hd * (paired ? 1u : 0u) + j, r));
  }
  float pooled = 0.0f;
  if (m > -5e29f) {
    float den = 0.0f, sum = 0.0f;
    for (uint32_t r = 0; r < ratio; ++r) {
      const float *a = nullptr, *b = nullptr;
      if (paired && win.at(p0 - (int64_t)ratio + r, &a, &b)) {
        const float wp = expf(score(b, j, r) - m);
        den += wp; sum += wp * a[j];
      }
      if (win.at(p0 + r, &a, &b)) {
        const uint32_t c = hd * (paired ? 1u : 0u) + j;
        const float wc = expf(score(b, c, r) - m);
        den += wc; sum += wc * a[c];
      }
    }
    pooled = den > 0.0f ? sum / den : 0.0f;
  }

  // ---- rms_norm, then the layer's own weight --------------------------------------------------
  s[j] = pooled * pooled;
  __syncthreads();
  for (uint32_t o = hd >> 1; o; o >>= 1) {
    if (j < o) s[j] += s[j + o];
    __syncthreads();
  }
  float v = pooled * (1.0f / sqrtf(s[0] / (float)hd + rms_eps)) * nrm[j];
  __syncthreads();

  // The latent V4.1's indexer projects its keys from — published HERE, before the RoPE, because
  // that is where the reference reads it (`Attention._compress_kv` runs the indexer and only then
  // rotates and quantises this same storage). Null on every V4 layer and on a V4.1 layer that is
  // not also an index-key owner.
  // DIM-major, [hd][n_out]: `wk` runs over these as a batched GEMM and that is the activation
  // layout it reads, so writing them this way costs a strided store here and saves a transpose
  // launch per layer per chunk.
  if (MODE == 0 && lat) lat[(size_t)j * gridDim.x + blockIdx.x] = v;

  // ---- rope_tail over the trailing n_rot, at the group's FIRST position ------------------------
  s[j] = v;
  __syncthreads();
  const uint32_t n_nope = hd - n_rot;
  if (j >= n_nope && ((j - n_nope) & 1u) == 0u)
    rope_pair(s + n_nope, (j - n_nope) >> 1, (uint32_t)(pe + 1 - (int64_t)ratio), freq_scale,
              theta_scale, ext_factor, lo, hi, 1.0f);
  __syncthreads();
  v = s[j];
  __syncthreads();

  if (MODE == 0) {
    // ---- fp8_kv_qat: E4M3 round trip over the NOPE part, one power-of-two scale per 64 ----------
    const uint32_t n_qat = (n_nope / 64u) * 64u;
    s[j] = (j < n_qat) ? fabsf(v) : 0.0f;
    __syncthreads();
    if (j < n_qat / 64u) {
      float amax = 0.0f;
      for (uint32_t i = 0; i < 64u; ++i) amax = fmaxf(amax, s[j * 64u + i]);
      s_sc[j] = exp2f(ceilf(log2f(fmaxf(amax, 1e-4f) / 448.0f)));
    }
    __syncthreads();
    if (j < n_qat) {
      const float q = s_sc[j >> 6];
      v = e4m3_decode(e4m3_enc(fminf(448.0f, fmaxf(-448.0f, v / q)))) * q;
    }
    __syncthreads();

    // ---- the cache row: f16 first, because that is the value the model committed to -------------
    //
    // The group reduction goes through LDS and not shuffles. Waves are 32 lanes here, so
    // `__shfl_xor(v, 32)` returns v unchanged and an apparent 64-lane reduction would take each
    // scale from its first 32 elements only — see kv_store_kernel, which had exactly that bug.
    v = __half2float(__float2half(v));
    uint8_t* slot = out + (size_t)(slot0 + blockIdx.x) * out_stride;
    s[j] = fabsf(v);
    __syncthreads();
    if (j < hd / 64u) {
      float amax = 0.0f;
      for (uint32_t i = 0; i < 64u; ++i) amax = fmaxf(amax, s[j * 64u + i]);
      s_sc[j] = amax;
    }
    __syncthreads();
    if (j == 0) {
      float amax = 0.0f;
      for (uint32_t g = 0; g < hd / 64u; ++g) amax = fmaxf(amax, s_sc[g]);
      const float q = exp2f(ceilf(log2f(fmaxf(amax, 1e-6f) / 448.0f)));
      s_sc[hd / 64u] = q;
      *(float*)(slot + hd) = q;
    }
    __syncthreads();
    slot[j] = e4m3_enc(fminf(448.0f, fmaxf(-448.0f, v / s_sc[hd / 64u])));
    return;
  }

  // ---- indexer_qat: the Hadamard round trip, then the E4M3 cache row ---------------------------
  // V4's index compressor. Shared with V4.1's key path below, which passes HADAMARD false.
  idx_key_store<true>(v, s, s_sc, j, hd, out, out_scale, out_stride, slot0 + blockIdx.x);
}

// ---- V4.1's index key: k_norm(wk(latent)), rotated, fp4'd ---------------------------------------
//
// One block a row, `head_dim` threads, over the `wk` GEMM's output. Everything that made mode 1 a
// single fused kernel is still fused here — it simply starts one projection later, because V4.1's
// keys come off the KV compressor's latent rather than off a compressor of their own.
//
// The RoPE position is the row's own: cache row `slot0 + blockIdx.x` IS group `slot0 + blockIdx.x`,
// and a latent stands for the first token of its group, so the position is that times `ratio`. Mode
// 0 derives the same number the long way round (`pe + 1 - ratio`, from the emitting position) —
// they agree, and this form is the one the reference writes.
__global__ void idx_key_kernel(const float* __restrict__ src, const float* __restrict__ k_norm,
                               uint8_t* __restrict__ out, float* __restrict__ out_scale,
                               uint64_t out_stride, uint32_t slot0, uint32_t hd, uint32_t ratio,
                               uint32_t n_rot, float freq_scale, float theta_scale,
                               float ext_factor, float lo, float hi, float rms_eps) {
  extern __shared__ float s[];
  float* s_sc = s + hd;
  const uint32_t j = threadIdx.x;
  const uint32_t row = slot0 + blockIdx.x;

  // `src` is dim-major, [hd][n_out] — see CompressJob::lat and the GEMM that produced it.
  float v = src[(size_t)j * gridDim.x + blockIdx.x];

  // ---- k_norm: RMSNorm over the 128, then the layer's own weight ------------------------------
  s[j] = v * v;
  __syncthreads();
  for (uint32_t o = hd >> 1; o; o >>= 1) {
    if (j < o) s[j] += s[j + o];
    __syncthreads();
  }
  v = v * (1.0f / sqrtf(s[0] / (float)hd + rms_eps)) * k_norm[j];
  __syncthreads();

  // ---- rope_tail over the trailing n_rot, at the group's own position --------------------------
  s[j] = v;
  __syncthreads();
  const uint32_t n_nope = hd - n_rot;
  if (j >= n_nope && ((j - n_nope) & 1u) == 0u)
    rope_pair(s + n_nope, (j - n_nope) >> 1, row * ratio, freq_scale, theta_scale, ext_factor, lo,
              hi, 1.0f);
  __syncthreads();
  v = s[j];
  __syncthreads();

  idx_key_store<false>(v, s, s_sc, j, hd, out, out_scale, out_stride, row);
}

// The chunk's last `hist` positions into their own ring slots, the main and index halves at once
// with blockIdx.y picking which. Grid.x is that count, so a chunk shorter than the ring leaves the
// older slots alone — which is right: they still hold positions the next chunk will need.
//
// The two halves touch different buffers and different widths and were two launches purely because
// they are two calls; at decode `take` is 1, so each is a SINGLE BLOCK moving a few KB and the
// launch costs more than the copy. A launch removed here is removed twice a layer for every layer,
// and decode issues thousands of them a token (bench/launch.hip).
struct HistJob {
  const float* kv;
  const float* sc;
  float* hkv;
  float* hsc;
  uint32_t width;
};

__global__ void hist2_kernel(HistJob a, HistJob b, uint32_t n, uint32_t pos0, uint32_t take,
                             uint32_t hist) {
  const HistJob j = blockIdx.y ? b : a;
  if (!j.kv) return;
  const uint32_t src = n - take + blockIdx.x;
  const uint32_t dst = (pos0 + src) & (hist - 1u);
  for (uint32_t i = threadIdx.x; i < j.width; i += blockDim.x) {
    j.hkv[(size_t)dst * j.width + i] = j.kv[(size_t)src * j.width + i];
    j.hsc[(size_t)dst * j.width + i] = j.sc[(size_t)src * j.width + i];
  }
}

__global__ void hist_reset_kernel(float* __restrict__ hkv, float* __restrict__ hsc,
                                  uint32_t width) {
  for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
    hkv[(size_t)blockIdx.x * width + i] = 0.0f;
    hsc[(size_t)blockIdx.x * width + i] = -1e30f;
  }
}

}  // namespace

void compressor_rows_hip(int mode, const CompressJob& j, uint32_t n, uint32_t pos0, uint32_t ratio,
                         uint32_t n_out, uint32_t n_rot, RopeDerived rope, float rms_eps,
                         void* stream) {
  if (!n_out || !n || !ratio || (ratio & (ratio - 1u))) return;   // the ring masks, so power of two
  const uint32_t hd = j.head_dim;
  // The indexer half's Hadamard normalisation is a 128-wide literal, so a different width would be
  // scaled by the wrong factor and the cache row would be silently wrong rather than absent.
  if (mode == 1 && hd != 128u) {
    aff::ui::fatal("fatal: the index compressor is compiled for a 128-wide head and was given %u.\n",
                 hd);
    std::abort();
  }
  Window w{j.kv, j.sc, j.hist_kv, j.hist_sc, pos0, comp_width(ratio, hd), comp_hist(ratio)};
  // head_dim floats for the row, then the group scales: hd/64 + 1 for the KV compressor's per-64
  // QAT and row scale, hd/32 for the indexer's per-32 E2M1. The larger of the two, always.
  const size_t lds = ((size_t)hd + hd / 32u + 1u) * sizeof(float);
  if (mode == 0)
    hipLaunchKernelGGL(compress_kernel<0>, dim3(n_out), dim3(hd), lds, (hipStream_t)stream, w, j.ape,
                       j.norm, j.out, j.out_scale, j.out_stride, j.slot0, hd, n_rot, ratio,
                       rope.freq_scale, rope.theta_scale, rope.ext_factor, rope.lo, rope.hi,
                       rms_eps, j.lat);
  else
    // `lat` is mode 0's alone: the indexer's own compressor has no consumer for its intermediate,
    // and V4.1 — the only caller that asks for one — has no mode-1 layer at all.
    hipLaunchKernelGGL(compress_kernel<1>, dim3(n_out), dim3(hd), lds, (hipStream_t)stream, w, j.ape,
                       j.norm, j.out, j.out_scale, j.out_stride, j.slot0, hd, n_rot, ratio,
                       rope.freq_scale, rope.theta_scale, rope.ext_factor, rope.lo, rope.hi,
                       rms_eps, nullptr);
  AFF_HIP_CHECK(hipGetLastError());
}

void idx_key_rows_hip(const float* src, const float* k_norm, uint8_t* out, float* out_scale,
                      uint64_t out_stride, uint32_t slot0, uint32_t n_out, uint32_t head_dim,
                      uint32_t ratio, uint32_t n_rot, RopeDerived rope, float rms_eps,
                      void* stream) {
  if (!n_out || !ratio) return;
  // Same 128-wide contract as mode 1, for the same reason — the scoring kernel is compiled for
  // kIdxDim and a key row of any other width is not a key row.
  if (head_dim != 128u) {
    aff::ui::fatal("fatal: the V4.1 index key is compiled for a 128-wide head and was given %u.\n",
                 head_dim);
    std::abort();
  }
  const size_t lds = ((size_t)head_dim + head_dim / 32u + 1u) * sizeof(float);
  hipLaunchKernelGGL(idx_key_kernel, dim3(n_out), dim3(head_dim), lds, (hipStream_t)stream, src,
                     k_norm, out, out_scale, out_stride, slot0, head_dim, ratio, n_rot,
                     rope.freq_scale, rope.theta_scale, rope.ext_factor, rope.lo, rope.hi, rms_eps);
  AFF_HIP_CHECK(hipGetLastError());
}


void compressor_hist2_hip(const float* kv_a, const float* sc_a, float* hkv_a, float* hsc_a,
                          uint32_t width_a, const float* kv_b, const float* sc_b, float* hkv_b,
                          float* hsc_b, uint32_t width_b, uint32_t n, uint32_t pos0, uint32_t hist,
                          void* stream) {
  if (!n || !kv_a) return;
  const uint32_t take = n < hist ? n : hist;
  const HistJob a{kv_a, sc_a, hkv_a, hsc_a, width_a};
  const HistJob b{kv_b, sc_b, hkv_b, hsc_b, width_b};
  hipLaunchKernelGGL(hist2_kernel, dim3(take, kv_b ? 2u : 1u), dim3(256), 0, (hipStream_t)stream,
                     a, b, n, pos0, take, hist);
  AFF_HIP_CHECK(hipGetLastError());
}

void compressor_hist_reset_hip(float* hist_kv, float* hist_sc, uint32_t width, uint32_t hist,
                               void* stream) {
  hipLaunchKernelGGL(hist_reset_kernel, dim3(hist), dim3(256), 0, (hipStream_t)stream, hist_kv,
                     hist_sc, width);
  AFF_HIP_CHECK(hipGetLastError());
}

}  // namespace aff
