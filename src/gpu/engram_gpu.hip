#include "gpu/engram_gpu.h"
#include "gpu/hip_common.h"
#include "gpu/wmma.h"

#include <hip/hip_runtime.h>

namespace aff {
namespace {

// The project's own bf16 conversion, not the HIP intrinsic: `b_hc` is written by batch_hc_seed and
// by hc_post through `wmma::f32_to_bf16`, which rounds to nearest even, and a second writer that
// truncated instead would bias this one contribution toward zero against every other.
__device__ __forceinline__ float ld_hc(const void* p, size_t i, bool bf16) {
  return bf16 ? wmma::bf16_to_f32(((const uint16_t*)p)[i]) : ((const float*)p)[i];
}
__device__ __forceinline__ void st_hc(void* p, size_t i, float v, bool bf16) {
  if (bf16) ((uint16_t*)p)[i] = wmma::f32_to_bf16(v);
  else      ((float*)p)[i] = v;
}

// One block a (hc copy, token tile); one thread a token. BF16 is a template parameter so the branch
// is resolved at compile time rather than inside two loops over `dim`.
template <bool BF16>
__global__ void engram_gate_kernel(void* __restrict__ hc, const float* __restrict__ kv,
                                   const float* __restrict__ w, uint32_t n_embd, uint32_t n_hc,
                                   uint32_t nb, uint32_t hs, float eps) {
  const uint32_t c = blockIdx.x;
  const uint32_t t = blockIdx.y * blockDim.x + threadIdx.x;
  if (t >= nb || c >= n_hc) return;

  const size_t band = (size_t)c * n_embd;
  const size_t vband = (size_t)n_hc * n_embd;          // the shared value's band in `kv`

  // Pass one: the two RMS terms and the weighted dot, in f32 whatever the state is stored as —
  // the reference does this in float and casts only on the way back.
  float sh = 0.0f, sk = 0.0f, dot = 0.0f;
  for (uint32_t i = 0; i < n_embd; ++i) {
    const float h = ld_hc(hc, (band + i) * hs + t, BF16);
    const float k = kv[(band + i) * nb + t];
    sh = fmaf(h, h, sh);
    sk = fmaf(k, k, sk);
    dot = fmaf(h * w[band + i], k, dot);
  }
  const float rstd = rsqrtf(sh / (float)n_embd + eps) * rsqrtf(sk / (float)n_embd + eps);
  const float d = dot * rstd * rsqrtf((float)n_embd);
  // Signed sqrt before the sigmoid, matching the training kernel: the magnitude is clamped away
  // from zero first, then the sign is put back.
  const float s = copysignf(sqrtf(fmaxf(fabsf(d), 1e-6f)), d);
  const float gate = 1.0f / (1.0f + __expf(-s));

  // Pass two: the write. ONE value for every copy, each scaled by its own gate.
  for (uint32_t i = 0; i < n_embd; ++i) {
    const size_t off = (band + i) * hs + t;
    st_hc(hc, off, ld_hc(hc, off, BF16) + gate * kv[(vband + i) * nb + t], BF16);
  }
}

}  // namespace

void engram_gate_hip(void* hc, bool hc_bf16, const float* kv, const float* w, uint32_t n_embd,
                     uint32_t n_hc, uint32_t nb, uint32_t hc_stride, float eps, void* stream) {
  if (!hc || !kv || !w || !n_embd || !n_hc || !nb || hc_stride < nb) return;
  const uint32_t thr = nb < 256u ? nb : 256u;
  const dim3 g(n_hc, (nb + thr - 1u) / thr), b(thr);
  if (hc_bf16)
    hipLaunchKernelGGL(engram_gate_kernel<true>, g, b, 0, (hipStream_t)stream, hc, kv, w, n_embd,
                       n_hc, nb, hc_stride, eps);
  else
    hipLaunchKernelGGL(engram_gate_kernel<false>, g, b, 0, (hipStream_t)stream, hc, kv, w, n_embd,
                       n_hc, nb, hc_stride, eps);
  AFF_HIP_CHECK(hipGetLastError());
}

}  // namespace aff
