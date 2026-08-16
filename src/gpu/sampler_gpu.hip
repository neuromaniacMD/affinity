#include "gpu/sampler_gpu.h"
#include "gpu/hip_common.h"

#include <cmath>

namespace aff {

namespace {

constexpr uint32_t kThr = 256;

// A monotone map from float to uint32, so a radix pass on the bits orders the values. The same trick
// the indexer's top-k uses; written for the general case even though probabilities are non-negative,
// so it cannot surprise anyone who reuses it.
__device__ __forceinline__ uint32_t mono(float v) {
  const uint32_t u = __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Everything the block shares, declared once. Shared memory inside a nested scope is a portability
// question nobody should have to ask while reading a sampler.
struct Shared {
  float    red[kThr];       // the general-purpose reduction buffer
  uint32_t red_i[kThr];     // ...and the ids that ride along with it
  float    mass[256];       // radix buckets by probability mass (top_p)
  uint32_t cnt[256];        // ...and by count (top_k)
  uint32_t prefix;          // the threshold key being narrowed, high byte first
  uint32_t slice;           // which thread's slice an inverse-CDF draw landed in
  float    slice_base;      // the cumulative mass before that slice
  uint32_t pick;
  float    keep_total;
  float    excl_total;
  float    p_query;
};

__device__ __forceinline__ float reduce_max(Shared& sh, float v) {
  sh.red[threadIdx.x] = v;
  __syncthreads();
  for (uint32_t k = kThr / 2; k; k >>= 1) {
    if (threadIdx.x < k) sh.red[threadIdx.x] = fmaxf(sh.red[threadIdx.x], sh.red[threadIdx.x + k]);
    __syncthreads();
  }
  const float r = sh.red[0];
  __syncthreads();
  return r;
}

__device__ __forceinline__ float reduce_sum(Shared& sh, float v) {
  sh.red[threadIdx.x] = v;
  __syncthreads();
  for (uint32_t k = kThr / 2; k; k >>= 1) {
    if (threadIdx.x < k) sh.red[threadIdx.x] += sh.red[threadIdx.x + k];
    __syncthreads();
  }
  const float r = sh.red[0];
  __syncthreads();
  return r;
}

// ---- the kernel ---------------------------------------------------------------------------------
//
// One block owns one position and walks the vocabulary several times. Every pass reads the same
// plane, so after the first it is L2-resident; at this vocabulary that beats any staging.
//
// The order is vLLM's — temperature, softmax, top_k, top_p, min_p, renormalise, sample — and it is
// not a detail: reordering changes which tokens survive.
__global__ __launch_bounds__(kThr) void sample_kernel(
    const float* __restrict__ logits, uint32_t vocab, uint32_t row_stride, uint32_t col_stride,
    const uint32_t* __restrict__ query, const float* __restrict__ u, float temperature,
    float top_p, uint32_t top_k, float min_p, SpecDraw* __restrict__ out) {
  __shared__ Shared sh;

  const uint32_t pos = blockIdx.x, tid = threadIdx.x;
  const float* const col = logits + (size_t)pos * col_stride;
  const uint32_t q = query ? query[pos] : 0xFFFFFFFFu;

  auto lg = [&](uint32_t v) { return col[(size_t)v * row_stride]; };

  const float m = reduce_max(sh, [&] {
    float best = -INFINITY;
    for (uint32_t v = tid; v < vocab; v += kThr) best = fmaxf(best, lg(v));
    return best;
  }());

  // ---- temperature 0 is greedy, and stays a real path ------------------------------------------
  //
  // Kept here rather than special-cased by the caller so that a request asking for determinism gets
  // it from the same code, with the SAME tie-break the argmax kernel uses (lowest id wins) — the two
  // must agree or a temperature-0 request stops matching a greedy run.
  if (!(temperature > 0.0f)) {
    float bv = -INFINITY;
    uint32_t bi = 0xFFFFFFFFu;
    for (uint32_t v = tid; v < vocab; v += kThr) {
      const float x = lg(v);
      if (x > bv || (x == bv && v < bi)) { bv = x; bi = v; }
    }
    sh.red[tid] = bv;
    sh.red_i[tid] = bi;
    __syncthreads();
    for (uint32_t k = kThr / 2; k; k >>= 1) {
      if (tid < k) {
        const float ov = sh.red[tid + k];
        const uint32_t oi = sh.red_i[tid + k];
        if (ov > sh.red[tid] || (ov == sh.red[tid] && oi < sh.red_i[tid])) {
          sh.red[tid] = ov; sh.red_i[tid] = oi;
        }
      }
      __syncthreads();
    }
    if (tid == 0) {
      out[pos].tok = sh.red_i[0];
      out[pos].tok_excl = sh.red_i[0];
      out[pos].p_query = (q == sh.red_i[0]) ? 1.0f : 0.0f;
    }
    return;
  }

  const float inv_t = 1.0f / temperature;
  float zs = 0.0f;
  for (uint32_t v = tid; v < vocab; v += kThr) zs += __expf((lg(v) - m) * inv_t);
  const float z = reduce_sum(sh, zs);
  const float inv_z = z > 0.0f ? 1.0f / z : 0.0f;
  auto prob = [&](uint32_t v) { return __expf((lg(v) - m) * inv_t) * inv_z; };

  // ---- top_k, as an exact threshold on the monotone key ----------------------------------------
  //
  // Four radix passes narrowing one byte at a time, counting tokens. That is the k-th largest
  // probability exactly, without sorting the vocabulary.
  uint32_t k_key = 0u;
  if (top_k > 0 && top_k < vocab) {
    if (tid == 0) sh.prefix = 0u;
    __syncthreads();
    uint32_t remaining = top_k;
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (tid < 256) sh.cnt[tid] = 0u;
      __syncthreads();
      const uint32_t mask = (shift == 24) ? 0u : (0xFFFFFFFFu << (shift + 8));
      for (uint32_t v = tid; v < vocab; v += kThr) {
        const uint32_t key = mono(prob(v));
        if ((key & mask) == (sh.prefix & mask)) atomicAdd(&sh.cnt[(key >> shift) & 0xFFu], 1u);
      }
      __syncthreads();
      if (tid == 0) {
        uint32_t acc = 0u, b = 0u;
        for (int i = 255; i >= 0; --i) {
          if (acc + sh.cnt[i] >= remaining) { b = (uint32_t)i; break; }
          acc += sh.cnt[i];
        }
        remaining -= acc;
        sh.prefix |= b << shift;
      }
      __syncthreads();
    }
    k_key = sh.prefix;
  }

  // ---- top_p, the same passes accumulating MASS ------------------------------------------------
  //
  // Descend the buckets summing probability until the nucleus target is reached; the bucket that
  // crosses it holds the threshold and the next pass narrows inside it. Ties AT the threshold are
  // all kept, which can overshoot top_p by the mass of the tie — the alternative is inventing an
  // order among equal probabilities.
  uint32_t p_key = 0u;
  if (top_p < 1.0f) {
    if (tid == 0) sh.prefix = 0u;
    __syncthreads();
    float target = top_p;
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (tid < 256) sh.mass[tid] = 0.0f;
      __syncthreads();
      const uint32_t mask = (shift == 24) ? 0u : (0xFFFFFFFFu << (shift + 8));
      for (uint32_t v = tid; v < vocab; v += kThr) {
        const float p = prob(v);
        const uint32_t key = mono(p);
        if ((key & mask) == (sh.prefix & mask) && (k_key == 0u || key >= k_key))
          atomicAdd(&sh.mass[(key >> shift) & 0xFFu], p);
      }
      __syncthreads();
      if (tid == 0) {
        float acc = 0.0f;
        uint32_t b = 0u;
        for (int i = 255; i >= 0; --i) {
          if (acc + sh.mass[i] >= target) { b = (uint32_t)i; break; }
          acc += sh.mass[i];
        }
        target -= acc;
        sh.prefix |= b << shift;
      }
      __syncthreads();
    }
    p_key = sh.prefix;
  }

  // min_p is a floor relative to the mode, and the mode's probability is exp(0)/Z.
  const float floor_p = min_p > 0.0f ? min_p * inv_z : 0.0f;

  auto keep = [&](float p) {
    const uint32_t key = mono(p);
    if (k_key && key < k_key) return false;
    if (p_key && key < p_key) return false;
    return !(p < floor_p);
  };

  // ---- the surviving mass, with and without the proposal ----------------------------------------
  float kt = 0.0f, ke = 0.0f, pq = 0.0f;
  for (uint32_t v = tid; v < vocab; v += kThr) {
    const float p = prob(v);
    if (!keep(p)) continue;
    kt += p;
    if (v == q) pq += p; else ke += p;
  }
  const float keep_total = reduce_sum(sh, kt);
  const float excl_total = reduce_sum(sh, ke);
  const float p_query = reduce_sum(sh, pq);
  if (tid == 0) {
    sh.keep_total = keep_total;
    sh.excl_total = excl_total;
    sh.p_query = p_query;
  }
  __syncthreads();

  // ---- inverse CDF, in two stages ---------------------------------------------------------------
  //
  // CONTIGUOUS slices, not the strided ones every other loop here uses, and that is the whole
  // correctness argument. An inverse CDF walks the distribution in some order, and any fixed order
  // samples correctly — but only ASCENDING TOKEN ID makes the answer a property of the uniform
  // rather than of the block size. Strided slices put token 256 second; a host reference, another
  // implementation, or this kernel at a different kThr would all disagree about which token a given
  // draw selects. Contiguous slices make the per-thread prefix the id prefix, so they agree.
  //
  // Each thread sums its own range, thread 0 finds which range the draw lands in, that thread
  // rescans it. Exact, and no sort. Every thread reaches both calls, so the barriers are uniform.
  const uint32_t chunk = (vocab + kThr - 1) / kThr;
  const uint32_t lo = tid * chunk;
  const uint32_t hi = lo + chunk < vocab ? lo + chunk : vocab;

  auto draw = [&](float uu, float total, uint32_t excl) -> uint32_t {
    if (!(total > 0.0f)) return 0u;
    const float want = uu * total;
    float mine = 0.0f;
    for (uint32_t v = lo; v < hi; ++v) {
      const float p = prob(v);
      if (!keep(p) || v == excl) continue;
      mine += p;
    }
    sh.red[tid] = mine;
    __syncthreads();
    if (tid == 0) {
      float acc = 0.0f;
      uint32_t who = kThr - 1;
      for (uint32_t k = 0; k < kThr; ++k) {
        if (acc + sh.red[k] >= want) { who = k; break; }
        acc += sh.red[k];
      }
      sh.slice = who;
      sh.slice_base = acc;
      sh.pick = 0xFFFFFFFFu;
    }
    __syncthreads();
    if (tid == sh.slice) {
      float acc = sh.slice_base;
      uint32_t last = 0xFFFFFFFFu;
      for (uint32_t v = lo; v < hi; ++v) {
        const float p = prob(v);
        if (!keep(p) || v == excl) continue;
        last = v;
        acc += p;
        if (acc >= want) { sh.pick = v; break; }
      }
      // Float drift can leave the walk a hair short at the very top of the range; the last kept
      // token in this range is the right answer there, not token 0.
      if (sh.pick == 0xFFFFFFFFu) sh.pick = last;
    }
    __syncthreads();
    const uint32_t r = sh.pick;
    __syncthreads();
    return r;
  };

  const uint32_t t1 = draw(u[2 * pos + 0], sh.keep_total, 0xFFFFFFFFu);
  const uint32_t t2 = draw(u[2 * pos + 1], sh.excl_total, q);

  if (tid == 0) {
    out[pos].tok = t1;
    out[pos].tok_excl = t2;
    out[pos].p_query = sh.keep_total > 0.0f ? sh.p_query / sh.keep_total : 0.0f;
  }
}

}  // namespace

bool batch_sample_hip(const float* logits, uint32_t vocab, uint32_t n,
                      uint32_t row_stride, uint32_t col_stride,
                      const uint32_t* d_query, const float* d_u, SampleParams p,
                      SpecDraw* d_out, void* stream) {
  if (!logits || !d_out || !d_u || !n || !vocab) return false;
  hipLaunchKernelGGL(sample_kernel, dim3(n), dim3(kThr), 0, (hipStream_t)stream, logits, vocab,
                     row_stride, col_stride, d_query, d_u, p.temperature, p.top_p, p.top_k,
                     p.min_p, d_out);
  return hipGetLastError() == hipSuccess;
}

}  // namespace aff
