#include "gpu/digest.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "gpu/hip_common.h"

#include "ui/log.h"
namespace aff {
namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaxDev = 8;

// FNV-1a's constants over 64 bits, plus a final avalanche. It has to be strong enough that one
// flipped float cannot cancel against another element's contribution — a weak hash here reads as
// "the arms agree", which is the one failure mode this instrument must not have.
__device__ inline uint64_t mixv(uint64_t h, uint64_t v) {
  h ^= v;
  h *= 1099511628211ull;
  return h ^ (h >> 29);
}

// `stamp` is the publish: the hash lands, a system fence drains it, and only then does the sequence
// number appear, so a host that sees `mark` is guaranteed to see the hash that goes with it.
__global__ __launch_bounds__(kThreads) void digest_kernel(const uint8_t* __restrict__ p, size_t n,
                                                          uint64_t* __restrict__ out,
                                                          uint32_t* __restrict__ stamp,
                                                          uint32_t mark) {
  __shared__ uint64_t red[kThreads];
  uint64_t h = 0;
  const size_t nw = n / 4;
  const uint32_t* w = (const uint32_t*)p;
  for (size_t i = threadIdx.x; i < nw; i += kThreads)
    h ^= mixv((uint64_t)i * 0x9e3779b97f4a7c15ull, (uint64_t)w[i]);
  for (size_t i = nw * 4 + threadIdx.x; i < n; i += kThreads)
    h ^= mixv((uint64_t)i * 0x9e3779b97f4a7c15ull, (uint64_t)p[i]);
  red[threadIdx.x] = h;
  __syncthreads();
  for (uint32_t s = kThreads / 2; s; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] ^= red[threadIdx.x + s];
    __syncthreads();
  }
  if (!threadIdx.x) {
    *out = red[0] ? red[0] : 1ull;                   // 0 stays "never written"
    if (stamp) {
      __threadfence_system();
      __hip_atomic_store(stamp, mark, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
  }
}

// ---- the cross-CONFIGURATION probe ------------------------------------------------------------
//
// The hash above answers "do the ranks agree", which is the shipping question. It cannot answer "do
// TWO RANK COUNTS agree", because they never do bitwise: a different number of partials sums in a
// different grouping and the last bits move on the first all-reduce. Comparing hashes across rank
// counts reports a difference at every layer and localises nothing.
//
// So this reduces the same buffer to a magnitude instead. Rounding leaves it equal to five or six
// figures; a sublayer that has gone wrong moves it grossly. Diff two runs' traces and the first
// line that parts is the sublayer to look at.
__global__ __launch_bounds__(kThreads) void l2_kernel(const float* __restrict__ p, size_t n,
                                                      double* __restrict__ out) {
  __shared__ double red[kThreads];
  double a = 0;
  for (size_t i = threadIdx.x; i < n; i += kThreads) a += (double)p[i] * (double)p[i];
  red[threadIdx.x] = a;
  __syncthreads();
  for (uint32_t s = kThreads / 2; s; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (!threadIdx.x) *out = red[0];
}

} // namespace

// AFF_NORM_TRACE=1 prints one line a visit. DIAGNOSTIC ONLY: it synchronises the stream to read the
// value back, so the engine it measures is not the engine that ships.
void xrank_trace(uint32_t rank, const char* tag, const void* p, size_t bytes, void* stream) {
  static const bool on = [] {
    const char* v = std::getenv("AFF_NORM_TRACE");
    return v && *v && *v != '0';
  }();
  if (!on || !p || !bytes) return;
  static double* h = nullptr;
  static double* d = nullptr;
  if (!h && (hipHostMalloc((void**)&h, sizeof(double), hipHostMallocMapped) != hipSuccess ||
             hipHostGetDevicePointer((void**)&d, h, 0) != hipSuccess))
    return;
  static uint64_t seq[kMaxDev] = {};
  if (rank >= kMaxDev) return;
  hipLaunchKernelGGL(l2_kernel, dim3(1), dim3(kThreads), 0, (hipStream_t)stream,
                     (const float*)p, bytes / sizeof(float), d);
  (void)hipStreamSynchronize((hipStream_t)stream);
  aff::ui::err("norm-trace r%u %-4s %6llu %.9e\n", rank, tag, (unsigned long long)seq[rank]++, *h);
}

// ---- the permanent cross-rank check -----------------------------------------------------------
namespace {

constexpr uint32_t kXrSlots = 8;      // samples in flight; the host is never more than a few behind
constexpr uint32_t kXrEvery = 512;    // calls between samples, per rank

struct XrState {
  uint64_t* hash = nullptr;           // [kMaxDev][kXrSlots], pinned and mapped
  uint32_t* stamp = nullptr;          // [kMaxDev][kXrSlots], 0 = never written
  uint64_t* hash_dev = nullptr;       // the same memory as the cards address it
  uint32_t* stamp_dev = nullptr;
  uint64_t  calls[kMaxDev] = {};
  uint32_t  done[kXrSlots] = {};      // the sample ordinal already compared in this slot
  uint64_t  compared = 0;
  // The positive control. A zero mismatch count means nothing if the counter cannot be made
  // non-zero: AFF_XRANK_INJECT=<n> perturbs the last rank's hash at sample n, so a build that
  // reports "all equal" can be made to report a mismatch and abort.
  uint32_t  inject = 0;
  bool      init = false, off = false;
};

XrState& xr() {
  static XrState s = [] {
    XrState t;
    const char* v = std::getenv("AFF_XRANK_INJECT");
    t.inject = (v && *v) ? (uint32_t)std::strtoul(v, nullptr, 10) : 0u;
    return t;
  }();
  return s;
}

}  // namespace

uint64_t xrank_compared() { return xr().compared; }

void xrank_check(uint32_t rank, uint32_t ranks, const void* p, size_t bytes, void* stream) {
  XrState& s = xr();
  if (s.off || ranks < 2 || rank >= kMaxDev || !p || !bytes) return;
  if (!s.init) {
    s.init = true;
    // Mapped host memory, exactly as tp_allreduce's timeout word is: one allocation both cards can
    // store into and the host reads with a plain load. A device buffer would need a copy and a
    // stream to order it on, which is the thing this must not have.
    if (hipHostMalloc((void**)&s.hash, (size_t)kMaxDev * kXrSlots * sizeof(uint64_t),
                      hipHostMallocMapped) != hipSuccess ||
        hipHostMalloc((void**)&s.stamp, (size_t)kMaxDev * kXrSlots * sizeof(uint32_t),
                      hipHostMallocMapped) != hipSuccess ||
        hipHostGetDevicePointer((void**)&s.hash_dev, s.hash, 0) != hipSuccess ||
        hipHostGetDevicePointer((void**)&s.stamp_dev, s.stamp, 0) != hipSuccess) {
      // Loud, and then off. A silent failure here would read for months as "the cards have never
      // disagreed", which is the exact shape of the bug this exists to catch.
      aff::ui::err("cross-rank check off: its pinned slots could not be allocated\n");
      s.off = true;
      return;
    }
    for (uint32_t i = 0; i < kMaxDev * kXrSlots; ++i) { s.hash[i] = 0; s.stamp[i] = 0; }
  }
  const uint64_t c = s.calls[rank]++;
  if (c % kXrEvery) return;
  const uint32_t ord = (uint32_t)(c / kXrEvery) + 1;   // 1-based, so 0 stays "never written"
  const size_t at = (size_t)rank * kXrSlots + (ord - 1) % kXrSlots;
  hipLaunchKernelGGL(digest_kernel, dim3(1), dim3(kThreads), 0, (hipStream_t)stream,
                     (const uint8_t*)p, bytes, s.hash_dev + at, s.stamp_dev + at, ord);

  // Once per visit, after every card has issued its launch. Nothing here waits: a slot whose ranks
  // have not all landed is skipped and picked up on a later visit, and a slot that gets reused
  // before the host looked at it is a sample lost, which a sampler is allowed to do.
  if (rank + 1 != ranks) return;
  for (uint32_t i = 0; i < kXrSlots; ++i) {
    const uint32_t m = __atomic_load_n(&s.stamp[i], __ATOMIC_ACQUIRE);
    if (!m || m == s.done[i]) continue;
    bool landed = true;
    for (uint32_t r = 1; r < ranks && landed; ++r)
      landed = __atomic_load_n(&s.stamp[(size_t)r * kXrSlots + i], __ATOMIC_ACQUIRE) == m;
    if (!landed) continue;
    s.done[i] = m;
    ++s.compared;
    if (s.inject == m) s.hash[(size_t)(ranks - 1) * kXrSlots + i] ^= 1ull;
    for (uint32_t r = 1; r < ranks; ++r)
      if (s.hash[(size_t)r * kXrSlots + i] != s.hash[i]) {
        aff::ui::fatal("fatal: the cards disagree about a replicated buffer at sample %u — "
                     "card 0 %016llx, card %u %016llx.\n"
                     "       Everything downstream inherits it, including the KV rows each card "
                     "writes, so the two cards are decoding different sequences.\n",
                     m, (unsigned long long)s.hash[i], r,
                     (unsigned long long)s.hash[(size_t)r * kXrSlots + i]);
        std::abort();
      }
  }
}

} // namespace aff
