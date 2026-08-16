// N-rank all-reduce for gfx12 cards, adapted from vllm-radiance's radiance_ar_ext.
//
// The original is a PyTorch extension for TP=2 across two PROCESSES, so it trades IPC handles
// (hipIpcGetMemHandle / open_shared) to give each rank a pointer into the other's scratch. This
// engine drives every card from ONE process, so that whole layer disappears: with
// hipDeviceEnablePeerAccess, a kernel on device 0 dereferences a device-1 allocation directly.
//
// What carries over unchanged, because it is what makes the crossing fast on this hardware:
//
//   * ONE-SHOT PUSH, not a ring. Each rank writes its input into every peer's scratch, then reduces
//     (my input + the peers' data now in my scratch) into out. One crossing per peer, not two.
//   * FINE-GRAINED allocation (hipExtMallocWithFlags + hipDeviceMallocFinegrained). Uncached, so
//     stores post straight to the fabric instead of sitting in L2 waiting for a writeback.
//   * `s_wait_storecnt 0x0` as the drain, NOT __threadfence_system. A threadfence emits an L2
//     writeback-invalidate per wavefront, which is pure waste on memory that was never cached and
//     floods the fabric at high occupancy. The wait drains posted writes with no invalidate, and
//     PCIe keeps posted writes to the same peer ordered, so the release flag lands after the data.
//   * PER-BLOCK flags and no grid barrier. Block b owns a contiguous chunk, pushes it, sets each
//     peer's flag for b, spins on its own flags for b, reduces its own chunk. Blocks are
//     independent.
//   * Scratch DOUBLE-BUFFERED by sequence parity, so a slot cannot be reused while a peer is
//     still draining the previous one.
//
// The sequence counter is device-resident and incremented by the kernel. In vllm that was for
// cudagraph replay safety; here it keeps the ranks in lockstep across calls without the host having
// to pass a matching number to all of them.

#include "gpu/tp_allreduce.h"
#include "gpu/dense_gpu.h"        // kMaxRanks, asserted equal to kTpMaxRanks in tp_init
#include "gpu/hip_common.h"

#include "ui/log.h"
#include <hip/hip_fp8.h>
#include <algorithm>

#include <cstdio>
#include <cstdlib>

namespace aff {

namespace {

// System scope: the reader is the CPU, and a device-scope store would leave it in a cache the host
// cannot see. `atomicMax` rather than a plain store so the FIRST failing sequence survives a later
// one — the interesting number is where it started, not where it was noticed.
__device__ inline void note_timeout(unsigned int* t, unsigned int s) {
  if (!t) return;
  __hip_atomic_fetch_max(t, s ? s : 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Two hipSetDevice calls a reduce, unconditionally. Caching what this translation unit last set is
// wrong: the engine changes the current device between entry points, so the fix if this ever shows
// up in a profile is for the caller to state which device is current, not to guess again.
inline void set_device(int d) { (void)hipSetDevice(d); }

__device__ inline void store_sys_rel(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ inline unsigned int load_sys_acq(const unsigned int* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// The peers a rank pushes to, resolved on the host so the kernel does no index arithmetic over rank
// ids. Both arrays are already offset to THIS rank's sub-slot in that peer's memory — see
// tp_peer_slot — so the kernel only ever adds the double-buffer slot.
template <int NPEER>
struct TpPush {
  float4*       scratch[NPEER];
  unsigned int* flags[NPEER];
};

__device__ __forceinline__ float4 f4add(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

// NPEER is a template argument, not a runtime one, so that at two ranks — the configuration this
// engine is tuned on — the push and the reduce collapse to exactly the single-peer straight line
// they were before the generalisation, with no loop and no indirection through the peer table.
template <int NPEER>
__global__ void tp_ar_kernel(const float4* __restrict__ in,
                             TpPush<NPEER> push, const float4* my_scratch,
                             float4* __restrict__ out,
                             unsigned int* my_flags,
                             unsigned int* seq_ctrs, unsigned int* timeout,
                             int n4, int slot_stride4, int peer_stride4, int own) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;

  __shared__ unsigned int s_seq;
  if (tid == 0) s_seq = atomicAdd(&seq_ctrs[b], 1u) + 1u;
  __syncthreads();
  const unsigned int s = s_seq;
  const int slot = (int)(s % TpContext::kTpSlots);

  float4* ps[NPEER];
  const float4* ms[NPEER];
#pragma unroll
  for (int j = 0; j < NPEER; ++j) {
    ps[j] = push.scratch[j] + (size_t)slot * slot_stride4;
    ms[j] = my_scratch + (size_t)slot * slot_stride4 + (size_t)j * peer_stride4;
  }

  const int w = (n4 + nb - 1) / nb;
  const int start = b * w;
  const int end = (start + w < n4) ? (start + w) : n4;

  // Push my chunk into every peer's scratch, unrolled for memory-level parallelism: PCIe wants many
  // writes in flight, and one per thread per iteration does not saturate it. `in` is read once and
  // fanned out from registers — the crossings are what this costs, not the local reads.
  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      const float4 v0 = in[k], v1 = in[k + nt], v2 = in[k + 2 * nt], v3 = in[k + 3 * nt];
#pragma unroll
      for (int j = 0; j < NPEER; ++j) {
        ps[j][k] = v0; ps[j][k + nt] = v1; ps[j][k + 2 * nt] = v2; ps[j][k + 3 * nt] = v3;
      }
    }
    for (; k < end; k += nt) {
      const float4 v = in[k];
#pragma unroll
      for (int j = 0; j < NPEER; ++j) ps[j][k] = v;
    }
  }

  // Drain posted writes without an L2 invalidate — see the header comment.
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();

  if (tid == 0) {
#pragma unroll
    for (int j = 0; j < NPEER; ++j) store_sys_rel(&push.flags[j][b], s);
    unsigned long long z = 0;
#pragma unroll
    for (int j = 0; j < NPEER; ++j) {
      while (load_sys_acq(&my_flags[j * kTpMaxBlocks + b]) < s) {
        if (++z > kTpSpinMax) { note_timeout(timeout, s); break; }
      }
    }
  }
  __syncthreads();

  // out = the sum of every rank's input, mine from `in` and the rest from my scratch.
  //
  // THE ORDER IS BY RANK, NOT "MINE FIRST". Float addition is commutative but not associative, so
  // summing my own contribution first and the peers after it gives rank 1 `x1+x0+x2` where rank 0
  // gets `x0+x1+x2` — the same value to within an ulp, and a DIFFERENT one in bits. An all-reduce
  // leaves a replicated tensor; two cards holding numbers that differ in the last bit are two cards
  // running slightly different models, and everything downstream of it compounds that. At two ranks
  // the sum has two terms and commutativity alone settles it, which is why the ordering only starts
  // to matter at three.
  {
    int k = start + tid;
    if constexpr (NPEER == 1) {
      // The two-rank straight line, kept verbatim: one peer, no source table, no select.
      for (; k + 3 * nt < end; k += 4 * nt) {
        const float4 a0 = in[k], a1 = in[k + nt], a2 = in[k + 2 * nt], a3 = in[k + 3 * nt];
        const float4 b0 = ms[0][k], b1 = ms[0][k + nt];
        const float4 b2 = ms[0][k + 2 * nt], b3 = ms[0][k + 3 * nt];
        out[k]          = f4add(a0, b0); out[k + nt]     = f4add(a1, b1);
        out[k + 2 * nt] = f4add(a2, b2); out[k + 3 * nt] = f4add(a3, b3);
      }
      for (; k < end; k += nt) out[k] = f4add(in[k], ms[0][k]);
    } else {
      // src[p] is where rank p's contribution lives as this rank sees it: `in` for my own, the
      // matching sub-slot for everyone else. Summing over p in order is then the same expression on
      // every rank by construction rather than by an argument about which slot holds what.
      const float4* src[NPEER + 1];
#pragma unroll
      for (int p = 0; p <= NPEER; ++p) src[p] = (p == own) ? in : ms[p < own ? p : p - 1];
      for (; k + 3 * nt < end; k += 4 * nt) {
        float4 a0 = src[0][k], a1 = src[0][k + nt];
        float4 a2 = src[0][k + 2 * nt], a3 = src[0][k + 3 * nt];
#pragma unroll
        for (int p = 1; p <= NPEER; ++p) {
          a0 = f4add(a0, src[p][k]);          a1 = f4add(a1, src[p][k + nt]);
          a2 = f4add(a2, src[p][k + 2 * nt]); a3 = f4add(a3, src[p][k + 3 * nt]);
        }
        out[k] = a0; out[k + nt] = a1; out[k + 2 * nt] = a2; out[k + 3 * nt] = a3;
      }
      for (; k < end; k += nt) {
        float4 a = src[0][k];
#pragma unroll
        for (int p = 1; p <= NPEER; ++p) a = f4add(a, src[p][k]);
        out[k] = a;
      }
    }
  }
}

// FP8 variant of the push. Only the CROSSING is quantised, and both ranks reduce against the
// quantised form of BOTH halves so the two cards end on the same bits — see the reduce loop.
//
// Scratch layout per slot: [ e4m3 bytes | ... | f32 scales ], scales at a fixed offset so both
// ranks agree without exchanging metadata.
__device__ __forceinline__ unsigned char f2e4m3(float v) { __hip_fp8_e4m3 q(v); return q.__x; }
__device__ __forceinline__ float e4m3_2f(unsigned char b) {
  __hip_fp8_e4m3 q; q.__x = b; return (float)q;
}

template <int NPEER>
struct TpPush8 {
  unsigned char* base[NPEER];
  unsigned int*  flags[NPEER];
};

template <int NPEER>
__global__ void tp_ar_fp8_kernel(const float* __restrict__ in,
                                 TpPush8<NPEER> push, const unsigned char* my_base,
                                 float* __restrict__ out,
                                 unsigned int* my_flags,
                                 unsigned int* seq_ctrs, unsigned int* timeout,
                                 int n, int slot_stride_bytes, int peer_stride_bytes,
                                 int scale_off_bytes, int own) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq;
  if (tid == 0) s_seq = atomicAdd(&seq_ctrs[b], 1u) + 1u;
  __syncthreads();
  const unsigned int s = s_seq;
  const int slot = (int)(s % TpContext::kTpSlots);

  unsigned char* pb[NPEER];
  float* p_scale[NPEER];
  const unsigned char* mb[NPEER];
  const float* m_scale[NPEER];
#pragma unroll
  for (int j = 0; j < NPEER; ++j) {
    pb[j] = push.base[j] + (size_t)slot * slot_stride_bytes;
    p_scale[j] = (float*)(pb[j] + scale_off_bytes);
    mb[j] = my_base + (size_t)slot * slot_stride_bytes + (size_t)j * peer_stride_bytes;
    m_scale[j] = (const float*)(mb[j] + scale_off_bytes);
  }

  // ONE WAVE per group, not one thread. Giving each thread a whole group and looping it serially
  // over amax, encode and decode leaves a handful of threads doing all the work while the rest of
  // the block idles, which is far slower than not quantising at all. Each lane now owns a few
  // elements and the amax is a wave reduction.
  const int n_groups = (n + kTpFp8Group - 1) / kTpFp8Group;
  const int waves = nt / kWave;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;
  const int gw = (n_groups + nb - 1) / nb;
  const int g0 = b * gw;
  const int g1 = (g0 + gw < n_groups) ? (g0 + gw) : n_groups;
  constexpr int kPerLane = kTpFp8Group / kWave;      // 4

  for (int g = g0 + wave; g < g1; g += waves) {
    const int lo = g * kTpFp8Group;
    float v[kPerLane];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const int idx = lo + lane * kPerLane + i;
      v[i] = (idx < n) ? in[idx] : 0.0f;
      amax = fmaxf(amax, fabsf(v[i]));
    }
    #pragma unroll
    for (int off = kWave / 2; off > 0; off >>= 1)
      amax = fmaxf(amax, __shfl_xor(amax, off, kWave));

    const float scale = (amax > 0.0f) ? amax * (1.0f / 448.0f) : 1.0f;
    const float inv = 1.0f / scale;
    #pragma unroll
    for (int j = 0; j < NPEER; ++j) if (lane == 0) p_scale[j][g] = scale;
    #pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const int idx = lo + lane * kPerLane + i;
      if (idx < n) {
        const unsigned char q = f2e4m3(v[i] * inv);
        #pragma unroll
        for (int j = 0; j < NPEER; ++j) pb[j][idx] = q;
      }
    }
  }

  asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    #pragma unroll
    for (int j = 0; j < NPEER; ++j) store_sys_rel(&push.flags[j][b], s);
    unsigned long long z = 0;
    #pragma unroll
    for (int j = 0; j < NPEER; ++j) {
      while (load_sys_acq(&my_flags[j * kTpMaxBlocks + b]) < s) {
        if (++z > kTpSpinMax) { note_timeout(timeout, s); break; }
      }
    }
  }
  __syncthreads();

  // Reduce against MY OWN QUANTISED value, not my exact one.
  //
  // An all-reduce leaves a REPLICATED tensor: both cards are supposed to hold the same number, and
  // everything downstream assumes it — this card's attention heads read the same hidden state the
  // other card's do, and each card writes its own copy of the KV rows from it. Keeping the local
  // half exact and quantising only what crossed makes each rank's answer closer to the true sum,
  // but it makes the two answers DIFFERENT, and a difference in a replicated tensor is not an
  // approximation, it is two cards running slightly different models.
  //
  // So both ranks sum the same two quantised numbers. The peer already holds Q(mine) — it is what
  // crossed — and this recovers the identical value locally from the identical bytes and scale, so
  // Q(mine) + Q(peer) and Q(peer) + Q(mine) agree bit for bit by commutativity. The crossing does
  // not move; what it costs is a second amax pass over data already in registers.
  for (int g = g0 + wave; g < g1; g += waves) {
    const int lo = g * kTpFp8Group;
    float sc[NPEER];
    #pragma unroll
    for (int j = 0; j < NPEER; ++j) sc[j] = m_scale[j][g];
    float v[kPerLane];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const int idx = lo + lane * kPerLane + i;
      v[i] = (idx < n) ? in[idx] : 0.0f;
      amax = fmaxf(amax, fabsf(v[i]));
    }
    #pragma unroll
    for (int off = kWave / 2; off > 0; off >>= 1)
      amax = fmaxf(amax, __shfl_xor(amax, off, kWave));
    // Recomputed rather than carried across the barrier: a wave owns far more groups than fit in
    // registers, and staging them through the scratch would be uncached traffic to trade for ALU.
    const float mine_scale = (amax > 0.0f) ? amax * (1.0f / 448.0f) : 1.0f;
    const float mine_inv = 1.0f / mine_scale;
    #pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const int idx = lo + lane * kPerLane + i;
      if (idx < n) {
        // Recovered from the identical bytes and scale the peers received, so every rank adds the
        // same numbers — and, from three ranks up, in the same RANK order. See the exact kernel.
        const float mine = e4m3_2f(f2e4m3(v[i] * mine_inv)) * mine_scale;
        if constexpr (NPEER == 1) {
          out[idx] = mine + e4m3_2f(mb[0][idx]) * sc[0];
        } else {
          float acc = 0.0f;
          #pragma unroll
          for (int p = 0; p <= NPEER; ++p) {
            if (p == own) { acc += mine; continue; }
            const int j = p < own ? p : p - 1;
            acc += e4m3_2f(mb[j][idx]) * sc[j];
          }
          out[idx] = acc;
        }
      }
    }
  }
}

} // namespace

// Sub-slots a rank keeps: one per peer, none for itself. Zero at one rank, which is why every
// allocation below is guarded rather than assumed non-empty.
static inline size_t tp_npeer(const TpContext* ctx) { return ctx->ranks ? ctx->ranks - 1 : 0; }

// Floats one double-buffer slot spans on one rank: every peer's message, back to back.
static inline size_t tp_slot_floats(const TpContext* ctx) {
  return tp_npeer(ctx) * ctx->max_floats;
}

bool tp_grow(TpContext* ctx, size_t max_floats, std::string* err) {
  if (!ctx->ready) return false;
  const size_t want = (max_floats + 3) / 4 * 4;
  if (want <= ctx->max_floats) return true;
  const size_t np = tp_npeer(ctx);
  if (!np) { ctx->max_floats = want; return true; }
  for (size_t r = 0; r < ctx->ranks; ++r) {
    if (!hip_try(hipSetDevice(ctx->dev[r]), "hipSetDevice", err)) return false;
    if (!hip_try(hipStreamSynchronize((hipStream_t)ctx->stream[r]), "drain before grow", err))
      return false;
    (void)hipFree(ctx->scratch[r]);
    ctx->scratch[r] = nullptr;
    void* sc = nullptr;
    if (!hip_try(hipExtMallocWithFlags(&sc, TpContext::kTpSlots * np * want * sizeof(float),
                                       hipDeviceMallocFinegrained),
                 "scratch regrow", err))
      return false;
    ctx->scratch[r] = (float*)sc;
  }
  ctx->max_floats = want;
  return true;
}

bool tp_fold_prepare(TpContext* ctx, size_t floats, std::string* err) {
  if (!ctx->ready) { if (err) *err = "tp not ready"; return false; }
  if (!floats) { if (err) *err = "zero-length fold"; return false; }
  // Two ranks only. The kernels that consume these buffers embed the single-peer protocol inside a
  // kernel doing other work, and there is no N-peer form of them — see the header. Refusing here is
  // how the callers learn to take the standalone collective instead, and it is not an error.
  if (ctx->ranks != 2) {
    if (err) *err = "the folded exchange is a two-rank path; this run has " +
                    std::to_string(ctx->ranks);
    return false;
  }
  const size_t want = (floats + 3) / 4 * 4;               // float4 alignment, as tp_init does
  if (want <= ctx->fold_floats) return true;
  for (int r = 0; r < 2; ++r) {
    if (!hip_try(hipSetDevice(ctx->dev[r]), "hipSetDevice", err)) return false;
    if (!hip_try(hipStreamSynchronize((hipStream_t)ctx->stream[r]), "drain before fold alloc", err))
      return false;
    (void)hipFree(ctx->fscratch[r]);
    ctx->fscratch[r] = nullptr;
    void* sc = nullptr;
    // Fine-grained for the same reason the main scratch is: the push has to POST to the fabric
    // rather than rest in this card's L2, where the peer would never see it.
    if (!hip_try(hipExtMallocWithFlags(&sc, TpContext::kTpSlots * want * sizeof(float),
                                       hipDeviceMallocFinegrained),
                 "fold scratch alloc", err))
      return false;
    ctx->fscratch[r] = (float*)sc;
    if (!ctx->fflags[r]) {
      void* fl = nullptr;
      if (!hip_try(hipExtMallocWithFlags(&fl, kTpMaxBlocks * sizeof(unsigned),
                                         hipDeviceMallocFinegrained), "fold flags alloc", err))
        return false;
      ctx->fflags[r] = (unsigned*)fl;
      if (!hip_try(hipMemset(ctx->fflags[r], 0, kTpMaxBlocks * sizeof(unsigned)), "memset fold flags",
                   err))
        return false;
      if (!hip_try(hipMalloc(&ctx->fseq[r], kTpMaxBlocks * sizeof(unsigned)), "fold seq alloc", err))
        return false;
      if (!hip_try(hipMemset(ctx->fseq[r], 0, kTpMaxBlocks * sizeof(unsigned)), "memset fold seq",
                   err))
        return false;
    }
  }
  ctx->fold_floats = want;
  return true;
}

bool tp_init(TpContext* ctx, size_t max_floats, std::string* err, ihipStream_t* const* ext,
             size_t n_ranks, const int* devs_in) {
  static_assert(kTpMaxRanks == kMaxRanks, "the collective and the engine must size ranks alike");
  std::vector<int> owned;
  if (!devs_in || !n_ranks) {
    owned = gfx12_devices();
    if (owned.empty()) { if (err) *err = "no gfx12 device for TP"; return false; }
    n_ranks = owned.size();
    devs_in = owned.data();
  }
  if (n_ranks > kTpMaxRanks) {
    if (err) *err = std::to_string(n_ranks) + " ranks exceeds kTpMaxRanks";
    return false;
  }
  ctx->ranks = n_ranks;
  for (size_t r = 0; r < n_ranks; ++r) ctx->dev[r] = devs_in[r];
  ctx->max_floats = (max_floats + 3) / 4 * 4;          // float4 alignment
  ctx->own_stream = (ext == nullptr);
  const size_t np = tp_npeer(ctx);

  for (size_t r = 0; r < n_ranks; ++r) {
    if (!hip_try(hipSetDevice(ctx->dev[r]), "hipSetDevice", err)) return false;
    // Peer access is what removes the IPC layer: a kernel here can write straight into another
    // card's allocation. Already-enabled is not an error, and neither is a peer that IS this device
    // — under AFF_LOGICAL_RANKS two ranks share a card, and a push between them is a local store
    // that needs no peer mapping at all.
    for (size_t q = 0; q < n_ranks; ++q) {
      if (q == r || ctx->dev[q] == ctx->dev[r]) continue;
      const hipError_t pe = hipDeviceEnablePeerAccess(ctx->dev[q], 0);
      if (pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled) {
        // Not every pair of cards in a machine can reach each other. Say which pair, because the
        // fix is a topology change (a different slot, or HIP_VISIBLE_DEVICES) and not a flag.
        if (err) *err = "hipDeviceEnablePeerAccess from device " + std::to_string(ctx->dev[r]) +
                        " to " + std::to_string(ctx->dev[q]) + ": " + hipGetErrorString(pe);
        return false;
      }
      // Tolerating the error is not the same as clearing it. hipGetLastError is STICKY, so an
      // already-enabled peer link reported here surfaced at the next kernel launch check instead —
      // "peer access is already enabled" as the reason a batch_hc_seed_kernel launch failed, which
      // sends you looking at a kernel that is perfectly fine.
      (void)hipGetLastError();
    }
    if (ext) ctx->stream[r] = ext[r];
    else if (!hip_try(hipStreamCreate((hipStream_t*)&ctx->stream[r]), "hipStreamCreate", err))
      return false;

    // One rank has no peers, so no scratch and no flags: tp_all_reduce is the identity there and
    // never reaches a kernel.
    if (!np) continue;

    // Fine-grained => uncached => stores post to the fabric rather than resting in L2.
    void* sc = nullptr;
    if (!hip_try(hipExtMallocWithFlags(&sc,
                                       TpContext::kTpSlots * np * ctx->max_floats * sizeof(float),
                                       hipDeviceMallocFinegrained), "scratch alloc", err))
      return false;
    ctx->scratch[r] = (float*)sc;

    void* fl = nullptr;
    if (!hip_try(hipExtMallocWithFlags(&fl, np * kTpMaxBlocks * sizeof(unsigned),
                                       hipDeviceMallocFinegrained), "flags alloc", err))
      return false;
    ctx->flags[r] = (unsigned*)fl;
    if (!hip_try(hipMemset(ctx->flags[r], 0, np * kTpMaxBlocks * sizeof(unsigned)), "memset flags",
                 err))
      return false;

    if (!hip_try(hipMalloc(&ctx->seq[r], kTpMaxBlocks * sizeof(unsigned)), "seq alloc", err))
      return false;
    if (!hip_try(hipMemset(ctx->seq[r], 0, kTpMaxBlocks * sizeof(unsigned)), "memset seq", err))
      return false;
  }
  // One word for every rank, in PINNED HOST memory so the host reads it with a load rather than a
  // copy — an instrument for a stall cannot itself need a synchronise. hipHostMalloc gives a pointer
  // every device can store to once peer access is up, which it is by here.
  if (!hip_try(hipHostMalloc((void**)&ctx->timeout, sizeof(unsigned), hipHostMallocMapped),
               "timeout alloc", err))
    return false;
  *ctx->timeout = 0u;
  if (!hip_try(hipHostGetDevicePointer((void**)&ctx->timeout_dev, ctx->timeout, 0),
               "timeout device pointer", err))
    return false;
  ctx->ready = true;
  return true;
}

void tp_check(const TpContext& ctx) {
  if (!ctx.ready || !ctx.timeout) return;
  const unsigned t = *ctx.timeout;
  if (!t) return;
  aff::ui::fatal("fatal: a rank gave up waiting for a peer in the all-reduce at collective %u and "
               "reduced against uninitialised scratch. Every token after that is wrong. This is a "
               "bug in the caller's ordering, not a slow peer: the spin bound is hours.\n", t);
  std::abort();
}

void tp_free(TpContext* ctx) {
  if (!ctx->ready) return;
  for (size_t r = 0; r < ctx->ranks; ++r) {
    (void)hipSetDevice(ctx->dev[r]);
    (void)hipFree(ctx->scratch[r]);
    (void)hipFree(ctx->flags[r]);
    (void)hipFree(ctx->seq[r]);
    ctx->scratch[r] = nullptr;
    ctx->flags[r] = nullptr;
    ctx->seq[r] = nullptr;
    if (r < 2) {
      (void)hipFree(ctx->fscratch[r]);
      (void)hipFree(ctx->fflags[r]);
      (void)hipFree(ctx->fseq[r]);
      ctx->fscratch[r] = nullptr;
      ctx->fflags[r] = nullptr;
      ctx->fseq[r] = nullptr;
    }
    if (ctx->own_stream) (void)hipStreamDestroy((hipStream_t)ctx->stream[r]);
    ctx->stream[r] = nullptr;
  }
  // Last: a rank that gave up is worth saying even on the way out, and after this the word is gone.
  tp_check(*ctx);
  (void)hipHostFree(ctx->timeout);
  ctx->timeout = nullptr;
  ctx->timeout_dev = nullptr;
  ctx->ready = false;
}

namespace {

// One launch per rank, with that rank's peer table filled in. NPEER is a template argument so the
// two-rank path keeps the straight-line kernel it had; the switch below is the only place the live
// rank count meets it.
template <int NPEER>
void tp_launch_exact(TpContext& ctx, float* const* in, float* const* out, int n4, int nb, int nt,
                     int slot4, int peer4) {
  for (size_t r = 0; r < ctx.ranks; ++r) {
    TpPush<NPEER> push{};
    for (size_t q = 0, j = 0; q < ctx.ranks; ++q) {
      if (q == r) continue;
      const size_t mine = tp_peer_slot(r, q);      // where rank r sits in rank q's tables
      push.scratch[j] = (float4*)(ctx.scratch[q] + mine * ctx.max_floats);
      push.flags[j] = ctx.flags[q] + mine * kTpMaxBlocks;
      ++j;
    }
    set_device(ctx.dev[r]);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(tp_ar_kernel<NPEER>), dim3(nb), dim3(nt), 0,
                       (hipStream_t)ctx.stream[r], (const float4*)in[r], push,
                       (const float4*)ctx.scratch[r], (float4*)out[r], ctx.flags[r], ctx.seq[r],
                       ctx.timeout_dev, n4, slot4, peer4, (int)r);
  }
}

template <int NPEER>
void tp_launch_fp8(TpContext& ctx, float* const* in, float* const* out, int n, int nb,
                   int slot_bytes, int peer_bytes, int scale_off) {
  for (size_t r = 0; r < ctx.ranks; ++r) {
    TpPush8<NPEER> push{};
    for (size_t q = 0, j = 0; q < ctx.ranks; ++q) {
      if (q == r) continue;
      const size_t mine = tp_peer_slot(r, q);
      push.base[j] = (unsigned char*)(ctx.scratch[q] + mine * ctx.max_floats);
      push.flags[j] = ctx.flags[q] + mine * kTpMaxBlocks;
      ++j;
    }
    set_device(ctx.dev[r]);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(tp_ar_fp8_kernel<NPEER>), dim3(nb), dim3(256), 0,
                       (hipStream_t)ctx.stream[r], in[r], push,
                       (const unsigned char*)ctx.scratch[r], out[r], ctx.flags[r], ctx.seq[r],
                       ctx.timeout_dev, n, slot_bytes, peer_bytes, scale_off, (int)r);
  }
}

}  // namespace

void tp_all_reduce(TpContext& ctx, float* const* in, float* const* out, size_t n, TpReduce mode) {
  if (!ctx.ready || n == 0) return;
  const size_t np = tp_npeer(&ctx);
  if (!np) {
    // One rank: the sum over one term. A copy only when the caller did not alias, which the engine
    // always does — so in the shipping single-card configuration this costs nothing at all.
    if (in[0] != out[0]) {
      set_device(ctx.dev[0]);
      AFF_HIP_CHECK(hipMemcpyAsync(out[0], in[0], n * sizeof(float), hipMemcpyDeviceToDevice,
                                   (hipStream_t)ctx.stream[0]));
    }
    return;
  }
  if (mode == TpReduce::Auto)
    mode = (n * sizeof(float) >= kTpFp8MinBytes) ? TpReduce::Fp8 : TpReduce::Exact;

  if (mode == TpReduce::Fp8) {
    const int nn = (int)n;
    int nb = (int)(((n + kTpFp8Group - 1) / kTpFp8Group + 63) / 64);
    nb = std::max(1, std::min(nb, kTpMaxBlocks));
    const int peer_bytes = (int)(ctx.max_floats * sizeof(float));
    const int slot_bytes = (int)(np * ctx.max_floats * sizeof(float));
    const int scale_off = (int)ctx.max_floats;      // codes occupy [0,n); scales start past them
    switch (np) {
      case 1: tp_launch_fp8<1>(ctx, in, out, nn, nb, slot_bytes, peer_bytes, scale_off); break;
      case 2: tp_launch_fp8<2>(ctx, in, out, nn, nb, slot_bytes, peer_bytes, scale_off); break;
      case 3: tp_launch_fp8<3>(ctx, in, out, nn, nb, slot_bytes, peer_bytes, scale_off); break;
      case 4: tp_launch_fp8<4>(ctx, in, out, nn, nb, slot_bytes, peer_bytes, scale_off); break;
      case 5: tp_launch_fp8<5>(ctx, in, out, nn, nb, slot_bytes, peer_bytes, scale_off); break;
      case 6: tp_launch_fp8<6>(ctx, in, out, nn, nb, slot_bytes, peer_bytes, scale_off); break;
      default: tp_launch_fp8<7>(ctx, in, out, nn, nb, slot_bytes, peer_bytes, scale_off); break;
    }
    return;
  }

  const int n4 = (int)((n + 3) / 4);
  const int peer4 = (int)(ctx.max_floats / 4);
  const int slot4 = (int)(np * ctx.max_floats / 4);

  // Block count from the payload. Small messages want ONE block: their cost is the handshake, and
  // extra blocks add flag traffic without adding transfer.
  int nb = (int)((n4 + kTpWordsPerBlock - 1) / kTpWordsPerBlock);
  nb = std::max(1, std::min(nb, kTpMaxBlocks));
  const int nt = 256;

  // Every rank's kernel must be IN FLIGHT together — each spins on the others' flags. Launching them
  // back to back on separate devices and streams achieves that; synchronising between the launches
  // would deadlock, and a blocking synchronise per all-reduce costs many times the crossing itself.
  switch (np) {
    case 1: tp_launch_exact<1>(ctx, in, out, n4, nb, nt, slot4, peer4); break;
    case 2: tp_launch_exact<2>(ctx, in, out, n4, nb, nt, slot4, peer4); break;
    case 3: tp_launch_exact<3>(ctx, in, out, n4, nb, nt, slot4, peer4); break;
    case 4: tp_launch_exact<4>(ctx, in, out, n4, nb, nt, slot4, peer4); break;
    case 5: tp_launch_exact<5>(ctx, in, out, n4, nb, nt, slot4, peer4); break;
    case 6: tp_launch_exact<6>(ctx, in, out, n4, nb, nt, slot4, peer4); break;
    default: tp_launch_exact<7>(ctx, in, out, n4, nb, nt, slot4, peer4); break;
  }
}

void tp_sync(TpContext& ctx) {
  if (!ctx.ready) return;
  for (size_t r = 0; r < ctx.ranks; ++r) {
    set_device(ctx.dev[r]);
    AFF_HIP_CHECK(hipStreamSynchronize((hipStream_t)ctx.stream[r]));
  }
}

} // namespace aff
