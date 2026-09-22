#include "gpu/dense_gpu.h"

#include "gpu/vmem.h"

#include "gpu/digest.h"
#include "gpu/sampler_gpu.h"
#include "gpu/expert_kernel_hip.h"
#include "gpu/attention_gpu.h"
#include "gpu/attn_flash.h"
#include "gpu/compressor_gpu.h"
#include "gpu/indexer_gpu.h"
#include "gpu/batch_kernels.h"
#include "gpu/hip_common.h"
#include "gpu/device_math.h"
#include "gpu/tp_allreduce.h"
#include "gpu/trace.h"
#include "engine/instrument.h"
#include "engine/route_trace.h"
#include "ui/log.h"
#include <hip/hip_fp16.h>

#include <string>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace aff {

// Largest index_topk the attention gather serves. Above it the top-k keeps the mask alone and
// attention walks every candidate row.
constexpr uint32_t kAdmCap = 512;

namespace {
// FP8 carries one UE8M0 scale per 128x128 tile picked by row, so every shard split must land on a
// tile boundary or a shard reads another's scales. Every dense tensor here divides cleanly at 128.
constexpr uint64_t kRowAlign = 128;

// ---- the small ops -----------------------------------------------------------------------------
//
// None of these move meaningful bytes. They are on device because they sat BETWEEN GPU matvecs on
// the host and each forced the activation back across the bus — ~730 round trips a token.

// yarn_ramp_d and rope_pair: gpu/device_math.h, one definition for every caller on device.

__global__ void rope_tail_kernel(float* __restrict__ x, uint32_t n_head, uint32_t head_dim,
                                 uint32_t n_rot, uint32_t pos, float freq_scale, float theta_scale,
                                 float ext_factor, float lo, float hi, float sin_sign) {
  const uint32_t pairs = n_rot / 2;
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n_head * pairs) return;
  const uint32_t h = t / pairs, k = t % pairs;
  rope_pair(x + (size_t)h * head_dim + (head_dim - n_rot), k, pos, freq_scale, theta_scale,
            ext_factor, lo, hi, sin_sign);
}

// A norm and the RoPE tail after it are two dispatches over at most 32 KiB, and a dispatch costs
// about what 12 MB of weights do. Fused: they are separated only by a barrier the block has anyway.
__global__ void head_rms_rope_kernel(float* __restrict__ x, uint32_t head_dim, float eps,
                                     uint32_t n_rot, uint32_t pos, float freq_scale,
                                     float theta_scale, float ext_factor, float lo, float hi,
                                     float sin_sign) {
  __shared__ float red[32];
  float* p = x + (size_t)blockIdx.x * head_dim;
  const uint32_t tid = threadIdx.x, nt = blockDim.x;
  float acc = 0.0f;
  for (uint32_t i = tid; i < head_dim; i += nt) { const float v = p[i]; acc += v * v; }
  const uint32_t lane = tid & 31, warp = tid >> 5;
  for (int o = 16; o; o >>= 1) acc += __shfl_down(acc, o);
  if (lane == 0) red[warp] = acc;
  __syncthreads();
  if (tid == 0) {
    float t = 0.0f;
    for (uint32_t w = 0; w < (nt + 31) / 32; ++w) t += red[w];
    red[0] = rsqrtf(t / (float)head_dim + eps);
  }
  __syncthreads();
  const float inv = red[0];
  for (uint32_t i = tid; i < head_dim; i += nt) p[i] = p[i] * inv;
  __syncthreads();                                   // the scaled values, before they are rotated
  for (uint32_t k = tid; k < n_rot / 2; k += nt)
    rope_pair(p + (head_dim - n_rot), k, pos, freq_scale, theta_scale, ext_factor, lo, hi,
              sin_sign);
}

__global__ void rms_norm_rope_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                     float* __restrict__ out, uint32_t n, float eps,
                                     uint32_t n_rot, uint32_t pos, float freq_scale,
                                     float theta_scale, float ext_factor, float lo, float hi,
                                     float sin_sign) {
  __shared__ float red[32];
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
  const float inv = red[0];
  for (uint32_t i = tid; i < n; i += nt) out[i] = x[i] * inv * (w ? w[i] : 1.0f);
  __syncthreads();
  for (uint32_t k = tid; k < n_rot / 2; k += nt)
    rope_pair(out + (n - n_rot), k, pos, freq_scale, theta_scale, ext_factor, lo, hi, sin_sign);
}

}

// Every full drain of a device stream bumps a counter, so code that only needs "did an async copy
// issued before now certainly land?" can answer it without synchronising again.
//
// Timed, because a phase counter cannot otherwise tell work from wait — a label naming host work can
// be almost entirely this drain. The clock pair runs under AFF_BLOCK or AFF_PROFILE.
//
// It must honour BOTH variables, like every other AFF_BLOCK_AT site. Gating a drain on AFF_PROFILE
// alone makes the table blind in both directions at once: `AFF_BLOCK=1` measures the shipped path
// and would report no drains at all, while `AFF_PROFILE=1` turns the hybrid census on, which walks
// the host's copy of the selection and so forces the very readback the table exists to show the
// absence of.
//
// TAGGED BY SITE, not just counted. A single aggregate is what lets a drain hide inside a phase
// named for host work; the per-site table in engine/instrument.h is the fix, and the target for
// each row is zero rather than "small in sum". See the second half of that header.
#define AFF_DRAIN_AT(D, SITE) do { \
    if (aff::block_timed()) { \
      const auto _d0 = std::chrono::steady_clock::now(); \
      AFF_HIP_CHECK(hipStreamSynchronize((D).stream)); \
      aff::block_record((SITE), (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>( \
          std::chrono::steady_clock::now() - _d0).count()); \
    } else { \
      AFF_HIP_CHECK(hipStreamSynchronize((D).stream)); \
    } \
    ++(D).drained; } while (0)

// Untagged sites still count, into `Other`. Widening the instrument must never shrink the total it
// is meant to widen, so there is no untracked spelling of a drain.
#define AFF_DRAIN(D) AFF_DRAIN_AT(D, aff::BlockSite::Other)

// ---- A PINNED STAGING RING, AND WHY EVERY H2D OFF THE HOT PATH NEEDS ONE --------------------------
//
// A host buffer that feeds `hipMemcpyAsync` is read by the DMA at some point AFTER the call returns.
// While a drain followed the copy that was invisible, because the drain guaranteed the transfer had
// landed before anything could overwrite the source. Take the drain away and every such buffer
// becomes a race whose symptom is not a crash: it is fluent text with a different hash, because the
// value that arrived was the NEXT layer's.
//
// The two wrong fixes, both of which this file has held at some point:
//
//   * pageable source. The driver then stages synchronously, which is correct and is a hidden host
//     stall — the very thing being removed, wearing a name no counter reads.
//   * one pinned slot with an event. Correct, and it reintroduces the stall exactly: in the steady
//     state the host IS ahead, so the wait binds on every call and pins the host back to one call of
//     lookahead, which is the drain again with extra steps.
//
// So: N slots, each with the event of the transfer that last read it. Deeper than the host's
// lookahead and it never binds; `slot()` waits only if it has wrapped onto a transfer still pending,
// and that wait is counted at the caller's site so the depth is checked by the instrument rather
// than by whoever picked the number.
// ---- HOW DEEP, AND WHY IT IS A BYTE BUDGET AND NOT A SLOT COUNT ----------------------------------
//
// Depth only has to exceed the host's lookahead, which is counted in CALLS of the site that owns the
// ring. A per-layer site needs tens; a per-block site needs a handful. But the stride is set by the
// widest chunk the engine will ever hand it, and for anything token-major that is a prefill chunk —
// 1408 x 16384 f32 is 92 MiB a slot, so a fixed 64 slots would ask for 5.9 GiB of PINNED memory.
//
// That is not merely large, it is the most expensive memory in the engine. Pinned pages come out of
// the same host RAM the expert pool is sized from, and a MiB not in the pool is an expert nearer the
// SSD tier. So a ring is given a BYTE budget and takes
// as many slots as fit, at least one. A site whose stride is small gets the depth it wants for free;
// a site handed a prefill-width chunk gets one slot and waits, which is correct, is confined to a
// path that is not the one being freed, and is visible in the table rather than silent.
struct StageRing {
  static constexpr uint32_t kMaxSlots = 64;
  uint8_t* host = nullptr;                  // pinned, slots * stride bytes
  hipEvent_t ev[kMaxSlots] = {};
  uint32_t next = 0;
  uint32_t slots = 0;
  size_t stride = 0;                        // bytes a slot

  // ---- WHY THIS IS `fit` AND NOT `alloc` --------------------------------------------------------
  //
  // The stride a site needs is a function of the CALL, not of the arena, and the two differ by two
  // orders of magnitude on the same site: `batch_begin` stages 1408 x 4096 f32 = 23 MiB for a
  // prefill chunk and 16 x 4096 = 256 KiB for a decode block. Size the ring from the arena and
  // decode inherits a one-slot ring that binds every block — the drain again, wearing a ring's name.
  // Size it from decode and prefill does not fit at all.
  //
  // So the ring follows the width, and reallocates when the width moves out of its range. The
  // hysteresis is what keeps that from thrashing: a stride is kept while it is big enough and not
  // wastefully big, so the prefill -> decode transition pays one realloc and the steady state pays
  // none. `nb` is constant within a phase, which is what makes this a transition cost and not a
  // per-call one.
  bool fit(size_t bytes_per_slot, size_t budget_bytes) {
    if (!bytes_per_slot) return false;
    if (ready() && stride >= bytes_per_slot && stride <= 4 * bytes_per_slot) return true;
    // Every outstanding transfer is reading the memory about to be freed. This is the one place the
    // ring blocks by design, and it is a phase change rather than a call.
    for (uint32_t i = 0; i < slots; ++i) if (ev[i]) (void)hipEventSynchronize(ev[i]);
    free();
    size_t n = budget_bytes / bytes_per_slot;
    if (n < 1) n = 1;
    if (n > kMaxSlots) n = kMaxSlots;
    if (hipHostMalloc((void**)&host, n * bytes_per_slot) != hipSuccess) { host = nullptr; return false; }
    stride = bytes_per_slot;
    slots = (uint32_t)n;
    next = 0;
    for (uint32_t i = 0; i < slots; ++i)
      if (hipEventCreateWithFlags(&ev[i], hipEventDisableTiming) != hipSuccess) { free(); return false; }
    return true;
  }
  void free() {
    if (host) { (void)hipHostFree(host); host = nullptr; }
    for (hipEvent_t& e : ev) if (e) { (void)hipEventDestroy(e); e = nullptr; }
    stride = 0; next = 0; slots = 0;
  }
  bool ready() const { return host != nullptr && slots != 0; }

  // Claim the next slot. Returns its pinned address; `s` receives the index so the caller can record
  // against it after issuing the copy. The wait is the caller's to wrap in AFF_BLOCK_AT: a ring that
  // is too shallow has to show up under the site that owns it, not under a shared one.
  uint8_t* claim(uint32_t* s) {
    const uint32_t i = next;
    next = (i + 1) % slots;
    if (s) *s = i;
    return host + (size_t)i * stride;
  }
  hipError_t wait(uint32_t s) { return ev[s] ? hipEventSynchronize(ev[s]) : hipSuccess; }
  hipError_t done(uint32_t s, hipStream_t st) { return hipEventRecord(ev[s], st); }
};

// ---- THE RING BUDGETS, AND WHY THEY ARE SMALL ----------------------------------------------------
//
// These are PINNED pages, and pinned pages come out of the same host RAM the expert pool is sized
// from, so a ring is priced in EXPERTS: at ~4.3 MiB a shard, 16 MiB of ring is about four of them.
// That is the right frame for choosing these, and the reason neither number is larger. A ring only
// has to out-run the host's lookahead, and the host's lookahead is a handful of blocks. Converting
// ring bytes into decode time needs the VRAM-to-decode rate, which FLATTENS as the hit rate rises —
// do not treat it as a constant when re-deriving these.
constexpr size_t kRingPlaneBudget = 16u << 20;   // strides that scale with tokens x n_embd
constexpr size_t kRingCountBudget =  4u << 20;   // strides that scale with tokens alone

// Headroom kept free when an allocation is sized against hipMemGetInfo. The driver's own working
// set grows with what we allocate, so a request that exactly fits the reported free figure fails
// later rather than here. Fixed rather than a fraction of the card, because what it covers is the
// driver's overhead and not a share of VRAM; the KV cache's is larger because its allocation is.
constexpr size_t kAllocHeadroom   = 256u << 20;
constexpr size_t kKvAllocHeadroom = 512u << 20;

struct DenseGpu::Impl {
  // Every device buffer the engine owns lives here, one instance per rank — including the hidden
  // state, the KV cache and the batch arena. Both cards carry the whole state and each computes its
  // own shard of every weight; a chain cannot run on card 1 if its activation only exists on card 0.
  struct Dev {
    int id = -1;
    hipStream_t stream = nullptr;
    // The shared expert's own stream. On one stream it and the routed dispatch serialise, though
    // they share only the activation they read — shexp writes d_p, the routed set d_block. Two
    // events per rank are the whole ordering the hardware needs.
    hipStream_t sstream = nullptr;
    hipEvent_t  ev_sx = nullptr;   // recorded on `stream`: d_norm is ready for the side stream
    hipEvent_t  ev_sd = nullptr;   // recorded on `sstream`: d_p holds the shared expert's partial
    bool shexp_side = false;       // this rank has a partial in d_p that collect() must fold in
    // ev_sd carries a record for THIS layer; waiting on one last recorded in some other layer, or
    // never, is a different ordering.
    bool sx_ev = false;
    // The held launch: everything shexp() would have passed to its two calls, kept until the router
    // has issued and the side stream can be released behind it. Shard pointers rather than copies —
    // `ent_` outlives every call and reg() does not move a shard once placed.
    const Shard* sx_g = nullptr; const Shard* sx_u = nullptr; const Shard* sx_w = nullptr;
    uint32_t sx_qg = 0, sx_qu = 0, sx_qw = 0;
    const float* sx_x = nullptr;
    uint32_t sx_li = 0, sx_hidden = 0;
    float sx_limit = 0.0f;
    float* d_x = nullptr;
    float* d_y = nullptr;
    // Scratch for fused chains: two intermediates plus the activation between them. Allocated once,
    // never freed until teardown, so a fused block issues no allocation at all.
    float* d_g = nullptr;
    float* d_u = nullptr;
    float* d_a = nullptr;
    // One more, for the Col half of a chain to write its PARTIAL into without clobbering the
    // operand the Row half left in d_a.
    float* d_p = nullptr;
    uint64_t cap_mid = 0;

    // ---- the hidden state, replicated ---------------------------------------------------------
    float* d_hc = nullptr;    float* d_hc_next = nullptr;
    float* d_pre = nullptr;   float* d_post = nullptr;    float* d_comb = nullptr;
    // V4.1: the mix the PREVIOUS sublayer computed, which is what this one collapses with.
    float* d_pre_prev = nullptr;  float* b_pre_prev = nullptr;
    float* d_mix = nullptr;
    // Per-block partial Gram matrices for the wide hc path: nblk * n_hc*(n_hc+1)/2 floats, 640
    // bytes at this shape, and the reason hc_fused_kernel need not be one block. See hc_gram_kernel.
    float* d_gpart = nullptr;
    float* d_mpart = nullptr;   // split x KM matvec partials
    float* d_cur = nullptr;   float* d_norm = nullptr;    float* d_block = nullptr;
    // qr_norm, kept where attn_q left it. d_u is scratch that attn_kv overwrites, and the single
    // token's indexer reads this two chains later.
    float* d_qrn = nullptr;   uint32_t d_qrn_cap = 0;

    // ---- attention, replicated cache and this rank's half of the heads -------------------------
    // ---- PER-SEQUENCE state, one set per SLOT (see --slots). Slot-major so a slot is a
    // contiguous, independently allocated sequence: its own KV ring, compressed cache, indexer
    // key cache and compressor window. Everything else in Dev is shared model state.
    std::vector<std::vector<uint8_t*>> a_raw_s, a_comp_s;
    // The compressed cache is reserved for the whole context the engine can serve and backed only
    // as far as the sequence has actually reached. `a_comp[l]` is the span's base and never moves,
    // so every kernel keeps its linear addressing. See gpu/vmem.h for why this and not paging.
    std::vector<std::vector<VirtualSpan>> a_comp_span_s;
    hipStream_t a_kv_zero = nullptr;   // shared by every span on this card; see attn_init
    std::vector<float*>   a_sink;
    // The indexer's compressed keys as the scoring kernel wants them: E4M3 rows with one f32 scale
    // each, written once when the compressor emits them.
    std::vector<std::vector<uint8_t*>> a_idx8_s;
    std::vector<std::vector<float*>>   a_idxs_s;
    // The compressor's cross-chunk window, per layer: kCompHist raw positions of each projection,
    // indexed by `position & 7`.
    std::vector<std::vector<float*>>   c_hkv_s, c_hsc_s, ci_hkv_s, ci_hsc_s;
    std::vector<uint32_t> c_ratio;      // the ring size follows from it, so it is remembered
    // Per-layer constants, uploaded on the layer's first chunk and kept.
    std::vector<float*>   c_ape, c_norm, ci_ape, ci_norm;
    // This chunk's four projections, token-major, which is what the pool indexes by position.
    float* c_kv = nullptr;  float* c_sc = nullptr;
    float* ci_kv = nullptr; float* ci_sc = nullptr;
    uint32_t c_cap_tok = 0;
    // The deferred decode compressor, per layer: `norm` for every position since this layer's last
    // pool, dim-major [n_embd][c_pcap], the layout batch_gemm_hip reads. Only the pool reads the
    // projections, and the ring between them is keyed to absolute position, so nothing needs them
    // until then.
    std::vector<float*>   c_pend;
    std::vector<uint32_t> c_np;         // positions buffered
    std::vector<uint32_t> c_ppos0;      // absolute position of column 0
    std::vector<uint32_t> c_pcap;       // = the layer's ratio
    float* c_py = nullptr;              // flush GEMM output, dim-major, shared across layers
    uint32_t c_py_cap = 0;              // in floats
    float*   a_q = nullptr;      // a_hpr heads
    float*   a_out = nullptr;    // a_hpr heads
    uint8_t* a_allow = nullptr;

    // ---- small f32 vectors and the router gate --------------------------------------------------
    // The vectors are replicated (norm weights, hyper-connection scales — a few KiB each).
    std::vector<float*> vec;
    // bf16: 2 MiB a layer rather than 4. Top-6 over a softmax needs the ORDER of the leading
    // logits, not their last bits, and the accumulator is f32 either way. Replicated on both cards;
    // `router_rows` splits the DECODE matvec over that replica and says nothing about the layout.
    std::vector<uint16_t*> router;
    // The aux-loss-free correction bias, one [n_expert] f32 vector per handle, in VRAM. Constant
    // for the run, and at 40 layers a token an H2D per layer is exactly the host traffic decode
    // cannot afford; the batched path uploads it per call because 1 KB against a chunk is nothing.
    std::vector<float*> rbias;
    uint64_t router_rows = 0;    // this rank's share of n_expert, for the decode matvec only
    uint64_t router_row0 = 0;
    float* d_rlog = nullptr;

    // ---- the batch arena ------------------------------------------------------------------------
    // bf16, not f32: the hc lanes are n_embd * n_hc = 16384 rows a token, the largest item in the
    // arena, and every kernel touching them was at its DRAM roofline. The GEMM that reads them
    // truncates to bf16 in LDS anyway; the two elementwise kernels round.
    uint16_t* b_hc = nullptr; uint16_t* b_hc_next = nullptr;
    float* b_mix = nullptr;
    float* b_pre = nullptr;  float* b_post = nullptr;    float* b_comb = nullptr;
    float* b_cur = nullptr;  float* b_norm = nullptr;    float* b_block = nullptr;
    // The narrow hc path's cross-block partials, [kHcnChunks][kNarrowTok] — 2 KiB, so the reduction
    // can span a grid instead of a workgroup.
    float* b_hcp = nullptr;
    float* b_q = nullptr;    float* b_kv = nullptr;
    // bf16, like b_hc: 16384 rows a token, and both readers (attention, which quantises Q to E4M3,
    // and wo_a's GEMM, which truncates in LDS) round anyway. b_heads aliases it.
    uint16_t* b_qtm = nullptr;
    float* b_hinv = nullptr;                              // [n_head/R][nb], the per-head RMS scale
    float* b_qr = nullptr;   float* b_g = nullptr;       float* b_u = nullptr;
    float* b_y = nullptr;    float* b_tm = nullptr;      float* b_emb = nullptr;
    // Aliases, not allocations — see batch_init for the lifetime proof. Head-sized buffers are a
    // large share of the arena between them, and the chunk size is rationed against the arena. Not
    // freed.
    float* b_htm = nullptr;                               // == b_q
    uint16_t* b_heads = nullptr;                          // == b_qtm
    // ---- the vocabulary head for a whole speculative block --------------------------------------
    // Sized for kNarrowTok columns and not for b_cap_: a block is six tokens and the plane is
    // [vocab][nb], so at a prefill chunk's width this would be 634 MB a card for a buffer only a
    // six-token pass ever writes. b_amx is the per-token (best, row) pair the argmax leaves behind,
    // which is all that crosses the bus — 129280 logits a token would be 3.1 MB a block.
    float* b_head = nullptr;
    float* b_amx = nullptr;
    uint64_t* b_mbest = nullptr;                          // the Markov step's grid-wide maximum
    // The Markov chain, entirely on the card: `b_ids` is [1 + kNarrowTok] with the block's first
    // token at 0 and each step's winner after it, `b_mpeer` the eight bytes of each other card's key,
    // `ev_mk` one event a step so the cards' steps can be ordered against each other without the
    // host waiting on any of them. `h_ids` is the single readback at the end of the block.
    uint32_t* b_ids = nullptr;  uint32_t* h_ids = nullptr;
    uint64_t* b_mpeer = nullptr;
    hipEvent_t ev_mk[kNarrowTok] = {};
    // The TARGET's head does the same exchange, once a block instead of once a step: `b_apeer` holds
    // the other cards' [2][nb] argmax planes, `b_hids` the combined winner per position and `h_hids`
    // the one readback. `ev_amx` is recorded after this card's argmax so the peer's stream can order
    // its copy against it. Kept separate from the Markov chain's buffers rather than aliased —
    // they are 64 and 128 bytes, and two logically different things sharing one allocation is how a
    // reordering becomes a silent wrong token.
    float* b_apeer = nullptr;
    uint32_t* b_hids = nullptr;  uint32_t* h_hids = nullptr;
    hipEvent_t ev_amx = nullptr;
    StageRing tok;                                        // the block's first token, 4 bytes a slot
    // The draft's own two: the target's tap staged for the projection, and the same transposed;
    // then main_norm(main_proj(.)), which is all three stages ever see of the target.
    float* b_dxt = nullptr;  float* b_dx = nullptr;  float* b_dm = nullptr;
    // The DSpark tap plane itself, [rows][taps * n_embd] token-major — written by batch_hc_tap as
    // the target runs and read by draft_proj, both on this card. It is the whole of what used to be
    // Block::main_hidden, a host vector that crossed the bus twice.
    // PER SLOT: the target's verify writes this plane and the NEXT block's draft reads it, so it
    // is per-sequence state, not scratch. Sharing it across slots leaves each sequence's draft
    // conditioned on its neighbour's hidden states — correct output (the verify rejects the bad
    // proposals) at a collapsed acceptance rate, i.e. a silent ~2.4x slowdown.
    std::vector<float*> b_tap_s;  std::vector<size_t> tap_bytes_s;
    // ---- the W8A8 activation ------------------------------------------------------------------
    // One slot, and no memo: a cache keyed on the source pointer would be wrong, because b_norm is
    // rewritten by hc_pre every layer and keeps its address. quant_act() is called where the
    // activation is ready and every GEMM reading it is issued before the next quant_act().
    uint8_t* q8 = nullptr;  float* qs = nullptr;
    uint32_t q_cap = 0;                                   // dim this slot was sized for
    uint32_t q_dim = 0;                                   // dim currently in it: the X row stride
    const float* q8_src = nullptr;                        // which plane the slot holds, or null
    float* h_tm = nullptr;                                // pinned, b_mid_ * b_cap_ floats
    uint8_t* b_allow = nullptr;                           // device, the same plane
    // ---- THE FOUR STAGING RINGS -----------------------------------------------------------------
    //
    // Every H2D on the hot path whose source the host rewrites. See StageRing for why each is a ring
    // and not a buffer; what follows is only what each one carries and how often it turns over.
    //
    //   ncomp   how many KV rows each token attends to, one uint32 a token. Once per SUB-BATCH per
    //           layer — not once per layer — so a prefill chunk split into groups reaches it several
    //           times more often than 43. This is the one that was a single buffer and produced an
    //           attention output that differed from an identical q, non-deterministically.
    //   rsel    a hash layer's selection and its per-token count, handed to the card because a hash
    //           is a host table lookup. Three layers a block.
    //   emb     the block's seed embedding. Its source is a member buffer, but one that is rebuilt
    //           in place every block, which is the same hazard with a longer fuse.
    //
    // If any of the ring rows is ever non-zero in the AFF_BLOCK table, that ring is shallower than
    // the host's lookahead and the fix is a larger budget, not a different design.
    StageRing ncomp, rsel, emb;
    uint32_t* b_ncomp = nullptr;                          // device, the same counts
    // The lightning indexer, all of it on device. `i_q`/`i_w`/`i_k` are the staged f32 inputs,
    // `i_q8`/`i_k8` their E4M3 forms with one scale a row, `i_sc` the [n][n_keys] score plane.
    // Its output IS b_allow, so the mask never crosses the bus in either direction.
    float* i_q = nullptr;   uint8_t* i_q8 = nullptr;  float* i_qs = nullptr;
    float* i_w = nullptr;
    float* i_k = nullptr;   uint8_t* i_k8 = nullptr;  float* i_ks = nullptr;
    float* i_sc = nullptr;
    // The flash key split's partial (O, m, l) arena. Preallocated in reserve_runtime and never
    // grown: the split is only taken when the natural grid is small, so this holds a handful of
    // tokens and not a prefill chunk. See attn_flash.h.
    FlashSplit fsplit;
    // The multi-block top-k's histogram and per-token state. Tiny — 8 tokens is 8.3 KiB — and
    // zeroed once here, because every pass of the select leaves it back at zero.
    TopkScratch topk;
    // The indexer's admitted-row LISTS, which is what attention gathers through. `n` holds the
    // count a token, written by the same kernel, so nothing about it crosses the bus.
    uint32_t* a_adm = nullptr;   uint32_t* a_admn = nullptr;    // decode: one token
    uint32_t* b_adm = nullptr;   uint32_t* b_admn = nullptr;    // prefill: the chunk
    float* hi_q = nullptr;  float* hi_w = nullptr;    float* hi_k = nullptr;   // pinned staging
    uint32_t* i_nc = nullptr;  uint32_t* hi_nc = nullptr;   // per-token key counts
    uint32_t i_cap_tok = 0, i_cap_key = 0;
    // The router's top-k, rank 0 only. sel/wt/nsel come back; bias and the residency mask go out.
    uint32_t* b_rsel = nullptr;  float* b_rwt = nullptr;  uint32_t* b_rn = nullptr;
    float* b_rbias = nullptr;                    // [n_router][n_expert], one slice a layer
    std::vector<uint8_t> rbias_up;               // which slices have been filled
    size_t r_back = 0, r_woff = 0, r_noff = 0;   // the sel/wt/n block and its interior offsets
    uint8_t* b_rres = nullptr;
    // ---- THE SELECTION, ON EVERY CARD ------------------------------------------------------------
    //
    // The top-k runs on rank 0 over the whole gate replica, so rank 0 owns the answer and every
    // other rank needs a copy of it before its own dispatch can name an expert. Today that copy goes
    // through the host: a drain, a D2H, and the host hands `sel` to both cards. A device-built
    // dispatch reads it on the card instead, so rank r gets its own block and rank 0 pushes into it
    // over the peer link — the same one-shot push with a release the all-reduce already runs.
    //
    // ---- A RING, AND ONE EVENT PER SLOT --------------------------------------------------------
    //
    // Both halves have to be ringed, and for the same reason: nothing orders the NEXT layer's push
    // after THIS layer's read of it. The forward edge exists — card r waits on the event card 0
    // records after the copy — but there is no back edge, and with the router's drain gone the host
    // is no longer bounded to a layer ahead.
    //
    // The event matters as much as the buffer. A single event re-recorded every layer is a single
    // signal: a wait submitted while that signal is still set from an EARLIER record is satisfied
    // immediately, and the card reads a mirror the push has not landed in yet. That is what three
    // different `ids-md5` values from three runs of one binary looked like, with the plan itself
    // verified identical on 1050 dispatches.
    //
    // 64 slots: more than the 43 layers of a block, so it cannot wrap inside one, and the block
    // boundary still has waits of its own. `route_wait` guards the wrap anyway — a host that IS 64
    // layers ahead is not the bottleneck, so blocking it there costs nothing.
    static constexpr uint32_t kRouteSlots = 64;
    uint32_t* b_rsel_m = nullptr;  float* b_rwt_m = nullptr;   // rank 0: its own block, unringed
    void* m_ring = nullptr;                      // rank > 0: kRouteSlots blocks
    void* ev_route[kRouteSlots] = {};            // rank 0: one per slot
    uint32_t* h_rsel = nullptr;  float* h_rwt = nullptr;  uint32_t* h_rn = nullptr;
    // Pinned staging for the host side of every transfer. A hipMemcpyAsync out of pageable memory is
    // neither async nor a DMA — the runtime pins and copies through its own bounce buffer, which
    // costs far more than the kernel it accompanies. A memcpy into pinned memory is nearly free at
    // the sizes these carry.
    //
    // One slot per call site, each with an event recorded after its H2D is enqueued: a site that
    // does not synchronise (attn_q and attn_kv leave their work in flight by design) would
    // otherwise let the next token's memcpy overwrite a buffer a DMA is still reading.
    float* h_y = nullptr;          // max_rows floats, D2H only: every reader synchronises first
    // Separate from h_y: the deferred `norm` copy stays in flight across the device router, and the
    // router lands its logits in h_y.
    float* h_norm = nullptr;
    // Bumped by AFF_DRAIN. See the macro for why this is cheaper than asking HIP.
    uint64_t drained = 0;
    struct Stage { float* h = nullptr; hipEvent_t ev = nullptr; };
    Stage stg[5];                  // indexed by StageSlot below
  };
  std::vector<Dev> dev;
  TpContext tp;
  ihipStream_t* const* streams() {
    for (size_t r = 0; r < dev.size(); ++r) sp[r] = (ihipStream_t*)dev[r].stream;
    return sp;
  }
  ihipStream_t* sp[kMaxRanks] = {nullptr};
};

// Which pinned slot a call site owns.
enum StageSlot { kStgMv = 0, kStgMvg = 1, kStgQ = 2, kStgKv = 3, kStgShexp = 4 };

DenseGpu::~DenseGpu() {
  if (!impl_) return;
  if (impl_->tp.ready) tp_free(&impl_->tp);
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    (void)hipSetDevice(D.id);
    for (Entry& e : ent_) {
      if (e.sh[d].data) (void)hipFree(e.sh[d].data);
      if (e.sh[d].scales) (void)hipFree(e.sh[d].scales);
    }
    float* const bufs[] = {
      D.d_g, D.d_u, D.d_a, D.d_p, D.d_x, D.d_y,
      D.d_hc, D.d_hc_next, D.d_pre, D.d_post, D.d_comb, D.d_mix, D.d_cur, D.d_norm, D.d_block,
      D.a_q, D.a_out, D.d_rlog,
      D.b_mix, D.b_pre, D.b_post, D.b_comb, D.b_cur, D.b_norm, D.b_block,
      D.b_emb, D.b_q, D.b_hinv, D.b_kv, D.b_qr, D.b_g, D.b_u, D.b_y, D.b_tm,
      D.b_head, D.b_amx, D.b_dxt, D.b_dx, D.b_dm};
    // b_htm and b_heads are deliberately absent: they alias b_q and b_qtm.
    for (float* p : bufs) if (p) (void)hipFree(p);
    for (uint16_t* p : {D.b_hc, D.b_hc_next, D.b_qtm}) if (p) (void)hipFree(p);
    for (float* p : D.vec) if (p) (void)hipFree(p);
    for (uint16_t* p : D.router) if (p) (void)hipFree(p);
    for (float* p : D.rbias) if (p) (void)hipFree(p);
    for (float* p : D.b_tap_s) if (p) (void)hipFree(p);
    for (auto& v : D.a_raw_s)  for (uint8_t* p : v) if (p) (void)hipFree(p);
    for (auto& v : D.a_comp_s) for (uint8_t* p : v) if (p) (void)hipFree(p);
    for (float* p : D.c_pend) if (p) (void)hipFree(p);
    if (D.c_py) (void)hipFree(D.c_py);
    for (float* p : D.a_sink) if (p) (void)hipFree(p);
    if (D.a_allow) (void)hipFree(D.a_allow);
    if (D.fsplit.o) (void)hipFree(D.fsplit.o);
    if (D.fsplit.ml) (void)hipFree(D.fsplit.ml);
    if (D.topk.hist) (void)hipFree(D.topk.hist);
    if (D.topk.state) (void)hipFree(D.topk.state);
    if (D.topk.tie) (void)hipFree(D.topk.tie);
    for (uint32_t* q : {D.a_adm, D.a_admn, D.b_adm, D.b_admn}) if (q) (void)hipFree(q);
    if (D.b_mbest) (void)hipFree(D.b_mbest);
    if (D.b_mpeer) (void)hipFree(D.b_mpeer);
    if (D.b_ids) (void)hipFree(D.b_ids);
    if (D.h_ids) (void)hipHostFree(D.h_ids);
    for (hipEvent_t& e : D.ev_mk) if (e) (void)hipEventDestroy(e);
    if (D.ev_amx) (void)hipEventDestroy(D.ev_amx);
    if (D.b_apeer) (void)hipFree(D.b_apeer);
    if (D.b_hids) (void)hipFree(D.b_hids);
    if (D.h_hids) (void)hipHostFree(D.h_hids);
    D.tok.free();
    if (D.h_y) (void)hipHostFree(D.h_y);
    if (D.h_norm) (void)hipHostFree(D.h_norm);
    if (D.h_tm) (void)hipHostFree(D.h_tm);
    if (D.b_allow) (void)hipFree(D.b_allow);
    for (StageRing* r : {&D.ncomp, &D.rsel, &D.emb}) r->free();
    if (D.b_ncomp) (void)hipFree(D.b_ncomp);
    for (void* p : {(void*)D.b_rsel, (void*)D.b_rwt, (void*)D.b_rn, (void*)D.b_rbias,
                    (void*)D.b_rres}) if (p) (void)hipFree(p);
    for (void* p : {(void*)D.h_rsel, (void*)D.h_rwt, (void*)D.h_rn}) if (p) (void)hipHostFree(p);
    for (auto& s : D.stg) {
      if (s.h) (void)hipHostFree(s.h);
      if (s.ev) (void)hipEventDestroy(s.ev);
    }
    if (D.ev_sx) (void)hipEventDestroy(D.ev_sx);
    if (D.ev_sd) (void)hipEventDestroy(D.ev_sd);
    if (D.sstream) (void)hipStreamDestroy(D.sstream);
    if (D.stream) (void)hipStreamDestroy(D.stream);
  }
  delete impl_;
  impl_ = nullptr;
}

// How many logical ranks to run, and which physical device backs each.
//
// AFF_LOGICAL_RANKS oversubscribes: N ranks are dealt round-robin over the cards that are actually
// present, so a two-card box can execute the N-rank sharding math, the N-way collective and the
// placement engine's per-rank loops at N=4 and N=8. Two ranks on one card get their own streams and
// their own allocations, so every protocol runs for real — a "peer" that happens to be the same
// device is a local write, which is the degenerate case and not a skipped one.
//
// It is a CORRECTNESS instrument and says nothing about speed: the ranks contend for one card's
// bandwidth and each still pays the full per-rank VRAM. Nothing in the shipping path reads it.
static size_t rank_plan(const std::vector<int>& devs, size_t want, std::vector<int>* out) {
  size_t use = want ? std::min(want, devs.size()) : devs.size();
  bool logical = false;
  if (const char* s = std::getenv("AFF_LOGICAL_RANKS")) {
    const long n = std::strtol(s, nullptr, 10);
    if (n > 0) { use = (size_t)n; logical = true; }
  }
  if (use > kMaxRanks) {
    ui::err("%zu ranks requested, this build's tables hold %zu — using %zu. Raise kMaxRanks in "
            "dense_gpu.h to go wider.\n", use, kMaxRanks, kMaxRanks);
    use = kMaxRanks;
  }
  out->resize(use);
  for (size_t r = 0; r < use; ++r) (*out)[r] = devs[r % devs.size()];
  if (logical)
    ui::err("AFF_LOGICAL_RANKS: %zu logical ranks over %zu physical cards. This is a correctness "
            "harness — timings from it mean nothing.\n", use, devs.size());
  return use;
}

bool DenseGpu::init(uint64_t max_rows, uint64_t max_cols, std::string* err, size_t want_ranks) {
  static_assert(kMaxRanks == (size_t)kMaxGpuDevices, "per-rank and per-device tables must agree");
  const std::vector<int> devs = gfx12_devices();
  if (devs.empty()) {
    if (err)
      *err = std::string("no ") + AFF_GPU_ARCH + " GPU found (of " +
             std::to_string(hip_device_count()) + " HIP devices)";
    // Name every card that was enumerated and rejected, so "found 0" with two cards installed is
    // diagnosable rather than mysterious.
    report_rejected_devices();
    return false;
  }
  std::vector<int> plan;
  const size_t use = rank_plan(devs, want_ranks, &plan);

  // Host-side spin-waiting instead of blocking on the completion interrupt lands inside the
  // run-to-run spread. Do not re-try it.

  impl_ = new Impl();
  impl_->dev.resize(use);
  // Every per-sequence vector is slot-major; give each its outer dimension before anything
  // indexes [slot_]. n_slots_ is 1 unless --slots raised it.
  init_slot_keys();
  for (auto& D : impl_->dev) {
    D.a_raw_s.resize(n_slots_);   D.a_comp_s.resize(n_slots_);
    D.a_comp_span_s.resize(n_slots_);
    D.a_idx8_s.resize(n_slots_);  D.a_idxs_s.resize(n_slots_);
    D.c_hkv_s.resize(n_slots_);   D.c_hsc_s.resize(n_slots_);
    D.ci_hkv_s.resize(n_slots_);  D.ci_hsc_s.resize(n_slots_);
  }
  for (size_t d = 0; d < use; ++d) {
    impl_->dev[d].id = plan[d];
    AFF_HIP_CHECK(hipSetDevice(plan[d]));
    // Each rank only ever writes its own row slice, but sizing to the full width keeps the
    // arithmetic obvious and costs half a megabyte.
    AFF_HIP_CHECK(hipMalloc(&impl_->dev[d].d_x, max_cols * sizeof(float)));
    AFF_HIP_CHECK(hipMalloc(&impl_->dev[d].d_y, max_rows * sizeof(float)));
    AFF_HIP_CHECK(hipStreamCreate(&impl_->dev[d].stream));
    // The trace watches ONE card's stream. Its events are created here, not lazily: an event
    // belongs to the device current at creation and may only be recorded on that device's stream.
    // Read a slot the two cards run differently (`reduce`) as "somebody waited", not as a cost.
    if (d == 0) trace::g_trace.bind(impl_->dev[d].stream);

    // Intermediates are bounded by the widest thing a fused chain carries: n_head*qk_width for the
    // query path, moe_inter for the shared expert. max_rows covers both.
    impl_->dev[d].cap_mid = max_rows;
    AFF_HIP_CHECK(hipMalloc(&impl_->dev[d].d_g, max_rows * sizeof(float)));
    AFF_HIP_CHECK(hipMalloc(&impl_->dev[d].d_u, max_rows * sizeof(float)));
    AFF_HIP_CHECK(hipMalloc(&impl_->dev[d].d_a, max_rows * sizeof(float)));
    AFF_HIP_CHECK(hipMalloc(&impl_->dev[d].d_p, max_rows * sizeof(float)));
    AFF_HIP_CHECK(hipHostMalloc(&impl_->dev[d].h_y, max_rows * sizeof(float),
                                hipHostMallocDefault));
    for (auto& s : impl_->dev[d].stg) {
      AFF_HIP_CHECK(hipHostMalloc(&s.h, max_cols * sizeof(float), hipHostMallocDefault));
      AFF_HIP_CHECK(hipEventCreateWithFlags(&s.ev, hipEventDisableTiming));
    }
    // hipEventDisableTiming: nothing asks these for elapsed time, and the timing variant costs a
    // host-visible timestamp write on every record, twice a layer a rank.
    //
    // ONE side stream serves both users. Within a layer the KV commit is issued, joined by
    // attn_run, and only then does shexp queue behind it, so sstream's FIFO order is program order
    // and neither user waits on the other.
    AFF_HIP_CHECK(hipStreamCreate(&impl_->dev[d].sstream));
    AFF_HIP_CHECK(hipEventCreateWithFlags(&impl_->dev[d].ev_sx, hipEventDisableTiming));
    AFF_HIP_CHECK(hipEventCreateWithFlags(&impl_->dev[d].ev_sd, hipEventDisableTiming));
  }
  st_.ranks = use;

  // The all-reduce runs on the ENGINE's streams: the reduction consumes what the previous kernel
  // on that stream produced, and a separate stream would need an event pair per call — 86 a token
  // to express an ordering the stream already had. batch_init grows it for a wider chunk.
  std::string terr;
  if (!tp_init(&impl_->tp, max_cols, &terr, impl_->streams(), impl_->dev.size(), plan.data())) {
    if (err) *err = "tp_init: " + terr;
    return false;
  }
  return true;
}

// n floats on each rank, summed, left on both. Enqueued on the ranks' own streams and NOT waited
// on: the caller's next kernel on that stream is already ordered behind it, and the one place a
// result has to reach the host synchronises there anyway.
void DenseGpu::reduce(float* const in[kMaxRanks], float* const out[kMaxRanks], uint64_t n) {
  AFF_TRACE(19);
  if (!impl_ || !impl_->tp.ready || !n) return;
  tp_all_reduce(impl_->tp, in, out, (size_t)n);
}

Model::DenseOps DenseGpu::ops() {
  Model::DenseOps o;
  o.ctx = this;
  o.reg = &DenseGpu::reg_cb;
  o.mv  = &DenseGpu::mv_cb;
  o.mvg = &DenseGpu::mvg_cb;
  o.mv_multi = &DenseGpu::mv_multi_cb;
  o.attn_pre = &DenseGpu::attn_pre_cb;
  o.comp_collect = &DenseGpu::comp_collect_cb;
  o.shexp = &DenseGpu::shexp_cb;
  o.shexp_collect = &DenseGpu::shexp_collect_cb;
  o.reg_vec = &DenseGpu::reg_vec_cb;
  o.reg_router = &DenseGpu::reg_router_cb;
  o.router = &DenseGpu::router_cb;
  o.attn_q = &DenseGpu::attn_q_cb;
  o.attn_kv = &DenseGpu::attn_kv_cb;
  o.kv_sinks = &DenseGpu::kv_sinks_cb;
  o.kv_commit = &DenseGpu::kv_commit_cb;
  o.compress_one = [](void* c, const Model::CompressArgs& a) {
    return static_cast<DenseGpu*>(c)->compress_one(a);
  };
  o.set_kv_owner = [](void* c, const uint32_t* owner, uint32_t n) {
    static_cast<DenseGpu*>(c)->set_kv_owner(owner, n);
  };
  o.set_shift_pre = [](void* c, bool v) { static_cast<DenseGpu*>(c)->set_shift_pre(v); };
  o.indexer_one = [](void* c, uint32_t l, int32_t h, uint64_t qr, uint32_t nh, uint32_t dim,
                     uint32_t nk, uint32_t nm, uint32_t topk, uint32_t pos, uint32_t nr,
                     RopeDerived rope) {
    return static_cast<DenseGpu*>(c)->indexer_one(l, h, qr, nh, dim, nk, nm, topk, pos, nr, rope);
  };
  // Called at the top of every layer, which is also where last layer's admissions are forgotten:
  // a token whose indexer stopped running would otherwise inherit them.
  o.want_qrn_dev = [](void* c, bool v) { static_cast<DenseGpu*>(c)->set_idx_want_qrn(v); };
  o.attend = &DenseGpu::attend_cb;
  o.attn_out = &DenseGpu::attn_out_cb;
  o.hc_ready = &DenseGpu::hc_ready_cb;
  o.hc_seed = &DenseGpu::hc_seed_cb;
  o.hc_pre = &DenseGpu::hc_pre_cb;
  o.norm_collect = &DenseGpu::norm_collect_cb;
  o.hc_post = &DenseGpu::hc_post_cb;
  o.block_add = &DenseGpu::block_add_cb;
  o.block_reduce = &DenseGpu::block_reduce_cb;
  o.fused_ok = &DenseGpu::fused_ok_cb;
  o.hc_read = &DenseGpu::hc_read_cb;
  return o;
}

int32_t DenseGpu::reg_cb(void* ctx, const char* n, const void* h, const uint8_t* s, uint32_t q,
                         uint64_t r, uint64_t c) {
  return static_cast<DenseGpu*>(ctx)->reg(n, h, s, q, r, c);
}
void DenseGpu::mv_cb(void* ctx, int32_t h, const float* x, uint64_t r, uint64_t c, float* y) {
  static_cast<DenseGpu*>(ctx)->mv(h, x, r, c, y);
}
void DenseGpu::mvg_cb(void* ctx, int32_t h, const float* x, uint32_t g, uint64_t gd, uint64_t rk,
                      float* y) {
  static_cast<DenseGpu*>(ctx)->mvg(h, x, g, gd, rk, y);
}
bool DenseGpu::mv_multi_cb(void* ctx, const int32_t* h, const uint64_t* rows, uint64_t cols,
                           uint32_t n, const float* x, float* const* y) {
  return static_cast<DenseGpu*>(ctx)->mv_multi(h, rows, cols, n, x, y);
}
bool DenseGpu::attn_pre_cb(void* ctx, int32_t qa, int32_t kv, uint64_t qr, uint32_t width,
                           const int32_t* h, const uint64_t* rows, uint64_t cols, uint32_t n,
                           float* const* y) {
  return static_cast<DenseGpu*>(ctx)->attn_pre(qa, kv, qr, width, h, rows, cols, n, y);
}
bool DenseGpu::comp_collect_cb(void* ctx) {
  return static_cast<DenseGpu*>(ctx)->comp_collect();
}
bool DenseGpu::shexp_cb(void* ctx, int32_t g, int32_t u, int32_t d, const float* x, uint64_t hid,
                        uint64_t inter, float lim, float* out, bool defer) {
  return static_cast<DenseGpu*>(ctx)->shexp(g, u, d, x, hid, inter, lim, out, defer);
}
bool DenseGpu::shexp_collect_cb(void* ctx, float* out, uint64_t hidden) {
  return static_cast<DenseGpu*>(ctx)->shexp_collect(out, hidden);
}

// Which fp8 tensors are stored in WMMA fragment order rather than row-major.
//
// An ALLOWLIST, and that direction is the safety argument: a tensor nobody thought about stays
// row-major and merely misses the speedup, whereas a missed exclusion is a wrong answer that
// decodes fluently. These seven are exactly the fp8 tensors whose only reader is the narrow W8A8
// GEMM. The two left out are attn_out_a (q8_mode leaves it on the bf16 grouped kernel, which reads
// rows, because W8A8 loses there) and idx_wq_b (its own kernels in indexer_gpu.hip).
//
// Everything that reads these bytes as rows aborts rather than proceeding; see `shard_rows`.
bool DenseGpu::ps_for(const char* name) {
  if (!name) return false;
  const char* s = std::strrchr(name, '.');
  s = s ? s + 1 : name;
  // The one exception, and it is a second reader rather than a second layout: a draft stage's wkv
  // is read by `batch_attn_qkv` like the target's AND by `draft_kv`, which goes through the bf16
  // kernel. The cost is that the draft's three stages lose the wq_a+wkv fusion — three launches a
  // block — against putting draft_kv on W8A8, which moves the emitted text. See draft_kv.
  if (!std::strcmp(s, "attn_kv") && !std::strncmp(name, "mtp.", 4)) return false;
  for (const char* w : {"attn_q_a", "attn_q_b", "attn_kv", "attn_out_b", "shexp_gate", "shexp_up",
                        "shexp_down"})
    if (!std::strcmp(s, w)) return true;
  return false;
}

// How a tensor is cut, from the name the loader binds it under: the loader has no business knowing
// about ranks, and these lines ARE the tensor-parallel plan. Anything unnamed is Row, which is both
// the safe default and the right one for every matrix whose result goes back to the host.
DenseGpu::Split DenseGpu::split_for(const char* name) {
  if (!name) return Split::Row;
  const char* s = std::strrchr(name, '.');
  s = s ? s + 1 : name;                               // "blk.17.attn_q_a" -> "attn_q_a"
  // Second matrix of a chain: consumes the first's shard over the INPUT dimension and yields a
  // partial the sublayer's one all-reduce completes.
  if (!std::strcmp(s, "attn_out_b") || !std::strcmp(s, "shexp_down")) return Split::Col;
  // Small, and both cards need the whole result: wq_a feeds an RMS over all q_lora_rank rows, wkv
  // feeds a cache that is replicated anyway (MQA). Splitting would trade less reading for more
  // exchange than it saves.
  if (!std::strcmp(s, "attn_q_a") || !std::strcmp(s, "attn_kv")) return Split::Rep;
  // strstr, not strncmp: the draft's head gate is `mtp_hc_head_fn`. Row-split it is 4 rows across
  // two cards, so rank 1 would get none and read whatever was in its shard — silently, since a
  // 0-row shard is legal.
  if (std::strstr(s, "hc_") && std::strstr(s, "_fn")) return Split::Rep;
  // The indexer needs WHOLE queries, and attention splits by head rather than by token, so both
  // cards want the same admission mask. Row-split, the result has to be drained to the host,
  // reassembled and uploaded again, every layer of every chunk.
  if (!std::strcmp(s, "idx_wq_b")) return Split::Rep;
  // The compressors, one step further on: their pool needs a whole projection row (a softmax over
  // 2*ratio candidates of one dimension) and emits into a replicated cache. ~6 MB a layer, so the
  // duplicate is cheaper than an exchange per layer.
  if (!std::strcmp(s, "comp_wkv") || !std::strcmp(s, "comp_wgate")) return Split::Rep;
  if (!std::strcmp(s, "idx_comp_wkv") || !std::strcmp(s, "idx_comp_wgate")) return Split::Rep;
  // 64 rows. The indexer weights every head by it and every card scores every head-group.
  if (!std::strcmp(s, "idx_proj")) return Split::Rep;
  // ---- DSpark ---------------------------------------------------------------------------------
  // main_proj is the one activation the whole draft is conditioned on and every stage reads all of
  // it, so a Row split needs a gather before stage one. 50 MB read twice beats 25 MB exchanged.
  if (!std::strcmp(s, "mtp_main_proj")) return Split::Rep;
  // The Markov correction. w1 is an embedding — one row chosen at run time — so Row-split only one
  // card would hold it. w2 is a [vocab][rank] head whose band must match the vocabulary head's, and
  // those are different tensors with no reason to agree on a boundary; replicated, each card slices
  // w2 by the head's own band. 66 MB a card each.
  if (!std::strcmp(s, "mtp_markov_w1") || !std::strcmp(s, "mtp_markov_w2")) return Split::Rep;
  return Split::Row;
}

int32_t DenseGpu::reg(const char* name, const void* host, const uint8_t* scales, uint32_t quant,
                      uint64_t rows, uint64_t cols) {
  if (!impl_ || !host || (quant != AFF_BF16 && quant != AFF_FP8_E4M3)) return -1;
  const auto t0 = std::chrono::steady_clock::now();
  const size_t R = impl_->dev.size();
  const uint64_t elem = (quant == AFF_FP8_E4M3) ? 1 : 2;
  const uint64_t sc_cols = (cols + 127) / 128;

  Entry e;
  e.quant = quant; e.rows = rows; e.cols = cols;
  e.split = split_for(name);

  // FP8 carries one UE8M0 scale per 128x128 tile, so a split of either axis has to land on a tile
  // boundary or a shard reads a neighbour's scales. A Col split that cannot align is demoted to
  // Rep rather than refused: replication is always arithmetically correct, only redundant.
  if (e.split == Split::Col && (cols % (kRowAlign * R)) != 0) {
    aff::ui::err("dense-gpu: %s has %llu columns, which do not split %zu ways on a %llu "
                 "boundary — replicating instead\n", name, (unsigned long long)cols, R,
                 (unsigned long long)kRowAlign);
    e.split = Split::Rep;
  }

  // Row: the output dimension. Rank r owns rows [r*chunk, ...) outright.
  uint64_t chunk = (rows + R - 1) / R;
  chunk = ((chunk + kRowAlign - 1) / kRowAlign) * kRowAlign;
  const uint64_t cchunk = cols / R;                   // Col: exact, guaranteed by the check above

  uint64_t placed = 0;
  for (size_t d = 0; d < R; ++d) {
    Shard& s = e.sh[d];
    switch (e.split) {
      case Split::Row:
        s.row0 = std::min(rows, (uint64_t)d * chunk);
        s.nrows = std::min(chunk, rows - s.row0);
        s.col0 = 0; s.ncols = cols;
        break;
      case Split::Col:
        s.row0 = 0; s.nrows = rows;
        s.col0 = (uint64_t)d * cchunk; s.ncols = cchunk;
        break;
      case Split::Rep:
        s.row0 = 0; s.nrows = rows;
        s.col0 = 0; s.ncols = cols;
        break;
    }
    if (s.nrows == 0 || s.ncols == 0) continue;

    AFF_HIP_CHECK(hipSetDevice(impl_->dev[d].id));
    const uint64_t nbytes = s.nrows * s.ncols * elem;
    const uint64_t sh_sc_cols = (s.ncols + 127) / 128;
    const uint64_t sc_bytes = (quant == AFF_FP8_E4M3) ? (s.nrows / 128) * sh_sc_cols : 0;

    size_t freeb = 0, totalb = 0;
    AFF_HIP_CHECK(hipMemGetInfo(&freeb, &totalb));
    if (freeb < nbytes + sc_bytes + kAllocHeadroom) {
      aff::ui::err("dense-gpu: %s shard %zu (%.1f MiB) does not fit, tensor stays on CPU\n",
                   name, d, nbytes / 1048576.0);
      ++st_.cpu_fallbacks;
      for (size_t k = 0; k < d; ++k) {
        (void)hipSetDevice(impl_->dev[k].id);
        if (e.sh[k].data) (void)hipFree(e.sh[k].data);
        if (e.sh[k].scales) (void)hipFree(e.sh[k].scales);
      }
      return -1;
    }
    if (hipMalloc(&s.data, nbytes) != hipSuccess) return -1;
    s.data_bytes = nbytes;   // amdnas-fixes weight-verify
    if (s.ncols == cols) {
      // Row and Rep: the slice is contiguous in the container's row-major layout.
      AFF_HIP_CHECK(hipMemcpy(s.data, (const uint8_t*)host + s.row0 * cols * elem, nbytes,
                              hipMemcpyHostToDevice));
    } else {
      // Col: a column band, so `rows` separate runs. Load-time only, and hipMemcpy2D expresses the
      // stride to the driver rather than issuing `rows` separate calls.
      AFF_HIP_CHECK(hipMemcpy2D(s.data, s.ncols * elem,
                                (const uint8_t*)host + s.col0 * elem, cols * elem,
                                s.ncols * elem, s.nrows, hipMemcpyHostToDevice));
    }
    if (sc_bytes) {
      if (hipMalloc((void**)&s.scales, sc_bytes) != hipSuccess) return -1;
      s.scales_bytes = sc_bytes;   // amdnas-fixes weight-verify
      if (sh_sc_cols == sc_cols) {
        AFF_HIP_CHECK(hipMemcpy(s.scales, scales + (s.row0 / 128) * sc_cols, sc_bytes,
                                hipMemcpyHostToDevice));
      } else {
        AFF_HIP_CHECK(hipMemcpy2D(s.scales, sh_sc_cols, scales + s.col0 / 128, sc_cols,
                                  sh_sc_cols, s.nrows / 128, hipMemcpyHostToDevice));
      }
    }
    // Fragment order, in place. The W8A8 GEMM otherwise reads A through
    // `Wp -> registers -> LDS -> barrier -> LDS -> fragment`; preshuffled, a wave reads its whole A
    // fragment with one coalesced 8-byte-a-lane load.
    //
    // This is the ONLY layout, not a second copy — the one reader that needed row-major was
    // `matvec_fp8_hip` via `forward_token`, which the supported configuration never calls, since with
    // DSpark on generation is seed/draft/verify and all of those are batched.
    //
    // A Col shard's row length is its own `ncols` and is stored that way, so the permutation and the
    // GEMM both use `s.ncols` and a Col split needs nothing special.
    if (ps_for(name) && quant == AFF_FP8_E4M3 && (s.nrows % 16) == 0 && (s.ncols % 16) == 0) {
      void* tmp = nullptr;
      if (hipMalloc(&tmp, nbytes) != hipSuccess) return -1;
      AFF_HIP_CHECK(hipMemcpy(tmp, s.data, nbytes, hipMemcpyDeviceToDevice));
      preshuffle_fp8_hip((const uint8_t*)tmp, (uint8_t*)s.data, (uint32_t)s.nrows,
                         (uint32_t)s.ncols, nullptr);
      AFF_HIP_CHECK(hipDeviceSynchronize());
      AFF_HIP_CHECK(hipFree(tmp));
      s.ps = true;
    }
    placed += nbytes + sc_bytes;
  }

  ent_.push_back(e);
  ++st_.tensors;
  st_.bytes += placed;
  st_.upload_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return (int32_t)(ent_.size() - 1);
}

// Copy `n` floats from pageable host memory into this device's pinned slot, then enqueue the H2D.
// The wait on the slot's own event marks the last transfer out of it, which is what makes
// overwriting safe when the call site never synchronises; already signalled in the steady state.
static inline void stage_up(float* pin, hipEvent_t ev, void* dst, const float* src, uint64_t n,
                            hipStream_t st) {
  AFF_HIP_CHECK(hipEventSynchronize(ev));
  std::memcpy(pin, src, n * sizeof(float));
  AFF_HIP_CHECK(hipMemcpyAsync(dst, pin, n * sizeof(float), hipMemcpyHostToDevice, st));
  AFF_HIP_CHECK(hipEventRecord(ev, st));
}
#define AFF_STAGE_UP(D, slot, dst, src, n, st) \
  stage_up((D).stg[slot].h, (D).stg[slot].ev, (dst), (src), (n), (st))

namespace {
// The same refusal as `shard_rows`, for readers that hold a Shard rather than ask for its rows.
inline void no_ps(bool ps, const char* where) {
  if (!ps) return;
  aff::ui::fatal("dense-gpu: %s reads a preshuffled fp8 shard as row-major. That layout is "
               "readable only by the W8A8 GEMM, and this is the single-token path DSpark replaces.\n",
               where);
  std::abort();
}

struct ShardRef { const void* data; const uint8_t* scales; uint32_t quant; bool ps = false; };
inline void launch_mv(const ShardRef& s, const float* dx, uint32_t rows, uint32_t cols, float* dy,
                      hipStream_t st) {
  no_ps(s.ps, "matvec");
  if (s.quant == AFF_FP8_E4M3)
    matvec_fp8_hip((const uint8_t*)s.data, s.scales, dx, rows, cols, dy, st);
  else
    matvec_bf16_hip((const uint16_t*)s.data, dx, rows, cols, dy, st);
}
}

int32_t DenseGpu::reg_vec_cb(void* ctx, const float* h, uint64_t n) {
  return static_cast<DenseGpu*>(ctx)->reg_vec(h, n);
}
int32_t DenseGpu::reg_router_cb(void* ctx, const uint16_t* w, const float* bias, uint32_t ne,
                                uint64_t nd) {
  return static_cast<DenseGpu*>(ctx)->reg_router(w, bias, ne, nd);
}
bool DenseGpu::router_cb(void* ctx, int32_t h, const float* x, uint32_t ne, uint64_t nd,
                         float* logits) {
  return static_cast<DenseGpu*>(ctx)->router(h, x, ne, nd, logits);
}
bool DenseGpu::attn_q_cb(void* ctx, int32_t qa, int32_t qb, int32_t qn, const float* x,
                         uint64_t hid, uint64_t qr, uint32_t nh, uint32_t w, uint32_t nr,
                         uint32_t pos, RopeDerived rope, float eps, float* q, float* qn_out) {
  return static_cast<DenseGpu*>(ctx)->attn_q(qa, qb, qn, x, hid, qr, nh, w, nr, pos, rope, eps,
                                             q, qn_out);
}
bool DenseGpu::attn_kv_cb(void* ctx, int32_t kv, int32_t kvn, const float* x, uint64_t hid,
                          uint32_t w, uint32_t nr, uint32_t pos, RopeDerived rope, float eps,
                          float* out) {
  return static_cast<DenseGpu*>(ctx)->attn_kv(kv, kvn, x, hid, w, nr, pos, rope, eps, out);
}

void DenseGpu::kv_sinks_cb(void* c, uint32_t l, const float* s, uint32_t n) {
  static_cast<DenseGpu*>(c)->attn_set_sinks(l, s, n);
}
bool DenseGpu::kv_commit_cb(void* c, uint32_t l, uint32_t pos, uint32_t w, uint32_t nr) {
  return static_cast<DenseGpu*>(c)->kv_commit(l, pos, w, nr);
}
bool DenseGpu::attend_cb(void* c, uint32_t l, const float* q, const uint8_t* a, uint32_t pos,
                         uint32_t nc, float sc, float* h) {
  return static_cast<DenseGpu*>(c)->attn_run(l, q, a, pos, nc, sc, h);
}

bool DenseGpu::hc_ready_cb(void* c) { return static_cast<DenseGpu*>(c)->hc_ready(); }
void DenseGpu::hc_seed_cb(void* c, const float* e, uint32_t n, uint32_t h) {
  static_cast<DenseGpu*>(c)->hc_seed(e, n, h);
}
bool DenseGpu::hc_pre_cb(void* c, int32_t fn, int32_t sv, int32_t bv, int32_t nw, uint32_t n,
                         uint32_t h, uint32_t it, float he, float re, float* nh, bool defer) {
  return static_cast<DenseGpu*>(c)->hc_pre(fn, sv, bv, nw, n, h, it, he, re, nh, defer);
}
bool DenseGpu::norm_collect_cb(void* c, float* nh, uint32_t n) {
  return static_cast<DenseGpu*>(c)->norm_collect(nh, n);
}
bool DenseGpu::hc_post_cb(void* c, bool hasc, uint32_t n, uint32_t h) {
  return static_cast<DenseGpu*>(c)->hc_post(hasc, n, h);
}
bool DenseGpu::block_add_cb(void* c, const float* src, uint32_t n) {
  return static_cast<DenseGpu*>(c)->block_add_host(src, n);
}
// Whether the device-resident path is live at all — the caller has to know before it leaves
// attention's heads in VRAM.
bool DenseGpu::fused_ok_cb(void* c) {
  DenseGpu* g = static_cast<DenseGpu*>(c);
  return g->hc_ok_ && g->attn_ok_;
}
bool DenseGpu::block_reduce_cb(void* c, uint32_t n) {
  return static_cast<DenseGpu*>(c)->block_reduce(n);
}
void DenseGpu::hc_read_cb(void* c, float* dst, uint32_t n, uint32_t h) {
  static_cast<DenseGpu*>(c)->hc_read(dst, n, h);
}
bool DenseGpu::attn_out_cb(void* c, int32_t wa, int32_t wb, uint32_t nh, uint32_t w, uint32_t nr,
                           uint32_t pos, RopeDerived rope, uint32_t ng, uint64_t rank,
                           uint64_t hid, float* blk) {
  return static_cast<DenseGpu*>(c)->attn_out(wa, wb, nh, w, nr, pos, rope, ng, rank, hid, blk);
}

// Replicated. These are norm weights and hyper-connection scales — a few KiB each, read by kernels
// that run identically on both cards.
int32_t DenseGpu::reg_vec(const float* host, uint64_t n) {
  if (!impl_ || !host || impl_->dev.empty()) return -1;
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    AFF_HIP_CHECK(hipSetDevice(impl_->dev[d].id));
    float* p = nullptr;
    if (hipMalloc(&p, n * sizeof(float)) != hipSuccess) return -1;
    AFF_HIP_CHECK(hipMemcpy(p, host, n * sizeof(float), hipMemcpyHostToDevice));
    impl_->dev[d].vec.push_back(p);
  }
  return (int32_t)(n_vec_++);
}

void DenseGpu::row_band(const Shard& s, uint64_t want, uint64_t* r0, uint64_t* nr) {
  *r0 = std::min(want, s.row0);
  *nr = std::min(s.nrows, want > *r0 ? want - *r0 : 0);
}
// Every reader that wants row-major bytes comes through here, so the check lives here rather than
// at fifteen call sites. A preshuffled shard read as rows is neither a crash nor garbage — it is a
// permuted-but-plausible activation that decodes fluently, so only the hash would move.
const void* DenseGpu::shard_rows(const Shard& s, uint32_t quant, uint64_t local) {
  if (s.ps) {
    aff::ui::fatal("dense-gpu: a preshuffled fp8 shard was asked for row-major rows. That "
                 "layout is readable only by the W8A8 GEMM, and this path is the single-token one "
                 "that DSpark replaces.\n");
    std::abort();
  }
  const uint64_t elem = (quant == AFF_FP8_E4M3) ? 1 : 2;
  return (const uint8_t*)s.data + local * s.ncols * elem;
}
const uint8_t* DenseGpu::shard_scales(const Shard& s, uint64_t local) {
  return s.scales ? s.scales + (local / 128) * ((s.ncols + 127) / 128) : nullptr;
}

bool DenseGpu::mv_multi(const int32_t* h, const uint64_t* rows, uint64_t cols, uint32_t n,
                        const float* x, float* const* y) {
  AFF_TRACE(0);
  if (!impl_ || impl_->dev.empty() || !n) return false;
  const size_t R = impl_->dev.size();
  for (uint32_t i = 0; i < n; ++i) {
    if (h[i] < 0 || (size_t)h[i] >= ent_.size() || !y[i]) return false;
    const Entry& e = ent_[h[i]];
    // Same x for all of them, so they must agree on width. Col has no host-side concatenation and
    // nothing routed through here is Col; refuse rather than silently return a partial.
    if (e.cols != cols || e.split == Split::Col) return false;
  }

  uint64_t r0[kMaxRanks][kMvBatchMax], nr[kMaxRanks][kMvBatchMax], total[kMaxRanks] = {0};
  if (n > kMvBatchMax) return false;
  for (size_t d = 0; d < R; ++d) {
    for (uint32_t i = 0; i < n; ++i) {
      row_band(ent_[h[i]].sh[d], rows[i], &r0[d][i], &nr[d][i]);
      total[d] += nr[d][i];
    }
    if (total[d] > impl_->dev[d].cap_mid) return false;
  }

  // Issue both cards before waiting on either.
  for (size_t d = 0; d < R; ++d) {
    Impl::Dev& D = impl_->dev[d];
    if (!total[d]) continue;
    AFF_HIP_CHECK(hipSetDevice(D.id));
    const float* xin = x ? D.d_x : D.d_norm;
    if (x) AFF_STAGE_UP(D, kStgMv, D.d_x, x, cols, D.stream);
    // ONE launch for all of them: they already shared a round trip, but not the grid. Issued
    // separately, `idx_comp_wkv`'s 256 rows ask for 32 blocks on a 64-CU card while `comp_wkv`'s
    // 1024 ask for 128, so each ran on a fraction of the machine. Batched, the grid is sized from
    // the rows they add up to.
    MvJob jobs[kMvBatchMax];
    uint32_t nj = 0;
    uint64_t off = 0;
    for (uint32_t i = 0; i < n; ++i) {
      if (!nr[d][i]) continue;
      const Entry& e = ent_[h[i]];
      const uint64_t local = r0[d][i] - e.sh[d].row0;
      jobs[nj++] = MvJob{shard_rows(e.sh[d], e.quant, local), shard_scales(e.sh[d], local),
                         e.quant, (uint32_t)nr[d][i], D.d_y + off};
      off += nr[d][i];
    }
    matvec_batch_hip(jobs, nj, xin, (uint32_t)cols, D.stream);
    // One copy back for all of them: the outputs were laid out contiguously on device for exactly
    // this reason.
    AFF_HIP_CHECK(hipMemcpyAsync(D.h_y, D.d_y, total[d] * sizeof(float), hipMemcpyDeviceToHost,
                                 D.stream));
  }
  for (size_t d = 0; d < R; ++d) {
    Impl::Dev& D = impl_->dev[d];
    if (!total[d]) continue;
    AFF_HIP_CHECK(hipSetDevice(D.id));
    AFF_DRAIN(D);
    uint64_t off = 0;
    for (uint32_t i = 0; i < n; ++i) {
      if (!nr[d][i]) continue;
      std::memcpy(y[i] + r0[d][i], D.h_y + off, nr[d][i] * sizeof(float));
      off += nr[d][i];
    }
  }
  st_.calls += n;
  return true;
}

// A SECOND STREAM OVERLAPS EXECUTION, NOT DISPATCH, which is why kv_commit stays on the main stream:
// an extra small launch costs the same on either stream, so moving one there measures flat. The
// shared expert's side stream pays only because shexp is bandwidth work with an idle memory pipeline
// beside it. Do not try it on `router`, `reduce`, `hc_pre`, `hc_post` or `comp_coll` — those are
// dispatch-bound, and the lever there is fewer launches, not more streams.

bool DenseGpu::attn_pre(int32_t qa, int32_t kv, uint64_t qr, uint32_t width, const int32_t* h,
                        const uint64_t* rows, uint64_t cols, uint32_t n, float* const* y) {
  AFF_TRACE(1);
  if (!impl_ || impl_->dev.empty()) return false;
  // `norm` must already be in VRAM: this is the one entry point with no host-x fallback, because
  // the whole point is that the batch reads the device copy exactly once for six matrices.
  if (!impl_->dev[0].d_norm || pre_q_ || pre_kv_ || pre_n_) return false;
  if (qa < 0 || kv < 0 || (size_t)qa >= ent_.size() || (size_t)kv >= ent_.size()) return false;
  if (n + 2 > kMvBatchMax) return false;
  const size_t R = impl_->dev.size();

  // Every check attn_q and attn_kv make about their FIRST matvec, made here. If one of them would
  // have refused, the caller must be left on a path where it still owns the whole chain. Both are
  // Rep, so the projections land whole at the same two offsets on both cards and attn_q/attn_kv
  // find them there without knowing anything about ranks.
  const Entry& A = ent_[qa];
  const Entry& K = ent_[kv];
  if (A.cols != cols || A.split != Split::Rep || A.rows < qr) return false;
  if (K.cols != cols || K.split != Split::Rep || K.rows < width) return false;
  for (uint32_t i = 0; i < n; ++i) {
    if (h[i] < 0 || (size_t)h[i] >= ent_.size() || !y[i]) return false;
    const Entry& e = ent_[h[i]];
    if (e.cols != cols || e.split == Split::Col) return false;
  }
  for (size_t d = 0; d < R; ++d) {
    uint64_t total = qr + width;
    for (uint32_t i = 0; i < n; ++i) {
      row_band(ent_[h[i]].sh[d], rows[i], &pre_r0_[d][i], &pre_nr_[d][i]);
      total += pre_nr_[d][i];
    }
    if (total > impl_->dev[d].cap_mid) return false;
    pre_total_[d] = total - (qr + width);
  }

  for (size_t d = 0; d < R; ++d) {
    Impl::Dev& D = impl_->dev[d];
    AFF_HIP_CHECK(hipSetDevice(D.id));

    MvJob jobs[kMvBatchMax];
    no_ps(A.sh[d].ps || K.sh[d].ps, "the fused single-token wq_a/wkv matvec");
    jobs[0] = MvJob{A.sh[d].data, A.sh[d].scales, A.quant, (uint32_t)qr, D.d_y};
    jobs[1] = MvJob{K.sh[d].data, K.sh[d].scales, K.quant, width, D.d_y + qr};
    uint32_t nj = 2;
    uint64_t off = qr + width;
    for (uint32_t i = 0; i < n; ++i) {
      if (!pre_nr_[d][i]) continue;
      const Entry& e = ent_[h[i]];
      const uint64_t local = pre_r0_[d][i] - e.sh[d].row0;
      jobs[nj++] = MvJob{shard_rows(e.sh[d], e.quant, local), shard_scales(e.sh[d], local),
                         e.quant, (uint32_t)pre_nr_[d][i], D.d_y + off};
      off += pre_nr_[d][i];
    }
    matvec_batch_hip(jobs, nj, D.d_norm, (uint32_t)cols, D.stream);

    // The compressor half goes to the host now and is WAITED FOR LATER. Between here and
    // comp_collect the caller issues the rest of both attention chains, and every one of those
    // launches is queued behind this copy on the same stream, so the wait costs nothing that was
    // not going to be spent anyway.
    if (pre_total_[d])
      AFF_HIP_CHECK(hipMemcpyAsync(D.h_y, D.d_y + qr + width, pre_total_[d] * sizeof(float),
                                   hipMemcpyDeviceToHost, D.stream));
  }
  pre_q_ = true; pre_kv_ = true;
  pre_qr_ = qr; pre_width_ = width;
  pre_n_ = n;
  for (uint32_t i = 0; i < n; ++i) pre_y_[i] = y[i];
  st_.calls += n + 2;
  return true;
}

bool DenseGpu::comp_collect() {
  if (!impl_ || impl_->dev.empty()) return false;
  // attn_q and attn_kv must both have taken their projection by now. If either did not, the batch
  // computed an operand nobody consumed and the chain that skipped it ran on stale scratch — silent
  // wrong output, so refuse rather than paper over it.
  if (pre_q_ || pre_kv_) {
    aff::ui::fatal("fatal: attn_pre's projections were never consumed (q=%d kv=%d)\n",
                 (int)pre_q_, (int)pre_kv_);
    std::abort();
  }
  const uint32_t n = pre_n_;
  pre_n_ = 0;
  if (!n) return true;
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    if (!pre_total_[d]) continue;
    AFF_HIP_CHECK(hipSetDevice(D.id));
    AFF_DRAIN(D);
    uint64_t off = 0;
    for (uint32_t i = 0; i < n; ++i) {
      if (!pre_nr_[d][i]) continue;
      std::memcpy(pre_y_[i] + pre_r0_[d][i], D.h_y + off, pre_nr_[d][i] * sizeof(float));
      off += pre_nr_[d][i];
    }
  }
  return true;
}

// Replicated allocation, row-split decode matvec — two decisions, both deliberate. Decode's logits
// go to the host for top-k either way, so a row split needs no exchange and each card writes its own
// half rather than leaving one idle. Prefill's top-k runs on device over all the logits at once and
// needs the whole gate in one place, and the gate is small enough to replicate. So `router_rows` is
// a compute split, not an allocation one.

int32_t DenseGpu::reg_router(const uint16_t* w, const float* bias, uint32_t n_expert,
                             uint64_t n_embd) {
  if (!impl_ || !w || impl_->dev.empty()) return -1;
  const size_t R = impl_->dev.size();
  const size_t parts = R;                             // one band of the gate a card
  for (size_t d = 0; d < R; ++d) {
    Impl::Dev& D = impl_->dev[d];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    const uint64_t per = ((uint64_t)n_expert + parts - 1) / parts;
    const uint64_t r0  = std::min<uint64_t>((uint64_t)d * per, n_expert);
    const uint64_t nr  = std::min<uint64_t>(per, (uint64_t)n_expert - r0);
    const uint64_t nbytes = (uint64_t)n_expert * n_embd * sizeof(uint16_t);  // bf16, half of f32
    // Refuse rather than displace experts if the card is tight. The host path is correct and only
    // slower, so a silent CPU fallback here is a degradation, not a failure.
    size_t freeb = 0, totalb = 0;
    AFF_HIP_CHECK(hipMemGetInfo(&freeb, &totalb));
    if (freeb < nbytes + kAllocHeadroom) return -1;
    uint16_t* p = nullptr;
    if (nbytes && hipMalloc(&p, nbytes) != hipSuccess) return -1;
    if (nbytes) AFF_HIP_CHECK(hipMemcpy(p, w, nbytes, hipMemcpyHostToDevice));
    if (!D.d_rlog && hipMalloc(&D.d_rlog, n_expert * sizeof(float)) != hipSuccess) {
      (void)hipFree(p);
      return -1;
    }
    float* pb = nullptr;
    if (bias) {
      if (hipMalloc(&pb, (size_t)n_expert * sizeof(float)) != hipSuccess) {
        (void)hipFree(p);
        return -1;
      }
      AFF_HIP_CHECK(hipMemcpy(pb, bias, (size_t)n_expert * sizeof(float), hipMemcpyHostToDevice));
    }
    D.router.push_back(p);
    D.rbias.push_back(pb);
    D.router_row0 = r0;
    D.router_rows = nr;
    st_.bytes += nbytes;
  }
  ++st_.tensors;
  return (int32_t)(n_router_++);
}

bool DenseGpu::router(int32_t h, const float* x, uint32_t n_expert, uint64_t n_embd,
                      float* logits) {
  AFF_TRACE(12);
  if (h < 0 || (size_t)h >= n_router_ || !logits) return false;
  const size_t R = impl_->dev.size();
  for (size_t d = 0; d < R; ++d) {
    Impl::Dev& D = impl_->dev[d];
    if (!D.d_rlog || !D.router_rows) continue;
    AFF_HIP_CHECK(hipSetDevice(D.id));
    // x == null: the norm is already in VRAM from hc_pre, so nothing crosses the bus on the way in
    // and only 512 B of logits comes back from each card.
    const float* xin = x ? D.d_x : D.d_norm;
    if (x) AFF_STAGE_UP(D, kStgMv, D.d_x, x, n_embd, D.stream);
    matvec_bf16_hip(D.router[h] + D.router_row0 * n_embd, xin, (uint32_t)D.router_rows,
                    (uint32_t)n_embd, D.d_rlog, D.stream);
    AFF_HIP_CHECK(hipMemcpyAsync(D.h_y, D.d_rlog, D.router_rows * sizeof(float),
                                 hipMemcpyDeviceToHost, D.stream));
  }
  // This drain exists because the host cannot dispatch routed experts until it knows which six won.
  for (size_t d = 0; d < R; ++d) {
    Impl::Dev& D = impl_->dev[d];
    if (!D.d_rlog || !D.router_rows) continue;
    AFF_HIP_CHECK(hipSetDevice(D.id));
    AFF_DRAIN(D);
    std::memcpy(logits + D.router_row0, D.h_y, D.router_rows * sizeof(float));
  }
  ++st_.calls;
  return true;
}

// The device router need not use the drained path's matvec kernel, only its arithmetic to within a
// rounding. It computes every row and the drained path computes half of them, so the two land on
// different row-splitting kernels and reduce in different orders. The selected SET agrees; only the
// renormalised weight differs by an ulp or two, from sqrtsoftplus going through the GPU's expf and
// log1pf rather than libm. Forcing the two onto one kernel costs more than that gap is worth.

namespace {

// ---- hyper-connections -------------------------------------------------------------------------
//
// The arithmetic here is trivial; the point is location. The 4-lane residual state IS the model's
// hidden state, and while it lived on the host every layer dragged it back across the bus. Moving it
// to the device is what lets a whole sublayer stay device-resident.

// The gates are computed one lane per matrix element and with no barriers, which needs the lane
// layout to be a power-of-two square inside one wave: comb[l] with l = src + dst * n_hc puts the src
// bits low and the dst bits directly above, so a sum over either axis is a butterfly over a
// contiguous bit field. The alternative runs the Sinkhorn sweeps on one thread while the rest of the
// block waits.
__device__ __host__ inline bool hc_gates_wave_ok(uint32_t n_hc) {
  return n_hc >= 1 && (n_hc & (n_hc - 1)) == 0 && n_hc * n_hc <= 32;
}
// Block width for every hc kernel. 256 is what makes the split-K arithmetic below exact.
constexpr uint32_t kHcThreads = 256;
// Split-K factor for hc_mixgram_kernel: 24 rows x 8 = 192 blocks, and 16384/8 = 2048 columns a
// block is exactly one uint4 a lane at 256 threads.
constexpr uint32_t kHcSplit = 8;

// Upper triangle of the symmetric n_hc x n_hc Gram, enumerated (a,b) with a<=b. NT floats is the
// entire dependency everything downstream has on the whole of hc.
template <uint32_t NH>
static constexpr uint32_t kHcTri = NH * (NH + 1) / 2;

// The gates split in two: neither half reads the other, since the sweeps produce `comb` alone and
// the map (cur, norm) needs only `pre` and the RMS scalar. Running them together parks the whole
// grid on one wave computing a value hc_post does not read until a sublayer later.
template <uint32_t NH>
__device__ inline void hc_gates_pre(const float* __restrict__ mix, float mix_scale,
                                    const float* __restrict__ scale,
                                    const float* __restrict__ base, float* pre, float* post,
                                    float hc_eps, uint32_t lane) {
  if (lane < NH) {
    const float z = mix[lane] * mix_scale * scale[0] + base[lane];
    pre[lane] = 1.0f / (1.0f + __expf(-z)) + hc_eps;
    const uint32_t o = NH + lane;
    const float z2 = mix[o] * mix_scale * scale[1] + base[o];
    post[lane] = 2.0f / (1.0f + __expf(-z2));
  }
}

// The Sinkhorn's shuffles go through DPP row_xmask rather than ds_bpermute. `__shfl_xor` lowers to
// ds_bpermute_b32 — an LDS round trip with a full wait after it — and this is ONE wave with nothing
// to interleave, so all of that is dependent latency.
//
// BIT-IDENTICAL, not an approximation: DPP16's xor is over a row of 16 lanes and the reduction is
// over lanes 0-15, exactly one row, so every lane reads the same source lane and merely skips LDS.
// Replacing the two IEEE divides with v_rcp_f32 would be shorter still, but it is about an ulp off,
// and a perturbation that size walks into a repetition loop a few tokens later.
template <int CTRL>
__device__ inline float hc_dpp_xor(float x) {
  return __builtin_bit_cast(
      float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x), CTRL, 0xf, 0xf, true));
}

// DPP's control field is an immediate, so the mask must reach it as a compile-time constant. The
// sweeps below unroll fully over a template NH, so `m` is constant in each copy and this folds to
// one instruction. A switch rather than a template parameter so the loop form survives.
__device__ inline float hc_sk_xor(float v, uint32_t m) {
  switch (m) {
    case 1: return hc_dpp_xor<0x161>(v);
    case 2: return hc_dpp_xor<0x162>(v);
    case 4: return hc_dpp_xor<0x164>(v);
    case 8: return hc_dpp_xor<0x168>(v);
    default: return __shfl_xor(v, (int)m);           // >16 lanes apart: not a DPP16 row
  }
}

// The Sinkhorn alone, writing wherever it is told — LDS when a later __syncthreads publishes it,
// global when nothing in the block waits. Every lane must reach the shuffles, including those past
// NH*NH: a shuffle is wave-wide and guarding it is undefined. Those lanes carry -1e30 and never
// contribute, because an xor within the low 2*log2(NH) bits keeps a valid lane's partner valid.
template <uint32_t NH>
__device__ inline void hc_sinkhorn_only(const float* __restrict__ mix, float mix_scale,
                                        const float* __restrict__ scale,
                                        const float* __restrict__ base, float* comb,
                                        uint32_t iters, float hc_eps, uint32_t lane) {
  constexpr uint32_t nn = NH * NH;
  // row_xmask is DPP16 — equivalent to __shfl_xor only while the mask stays inside a 16-lane row.
  static_assert(nn <= 16, "DPP row_xmask is a 16-lane row; NH*NH must fit in one");
  float c = lane < nn ? mix[2 * NH + lane] * mix_scale * scale[2] + base[2 * NH + lane] : -1e30f;
  // Softmax over src per dst, then columns once. Not the loop body: starting with a column sweep
  // converges somewhere else.
  float m = c;
#pragma unroll
  for (uint32_t o = 1; o < NH; o <<= 1) m = fmaxf(m, hc_sk_xor(m, o));
  c = __expf(c - m);
  float s = c;
#pragma unroll
  for (uint32_t o = 1; o < NH; o <<= 1) s += hc_sk_xor(s, o);
  c = c / s + hc_eps;
  s = c;
#pragma unroll
  for (uint32_t o = NH; o < nn; o <<= 1) s += hc_sk_xor(s, o);
  c *= 1.0f / (s + hc_eps);
  for (uint32_t it = 1; it < iters; ++it) {
    s = c;
#pragma unroll
    for (uint32_t o = 1; o < NH; o <<= 1) s += hc_sk_xor(s, o);
    c *= 1.0f / (s + hc_eps);
    s = c;
#pragma unroll
    for (uint32_t o = NH; o < nn; o <<= 1) s += hc_sk_xor(s, o);
    c *= 1.0f / (s + hc_eps);
  }
  if (lane < nn) comb[lane] = c;
}

// hc_pre's larger half: the hc_*_fn matvec and the Gram, in one launch. Two things shape it:
//
//   * `fn` is read one uint4 a lane, so a wave-pass covers a whole cacheline. A narrower load
//     discards most of every line it fetches and runs at a fraction of the coalesced ceiling.
//   * there are only a couple of dozen output rows, far too few blocks to fill the card. The
//     parallelism is in the long REDUCTION, not the outputs, so the grid is (rows x SPLIT) and pass
//     two sums the partials.
//
// The Gram rides on the first KM blocks, which is what makes this one launch instead of two — it is
// small beside the matvec and reads the same `hc`.
//
// WHICH BLOCKS CARRY THE GRAM MUST NOT CHANGE gpart's DECOMPOSITION: pass two's summation order
// follows it, and a perturbation of a few ulp flips an argmax that becomes the next input. A version
// that keyed the Gram to a swept parameter passed every accuracy check against a double reference
// and still walked into a repetition loop ten tokens later.
template <uint32_t NH, uint32_t SPLIT>
__global__ __launch_bounds__(kHcThreads) void hc_mixgram_kernel(const float* __restrict__ hc,
                                                                const uint16_t* __restrict__ fn,
                                                                float* __restrict__ mpart,
                                                                float* __restrict__ gpart,
                                                                uint32_t n_embd) {
  constexpr uint32_t KM = 2 * NH + NH * NH, NT = kHcTri<NH>;
  const uint32_t flat = n_embd * NH;
  const uint32_t r = blockIdx.x / SPLIT, s = blockIdx.x % SPLIT;
  const uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint32_t nwarp = blockDim.x >> 5;
  __shared__ float red[kHcThreads / 32];

  const uint32_t per = flat / SPLIT;
  const uint32_t c0 = s * per;
  float acc = 0.0f;
  for (uint32_t c = c0 + tid * 8u; c < c0 + per; c += blockDim.x * 8u) {
    const uint4 wv = *(const uint4*)(fn + (size_t)r * flat + c);
    const float4 x0 = *(const float4*)(hc + c);
    const float4 x1 = *(const float4*)(hc + c + 4);
    const uint16_t* wh = (const uint16_t*)&wv;
#define AFF_HCW(i) __uint_as_float((uint32_t)wh[i] << 16)
    acc += AFF_HCW(0) * x0.x + AFF_HCW(1) * x0.y + AFF_HCW(2) * x0.z + AFF_HCW(3) * x0.w;
    acc += AFF_HCW(4) * x1.x + AFF_HCW(5) * x1.y + AFF_HCW(6) * x1.z + AFF_HCW(7) * x1.w;
#undef AFF_HCW
  }
  for (int o = 16; o; o >>= 1) acc += __shfl_down(acc, o);
  if (lane == 0) red[warp] = acc;
  __syncthreads();
  if (tid == 0) {
    float t = 0.0f;
    for (uint32_t w = 0; w < nwarp; ++w) t += red[w];
    mpart[(size_t)s * KM + r] = t;
  }

  if (blockIdx.x >= KM) return;                          // the first KM blocks carry the Gram
  const uint32_t gg = blockIdx.x;
  __shared__ float redg[NT][kHcThreads / 32];
  float gacc[NT];
#pragma unroll
  for (uint32_t i = 0; i < NT; ++i) gacc[i] = 0.0f;
  for (uint32_t d = gg * blockDim.x + tid; d < n_embd; d += KM * blockDim.x) {
    float v[NH];
#pragma unroll
    for (uint32_t h = 0; h < NH; ++h) v[h] = hc[(size_t)h * n_embd + d];
    uint32_t k = 0;
#pragma unroll
    for (uint32_t a = 0; a < NH; ++a)
#pragma unroll
      for (uint32_t b = a; b < NH; ++b) gacc[k++] += v[a] * v[b];
  }
#pragma unroll
  for (uint32_t i = 0; i < NT; ++i) {
    float t = gacc[i];
    for (int o = 16; o; o >>= 1) t += __shfl_down(t, o);
    if (lane == 0) redg[i][warp] = t;
  }
  __syncthreads();
  if (tid < NT) {
    float t = 0.0f;
    for (uint32_t w = 0; w < nwarp; ++w) t += redg[tid][w];
    gpart[(size_t)gg * NT + tid] = t;
  }
}

// Pass two. Most of this pair's cost is the two launch boundaries and the Sinkhorn — a dependent
// chain on one wave that the whole grid waits behind — rather than pass one's traffic. So the fixes
// live here:
//
//   * NBG and NBM are template parameters. As runtime arguments each gather becomes a chain of
//     dependent global loads the compiler cannot unroll.
//   * The Sinkhorn gets a block of its own, which computes nothing else and maps nothing, so the
//     kernel costs max(map, Sinkhorn) rather than Sinkhorn then map. Running it on block 0 without
//     that overlap buys nothing.
//   * That block skips the gates and the fp64 quadratic form, which are dead on it and sit ahead of
//     the Sinkhorn on the same wave; and the map's loads hoist above the first barrier, since
//     nothing in the warp-0 chain feeds their addresses.
//
// All of it is bit-identical by construction: same values, same summation order, work removed only
// where its result was never read.
template <uint32_t NH, uint32_t NBG, uint32_t NBM>
__global__ __launch_bounds__(kHcThreads) void hc_apply_mg2_kernel(
    const float* __restrict__ hc, const float* __restrict__ gpart,
    const float* __restrict__ mpart,
    const float* __restrict__ scale, const float* __restrict__ base, const float* __restrict__ nw,
    float* __restrict__ pre_out, float* __restrict__ post_out, float* __restrict__ comb_out,
    float* __restrict__ cur, float* __restrict__ norm, const float* __restrict__ pre_in,
    uint32_t n_embd, uint32_t iters, float hc_eps, float rms_eps) {
  constexpr uint32_t KM = 2 * NH + NH * NH, NT = kHcTri<NH>;
  // s_pre holds the mix this collapse USES; s_pre_new the one this sublayer COMPUTED. They are the
  // same on V4. On V4.1 the block collapses with what the previous sublayer computed (`pre_in`) and
  // hands its own forward — see the shift in the reference's Block.forward.
  __shared__ float s_g[NH * NH], s_pre[NH], s_pre_new[NH], s_post[NH], s_inv, s_mix[KM];
  const uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  // The last block runs the Sinkhorn and maps nothing. `blockIdx.x != 0` guards a one-block grid,
  // where block 0 would be `last` and skipping the gates would drop the pre/post publish.
  const bool last = blockIdx.x == gridDim.x - 1;
  const bool dead = last && blockIdx.x != 0;

  // Issued FIRST, before the gathers: these are the map's loads and the map is the last thing in
  // the kernel, but nothing in the warp-0 chain feeds their ADDRESSES — only the s_pre[h] they are
  // multiplied by. The deployed geometry is n_embd=4096 over 16 mapping blocks of 256, exactly one
  // iteration a thread, so this prefetches the whole map; the loop keeps its general form.
  const uint32_t d0 = blockIdx.x * blockDim.x + tid;
  const bool pf = !dead && d0 < n_embd;
  float pv[NH], pw = 1.0f;
  if (pf) {
#pragma unroll
    for (uint32_t h = 0; h < NH; ++h) pv[h] = hc[(size_t)h * n_embd + d0];
    pw = nw ? nw[d0] : 1.0f;
  }

  if (tid < NT) {
    float s = 0.0f;
#pragma unroll
    for (uint32_t b = 0; b < NBG; ++b) s += gpart[(size_t)b * NT + tid];
    uint32_t a = 0, bb = 0, k = 0;                   // (a,b) from the triangle index
#pragma unroll
    for (uint32_t i = 0; i < NH; ++i)
#pragma unroll
      for (uint32_t j = i; j < NH; ++j) { if (k == tid) { a = i; bb = j; } ++k; }
    s_g[a * NH + bb] = s;
    s_g[bb * NH + a] = s;
  }
  if (tid < KM) {
    float s = 0.0f;
#pragma unroll
    for (uint32_t b = 0; b < NBM; ++b) s += mpart[(size_t)b * KM + tid];
    s_mix[tid] = s;
  }
  __syncthreads();

  if (warp == 0) {
    float tr = 0.0f;
#pragma unroll
    for (uint32_t h = 0; h < NH; ++h) tr += s_g[h * NH + h];
    const float mix_scale = rsqrtf(tr / (float)(n_embd * NH) + rms_eps);
    if (!dead) {
      hc_gates_pre<NH>(s_mix, mix_scale, scale, base, s_pre, s_post, hc_eps, lane);
      // Same wave, so the LDS write below is visible to lane 0's read that follows — the pattern
      // hc_gates_pre itself relies on.
      if (lane < NH) {
        s_pre_new[lane] = s_pre[lane];
        if (pre_in) s_pre[lane] = pre_in[lane];
      }
      if (lane == 0) {
        // Double, because pre^T G pre is algebraically a sum of squares but not numerically: G's
        // off-diagonals are signed, and an f32 contraction of the form can cancel several digits for
        // anti-correlated lanes. f32 measures identical on random hc — which is exactly the input
        // that cannot show the failure — so this stays until an anti-correlated case says otherwise.
        double q = 0.0;
#pragma unroll
        for (uint32_t a = 0; a < NH; ++a)
#pragma unroll
          for (uint32_t b = 0; b < NH; ++b)
            q += (double)s_pre[a] * (double)s_pre[b] * (double)s_g[a * NH + b];
        s_inv = rsqrtf((float)(q / (double)n_embd) + rms_eps);
      }
    }
    // Nothing in this block waits on `comb`, so it goes straight to global and skips the LDS.
    if (last) hc_sinkhorn_only<NH>(s_mix, mix_scale, scale, base, comb_out, iters, hc_eps, lane);
  }
  __syncthreads();

  if (blockIdx.x == 0 && tid < NH) {
    pre_out[tid] = pre_in ? s_pre_new[tid] : s_pre[tid];
    post_out[tid] = s_post[tid];
  }
  if (last) return;                                  // the extra block maps nothing
  const float inv = s_inv;
  const uint32_t nmap = gridDim.x - 1u;
  bool first = true;
  for (uint32_t d = d0; d < n_embd; d += nmap * blockDim.x) {
    const bool have = pf && first;                   // already prefetched above
    float a2 = 0.0f;
#pragma unroll
    for (uint32_t h = 0; h < NH; ++h)
      a2 += (have ? pv[h] : hc[(size_t)h * n_embd + d]) * s_pre[h];
    cur[d] = a2;
    norm[d] = a2 * inv * (have ? pw : (nw ? nw[d] : 1.0f));
    first = false;
  }
}

// The expand arithmetic, factored out so the plain kernel and the fused one cannot drift apart. `b`
// is the block value for this `d`, however obtained: read from global by the kernel below, or still
// in a register from the exchange by the one after it.
//
// NH is a template parameter and everything is loaded once into registers because the loops only
// unroll if the compiler can see the trip count — otherwise `hc[src * n_embd + d]` sits in the
// inner loop (16 loads of 4 values at NH=4) and `post`/`comb` are re-read every iteration.
template <uint32_t NH>
__device__ inline void hc_expand_body(float* __restrict__ out_hc, float b,
                                      const float* __restrict__ hc, const float* __restrict__ post,
                                      const float* __restrict__ comb, uint32_t n_embd, uint32_t d) {
  float v[NH], p[NH], c[NH * NH];
#pragma unroll
  for (uint32_t src = 0; src < NH; ++src) v[src] = hc[(size_t)src * n_embd + d];
#pragma unroll
  for (uint32_t i = 0; i < NH; ++i) p[i] = post[i];
#pragma unroll
  for (uint32_t i = 0; i < NH * NH; ++i) c[i] = comb[i];
#pragma unroll
  for (uint32_t dst = 0; dst < NH; ++dst) {
    float acc = b * p[dst];
#pragma unroll
    for (uint32_t src = 0; src < NH; ++src) acc += c[dst + src * NH] * v[src];
    out_hc[(size_t)dst * n_embd + d] = acc;
  }
}

// Every consumer leaves the block ZEROED. The FFN sublayer accumulates into `d_block` from two
// independent sources, so the block must be zero before either runs — otherwise a hipMemsetAsync per
// layer per rank, which is a blit kernel and a launch boundary for 16 KiB. hc_post is the last
// reader in every sublayer and its grid partitions the same elements the same way, so it stores the
// zero on its way out. The attention sublayer never needed one: `attn_out`'s wo_b matvec stores
// rather than accumulates.
//
// Safe against the shared expert, the one writer not on this stream, because its launches are held
// until shexp_collect() releases them, which is after hc_post.
template <uint32_t NH>
__global__ void hc_expand_reg_kernel(float* __restrict__ out_hc, float* __restrict__ block,
                                     const float* __restrict__ hc, const float* __restrict__ post,
                                     const float* __restrict__ comb, uint32_t n_embd) {
  const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= n_embd) return;
  hc_expand_body<NH>(out_hc, block[d], hc, post, comb, n_embd, d);
  block[d] = 0.0f;
}

// The plain residual path, for layers without a hyper-connection control. ZB says whether `block`
// is the sublayer's accumulator, which this call drains and must leave zero, or a staging buffer
// somebody else owns — `block_add_host` passes the host tier's partial here and zeroing THAT would
// throw the contribution away on the next call that read it.
template <bool ZB>
__global__ void hc_add_kernel(float* __restrict__ hc, float* __restrict__ block,
                              uint32_t n_embd) {
  const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d < n_embd) {
    hc[d] += block[d];
    if (ZB) block[d] = 0.0f;
  }
}

// ---- the sublayer's all-reduce, folded into the kernel that consumes its result ----------------
//
// `block_reduce` and `hc_post` are adjacent at every call site, and in decode every reduce the
// engine issues is one of that pair — two a layer, 86 a token. So the exchange wrote 16 KiB into
// `d_block` and the very next kernel read it straight back out, a whole launch to move a value that
// could have stayed in a register.
//
// Exact because the expand is elementwise in `d`: block b of the expand owns the chunk block b of
// the exchange would have owned, so the grids line up block for block and no cross-block
// communication is added. The sum is the same f32 add in the same operand order as tp_ar_kernel's
// (mine first, peer second), so this is bit-identical to reduce-then-expand.
//
// It does NOT remove the handshake — both cards still pay the fabric round trip and the late one
// still waits. The `reduce` slot vanishes INTO `hc_post`; read the token rate, not the table.
__device__ inline void hc_ar_store_rel(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ inline unsigned int hc_ar_load_acq(const unsigned int* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Push this block's chunk, release it, wait for the peer's, and return mine + the peer's for `d`.
// EVERY thread of the block must reach this — it contains two __syncthreads, so the usual
// `if (d >= n) return;` prologue has to move BELOW it or a partial block wedges the grid.
__device__ inline float hc_ar_exchange(const float* __restrict__ block, uint32_t d, bool live,
                                       float* peer_scratch, const float* __restrict__ my_scratch,
                                       unsigned int* peer_flags, unsigned int* my_flags,
                                       unsigned int* seq, uint32_t slot_stride) {
  __shared__ unsigned int s_seq;
  if (threadIdx.x == 0) s_seq = atomicAdd(&seq[blockIdx.x], 1u) + 1u;
  __syncthreads();
  const unsigned int s = s_seq;
  float* ps = peer_scratch + (size_t)(s & 1u) * slot_stride;
  const float* ms = my_scratch + (size_t)(s & 1u) * slot_stride;
  const float mine = live ? block[d] : 0.0f;
  if (live) ps[d] = mine;
  // Drain posted writes with no L2 invalidate, not __threadfence_system — see tp_allreduce.hip for
  // why the invalidate is waste on memory that was never cached. PCIe keeps posted writes to one
  // peer ordered, so the release below cannot overtake the data.
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    hc_ar_store_rel(&peer_flags[blockIdx.x], s);
    unsigned long long z = 0;
    // Same generous bound as tp_ar_kernel: a lost peer degrades to a wrong answer, not a hung GPU.
    while (hc_ar_load_acq(&my_flags[blockIdx.x]) < s) { if (++z > (1ull << 32)) break; }
  }
  __syncthreads();
  return live ? mine + ms[d] : 0.0f;
}

// The zero is safe under the exchange too: the peer never reads MY block, it reads what I pushed
// into ITS scratch, and that push happens inside hc_ar_exchange before this store.
template <uint32_t NH>
__global__ void hc_expand_ar_kernel(float* __restrict__ out_hc, float* __restrict__ block,
                                    const float* __restrict__ hc, const float* __restrict__ post,
                                    const float* __restrict__ comb, uint32_t n_embd,
                                    float* peer_scratch, const float* __restrict__ my_scratch,
                                    unsigned int* peer_flags, unsigned int* my_flags,
                                    unsigned int* seq, uint32_t slot_stride) {
  const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
  const bool live = d < n_embd;
  const float b = hc_ar_exchange(block, d, live, peer_scratch, my_scratch, peer_flags, my_flags,
                                 seq, slot_stride);
  if (!live) return;
  hc_expand_body<NH>(out_hc, b, hc, post, comb, n_embd, d);
  block[d] = 0.0f;
}

// The same fold for the layers without a hyper-connection control.
__global__ void hc_add_ar_kernel(float* __restrict__ hc, float* __restrict__ block,
                                 uint32_t n_embd, float* peer_scratch,
                                 const float* __restrict__ my_scratch, unsigned int* peer_flags,
                                 unsigned int* my_flags, unsigned int* seq, uint32_t slot_stride) {
  const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
  const bool live = d < n_embd;
  const float b = hc_ar_exchange(block, d, live, peer_scratch, my_scratch, peer_flags, my_flags,
                                 seq, slot_stride);
  if (live) { hc[d] += b; block[d] = 0.0f; }
}

// The norm weight ships f32 for these tensors, so this takes it as f32 and keeps the inner loop free
// of a widening branch per element.
__global__ void rms_norm_f32_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                    float* __restrict__ out, uint32_t n, float eps) {
  __shared__ float red[32];
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
  const float inv = red[0];
  for (uint32_t i = tid; i < n; i += nt) out[i] = x[i] * inv * (w ? w[i] : 1.0f);
}
}

// Attention is head-parallel, which is why this sublayer needs one exchange rather than one per
// step: wq_a is replicated so both cards produce the same q_lora vector and normalise it, and wq_b
// is row-split at a head boundary, so card r owns heads [r*32, r*32+32) through RoPE, the attention
// and wo_a's groups. Nothing crosses until wo_b.
bool DenseGpu::attn_q(int32_t qa, int32_t qb, int32_t qn, const float* x, uint64_t hidden,
                      uint64_t qr, uint32_t n_head, uint32_t width, uint32_t n_rot, uint32_t pos,
                      RopeDerived rope, float eps, float* q_out, float* qrnorm_out) {
  AFF_TRACE(3);
  if (qa < 0 || qb < 0 || qn < 0) return false;
  if ((size_t)qa >= ent_.size() || (size_t)qb >= ent_.size() || (size_t)qn >= n_vec_) return false;
  const Entry& A = ent_[qa]; const Entry& B = ent_[qb];
  if (A.split != Split::Rep || B.split != Split::Row) return false;
  const size_t R = impl_->dev.size();
  for (size_t d = 0; d < R; ++d) {
    // A shard that ends mid-head would hand this rank a partial head to normalise and rotate.
    if (B.sh[d].row0 % width || B.sh[d].nrows % width) return false;
    if (B.sh[d].nrows > impl_->dev[d].cap_mid || qr > impl_->dev[d].cap_mid) return false;
  }

  for (size_t d = 0; d < R; ++d) {
    Impl::Dev& D = impl_->dev[d];
    const uint64_t nq = B.sh[d].nrows;                // this card's heads, in floats
    const uint32_t nh = (uint32_t)(nq / width);
    AFF_HIP_CHECK(hipSetDevice(D.id));
    const float* xin = x ? D.d_x : D.d_norm;
    if (x) AFF_STAGE_UP(D, kStgQ, D.d_x, x, hidden, D.stream);
    // wq_a may already be done: attn_pre puts it in the layer's one batched launch and leaves the
    // result at the head of d_y — on both cards, because wq_a is replicated. The flag is cleared
    // here so a second chain cannot read a buffer the first one's successor has overwritten.
    const float* qa_out = D.d_g;
    if (pre_q_ && !x && pre_qr_ == qr)
      qa_out = D.d_y;
    else
      launch_mv({A.sh[d].data, A.sh[d].scales, A.quant, A.sh[d].ps}, xin, (uint32_t)qr,
                (uint32_t)hidden, D.d_g, D.stream);
    // q goes straight into this card's attention input buffer, which holds ITS heads at offset 0; a
    // device-to-device copy from d_a instead is a large fraction of the matvec it would follow.
    // attend() runs on this stream ordered behind the rope, so there is no window in which a_q holds
    // a half-written q.
    const bool to_aq = attn_ok_ && nq <= (uint64_t)a_hpr_ * a_width_;
    float* const qdst = to_aq ? D.a_q : D.d_a;
    // The RMSNorm between wq_a and wq_b is a single-block launch on its own — one CU reducing a few
    // KB. wq_b reads exactly what it wrote, hundreds of blocks over, so it does the reduction itself;
    // bit-identical, see matvec_fp8_rn_kernel. d_u is only materialised for readers that want it.
    const bool want_u = (qrnorm_out && d == 0) || idx_want_qrn_;
    if (B.quant == AFF_FP8_E4M3 && D.vec[qn]) {
      no_ps(B.sh[d].ps, "the fused wq_b matvec+rmsnorm");
      matvec_fp8_rmsnorm_hip((const uint8_t*)B.sh[d].data, B.sh[d].scales, qa_out, D.vec[qn], eps,
                             (uint32_t)nq, (uint32_t)qr, qdst, want_u ? D.d_u : nullptr, D.stream);
    } else {
      hipLaunchKernelGGL(rms_norm_f32_kernel, dim3(1), dim3(256), 0, D.stream,
                         qa_out, D.vec[qn], D.d_u, (uint32_t)qr, eps);
      AFF_HIP_CHECK(hipGetLastError());
      launch_mv({B.sh[d].data, B.sh[d].scales, B.quant, B.sh[d].ps}, D.d_u, (uint32_t)nq,
                (uint32_t)qr, qdst, D.stream);
    }
    // Per-head norm and the RoPE tail in one launch: a block owns a head, and the rotation only
    // touches the head that block just normalised.
    hipLaunchKernelGGL(head_rms_rope_kernel, dim3(nh), dim3(128), 0, D.stream,
                       qdst, width, eps, n_rot, pos, rope.freq_scale, rope.theta_scale,
                       rope.ext_factor, rope.lo, rope.hi, 1.0f);
    AFF_HIP_CHECK(hipGetLastError());
    bool need_sync = false;
    if (!to_aq && q_out) {
      AFF_HIP_CHECK(hipMemcpyAsync(q_out + B.sh[d].row0, qdst, nq * sizeof(float),
                                   hipMemcpyDeviceToHost, D.stream));
      need_sync = true;
    }
    // qr_norm comes back only when the lightning indexer will read it, which needs index_topk
    // (512) below n_comp = pos/4 — so not until ~2048 positions. Rank 0 only: wq_a is replicated,
    // so the two are identical.
    if (qrnorm_out && d == 0) {
      AFF_HIP_CHECK(hipMemcpyAsync(qrnorm_out, D.d_u, qr * sizeof(float), hipMemcpyDeviceToHost,
                                   D.stream));
      need_sync = true;
    }
    // Kept on device for the indexer instead, when the caller asked for that. d_u is scratch and
    // attn_kv is the very next chain to write it, so the copy has to happen here or not at all.
    if (idx_want_qrn_) {
      if (D.d_qrn_cap < qr) {
        if (D.d_qrn) (void)hipFree(D.d_qrn);
        if (hipMalloc(&D.d_qrn, (size_t)qr * sizeof(float)) != hipSuccess) return false;
        D.d_qrn_cap = (uint32_t)qr;
      }
      AFF_HIP_CHECK(hipMemcpyAsync(D.d_qrn, D.d_u, qr * sizeof(float), hipMemcpyDeviceToDevice,
                                   D.stream));
    }
    if (need_sync) AFF_DRAIN(D);
  }
  pre_q_ = false;
  ++st_.calls;
  return true;
}

// Replicated end to end: wkv is small, and the KV cache it feeds lives on both cards because every
// card attends over all of it (MQA — one KV head). Splitting its outputs would buy less reading than
// the exchange costs, twice per layer.
bool DenseGpu::attn_kv(int32_t kv, int32_t kvn, const float* x, uint64_t hidden, uint32_t width,
                       uint32_t n_rot, uint32_t pos, RopeDerived rope, float eps, float* out) {
  AFF_TRACE(4);
  if (kv < 0 || kvn < 0) return false;
  if ((size_t)kv >= ent_.size() || (size_t)kvn >= n_vec_) return false;
  const Entry& K = ent_[kv];
  if (K.split != Split::Rep) return false;
  // Every condition kv_commit is about to test, tested here, plus the caller's own signal that it
  // is going to call it at all.
  const bool defer_kvn = !out && attn_ok_ && a_ring_ && width == a_width_ &&
                         n_rot == a_rot_;
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    const float* xin = x ? D.d_x : D.d_norm;
    if (x) AFF_STAGE_UP(D, kStgKv, D.d_x, x, hidden, D.stream);
    // Same handoff as attn_q: wkv's rows sit at d_y + qr when attn_pre ran.
    const float* kv_out = D.d_g;
    if (pre_kv_ && !x && pre_width_ == width) kv_out = D.d_y + pre_qr_;
    else
      launch_mv({K.sh[d].data, K.sh[d].scales, K.quant, K.sh[d].ps}, xin, width, (uint32_t)hidden, D.d_g,
                D.stream);
    // No dispatch at all when it can be deferred: there is one KV head, so this step is 512 floats,
    // and kv_commit is the next thing to touch the latent and already reads every element, so it
    // does the norm and the rotation on the way in. `defer` is only true when THIS call has checked
    // the shape kv_commit will demand, so the latent cannot end up normalised by nobody.
    if (defer_kvn) {
      kvn_src_[d] = kv_out;
      kvn_w_[d] = D.vec[kvn];
    } else {
      hipLaunchKernelGGL(rms_norm_rope_kernel, dim3(1), dim3(256), 0, D.stream,
                         kv_out, D.vec[kvn], D.d_u, width, eps, n_rot, pos, rope.freq_scale,
                         rope.theta_scale, rope.ext_factor, rope.lo, rope.hi, 1.0f);
      AFF_HIP_CHECK(hipGetLastError());
    }
    // No sync when the latent stays on device: kv_commit and attend run on this same stream and are
    // already ordered behind it. Syncing here was 43 stalls a token for nothing.
    if (out && d == 0) {
      AFF_HIP_CHECK(hipMemcpyAsync(D.h_y, D.d_u, width * sizeof(float), hipMemcpyDeviceToHost,
                                   D.stream));
      AFF_DRAIN(D);
      std::memcpy(out, D.h_y, width * sizeof(float));
    }
  }
  pre_kv_ = false;
  kvn_pending_ = defer_kvn;
  kvn_eps_ = eps;
  kvn_pos_ = pos;
  kvn_rope_ = rope;
  ++st_.calls;
  return true;
}

bool DenseGpu::kv_commit(uint32_t layer, uint32_t pos, uint32_t width, uint32_t n_rot) {
  AFF_TRACE(11);
  // A refusal here has to clear the deferral too. attn_kv already tested everything below except
  // the layer bound, but a pending flag surviving a refusal would hand the NEXT kv_commit a
  // kvn_src_ from the wrong layer — a stale pointer that still dereferences.
  if (!attn_ok_ || layer >= impl_->dev[0].a_raw_s[slot_].size() || !a_ring_) { kvn_pending_ = false; return false; }
  // The cache was sized from a_rot_ at init; committing against a different tail would write rows
  // the attend kernel then reads with the wrong stride.
  if (width != a_width_ || n_rot != a_rot_) { kvn_pending_ = false; return false; }

  // Main stream, always: this is a dispatch plus one block's dependent chain, and a side stream
  // overlaps execution rather than dispatch, so the event round trip would cost what the overlap
  // saved.
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    hipStream_t kst = D.stream;
    // The same fused kernel the batched path uses, at n = 1 — d_u is one token's contiguous [width]
    // latent, which IS the dim-major [width][nb] layout at nb = 1, b0 = 0.
    //
    // When attn_kv deferred, the latent arriving here is the RAW wkv output and the norm and the
    // rotation happen in this kernel's prologue. Norm is a template parameter on the one kernel,
    // not a second kernel that has to be kept in step.
    bool done = false;
    if (kvn_pending_ && kvn_src_[d])
      done = kv_norm_commit_hip(a_dt_, kvn_src_[d], kvn_w_[d], kvn_eps_, D.a_raw_s[slot_][layer], pos,
                                a_ring_, width, n_rot, kvn_rope_.freq_scale, kvn_rope_.theta_scale,
                                kvn_rope_.ext_factor, kvn_rope_.lo, kvn_rope_.hi, kst);
    if (!done) {
      // The prologue declined, so its half has to happen here before the commit reads the latent.
      if (kvn_pending_ && kvn_src_[d]) {
        hipLaunchKernelGGL(rms_norm_rope_kernel, dim3(1), dim3(256), 0, D.stream, kvn_src_[d],
                           kvn_w_[d], D.d_u, width, kvn_eps_, n_rot, kvn_pos_,
                           kvn_rope_.freq_scale, kvn_rope_.theta_scale, kvn_rope_.ext_factor,
                           kvn_rope_.lo, kvn_rope_.hi, 1.0f);
        AFF_HIP_CHECK(hipGetLastError());
      }
      kv_commit_batch_hip(a_dt_, D.d_u, 1, 0, 1, D.a_raw_s[slot_][layer], pos, a_ring_, width, n_rot,
                          D.stream);
    }
  }
  kvn_pending_ = false;                               // consumed exactly once, like pre_kv_
  return true;                                        // no sync: attend() will order behind this
}

// The canonical Megatron FFN: gate and up are Row so each card owns half the intermediate, SwiGLU
// is elementwise and therefore local, and `down` is Col so each card's partial covers the whole
// hidden dimension. The partials are left in the two blocks and the sublayer's one all-reduce at
// the end completes them — which is also what folds in the routed experts, since those deposit
// their partials into the very same blocks.
bool DenseGpu::shexp(int32_t hg, int32_t hu, int32_t hd, const float* x, uint64_t hidden,
                     uint64_t inter, float limit, float* out, bool defer) {
  AFF_TRACE(2);
  if (hg < 0 || hu < 0 || hd < 0) return false;
  if ((size_t)hg >= ent_.size() || (size_t)hu >= ent_.size() || (size_t)hd >= ent_.size())
    return false;
  const Entry& G = ent_[hg]; const Entry& U = ent_[hu]; const Entry& W = ent_[hd];
  if (G.split != Split::Row || U.split != Split::Row || W.split != Split::Col) return false;
  const size_t R = impl_->dev.size();
  for (size_t d = 0; d < R; ++d) {
    // gate's row band and down's column band are the same slice of the intermediate. They are cut
    // by independent arithmetic in reg(), so check rather than assume: a mismatch would multiply
    // one card's activations by the other card's weights and still produce finite text.
    if (G.sh[d].nrows != U.sh[d].nrows || G.sh[d].nrows != W.sh[d].ncols) return false;
    if (G.sh[d].row0 != W.sh[d].col0) return false;
    if (G.sh[d].nrows > impl_->dev[d].cap_mid) return false;
  }

  // When the result stays on device (out == nullptr) the launches are HELD rather than issued, and
  // shexp_collect() releases them onto the side stream after the routed dispatch has gone out. That
  // is where the overlap has to land: releasing as soon as d_norm is ready overlaps shexp with the
  // router instead, which is dispatch-bound rather than bandwidth-bound, so the two contend and the
  // routed FFN — the one thing worth hiding behind — barely moves.
  //
  // `defer` is the caller's promise that shexp_collect() will run. The model has a second,
  // non-deferred call site for when this one refuses; deferring there would leave the shared expert
  // out of the block entirely.
  const bool hold = !out && defer;

  float* in[kMaxRanks]; float* outp[kMaxRanks];
  for (size_t d = 0; d < R; ++d) {
    Impl::Dev& D = impl_->dev[d];
    const uint32_t li = (uint32_t)G.sh[d].nrows;      // this card's share of the intermediate
    AFF_HIP_CHECK(hipSetDevice(D.id));
    const float* xin = x ? D.d_x : D.d_norm;
    if (x) AFF_STAGE_UP(D, kStgShexp, D.d_x, x, hidden, D.stream);
    if (hold) {
      // Remembered, not launched. shexp_collect() flushes it — a shared expert that silently did
      // not run is the one failure this whole sublayer cannot detect.
      D.sx_g = &G.sh[d]; D.sx_u = &U.sh[d]; D.sx_w = &W.sh[d];
      D.sx_qg = G.quant; D.sx_qu = U.quant; D.sx_qw = W.quant;
      D.sx_x = xin; D.sx_li = li; D.sx_hidden = (uint32_t)hidden; D.sx_limit = limit;
      D.shexp_side = true; D.sx_ev = false;
      in[d] = D.d_p; outp[d] = D.d_p;
      continue;
    }

    // Gate and up are independent and read the same activation AND the same row index, so one wave
    // takes both dots and applies the SwiGLU on the spot: one launch, not two plus a round trip
    // through VRAM. The bytes were never the problem — at these sizes a matvec is a few microseconds
    // of bandwidth under a fixed per-launch cost several times that, and each launch's ramp and
    // drain serialises behind the last.
    no_ps(G.sh[d].ps || U.sh[d].ps, "the fused single-token gate/up matvec");
    mv_gate_up_swiglu_hip(G.sh[d].data, G.sh[d].scales, G.quant,
                          U.sh[d].data, U.sh[d].scales, U.quant,
                          xin, D.d_a, li, (uint32_t)hidden, limit, D.stream);
    launch_mv({W.sh[d].data, W.sh[d].scales, W.quant, W.sh[d].ps}, D.d_a, (uint32_t)hidden, li,
              D.d_p, D.stream);
    in[d] = D.d_p; outp[d] = D.d_p;
    D.shexp_side = false;
    D.sx_ev = false;
    if (!out) {
      // ACCUMULATE the partial into this card's block for the sublayer's reduce to finish — the
      // routed experts write there too, so a plain overwrite would drop them.
      const uint32_t thr = 256, blk = (uint32_t)((hidden + thr - 1) / thr);
      // ZB=false: `D.d_p` is the shared expert's own output, not the sublayer accumulator.
      hipLaunchKernelGGL(HIP_KERNEL_NAME(hc_add_kernel<false>), dim3(blk), dim3(thr), 0, D.stream,
                         D.d_block, D.d_p, (uint32_t)hidden);
      AFF_HIP_CHECK(hipGetLastError());
    }
  }
  ++st_.calls;
  if (!out) return true;

  // The host wants the whole vector, so the partials are summed here rather than at the end of the
  // sublayer — but the chain is left in flight and the caller gets a collect(). Synchronising here
  // would spend the one window in the sublayer where the cards and the host have independent work.
  reduce(in, outp, hidden);
  Impl::Dev& D = impl_->dev[0];
  AFF_HIP_CHECK(hipSetDevice(D.id));
  AFF_HIP_CHECK(hipMemcpyAsync(D.h_y, D.d_p, hidden * sizeof(float), hipMemcpyDeviceToHost,
                               D.stream));
  if (defer) return true;                    // caller collects
  AFF_DRAIN(D);
  std::memcpy(out, D.h_y, hidden * sizeof(float));
  return true;
}

// Issue what shexp() held back. See shexp() for why the launches do not simply go out where they
// were computed.
void DenseGpu::shexp_release(size_t d) {
  Impl::Dev& D = impl_->dev[d];
  if (!D.sx_g) return;
  AFF_HIP_CHECK(hipSetDevice(D.id));
  // Recorded HERE, so the side stream starts behind everything the main stream has been given since
  // shexp() ran — the router's two kernels and the routed dispatch.
  AFF_HIP_CHECK(hipEventRecord(D.ev_sx, D.stream));
  AFF_HIP_CHECK(hipStreamWaitEvent(D.sstream, D.ev_sx, 0));
  mv_gate_up_swiglu_hip(D.sx_g->data, D.sx_g->scales, D.sx_qg,
                        D.sx_u->data, D.sx_u->scales, D.sx_qu,
                        D.sx_x, D.d_a, D.sx_li, D.sx_hidden, D.sx_limit, D.sstream);
  launch_mv({D.sx_w->data, D.sx_w->scales, D.sx_qw, D.sx_w->ps}, D.d_a, D.sx_hidden, D.sx_li, D.d_p,
            D.sstream);
  AFF_HIP_CHECK(hipEventRecord(D.ev_sd, D.sstream));
  D.sx_g = nullptr; D.sx_u = nullptr; D.sx_w = nullptr;
  D.sx_ev = true;
}

// Finish a deferred shexp: wait for the chain and copy the result out of the pinned landing buffer.
bool DenseGpu::shexp_collect(float* out, uint64_t hidden) {
  AFF_TRACE(2);
  if (!impl_ || impl_->dev.empty()) return false;
  // The deferred fold. The caller arrives having already issued the routed dispatch, so the two
  // chains have been in flight together; this is where the block stops being two partials. Still
  // asynchronous — the wait is a stream dependency, not a drain.
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    if (!D.shexp_side) continue;
    D.shexp_side = false;
    shexp_release(d);                              // no-op unless shexp() held the launches back
    AFF_HIP_CHECK(hipSetDevice(D.id));
    if (D.sx_ev) AFF_HIP_CHECK(hipStreamWaitEvent(D.stream, D.ev_sd, 0));
    D.sx_ev = false;
    const uint32_t thr = 256, blk = (uint32_t)((hidden + thr - 1) / thr);
    // ZB=false: `D.d_p` is the shared expert's own output, not the sublayer accumulator.
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hc_add_kernel<false>), dim3(blk), dim3(thr), 0, D.stream,
                       D.d_block, D.d_p, (uint32_t)hidden);
    AFF_HIP_CHECK(hipGetLastError());
  }
  // Nothing to collect when the result stayed on device: it was accumulated into the blocks, and
  // block_reduce / hc_post run on these same streams already ordered behind it.
  if (!out) return true;
  Impl::Dev& D = impl_->dev[0];
  AFF_HIP_CHECK(hipSetDevice(D.id));
  AFF_DRAIN(D);
  std::memcpy(out, D.h_y, hidden * sizeof(float));
  return true;
}


// Replicated: the 4-lane state is 64 KiB, so a split would be pure communication. Both cards run
// the identical control kernel over identical data and arrive at identical `norm` and `cur`, which
// is what lets every downstream chain read `D.d_norm` on its own card with no exchange first.
bool DenseGpu::hc_init(uint32_t n_embd, uint32_t n_hc, std::string* err) {
  if (!impl_ || impl_->dev.empty()) { if (err) *err = "no device"; return false; }
  const uint64_t hc_dim = (uint64_t)n_embd * n_hc;
  const uint32_t n_out = 2 * n_hc + n_hc * n_hc;
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    AFF_HIP_CHECK(hipMalloc(&D.d_hc, hc_dim * sizeof(float)));
    AFF_HIP_CHECK(hipMalloc(&D.d_hc_next, hc_dim * sizeof(float)));
    AFF_HIP_CHECK(hipMalloc(&D.d_mix, n_out * sizeof(float)));
    AFF_HIP_CHECK(hipMalloc(&D.d_pre, n_hc * sizeof(float)));
    AFF_HIP_CHECK(hipMalloc(&D.d_pre_prev, n_hc * sizeof(float)));
    AFF_HIP_CHECK(hipMalloc(&D.d_post, n_hc * sizeof(float)));
    AFF_HIP_CHECK(hipMalloc(&D.d_comb, (uint64_t)n_hc * n_hc * sizeof(float)));
    AFF_HIP_CHECK(hipMalloc(&D.d_cur, n_embd * sizeof(float)));
    AFF_HIP_CHECK(hipMalloc(&D.d_norm, n_embd * sizeof(float)));
    AFF_HIP_CHECK(hipMalloc(&D.d_block, n_embd * sizeof(float)));
    // Sized for the largest triangle hc_pre will accept (n_hc <= 8 -> 36), so the buffer does not
    // depend on which n_hc the wide path ends up taking.
    const uint32_t nblk = (n_embd + kHcThreads - 1) / kHcThreads;
    AFF_HIP_CHECK(hipMalloc(&D.d_gpart, (uint64_t)nblk * 36 * sizeof(float)));
    // SPLIT x KM at the largest n_hc this kernel serves. Generous and allocated once.
    AFF_HIP_CHECK(hipMalloc(&D.d_mpart, (uint64_t)kHcSplit * 72 * sizeof(float)));
  }
  hc_ok_ = true;
  return true;
}

// The mix a V4.1 sublayer collapses with, for the FIRST sublayer of a sequence: the reference's
// `make_identity_pre_mix`, one-hot on lane 0. Lanes are laid out [h][tok], so lane 0's row is the
// leading `n` floats and the rest are zero.
void DenseGpu::seed_pre_identity(float* pre, uint32_t n_hc, uint32_t n, void* stream) {
  if (!pre) return;
  const float one = 1.0f;
  uint32_t bits;
  std::memcpy(&bits, &one, 4);
  AFF_HIP_CHECK(hipMemsetAsync(pre, 0, (size_t)n_hc * n * sizeof(float), (hipStream_t)stream));
  AFF_HIP_CHECK(hipMemsetD32Async((hipDeviceptr_t)pre, bits, n, (hipStream_t)stream));
}

void DenseGpu::hc_seed(const float* embed, uint32_t n_embd, uint32_t n_hc) {
  // A new token: a_allow/a_adm/a_admn hold the PREVIOUS token's admissions. indexer_one re-arms
  // this per layer when the indexer engages; without it a stale layer match reads old admissions.
  a_idx_layer_[slot_] = -1;
  if (!hc_ok_) return;
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    for (uint32_t h = 0; h < n_hc; ++h)
      AFF_HIP_CHECK(hipMemcpyAsync(D.d_hc + (size_t)h * n_embd, embed, n_embd * sizeof(float),
                                   hipMemcpyHostToDevice, D.stream));
    if (shift_pre_) seed_pre_identity(D.d_pre_prev, n_hc, 1, D.stream);
  }
}

void DenseGpu::hc_read(float* dst, uint32_t n_embd, uint32_t n_hc) {
  if (!hc_ok_) return;
  Impl::Dev& D = impl_->dev[0];                       // the two are identical
  AFF_HIP_CHECK(hipSetDevice(D.id));
  AFF_HIP_CHECK(hipMemcpyAsync(dst, D.d_hc, (uint64_t)n_embd * n_hc * sizeof(float),
                               hipMemcpyDeviceToHost, D.stream));
  AFF_DRAIN(D);
}

bool DenseGpu::hc_pre(int32_t fn, int32_t scale_v, int32_t base_v, int32_t norm_w,
                      uint32_t n_embd, uint32_t n_hc, uint32_t iters, float hc_eps, float rms_eps,
                      float* norm_host, bool defer) {
  AFF_TRACE(7);
  if (!hc_ok_) return false;
  const uint64_t hc_dim = (uint64_t)n_embd * n_hc;

  // ONE path, two launches: hc_mixgram_kernel does the 24-row matvec AND the Gram together, split
  // kHcSplit ways over the flat n_hc*n_embd state, and hc_apply_mg2_kernel reduces those partials,
  // runs the Sinkhorn on a block of its own and maps the result.
  //
  // Every precondition is a property of the container, so this is decided once a call and never
  // varies per device. bf16 ONLY — the uint4 path decodes the weight itself and would read an fp8
  // plane as bf16 pairs, which is finite, wrong and fluent.
  //
  // A refusal is the caller's cue to ABORT (model.cpp does), not to fall back: every slower variant
  // is gone, so reaching this with a container these conditions reject means the placement is wrong
  // and a silent slow path would hide it.
  if (fn < 0 || (size_t)fn >= ent_.size() || scale_v < 0 || base_v < 0) return false;
  if (ent_[fn].split != Split::Rep || ent_[fn].quant != AFF_BF16) return false;
  if (!hc_gates_wave_ok(n_hc) || n_hc > 4) return false;
  // 8 bf16 a lane over 256 lanes is 2048 columns a block, and the loop is not tail-guarded.
  if (hc_dim % ((uint64_t)kHcSplit * kHcThreads * 8u) != 0) return false;
  for (size_t d = 0; d < impl_->dev.size(); ++d)
    if (!ent_[fn].sh[d].data || !impl_->dev[d].d_mpart) return false;

  const Entry& F = ent_[fn];
  const uint32_t nblk = (n_embd + kHcThreads - 1) / kHcThreads;
  const uint32_t KM = 2 * n_hc + n_hc * n_hc;
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    const float* nw = (norm_w >= 0 && (size_t)norm_w < n_vec_) ? D.vec[norm_w] : nullptr;
    // The apply's grid is nblk + 1: the extra block runs the Sinkhorn and maps nothing.
#define AFF_HC_LAUNCH(NH)                                                                        \
  do {                                                                                           \
    constexpr uint32_t KMR = 2 * (NH) + (NH) * (NH);                                             \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hc_mixgram_kernel<NH, kHcSplit>),                         \
                       dim3(KM * kHcSplit), dim3(kHcThreads), 0, D.stream, D.d_hc,               \
                       (const uint16_t*)F.sh[d].data, D.d_mpart, D.d_gpart, n_embd);             \
    AFF_HIP_CHECK(hipGetLastError());                                                            \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hc_apply_mg2_kernel<NH, KMR, kHcSplit>),                   \
                       dim3(nblk + 1), dim3(kHcThreads), 0, D.stream, D.d_hc, D.d_gpart,         \
                       D.d_mpart, D.vec[scale_v], D.vec[base_v], nw, D.d_pre, D.d_post,          \
                       D.d_comb, D.d_cur, D.d_norm, shift_pre_ ? D.d_pre_prev : nullptr,         \
                       n_embd, iters, hc_eps, rms_eps);                                          \
    AFF_HIP_CHECK(hipGetLastError());                                                            \
  } while (0)
    if (n_hc == 1)      AFF_HC_LAUNCH(1);
    else if (n_hc == 2) AFF_HC_LAUNCH(2);
    else                AFF_HC_LAUNCH(4);
#undef AFF_HC_LAUNCH
    // What this sublayer computed is what the NEXT one collapses with (V4.1). Done after the
    // launch, not before, so the identity seeded at the start of a token survives to the first.
    if (shift_pre_)
      AFF_HIP_CHECK(hipMemcpyAsync(D.d_pre_prev, D.d_pre, n_hc * sizeof(float),
                                   hipMemcpyDeviceToDevice, D.stream));
  }
  {
    // The pinned landing buffer exists whether or not this call wants the copy, because
    // norm_collect can issue one on demand — see there.
    Impl::Dev& D = impl_->dev[0];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    if (norm_cap_ < n_embd) {
      if (D.h_norm) AFF_HIP_CHECK(hipHostFree(D.h_norm));
      AFF_HIP_CHECK(hipHostMalloc(&D.h_norm, (size_t)n_embd * sizeof(float),
                                  hipHostMallocDefault));
      norm_cap_ = n_embd;
    }
    if (norm_host) {
      // Rank 0's copy — the two are bit-identical, both cards having run the same kernel on the
      // same input.
      AFF_HIP_CHECK(hipMemcpyAsync(D.h_norm, D.d_norm, n_embd * sizeof(float),
                                   hipMemcpyDeviceToHost, D.stream));
      norm_pending_ = n_embd;
      // Which drain the copy waits behind. Anything that fully synchronises this stream afterwards
      // lands it as a side effect, and on the common path the device router does exactly that.
      norm_epoch_ = D.drained;
      if (!defer) (void)norm_collect(norm_host, n_embd);
    }
  }
  return true;
}

bool DenseGpu::norm_collect(float* norm_host, uint32_t n_embd) {
  AFF_TRACE(13);
  if (!norm_host) return false;
  Impl::Dev& D = impl_->dev[0];
  if (!norm_pending_) {
    // hc_pre was told nobody wanted `norm` this sublayer, and somebody does. Issue the copy now and
    // pay the drain — this is what lets the caller's predicate be WRONG WITHOUT BEING UNSAFE. The
    // copy is a blit and a launch boundary, dozens of times a token, and on the all-resident
    // device-routed path the host never reads it; skipping it on a guess without this fallback would
    // hand out a stale activation, which reads as fluent text. A mis-guess costs a stall instead.
    if (!hc_ok_ || !D.d_norm || !D.h_norm || norm_cap_ < n_embd) return false;
    AFF_HIP_CHECK(hipSetDevice(D.id));
    AFF_HIP_CHECK(hipMemcpyAsync(D.h_norm, D.d_norm, n_embd * sizeof(float),
                                 hipMemcpyDeviceToHost, D.stream));
    AFF_DRAIN(D);
    std::memcpy(norm_host, D.h_norm, (size_t)n_embd * sizeof(float));
    return true;
  }
  // Only synchronise if NOTHING has drained the stream since the copy was enqueued: a drain waits
  // for everything queued before it, so a later one has already landed this copy. On a non-hash
  // layer the device router drains between the two and this skips; on the hash-routed layers there
  // is no device router and the wait is real, and large.
  if (D.drained == norm_epoch_) {
    AFF_HIP_CHECK(hipSetDevice(D.id));
    AFF_DRAIN(D);
  }
  std::memcpy(norm_host, D.h_norm,
              (n_embd < norm_pending_ ? n_embd : norm_pending_) * sizeof(float));
  norm_pending_ = 0;
  return true;
}

// The routed experts that ran on the CPU, added to rank 0's block ONLY: a host contribution is
// already whole rather than a partial, so adding it to both would have the sublayer's all-reduce
// count it twice.
bool DenseGpu::block_add_host(const float* src, uint32_t n_embd) {
  if (!hc_ok_) return false;
  Impl::Dev& D = impl_->dev[0];
  AFF_HIP_CHECK(hipSetDevice(D.id));
  AFF_STAGE_UP(D, kStgShexp, D.d_x, src, n_embd, D.stream);
  const uint32_t thr = 256, blk = (n_embd + thr - 1) / thr;
  // ZB=false: `D.d_x` is the staging buffer this partial arrived in, not the sublayer accumulator.
  hipLaunchKernelGGL(HIP_KERNEL_NAME(hc_add_kernel<false>), dim3(blk), dim3(thr), 0, D.stream,
                     D.d_block, D.d_x, n_embd);
  AFF_HIP_CHECK(hipGetLastError());
  return true;
}

// The sublayer's one exchange. Every Col matrix — wo_b, shexp_down, every resident expert's down —
// has left a PARTIAL in its own card's block; this is where the two become the one value both cards
// need before hc_post folds it into the replicated hidden state.
bool DenseGpu::block_reduce(uint32_t n_embd) {
  if (!hc_ok_) return false;
  // Defer rather than issue: hc_post runs next at every call site, visits exactly the elements this
  // would have written, and partitions them the same way. Nothing may run in between — and nothing
  // does — so the only state carried is "the next hc_post owes an exchange". If hc_post finds it
  // set for a different length it falls back rather than guessing.
  //
  // The fold is a TWO-RANK path — the kernel that carries it knows one peer — so tp_fold_prepare
  // declines at any other rank count and this falls through to the standalone collective, which
  // costs the launch the fold exists to save and is otherwise identical.
  if (impl_ && impl_->tp.ready && impl_->dev.size() == 2) {
    std::string err;
    if (tp_fold_prepare(&impl_->tp, n_embd, &err)) {
      ar_pending_ = n_embd;
      return true;
    }
    static bool warned = false;
    if (!warned) {
      aff::ui::err("dense-gpu: the fold buffers would not allocate (%s) — using the "
                   "separate exchange\n", err.c_str());
      warned = true;
    }
  }
  float* p[kMaxRanks];
  for (size_t d = 0; d < impl_->dev.size(); ++d) p[d] = impl_->dev[d].d_block;
  reduce(p, p, n_embd);
  return true;
}

int DenseGpu::device(size_t r) const {
  return (impl_ && r < impl_->dev.size()) ? impl_->dev[r].id : -1;
}
void* DenseGpu::stream(size_t r) const {
  return (impl_ && r < impl_->dev.size()) ? (void*)impl_->dev[r].stream : nullptr;
}
float* DenseGpu::batch_norm_dev(size_t r) const {
  return (impl_ && r < impl_->dev.size()) ? impl_->dev[r].b_norm : nullptr;
}
float* DenseGpu::batch_block_dev(size_t r) const {
  return (impl_ && r < impl_->dev.size()) ? impl_->dev[r].b_block : nullptr;
}

bool DenseGpu::hc_post(bool has_control, uint32_t n_embd, uint32_t n_hc) {
  AFF_TRACE(8);
  if (!hc_ok_) return false;
  const uint32_t thr = 256, blk = (n_embd + thr - 1) / thr;
  // An exchange block_reduce deferred to this call, and whether its shape can carry it: the
  // deferred length must be the one about to be mapped, the grid must fit the per-block flag array
  // (block b uses flag b), and the control path needs a compile-time n_hc. Anything else issues the
  // exchange here and falls through to the pair below — a silently dropped deferral would leave the
  // block half-summed, which is a plausible-looking output rather than a crash.
  const bool ar_fits = ar_pending_ == n_embd && blk <= (uint32_t)kTpMaxBlocks &&
                       (!has_control || n_hc == 1 || n_hc == 2 || n_hc == 4 || n_hc == 8);
  const bool ar = ar_pending_ != 0 && ar_fits;
  if (ar_pending_ && !ar_fits) {
    float* p[kMaxRanks];
    for (size_t r = 0; r < impl_->dev.size(); ++r) p[r] = impl_->dev[r].d_block;
    reduce(p, p, ar_pending_);
  }
  ar_pending_ = 0;                                    // cleared on EVERY path out of here
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    // The peer's index. `ar` is only ever set at TWO ranks — block_reduce checks the count before
    // arming it and tp_fold_prepare refuses otherwise — so `1 - d` is in range exactly when it is
    // read. The fold's buffers are per rank as tp_all_reduce's are, and the kernel writes the PEER's
    // scratch and reads its OWN.
    const size_t pr = ar ? 1 - d : 0;
    float* const f_peer = ar ? impl_->tp.fscratch[pr] : nullptr;
    const float* const f_mine = ar ? impl_->tp.fscratch[d] : nullptr;
    unsigned* const fl_peer = ar ? impl_->tp.fflags[pr] : nullptr;
    unsigned* const fl_mine = ar ? impl_->tp.fflags[d] : nullptr;
    unsigned* const f_seq = ar ? impl_->tp.fseq[d] : nullptr;
    const uint32_t f_stride = ar ? (uint32_t)impl_->tp.fold_floats : 0;
    if (has_control && ar) {
      // Bit-identical to the reduce-then-expand pair below.
      switch (n_hc) {
        case 1: hipLaunchKernelGGL(HIP_KERNEL_NAME(hc_expand_ar_kernel<1>), dim3(blk), dim3(thr), 0,
                  D.stream, D.d_hc_next, D.d_block, D.d_hc, D.d_post, D.d_comb, n_embd, f_peer,
                  f_mine, fl_peer, fl_mine, f_seq, f_stride); break;
        case 2: hipLaunchKernelGGL(HIP_KERNEL_NAME(hc_expand_ar_kernel<2>), dim3(blk), dim3(thr), 0,
                  D.stream, D.d_hc_next, D.d_block, D.d_hc, D.d_post, D.d_comb, n_embd, f_peer,
                  f_mine, fl_peer, fl_mine, f_seq, f_stride); break;
        case 8: hipLaunchKernelGGL(HIP_KERNEL_NAME(hc_expand_ar_kernel<8>), dim3(blk), dim3(thr), 0,
                  D.stream, D.d_hc_next, D.d_block, D.d_hc, D.d_post, D.d_comb, n_embd, f_peer,
                  f_mine, fl_peer, fl_mine, f_seq, f_stride); break;
        default: hipLaunchKernelGGL(HIP_KERNEL_NAME(hc_expand_ar_kernel<4>), dim3(blk), dim3(thr),
                  0, D.stream, D.d_hc_next, D.d_block, D.d_hc, D.d_post, D.d_comb, n_embd, f_peer,
                  f_mine, fl_peer, fl_mine, f_seq, f_stride); break;
      }
      AFF_HIP_CHECK(hipGetLastError());
      std::swap(D.d_hc, D.d_hc_next);
    } else if (ar) {
      hipLaunchKernelGGL(hc_add_ar_kernel, dim3(blk), dim3(thr), 0, D.stream, D.d_hc, D.d_block,
                         n_embd, f_peer, f_mine, fl_peer, fl_mine, f_seq, f_stride);
      AFF_HIP_CHECK(hipGetLastError());
    } else if (has_control) {
      // n_hc is a TEMPLATE argument here, not a runtime one: the runtime-n_hc kernel it replaced
      // re-read hc once per destination and could not unroll either loop. Bit-identical, so the
      // switch below is the whole difference.
#define AFF_HC_XP_LAUNCH(NH)                                                                     \
  hipLaunchKernelGGL(HIP_KERNEL_NAME(hc_expand_reg_kernel<NH>), dim3(blk), dim3(thr), 0,         \
                     D.stream, D.d_hc_next, D.d_block, D.d_hc, D.d_post, D.d_comb, n_embd)
      if (n_hc == 1)      { AFF_HC_XP_LAUNCH(1); }
      else if (n_hc == 2) { AFF_HC_XP_LAUNCH(2); }
      else if (n_hc == 8) { AFF_HC_XP_LAUNCH(8); }
      else                { AFF_HC_XP_LAUNCH(4); }
      AFF_HIP_CHECK(hipGetLastError());
#undef AFF_HC_XP_LAUNCH
      std::swap(D.d_hc, D.d_hc_next);
    } else {
      hipLaunchKernelGGL(HIP_KERNEL_NAME(hc_add_kernel<true>), dim3(blk), dim3(thr), 0, D.stream,
                         D.d_hc, D.d_block, n_embd);
      AFF_HIP_CHECK(hipGetLastError());
    }
  }
  return true;
}

// Rows layer `l` needs to hold `pos` positions. a_comp_rows_[l] was computed by the caller as
// max_pos/ratio + 2, so the ratio is recoverable from the pair rather than passed again — and the
// +2 slack rides along, which is what the compressor's cross-chunk state needs.
uint32_t DenseGpu::comp_rows_for(uint32_t l, uint64_t pos) const {
  const uint32_t full = a_comp_rows_[l];
  if (!full || !a_max_pos_ || pos >= a_max_pos_) return full;
  const uint64_t r = (uint64_t)full * pos / a_max_pos_ + 2;
  return r >= full ? full : (uint32_t)r;
}

bool DenseGpu::kv_commit_positions(uint64_t positions, std::string* err) {
  if (!impl_ || !attn_ok_) { if (err) *err = "kv_commit_positions before attn_init"; return false; }
  // RESTORE THE CALLER'S DEVICE. This runs mid-run from two threads: the grower, where the device
  // is this thread's own business, and the prefill path, where it is the DISPATCH thread's. Leaving
  // it pointed at the last card sends the next launch to a stream that does not belong to it, which
  // surfaces as "invalid resource handle" from a batched GEMM and nothing that names this function.
  int prev = 0;
  AFF_HIP_CHECK(hipGetDevice(&prev));
  struct Restore {
    int d;
    ~Restore() { (void)hipSetDevice(d); }
  } restore{prev};
  if (positions > a_max_pos_) {
    if (err) *err = "context " + std::to_string(positions) + " exceeds --kv-size " +
                    std::to_string(a_max_pos_);
    return false;
  }
  const uint64_t rb = kv_row_bytes(a_dt_, a_width_, a_rot_);
  for (Impl::Dev& D : impl_->dev) {
    AFF_HIP_CHECK(hipSetDevice(D.id));
    for (size_t l = 0; l < D.a_comp_span_s[slot_].size(); ++l) {
      if (!D.a_comp_span_s[slot_][l].reserved()) continue;
      const size_t was = D.a_comp_span_s[slot_][l].committed();
      if (!D.a_comp_span_s[slot_][l].commit_to((uint64_t)comp_rows_for((uint32_t)l, positions) * rb,
                                      D.a_kv_zero, err))
        return false;
      st_.kv_bytes += D.a_comp_span_s[slot_][l].committed() - was;
    }
  }
  return true;
}

void DenseGpu::kv_autogrow_start(uint64_t margin) {
  if (kv_grow_run_.load(std::memory_order_relaxed)) return;
  kv_margin_.store(margin, std::memory_order_relaxed);
  kv_backed_.store(a_init_pos_, std::memory_order_release);
  kv_grow_run_.store(true, std::memory_order_release);
  kv_grow_ = std::thread([this] {
    while (kv_grow_run_.load(std::memory_order_acquire)) {
      const uint64_t pos = kv_pos_.load(std::memory_order_relaxed);
      const uint64_t m = kv_margin_.load(std::memory_order_relaxed);
      // Quantised to whole margins, so growth fires ONCE every `m` positions in one jump rather
      // than mapping a few pages on every tick as the sequence creeps forward. Each growth is 86
      // driver calls (43 layers, 2 cards) and they are only cheap because they are rare.
      const uint64_t want = std::min(((pos + m) / m + 1) * m, a_max_pos_);
      if (want > kv_backed_.load(std::memory_order_relaxed)) {
        std::string err;
        if (kv_commit_positions(want, &err)) {
          kv_backed_.store(want, std::memory_order_release);
        } else {
          // Out of VRAM to grow into. Say so once and stop trying: the dispatch checks
          // kv_backed_positions() before it writes, so this surfaces as a refusal to continue the
          // sequence rather than as a fault in an attention kernel.
          if (kv_grow_err_.empty()) {
            kv_grow_err_ = err;
            ui::err("kv cache: cannot grow past %llu positions: %s\n",
                    (unsigned long long)kv_backed_.load(std::memory_order_relaxed), err.c_str());
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  });
}

void DenseGpu::kv_autogrow_stop() {
  if (!kv_grow_run_.exchange(false, std::memory_order_acq_rel)) return;
  if (kv_grow_.joinable()) kv_grow_.join();
}

uint64_t DenseGpu::kv_committed_bytes() const {
  uint64_t t = 0;
  if (impl_)
    for (const Impl::Dev& D : impl_->dev)
      for (const VirtualSpan& s : D.a_comp_span_s[slot_]) t += s.committed();
  return t;
}

bool DenseGpu::attn_init(uint32_t n_layer, uint32_t n_head, uint32_t width, uint32_t n_rot,
                         uint32_t sliding, const uint32_t* comp_rows, KvDtype dt,
                         uint64_t max_pos, uint64_t init_pos, std::string* err) {
  if (!impl_ || impl_->dev.empty()) { if (err) *err = "no device"; return false; }
  if (!comp_rows) { if (err) *err = "no per-layer row counts"; return false; }
  const size_t R = impl_->dev.size();
  // Attention is head-parallel, so the rank count has to divide the head count. There is no partial
  // head: a rank holding three and a half of them would need the KV rows split mid-row as well.
  // 64 heads means 1, 2, 4, 8, 16, 32 or 64 ranks; --gpus is how to land on one of those with an
  // awkward number of cards installed.
  if (n_head % R) {
    std::string ok;
    for (uint32_t c = 1; c <= n_head; ++c)
      if (n_head % c == 0 && c <= (uint32_t)kMaxRanks) ok += (ok.empty() ? "" : ", ") +
                                                            std::to_string(c);
    if (err) *err = std::to_string(n_head) + " attention heads do not split " + std::to_string(R) +
                    " ways. Usable rank counts here: " + ok + " — set --gpus.";
    return false;
  }
  a_head_ = n_head; a_hpr_ = n_head / (uint32_t)R;
  // The flash kernel wants whole 16-head tiles. Past that it still runs, on the general MQA path,
  // which is correct and slower — worth saying once rather than leaving as an unexplained drop.
  if (a_hpr_ % 16u)
    aff::ui::err("attention: %u heads a card does not fill a 16-head tile, so the flash kernel is "
                 "off and the general path runs instead. Fewer ranks would keep it.\n", a_hpr_);
  a_width_ = width; a_rot_ = n_rot;
  a_sliding_ = sliding; a_dt_ = dt;
  // Per layer, sized by THAT layer's compress ratio: 21 of 43 layers are ratio 4, 20 are ratio 128
  // and need a thirty-second as much, 2 are ratio 0 and need none. Handing every layer the ratio-4
  // count was 5.55 GiB a card at a 1M cache where 2.80 is enough.
  //
  // `a_maxcomp_` is the MAXIMUM over them, because the per-token scratch that is not the cache —
  // the admission mask plane, the indexer's score plane, its key cache — is sized for the widest
  // layer any token will meet rather than for a particular one.
  a_max_pos_ = max_pos ? max_pos : 1;
  a_init_pos_ = init_pos > a_max_pos_ ? a_max_pos_ : init_pos;
  a_comp_rows_.assign(comp_rows, comp_rows + n_layer);
  a_maxcomp_ = 0;
  for (uint32_t l = 0; l < n_layer; ++l) a_maxcomp_ = std::max(a_maxcomp_, comp_rows[l]);
  a_ring_ = attn_ring_slots(sliding);
  const uint64_t rb = kv_row_bytes(dt, width, n_rot);
  // sliding + kAttnBatch - 1 slots, not `sliding` — that is what lets a whole sub-batch commit
  // before any of it attends. ~7 MB a card across 43 layers at kAttnBatch 256.
  const uint64_t raw_b = (uint64_t)a_ring_ * rb;
  uint64_t comp_total = 0;
  for (uint32_t l = 0; l < n_layer; ++l) comp_total += (uint64_t)a_comp_rows_[l] * rb;
  // The CACHE is replicated and the HEADS are split, so a card holds the whole cache and half the
  // query/output buffers — which is what keeps the sublayer free of exchanges. MQA, so a row is
  // 2 KiB and the duplicate costs about one expert of VRAM.
  for (size_t d = 0; d < R; ++d) {
    Impl::Dev& D = impl_->dev[d];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    size_t freeb = 0, totalb = 0;
    AFF_HIP_CHECK(hipMemGetInfo(&freeb, &totalb));
    // Only what is COMMITTED counts against free VRAM. The rest of the compressed cache is address
    // space, and address space is what placement is allowed to take: this is the whole point of the
    // change, since placement measures free VRAM after this call and the difference is 962 expert
    // slots at the shipped --kv-size.
    uint64_t comp_now = 0;
    for (uint32_t l = 0; l < n_layer; ++l)
      comp_now += (uint64_t)comp_rows_for(l, a_init_pos_) * rb;
    const uint64_t need = (raw_b * n_layer + comp_now) * n_slots_ +
                          (uint64_t)a_hpr_ * width * 2 * sizeof(float);
    if (freeb < need + kKvAllocHeadroom) {
      if (err) *err = "KV cache needs " + std::to_string(need >> 20) + " MiB on card " +
                      std::to_string(d) + ", only " + std::to_string(freeb >> 20) + " free";
      return false;
    }
    D.a_sink.assign(n_layer, nullptr);            // shared model state, not per-sequence
    // The slot dimension is sized HERE, not at device creation: --slots is parsed after the
    // devices exist, so init()'s sizing ran while n_slots_ was still 1.
    D.b_tap_s.assign(n_slots_, nullptr); D.tap_bytes_s.assign(n_slots_, 0);
    D.a_raw_s.resize(n_slots_);   D.a_comp_s.resize(n_slots_);
    D.a_comp_span_s.resize(n_slots_);
    D.a_idx8_s.resize(n_slots_);  D.a_idxs_s.resize(n_slots_);
    D.c_hkv_s.resize(n_slots_);   D.c_hsc_s.resize(n_slots_);
    D.ci_hkv_s.resize(n_slots_);  D.ci_hsc_s.resize(n_slots_);
    for (uint32_t sl = 0; sl < n_slots_; ++sl) {
      D.a_raw_s[sl].assign(n_layer, nullptr);
      D.a_comp_s[sl].assign(n_layer, nullptr);
      D.a_comp_span_s[sl].clear();
      D.a_comp_span_s[sl].resize(n_layer);
    }
    // ONE zeroing stream a card, shared by every layer's span. Per-span streams cost 640 MiB of
    // host RAM across 86 of them, which comes out of the expert pool and strands experts on the
    // SSD tier — the pool is sized from what is left after everything else the process holds.
    if (!D.a_kv_zero)
      AFF_HIP_CHECK(hipStreamCreateWithFlags(&D.a_kv_zero, hipStreamNonBlocking));
    for (uint32_t sl = 0; sl < n_slots_; ++sl)
    for (uint32_t l = 0; l < n_layer; ++l) {
      AFF_HIP_CHECK(hipMalloc(&D.a_raw_s[sl][l], raw_b));
      AFF_HIP_CHECK(hipMemset(D.a_raw_s[sl][l], 0, raw_b));
      const uint64_t cb = (uint64_t)a_comp_rows_[l] * rb;
      if (cb) {
        std::string verr;
        if (!D.a_comp_span_s[sl][l].reserve(D.id, cb, &verr) ||
            !D.a_comp_span_s[sl][l].commit_to((uint64_t)comp_rows_for(l, a_init_pos_) * rb,
                                        D.a_kv_zero, &verr)) {
          if (err) *err = "KV layer " + std::to_string(l) + " on card " + std::to_string(d) +
                          " slot " + std::to_string(sl) + ": " + verr;
          return false;
        }
        D.a_comp_s[sl][l] = static_cast<uint8_t*>(D.a_comp_span_s[sl][l].ptr());
      }
    }
    AFF_HIP_CHECK(hipMalloc(&D.a_q, (uint64_t)a_hpr_ * width * sizeof(float)));
    AFF_HIP_CHECK(hipMalloc(&D.a_out, (uint64_t)a_hpr_ * width * sizeof(float)));
    AFF_HIP_CHECK(hipMalloc(&D.a_allow, a_maxcomp_ ? a_maxcomp_ : 1));
    // What is BACKED, not what is reserved: the reservation is address space and the whole point
    // of it is that it costs no VRAM. A ledger that counted the reservation would report 4.3 GiB a
    // card that placement is in fact using for experts.
    st_.kv_bytes += (raw_b * n_layer + comp_now) * n_slots_;
  }
  attn_ok_ = true;
  // Everything the loader handed over before this point, now that there is somewhere to put it.
  for (uint32_t l = 0; l < n_layer; ++l) upload_sink(l);
  return true;
}

// Held on the host until attn_init, which runs after Model::load() because it needs the layer
// count, while the loader binds sinks DURING that walk. Guarding this on attn_ok_ instead dropped
// every sink and still produced fluent text: the sink seeds the max and adds exp(sink - m) to the
// denominator without contributing a value vector, so without it every attention output in every
// layer is scaled up. Against tools/ref_forward.py that was rel L2 1.192e-01 rather than 4.016e-05.
void DenseGpu::attn_set_sinks(uint32_t layer, const float* sinks, uint32_t n) {
  if (!sinks || !n) return;
  if (a_sink_host_.size() <= layer) a_sink_host_.resize(layer + 1);
  a_sink_host_[layer].assign(sinks, sinks + n);
  if (attn_ok_) upload_sink(layer);
}

void DenseGpu::upload_sink(uint32_t layer) {
  if (layer >= a_sink_host_.size()) return;
  const std::vector<float>& h = a_sink_host_[layer];
  if (h.empty()) return;
  // One sink per head, so each card takes the slice belonging to the heads it owns — the attend
  // kernel indexes them from zero against the heads it was given.
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    if (layer >= D.a_sink.size()) continue;
    AFF_HIP_CHECK(hipSetDevice(D.id));
    const size_t off = std::min(h.size(), (size_t)d * a_hpr_);
    const size_t n = std::min((size_t)a_hpr_, h.size() - off);
    if (!n) continue;
    if (!D.a_sink[layer]) AFF_HIP_CHECK(hipMalloc(&D.a_sink[layer], n * sizeof(float)));
    AFF_HIP_CHECK(hipMemcpy(D.a_sink[layer], h.data() + off, n * sizeof(float),
                            hipMemcpyHostToDevice));
  }
}

// ---- checkpointing the attention state ----------------------------------------------------------

bool DenseGpu::kv_part_bufs(KvPart part, uint32_t layer, size_t rank, uint64_t* row_bytes,
                            uint64_t* rows_each, void* bufs[2], uint32_t* n_buf) const {
  if (!impl_ || !attn_ok_ || rank >= impl_->dev.size()) return false;
  Impl::Dev& D = impl_->dev[rank];
  if (layer >= D.a_raw_s[slot_].size()) return false;
  const uint64_t rb = kv_row_bytes(a_dt_, a_width_, a_rot_);
  // reserve_runtime only grows the compressor where the layer compresses, so c_ratio is both the
  // ratio and the "is there one at all" flag — a ratio-0 layer and a draft stage both read 0.
  const uint32_t ratio = layer < D.c_ratio.size() ? D.c_ratio[layer] : 0u;
  bufs[0] = bufs[1] = nullptr;
  *n_buf = 1;
  switch (part) {
    case KvPart::Raw:
      bufs[0] = D.a_raw_s[slot_][layer];  *row_bytes = rb;  *rows_each = a_ring_;
      break;
    case KvPart::Comp:
      bufs[0] = D.a_comp_s[slot_][layer]; *row_bytes = rb;  *rows_each = a_comp_rows_[layer];
      break;
    case KvPart::Idx8:
      if (layer >= D.a_idx8_s[slot_].size()) return false;
      bufs[0] = D.a_idx8_s[slot_][layer]; *row_bytes = kIdxDim; *rows_each = a_maxcomp_;
      break;
    case KvPart::IdxScale:
      if (layer >= D.a_idxs_s[slot_].size()) return false;
      bufs[0] = D.a_idxs_s[slot_][layer]; *row_bytes = sizeof(float); *rows_each = a_maxcomp_;
      break;
    case KvPart::Hist:
      if (!ratio) return false;
      bufs[0] = D.c_hkv_s[slot_][layer];  bufs[1] = D.c_hsc_s[slot_][layer];
      *row_bytes = (uint64_t)comp_width(ratio, a_width_) * sizeof(float);
      *rows_each = comp_hist(ratio);
      *n_buf = 2;
      break;
    case KvPart::IdxHist:
      // The index compressor exists on the ratio-4 layers alone, which is the rule compress_grow
      // applies when it decides whether to allocate one.
      if (ratio != 4u) return false;
      bufs[0] = D.ci_hkv_s[slot_][layer]; bufs[1] = D.ci_hsc_s[slot_][layer];
      *row_bytes = (uint64_t)comp_width(ratio, kIdxDim) * sizeof(float);
      *rows_each = comp_hist(ratio);
      *n_buf = 2;
      break;
  }
  if (!bufs[0] || (*n_buf == 2 && !bufs[1])) return false;
  return *row_bytes != 0 && *rows_each != 0;
}

namespace {
// Cuts [row0, row0+nrows) across the equal-length buffers a part spans, handing each piece to `f`
// as (offset into the caller's host block, device address, bytes). At most two calls, and the
// second only for the rings, whose two halves are separate allocations.
template <class F>
bool kv_walk(void* const bufs[2], uint32_t n_buf, uint64_t row_bytes, uint64_t rows_each,
             uint64_t row0, uint64_t nrows, F&& f) {
  if (row0 + nrows > rows_each * (uint64_t)n_buf) return false;
  uint64_t done = 0;
  for (uint32_t i = 0; i < n_buf && done < nrows; ++i) {
    const uint64_t lo = (uint64_t)i * rows_each;
    if (row0 + done >= lo + rows_each) continue;
    const uint64_t first = row0 + done - lo;
    const uint64_t take = std::min(nrows - done, rows_each - first);
    f(done * row_bytes, (uint8_t*)bufs[i] + first * row_bytes, take * row_bytes);
    done += take;
  }
  return done == nrows;
}
}  // namespace

bool DenseGpu::kv_part_geometry(KvPart part, uint32_t layer, uint64_t* row_bytes,
                                uint64_t* rows) const {
  void* b[2]; uint32_t nb = 0; uint64_t re = 0, rbz = 0;
  const bool ok = kv_part_bufs(part, layer, 0, &rbz, &re, b, &nb);
  if (row_bytes) *row_bytes = ok ? rbz : 0;
  if (rows) *rows = ok ? re * nb : 0;
  return ok;
}

bool DenseGpu::kv_part_save(KvPart part, uint32_t layer, uint32_t rank, uint64_t row0,
                            uint64_t nrows, void* dst) const {
  if (!dst || !nrows || !impl_ || rank >= impl_->dev.size()) return false;
  void* b[2]; uint32_t nb = 0; uint64_t re = 0, rb = 0;
  if (!kv_part_bufs(part, layer, rank, &rb, &re, b, &nb)) return false;
  // Issued on the rank's own stream so the copy is ordered behind the kernels that wrote the rows,
  // and the caller's next launch behind the copy.
  Impl::Dev& D = impl_->dev[rank];
  AFF_HIP_CHECK(hipSetDevice(D.id));
  hipError_t e = hipSuccess;
  const bool ok = kv_walk(b, nb, rb, re, row0, nrows,
                          [&](uint64_t off, void* src, uint64_t bytes) {
    if (e == hipSuccess)
      e = hipMemcpyAsync((uint8_t*)dst + off, src, bytes, hipMemcpyDeviceToHost, D.stream);
  });
  AFF_HIP_CHECK(hipStreamSynchronize(D.stream));
  AFF_HIP_CHECK(e);
  return ok;
}

bool DenseGpu::kv_part_load(KvPart part, uint32_t layer, uint32_t rank, uint64_t row0,
                            uint64_t nrows, const void* src) {
  if (!src || !nrows || !impl_ || rank >= impl_->dev.size()) return false;
  void* b[2]; uint32_t nb = 0; uint64_t re = 0, rb = 0;
  if (!kv_part_bufs(part, layer, rank, &rb, &re, b, &nb)) return false;
  Impl::Dev& D = impl_->dev[rank];
  AFF_HIP_CHECK(hipSetDevice(D.id));
  hipError_t e = hipSuccess;
  const bool ok = kv_walk(b, nb, rb, re, row0, nrows,
                          [&](uint64_t off, void* dst, uint64_t bytes) {
    if (e == hipSuccess)
      e = hipMemcpyAsync(dst, (const uint8_t*)src + off, bytes, hipMemcpyHostToDevice, D.stream);
  });
  AFF_HIP_CHECK(hipStreamSynchronize(D.stream));
  AFF_HIP_CHECK(e);
  return ok;
}

bool DenseGpu::kv_part_hash(KvPart part, uint32_t layer, uint64_t row0, uint64_t nrows, size_t rank,
                            uint64_t* h) const {
  if (!h || !nrows) return false;
  void* b[2]; uint32_t nb = 0; uint64_t re = 0, rb = 0;
  if (!kv_part_bufs(part, layer, rank, &rb, &re, b, &nb)) return false;
  Impl::Dev& D = impl_->dev[rank];
  AFF_HIP_CHECK(hipSetDevice(D.id));
  // Staged a few MiB at a time: a_comp is 138 MB a layer at a 1M cache, and a diagnostic that has
  // to allocate that much to run is one that stops being run.
  std::vector<uint8_t> stage;
  uint64_t acc = *h;
  hipError_t e = hipSuccess;
  const bool ok = kv_walk(b, nb, rb, re, row0, nrows, [&](uint64_t, void* src, uint64_t bytes) {
    constexpr uint64_t kChunk = 4u << 20;
    for (uint64_t off = 0; off < bytes && e == hipSuccess; off += kChunk) {
      const uint64_t n = std::min(kChunk, bytes - off);
      stage.resize((size_t)n);
      e = hipMemcpy(stage.data(), (uint8_t*)src + off, (size_t)n, hipMemcpyDeviceToHost);
      if (e != hipSuccess) return;
      for (uint64_t i = 0; i < n; ++i) acc = (acc ^ stage[(size_t)i]) * 0x100000001b3ull;
    }
  });
  AFF_HIP_CHECK(e);
  *h = acc;
  return ok;
}

bool DenseGpu::attn_run(uint32_t layer, const float* q, const uint8_t* allowed, uint32_t pos,
                        uint32_t n_comp, float scale, float* heads_out) {
  AFF_TRACE(5);
  if (!attn_ok_ || layer >= impl_->dev[0].a_raw_s[slot_].size() || !a_ring_) return false;
  // This layer's own allocation; the caller falls back rather than truncating.
  if (n_comp > a_comp_rows_[kvl(layer)]) return false;
  // The window is a function of the position and nothing else — the same rule the batched kernel
  // derives per block, so decode and prefill cannot disagree about which rows a token may see.
  const uint32_t n_raw = std::min(pos + 1u, a_sliding_);
  const uint32_t raw0 = (pos + 1u - n_raw) % a_ring_;
  // A sink that was handed over but never reached the card is a silently wrong softmax, which is
  // exactly what this cost once. Refuse instead.
  if (layer < a_sink_host_.size() && !a_sink_host_[layer].empty() && !impl_->dev[0].a_sink[layer]) {
    aff::ui::fatal("fatal: layer %u has attention sinks that never reached the device\n",
                 layer);
    std::abort();
  }
  const uint64_t nq = (uint64_t)a_hpr_ * a_width_;    // this card's heads
  bool need_sync[kMaxRanks] = {};
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    // Every host buffer touched below is a caller local that dies at the end of the layer, so an
    // async copy out of one obliges this call to synchronise before returning. When nothing crosses
    // the bus — q already in a_q, no indexer mask, heads staying for attn_out — there is nothing to
    // wait for, and attn_out runs on this same stream already ordered behind the kernel.
    if (q) {
      AFF_HIP_CHECK(hipMemcpyAsync(D.a_q, q + (size_t)d * nq, nq * sizeof(float),
                                   hipMemcpyHostToDevice, D.stream));
      need_sync[d] = true;
    }
    // The indexer writes a_allow for its OWN layer only. Keyed by layer rather than a bare flag: a
    // stale one would hand attention the previous layer's admissions, which is not a crash — it is
    // a model that quietly attends to the wrong rows.
    const bool dev_mask = a_idx_layer_[slot_] == (int64_t)layer && n_comp;
    if (!dev_mask && allowed && n_comp) {
      // The compressed-row mask is per TOKEN, not per head, so both cards get all of it.
      AFF_HIP_CHECK(hipMemcpyAsync(D.a_allow, allowed, n_comp, hipMemcpyHostToDevice, D.stream));
      need_sync[d] = true;
    }
    // The same fp8 WMMA kernel the batched path uses, at n = 1. The VALU kernel below it walks keys
    // one at a time — a serial chain of one dependent byte load per key — so its cost is linear in
    // context where flash's is not, and the gap widens as the cache grows. At one token the tiled
    // dimension is HEADS, which is what makes this work: MQA means many query heads share one key
    // row, and flash wants whole 16-head tiles.
    const uint8_t* mask = (dev_mask || (allowed && n_comp)) ? D.a_allow : nullptr;
    const bool flash = a_dt_ == KvDtype::FP8R && a_width_ == kFlashWidth && (a_hpr_ % 16u) == 0u;
    if (flash) {
      // The count goes in as a kernel argument, not a device word: as an upload it is a blit kernel
      // and a launch boundary per layer per rank, to store one integer.
      //
      // nb = 1 and b0 = 0: a_q and a_out are one token's [n_head][width], which IS the token-major
      // layout flash wants at a single token. The window comes from `pos` exactly as in the batched
      // kernel, so decode and prefill cannot disagree about which rows are visible.
      //
      // GATHERED when the indexer left a list for this layer: attention walks the sliding window
      // plus at most index_topk admitted rows rather than every compressed row in the cache. At
      // 123K context that is 640 key rows against 30848.
      const bool gather = dev_mask && a_idx_gather_[slot_];
      mqa_attend_flash_hip(D.a_q, D.a_raw_s[slot_][layer], D.a_comp_s[slot_][kvl(layer)], gather ? nullptr : mask,
                           gather ? D.a_admn : nullptr, gather ? 0u : n_comp,
                           D.a_sink[layer],
                           D.a_out, 0, 1, 1, pos, a_sliding_, a_ring_, a_maxcomp_, a_hpr_, scale,
                           D.stream, /*q_bf16=*/false, &D.fsplit, gather ? a_idx_topk_[slot_] : 0u,
                           gather ? D.a_adm : nullptr, kAdmCap);
    } else {
      mqa_attend_hip(a_dt_, D.a_q, D.a_raw_s[slot_][layer], D.a_comp_s[slot_][kvl(layer)], mask, D.a_sink[layer], D.a_out,
                     a_hpr_, raw0, n_raw, a_ring_, n_comp, a_width_, a_rot_, scale, D.stream);
    }
    // heads_out is null when the caller will run attn_out() next and the heads should stay put.
    if (heads_out) {
      AFF_HIP_CHECK(hipMemcpyAsync(heads_out + (size_t)d * nq, D.a_out, nq * sizeof(float),
                                   hipMemcpyDeviceToHost, D.stream));
      need_sync[d] = true;
    }
  }
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    if (!need_sync[d]) continue;
    AFF_HIP_CHECK(hipSetDevice(impl_->dev[d].id));
    AFF_DRAIN(impl_->dev[d]);
  }
  ++st_.calls;
  return true;
}

// The end of the attention sublayer and its ONE exchange. wo_a is Row and its groups follow the
// heads — 4 of the 8 groups on each card, because a group is 8 heads and a card owns 32 — so its
// output is this card's half of the concatenated [n_groups*rank] vector. wo_b is Col over exactly
// that half, which makes each card's result a partial over the whole hidden dimension, and the
// partials are left in the blocks for block_reduce.
bool DenseGpu::attn_out(int32_t hwa, int32_t hwb, uint32_t n_head, uint32_t width, uint32_t n_rot,
                        uint32_t pos, RopeDerived rope, uint32_t n_groups, uint64_t rank,
                        uint64_t hidden, float* block_out) {
  AFF_TRACE(6);
  if (!attn_ok_ || hwa < 0 || hwb < 0) return false;
  if ((size_t)hwa >= ent_.size() || (size_t)hwb >= ent_.size()) return false;
  const Entry& A = ent_[hwa]; const Entry& B = ent_[hwb];
  if (A.split != Split::Row || B.split != Split::Col) return false;
  // The one-launch grouped path needs each group to start on a scale-tile row.
  if (A.quant != AFF_FP8_E4M3 || (rank % 128)) return false;
  const size_t R = impl_->dev.size();
  const uint64_t group_dim = (uint64_t)(n_head / n_groups) * width;
  for (size_t d = 0; d < R; ++d) {
    // wo_a's row band has to be a whole number of groups, and it has to be the same band of the
    // concatenation that wo_b's column band consumes. Both are cut independently in reg().
    if (A.sh[d].row0 % rank || A.sh[d].nrows % rank) return false;
    if (A.sh[d].nrows != B.sh[d].ncols || A.sh[d].row0 != B.sh[d].col0) return false;
  }

  float* in[kMaxRanks]; float* outp[kMaxRanks];
  for (size_t d = 0; d < R; ++d) {
    Impl::Dev& D = impl_->dev[d];
    const uint32_t ng = (uint32_t)(A.sh[d].nrows / rank);   // groups on this card
    AFF_HIP_CHECK(hipSetDevice(D.id));
    // V carried the rotated tail, so the rotation has to be undone on the heads attention produced.
    // wo_a is the only thing that reads them again, so it undoes it on the way in rather than paying
    // a tiny launch of its own to do it in place first.
    no_ps(A.sh[d].ps, "the grouped wo_a matvec");
    if (!matvec_fp8_grouped_rope_hip(
            (const uint8_t*)A.sh[d].data, A.sh[d].scales, D.a_out, ng, (uint32_t)group_dim,
            (uint32_t)rank, width, n_rot, pos, rope.freq_scale, rope.theta_scale, rope.ext_factor,
            rope.lo, rope.hi, D.d_g, D.stream)) {
      const uint32_t pairs = n_rot / 2, thr = 256;
      const uint32_t blk = (a_hpr_ * pairs + thr - 1) / thr;
      hipLaunchKernelGGL(rope_tail_kernel, dim3(blk), dim3(thr), 0, D.stream, D.a_out, a_hpr_,
                         width, n_rot, pos, rope.freq_scale, rope.theta_scale, rope.ext_factor,
                         rope.lo, rope.hi, -1.0f);     // sine negated: this is the INVERSE
      AFF_HIP_CHECK(hipGetLastError());
      matvec_fp8_grouped_hip((const uint8_t*)A.sh[d].data, A.sh[d].scales, D.a_out, ng,
                             (uint32_t)group_dim, (uint32_t)rank, D.d_g, D.stream);
    }
    launch_mv({B.sh[d].data, B.sh[d].scales, B.quant, B.sh[d].ps}, D.d_g, (uint32_t)hidden,
              (uint32_t)B.sh[d].ncols, block_out ? D.d_p : D.d_block, D.stream);
    in[d] = block_out ? D.d_p : D.d_block;
    outp[d] = in[d];
  }
  if (block_out) {
    // The host wants the whole vector rather than a partial, so reduce here instead of leaving it
    // to the sublayer's block_reduce.
    reduce(in, outp, hidden);
    Impl::Dev& D = impl_->dev[0];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    AFF_HIP_CHECK(hipMemcpyAsync(D.h_y, D.d_p, hidden * sizeof(float), hipMemcpyDeviceToHost,
                                 D.stream));
    AFF_DRAIN(D);
    std::memcpy(block_out, D.h_y, hidden * sizeof(float));
  }
  ++st_.calls;
  return true;
}

void DenseGpu::mv(int32_t h, const float* x, uint64_t rows, uint64_t cols, float* y) {
  AFF_TRACE(0);
  if (h < 0 || (size_t)h >= ent_.size()) return;
  const Entry& e = ent_[h];
  Impl& I = *impl_;

  // Col and Rep do not partition the OUTPUT, so the host-facing concatenation below does not apply
  // to them. This is the unfused fallback — the chains that own those tensors keep them on device
  // — so it is written for correctness rather than for speed: reduce, then take rank 0's copy.
  if (e.split != Split::Row) {
    float* in[kMaxRanks]; float* out[kMaxRanks];
    for (size_t d = 0; d < I.dev.size(); ++d) {
      const Shard& s = e.sh[d];
      AFF_HIP_CHECK(hipSetDevice(I.dev[d].id));
      AFF_STAGE_UP(I.dev[d], kStgMv, I.dev[d].d_x, x + s.col0, s.ncols, I.dev[d].stream);
      launch_mv({s.data, s.scales, e.quant, s.ps}, I.dev[d].d_x, (uint32_t)rows, (uint32_t)s.ncols,
                I.dev[d].d_y, I.dev[d].stream);
      in[d] = I.dev[d].d_y; out[d] = I.dev[d].d_y;
    }
    if (e.split == Split::Col) reduce(in, out, rows);
    AFF_HIP_CHECK(hipSetDevice(I.dev[0].id));
    AFF_HIP_CHECK(hipMemcpyAsync(I.dev[0].h_y, I.dev[0].d_y, rows * sizeof(float),
                                 hipMemcpyDeviceToHost, I.dev[0].stream));
    AFF_DRAIN(I.dev[0]);
    std::memcpy(y, I.dev[0].h_y, rows * sizeof(float));
    ++st_.calls;
    return;
  }

  // Issue every rank before synchronising any of them, so the two cards overlap. Synchronising
  // inside the loop would serialise them and give back exactly the parallelism this is for.
  for (size_t d = 0; d < I.dev.size(); ++d) {
    const Shard& s = e.sh[d];
    if (!s.nrows || s.row0 >= rows) continue;
    const uint64_t n = std::min(s.nrows, rows - s.row0);
    AFF_HIP_CHECK(hipSetDevice(I.dev[d].id));
    AFF_STAGE_UP(I.dev[d], kStgMv, I.dev[d].d_x, x, cols, I.dev[d].stream);
    no_ps(s.ps, "mv");
    if (e.quant == AFF_FP8_E4M3)
      matvec_fp8_hip((const uint8_t*)s.data, s.scales, I.dev[d].d_x, (uint32_t)n, (uint32_t)cols,
                     I.dev[d].d_y, I.dev[d].stream);
    else
      matvec_bf16_hip((const uint16_t*)s.data, I.dev[d].d_x, (uint32_t)n, (uint32_t)cols,
                      I.dev[d].d_y, I.dev[d].stream);
    AFF_HIP_CHECK(hipMemcpyAsync(I.dev[d].h_y, I.dev[d].d_y, n * sizeof(float),
                                 hipMemcpyDeviceToHost, I.dev[d].stream));
  }
  for (size_t d = 0; d < I.dev.size(); ++d) {
    const Shard& s = e.sh[d];
    if (!s.nrows || s.row0 >= rows) continue;
    AFF_HIP_CHECK(hipSetDevice(I.dev[d].id));
    AFF_DRAIN(I.dev[d]);
    std::memcpy(y + s.row0, I.dev[d].h_y, std::min(s.nrows, rows - s.row0) * sizeof(float));
  }
  ++st_.calls;
}

void DenseGpu::mvg(int32_t h, const float* x, uint32_t n_groups, uint64_t group_dim, uint64_t rank,
                   float* y) {
  AFF_TRACE(0);
  if (h < 0 || (size_t)h >= ent_.size()) return;
  const Entry& e = ent_[h];
  Impl& I = *impl_;
  const uint64_t sc_cols = (group_dim + 127) / 128;

  // The groups are contiguous row blocks, so the row split already IS a group split as long as a
  // shard boundary lands on one. rank is 1024 here and the shard boundary is 4096, so rank 0 owns
  // groups 0-3 and rank 1 owns 4-7. The guard below refuses rather than reading across a boundary.
  for (size_t d = 0; d < I.dev.size(); ++d) {
    const Shard& s = e.sh[d];
    if (!s.nrows) continue;
    if (s.row0 % rank || s.nrows % rank) {
      static bool warned = false;
      if (!warned) {
        aff::ui::err("dense-gpu: grouped shard is not group-aligned (row0 %llu, rows %llu,"
                     " rank %llu) — leaving this tensor on the CPU\n",
                     (unsigned long long)s.row0, (unsigned long long)s.nrows,
                     (unsigned long long)rank);
        warned = true;
      }
      return;
    }
    AFF_HIP_CHECK(hipSetDevice(I.dev[d].id));
    const uint32_t g0 = (uint32_t)(s.row0 / rank), gn = (uint32_t)(s.nrows / rank);
    // The WHOLE input, staged once. Per-group staging would race: the host memcpy for group g+1 is
    // not ordered against group g's in-flight H2D out of the same buffer, and nothing on the stream
    // says otherwise. One buffer, distinct offsets, no ordering question.
    no_ps(s.ps, "mvg");
    float* const pin = I.dev[d].stg[kStgMvg].h;
    AFF_HIP_CHECK(hipEventSynchronize(I.dev[d].stg[kStgMvg].ev));
    std::memcpy(pin, x, (size_t)n_groups * group_dim * sizeof(float));
    for (uint32_t g = g0; g < g0 + gn && g < n_groups; ++g) {
      const uint64_t local = (uint64_t)(g - g0) * rank;          // row offset inside this shard
      AFF_HIP_CHECK(hipMemcpyAsync(I.dev[d].d_x, pin + (uint64_t)g * group_dim,
                                   group_dim * sizeof(float), hipMemcpyHostToDevice,
                                   I.dev[d].stream));
      if (e.quant == AFF_FP8_E4M3)
        matvec_fp8_hip((const uint8_t*)s.data + local * group_dim,
                       s.scales + (local / 128) * sc_cols, I.dev[d].d_x,
                       (uint32_t)rank, (uint32_t)group_dim, I.dev[d].d_y, I.dev[d].stream);
      else
        matvec_bf16_hip((const uint16_t*)s.data + local * group_dim, I.dev[d].d_x,
                        (uint32_t)rank, (uint32_t)group_dim, I.dev[d].d_y, I.dev[d].stream);
      AFF_HIP_CHECK(hipMemcpyAsync(I.dev[d].h_y + (uint64_t)g * rank, I.dev[d].d_y,
                                   rank * sizeof(float), hipMemcpyDeviceToHost, I.dev[d].stream));
    }
    AFF_HIP_CHECK(hipEventRecord(I.dev[d].stg[kStgMvg].ev, I.dev[d].stream));
  }
  for (size_t d = 0; d < I.dev.size(); ++d) {
    const Shard& s = e.sh[d];
    if (!s.nrows) continue;
    AFF_HIP_CHECK(hipSetDevice(I.dev[d].id));
    AFF_DRAIN(I.dev[d]);
    const uint32_t g0 = (uint32_t)(s.row0 / rank), gn = (uint32_t)(s.nrows / rank);
    const uint32_t g1 = std::min<uint32_t>(g0 + gn, n_groups);
    if (g1 > g0)
      std::memcpy(y + (uint64_t)g0 * rank, I.dev[d].h_y + (uint64_t)g0 * rank,
                  (size_t)(g1 - g0) * rank * sizeof(float));
  }
  ++st_.calls;
}


// ==================================================================================================
// Batched prefill
// ==================================================================================================
//
// Same weights, same KV cache, same arithmetic; a chunk of tokens instead of one. The buffers here
// are a SECOND arena — nothing touches d_hc, d_norm or d_block — so a decode step issued after a
// batched prefill runs byte for byte the code it always did.
//
// The tensor-parallel shape is identical to the single-token path above and for the same reasons.
// What differs is only the size of the exchange — [n_embd][nb] instead of [n_embd] — which is large
// enough that the FP8 crossing pays. tp_all_reduce picks that on its own.
//
// Every entry point here is a plural of one above it; where a comment there explains why a step
// exists, it is not repeated.

bool DenseGpu::batch_init(uint32_t max_tokens, uint32_t n_embd, uint32_t n_hc, uint32_t n_head,
                          uint32_t width, uint32_t max_mid, uint32_t vocab, std::string* err) {
  if (!impl_ || impl_->dev.empty() || !max_tokens) return false;
  const size_t R = impl_->dev.size();
  if (n_head % R) { if (err) *err = "n_head does not split across ranks"; return false; }
  // Round to the GEMM's token tile: a chunk that is not a multiple of it pays for a whole extra
  // pass over the weights to compute padding columns. `batch_round` is the one place that decides
  // this, and `batch_begin` calls the same function -- see the note there.
  const uint32_t nb = batch_round(max_tokens);

  // Only this card's heads, which is what halves the two largest buffers in the arena.
  const uint64_t nq  = (uint64_t)(n_head / R) * width;
  const uint64_t hcd = (uint64_t)n_embd * n_hc;
  const uint64_t nout = 2ull * n_hc + (uint64_t)n_hc * n_hc;
  for (size_t d = 0; d < R; ++d) {
    Impl::Dev& D = impl_->dev[d];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    struct Alloc { float** p; uint64_t rows; const char* name; };
    const Alloc as[] = {
      {&D.b_mix, nout, "mix"},
      {&D.b_pre, n_hc, "pre"}, {&D.b_pre_prev, n_hc, "pre prev"}, {&D.b_post, n_hc, "post"}, {&D.b_comb, (uint64_t)n_hc * n_hc, "comb"},
      {&D.b_cur, n_embd, "cur"}, {&D.b_norm, n_embd, "norm"}, {&D.b_block, n_embd, "block"},
      // Rows, not tokens: TWO [kHcnChunks][nb] planes and the loop below multiplies by nb. Two,
      // because the expand kernel reads one to re-derive the gates and writes the other.
      {&D.b_hcp, 2 * kHcnChunks, "hc partials"},
      {&D.b_emb, n_embd, "emb"},
      {&D.b_q, nq, "q"}, {&D.b_kv, width, "kv"},
      // One float a head a token: the per-head RMS scale, handed from the pass that can compute it
      // to the transpose that has to touch every element anyway. 32 rows of 123440.
      {&D.b_hinv, nq / width, "hinv"},
      // b_g is TWO max_mid bands: a fused pair writes its second job right after the first, and
      // the consumer reads it there instead of from b_u. b_u stays for the unfused paths.
      {&D.b_qr, max_mid, "qr"}, {&D.b_g, 2 * max_mid, "g"}, {&D.b_u, max_mid, "u"},
      {&D.b_y, max_mid, "y"}, {&D.b_tm, max_mid, "tm"},
    };
    uint64_t arena_rows = 0;
    for (const Alloc& a : as) {
      if (!hip_try(hipMalloc(a.p, a.rows * nb * sizeof(float)), "hipMalloc batch", err))
        return false;
      AFF_HIP_CHECK(hipMemset(*a.p, 0, a.rows * nb * sizeof(float)));
      arena_rows += a.rows;
    }
    // The hyper-connection lanes, in bf16 and therefore not in the f32 list above. Two of them,
    // hcd = n_embd * n_hc rows each. Holding them in bf16 rather than f32 takes a quarter off the
    // arena, for a buffer whose only GEMM consumer already reads it as bf16.
    struct BAlloc { uint16_t** p; uint64_t rows; };
    const BAlloc bs[] = {{&D.b_hc, hcd}, {&D.b_hc_next, hcd}, {&D.b_qtm, nq}};
    uint64_t brows = 0;
    for (const BAlloc& a : bs) {
      if (!hip_try(hipMalloc(a.p, a.rows * nb * sizeof(uint16_t)), "hipMalloc batch bf16", err))
        return false;
      AFF_HIP_CHECK(hipMemset(*a.p, 0, a.rows * nb * sizeof(uint16_t)));
      brows += a.rows;
    }
    const uint64_t arena_bytes = arena_rows * nb * sizeof(float) + brows * nb * sizeof(uint16_t);
    // ---- two aliases, and why they are safe ----------------------------------------------------
    //
    // The four head-sized buffers are 4 * nq f32 rows a token, a large share of the arena, and the
    // arena caps the prefill chunk — which decides how many times a layer's 2.875-bpw expert
    // weights are staged and dequantised over one prompt. Two of the four are never live at the
    // same time as their partner. A layer runs qkv -> attend -> attn_out on one stream:
    //
    //   b_q     written by wq_b, roped, then READ as the source of the transpose to b_qtm.
    //           Dead from that transpose onward.
    //   b_htm   first WRITTEN by mqa_attend, which runs after that transpose.   -> alias b_q
    //   b_qtm   written by that transpose, READ by mqa_attend. Dead after it.
    //   b_heads first WRITTEN by the transpose in attn_out, which runs after.   -> alias b_qtm
    //
    // Neither transpose has the same buffer on both sides: attn_out reads b_htm (== b_q) and writes
    // b_heads (== b_qtm), qkv reads b_q and writes b_qtm. Across layers the next qkv writes b_q
    // only after this attn_out has finished with b_heads. Verified against a separately-allocated
    // pair, with both allocated in both arms so every subsequent address stayed identical, and the
    // emitted text was unchanged.
    //
    // The generated text DOES move when the arena's size changes at all, and that is not a bug:
    // free VRAM decides which experts the static placement holds, a resident expert is computed by
    // the 2.875-bpw GPU kernel and a non-resident one by the host kernel, and the two round
    // differently. Only an A/B that holds the allocations fixed says anything about a change here.
    D.b_htm   = D.b_q;
    D.b_heads = D.b_qtm;
    // The chunk is the biggest lever on the routed-expert GEMM and is capped by this number, so it
    // belongs in the log next to the free VRAM it competes with. Rows-per-token too, because that
    // is what a shrink has to move.
    if (d == 0) {
      size_t freeb = 0, totb = 0;
      (void)hipMemGetInfo(&freeb, &totb);
      aff::ui::err("prefill arena: %.2f GiB/card at %u tokens (%llu f32 + %llu bf16 rows a token, "
                   "%.0f KiB), %.2f GiB free after\n",
                   (double)arena_bytes / 1073741824.0, nb,
                   (unsigned long long)arena_rows, (unsigned long long)brows,
                   (double)arena_bytes / (double)nb / 1024.0, (double)freeb / 1073741824.0);
    }
    if (!hip_try(hipHostMalloc(&D.h_tm, (size_t)max_mid * nb * sizeof(float), hipHostMallocDefault),
                 "hipHostMalloc batch staging", err))
      return false;
    // One mask slot per token in the chunk, device only: `batch_indexer` writes this in place on
    // the same stream, so the mask never crosses the bus. There is no host staging buffer — at a
    // large KV cache it would be the biggest single allocation here, and nothing reads it.
    if (a_maxcomp_) {
      if (!hip_try(hipMalloc(&D.b_allow, (size_t)nb * a_maxcomp_), "hipMalloc batch masks", err))
        return false;
    }
    if (!hip_try(hipMalloc(&D.b_ncomp, (size_t)nb * sizeof(uint32_t)), "hipMalloc ncomp", err))
      return false;
    // The block head's output plane and the argmax that reads it. Here rather than in draft_init:
    // the verify block's head is the TARGET's, so with the draft off nothing would allocate them
    // and batch_head would refuse every block on `vocab > b_vocab_`. The draft's own head writes
    // this same plane.
    //
    // kNarrowTok columns, not nb: a verify block is at most kNarrowTok wide, so this is 8.3 MB a
    // card rather than the 634 MB a prompt chunk's width would ask for.
    if (!hip_try(hipMalloc(&D.b_head, (size_t)vocab * kNarrowTok * sizeof(float)), "block head", err))
      return false;
    AFF_HIP_CHECK(hipMemset(D.b_head, 0, (size_t)vocab * kNarrowTok * sizeof(float)));
    if (!hip_try(hipMalloc(&D.b_amx, 2u * kNarrowTok * sizeof(float)), "argmax", err)) return false;
    // The cross-card combine's three, on every card because `batch_head` runs the exchange on rank
    // 0 and the peer copy needs a destination there — 320 bytes a card for a round trip a block.
    if (!hip_try(hipMalloc(&D.b_apeer, (kMaxRanks - 1) * 2u * kNarrowTok * sizeof(float)),
                 "argmax peer", err))
      return false;
    if (!hip_try(hipMalloc(&D.b_hids, (size_t)kNarrowTok * sizeof(uint32_t)), "block ids", err))
      return false;
    if (!hip_try(hipHostMalloc(&D.h_hids, (size_t)kNarrowTok * sizeof(uint32_t),
                               hipHostMallocDefault), "block ids staging", err))
      return false;
    if (!hip_try(hipEventCreateWithFlags(&D.ev_amx, hipEventDisableTiming), "argmax event", err))
      return false;
  }
  // The chunk's block is the largest thing that ever crosses between the cards.
  if (!tp_grow(&impl_->tp, (size_t)n_embd * nb, err)) return false;
  b_cap_ = nb;
  b_mid_ = max_mid;
  b_vocab_ = vocab;
  return true;
}

// Everything the DSpark draft allocates, in one place and once, after batch_init and before static
// placement — the same rule every other dense allocation here follows, because placement decides
// how many expert shards fit by asking the driver what is free.
//
// Nothing here scales with the prefill chunk. A draft is `block` tokens and a verify pass is
// `block + 1`, both under kNarrowTok, so every plane is kNarrowTok columns wide: the vocabulary
// head's would be 634 MB a card at a prompt chunk's width and is 8.3 MB at this one.
bool DenseGpu::draft_init(uint32_t n_embd, uint32_t tap_in, uint32_t vocab, uint32_t markov_rank,
                          std::string* err) {
  if (!impl_ || impl_->dev.empty()) { if (err) *err = "no device"; return false; }
  if (!b_cap_) { if (err) *err = "batch_init must run first"; return false; }
  if (!vocab || !tap_in || !n_embd) { if (err) *err = "draft shapes are zero"; return false; }
  (void)markov_rank;
  uint64_t bytes = 0;
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    struct A { float** p; uint64_t rows; const char* name; };
    const A as[] = {
      {&D.b_dxt, tap_in, "draft tap staging"},     // token-major, as the host holds the tap
      {&D.b_dx, tap_in, "draft tap"},              // dim-major, as the GEMM reads it
      {&D.b_dm, n_embd, "draft main_x"},
    };
    for (const A& a : as) {
      if (!hip_try(hipMalloc(a.p, a.rows * kNarrowTok * sizeof(float)), a.name, err)) return false;
      AFF_HIP_CHECK(hipMemset(*a.p, 0, a.rows * kNarrowTok * sizeof(float)));
      bytes += a.rows * kNarrowTok * sizeof(float);
    }
    if (!hip_try(hipMalloc(&D.b_mbest, sizeof(uint64_t)), "markov best", err)) return false;
    if (!hip_try(hipMalloc(&D.b_mpeer, (kMaxRanks - 1) * sizeof(uint64_t)), "markov peer keys", err))
      return false;
    if (!hip_try(hipMalloc(&D.b_ids, (size_t)(kNarrowTok + 1) * sizeof(uint32_t)),
                 "markov id chain", err))
      return false;
    if (!hip_try(hipHostMalloc(&D.h_ids, (size_t)kNarrowTok * sizeof(uint32_t),
                               hipHostMallocDefault), "markov ids readback", err))
      return false;
    for (hipEvent_t& e : D.ev_mk)
      if (!hip_try(hipEventCreateWithFlags(&e, hipEventDisableTiming), "markov event", err))
        return false;
    // Four bytes a slot, so the depth is the ring's maximum whatever the budget says.
    if (!D.tok.fit(sizeof(uint32_t), kRingCountBudget)) {
      if (err) *err = "markov token ring";
      return false;
    }
  }
  // b_head, b_amx and b_vocab_ are batch_init's — the draft writes the target's plane.
  if (vocab > b_vocab_) { if (err) *err = "draft vocab exceeds the block head"; return false; }
  b_tapin_ = tap_in;
  aff::ui::err("draft arena: %.1f MiB/card\n", (double)bytes / 1048576.0);
  return true;
}

// Everything the compressor and the indexer would otherwise allocate on FIRST USE, taken here at
// final size. Not only to keep allocations out of the middle of a token: `StaticPlacement` decides
// how many expert shards fit from `hipMemGetInfo` and a fixed reserve, so anything allocated after
// it is invisible to that decision. At an 8K cache the stragglers fit inside the reserve; at 1M the
// indexer's key cache and score plane alone are 2.56 GiB and the run dies with an out-of-memory
// surfacing at whatever kernel calls hipGetLastError first, nowhere near the allocation.
//
// Every helper below is capacity-guarded and idempotent, so calling them here with the widest
// arguments the run can present makes every later call a no-op rather than a resize.
bool DenseGpu::reserve_runtime(uint32_t n_layer, const uint32_t* ratios, uint32_t width,
                               uint32_t idx_dim, uint32_t idx_heads, std::string* err) {
  if (!impl_ || impl_->dev.empty()) { if (err) *err = "no device"; return false; }
  if (!attn_ok_) { if (err) *err = "attn_init must run first"; return false; }
  if (!ratios) { if (err) *err = "no per-layer ratios"; return false; }
  // b_cap_ is 0 when prefill batching is off; the decode path still wants one token's worth.
  const uint32_t cap = b_cap_ ? b_cap_ : 1u;
  const uint32_t aw = 2u * width, aiw = 2u * idx_dim;
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    for (uint32_t l = 0; l < n_layer; ++l) {
      const uint32_t r = ratios[l];
      if (!r) continue;                       // ratio 0 compresses nothing and allocates nothing
      // The indexer half exists on the ratio-4 layers alone — the same rule batch_compress applies
      // when it derives `has_idx` from whether the caller bound idx_comp_wkv/wgate.
      if (!compress_grow(&D, l, r, comp_width(r, width),
                         r == 4u ? comp_width(r, idx_dim) : 0u)) {
        if (err) *err = "compressor state for layer " + std::to_string(l);
        return false;
      }
    }
    // The compressor's per-chunk projection arena. batch_compress sizes this from b_nb_, THIS
    // chunk's token count, which is never more than b_cap_.
    if (D.c_cap_tok < cap) {
      for (void** p : {(void**)&D.c_kv, (void**)&D.c_sc, (void**)&D.ci_kv, (void**)&D.ci_sc})
        if (*p) { (void)hipFree(*p); *p = nullptr; }
      if (!hip_try(hipMalloc(&D.c_kv, (size_t)cap * aw * sizeof(float)), "compressor kv", err) ||
          !hip_try(hipMalloc(&D.c_sc, (size_t)cap * aw * sizeof(float)), "compressor sc", err) ||
          !hip_try(hipMalloc(&D.ci_kv, (size_t)cap * aiw * sizeof(float)), "indexer kv", err) ||
          !hip_try(hipMalloc(&D.ci_sc, (size_t)cap * aiw * sizeof(float)), "indexer sc", err))
        return false;
      D.c_cap_tok = cap;
    }
    if (idx_heads && !idx_grow(&D, cap, idx_heads, idx_dim)) {
      if (err) *err = "indexer query and score planes";
      return false;
    }
    // The flash key split. kSplitTok is the launch size above which the natural grid already fills
    // the device, so the arena never has to hold a prefill chunk: at 32 heads a card and 512 dims a
    // head it is 8 * 32 * 512 * 4 B = 512 KiB a slice, 16 MiB for all 32 of them.
    if (a_width_ == kFlashWidth && !D.fsplit.o) {
      constexpr uint32_t kSplitTok = 8, kSplitSlots = 32;
      uint64_t mlb = 0;
      const uint64_t ob = flash_split_bytes(kSplitTok, a_hpr_, kSplitSlots, &mlb);
      if (!hip_try(hipMalloc(&D.fsplit.o, ob), "flash split numerators", err) ||
          !hip_try(hipMalloc(&D.fsplit.ml, mlb), "flash split softmax state", err))
        return false;
      D.fsplit.tok_cap = kSplitTok;
      D.fsplit.slots = kSplitSlots;
    }
    if (!D.topk.hist) {
      constexpr uint32_t kTopkTok = 8;
      uint64_t sb = 0, tb = 0;
      const uint64_t hb = idx_topk_scratch_bytes(kTopkTok, &sb, &tb);
      if (!hip_try(hipMalloc(&D.topk.hist, hb), "top-k histogram", err) ||
          !hip_try(hipMalloc(&D.topk.state, sb), "top-k select state", err) ||
          !hip_try(hipMalloc(&D.topk.tie, tb), "top-k tie list", err))
        return false;
      AFF_HIP_CHECK(hipMemset(D.topk.hist, 0, hb));
      AFF_HIP_CHECK(hipMemset(D.topk.state, 0, sb));
      AFF_HIP_CHECK(hipMemset(D.topk.tie, 0, tb));
      D.topk.tok_cap = kTopkTok;
    }
    // The admitted-row lists. kAdmCap is the largest index_topk this path will serve; above it the
    // top-k keeps the mask and attention keeps walking every row, which is correct and slow rather
    // than wrong. 512 is what this model asks for.
    if (!D.a_adm) {
      if (!hip_try(hipMalloc(&D.a_adm, (size_t)kAdmCap * sizeof(uint32_t)), "admitted list", err) ||
          !hip_try(hipMalloc(&D.a_admn, sizeof(uint32_t)), "admitted count", err) ||
          !hip_try(hipMalloc(&D.b_adm, (size_t)cap * kAdmCap * sizeof(uint32_t)),
                   "admitted list, chunk", err) ||
          !hip_try(hipMalloc(&D.b_admn, (size_t)cap * sizeof(uint32_t)), "admitted counts", err))
        return false;
      AFF_HIP_CHECK(hipMemset(D.a_admn, 0, sizeof(uint32_t)));
      AFF_HIP_CHECK(hipMemset(D.b_admn, 0, (size_t)cap * sizeof(uint32_t)));
    }
  }
  return true;
}

// The routed-expert dispatch runs on BOTH cards and leaves whichever it touched last current, so
// every batch entry point makes its own card current before it launches. A kernel issued on a
// stream that belongs to a different device than the current one fails with "invalid resource
// handle" at the launch — which reads like a corrupt pointer and is not one.
void DenseGpu::batch_gemm_ent(size_t r, int32_t h, const float* x, float* y, uint64_t rows,
                              uint64_t cols, const char* aff_file, uint32_t aff_line) {
  const Entry& e = ent_[h];
  // The bf16 kernel, which dequants E4M3 through an LDS LUT and therefore reads ROWS. For an
  // allowlisted tensor this is only reachable when `q8_ok` refuses — cols % 128, which none of them
  // has — so it is a guard rather than a path.
  if (e.sh[r].ps) {
    aff::ui::fatal("dense-gpu: the bf16 batched GEMM got preshuffled %llux%llu (shard %llux%llu"
                 ", quant %u, nb %u)\n", (unsigned long long)e.rows, (unsigned long long)e.cols,
                 (unsigned long long)e.sh[r].nrows, (unsigned long long)e.sh[r].ncols, e.quant,
                 b_nb_);
    std::abort();
  }
  batch_gemm_hip(e.quant, e.sh[r].data, e.sh[r].scales, x, y, (uint32_t)rows, (uint32_t)cols,
                 b_nb_, impl_->dev[r].stream, aff_file, aff_line);
}

// ---- W8A8 for the dense prefill GEMMs ----------------------------------------------------------
//
// The bf16 path reads the E4M3 weight and converts it through an LDS lookup table before
// multiplying, so it runs the matrix core at bf16 rate on operands that are already fp8. Feeding the
// core fp8 directly is worth roughly a factor of two on every dense GEMM here.
//
// The activation costs one pass — [dim][nb] f32 -> [nb][dim] E4M3, a scale per (token, 128 columns)
// — which is small against a GEMM and is paid ONCE per activation rather than once per GEMM. That is
// why the caller places this call instead of the GEMM wrapper doing it for itself.
//
// `dim % 128` is a precondition, not a preference: the scale block is 128 wide and the GEMM's fold
// at the end of its K loop assumes a whole one.
bool DenseGpu::q_slot(size_t r, uint32_t dim) {
  Impl::Dev& D = impl_->dev[r];
  if (dim % 128u) return false;
  if (D.q_cap < dim) {
    for (void** p : {(void**)&D.q8, (void**)&D.qs}) if (*p) { (void)hipFree(*p); *p = nullptr; }
    if (hipMalloc(&D.q8, (size_t)b_cap_ * dim) != hipSuccess) return false;
    if (hipMalloc(&D.qs, (size_t)b_cap_ * (dim / 128u) * sizeof(float)) != hipSuccess) return false;
    D.q_cap = dim;
  }
  D.q_dim = dim;
  return true;
}

bool DenseGpu::quant_act(size_t r, const float* src, uint32_t dim) {
  if (!q_slot(r, dim)) return false;
  Impl::Dev& D = impl_->dev[r];
  // Already there: hc_pre quantises `norm` in the kernel that writes it. `q8_src` is cleared by
  // every other path that overwrites the slot, so this can only match the plane actually in it.
  if (D.q8_src == src && D.q_dim == dim) return true;
  batch_quant_fp8_hip(src, D.q8, D.qs, dim, b_nb_, 0, D.stream);
  D.q8_src = src;
  return true;
}

bool DenseGpu::quant_act_swiglu(size_t r, const float* g, const float* u, uint32_t dim,
                                float limit) {
  if (!q_slot(r, dim)) return false;
  Impl::Dev& D = impl_->dev[r];
  batch_swiglu_quant_fp8_hip(g, u, D.q8, D.qs, dim, b_nb_, limit, D.stream);
  D.q8_src = nullptr;
  return true;
}

// A GEMM against whatever quant_act() last put in the slot. `x_off` is a column window into it, for
// the grouped output projection; the row stride is the slot's own dim, not this GEMM's `cols`.
void DenseGpu::batch_gemm_ent_q8(size_t r, int32_t h, float* y, uint64_t rows, uint64_t cols,
                                 uint32_t x_off, uint32_t w_row_off, bool accum,
                                 const char* aff_file, uint32_t aff_line) {
  const Entry& e = ent_[h];
  Impl::Dev& D = impl_->dev[r];
  // w_row_off is nonzero only for the grouped output projection, whose eight groups share one
  // shard: the kernel needs it to reach both the weight rows and the right row of the scale plane.
  batch_gemm_fp8_hip((const uint8_t*)e.sh[r].data, e.sh[r].scales, D.q8, D.qs, y, (uint32_t)rows,
                     (uint32_t)cols, b_nb_, x_off, w_row_off, D.stream, e.sh[r].ps,
                     D.q_dim, D.q_dim / 128u, accum, 0u, aff_file, aff_line);
}

// True when this entry can go through the W8A8 path at all: E4M3 weights, and a K that is a whole
// number of scale blocks. Everything else keeps the bf16 kernel.
bool DenseGpu::q8_ok(int32_t h, uint64_t cols) const {
  return h >= 0 && (size_t)h < ent_.size() && ent_[h].quant == AFF_FP8_E4M3 && (cols % 128) == 0;
}

void DenseGpu::batch_stage_out(size_t r, const float* src_dm, uint32_t n, uint32_t off) {
  Impl::Dev& D = impl_->dev[r];
  AFF_HIP_CHECK(hipSetDevice(D.id));
  batch_transpose_hip(src_dm, D.b_tm + (size_t)off * b_nb_, n, b_nb_, 1, D.stream);
  AFF_HIP_CHECK(hipMemcpyAsync(D.h_tm + (size_t)off * b_nb_, D.b_tm + (size_t)off * b_nb_,
                               (size_t)b_live_ * n * sizeof(float), hipMemcpyDeviceToHost,
                               D.stream));
}

bool DenseGpu::batch_begin(uint32_t n_live, uint32_t pos0, const float* emb, uint32_t n_embd,
                           uint32_t n_hc) {
  AFF_TRACE(20);
  if (!b_cap_ || !n_live) return false;
  // A new chunk: whatever admissions the b_* buffers hold were written for a previous chunk's
  // token count and are garbage past it. batch_indexer re-arms this for its own layer.
  b_idx_layer_[slot_] = -1;
  // The SAME function batch_init sized the arena with, deliberately. As a second copy of the
  // expression it drifted: a chunk asked for a stride larger than its own arena, was refused here,
  // and the caller quietly fell back to one token at a time — so prefill reported decode's rate,
  // because it WAS decode.
  const uint32_t nb = batch_round(n_live);
  if (nb > b_cap_) {
    // The arena was sized by the same function on a cap this chunk cannot exceed, so this is a bug
    // in one of the two rather than a configuration the caller can reach. The failure mode would be
    // a plausible-looking number rather than a crash, so say so.
    aff::ui::fatal("batch_begin: %u live -> stride %u exceeds the %u-token arena\n", n_live,
                 nb, b_cap_);
    std::abort();
  }
  b_nb_ = nb; b_live_ = n_live; b_pos0_ = pos0;
  const size_t live_bytes = (size_t)n_live * n_embd * sizeof(float);
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    // ---- through the ring, which is what deleted the drain that used to be here ------------------
    //
    // `emb` belongs to the caller. For the draft it is `dspark_emb_`, a member — but one whose row 0
    // is rebuilt in place every block, so "not a local" was never the property that mattered. What
    // matters is that the DMA reads it after this call returns, and the drain was the only thing
    // guaranteeing it had. A ring slot per call carries the same guarantee without pinning the host
    // back to one block of lookahead.
    if (!D.emb.fit(live_bytes, kRingPlaneBudget)) return false;
    uint32_t es = 0;
    uint8_t* const pin = D.emb.claim(&es);
    AFF_BLOCK_AT(aff::BlockSite::Begin, AFF_HIP_CHECK(D.emb.wait(es)));
    std::memcpy(pin, emb, live_bytes);
    // The padding columns run through every kernel; zeroing the seed is what keeps them finite, and
    // a NaN there would be invisible until it reached a reduction that crosses tokens.
    AFF_HIP_CHECK(hipMemsetAsync(D.b_emb, 0, (size_t)nb * n_embd * sizeof(float), D.stream));
    AFF_HIP_CHECK(hipMemcpyAsync(D.b_emb, pin, live_bytes, hipMemcpyHostToDevice, D.stream));
    AFF_HIP_CHECK(D.emb.done(es, D.stream));
    batch_hc_seed_hip(D.b_emb, D.b_hc, n_embd, n_hc, nb, D.stream);
    // Every token in the chunk starts its stack with the reference's one-hot mix.
    if (shift_pre_) seed_pre_identity(D.b_pre_prev, n_hc, nb, D.stream);
  }
  return true;
}

bool DenseGpu::batch_hc_pre(int32_t fn, int32_t sv, int32_t bv, int32_t nw, uint32_t n_embd,
                            uint32_t n_hc, uint32_t iters, float hc_eps, float rms_eps,
                            float* norm_host) {
  if (!b_nb_ || n_hc > 8) return false;
  AFF_TRACE(21);
  const bool ctl = fn >= 0 && (size_t)fn < ent_.size() && sv >= 0 && bv >= 0;
  if (ctl && ent_[fn].split != Split::Rep) return false;
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    if (ctl) {
      AFF_TRACE(37);
      // b_hc is bf16, so this is the bf16-activation entry rather than batch_gemm_ent: the same
      // operand at the matrix core — the f32 form truncated it in LDS staging — and half the bytes.
      const Entry& F = ent_[fn];
      if (!batch_gemm_bf16x_hip(F.quant, F.sh[d].data, F.sh[d].scales, D.b_hc, D.b_mix,
                                (uint32_t)(2ull * n_hc + (uint64_t)n_hc * n_hc),
                                (uint32_t)((uint64_t)n_embd * n_hc), b_nb_, D.stream))
        return false;
    }
    const float* nwp = (nw >= 0 && (size_t)nw < n_vec_) ? D.vec[nw] : nullptr;
    {
      AFF_TRACE(38);
    // A speculative block takes the narrow form. The wide kernel's grid is a function of the token
    // count, so at b_nb_ = kNarrowTok it is two workgroups; this one's is a function of n_embd.
    if (b_nb_ <= kNarrowTok) {
      // Quantise `norm` inside the kernel that writes it, when the slot can be sized for it: every
      // W8A8 consumer of norm calls batch_quant_fp8_hip on it as its very next act.
      const bool fold = (n_embd % kHcnThreads) == 0 && q_slot(d, (uint32_t)n_embd);
      batch_hc_fused_narrow_hip(D.b_hc, D.b_mix, ctl ? D.vec[sv] : nullptr,
                                ctl ? D.vec[bv] : nullptr, nwp, D.b_pre, D.b_post, D.b_comb,
                                D.b_cur, D.b_norm, D.b_hcp, fold ? D.q8 : nullptr,
                                fold ? D.qs : nullptr, n_embd, n_hc, b_nb_, iters, hc_eps,
                                rms_eps, ctl ? 1 : 0, D.stream,
                                shift_pre_ ? D.b_pre_prev : nullptr);
      if (shift_pre_)
        AFF_HIP_CHECK(hipMemcpyAsync(D.b_pre_prev, D.b_pre, (size_t)n_hc * b_nb_ * sizeof(float),
                                     hipMemcpyDeviceToDevice, D.stream));
      D.q8_src = fold ? D.b_norm : nullptr;
    }
      else
        batch_hc_fused_hip(D.b_hc, D.b_mix, ctl ? D.vec[sv] : nullptr, ctl ? D.vec[bv] : nullptr,
                           nwp, D.b_pre, D.b_post, D.b_comb, D.b_cur, D.b_norm, n_embd, n_hc, b_nb_,
                           iters, hc_eps, rms_eps, ctl ? 1 : 0, D.stream,
                           shift_pre_ ? D.b_pre_prev : nullptr);
        if (shift_pre_)
          AFF_HIP_CHECK(hipMemcpyAsync(D.b_pre_prev, D.b_pre, (size_t)n_hc * b_nb_ * sizeof(float),
                                       hipMemcpyDeviceToDevice, D.stream));
    }
    // ---- the one place both cards are supposed to hold the same bytes -----------------------------
    //
    // `b_norm` is derived from `b_hc`, which every crossing leaves replicated, so the two cards
    // agreeing here is what makes the whole sublayer below one computation rather than two similar
    // ones. Nothing checked it. Sampled, never waited on, and fatal on a mismatch — see digest.h for
    // why a reference check could not see the failure this is for.
    xrank_check((uint32_t)d, (uint32_t)impl_->dev.size(), D.b_norm,
                (size_t)n_embd * b_nb_ * sizeof(float), D.stream);
    xrank_trace((uint32_t)d, "nrm", D.b_norm, (size_t)n_embd * b_nb_ * sizeof(float), D.stream);
  }
  b_norm_rows_ = n_embd;
  if (norm_host) {
    Impl::Dev& D = impl_->dev[0];                     // replicated: the two are identical
    batch_stage_out(0, D.b_norm, n_embd, 0);
    AFF_DRAIN_AT(D, aff::BlockSite::HcPre);
    std::memcpy(norm_host, D.h_tm, (size_t)b_live_ * n_embd * sizeof(float));
  }
  ++st_.calls;
  return true;
}

bool DenseGpu::batch_attn_qkv(int32_t qa, int32_t qb, int32_t qnorm, int32_t kv, int32_t kvnorm,
                              uint64_t hidden, uint64_t qr, uint32_t n_head, uint32_t width,
                              uint32_t n_rot, RopeDerived rope, float eps, float* qrnorm_host) {
  if (!b_nb_) return false;
  AFF_TRACE(22);
  if (qa < 0 || qb < 0 || kv < 0 || qnorm < 0 || kvnorm < 0) return false;
  if (qr > b_mid_ || width > b_mid_) return false;
  const Entry& A = ent_[qa]; const Entry& B = ent_[qb]; const Entry& K = ent_[kv];
  if (A.split != Split::Rep || K.split != Split::Rep || B.split != Split::Row) return false;
  for (size_t d = 0; d < impl_->dev.size(); ++d)
    if (B.sh[d].row0 % width || B.sh[d].nrows % width) return false;

  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    const uint64_t nq = B.sh[d].nrows;
    const uint32_t nh = (uint32_t)(nq / width);
    AFF_HIP_CHECK(hipSetDevice(D.id));
    // wq_a and wkv both read `norm`, so they are issued together against ONE quantisation of it —
    // and wkv is hoisted above the q chain to make that possible, which is free because nothing in
    // the q chain feeds it.
    bool q8;
    float* ykv = D.b_u;                 // where wkv landed: b_u unfused, inside b_g when fused
    {
      AFF_TRACE(39);
      q8 = q8_ok(qa, hidden) && q8_ok(kv, hidden) && q8_ok(qb, qr) &&
           quant_act(d, D.b_norm, (uint32_t)hidden);
      if (q8) {
        // Same fusion as shexp: one activation, two weights, one launch. wkv lands after wq_a.
        const BatchGemmJob bj[2] = {{A.sh[d].data, A.sh[d].scales, D.b_g, (uint32_t)qr},
                                    {K.sh[d].data, K.sh[d].scales, nullptr, width}};
        // One layout flag for the group. Both are fp8 with the same K, so the load-time preshuffle
        // took both or neither; a mixed group would silently read one in the wrong order, so it
        // refuses the fusion and each goes on its own.
        if (A.sh[d].ps == K.sh[d].ps &&
            batch_gemm_fp8_multi_hip(bj, 2, D.q8, D.qs, (uint32_t)hidden, b_nb_, D.q_dim,
                                     D.q_dim / 128u, D.stream, A.sh[d].ps)) {
          ykv = D.b_g + (size_t)qr * b_nb_;
        } else {
          batch_gemm_ent_q8(d, qa, D.b_g, qr, hidden);
          batch_gemm_ent_q8(d, kv, D.b_u, width, hidden);
        }
      } else {
        batch_gemm_ent(d, qa, D.b_norm, D.b_g, qr, hidden);
        batch_gemm_ent(d, kv, D.b_norm, D.b_u, width, hidden);
      }
    }
    {
      AFF_TRACE(40);
      // The q-lora norm emits its own E4M3 when it can: one block a token at this width, holding
      // every dim of that token, so the scale blocks are local and the quantise launch goes.
      bool qr_q8 = false;
      if (q8 && q_slot(d, (uint32_t)qr))
        qr_q8 = batch_rms_norm_q8_hip(D.b_g, D.vec[qnorm], nullptr, D.b_qr, D.q8, D.qs,
                                      (uint32_t)qr, b_nb_, eps, D.stream);
      if (qr_q8)
        D.q8_src = D.b_qr;
      else
        batch_rms_norm_hip(D.b_g, D.vec[qnorm], nullptr, D.b_qr, (uint32_t)qr, b_nb_, eps,
                           D.stream);
      if (q8 && (qr_q8 || quant_act(d, D.b_qr, (uint32_t)qr)))
        batch_gemm_ent_q8(d, qb, D.b_q, nq, qr);
      else
        batch_gemm_ent(d, qb, D.b_qr, D.b_q, nq, qr);
    }
    {
      AFF_TRACE(41);
      // Per-head RMS, the rope tail and the dim-major -> token-major transpose attention needs, in
      // two passes over b_q rather than five. b_q is left unscaled: this transpose was its only
      // reader, and b_q is b_htm, which mqa_attend overwrites next.
      if (!batch_head_rms_rope_tr_hip(D.b_q, D.b_qtm, D.b_hinv, nh, width, n_rot, b_pos0_, rope,
                                      eps, b_nb_, D.stream))
        return false;
      batch_rms_norm_rope_hip(ykv, D.vec[kvnorm], D.b_kv, width, n_rot, b_pos0_, rope, eps, b_nb_,
                              D.stream);
    }
  }
  if (qrnorm_host) {
    Impl::Dev& D = impl_->dev[0];                     // wq_a is replicated
    batch_stage_out(0, D.b_qr, (uint32_t)qr, 0);
    AFF_DRAIN(D);
    std::memcpy(qrnorm_host, D.h_tm, (size_t)b_live_ * qr * sizeof(float));
  }
  ++st_.calls;
  return true;
}

bool DenseGpu::batch_mv_host(int src, const int32_t* h, const uint64_t* rows, uint64_t cols,
                             uint32_t n, float* const* y) {
  if (!b_nb_ || !n || n > kMvBatchMax) return false;
  AFF_TRACE(23);
  const size_t R = impl_->dev.size();
  for (uint32_t j = 0; j < n; ++j) {
    if (h[j] < 0 || (size_t)h[j] >= ent_.size()) return false;
    if (ent_[h[j]].split == Split::Col) return false;
  }
  uint64_t r0[kMaxRanks][kMvBatchMax], nr[kMaxRanks][kMvBatchMax], total[kMaxRanks] = {0};
  for (size_t d = 0; d < R; ++d) {
    for (uint32_t j = 0; j < n; ++j) {
      row_band(ent_[h[j]].sh[d], rows[j], &r0[d][j], &nr[d][j]);
      total[d] += nr[d][j];
    }
    if (total[d] > b_mid_) return false;
  }

  // All the GEMMs first, then all the transposes and copies, then ONE wait. Issued job by job this
  // was a synchronise per matrix, which is exactly the shape the batched path exists to delete.
  for (size_t d = 0; d < R; ++d) {
    Impl::Dev& D = impl_->dev[d];
    if (!total[d]) continue;
    AFF_HIP_CHECK(hipSetDevice(D.id));
    const float* x = (src == 0) ? D.b_norm : D.b_qr;
    uint64_t off = 0;
    // One launch when the group agrees on a format and fits — see the compressor above.
    BatchGemmJob bj[kMvBatchMax];
    uint32_t nj = 0;
    bool one = b_nb_ <= kNarrowTok && n <= kGemmJobsMax;
    for (uint32_t j = 0; j < n && one; ++j) {
      if (!nr[d][j]) continue;
      const Entry& e = ent_[h[j]];
      const uint64_t local = r0[d][j] - e.sh[d].row0;
      one = e.quant == ent_[h[0]].quant && e.quant != AFF_F32;
      bj[nj++] = {shard_rows(e.sh[d], e.quant, local), shard_scales(e.sh[d], local),
                  D.b_y + off * b_nb_, (uint32_t)nr[d][j]};
      off += nr[d][j];
    }
    if (one && nj)
      one = batch_gemm_multi_hip(ent_[h[0]].quant, bj, nj, x, (uint32_t)cols, b_nb_, false,
                                 D.stream);
    if (!one) {
      off = 0;
      for (uint32_t j = 0; j < n; ++j) {
        if (!nr[d][j]) continue;
        const Entry& e = ent_[h[j]];
        const uint64_t local = r0[d][j] - e.sh[d].row0;
        batch_gemm_hip(e.quant, shard_rows(e.sh[d], e.quant, local), shard_scales(e.sh[d], local),
                       x, D.b_y + off * b_nb_, (uint32_t)nr[d][j], (uint32_t)cols, b_nb_, D.stream);
        off += nr[d][j];
      }
    }
    off = 0;
    for (uint32_t j = 0; j < n; ++j) {
      if (!nr[d][j]) continue;
      batch_stage_out(d, D.b_y + off * b_nb_, (uint32_t)nr[d][j], (uint32_t)off);
      off += nr[d][j];
    }
  }
  for (size_t d = 0; d < R; ++d) {
    Impl::Dev& D = impl_->dev[d];
    if (!total[d]) continue;
    AFF_HIP_CHECK(hipSetDevice(D.id));
    AFF_DRAIN(D);
    uint64_t off = 0;
    for (uint32_t j = 0; j < n; ++j) {
      if (!nr[d][j]) continue;
      // Token-major on the host, so this rank's row band is a stride-`rows[j]` window in each
      // token's slot rather than one contiguous run.
      for (uint32_t b = 0; b < b_live_; ++b)
        std::memcpy(y[j] + (size_t)b * rows[j] + r0[d][j], D.h_tm + off * b_nb_ + (size_t)b * nr[d][j],
                    nr[d][j] * sizeof(float));
      off += nr[d][j];
    }
  }
  ++st_.calls;
  return true;
}

bool DenseGpu::batch_kv_commit(uint32_t layer, uint32_t b0, uint32_t n, uint32_t pos0,
                               uint32_t width, uint32_t n_rot) {
  if (!attn_ok_ || layer >= impl_->dev[0].a_raw_s[slot_].size() || !a_ring_ || !n) return false;
  AFF_TRACE(24);
  if (width != a_width_ || n_rot != a_rot_) return false;
  if (n > a_ring_) return false;                      // a sub-batch may not lap its own ring
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    // One launch, straight from the chunk's dim-major latent. The per-token version staged a 2 KiB
    // column through d_u and then ran two more kernels on it, three launches a token a layer.
    kv_commit_batch_hip(a_dt_, D.b_kv, b_nb_, b0, n, D.a_raw_s[slot_][layer], pos0, a_ring_, width, n_rot,
                        D.stream);
  }
  return true;
}

// The lightning indexer for a whole chunk, in five launches a layer a card. The per-token HOST loop
// it replaces — score every compressed row against 64 heads, then partial_sort — was 8.3 s of a
// 12.3 s long-prompt prefill with the GPU idle. See `indexer_gpu.h`. The mask is produced straight
// into `b_allow` where `batch_attend` already reads it, so it never crosses the bus.
//
// Grows the indexer's per-token arenas; both entry points call it, since the query projection lands
// in i_q before batch_indexer runs and whichever comes first must have allocated them.
bool DenseGpu::idx_grow(void* devp, uint32_t n, uint32_t n_head, uint32_t dim) {
  Impl::Dev& D = *(Impl::Dev*)devp;
  if (D.i_cap_tok >= n) return true;
  const uint32_t cap = n;
  for (void** p : {(void**)&D.i_q, (void**)&D.i_q8, (void**)&D.i_qs, (void**)&D.i_w,
                   (void**)&D.i_sc, (void**)&D.hi_w, (void**)&D.i_nc, (void**)&D.hi_nc})
    if (*p) { (void)hipFree(*p); *p = nullptr; }
  if (hipMalloc(&D.i_q, (size_t)cap * n_head * dim * sizeof(float)) != hipSuccess) return false;
  if (hipMalloc(&D.i_q8, (size_t)cap * n_head * dim) != hipSuccess) return false;
  if (hipMalloc(&D.i_qs, (size_t)cap * sizeof(float)) != hipSuccess) return false;
  if (hipMalloc(&D.i_w, (size_t)cap * n_head * sizeof(float)) != hipSuccess) return false;
  // The score plane is the only buffer here that scales with CONTEXT as well as the chunk —
  // [tokens][every candidate compressed row] — so it is capped rather than sized. Below the budget
  // nothing changes (one tile, one pass of batch_indexer's loop); above it the chunk is tiled.
  // 256 MiB keeps the tile at the full 1536 up to 128K context, and gives 256 tokens a tile at 1M
  // against 1.50 GiB untiled.
  const size_t kScoreBudget = (size_t)256 << 20;
  const size_t row = (size_t)a_maxcomp_ * sizeof(float);
  i_tile_ = (uint32_t)std::min<size_t>(cap, std::max<size_t>(1, row ? kScoreBudget / row : cap));
  if (hipMalloc(&D.i_sc, (size_t)i_tile_ * row) != hipSuccess) return false;
  if (hipHostMalloc((void**)&D.hi_w, (size_t)cap * n_head * sizeof(float)) != hipSuccess)
    return false;
  if (hipMalloc(&D.i_nc, (size_t)cap * sizeof(uint32_t)) != hipSuccess) return false;
  if (hipHostMalloc((void**)&D.hi_nc, (size_t)cap * sizeof(uint32_t)) != hipSuccess) return false;
  D.i_cap_tok = cap;
  return true;
}

// The indexer's query projection, left ON DEVICE in i_q. Same GEMM batch_mv_host would run, minus
// the drain and the copy back — `idx_wq_b` is replicated, so this rank's shard is the whole matrix.
bool DenseGpu::batch_indexer_q(int32_t h, uint64_t rows, uint64_t cols, uint32_t n_head,
                               uint32_t dim) {
  if (!b_nb_ || h < 0 || (size_t)h >= ent_.size()) return false;
  AFF_TRACE(23);
  const Entry& e = ent_[h];
  if (e.split != Split::Rep || rows != (uint64_t)n_head * dim) return false;
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    if (!idx_grow(&D, b_live_, n_head, dim)) return false;
    batch_gemm_hip(e.quant, shard_rows(e.sh[d], e.quant, 0), shard_scales(e.sh[d], 0), D.b_qr,
                   D.b_y, (uint32_t)rows, (uint32_t)cols, b_nb_, D.stream);
    // Dim-major [rows][nb] out of the GEMM; the indexer wants token-major [nb][rows].
    batch_transpose_hip(D.b_y, D.i_q, (uint32_t)rows, b_nb_, 1, D.stream);
  }
  return true;
}

// ---- the compressors, both of them, for a whole chunk -------------------------------------------
//
// Nine launches a layer: five GEMMs, four transposes, two compressors, two ring updates. The rows
// land in their final cache form and nothing crosses the bus in either direction. The per-token
// HOST walk this replaces was a quarter of the wall, most of it the device idling in a drain.
// See gpu/compressor_gpu.h for why the pooling was never sequential.
bool DenseGpu::compress_grow(void* devp, uint32_t layer, uint32_t ratio, uint32_t width,
                             uint32_t idx_width) {
  Impl::Dev& D = *(Impl::Dev*)devp;
  const uint32_t nl = (uint32_t)D.a_comp_s[slot_].size();
  if (layer >= nl) return false;
  // Shared per-layer state: sized ONCE. Re-running this when a second slot first touches the
  // layer would null pointers slot 0 already allocated.
  if (D.c_ratio.empty()) {
    D.c_ape.assign(nl, nullptr);  D.c_norm.assign(nl, nullptr);
    D.ci_ape.assign(nl, nullptr); D.ci_norm.assign(nl, nullptr);
    D.c_ratio.assign(nl, 0u);
  }
  // Per-sequence state: sized once PER SLOT, lazily, on that slot's first touch.
  if (D.c_hkv_s[slot_].empty()) {
    D.c_hkv_s[slot_].assign(nl, nullptr);  D.c_hsc_s[slot_].assign(nl, nullptr);
    D.ci_hkv_s[slot_].assign(nl, nullptr); D.ci_hsc_s[slot_].assign(nl, nullptr);
    D.a_idx8_s[slot_].assign(nl, nullptr); D.a_idxs_s[slot_].assign(nl, nullptr);
  }
  if (D.c_hkv_s[slot_][layer]) return true;
  const uint32_t hist = comp_hist(ratio);
  const size_t hb = (size_t)hist * width * sizeof(float);
  if (hipMalloc(&D.c_hkv_s[slot_][layer], hb) != hipSuccess) return false;
  if (hipMalloc(&D.c_hsc_s[slot_][layer], hb) != hipSuccess) return false;
  compressor_hist_reset_hip(D.c_hkv_s[slot_][layer], D.c_hsc_s[slot_][layer], width, hist, D.stream);
  // The indexer half exists only on the ratio-4 layers; idx_width 0 says so.
  if (idx_width) {
    const size_t ihb = (size_t)hist * idx_width * sizeof(float);
    if (hipMalloc(&D.ci_hkv_s[slot_][layer], ihb) != hipSuccess) return false;
    if (hipMalloc(&D.ci_hsc_s[slot_][layer], ihb) != hipSuccess) return false;
    if (hipMalloc(&D.a_idx8_s[slot_][layer], (size_t)a_maxcomp_ * kIdxDim) != hipSuccess) return false;
    if (hipMalloc(&D.a_idxs_s[slot_][layer], (size_t)a_maxcomp_ * sizeof(float)) != hipSuccess) return false;
    compressor_hist_reset_hip(D.ci_hkv_s[slot_][layer], D.ci_hsc_s[slot_][layer], idx_width, hist, D.stream);
  }
  D.c_ratio[layer] = ratio;
  return true;
}

void DenseGpu::set_kv_owner(const uint32_t* owner, uint32_t n) {
  kv_owner_.assign(owner, owner + n);
}

bool DenseGpu::batch_compress(const Model::CompressArgs& a) {
  if (!b_nb_ || !a.n || !a.ratio || (a.ratio & (a.ratio - 1u)) || !attn_ok_) return false;
  if (a.h_kv < 0) return false;
  // V4.1's ratio-1 layers have no gate projection: one position a group, so the pool IS the
  // projection. Everything below that touches the score half is conditioned on this.
  const bool gated = a.h_sc >= 0;
  const bool has_idx = a.h_ikv >= 0 && a.h_isc >= 0;      // ratio 4 only
  if (has_idx && a.idx_dim != kIdxDim) return false;
  if (a.layer >= impl_->dev[0].a_comp_s[slot_].size()) return false;
  // This layer's own allocation for the KV half; the indexer half exists only on ratio-4 layers and
  // those are the widest, so a_maxcomp_ is exactly right for it.
  if (a.slot0 + a.n_out > a_comp_rows_[a.layer]) return false;
  if (has_idx && a.idx_slot0 + a.n_out > a_maxcomp_) return false;
  AFF_TRACE(32);
  const uint32_t cw = comp_width(a.ratio, a.width);
  const uint32_t iw = has_idx ? comp_width(a.ratio, a.idx_dim) : 0u;
  // Replicated on purpose: the pool needs whole rows and both cards must reach the same compressed
  // cache, so a row split would mean an exchange per layer to reassemble what neither card can use
  // half of. Checked rather than assumed — a Row split here is silently half a key.
  const int32_t hs[5] = {a.h_kv, a.h_sc, a.h_ikv, a.h_isc, a.h_iproj};
  const uint64_t rws[5] = {cw, cw, iw, iw, a.idx_heads};
  for (int i = 0; i < 5; ++i) {
    if (hs[i] < 0) continue;
    if ((size_t)hs[i] >= ent_.size() || ent_[hs[i]].split != Split::Rep) return false;
    if (ent_[hs[i]].cols != a.n_embd || ent_[hs[i]].rows < rws[i]) return false;
  }
  if ((uint64_t)2 * (cw + iw) > b_mid_) return false;

  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    hipStream_t st = D.stream;
    if (!compress_grow(&D, a.layer, a.ratio, cw, iw)) return false;
    // One arena for every layer, sized from the widest: the ratio-4 layers carry both halves of the
    // projection in one row and the ratio-128 layers half of that.
    const uint32_t aw = 2u * a_width_, aiw = 2u * kIdxDim;
    if (D.c_cap_tok < b_nb_) {
      for (void** p : {(void**)&D.c_kv, (void**)&D.c_sc, (void**)&D.ci_kv, (void**)&D.ci_sc})
        if (*p) { (void)hipFree(*p); *p = nullptr; }
      if (hipMalloc(&D.c_kv, (size_t)b_nb_ * aw * sizeof(float)) != hipSuccess) return false;
      if (hipMalloc(&D.c_sc, (size_t)b_nb_ * aw * sizeof(float)) != hipSuccess) return false;
      if (hipMalloc(&D.ci_kv, (size_t)b_nb_ * aiw * sizeof(float)) != hipSuccess) return false;
      if (hipMalloc(&D.ci_sc, (size_t)b_nb_ * aiw * sizeof(float)) != hipSuccess) return false;
      D.c_cap_tok = b_nb_;
    }
    // The layer's ape and norm weights, once, and ASYNCHRONOUSLY. A blocking hipMemcpy here waits
    // for the device, and by the time layer L runs there is a whole 1536-token layer in the queue —
    // four of them a layer a card across 43 layers, on the one pass where this guard is false,
    // which is why hybrid prefill's spread was always in the first chunk.
    //
    // Async on the layer's own stream needs no sync: the kernels that read these are issued on `st`
    // behind them, and the sources are the model's own weights, which outlive the process.
    //
    // Keyed on `c_norm`, not on `c_ape`: V4.1 has no ape, so an ape-keyed guard would never latch
    // and every chunk would allocate and copy the norm again.
    if (!D.c_norm[a.layer]) {
      if (hipMalloc(&D.c_norm[a.layer], (size_t)a.width * sizeof(float)) != hipSuccess) return false;
      AFF_HIP_CHECK(hipMemcpyAsync(D.c_norm[a.layer], a.norm, (size_t)a.width * sizeof(float),
                                   hipMemcpyHostToDevice, st));
      if (a.ape) {
        const size_t ab = (size_t)a.ratio * cw * sizeof(float);
        if (hipMalloc(&D.c_ape[a.layer], ab) != hipSuccess) return false;
        AFF_HIP_CHECK(hipMemcpyAsync(D.c_ape[a.layer], a.ape, ab, hipMemcpyHostToDevice, st));
      }
      if (has_idx) {
        if (hipMalloc(&D.ci_norm[a.layer], (size_t)a.idx_dim * sizeof(float)) != hipSuccess)
          return false;
        AFF_HIP_CHECK(hipMemcpyAsync(D.ci_norm[a.layer], a.inorm,
                                     (size_t)a.idx_dim * sizeof(float), hipMemcpyHostToDevice, st));
        if (a.iape) {
          const size_t iab = (size_t)a.ratio * iw * sizeof(float);
          if (hipMalloc(&D.ci_ape[a.layer], iab) != hipSuccess) return false;
          AFF_HIP_CHECK(hipMemcpyAsync(D.ci_ape[a.layer], a.iape, iab, hipMemcpyHostToDevice, st));
        }
      }
    }

    // GEMMs into one dim-major staging block, then transposes: the pool indexes by POSITION, so it
    // wants token-major, and the transpose is the same one batch_indexer_q uses.
    struct Job { int32_t h; uint32_t rows; float* dst; };
    // The gate half is skipped entirely when the layer has none (V4.1 ratio 1), so the list is
    // built rather than indexed: an empty slot in the middle would shift every later offset.
    Job job[4];
    int nj = 0;
    job[nj++] = {a.h_kv, cw, D.c_kv};
    if (gated) job[nj++] = {a.h_sc, cw, D.c_sc};
    if (has_idx) { job[nj++] = {a.h_ikv, iw, D.ci_kv}; job[nj++] = {a.h_isc, iw, D.ci_sc}; }
    // ONE launch when they agree on a format, which they do — all four are the same container's
    // compressor tensors. Four small GEMMs against one activation is exactly the shape that reads
    // well under its streaming rate apart and at the ceiling fused.
    uint64_t off = 0;
    BatchGemmJob bj[4];
    bool one = b_nb_ <= kNarrowTok;
    for (int i = 0; i < nj && one; ++i) {
      const Entry& e = ent_[job[i].h];
      one = e.quant == ent_[job[0].h].quant && e.quant != AFF_F32;
      bj[i] = {shard_rows(e.sh[d], e.quant, 0), shard_scales(e.sh[d], 0), D.b_y + off * b_nb_,
               job[i].rows};
      off += job[i].rows;
    }
    if (!one || !batch_gemm_multi_hip(ent_[job[0].h].quant, bj, (uint32_t)nj, D.b_norm,
                                      (uint32_t)a.n_embd, b_nb_, false, st)) {
      off = 0;
      for (int i = 0; i < nj; ++i) {
        const Entry& e = ent_[job[i].h];
        batch_gemm_hip(e.quant, shard_rows(e.sh[d], e.quant, 0), shard_scales(e.sh[d], 0), D.b_norm,
                       D.b_y + off * b_nb_, job[i].rows, (uint32_t)a.n_embd, b_nb_, st);
        off += job[i].rows;
      }
    }
    off = 0;
    {
      const float* tsrc[4]; float* tdst[4]; uint32_t tn[4];
      for (int i = 0; i < nj; ++i) {
        tsrc[i] = D.b_y + off * b_nb_; tdst[i] = job[i].dst; tn[i] = job[i].rows;
        off += job[i].rows;
      }
      batch_transpose_multi_hip(tsrc, tdst, tn, (uint32_t)nj, b_nb_, st);
    }
    // The indexer's per-token head weights, with its 1/sqrt(dim*heads) folded in here rather than
    // in a host loop over 64 floats a token.
    if (a.h_iproj >= 0 && a.idx_heads) {
      if (!idx_grow(&D, b_live_, a.idx_heads, a.idx_dim)) return false;
      const Entry& e = ent_[a.h_iproj];
      batch_gemm_hip(e.quant, shard_rows(e.sh[d], e.quant, 0), shard_scales(e.sh[d], 0), D.b_norm,
                     D.b_y, a.idx_heads, (uint32_t)a.n_embd, b_nb_, st);
      batch_scale_transpose_hip(D.b_y, D.i_w, a.idx_heads, b_nb_, 1, a.iscale, st);
    }

    if (a.n_out) {
      CompressJob c{};
      c.kv = D.c_kv; c.sc = gated ? D.c_sc : nullptr;   // null sc = weight every candidate equally
      c.hist_kv = D.c_hkv_s[slot_][a.layer]; c.hist_sc = D.c_hsc_s[slot_][a.layer];
      c.ape = D.c_ape[a.layer]; c.norm = D.c_norm[a.layer];
      c.out = D.a_comp_s[slot_][a.layer];
      c.out_stride = kv_row_bytes(a_dt_, a_width_, a_rot_);
      c.slot0 = a.slot0; c.head_dim = a.width;
      compressor_rows_hip(0, c, a.n, a.pos0, a.ratio, a.n_out, a.n_rot, a.rope, a.eps, st);

      if (has_idx) {
        CompressJob i{};
        i.kv = D.ci_kv; i.sc = D.ci_sc;
        i.hist_kv = D.ci_hkv_s[slot_][a.layer]; i.hist_sc = D.ci_hsc_s[slot_][a.layer];
        i.ape = D.ci_ape[a.layer]; i.norm = D.ci_norm[a.layer];
        i.out = D.a_idx8_s[slot_][a.layer]; i.out_scale = D.a_idxs_s[slot_][a.layer];
        i.out_stride = a.idx_dim;
        i.slot0 = a.idx_slot0; i.head_dim = a.idx_dim;
        compressor_rows_hip(1, i, a.n, a.pos0, a.ratio, a.n_out, a.n_rot, a.rope, a.eps, st);
      }
    }
    // AFTER the compressors, on the same stream: the window they read is the window this
    // overwrites, and stream order is the whole reason that is safe.
    const uint32_t hist = comp_hist(a.ratio);
    compressor_hist2_hip(D.c_kv, D.c_sc, D.c_hkv_s[slot_][a.layer], D.c_hsc_s[slot_][a.layer], cw,
                         has_idx ? D.ci_kv : nullptr, D.ci_sc,
                         has_idx ? D.ci_hkv_s[slot_][a.layer] : nullptr,
                         has_idx ? D.ci_hsc_s[slot_][a.layer] : nullptr, iw, a.n, a.pos0, hist, st);
  }
  ++st_.calls;
  return true;
}

// The same two compressors for ONE token, on the decode path. It has to be the same implementation
// rather than an equivalent one: the cross-chunk window lives in VRAM, so a decode walking a
// host-side copy would pool over a state prefill stopped updating — a plausible continuation from
// the wrong eight positions, not a crash.
//
// At n = 1 a dim-major GEMM result IS token-major, so the four projections go straight from d_y
// into the kernel with no transpose.
bool DenseGpu::compress_one(const Model::CompressArgs& a) {
  if (!a.ratio || (a.ratio & (a.ratio - 1u)) || !attn_ok_ || a.n != 1u) return false;
  if (a.h_kv < 0) return false;
  const bool gated = a.h_sc >= 0;       // V4.1 ratio 1 has no gate — see batch_compress
  const bool has_idx = a.h_ikv >= 0 && a.h_isc >= 0;
  if (has_idx && a.idx_dim != kIdxDim) return false;
  if (a.layer >= impl_->dev[0].a_comp_s[slot_].size()) return false;
  if (a.slot0 + a.n_out > a_comp_rows_[a.layer]) return false;
  if (has_idx && a.idx_slot0 + a.n_out > a_maxcomp_) return false;
  AFF_TRACE(32);
  const uint32_t cw = comp_width(a.ratio, a.width);
  const uint32_t iw = has_idx ? comp_width(a.ratio, a.idx_dim) : 0u;
  const int32_t hs[5] = {a.h_kv, a.h_sc, a.h_ikv, a.h_isc, a.h_iproj};
  const uint64_t rws[5] = {cw, cw, iw, iw, a.idx_heads};
  for (int i = 0; i < 5; ++i) {
    if (hs[i] < 0) continue;
    if ((size_t)hs[i] >= ent_.size() || ent_[hs[i]].split != Split::Rep) return false;
    if (ent_[hs[i]].cols != a.n_embd || ent_[hs[i]].rows < rws[i]) return false;
  }
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    hipStream_t st = D.stream;
    if (!compress_grow(&D, a.layer, a.ratio, cw, iw)) return false;
    if (2u * (uint64_t)(cw + iw) + a.idx_heads > D.cap_mid) return false;
    // Same lazy upload as the batched path above, same reason for being async — see there.
    //
    // Keyed on `c_norm`, not on `c_ape`: V4.1 has no ape, so an ape-keyed guard would never latch
    // and every chunk would allocate and copy the norm again.
    if (!D.c_norm[a.layer]) {
      if (hipMalloc(&D.c_norm[a.layer], (size_t)a.width * sizeof(float)) != hipSuccess) return false;
      AFF_HIP_CHECK(hipMemcpyAsync(D.c_norm[a.layer], a.norm, (size_t)a.width * sizeof(float),
                                   hipMemcpyHostToDevice, st));
      if (a.ape) {
        const size_t ab = (size_t)a.ratio * cw * sizeof(float);
        if (hipMalloc(&D.c_ape[a.layer], ab) != hipSuccess) return false;
        AFF_HIP_CHECK(hipMemcpyAsync(D.c_ape[a.layer], a.ape, ab, hipMemcpyHostToDevice, st));
      }
      if (has_idx) {
        if (hipMalloc(&D.ci_norm[a.layer], (size_t)a.idx_dim * sizeof(float)) != hipSuccess)
          return false;
        AFF_HIP_CHECK(hipMemcpyAsync(D.ci_norm[a.layer], a.inorm,
                                     (size_t)a.idx_dim * sizeof(float), hipMemcpyHostToDevice, st));
        if (a.iape) {
          const size_t iab = (size_t)a.ratio * iw * sizeof(float);
          if (hipMalloc(&D.ci_ape[a.layer], iab) != hipSuccess) return false;
          AFF_HIP_CHECK(hipMemcpyAsync(D.ci_ape[a.layer], a.iape, iab, hipMemcpyHostToDevice, st));
        }
      }
    }
    // ---- buffer `norm`, project the whole group at the pool -------------------------------------
    //
    // The four compressor projections are read by exactly one thing, the pool, which runs every
    // `ratio` positions. Between pools they go into a ring hist2_kernel fills by ABSOLUTE position
    // — a copy with no accumulation and no order dependence — so replaying the group as one batched
    // GEMM before the pool lands on exactly the same ring contents. Window::at() already reads
    // current-batch-or-history by absolute position and compress_kernel derives its emitting
    // positions from pos0, so a ragged first group after prefill needs no special case.
    //
    // What this buys is the weight read: those four are bf16 and Split::Rep, so undeferred each card
    // re-reads all of them every token. Deferring divides that by the layer's `ratio`.
    //
    // idx_proj is the exception and is issued further down: its rows are the indexer's per-token head
    // weights and the indexer scores every token, so it cannot be deferred. It is also much the
    // smallest of the five.
    if (D.c_pend.empty()) {
      const size_t nl = D.a_comp_s[slot_].size();
      D.c_pend.assign(nl, nullptr); D.c_np.assign(nl, 0u);
      D.c_ppos0.assign(nl, 0u);     D.c_pcap.assign(nl, 0u);
    }
    const uint32_t cap = a.ratio;                     // one flush a pool, never per token
    // A group small enough for matvec_multix_hip is staged TOKEN-major, because that is the layout
    // its `x` wants and it makes the per-token staging a contiguous 16 KiB store instead of a
    // scatter at stride `cap`. A larger one is staged dim-major for batch_gemm_hip. The ratio-4
    // layers take the first branch and the ratio-128 layers the second, which is exactly where each
    // kernel is the right one: four tokens through a kWN=128 GEMM would compute 124 dead columns,
    // and 128 tokens will not fit in the multix register file.
    const bool tmaj = cap <= kMvMultiXMax;
    if (!D.c_pend[a.layer]) {
      const size_t pb = (size_t)a.n_embd * cap * sizeof(float);
      if (hipMalloc(&D.c_pend[a.layer], pb) != hipSuccess) return false;
      // Zeroed because a RAGGED group still runs the GEMM over all `cap` columns — see the flush.
      AFF_HIP_CHECK(hipMemset(D.c_pend[a.layer], 0, pb));
      D.c_pcap[a.layer] = cap;
      D.c_ppos0[a.layer] = a.pos0;
    }
    // The projections land token-major in the same buffers the batched path uses, so they must hold
    // a whole group as well as a whole prefill chunk.
    const uint32_t need_tok = std::max(b_nb_, cap);
    const uint32_t aw = 2u * a_width_, aiw = 2u * kIdxDim;
    if (D.c_cap_tok < need_tok) {
      for (void** p : {(void**)&D.c_kv, (void**)&D.c_sc, (void**)&D.ci_kv, (void**)&D.ci_sc})
        if (*p) { (void)hipFree(*p); *p = nullptr; }
      if (hipMalloc(&D.c_kv, (size_t)need_tok * aw * sizeof(float)) != hipSuccess) return false;
      if (hipMalloc(&D.c_sc, (size_t)need_tok * aw * sizeof(float)) != hipSuccess) return false;
      if (hipMalloc(&D.ci_kv, (size_t)need_tok * aiw * sizeof(float)) != hipSuccess) return false;
      if (hipMalloc(&D.ci_sc, (size_t)need_tok * aiw * sizeof(float)) != hipSuccess) return false;
      D.c_cap_tok = need_tok;
    }
    const uint32_t py_need = (aw + aw + aiw + aiw) * cap;
    if (D.c_py_cap < py_need) {
      if (D.c_py) (void)hipFree(D.c_py);
      D.c_py = nullptr;
      if (hipMalloc(&D.c_py, (size_t)py_need * sizeof(float)) != hipSuccess) return false;
      D.c_py_cap = py_need;
    }

    // A buffered group is only meaningful for the positions that follow it. If the caller jumps —
    // a new sequence, a rewind — the columns held belong to a stream nothing will ask about again,
    // and flushing them would write the ring at positions this one is about to claim. Drop them.
    if (D.c_np[a.layer] && a.pos0 != D.c_ppos0[a.layer] + D.c_np[a.layer]) D.c_np[a.layer] = 0;
    if (D.c_np[a.layer] == 0) D.c_ppos0[a.layer] = a.pos0;

    // Where this token's `norm` goes. Token-major puts it at a contiguous offset; dim-major
    // scatters it down a column at stride `cap`. One `dst[i*stride] = src[i]` covers both.
    MvStage sg{};
    sg.src = D.d_norm;
    sg.n = (uint32_t)a.n_embd;
    if (tmaj) {
      sg.dst = D.c_pend[a.layer] + (size_t)D.c_np[a.layer] * a.n_embd;
      sg.stride = 1;
    } else {
      sg.dst = D.c_pend[a.layer] + D.c_np[a.layer];
      sg.stride = cap;
    }

    // idx_proj, with the staging column riding on its launch. When the layer has no indexer there
    // is no matvec to ride on and the copy pays for its own launch, as it always did.
    if (a.h_iproj >= 0 && a.idx_heads) {
      if (!idx_grow(&D, 1, a.idx_heads, a.idx_dim)) return false;
      const Entry& e = ent_[a.h_iproj];
      // Straight into i_w with the scale folded into the store. At one token the transpose that
      // used to follow was the identity — a whole launch, 21 a token, to multiply 64 floats by a
      // constant and copy them. The batched path above still needs its transpose, where nb > 1
      // makes it a real one.
      MvJob j{e.sh[d].data, e.sh[d].scales, e.quant, (uint32_t)a.idx_heads, D.i_w, a.iscale};
      matvec_batch_hip(&j, 1, D.d_norm, (uint32_t)a.n_embd, st, &sg);
    } else {
      batch_stage_col_hip(D.d_norm, tmaj ? sg.dst : D.c_pend[a.layer], (uint32_t)a.n_embd,
                          tmaj ? 1u : cap, tmaj ? 0u : D.c_np[a.layer], st);
    }
    ++D.c_np[a.layer];

    // Flush at the pool, or when the group is full without one — n_out can be suppressed by a full
    // compressed cache, and a buffer that never drains would run off the end of its columns.
    if (a.n_out || D.c_np[a.layer] >= cap) {
      const uint32_t nb = D.c_np[a.layer], pos0 = D.c_ppos0[a.layer];
      struct Job { int32_t h; uint32_t rows; float* dst; };
      // Built, not indexed — see the batched path: a layer with no gate has no score projection and
      // a hole in the middle would shift every later offset into the staging block.
      Job job[4];
      int nj = 0;
      job[nj++] = {a.h_kv, cw, D.c_kv};
      if (gated) job[nj++] = {a.h_sc, cw, D.c_sc};
      if (has_idx) { job[nj++] = {a.h_ikv, iw, D.ci_kv}; job[nj++] = {a.h_isc, iw, D.ci_sc}; }
      bool done = false;
      if (tmaj) {
        // One launch for the whole flush, writing the pool's layout directly: no GEMM, no
        // transposes, and the live count is a kernel parameter so a ragged group costs nothing.
        MvJob mj[4];
        for (int i = 0; i < nj; ++i) {
          const Entry& e = ent_[job[i].h];
          mj[i] = MvJob{e.sh[d].data, e.sh[d].scales, e.quant, job[i].rows, job[i].dst};
        }
        done = matvec_multix_hip(mj, (uint32_t)nj, D.c_pend[a.layer], (uint32_t)a.n_embd, nb,
                                 (uint32_t)a.n_embd, st);
      }
      if (!done) {
        // The GEMM runs over all `cap` columns even when only `nb` are live, because `nb` is the
        // STRIDE of X as well as its count and the staging block's stride is fixed at `cap`. A
        // group is ragged only where a pool arrives before the buffer fills — the first one after
        // prefill — and the dead columns are zeros, whose output no one reads. The transpose
        // therefore also runs at `cap`, and only the pool and the ring are told the live count.
        uint64_t goff = 0;
        for (int i = 0; i < nj; ++i) {
          const Entry& e = ent_[job[i].h];
          batch_gemm_hip(e.quant, shard_rows(e.sh[d], e.quant, 0), shard_scales(e.sh[d], 0),
                         D.c_pend[a.layer], D.c_py + goff * cap, job[i].rows, (uint32_t)a.n_embd,
                         cap, st);
          goff += job[i].rows;
        }
        goff = 0;
        for (int i = 0; i < nj; ++i) {
          // Token-major: row t of the result is token t's whole projection, so the live prefix is a
          // prefix of the rows and the pool can read it with no further arithmetic.
          batch_transpose_hip(D.c_py + goff * cap, job[i].dst, job[i].rows, cap, 1, st);
          goff += job[i].rows;
        }
      }

      if (a.n_out) {
        CompressJob c{};
        c.kv = D.c_kv; c.sc = gated ? D.c_sc : nullptr;
        c.hist_kv = D.c_hkv_s[slot_][a.layer]; c.hist_sc = D.c_hsc_s[slot_][a.layer];
        c.ape = D.c_ape[a.layer]; c.norm = D.c_norm[a.layer];
        c.out = D.a_comp_s[slot_][a.layer];
        c.out_stride = kv_row_bytes(a_dt_, a_width_, a_rot_);
        c.slot0 = a.slot0; c.head_dim = a.width;
        compressor_rows_hip(0, c, nb, pos0, a.ratio, a.n_out, a.n_rot, a.rope, a.eps, st);

        if (has_idx) {
          CompressJob i{};
          i.kv = D.ci_kv; i.sc = D.ci_sc;
          i.hist_kv = D.ci_hkv_s[slot_][a.layer]; i.hist_sc = D.ci_hsc_s[slot_][a.layer];
          i.ape = D.ci_ape[a.layer]; i.norm = D.ci_norm[a.layer];
          i.out = D.a_idx8_s[slot_][a.layer]; i.out_scale = D.a_idxs_s[slot_][a.layer];
          i.out_stride = a.idx_dim;
          i.slot0 = a.idx_slot0; i.head_dim = a.idx_dim;
          compressor_rows_hip(1, i, nb, pos0, a.ratio, a.n_out, a.n_rot, a.rope, a.eps, st);
        }
      }
      // AFTER the pool, on the same stream: the window it reads is the window this overwrites.
      const uint32_t hist = comp_hist(a.ratio);
      compressor_hist2_hip(D.c_kv, D.c_sc, D.c_hkv_s[slot_][a.layer], D.c_hsc_s[slot_][a.layer], cw,
                           has_idx ? D.ci_kv : nullptr, D.ci_sc,
                           has_idx ? D.ci_hkv_s[slot_][a.layer] : nullptr,
                           has_idx ? D.ci_hsc_s[slot_][a.layer] : nullptr, iw, nb, pos0, hist, st);
      D.c_np[a.layer] = 0;
    }
  }
  ++st_.calls;
  return true;
}

// The lightning indexer for one token. Same five kernels the batched path runs, with n = 1 and the
// mask landing in `a_allow`, which is where attn_run reads it — so nothing crosses the bus and the
// host never sees a score.
bool DenseGpu::indexer_one(uint32_t layer, int32_t h, uint64_t qr, uint32_t n_head, uint32_t dim,
                           uint32_t n_keys, uint32_t n_mask, uint32_t topk, uint32_t pos,
                           uint32_t n_rot, RopeDerived rope) {
  if (!attn_ok_ || h < 0 || (size_t)h >= ent_.size() || dim != kIdxDim || !n_keys) return false;
  AFF_TRACE(28);
  const Entry& e = ent_[h];
  if (e.split != Split::Rep || e.rows < (uint64_t)n_head * dim || n_keys > a_maxcomp_) return false;
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    hipStream_t st = D.stream;
    if (!D.d_qrn || layer >= D.a_idx8_s[slot_].size() || !D.a_idx8_s[slot_][kvl(layer)] || !D.i_w) return false;
    if (!idx_grow(&D, 1, n_head, dim)) return false;
    launch_mv({e.sh[d].data, e.sh[d].scales, e.quant, e.sh[d].ps}, D.d_qrn, (uint32_t)(n_head * dim),
              (uint32_t)qr, D.i_q, st);
    idx_rope_qat_hip(D.i_q, 1, n_head, n_rot, pos, rope.freq_scale, rope.theta_scale,
                     rope.ext_factor, rope.lo, rope.hi, st);
    idx_quant_rows_hip(D.i_q, D.i_q8, D.i_qs, 1, n_head * dim, st);
    indexer_scores_hip(D.i_q8, D.i_qs, D.a_idx8_s[slot_][kvl(layer)], D.a_idxs_s[slot_][kvl(layer)], D.i_w, D.i_sc, 1, n_keys,
                       n_head, st);
    // Null counts: at one token the score row IS its own prefix. The clear is over `n_mask`, the
    // attention's own row count for this token, and NOT over the plane's pitch: the pitch comes
    // from the cache's capacity, so at a 1M cache it is 262146 bytes however short the context is.
    const bool gather = topk <= kAdmCap;
    idx_topk_mask_hip(D.i_sc, D.a_allow, nullptr, 1, n_keys, a_maxcomp_,
                      n_mask > a_maxcomp_ ? a_maxcomp_ : n_mask, topk, st, &D.topk,
                      gather ? D.a_adm : nullptr, gather ? D.a_admn : nullptr, kAdmCap);
  }
  a_idx_layer_[slot_] = (int64_t)layer;
  a_idx_gather_[slot_] = topk <= kAdmCap;
  a_idx_topk_[slot_] = topk;
  ++st_.calls;
  return true;
}

bool DenseGpu::compress_reset() {
  if (!impl_) return false;
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    if (D.c_hkv_s[slot_].empty()) continue;
    AFF_HIP_CHECK(hipSetDevice(D.id));
    for (size_t l = 0; l < D.c_hkv_s[slot_].size(); ++l) {
      if (!D.c_hkv_s[slot_][l]) continue;
      const uint32_t r = D.c_ratio[l], hist = comp_hist(r);
      compressor_hist_reset_hip(D.c_hkv_s[slot_][l], D.c_hsc_s[slot_][l], comp_width(r, a_width_), hist, D.stream);
      if (D.ci_hkv_s[slot_][l])
        compressor_hist_reset_hip(D.ci_hkv_s[slot_][l], D.ci_hsc_s[slot_][l], comp_width(r, kIdxDim), hist, D.stream);
    }
  }
  return true;
}

bool DenseGpu::batch_indexer(uint32_t layer, uint32_t n, const uint32_t* n_comp, uint32_t n_keys,
                             uint32_t n_head, uint32_t dim, uint32_t topk, uint32_t pos0,
                             uint32_t n_rot, RopeDerived rope) {
  if (!b_nb_ || !n || !n_keys || dim != kIdxDim) return false;
  AFF_TRACE(28);
  if (n > b_nb_ || n_keys > a_maxcomp_) return false;
  // What the ATTENTION will read of each mask row, which is its own compressed-row count and not
  // the indexer's key count — the two compressors have separate capacities. The last token of the
  // chunk holds the largest, since n_comp only grows.
  uint32_t n_mask = n_keys;
  for (uint32_t t = 0; t < n; ++t) n_mask = std::max(n_mask, n_comp[t]);
  if (n_mask > a_maxcomp_) return false;
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    hipStream_t st = D.stream;
    if (!idx_grow(&D, n, n_head, dim)) return false;
    // The keys are ALREADY on device, in the form this wants: batch_compress wrote them as E4M3
    // rows with one f32 scale each when it emitted them. They used to arrive as host f32 — a
    // n_keys x 128 upload and a quantise pass, per layer per chunk, of rows that had not changed
    // since the chunk that produced them.
    if (layer >= D.a_idx8_s[slot_].size() || !D.a_idx8_s[slot_][kvl(layer)]) return false;
    std::memcpy(D.hi_nc, n_comp, (size_t)n * sizeof(uint32_t));
    AFF_HIP_CHECK(hipMemcpyAsync(D.i_nc, D.hi_nc, (size_t)n * sizeof(uint32_t),
                                 hipMemcpyHostToDevice, st));
    // RoPE and QAT here, not on the host: i_q holds the projection's RAW output, and i_w the head
    // weights — batch_indexer_q and batch_compress wrote both, on device.
    idx_rope_qat_hip(D.i_q, n, n_head, n_rot, pos0, rope.freq_scale, rope.theta_scale,
                     rope.ext_factor, rope.lo, rope.hi, st);
    // One scale a token over all 64 heads; the keys' own scales came with them.
    idx_quant_rows_hip(D.i_q, D.i_q8, D.i_qs, n, n_head * dim, st);
    // Scores in TILES of `i_tile_` tokens, because the score plane is [tokens][every candidate row]
    // and the candidate count is the context. Holding a whole 1536-token chunk of it costs 1.50 GiB
    // a card at a 1M cache, to carry values that idx_topk_mask consumes immediately and never reads
    // again. i_tile_ comes from a fixed BYTE budget (see idx_grow), so at every context short enough
    // for the plane to fit the tile is the whole chunk and this loop runs exactly once — same
    // launches, same arithmetic, same answer as before.
    //
    // Each token's row is independent, so tiling cannot move a score or a tie.
    for (uint32_t t0 = 0; t0 < n; t0 += i_tile_) {
      const uint32_t nt = std::min(i_tile_, n - t0);
      indexer_scores_hip(D.i_q8 + (size_t)t0 * n_head * dim, D.i_qs + t0, D.a_idx8_s[slot_][kvl(layer)],
                         D.a_idxs_s[slot_][kvl(layer)], D.i_w + (size_t)t0 * n_head, D.i_sc, nt, n_keys,
                         n_head, st);
      // Straight into b_allow at the CHUNK's own token indexing; batch_attend then offsets by its
      // sub-batch. The mask never leaves the card.
      const bool gather = topk <= kAdmCap;
      idx_topk_mask_hip(D.i_sc, D.b_allow + (size_t)t0 * a_maxcomp_, D.i_nc + t0, nt, n_keys,
                        a_maxcomp_, n_mask, topk, st, nullptr,
                        gather ? D.b_adm + (size_t)t0 * kAdmCap : nullptr,
                        gather ? D.b_admn + t0 : nullptr, kAdmCap);
    }
  }
  b_idx_layer_[slot_] = (int64_t)layer;
  a_idx_gather_[slot_] = topk <= kAdmCap;
  a_idx_topk_[slot_] = topk;
  ++st_.calls;
  return true;
}

bool DenseGpu::batch_attend(uint32_t layer, uint32_t b0, uint32_t n, uint32_t pos0,
                            const uint8_t* allowed, uint32_t mask_stride, const uint32_t* n_comp,
                            float scale) {
  if (!attn_ok_ || layer >= impl_->dev[0].a_raw_s[slot_].size() || !b_nb_ || !n || !a_ring_) return false;
  AFF_TRACE(25);
  if (b0 + n > b_nb_) return false;
  // The indexer writes b_allow for its OWN layer only. Keyed by layer rather than a bare flag: a
  // stale one would hand attention the previous layer's admissions, which is not a crash — it is a
  // model that quietly attends to the wrong rows.
  const bool dev_mask = b_idx_layer_[slot_] == (int64_t)layer;
  uint32_t n_comp_max = 0;
  if (n_comp) for (uint32_t t = 0; t < n; ++t) n_comp_max = std::max(n_comp_max, n_comp[t]);
  if (n_comp_max > a_maxcomp_) return false;          // caller falls back rather than truncating
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    if (!D.b_ncomp || !D.ncomp.fit((size_t)b_nb_ * sizeof(uint32_t), kRingCountBudget)) return false;
    // The ring slot, and the wait that says its last transfer is done. Only binds when the host is a
    // whole ring of layers ahead of the card, where it is not the thing holding anything up.
    uint32_t ns = 0;
    uint32_t* const pin = (uint32_t*)D.ncomp.claim(&ns);
    AFF_BLOCK_AT(aff::BlockSite::NcompRing, AFF_HIP_CHECK(D.ncomp.wait(ns)));
    if (n_comp) std::memcpy(pin, n_comp, n * sizeof(uint32_t));
    else        std::memset(pin, 0, n * sizeof(uint32_t));
    AFF_HIP_CHECK(hipMemcpyAsync(D.b_ncomp, pin, n * sizeof(uint32_t), hipMemcpyHostToDevice,
                                 D.stream));
    AFF_HIP_CHECK(D.ncomp.done(ns, D.stream));
    // b_allow is already correct for this layer: batch_indexer wrote it in place on this stream.
    // A caller arriving with a host-side mask instead has nowhere to put it, and proceeding would
    // hand attention the previous layer's admissions — not a crash, a model that quietly attends to
    // the wrong rows. So this refuses.
    if (!dev_mask && allowed && n_comp_max) {
      aff::ui::err("batch_attend: layer %u wants a host-supplied indexer mask, and the pinned "
                   "staging for one no longer exists. The device indexer covers every ratio-4 "
                   "layer; a layer outside that set must attend unmasked.\n", layer);
      return false;
    }
    // The flash kernel is the fp8 WMMA one; it reads FP8R and only FP8R, because a per-64 scale
    // cannot be folded out of the PV matrix multiply (see attention_gpu.h). It is compiled for a
    // 512-wide head and needs whole 16-head tiles, so anything else falls through to the kernel that
    // walks keys on the VALU. In isolation that is two orders of magnitude at short context, and the
    // difference between running and not at long context.
    const bool flash = a_dt_ == KvDtype::FP8R && a_width_ == kFlashWidth && (a_hpr_ % 16u) == 0u;
    const bool gather = dev_mask && a_idx_gather_[slot_];
    if (flash)
      mqa_attend_flash_hip(D.b_qtm, D.a_raw_s[slot_][layer], D.a_comp_s[slot_][kvl(layer)],
                           gather ? nullptr
                                  : (dev_mask ? D.b_allow + (size_t)b0 * a_maxcomp_
                                              : ((allowed && n_comp_max) ? D.b_allow : nullptr)),
                           gather ? D.b_admn + b0 : D.b_ncomp, /*n_comp1=*/0u, D.a_sink[layer],
                           D.b_htm, b0, n, b_nb_, pos0,
                           a_sliding_, a_ring_, a_maxcomp_, a_hpr_, scale, D.stream,
                           /*q_bf16=*/true, &D.fsplit, gather ? a_idx_topk_[slot_] : n_comp_max,
                           gather ? D.b_adm + (size_t)b0 * kAdmCap : nullptr, kAdmCap);
    else
      mqa_attend_batch_hip(a_dt_, D.b_qtm, D.a_raw_s[slot_][layer], D.a_comp_s[slot_][kvl(layer)],
                           dev_mask ? D.b_allow + (size_t)b0 * a_maxcomp_
                                    : ((allowed && n_comp_max) ? D.b_allow : nullptr),
                           D.b_ncomp, D.a_sink[layer], D.b_htm, b0, n, b_nb_, pos0, a_sliding_,
                           a_ring_, n_comp_max, a_maxcomp_, a_hpr_, a_width_, a_rot_, scale,
                           D.stream, /*q_bf16=*/true);
  }
  ++st_.calls;
  return true;                                        // no sync: every source was pinned
}

bool DenseGpu::batch_attn_out(int32_t hwa, int32_t hwb, uint32_t n_head, uint32_t width,
                              uint32_t n_rot, RopeDerived rope, uint32_t n_groups, uint64_t rank,
                              uint64_t hidden) {
  if (!b_nb_ || hwa < 0 || hwb < 0) return false;
  AFF_TRACE(26);
  if ((size_t)hwa >= ent_.size() || (size_t)hwb >= ent_.size()) return false;
  const Entry& A = ent_[hwa]; const Entry& B = ent_[hwb];
  if (A.split != Split::Row || B.split != Split::Col) return false;
  if (A.quant == AFF_FP8_E4M3 && (rank % 128)) return false;
  const size_t R = impl_->dev.size();
  const uint32_t group_dim = (n_head / n_groups) * width;
  for (size_t d = 0; d < R; ++d) {
    if (A.sh[d].row0 % rank || A.sh[d].nrows % rank) return false;
    if (A.sh[d].nrows != B.sh[d].ncols || A.sh[d].row0 != B.sh[d].col0) return false;
    if (A.sh[d].nrows > b_mid_) return false;
  }
  for (size_t d = 0; d < R; ++d) {
    Impl::Dev& D = impl_->dev[d];
    const uint32_t ng = (uint32_t)(A.sh[d].nrows / rank);
    AFF_HIP_CHECK(hipSetDevice(D.id));
    // Both sides of the flash kernel, taken here because here is where both are still live and the
    // shapes are in scope. b_heads aliases b_qtm and at this point still holds the transposed q the
    // kernel READ; b_htm aliases b_q and holds what it WROTE. See the alias note in batch_init.
    {
      AFF_TRACE(42);
      batch_transpose_bf16_hip(D.b_htm, D.b_heads, (uint32_t)((uint64_t)a_hpr_ * width), b_nb_, 0,
                               D.stream);
      // V carried the rotated tail, so undo the rotation. Sine negated: this is the INVERSE.
      batch_rope_tail_bf16_hip(D.b_heads, a_hpr_, width, n_rot, b_pos0_, rope, -1.0f, b_nb_,
                               D.stream);
    }
    // wo_a stays on the bf16 kernel and only wo_b goes W8A8. The grouped projection is the one shape
    // where W8A8 LOSES, because of the activation stride: its operand is the heads buffer, 16384
    // wide per card, and each group reads a 4096-wide window, so a token's row is 16 KiB from the
    // next and the fp8 kernel stages 128 scattered 32-byte pieces per step where the bf16 kernel
    // reads 32 consecutive floats a wave. On top of that, one quantise pass over [16384][nb] is
    // 67 MB of reads for four small GEMMs. Quantising each window compactly would fix both, at four
    // passes rather than one.
    {
      AFF_TRACE(43);
      batch_gemm_grouped_hip(A.quant, A.sh[d].data, A.sh[d].scales, D.b_heads, D.b_g, ng,
                             group_dim, (uint32_t)rank, b_nb_, D.stream, /*x_bf16=*/true);
    }
    // Row-split, so no single card's value is comparable between two rank counts — but the SUM over
    // cards is, because it is the same 64 heads either way, only dealt differently.
    xrank_trace((uint32_t)d, "woa", D.b_g, (size_t)ng * rank * b_nb_ * sizeof(float), D.stream);
    // Straight into the block: the attention sublayer's block is written, not accumulated, and
    // what lands here is this card's PARTIAL for block_reduce to finish.
    {
      AFF_TRACE(44);
      if (q8_ok(hwb, B.sh[d].ncols) && quant_act(d, D.b_g, (uint32_t)B.sh[d].ncols))
        batch_gemm_ent_q8(d, hwb, D.b_block, hidden, B.sh[d].ncols);
      else
        batch_gemm_ent(d, hwb, D.b_g, D.b_block, hidden, B.sh[d].ncols);
    }
    // This card's partial, BEFORE the crossing. With `attn_red` it brackets the all-reduce, which is
    // the only thing between two sublayers that both cards touch. See gpu/digest.h.
  }
  ++st_.calls;
  return true;
}

bool DenseGpu::batch_hc_post(bool has_control, uint32_t n_embd, uint32_t n_hc) {
  if (!b_nb_) return false;
  AFF_TRACE(27);
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    if (has_control) {
      // `zb`: this pass visits every element of the sublayer accumulator, so it leaves it zeroed
      // and the next sublayer needs no memset. See batch_hc_expand_reg_kernel.
      batch_hc_expand_hip(D.b_hc_next, D.b_block, D.b_hc, D.b_post, D.b_comb, n_embd, n_hc, b_nb_,
                          /*zb=*/1, D.stream);
      std::swap(D.b_hc, D.b_hc_next);
    } else {
      // Lane 0 only, exactly as the host and single-token paths do it for a layer with no control.
      batch_hc_add_bf16_hip(D.b_hc, D.b_block, (uint64_t)n_embd * b_nb_, /*zb=*/1, D.stream);
    }
  }
  return true;
}

bool DenseGpu::batch_router(int32_t h, uint32_t n_expert, uint64_t n_embd, float* logits) {
  AFF_TRACE(28);
  if (h < 0 || (size_t)h >= n_router_ || !b_nb_ || n_expert > b_mid_) return false;
  const size_t R = impl_->dev.size();
  for (size_t d = 0; d < R; ++d) {
    Impl::Dev& D = impl_->dev[d];
    if (!D.router_rows) continue;
    AFF_HIP_CHECK(hipSetDevice(D.id));
    batch_gemm_hip(AFF_BF16, D.router[h] + D.router_row0 * n_embd, nullptr, D.b_norm, D.b_y,
                   (uint32_t)D.router_rows, (uint32_t)n_embd, b_nb_, D.stream);
    batch_stage_out(d, D.b_y, (uint32_t)D.router_rows, 0);
  }
  for (size_t d = 0; d < R; ++d) {
    Impl::Dev& D = impl_->dev[d];
    if (!D.router_rows) continue;
    AFF_HIP_CHECK(hipSetDevice(D.id));
    AFF_DRAIN_AT(D, aff::BlockSite::Router);
    for (uint32_t b = 0; b < b_live_; ++b)
      std::memcpy(logits + (size_t)b * n_expert + D.router_row0,
                  D.h_tm + (size_t)b * D.router_rows, D.router_rows * sizeof(float));
  }
  ++st_.calls;
  return true;
}

bool DenseGpu::batch_norm_fetch(float* norm_host) {
  if (!b_nb_ || !norm_host) return false;
  AFF_TRACE(30);
  Impl::Dev& D = impl_->dev[0];                       // replicated: the two are identical
  AFF_HIP_CHECK(hipSetDevice(D.id));
  batch_stage_out(0, D.b_norm, b_norm_rows_, 0);
  AFF_DRAIN_AT(D, aff::BlockSite::NormFetch);
  std::memcpy(norm_host, D.h_tm, (size_t)b_live_ * b_norm_rows_ * sizeof(float));
  ++st_.calls;
  return true;
}

bool DenseGpu::router_scratch(size_t rank, uint32_t n_expert, uint32_t k) {
  if (rank >= impl_->dev.size() || !b_cap_ || !n_expert || !k) return false;
  Impl::Dev& D = impl_->dev[rank];
  if (D.h_rn) return true;
  // b_cap_, NOT b_nb_. b_nb_ is THIS chunk's rounded token count, and the first chunk of a run can
  // be a short prompt; sizing from it and then meeting a full chunk overruns the allocation, which
  // surfaces as "invalid argument" on the copy rather than as a bad answer. h_rn is assigned last
  // and is the guard, so a partial failure retries rather than leaving half-null pointers behind a
  // satisfied test.
  const size_t nk = (size_t)b_cap_ * k;
  // sel, wt and n in ONE allocation, device and pinned alike, in that order: the three come back
  // together every layer, and as three allocations that is three D2H copies — each of which ROCm
  // runs as a copyBuffer kernel carrying a full dispatch gap. Contiguous, it is one copy. bias and
  // res stay separate: they go the other way, and only when placement supplies them.
  const size_t o_wt = nk * sizeof(uint32_t), o_n = o_wt + nk * sizeof(float);
  const size_t back = o_n + (size_t)b_cap_ * sizeof(uint32_t);
  void* p[5] = {nullptr};
  bool ok = hipMalloc(&p[0], back) == hipSuccess &&
            // One slice per router handle, not one buffer reused: the bias is a per-LAYER constant
            // and uploading it per call was a pageable, host-synchronous copy on the hot path.
            // 43 x 256 floats = 44 KB for the whole model.
            hipMalloc(&p[1], (size_t)n_router_ * n_expert * sizeof(float)) == hipSuccess &&
            hipMalloc(&p[2], n_expert) == hipSuccess &&
            // Pinned: these come straight back, and a pageable D2H is a bounce-buffer copy.
            hipHostMalloc(&p[3], back, hipHostMallocDefault) == hipSuccess;
  if (!ok) {
    for (int i = 0; i < 3; ++i) if (p[i]) (void)hipFree(p[i]);
    if (p[3]) (void)hipHostFree(p[3]);
    return false;
  }
  // The UPWARD staging is a ring of the same block, because a hash layer hands the card a selection
  // the host looked up and then goes straight on to the next layer. `hb` above stays a single block:
  // it is the D2H direction, where the host is the consumer and the copy has already landed by the
  // time anything reads it. Only the H2D direction can be overwritten under a pending DMA.
  if (!D.rsel.fit(back, kRingCountBudget)) {
    for (int i = 0; i < 3; ++i) if (p[i]) (void)hipFree(p[i]);
    (void)hipHostFree(p[3]);
    return false;
  }
  uint8_t* const db = (uint8_t*)p[0];
  uint8_t* const hb = (uint8_t*)p[3];
  D.b_rsel = (uint32_t*)db; D.b_rwt = (float*)(db + o_wt); D.b_rn = (uint32_t*)(db + o_n);
  D.b_rbias = (float*)p[1]; D.b_rres = (uint8_t*)p[2];
  D.rbias_up.assign(n_router_, 0u);
  D.r_back = back; D.r_woff = o_wt; D.r_noff = o_n;
  D.h_rsel = (uint32_t*)hb; D.h_rwt = (float*)(hb + o_wt);
  D.b_rsel_m = D.b_rsel; D.b_rwt_m = D.b_rwt;        // rank 0 reads its own block
  D.h_rn = (uint32_t*)(hb + o_n);                    // last: it is the guard above
  return true;
}

// ---- the selection, pushed to every other card ---------------------------------------------------
//
// One allocation per rank laid out exactly like rank 0's, so the push is a single peer copy of the
// whole sel/wt/n block rather than three, and `route_dev` hands every rank the same shape.
bool DenseGpu::route_mirror(uint32_t n_expert, uint32_t k) {
  const size_t R = impl_->dev.size();
  if (R < 2) return true;
  Impl::Dev& D0 = impl_->dev[0];
  if (!D0.r_back) return false;
  for (size_t r = 1; r < R; ++r) {
    Impl::Dev& D = impl_->dev[r];
    if (D.m_ring) continue;
    AFF_HIP_CHECK(hipSetDevice(D.id));
    void* p = nullptr;
    if (hipMalloc(&p, D0.r_back * Impl::Dev::kRouteSlots) != hipSuccess) return false;
    D.m_ring = p;
  }
  AFF_HIP_CHECK(hipSetDevice(D0.id));
  for (uint32_t s = 0; s < Impl::Dev::kRouteSlots; ++s)
    if (!D0.ev_route[s] &&
        hipEventCreateWithFlags((hipEvent_t*)&D0.ev_route[s], hipEventDisableTiming) != hipSuccess)
      return false;
  (void)n_expert; (void)k;
  return true;
}

// Send rank 0's selection to every other card and record the event the readers wait on.
//
// Called from BOTH router paths, and that is the whole point of it being a function. The top-k path
// computes the selection on the card; the HASH path uploads one the host already has. Pushing from
// only the first left every hash layer's mirror holding whatever the previous routed layer put
// there — a selection that is in range, plausible, and for the wrong tokens. It cost three
// different `ids-md5` values from three runs of the same binary before the differential check named
// the layer.
void DenseGpu::route_publish() {
  if (impl_->dev.size() < 2 || route_ready_ != 1) return;
  Impl::Dev& D = impl_->dev[0];
  const uint32_t slot = (uint32_t)(route_seq_ % Impl::Dev::kRouteSlots);
  // Reusing a slot means reusing its event, so the record from a whole ring ago has to be done
  // first. Only blocks when the host is kRouteSlots layers ahead of the card, where it is not the
  // thing holding anything up.
  if (route_seq_ >= Impl::Dev::kRouteSlots)
    AFF_BLOCK_AT(aff::BlockSite::RouteRing,
                 AFF_HIP_CHECK(hipEventSynchronize((hipEvent_t)D.ev_route[slot])));
  for (size_t r = 1; r < impl_->dev.size(); ++r) {
    Impl::Dev& Dr = impl_->dev[r];
    if (!Dr.m_ring) continue;
    // THIS COSTS NOTHING, so do not replace it with a recompute. Both cards hold the full gate
    // replica, so each could route independently and skip the crossing entirely. Priced by issuing
    // this copy TWICE, which is bit-identical (same source, same destination, same stream) and
    // charges exactly one extra push: the doubled arm sits inside the run-to-run spread, so the
    // push is free. Recomputing would buy at most that, while putting the two cards back in the
    // business of deciding something independently — the failure mode the FP8 all-reduce asymmetry
    // already cost a long hunt to find.
    AFF_HIP_CHECK(hipMemcpyPeerAsync((uint8_t*)Dr.m_ring + (size_t)slot * D.r_back, Dr.id,
                                     D.b_rsel, D.id, D.r_back, D.stream));
  }
  AFF_HIP_CHECK(hipEventRecord((hipEvent_t)D.ev_route[slot], D.stream));
  ++route_seq_;
}

DenseGpu::RouteDev DenseGpu::route_dev(size_t r) const {
  RouteDev R{};
  if (!impl_ || r >= impl_->dev.size()) return R;
  // A refused mirror hands back nulls rather than rank 0's pointers, which on another card would be
  // in-bounds, wrong, and silent.
  if (impl_->dev.size() > 1 && route_ready_ != 1) return R;
  if (!route_seq_) return R;                   // nothing published yet
  const Impl::Dev& D = impl_->dev[r];
  // The slot the LAST publish used, which is the selection the caller is about to dispatch on.
  const uint32_t slot = (uint32_t)((route_seq_ - 1) % Impl::Dev::kRouteSlots);
  if (r == 0) {
    // Rank 0's producer is the top-k kernel and its consumer is the plan kernel, both on rank 0's
    // compute stream, so same-stream ordering is the whole guarantee and no ring is needed.
    R.sel = D.b_rsel_m; R.wt = D.b_rwt_m;
  } else {
    uint8_t* const base = (uint8_t*)D.m_ring + (size_t)slot * impl_->dev[0].r_back;
    R.sel = (const uint32_t*)base;
    R.wt = (const float*)(base + impl_->dev[0].r_woff);
  }
  // The event is rank 0's whatever rank is asking: it marks the instant the selection exists AND
  // has been pushed, which is the one thing every rank waits for.
  R.ready = impl_->dev[0].ev_route[slot];
  return R;
}

// The gate GEMM plus a gather, for the layers whose selection is already known.
//
// Everything here is batch_router_topk's, minus the top-k and plus an upload of the selection. The
// gate is the SAME [n_expert, n_embd] replica read the same way, so a hash layer's logits and a
// routed layer's logits are computed by one code path rather than two — which matters, because the
// host loop this replaces computed them a third way and the three would have had to be kept in
// agreement by hand.
bool DenseGpu::batch_router_hash(int32_t h, uint32_t n_expert, uint64_t n_embd, uint32_t k,
                                 const uint32_t* sel, const uint32_t* nsel, int norm_topk_prob,
                                 float routed_scaling, float* wt) {
  AFF_TRACE(31);
  if (h < 0 || (size_t)h >= n_router_ || !b_nb_ || n_expert > b_mid_ || !k) return false;
  // ---- A NULL `wt` MEANS "DO NOT BRING IT BACK" -------------------------------------------------
  //
  // The same contract batch_router_topk's null `sel` carries, for the same reason. The weights are
  // read by exactly one thing — the host's per-token expert arm — and under a device-built dispatch
  // that arm is unreachable, so the D2H below fed nothing and the drain behind it waited for the
  // whole previous layer to hand the host a number it would not look at.
  if (!sel || !nsel) return false;
  Impl::Dev& D = impl_->dev[0];
  if (!D.router[h]) return false;
  AFF_HIP_CHECK(hipSetDevice(D.id));
  if (!router_scratch(0, n_expert, k)) return false;
  // The mirror is latched by whichever router path runs first, and on this model that is a HASH
  // layer — layer 0. Latching it only in the top-k path left the first routed layers pushing
  // nothing at all.
  if (!route_ready_) route_ready_ = route_mirror(n_expert, k) ? 1 : 2;
  if (b_live_ > b_cap_) return false;

  // Up through a RING of the pinned block, not the single slot the results come back through.
  // `sel` is a caller local and the copy is asynchronous, so the DMA reads the staging after this
  // returns; one slot was safe only while the drain at the end of this function guaranteed it had.
  // `batch_attend` had exactly this shape without a drain to save it, and the result was an
  // attention output that differed from an identical q, non-deterministically.
  const size_t nk = (size_t)b_live_ * k;
  if (!D.rsel.ready()) return false;
  uint32_t rs = 0;
  uint8_t* const pin = D.rsel.claim(&rs);
  AFF_BLOCK_AT(aff::BlockSite::RouterHash, AFF_HIP_CHECK(D.rsel.wait(rs)));
  std::memcpy(pin, sel, nk * sizeof(uint32_t));
  std::memcpy(pin + D.r_noff, nsel, (size_t)b_live_ * sizeof(uint32_t));
  // The copy below is the WHOLE block, so it also carries the weight region — which the host does
  // not fill, because `router_hash_wt_hip` computes it on the card. With one staging buffer that
  // region held stale bytes, but the SAME stale bytes every call; with a ring it would hold whatever
  // that slot last had, and a padding slot's weight that no kernel overwrites would then differ
  // between two runs of one binary. Zeroed rather than narrowed to two copies: it is 144 bytes at
  // decode, and one DMA of a contiguous block is why the block is laid out contiguously.
  std::memset(pin + D.r_woff, 0, nk * sizeof(float));
  AFF_HIP_CHECK(hipMemcpyAsync(D.b_rsel, pin, D.r_back, hipMemcpyHostToDevice, D.stream));
  AFF_HIP_CHECK(D.rsel.done(rs, D.stream));

  { AFF_TRACE(33);
    batch_gemm_hip(AFF_BF16, D.router[h], nullptr, D.b_norm, D.b_y, n_expert,
                   (uint32_t)n_embd, b_nb_, D.stream); }
  { AFF_TRACE(34);
    router_hash_wt_hip(D.b_y, D.b_rsel, D.b_rn, D.b_rwt, b_nb_, b_live_, k, norm_topk_prob,
                       routed_scaling, D.stream); }
  // The mirror, before the readback and for the same reason: a hash layer's dispatch is device-built
  // too, and its selection came from the host rather than from the top-k.
  route_publish();
  if (wt) {
    AFF_HIP_CHECK(hipMemcpyAsync(D.h_rwt, D.b_rwt, nk * sizeof(float), hipMemcpyDeviceToHost,
                                 D.stream));
    AFF_DRAIN_AT(D, aff::BlockSite::RouterHash);
    std::memcpy(wt, D.h_rwt, nk * sizeof(float));  // NOLINT: h_rwt is a slice of the pinned block
  }
  ++st_.calls;
  return true;
}

bool DenseGpu::batch_router_topk(int32_t h, uint32_t n_expert, uint64_t n_embd, uint32_t k,
                                 const float* bias, const uint8_t* resident, int norm_topk_prob,
                                 float routed_scaling, uint32_t* sel, float* wt, uint32_t* nsel) {
  AFF_TRACE(31);
  if (h < 0 || (size_t)h >= n_router_ || !b_nb_ || n_expert > b_mid_ || !k) return false;
  // ---- A NULL `sel` MEANS "DO NOT BRING IT BACK" -------------------------------------------------
  //
  // The caller passes null when the dispatch will be built on the device and nothing else in the
  // layer wants the host copy. That deletes the D2H AND the drain behind it, which is the whole
  // point: the drain is a hipStreamSynchronize on the rank-0 compute stream, so it waits for the
  // entire previous layer including the hybrid tier's PCIe reads, 43 times a block.
  //
  // Spelled as a null out-parameter rather than a flag because the two cannot then drift: there is
  // no way to ask for the readback and not get it, or to skip it and still read `sel`.
  const bool back = sel != nullptr;
  if (back && (!wt || !nsel)) return false;
  // Rank 0, and the WHOLE gate — the top-k is a choice over all 256 logits and runs on device here,
  // so this path reads the full replica rather than `router_rows`, which is the decode split.
  Impl::Dev& D = impl_->dev[0];
  if (!D.router[h]) return false;
  AFF_HIP_CHECK(hipSetDevice(D.id));

  if (!router_scratch(0, n_expert, k)) return false;
  if (!route_ready_) route_ready_ = route_mirror(n_expert, k) ? 1 : 2;
  if (b_live_ > b_cap_) return false;
  // ---- the bias is per LAYER and never changes; the residency mask is per CALL -------------------
  //
  // Uploading them together, per call, on the argument that 1.25 KB a layer is cheaper than the
  // bookkeeping to keep a device copy in sync with placement, is true of the mask and false of the
  // bias — and the two are not separable while they share a buffer.
  //
  // The bias is `w.router_b`, a slice of the mmap'd container: one value per expert per layer, fixed
  // for the life of the process. 43 layers x 256 floats is 44 KB for the WHOLE MODEL, so it is
  // uploaded once per router handle into its own slice and never touched again.
  //
  // What that removes is not the kilobyte. The source is mmap'd model memory, which is PAGEABLE, so
  // the driver stages the copy through its own bounce buffer and the call is host-synchronous at
  // this size — an invisible stall, on the per-layer path, immediately before the drain below. It
  // does not appear in any profile as a copy; it appears as the drain being slightly longer.
  const float* dbias = nullptr;
  if (bias) {
    float* const slot = D.b_rbias + (size_t)h * n_expert;
    if (!D.rbias_up[(size_t)h]) {
      AFF_HIP_CHECK(hipMemcpyAsync(slot, bias, n_expert * sizeof(float), hipMemcpyHostToDevice,
                                   D.stream));
      D.rbias_up[(size_t)h] = 1;                 // ordered on D.stream against the top-k below
    }
    dbias = slot;
  }
  if (resident)
    AFF_HIP_CHECK(hipMemcpyAsync(D.b_rres, resident, n_expert, hipMemcpyHostToDevice, D.stream));

  { AFF_TRACE(33);
    batch_gemm_hip(AFF_BF16, D.router[h], nullptr, D.b_norm, D.b_y, n_expert,
                   (uint32_t)n_embd, b_nb_, D.stream); }
  { AFF_TRACE(34);
    router_topk_hip(D.b_y, dbias, resident ? D.b_rres : nullptr, D.b_rsel,
                    D.b_rwt, D.b_rn, n_expert, b_nb_, b_live_, k, norm_topk_prob, routed_scaling,
                    D.stream); }
  const size_t nk = (size_t)b_live_ * k;
  // ---- the push, before the readback -------------------------------------------------------------
  //
  // Every other card's dispatch reads the selection off its own memory rather than off a host copy,
  // so rank 0 sends it over the peer link and records one event the readers wait on. Issued BEFORE
  // the D2H because the D2H is the thing that is going away: the push is on the critical path and
  // the readback is not.
  //
  // The whole block, `sel` and `wt` and `n` together, for the reason router_scratch gives — three
  // copies is three dispatches, and at 43 layers a block that is the cost of the thing itself.
  route_publish();
  if (back) {
    // One copy of the whole sel/wt/n block, padding included — see router_scratch.
    AFF_HIP_CHECK(hipMemcpyAsync(D.h_rsel, D.b_rsel, D.r_back, hipMemcpyDeviceToHost, D.stream));
    AFF_DRAIN_AT(D, aff::BlockSite::RouterTopk);
    std::memcpy(sel, D.h_rsel, nk * sizeof(uint32_t));
    std::memcpy(wt, D.h_rwt, nk * sizeof(float));
    std::memcpy(nsel, D.h_rn, (size_t)b_live_ * sizeof(uint32_t));
  }
  // ---- the gate line: what the router thought of the 250 experts it did NOT pick -----------------
  //
  // ON THE --route-trace PATH ONLY, which already forces the host build and already drains here, so
  // this costs a 16 KiB readback on a path whose tok/s is not the shipping rate anyway.
  //
  // Recomputed on the host from the raw logits rather than read out of the kernel, because what
  // matters is the value the top-k ORDERS BY — sqrt(softplus(x)) + bias — and the kernel keeps only
  // the six it chose. `b_y` is [n_expert][b_nb_]; only the first b_live_ columns are real.
  if (route_trace().on() && back) {
    std::vector<float> hy((size_t)n_expert * b_nb_);
    std::vector<float> sc(n_expert);
    std::vector<uint32_t> q(n_expert);
    AFF_HIP_CHECK(hipMemcpyAsync(hy.data(), D.b_y, hy.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, D.stream));
    AFF_HIP_CHECK(hipStreamSynchronize(D.stream));
    float lo = 3.0e38f, hi = -3.0e38f;
    for (uint32_t e = 0; e < n_expert; ++e) {
      double acc = 0.0;
      for (uint32_t b = 0; b < b_live_; ++b) {
        const float x = hy[(size_t)e * b_nb_ + b];
        const float sp = (x > 20.0f) ? x : std::log1p(std::exp(x));
        acc += (double)std::sqrt(sp);
      }
      sc[e] = (float)acc + (bias ? bias[e] * (float)b_live_ : 0.0f);
      if (sc[e] < lo) lo = sc[e];
      if (sc[e] > hi) hi = sc[e];
    }
    // Quantised against the LINE'S OWN range, not a global one: the absolute scale of a gate value
    // means nothing across layers, the ordering and the spacing within a dispatch are the signal.
    const float span = hi > lo ? hi - lo : 1.0f;
    for (uint32_t e = 0; e < n_expert; ++e)
      q[e] = (uint32_t)(1000.0f * (sc[e] - lo) / span + 0.5f);
    route_trace().gate((uint32_t)h, q.data(), n_expert);
  }
  ++st_.calls;
  return true;
}

bool DenseGpu::batch_shexp(int32_t hg, int32_t hu, int32_t hd, uint64_t hidden, uint64_t inter,
                           float limit) {
  if (!b_nb_ || hg < 0 || hu < 0 || hd < 0) return false;
  AFF_TRACE(29);
  if (inter > b_mid_ || hidden > b_mid_) return false;
  const Entry& G = ent_[hg]; const Entry& U = ent_[hu]; const Entry& W = ent_[hd];
  if (G.split != Split::Row || U.split != Split::Row || W.split != Split::Col) return false;
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    if (G.sh[d].nrows != U.sh[d].nrows || G.sh[d].nrows != W.sh[d].ncols) return false;
    if (G.sh[d].row0 != W.sh[d].col0) return false;
  }
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    const uint32_t li = (uint32_t)G.sh[d].nrows;
    AFF_HIP_CHECK(hipSetDevice(D.id));
    const bool q8 = q8_ok(hg, hidden) && q8_ok(hu, hidden) && quant_act(d, D.b_norm, (uint32_t)hidden);
    // gate and up read the SAME activation and differ only in the weight, so fusing them lifts the
    // pair onto the streaming ceiling. Fused, `up` lands immediately after `gate` in b_g and `yu` is
    // where the SwiGLU reads it.
    float* yu = D.b_u;
    if (q8) {
      const BatchGemmJob bj[2] = {{G.sh[d].data, G.sh[d].scales, D.b_g, (uint32_t)li},
                                  {U.sh[d].data, U.sh[d].scales, nullptr, (uint32_t)li}};
      if (G.sh[d].ps == U.sh[d].ps &&                             // see attn_qkv: one flag a group
          batch_gemm_fp8_multi_hip(bj, 2, D.q8, D.qs, (uint32_t)hidden, b_nb_, D.q_dim,
                                   D.q_dim / 128u, D.stream, G.sh[d].ps)) {
        yu = D.b_g + (size_t)li * b_nb_;
      } else {
        batch_gemm_ent_q8(d, hg, D.b_g, li, hidden);
        batch_gemm_ent_q8(d, hu, D.b_u, li, hidden);
      }
    } else {
      batch_gemm_ent(d, hg, D.b_norm, D.b_g, li, hidden);
      batch_gemm_ent(d, hu, D.b_norm, D.b_u, li, hidden);
    }
    // Straight into the block, which hc_post left zeroed and the routed experts also add to.
    // (There is no block_zero anywhere any more — hc_post leaves its accumulator zeroed.
    // This one is still a memset because prefill's arena is [nb][n_embd] and hc_post's grid covers
    // one token's worth of it, so there is no consumer whose grid already partitions the whole.)
    // The scratch this replaced cost a 25 MB write, a 25 MB read and a 50 MB read-modify-write a
    // call for an addition the GEMM's epilogue can do for nothing. Only the W8A8 path can: the bf16
    // kernel splits K when the grid is narrow and its reduce WRITES, so an accumulate there would
    // need the split-K pass to accumulate too. That path is a fallback for a container whose shexp
    // weights are not fp8 and it keeps the scratch.
    //
    // The SwiGLU is INSIDE the quantisation on the W8A8 path — `b_y` is never written there, and
    // the else branch below is the only thing that still needs it. Same expression in the same
    // order as batch_swiglu_hip, so the E4M3 that comes out is bit-identical to running the two.
    if (q8_ok(hd, li) && quant_act_swiglu(d, D.b_g, yu, li, limit)) {
      batch_gemm_ent_q8(d, hd, D.b_block, hidden, li, 0, 0, /*accum=*/true);
    } else {
      batch_swiglu_hip(D.b_g, yu, D.b_y, li, b_nb_, limit, D.stream);
      batch_gemm_ent(d, hd, D.b_y, D.b_qr, hidden, li);
      batch_hc_add_hip(D.b_block, D.b_qr, hidden * b_nb_, D.stream);
    }
  }
  ++st_.calls;
  return true;
}

// The host tier's contribution, added to rank 0 only. See block_add_host: it is already whole
// rather than a partial, so adding it on both cards would have block_reduce double it.
bool DenseGpu::batch_block_add(const float* src, uint32_t n_embd) {
  if (!b_nb_ || n_embd > b_mid_) return false;
  Impl::Dev& D = impl_->dev[0];
  AFF_HIP_CHECK(hipSetDevice(D.id));
  AFF_HIP_CHECK(hipMemsetAsync(D.b_emb, 0, (size_t)b_nb_ * n_embd * sizeof(float), D.stream));
  AFF_HIP_CHECK(hipMemcpyAsync(D.b_emb, src, (size_t)b_live_ * n_embd * sizeof(float),
                               hipMemcpyHostToDevice, D.stream));
  batch_transpose_hip(D.b_emb, D.b_qr, n_embd, b_nb_, 0, D.stream);
  batch_hc_add_hip(D.b_block, D.b_qr, (uint64_t)n_embd * b_nb_, D.stream);
  AFF_DRAIN_AT(D, aff::BlockSite::BlockAdd);                                       // `src` is a caller local
  return true;
}

bool DenseGpu::batch_block_reduce(uint32_t n_embd) {
  if (!b_nb_) return false;
  AFF_TRACE(30);
  float* p[kMaxRanks];
  for (size_t d = 0; d < impl_->dev.size(); ++d) p[d] = impl_->dev[d].b_block;
  reduce(p, p, (uint64_t)n_embd * b_nb_);
  // Post-reduce, so it is the one value every card holds and therefore the one thing comparable
  // between two RANK COUNTS. The partials before it are each a different share and always differ.
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    AFF_HIP_CHECK(hipSetDevice(impl_->dev[d].id));   // reduce() left the last rank's device current
    xrank_trace((uint32_t)d, "blk", impl_->dev[d].b_block,
                (size_t)n_embd * b_nb_ * sizeof(float), impl_->dev[d].stream);
  }
  // Did a rank ever give up waiting for its peer? A plain load of pinned memory — no copy and no
  // synchronise — but it is an uncached PCIe read, so not every call. Whatever has landed by the
  // 512th is what gets seen; the point is that the process stops rather than emitting a whole run
  // of wrong text, not that it stops on the exact collective.
  if ((++tp_calls_ & 511u) == 0u) tp_check(impl_->tp);
  return true;
}

bool DenseGpu::batch_hc_read(uint32_t b, float* dst, uint32_t n_embd, uint32_t n_hc) {
  if (!b_nb_) return false;
  Impl::Dev& D = impl_->dev[0];                       // replicated
  AFF_HIP_CHECK(hipSetDevice(D.id));
  const uint32_t n = n_embd * n_hc;
  batch_column_bf16_hip(D.b_hc, D.d_g, n, b_nb_, b, D.stream);
  AFF_HIP_CHECK(hipMemcpyAsync(D.h_y, D.d_g, (size_t)n * sizeof(float), hipMemcpyDeviceToHost,
                               D.stream));
  AFF_DRAIN_AT(D, aff::BlockSite::HcRead);
  std::memcpy(dst, D.h_y, (size_t)n * sizeof(float));
  return true;
}

// ---- the DSpark tap, and why it has no host side at all ------------------------------------------
//
// `h.mean(dim=2)` — the activation DSpark conditions its draft on — for one tap layer of one chunk.
//
// Routing it through the host means: mean into a dim-major scratch, transpose, D2H, DRAIN because
// the host wants the bytes, memcpy into a host plane, and later an H2D of that plane back onto both
// cards so draft_proj can read it. That drain was the single most expensive site in the decode
// blocking table, and every byte of the round trip was a value the card had just computed and would
// want again a few calls later.
//
// So the plane lives in VRAM on both cards and neither direction happens. `tap_rows` is at most
// `sliding` (128 here) and the plane is [rows][taps * n_embd], so it is 6 MiB a card — well under
// what one resident expert is worth in decode time.
//
// Both cards, not rank 0 and a push: the hc lanes are replicated, so each card recomputes the
// identical mean from identical inputs. That is one small kernel against a peer copy plus the event
// pair that would have to order it.
bool DenseGpu::batch_hc_tap(uint32_t slot, uint32_t n_taps, uint32_t b_lo, uint32_t n_keep,
                            uint32_t row0, uint32_t rows_cap, uint32_t n_embd, uint32_t n_hc) {
  if (!b_nb_ || n_embd > b_mid_ || !n_taps || slot >= n_taps) return false;
  if (b_lo + n_keep > b_live_) return false;
  const size_t stride = (size_t)n_taps * n_embd;
  const size_t want = (size_t)rows_cap * stride * sizeof(float);
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    // Grown, never shrunk: `rows_cap` is `sliding` for the prompt's tail and B+1 for every block
    // after it, so one allocation on the first chunk serves the whole run.
    if (D.tap_bytes_s[slot_] < want) {
      if (D.b_tap_s[slot_]) (void)hipFree(D.b_tap_s[slot_]);
      D.b_tap_s[slot_] = nullptr; D.tap_bytes_s[slot_] = 0;
      if (hipMalloc(&D.b_tap_s[slot_], want) != hipSuccess) return false;
      // Zeroed once so a row the taps never reach — a chunk shorter than the window — carries
      // zeros rather than whatever the allocator returned. The host plane it replaces was
      // value-initialised for the same reason.
      AFF_HIP_CHECK(hipMemsetAsync(D.b_tap_s[slot_], 0, want, D.stream));
      D.tap_bytes_s[slot_] = want;
    }
    batch_hc_tap_hip(D.b_hc, D.b_tap_s[slot_], n_embd, n_hc, b_nb_, b_lo, n_keep, row0, rows_cap,
                     (uint32_t)stride, slot * n_embd, D.stream);
  }
  return true;
}

// The vocabulary head for EVERY token of the block, and only the winner comes back.
//
// Prefill runs the head once, for the last token, through DenseOps::mv. A speculative block needs
// every position — position j's argmax is what the draft's token j+1 is checked against — and the
// head is the largest weight on the card. One matvec per position reads it once per position; one
// GEMM over those columns reads it once, which is the same reason the batched prefill exists at all.
//
// `x` is the collapsed, normed hidden state, token-major, because that is the shape the host
// hc_head leaves it in. It is transposed on the card rather than by the caller.
// ---- the head, sampled --------------------------------------------------------------------------
//
// Same GEMM as batch_head; what differs is the reduction. An argmax can be taken per card and
// combined from two (value, id) pairs, because the largest of two maxima is the maximum. A DRAW
// cannot: temperature, a nucleus and a renormalisation are all properties of the whole distribution,
// and half of one is not a distribution.
//
// So the peer's band is gathered first. `b_head` is already allocated at full vocabulary on every
// card — only the card's own band is written by the GEMM — so the gather is a peer copy into the
// hole rather than a new allocation. It is ~1.5 MiB for a verify block, tens of microseconds, and it
// buys a sampler that needs no cross-card step at all afterwards.
bool DenseGpu::batch_head_sample(int32_t h, uint64_t vocab, uint64_t n_embd, uint32_t n,
                                 const float* x, const uint32_t* query, const float* u,
                                 float temperature, float top_p, uint32_t top_k, float min_p,
                                 Model::PosDraw* out) {
  // Named refusals, not bare ones. The only caller aborts the run on a false return, and "the
  // verify block refused" with no reason costs a build-and-bisect to answer a question the failing
  // condition already knew.
  auto no = [&](const char* why) {
    aff::ui::err("batch_head_sample: %s (nb %u live %u n %u vocab %llu of %llu)\n", why, b_nb_,
                 b_live_, n, (unsigned long long)vocab, (unsigned long long)b_vocab_);
    return false;
  };
  if (!b_nb_ || !n || !out || h < 0 || (size_t)h >= ent_.size()) return no("no block, or a bad handle");
  if (n > b_live_) return no("more positions than the block is live for");
  if (b_nb_ > kNarrowTok) return no("the block is wider than the narrow head");
  if (vocab > b_vocab_) return no("vocabulary wider than the block head plane");
  const Entry& e = ent_[h];
  if (e.split == Split::Col) return no("the head is Col-split; it must be Row or Rep");
  if (e.cols != n_embd) return no("the head's column count is not n_embd");
  const size_t R = impl_->dev.size();

  // ---- THE BANDS MUST TILE THE VOCABULARY --------------------------------------------------
  //
  // Every row of the plane is written by exactly one card and read by the sampler. A row no card
  // owns keeps whatever was there — zero at allocation — and a zero logit is not an absent token,
  // it is a MID-RANKED one, so the nucleus keeps it and the draw can land on it. The symptom is a
  // narrow, repetitive vocabulary rather than an error, which is why this is asserted rather than
  // assumed: the row split rounds its chunk up to kRowAlign, so whether the last band reaches the
  // end is a property of the rank count and the vocabulary size together.
  uint64_t covered = 0;
  // The GEMM, not the multi-x matvec: sampling reads the plane with a row stride, and the matvec's
  // token-major output would need a different one. The verify block is past kMvMultiXFast anyway,
  // so this is the path it would have taken.
  for (size_t d = 0; d < R; ++d) {
    Impl::Dev& D = impl_->dev[d];
    if (!D.b_head) return false;
    AFF_HIP_CHECK(hipSetDevice(D.id));
    if (x) {
      AFF_HIP_CHECK(hipMemsetAsync(D.b_emb, 0, (size_t)n_embd * b_nb_ * sizeof(float), D.stream));
      AFF_HIP_CHECK(hipMemcpyAsync(D.b_emb, x, (size_t)n * n_embd * sizeof(float),
                                   hipMemcpyHostToDevice, D.stream));
      batch_transpose_hip(D.b_emb, D.b_norm, (uint32_t)n_embd, b_nb_, 0, D.stream);
    }
    uint64_t r0 = 0, nr = 0;
    row_band(e.sh[d], vocab, &r0, &nr);
    covered += nr;
    if (!nr) continue;
    // Into the band's own place in the full plane, so the gather below is a copy and not a shuffle.
    batch_gemm_hip(e.quant, shard_rows(e.sh[d], e.quant, r0 - e.sh[d].row0),
                   shard_scales(e.sh[d], r0 - e.sh[d].row0), D.b_norm,
                   D.b_head + (size_t)r0 * b_nb_, (uint32_t)nr, (uint32_t)n_embd, b_nb_, D.stream);
  }
  if (covered != vocab)
    return no("the cards' bands do not tile the vocabulary");
  for (size_t d = 0; d < R; ++d) {
    AFF_HIP_CHECK(hipSetDevice(impl_->dev[d].id));
    AFF_DRAIN_AT(impl_->dev[d], aff::BlockSite::HeadSample);
  }

  // Gather every other card's band onto card 0.
  Impl::Dev& D0 = impl_->dev[0];
  for (size_t d = 1; d < R; ++d) {
    uint64_t r0 = 0, nr = 0;
    row_band(ent_[h].sh[d], vocab, &r0, &nr);
    if (!nr) continue;
    AFF_HIP_CHECK(hipMemcpyPeer(D0.b_head + (size_t)r0 * b_nb_, D0.id,
                                impl_->dev[d].b_head + (size_t)r0 * b_nb_, impl_->dev[d].id,
                                (size_t)nr * b_nb_ * sizeof(float)));
  }

  AFF_HIP_CHECK(hipSetDevice(D0.id));
  if (!smp_cap_ || smp_cap_ < n) {
    for (void** p : {(void**)&smp_q_, (void**)&smp_u_, (void**)&smp_out_})
      if (*p) { (void)hipFree(*p); *p = nullptr; }
    if (hipMalloc(&smp_q_, (size_t)kNarrowTok * sizeof(uint32_t)) != hipSuccess) return false;
    if (hipMalloc(&smp_u_, (size_t)2 * kNarrowTok * sizeof(float)) != hipSuccess) return false;
    if (hipMalloc(&smp_out_, (size_t)kNarrowTok * sizeof(SpecDraw)) != hipSuccess) return false;
    smp_cap_ = kNarrowTok;
  }
  AFF_HIP_CHECK(hipMemcpyAsync(smp_q_, query, (size_t)n * sizeof(uint32_t), hipMemcpyHostToDevice,
                               D0.stream));
  AFF_HIP_CHECK(hipMemcpyAsync(smp_u_, u, (size_t)2 * n * sizeof(float), hipMemcpyHostToDevice,
                               D0.stream));
  SampleParams sp;
  sp.temperature = temperature; sp.top_p = top_p; sp.top_k = top_k; sp.min_p = min_p;
  // [vocab][nb] dim-major, as the GEMM leaves it: row stride nb, column stride 1.
  if (!batch_sample_hip(D0.b_head, (uint32_t)vocab, n, b_nb_, 1, (const uint32_t*)smp_q_,
                        (const float*)smp_u_, sp, (SpecDraw*)smp_out_, D0.stream))
    return no("batch_sample_hip refused the plane");
  std::vector<SpecDraw> host(n);
  AFF_HIP_CHECK(hipMemcpyAsync(host.data(), smp_out_, (size_t)n * sizeof(SpecDraw),
                               hipMemcpyDeviceToHost, D0.stream));
  AFF_DRAIN_AT(D0, aff::BlockSite::HeadSample);
  for (uint32_t i = 0; i < n; ++i)
    out[i] = Model::PosDraw{host[i].tok, host[i].tok_excl, host[i].p_query};
  ++st_.calls;
  return true;
}

// `x == nullptr` means `collapse` has already left the normed activation on the card, in both
// layouts — b_norm dim-major and b_emb token-major — and there is nothing to stage. That is the
// block path; a host pointer is prefill's, and the reference epilogue is what it was checked
// against. Same contract draft_head carries, for the same reason.
bool DenseGpu::batch_head(int32_t h, uint64_t vocab, uint64_t n_embd, uint32_t n, const float* x,
                          uint32_t* ids) {
  if (!b_nb_ || !n || !ids || h < 0 || (size_t)h >= ent_.size()) return false;
  if (n > b_live_ || b_nb_ > kNarrowTok || vocab > b_vocab_) return false;
  const Entry& e = ent_[h];
  if (e.split == Split::Col || e.cols != n_embd) return false;
  const size_t R = impl_->dev.size();
  // kMvMultiXFast and not kMvMultiXMax: the verify block is B+1 = 6 positions wide, which is one
  // past the cliff, so the target's head keeps the GEMM. See the constant.
  const bool mx = e.quant == AFF_BF16 && n <= kMvMultiXFast;
  uint64_t band[kMaxRanks] = {};                  // nr per card, for the combine below
  for (size_t d = 0; d < R; ++d) {
    Impl::Dev& D = impl_->dev[d];
    if (!D.b_head || !D.b_amx || !D.b_apeer || !D.b_hids || !D.h_hids || !D.ev_amx) return false;
    AFF_HIP_CHECK(hipSetDevice(D.id));
    // The padding columns go through the transpose and the GEMM like any other, so the source of
    // them has to be finite; b_emb is the arena's upload slot and nothing else reads it here. The
    // matvec reads only the `n` live vectors, so neither the zeroing nor the transpose is its
    // business — two launches that exist for the fallback and are skipped when it is not taken.
    if (x) {
      if (!mx) {
        AFF_HIP_CHECK(hipMemsetAsync(D.b_emb, 0, (size_t)n_embd * b_nb_ * sizeof(float),
                                     D.stream));
      }
      AFF_HIP_CHECK(hipMemcpyAsync(D.b_emb, x, (size_t)n * n_embd * sizeof(float),
                                   hipMemcpyHostToDevice, D.stream));
      if (!mx) batch_transpose_hip(D.b_emb, D.b_norm, (uint32_t)n_embd, b_nb_, 0, D.stream);
    }
    uint64_t r0 = 0, nr = 0;
    row_band(e.sh[d], vocab, &r0, &nr);
    band[d] = nr;
    if (!nr) continue;
    // The same choice draft_head makes, going the other way at the verify block's width: B+1 = 6
    // columns is one past kMvMultiXFast, so `mx` is false and this is the GEMM. The branch stays
    // because `n` is the caller's and a narrower block would want the matvec.
    if (mx) {
      const MvJob j{shard_rows(e.sh[d], e.quant, r0 - e.sh[d].row0), nullptr, e.quant,
                    (uint32_t)nr, D.b_head, 1.0f};
      if (!matvec_multix_hip(&j, 1, D.b_emb, (uint32_t)n_embd, n, (uint32_t)n_embd, D.stream))
        return false;
    } else {
      batch_gemm_hip(e.quant, shard_rows(e.sh[d], e.quant, r0 - e.sh[d].row0),
                     shard_scales(e.sh[d], r0 - e.sh[d].row0), D.b_norm, D.b_head, (uint32_t)nr,
                     (uint32_t)n_embd, b_nb_, D.stream);
    }
    batch_argmax_hip(D.b_head, D.b_amx, (uint32_t)nr, b_nb_, n, (uint32_t)r0, D.stream,
                     mx ? 1 : 0);
    // The instant this card's plane exists. Rank 0's stream waits on rank 1's record; nothing waits
    // on rank 0's, and it costs one event record a block to have it.
    AFF_HIP_CHECK(hipEventRecord(D.ev_amx, D.stream));
  }
  // ---- THE CROSS-CARD COMBINE, ON THE CARD -----------------------------------------------------
  //
  // The vocabulary is Row-split, so each card knows the best of its own band and nothing about the
  // others'. Folding one (value, id) pair a position a card is the whole exchange — shipping the
  // logits themselves would be 3.1 MB a block over PCIe for an argmax.
  //
  // It used to be run on the HOST, which meant a `hipStreamSynchronize` PER CARD to get the planes
  // back: a drain a card a verify block, each waiting on that card's whole queued chain. The
  // comparison is `higher value, then lower id`, which is one packed 64-bit max, so it is 128 bytes
  // a card over the peer link and a kernel — and then ONE readback of the answer instead of one of
  // every input. Exactly the shape draft_head's Markov chain already took.
  //
  // Bit-identity is not an argument here, it is arithmetic: the kernel applies the same comparison
  // to the same pairs, `markov_key`'s float map is order-preserving on finite logits, and `max` is
  // associative, so the fold is independent of how many cards there are and of their order.
  {
    size_t nband = 0;
    for (size_t d = 0; d < R; ++d) nband += band[d] ? 1 : 0;
    if (!nband) return false;                             // no card owns any of the vocabulary
    Impl::Dev& D = impl_->dev[0];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    // A card with no band skipped its argmax, so its `b_amx` holds whatever the last call left
    // there and must not enter the fold. Only the cards that DID write one are staged.
    const size_t plane = 2u * (size_t)b_nb_;
    uint32_t staged = 0;
    for (size_t d = 1; d < R; ++d) {
      if (!band[d]) continue;
      Impl::Dev& P = impl_->dev[d];
      AFF_HIP_CHECK(hipStreamWaitEvent(D.stream, P.ev_amx, 0));
      AFF_HIP_CHECK(hipMemcpyPeerAsync(D.b_apeer + staged * plane, D.id, P.b_amx, P.id,
                                       plane * sizeof(float), D.stream));
      ++staged;
    }
    // When rank 0 has no band its own plane is meaningless, so the first staged peer stands in as
    // the fold's seed and the rest follow it. `max` is commutative, so which one seeds it cannot
    // change the winner.
    const float* const own = band[0] ? D.b_amx : D.b_apeer;
    const float* const peers = band[0] ? D.b_apeer : D.b_apeer + plane;
    const uint32_t npeer = band[0] ? staged : staged - 1;
    batch_head_combine_hip(own, peers, npeer, D.b_hids, n, b_nb_, D.stream);
    AFF_HIP_CHECK(hipMemcpyAsync(D.h_hids, D.b_hids, (size_t)n * sizeof(uint32_t),
                                 hipMemcpyDeviceToHost, D.stream));
    AFF_DRAIN_AT(D, aff::BlockSite::Head);
    std::memcpy(ids, D.h_hids, (size_t)n * sizeof(uint32_t));
  }
  ++st_.calls;
  return true;
}

// ==================================================================================================
// The DSpark draft
// ==================================================================================================
//
// Four entry points, and everything else the draft needs is the batched path's, unchanged: the
// stages are structurally ordinary ratio-0 layers, so hc_pre, attn_qkv, attn_out, hc_post, the
// router and the shared expert are called exactly as prefill calls them, with the stage's handles
// and the layer index n_layer + stage.
//
// What is NOT shared is here:
//   draft_proj    main_norm(main_proj(tap)) — the one activation the draft sees of the target.
//   draft_kv      that activation as each stage's KV row, at its own position in the ring.
//   draft_attend  the flat, non-causal window a jointly-predicted block wants.
//   draft_head    the vocabulary head, then the serial Markov correction.

// main_norm(main_proj(tap)) for `n` positions, left in b_dm for the stages to read.
bool DenseGpu::draft_proj(int32_t h, int32_t norm_w, uint64_t rows, uint64_t cols, uint32_t n,
                          uint32_t tap_row0, float eps) {
  if (!n || n > kNarrowTok || h < 0 || (size_t)h >= ent_.size() || norm_w < 0) return false;
  if (cols > b_tapin_) return false;
  const Entry& e = ent_[h];
  // Rep, so both cards produce the whole vector and no exchange stands between the tap and the
  // first stage. split_for says so; this is where it would be noticed if it stopped.
  if (e.split != Split::Rep || e.rows != rows || e.cols != cols) return false;
  b_nb_ = kNarrowTok; b_live_ = n;
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    if (!D.b_dxt || !D.b_dx || !D.b_dm || !D.b_tap_s[slot_]) return false;
    if ((size_t)(tap_row0 + n) * cols * sizeof(float) > D.tap_bytes_s[slot_]) return false;
    AFF_HIP_CHECK(hipSetDevice(D.id));
    // The padding columns are transposed and multiplied like every other; zeroing the source is
    // what keeps them finite.
    AFF_HIP_CHECK(hipMemsetAsync(D.b_dxt, 0, cols * kNarrowTok * sizeof(float), D.stream));
    // ON THE CARD. `b_tap` already holds exactly this layout — [row][taps * n_embd], token-major,
    // written by batch_hc_tap as the target ran — so the tap's rows are a contiguous run and this is
    // a device-to-device copy of one. It replaces an H2D of a host plane whose bytes this card had
    // computed itself, and the drain that used to follow it.
    AFF_HIP_CHECK(hipMemcpyAsync(D.b_dxt, D.b_tap_s[slot_] + (size_t)tap_row0 * cols,
                                 (size_t)n * cols * sizeof(float), hipMemcpyDeviceToDevice,
                                 D.stream));
    batch_transpose_hip(D.b_dxt, D.b_dx, (uint32_t)cols, kNarrowTok, 0, D.stream);
    batch_gemm_ent(d, h, D.b_dx, D.b_dm, rows, cols);
    if (!D.vec[norm_w]) return false;
    batch_rms_norm_hip(D.b_dm, D.vec[norm_w], nullptr, D.b_dm, (uint32_t)rows, kNarrowTok, eps,
                       D.stream);
  }
  ++st_.calls;
  return true;
}

// `kv_norm(wkv(main_x))`, roped at each position, into the stage's ring — one row per position that
// has just become real. This is the ONLY way the target reaches the draft: the draft's residual
// stream starts from the noise token's embedding and knows nothing else.
uint32_t DenseGpu::kv_layers_debug() const {
  if (!impl_ || impl_->dev.empty() || slot_ >= impl_->dev[0].a_raw_s.size()) return 0;
  return (uint32_t)impl_->dev[0].a_raw_s[slot_].size();
}

uint64_t DenseGpu::kv_hash_debug(uint32_t layer) {
  if (!impl_ || impl_->dev.empty()) return 0;
  Impl::Dev& D = impl_->dev[0];
  if (slot_ >= D.a_raw_s.size() || layer >= D.a_raw_s[slot_].size() || !D.a_raw_s[slot_][layer])
    return 0;
  const size_t n = std::min<size_t>((size_t)a_ring_ * kv_row_bytes(a_dt_, a_width_, a_rot_), 4u*1024u*1024u);
  std::vector<uint8_t> h(n);
  AFF_HIP_CHECK(hipSetDevice(D.id));
  AFF_HIP_CHECK(hipMemcpy(h.data(), D.a_raw_s[slot_][layer], n, hipMemcpyDeviceToHost));
  uint64_t v = 1469598103934665603ull;
  for (size_t i = 0; i < n; ++i) { v ^= h[i]; v *= 1099511628211ull; }
  return v;
}

void DenseGpu::tp_reset_seq() {
  if (impl_ && impl_->tp.ready) aff::tp_reset_seq(impl_->tp);
}

void DenseGpu::sync_devices() {
  if (!impl_) return;
  for (auto& D : impl_->dev) { AFF_HIP_CHECK(hipSetDevice(D.id)); AFF_HIP_CHECK(hipDeviceSynchronize()); }
}

uint64_t DenseGpu::tap_hash_debug() {
  if (!impl_ || impl_->dev.empty()) return 0;
  Impl::Dev& D = impl_->dev[0];
  if (slot_ >= D.b_tap_s.size() || !D.b_tap_s[slot_] || !D.tap_bytes_s[slot_]) return 0;
  const size_t n = std::min<size_t>(D.tap_bytes_s[slot_], 4u*1024u*1024u);
  std::vector<uint8_t> h(n);
  AFF_HIP_CHECK(hipSetDevice(D.id));
  AFF_HIP_CHECK(hipMemcpy(h.data(), D.b_tap_s[slot_], n, hipMemcpyDeviceToHost));
  uint64_t v = 1469598103934665603ull;
  for (size_t i = 0; i < n; ++i) { v ^= h[i]; v *= 1099511628211ull; }
  return v;
}

bool DenseGpu::draft_kv(uint32_t layer, int32_t kv, int32_t kvnorm, uint32_t n, uint32_t pos0,
                        uint64_t hidden, uint32_t width, uint32_t n_rot, RopeDerived rope,
                        float eps) {
  if (!n || n > kNarrowTok || kv < 0 || (size_t)kv >= ent_.size() || kvnorm < 0) return false;
  if (!attn_ok_ || layer >= impl_->dev[0].a_raw_s[slot_].size()) return false;
  b_nb_ = kNarrowTok; b_live_ = n;
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    if (!D.b_dm || !D.vec[kvnorm]) return false;
    AFF_HIP_CHECK(hipSetDevice(D.id));
    // Still the bf16 kernel, which dequants this E4M3 weight through an LDS LUT and so runs the
    // matrix core at half rate on operands that are already fp8. W8A8 here is flat on ms a cycle and
    // it MOVES THE EMITTED TEXT — acceptance rose sharply but the speculative output stopped
    // matching the non-speculative one, which it had matched exactly. A draft that changes what the
    // target emits is not drafting.
    batch_gemm_ent(d, kv, D.b_dm, D.b_u, width, hidden);
    batch_rms_norm_rope_hip(D.b_u, D.vec[kvnorm], D.b_kv, width, n_rot, pos0, rope, eps,
                            kNarrowTok, D.stream);
  }
  return batch_kv_commit(layer, 0, n, pos0, width, n_rot);
}

// The draft's attention. Every query sees the same key set — see mqa_attend_batch_hip's header —
// so the window is computed once, on the host, out of positions rather than out of each query's.
bool DenseGpu::draft_attend(uint32_t layer, uint32_t n, uint32_t pos0, uint32_t win_lo,
                            float scale) {
  if (!attn_ok_ || layer >= impl_->dev[0].a_raw_s[slot_].size() || !b_nb_ || !n) return false;
  const uint32_t hi = pos0 + n - 1;
  if (win_lo > hi) return false;
  const uint32_t win = hi - win_lo + 1u;
  if (win > a_ring_) return false;                // the window may not lap its own ring
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    // Deliberately the VALU kernel and not the flash one: five queries over ~133 keys is a hundredth
    // of what the flash path's tiling is for, and this way the draft cannot perturb it.
    mqa_attend_batch_hip(a_dt_, D.b_qtm, D.a_raw_s[slot_][layer], nullptr, nullptr, nullptr,
                         D.a_sink[layer], D.b_htm, 0, n, b_nb_, pos0, a_sliding_, a_ring_, 0,
                         a_maxcomp_, a_hpr_, a_width_, a_rot_, scale, D.stream,
                         /*q_bf16=*/true, win, win_lo % a_ring_);
  }
  ++st_.calls;
  return true;
}

// The draft's head epilogue: structurally hc_pre with the expansion deleted — see
// batch_hc_head_narrow_hip. `fn` is the [n_hc, n_hc*n_embd] head gate, Rep-split so both cards
// derive the same four gates. As host code this was five round trips and 640 KB across the bus in
// both directions per block, for three kernels over data that never had to leave.
// V4.1's head: no mixer, no GEMM. `b_pre` still holds what the last layer's FFN computed — nothing
// between that hc_pre and here writes it — so the epilogue reuses it and is otherwise the same
// kernel, in the same order, as the V4 path above.
bool DenseGpu::collapse_pre(int32_t nw, uint32_t n_embd, uint32_t n_hc, float hc_eps,
                            float rms_eps) {
  if (!b_nb_ || b_nb_ != kNarrowTok || n_hc > 8) return false;
  if (nw < 0 || (size_t)nw >= n_vec_) return false;
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    if (!D.b_hc || !D.b_pre || !D.b_cur || !D.b_norm || !D.b_hcp) return false;
    AFF_HIP_CHECK(hipSetDevice(D.id));
    batch_hc_head_narrow_hip(D.b_hc, nullptr, nullptr, nullptr, D.vec[nw], D.b_cur, D.b_norm,
                             D.b_emb, D.b_hcp, n_embd, n_hc, b_nb_, hc_eps, rms_eps, D.stream,
                             D.b_pre);
    D.q8_src = nullptr;
  }
  b_norm_rows_ = n_embd;
  ++st_.calls;
  return true;
}

bool DenseGpu::draft_collapse(int32_t fn, int32_t sv, int32_t bv, int32_t nw, uint32_t n_embd,
                              uint32_t n_hc, float hc_eps, float rms_eps) {
  if (!b_nb_ || b_nb_ != kNarrowTok || n_hc > 8) return false;
  if (fn < 0 || (size_t)fn >= ent_.size() || sv < 0 || bv < 0 || nw < 0) return false;
  if ((size_t)sv >= n_vec_ || (size_t)bv >= n_vec_ || (size_t)nw >= n_vec_) return false;
  const Entry& F = ent_[fn];
  if (F.split != Split::Rep || F.rows != n_hc || F.cols != (uint64_t)n_embd * n_hc) return false;
  for (size_t d = 0; d < impl_->dev.size(); ++d) {
    Impl::Dev& D = impl_->dev[d];
    if (!D.b_hc || !D.b_mix || !D.b_cur || !D.b_norm || !D.b_hcp) return false;
    AFF_HIP_CHECK(hipSetDevice(D.id));
    if (!batch_gemm_bf16x_hip(F.quant, F.sh[d].data, F.sh[d].scales, D.b_hc, D.b_mix, n_hc,
                              (uint32_t)((uint64_t)n_embd * n_hc), b_nb_, D.stream))
      return false;
    // Both layouts, because the head takes whichever of its two kernels the shard allows: b_norm
    // dim-major for the WMMA GEMM, b_emb token-major for the multi-x matvec. The second is a store,
    // not a pass — see the note on hcn_norm_kernel.
    batch_hc_head_narrow_hip(D.b_hc, D.b_mix, D.vec[sv], D.vec[bv], D.vec[nw], D.b_cur, D.b_norm,
                             D.b_emb, D.b_hcp, n_embd, n_hc, b_nb_, hc_eps, rms_eps, D.stream,
                             nullptr);
    D.q8_src = nullptr;                               // b_norm is a different vector now
  }
  b_norm_rows_ = n_embd;
  ++st_.calls;
  return true;
}

// The vocabulary head for the block, then the Markov correction, then the block's ids.
//
// The correction is what makes the block autoregressive without running the three stages again:
// token i's embedding biases position i's logits, and the argmax of that IS token i+1. So the five
// steps are serial by construction and each one needs its predecessor's winner back on the host —
// the vocabulary is split across the cards and only the host sees both bands.
bool DenseGpu::draft_head(int32_t h_head, uint64_t vocab, uint64_t n_embd, uint32_t n,
                          const float* x, int32_t h_w1, int32_t h_w2, uint32_t rank,
                          uint32_t first_id, uint32_t* ids) {
  // x == nullptr means draft_collapse has already left the activation in b_norm, transposed, and
  // there is nothing to stage. That is the live path; the host form is kept because the reference
  // epilogue is what this was checked against.
  if (!n || n > kNarrowTok || !ids) return false;
  if (h_head < 0 || h_w1 < 0 || h_w2 < 0) return false;
  if ((size_t)h_head >= ent_.size() || (size_t)h_w1 >= ent_.size() ||
      (size_t)h_w2 >= ent_.size())
    return false;
  if (vocab > b_vocab_) return false;
  const Entry& H = ent_[h_head];
  const Entry& W1 = ent_[h_w1];
  const Entry& W2 = ent_[h_w2];
  if (H.split == Split::Col || H.cols != n_embd) return false;
  // Both replicated, so each card can slice w2 by whatever band its own head produced. split_for
  // says so; a Row split here would silently score the wrong rows.
  if (W1.split != Split::Rep || W2.split != Split::Rep) return false;
  if (W1.cols != rank || W2.cols != rank || W1.quant != AFF_BF16 || W2.quant != AFF_BF16)
    return false;
  const size_t R = impl_->dev.size();
  b_nb_ = kNarrowTok; b_live_ = n;
  uint64_t r0[kMaxRanks] = {0}, nr[kMaxRanks] = {0};
  // The multi-x matvec is bf16-only, needs the token-major activation draft_collapse leaves in
  // b_emb — so not when the caller staged its own — and is only the faster of the two kernels up to
  // kMvMultiXFast columns. The draft's block is B = 5, which is exactly the last width where it is.
  const bool mx = !x && H.quant == AFF_BF16 && n <= kMvMultiXFast;
  for (size_t d = 0; d < R; ++d) {
    Impl::Dev& D = impl_->dev[d];
    if (!D.b_head || !D.b_amx || !D.b_mbest) return false;
    AFF_HIP_CHECK(hipSetDevice(D.id));
    if (x) {
      AFF_HIP_CHECK(hipMemsetAsync(D.b_emb, 0, (size_t)n_embd * kNarrowTok * sizeof(float),
                                   D.stream));
      AFF_HIP_CHECK(hipMemcpyAsync(D.b_emb, x, (size_t)n * n_embd * sizeof(float),
                                   hipMemcpyHostToDevice, D.stream));
      batch_transpose_hip(D.b_emb, D.b_norm, (uint32_t)n_embd, kNarrowTok, 0, D.stream);
    }
    row_band(H.sh[d], vocab, &r0[d], &nr[d]);
    if (!nr[d]) continue;
    // THE MATVEC AND NOT THE GEMM, at this width and no wider. This is the largest single kernel in
    // the draft either way. The WMMA GEMM is held below the streaming ceiling by the LDS round trip
    // between the weight load and the accumulator, where the multi-x matvec reaches it — but only
    // while the activations it re-reads still fit in cache. kMvMultiXFast is where that stops, and
    // past it the matvec is much the worse of the two, so this does not generalise upward.
    //
    // `x` is token-major here, which is why draft_collapse writes both layouts, and so is `out`,
    // which is why markov_step and batch_argmax are told so.
    if (mx) {
      const MvJob j{shard_rows(H.sh[d], H.quant, r0[d] - H.sh[d].row0), nullptr, H.quant,
                    (uint32_t)nr[d], D.b_head, 1.0f};
      if (!matvec_multix_hip(&j, 1, D.b_emb, (uint32_t)n_embd, n, (uint32_t)n_embd, D.stream))
        return false;
    } else {
      batch_gemm_hip(H.quant, shard_rows(H.sh[d], H.quant, r0[d] - H.sh[d].row0),
                     shard_scales(H.sh[d], r0[d] - H.sh[d].row0), D.b_norm, D.b_head,
                     (uint32_t)nr[d], (uint32_t)n_embd, kNarrowTok, D.stream);
    }
  }
  // ---- THE CHAIN, ON THE CARDS -----------------------------------------------------------------
  //
  // Step i's winner is step i+1's input, so the five steps are serial. They used to be serial
  // THROUGH THE HOST — two stream drains a step, ten a block — because the winner was a launch
  // argument and a launch argument has to be a host value. It is now a device word, and the whole
  // chain is issued without the host learning any of it.
  //
  // The cross-card exchange is eight bytes a peer. Each card's step leaves its band's packed (value,
  // absolute row) maximum in `b_mbest`; each card then copies every OTHER card's key over the peer
  // link and takes the largest, which is the same "higher value, then lower id" comparison the host
  // used to run. Every card does it, on its own copy, so none is the authority and no second
  // crossing is needed to tell the losers what won — and `max` is associative, so the fold lands on
  // the same winner whatever order a card visits its peers in.
  //
  // Ordering is events, not waits: a card records after its step and the others' streams wait on
  // that record. `hipStreamWaitEvent` blocks the STREAM, never the thread issuing it.
  for (size_t d = 0; d < R; ++d) {
    Impl::Dev& D = impl_->dev[d];
    if (!D.b_ids || !D.b_mpeer || !D.tok.ready()) return false;
    AFF_HIP_CHECK(hipSetDevice(D.id));
    // The block's first token, the one host value the chain needs. Through a ring slot, because a
    // 4-byte H2D reads its source after this call returns exactly like a 4 MiB one does.
    uint32_t ts = 0;
    uint8_t* const pin = D.tok.claim(&ts);
    AFF_BLOCK_AT(aff::BlockSite::DraftHead, AFF_HIP_CHECK(D.tok.wait(ts)));
    std::memcpy(pin, &first_id, sizeof(uint32_t));
    AFF_HIP_CHECK(hipMemcpyAsync(D.b_ids, pin, sizeof(uint32_t), hipMemcpyHostToDevice, D.stream));
    AFF_HIP_CHECK(D.tok.done(ts, D.stream));
  }
  for (uint32_t i = 0; i < n; ++i) {
    for (size_t d = 0; d < R; ++d) {
      if (!nr[d]) continue;
      Impl::Dev& D = impl_->dev[d];
      AFF_HIP_CHECK(hipSetDevice(D.id));
      markov_step_hip((const uint16_t*)W1.sh[d].data, (const uint16_t*)W2.sh[d].data, D.b_ids,
                      D.b_head, D.b_mbest, (uint32_t)nr[d], (uint32_t)r0[d], rank, kNarrowTok, i,
                      D.stream, mx ? 1 : 0);
      AFF_HIP_CHECK(hipEventRecord(D.ev_mk[i], D.stream));
    }
    for (size_t d = 0; d < R; ++d) {
      if (!nr[d]) continue;
      Impl::Dev& D = impl_->dev[d];
      AFF_HIP_CHECK(hipSetDevice(D.id));
      // Every OTHER card that produced a key, pulled into this card's staging array. A card with no
      // band of its own contributes nothing, and a fold over no peers is the identity — which is
      // also the whole of the single-card path.
      uint32_t staged = 0;
      for (size_t o = 0; o < R; ++o) {
        if (o == d || !nr[o]) continue;
        Impl::Dev& P = impl_->dev[o];
        AFF_HIP_CHECK(hipStreamWaitEvent(D.stream, P.ev_mk[i], 0));
        AFF_HIP_CHECK(hipMemcpyPeerAsync(D.b_mpeer + staged, D.id, P.b_mbest, P.id,
                                         sizeof(uint64_t), D.stream));
        ++staged;
      }
      markov_combine_hip(D.b_mbest, D.b_mpeer, staged, D.b_ids, i, D.stream);
    }
  }
  // ONE readback for the whole block, where there were ten. `b_ids[0]` is the input token, so the
  // block's proposals are the n words after it.
  {
    Impl::Dev& D = impl_->dev[0];
    AFF_HIP_CHECK(hipSetDevice(D.id));
    AFF_HIP_CHECK(hipMemcpyAsync(D.h_ids, D.b_ids + 1, (size_t)n * sizeof(uint32_t),
                                 hipMemcpyDeviceToHost, D.stream));
    AFF_DRAIN_AT(D, aff::BlockSite::DraftHead);
    std::memcpy(ids, D.h_ids, (size_t)n * sizeof(uint32_t));
  }
  ++st_.calls;
  return true;
}

Model::BatchOps DenseGpu::batch_ops() {
  Model::BatchOps o;
  o.ctx = this;
  o.cap = [](void* c) { return static_cast<DenseGpu*>(c)->b_cap_; };
  o.begin = [](void* c, uint32_t nl, uint32_t p0, const float* e, uint32_t ne, uint32_t nh) {
    return static_cast<DenseGpu*>(c)->batch_begin(nl, p0, e, ne, nh);
  };
  o.hc_pre = [](void* c, int32_t fn, int32_t sv, int32_t bv, int32_t nw, uint32_t ne, uint32_t nh,
                uint32_t it, float he, float re, float* n) {
    return static_cast<DenseGpu*>(c)->batch_hc_pre(fn, sv, bv, nw, ne, nh, it, he, re, n);
  };
  o.attn_qkv = [](void* c, int32_t qa, int32_t qb, int32_t qn, int32_t kv, int32_t kvn,
                  uint64_t hid, uint64_t qr, uint32_t nh, uint32_t w, uint32_t nr, RopeDerived rp,
                  float eps, float* qo) {
    return static_cast<DenseGpu*>(c)->batch_attn_qkv(qa, qb, qn, kv, kvn, hid, qr, nh, w, nr, rp,
                                                     eps, qo);
  };
  o.mv_host = [](void* c, int src, const int32_t* h, const uint64_t* r, uint64_t cols, uint32_t n,
                 float* const* y) {
    return static_cast<DenseGpu*>(c)->batch_mv_host(src, h, r, cols, n, y);
  };
  o.head_sample = [](void* c, int32_t h, uint64_t vocab, uint64_t ne, uint32_t n, const float* x,
                     const uint32_t* q, const float* u, float t, float tp, uint32_t tk, float mp,
                     Model::PosDraw* out) {
    return static_cast<DenseGpu*>(c)->batch_head_sample(h, vocab, ne, n, x, q, u, t, tp, tk, mp, out);
  };
  o.hc_tap = [](void* c, uint32_t slot, uint32_t nt, uint32_t blo, uint32_t nk, uint32_t r0,
                uint32_t rc, uint32_t ne, uint32_t nh) {
    return static_cast<DenseGpu*>(c)->batch_hc_tap(slot, nt, blo, nk, r0, rc, ne, nh);
  };
  o.head = [](void* c, int32_t h, uint64_t v, uint64_t ne, uint32_t n, const float* x,
              uint32_t* ids) {
    return static_cast<DenseGpu*>(c)->batch_head(h, v, ne, n, x, ids);
  };
  o.draft_proj = [](void* c, int32_t h, int32_t nw, uint64_t r, uint64_t co, uint32_t n,
                    uint32_t row0, float eps) {
    return static_cast<DenseGpu*>(c)->draft_proj(h, nw, r, co, n, row0, eps);
  };
  o.draft_kv = [](void* c, uint32_t l, int32_t kv, int32_t kvn, uint32_t n, uint32_t p0,
                  uint64_t hid, uint32_t w, uint32_t nr, RopeDerived rp, float eps) {
    return static_cast<DenseGpu*>(c)->draft_kv(l, kv, kvn, n, p0, hid, w, nr, rp, eps);
  };
  o.draft_attend = [](void* c, uint32_t l, uint32_t n, uint32_t p0, uint32_t lo, float sc) {
    return static_cast<DenseGpu*>(c)->draft_attend(l, n, p0, lo, sc);
  };
  // The target's block epilogue IS the draft's: same three kernels, different handles.
  o.collapse = [](void* c, int32_t fn, int32_t sv, int32_t bv, int32_t nw, uint32_t ne,
                  uint32_t nh, float he, float re) {
    return static_cast<DenseGpu*>(c)->draft_collapse(fn, sv, bv, nw, ne, nh, he, re);
  };
  o.collapse_pre = [](void* c, int32_t nw, uint32_t ne, uint32_t nh, float he, float re) {
    return static_cast<DenseGpu*>(c)->collapse_pre(nw, ne, nh, he, re);
  };
  o.draft_collapse = [](void* c, int32_t fn, int32_t sv, int32_t bv, int32_t nw, uint32_t ne,
                        uint32_t nh, float he, float re) {
    return static_cast<DenseGpu*>(c)->draft_collapse(fn, sv, bv, nw, ne, nh, he, re);
  };
  o.draft_head = [](void* c, int32_t h, uint64_t v, uint64_t ne, uint32_t n, const float* x,
                    int32_t w1, int32_t w2, uint32_t rk, uint32_t fid, uint32_t* ids) {
    return static_cast<DenseGpu*>(c)->draft_head(h, v, ne, n, x, w1, w2, rk, fid, ids);
  };
  o.kv_commit = [](void* c, uint32_t l, uint32_t b0, uint32_t n, uint32_t p0, uint32_t w,
                   uint32_t nr) {
    return static_cast<DenseGpu*>(c)->batch_kv_commit(l, b0, n, p0, w, nr);
  };
  o.compress = [](void* c, const Model::CompressArgs& a) {
    return static_cast<DenseGpu*>(c)->batch_compress(a);
  };
  o.compress_reset = [](void* c) { return static_cast<DenseGpu*>(c)->compress_reset(); };
  o.indexer_q = [](void* c, int32_t h, uint64_t r, uint64_t cols, uint32_t nh, uint32_t dim) {
    return static_cast<DenseGpu*>(c)->batch_indexer_q(h, r, cols, nh, dim);
  };
  o.indexer = [](void* c, uint32_t l, uint32_t n, const uint32_t* nc, uint32_t nk, uint32_t nh,
                 uint32_t dim, uint32_t topk, uint32_t pos0, uint32_t n_rot, RopeDerived rope) {
    return static_cast<DenseGpu*>(c)->batch_indexer(l, n, nc, nk, nh, dim, topk, pos0, n_rot, rope);
  };
  o.attend = [](void* c, uint32_t l, uint32_t b0, uint32_t n, uint32_t p0, const uint8_t* a,
                uint32_t ms, const uint32_t* nc, float s) {
    return static_cast<DenseGpu*>(c)->batch_attend(l, b0, n, p0, a, ms, nc, s);
  };
  o.norm_fetch = [](void* c, float* n) { return static_cast<DenseGpu*>(c)->batch_norm_fetch(n); };
  o.router_topk = [](void* c, int32_t h, uint32_t ne, uint64_t nd, uint32_t k, const float* bias,
                     const uint8_t* res, int np, float rs, uint32_t* sel, float* wt, uint32_t* n) {
    return static_cast<DenseGpu*>(c)->batch_router_topk(h, ne, nd, k, bias, res, np, rs, sel, wt, n);
  };
  o.attn_out = [](void* c, int32_t wa, int32_t wb, uint32_t nh, uint32_t w, uint32_t nr,
                  RopeDerived rp, uint32_t ng, uint64_t rk, uint64_t hid) {
    return static_cast<DenseGpu*>(c)->batch_attn_out(wa, wb, nh, w, nr, rp, ng, rk, hid);
  };
  o.hc_post = [](void* c, bool hc, uint32_t ne, uint32_t nh) {
    return static_cast<DenseGpu*>(c)->batch_hc_post(hc, ne, nh);
  };
  o.router = [](void* c, int32_t h, uint32_t nx, uint64_t ne, float* lg) {
    return static_cast<DenseGpu*>(c)->batch_router(h, nx, ne, lg);
  };
  o.shexp = [](void* c, int32_t g, int32_t u, int32_t d, uint64_t hid, uint64_t it, float lim) {
    return static_cast<DenseGpu*>(c)->batch_shexp(g, u, d, hid, it, lim);
  };
  o.block_add = [](void* c, const float* s, uint32_t ne) {
    return static_cast<DenseGpu*>(c)->batch_block_add(s, ne);
  };
  o.block_reduce = [](void* c, uint32_t ne) {
    return static_cast<DenseGpu*>(c)->batch_block_reduce(ne);
  };
  o.hc_read = [](void* c, uint32_t b, float* d, uint32_t ne, uint32_t nh) {
    return static_cast<DenseGpu*>(c)->batch_hc_read(b, d, ne, nh);
  };
  o.router_hash = [](void* c, int32_t h, uint32_t ne, uint64_t nd, uint32_t k, const uint32_t* sel,
                     const uint32_t* nsel, int np, float rs, float* wt) {
    return static_cast<DenseGpu*>(c)->batch_router_hash(h, ne, nd, k, sel, nsel, np, rs, wt);
  };
  return o;
}

std::vector<VramSample> vram_sample() {
  std::vector<VramSample> out;
  int cur = 0;
  if (hipGetDevice(&cur) != hipSuccess) return out;
  for (int d : gfx12_devices()) {
    size_t freeb = 0, totb = 0;
    if (hipSetDevice(d) != hipSuccess || hipMemGetInfo(&freeb, &totb) != hipSuccess) continue;
    out.push_back({(double)(totb - freeb) / 1073741824.0, (double)totb / 1073741824.0});
  }
  // Restoring is not tidiness: the caller is mid-load with a current device, and every allocation
  // after this would land on whichever card the loop finished on.
  (void)hipSetDevice(cur);
  return out;
}

std::vector<GpuDeviceDesc> gpu_devices() {
  std::vector<GpuDeviceDesc> out;
  for (int d : gfx12_devices()) {
    hipDeviceProp_t pr{};
    if (hipGetDeviceProperties(&pr, d) != hipSuccess) continue;
    GpuDeviceDesc g;
    g.name = pr.name;
    // gcnArchName carries feature suffixes (":sramecc+:xnack-"); only the base name is the arch.
    g.arch = std::string(pr.gcnArchName).substr(0, std::string(pr.gcnArchName).find(':'));
    char bus[32] = {};
    if (hipDeviceGetPCIBusId(bus, (int)sizeof bus, d) == hipSuccess) g.bus_id = bus;
    g.cus = (int)device_cus(d);
    g.vram_total_gib = (double)pr.totalGlobalMem / 1073741824.0;
    out.push_back(std::move(g));
  }
  return out;
}

} // namespace aff

// ---- amdnas-fixes weight-verify -------------------------------------------------------------------------------------
namespace {
uint64_t aff_wv_hash(const uint8_t* p, size_t n, uint64_t h) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    uint64_t w; std::memcpy(&w, p + i, 8);
    h ^= w; h *= 0x9E3779B97F4A7C15ull; h ^= h >> 29;
  }
  for (; i < n; ++i) { h ^= p[i]; h *= 0x100000001B3ull; }
  return h;
}
uint64_t aff_wv_digest_dev(const void* dptr, uint64_t bytes, std::vector<uint8_t>& stage) {
  uint64_t h = 0xCBF29CE484222325ull;
  const size_t chunk = 64u << 20;
  if (stage.size() < chunk) stage.resize(chunk);
  for (uint64_t off = 0; off < bytes; off += chunk) {
    const size_t n = (size_t)std::min<uint64_t>(chunk, bytes - off);
    if (hipMemcpy(stage.data(), (const uint8_t*)dptr + off, n, hipMemcpyDeviceToHost) != hipSuccess) return 0xDEADull;
    h = aff_wv_hash(stage.data(), n, h);
  }
  return h;
}
}  // namespace

uint64_t aff::DenseGpu::digest_weights(std::vector<std::pair<std::string, uint64_t>>* out) const {
  if (!impl_) return 0;
  int prev = 0; (void)hipGetDevice(&prev);
  static std::vector<uint8_t> stage;
  uint64_t total = 0;
  for (size_t i = 0; i < ent_.size(); ++i) {
    const Entry& e = ent_[i];
    for (size_t d = 0; d < impl_->dev.size() && d < kMaxRanks; ++d) {
      const Shard& sh = e.sh[d];
      if (!sh.data || !sh.data_bytes) continue;
      (void)hipSetDevice(impl_->dev[d].id);
      out->emplace_back("ent" + std::to_string(i) + " rank" + std::to_string(d) + " data",
                        aff_wv_digest_dev(sh.data, sh.data_bytes, stage));
      total += sh.data_bytes;
      if (sh.scales && sh.scales_bytes) {
        out->emplace_back("ent" + std::to_string(i) + " rank" + std::to_string(d) + " scales",
                          aff_wv_digest_dev(sh.scales, sh.scales_bytes, stage));
        total += sh.scales_bytes;
      }
    }
  }
  (void)hipSetDevice(prev);
  return total;
}
