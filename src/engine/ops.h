// DeepSeek-V4-Flash operators beyond attention.
//
// Details that are easy to get wrong, from antirez/ds4's working implementation rather than from
// the config alone:
//
//  * The compressor is a per-dimension SOFTMAX POOL with a learned absolute-position embedding
//    added to the score — not mean-pooling. At ratio 4 it keeps an overlapping dual-lane state,
//    so each compressed entry draws from 2m raw entries.
//  * The lightning indexer's weight projection reads the ATTENTION-NORMED activation, the same
//    tensor the compressor sees — not the pre-norm hyper-connection residual.
//  * The lightning indexer has NO softmax: I(t,s) = sum_h w_h * relu(q_h . k_s).
//  * Sinkhorn iteration 0 differs from 1..19, and eps appears in three distinct places.
//  * The router bias affects SELECTION but never the WEIGHTS, and the result is scaled by 1.5.
//  * The SwiGLU clamp is ASYMMETRIC — the gate is upper-bounded only.
//  * Hash routing (layers 0-2) has no hash function: it is an I32[6, vocab] lookup keyed by
//    token id, so those experts are known at tokenise time.

#pragma once

#include "format/aff_format.h"   // AffQuant, for the dense weight tag

#include <cstdint>
#include <vector>

namespace aff {

// The widest batch the speculative path ever presents: wmma::kN, the narrow W8A8 instantiation.
// Here rather than in gpu/batch_kernels.h because the engine core sizes its own runs against it and
// must not include a HIP header to do so — and two copies of this number disagreeing is not a slow
// path, it is a buffer overrun. gpu/batch_kernels.h's kNarrowTok is this.
constexpr uint32_t kSpecTok = 16;

// ---- elementwise / norms ----------------------------------------------------------------------

void  rms_norm(const float* x, const float* weight, uint64_t n, float eps, float* out) noexcept;
float sqrtsoftplus(float x) noexcept;          // sqrt(log(1+e^x)), the router's scoring fn
void  softmax_inplace(float* x, uint64_t n) noexcept;

// SwiGLU with DeepSeek's two clamps, which differ in shape: the GATE is bounded above only, while
// `up` is bounded on BOTH sides at +/- `limit`.
void swiglu(const float* gate, const float* up, uint64_t n, float limit, float* out) noexcept;

// ---- router -----------------------------------------------------------------------------------

// The widest routed selection RouterOut carries. A config asking for more would overrun the two
// arrays below, so the routers refuse rather than truncate — a truncated selection is a different
// model that still produces text.
constexpr uint32_t kMaxTopK = 8;

struct RouterOut {
  uint32_t expert[kMaxTopK];
  float    weight[kMaxTopK];
  uint32_t n = 0;
};

// noaux_tc routing. `bias` (aux-loss-free correction) shifts SELECTION only; the returned weights
// come from the unbiased scores. Weights are renormalised over the top-k then scaled.
void route_topk(const float* logits, const float* bias, uint32_t n_expert, uint32_t top_k,
                bool norm_topk_prob, float routed_scaling, RouterOut* out) noexcept;

// Layers 0-2: expert ids are a pure function of the token id. `table` is [n_expert_used, vocab]
// row-major as shipped (`ffn_gate_tid2eid`). Known at tokenise time -> perfect prefetch.
void route_hash(const int32_t* table, uint32_t vocab, uint32_t top_k, uint32_t token_id,
                RouterOut* out) noexcept;

// ---- compressor (CSA / HCA) ---------------------------------------------------------------------

struct CompressorConfig {
  uint32_t width = 576;     // latent width being pooled
  uint32_t ratio = 4;       // m: raw entries per compressed entry (4 = CSA, 128 = HCA)
  bool     overlap = true;  // ratio-4 keeps a dual-lane overlapping window (draws from 2m)
};

// Softmax-pool `n_raw` latents into ceil(n_raw/ratio) compressed rows.
// `score_w` : [width] learned projection producing the pooling logit
// `ape`     : [ratio*2] learned absolute-position embedding ADDED to the logit (may be null)
void compress_pool(const CompressorConfig& c, const float* raw, uint64_t n_raw,
                   const float* score_w, const float* ape,
                   float* out, uint64_t* n_out) noexcept;

// ---- lightning indexer + top-k --------------------------------------------------------------------

struct IndexerConfig {
  uint32_t n_head = 64;
  uint32_t head_dim = 128;
  uint32_t topk = 512;
};

// I(t,s) = sum_h w_h * relu(q_h . k_s)   — deliberately NO softmax.
void indexer_scores(const IndexerConfig& c, const float* q, const float* w,
                    const float* keys, uint64_t n_keys, float* scores) noexcept;

// Exact top-k by chunk + tree merge; scales to hundreds of thousands of rows.
void top_k_select(const float* scores, uint64_t n, uint32_t k, uint64_t* out_idx,
                  uint64_t* n_out) noexcept;

// ---- hyper-connections (mHC) ------------------------------------------------------------------

struct HcConfig {
  uint32_t mult = 4;          // hc_mult: number of residual lanes
  uint32_t sinkhorn_iters = 20;
  float    eps = 1e-6f;
};

// Sinkhorn normalisation of an [n x n] matrix. Iteration 0 is not the same as 1..N-1: the first
// pass normalises rows only, before the alternating row/column sweeps begin.
void sinkhorn(float* m, uint32_t n, uint32_t iters, float eps) noexcept;

// Mixes `mult` residual lanes using a Sinkhorn-normalised mixing matrix derived from `fn`/`scale`.
void hc_mix(const HcConfig& c, const float* lanes, uint32_t width,
            const float* fn, const float* scale, float* out) noexcept;


// ==============================================================================================
// Exact DeepSeek-V4-Flash primitives.
//
// The generic ops above were written from the tech report. Everything below was read off
// antirez/ds4's working CPU path and matches it operation for operation, including the parts the
// report does not mention: the per-head RMS norm on Q with no weight, the INVERSE RoPE applied to
// the attention output before the grouped projection (V carries the rotated tail, so it has to be
// unrotated), and the two QAT round-trips the model was trained with.
// ==============================================================================================

// ---- dense matvec over bf16 weights (row-major, out[r] = sum_c W[r][c] * x[c]) ----------------
void matvec_bf16(const uint16_t* W, const float* x, uint64_t rows, uint64_t cols,
                 float* out) noexcept;


// A/B selection for the dense matvec: 1 = a single accumulator chain, 2 = four (default).
void matvec_bf16_variant(int v);
int  matvec_bf16_variant();
void matvec_bf16_grouped_mt(const uint16_t* W, const float* x, uint32_t n_groups,
                            uint64_t group_dim, uint64_t rank, float* out);

// ---- dense matvec over whatever the container actually stores ----------------------------------
//
// A dense weight is bf16 OR the checkpoint's native FP8-E4M3 with a UE8M0 scale per 128x128 tile.
// The engine used to reinterpret every dense tensor as bf16 unconditionally, so the first container
// that stored FP8 verbatim produced NaN logits with nothing anywhere reporting a type error. The
// quant tag travels WITH the pointer now, and matvec_dense_* dispatches on it, so the two can no
// longer disagree silently.
struct DenseW {
  const void*    data   = nullptr;
  const uint8_t* scales = nullptr;   // FP8 only; null for bf16
  uint32_t       quant  = 0;         // AffQuant; AFF_BF16 or AFF_FP8_E4M3
  // Handle into the GPU dense executor, or -1 when this weight is not device-resident. Dense is
  // several times the expert traffic and reads every byte every token, so the intent is that this is
  // ALWAYS >= 0 in a configured engine and the CPU paths below are the fallback, not the design.
  int32_t        gpu    = -1;
  explicit operator bool() const noexcept { return data != nullptr; }
};

void matvec_dense(const DenseW& w, const float* x, uint64_t rows, uint64_t cols, float* out) noexcept;
void matvec_dense_mt(const DenseW& w, const float* x, uint64_t rows, uint64_t cols, float* out);
void matvec_dense_grouped_mt(const DenseW& w, const float* x, uint32_t n_groups,
                             uint64_t group_dim, uint64_t rank, float* out);

// ---- norms --------------------------------------------------------------------------------
void rms_norm_noweight(const float* x, uint64_t n, float eps, float* out) noexcept;
// Per-head RMS norm with no learned weight, applied to Q after wq_b. Absent from the config;
// omitting it changes every attention score.
void head_rms_norm_inplace(float* x, uint32_t n_head, uint32_t head_dim, float eps) noexcept;

// ---- RoPE ------------------------------------------------------------------------------------
// YaRN parameters resolved per layer: compressed layers use theta 160000 with interpolation,
// uncompressed layers use theta 10000 with none.
struct RopeParams {
  float    freq_base   = 10000.0f;
  float    freq_scale  = 1.0f;   // 1/factor on compressed layers, 1 otherwise
  float    ext_factor  = 0.0f;   // 1 enables the YaRN frequency blend, 0 disables it
  float    beta_fast   = 32.0f;
  float    beta_slow   = 1.0f;
  uint32_t orig_ctx    = 0;
};

// The scalars rope_tail derives from RopeParams before its loops. Hoisted so a device kernel can
// take them as arguments instead of reimplementing the YaRN corner cases.
struct RopeDerived { float freq_scale, theta_scale, ext_factor, lo, hi; };
RopeDerived rope_derive(const RopeParams& p, uint32_t n_rot) noexcept;


// Rotates only the trailing `n_rot` of each head. `inverse` flips the sine sign.
void rope_tail(float* x, uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
               uint32_t pos, const RopeParams& p, bool inverse) noexcept;

// ---- QAT round-trips the model was trained through -------------------------------------------
// The non-RoPE part of every cached KV row goes through an E4M3 round trip with a per-64
// power-of-two scale. Skipping it makes cached keys differ from what the model saw in training.
void fp8_kv_qat(float* x, uint32_t head_dim, uint32_t n_rot) noexcept;
// The indexer's own round trip — 128-wide Hadamard then FP4-E2M1 per 32 — has no host version:
// it only ever ran inside compressor_step, and both moved to the device. See src/gpu/compressor_gpu.

// ---- hyper-connections -----------------------------------------------------------------------
// Control values decoded from the HC state for one sublayer.
struct HcControl {
  float pre[8]   = {0};   // reduce weights, sigmoid + eps
  float post[8]  = {0};   // inject gates, 2*sigmoid
  float comb[64] = {0};   // [dst + src*n_hc], Sinkhorn-normalised
};

// rms_norm(flat HC state) -> fn -> {pre, post, comb}. `fn` is [3*n_hc + n_hc^2, n_hc*n_embd] bf16.
void hc_control(const uint16_t* fn, const float* scale, const float* base,
                const float* residual_hc, uint32_t n_embd, uint32_t n_hc,
                uint32_t iters, float hc_eps, float rms_eps, HcControl* out) noexcept;

// out[d] = sum_h residual_hc[h][d] * pre[h]  — the sublayer's plain input.
void hc_reduce(const float* residual_hc, const float* pre, uint32_t n_embd, uint32_t n_hc,
               float* out) noexcept;

// out_hc[dst][d] = block_out[d]*post[dst] + sum_src comb[dst + src*n_hc] * residual_hc[src][d].
void hc_expand(float* out_hc, const float* block_out, const float* residual_hc,
               const float* post, const float* comb, uint32_t n_embd, uint32_t n_hc) noexcept;

} // namespace aff
