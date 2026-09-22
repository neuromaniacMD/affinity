// Model runtime: weight binding and the decode forward pass.
//
// The layer schedule comes from `compress_ratios` in the checkpoint config (44 entries = 43
// layers + the MTP block): 0 on layers 0-1 (uncompressed), then alternating 4 (CSA, indexed) and
// 128 (HCA, dense over a short compressed cache). Hash routing applies to layers 0-2.
//
// Ordering below is validated against antirez/ds4's CPU decode path operation for operation. The
// steps that are NOT derivable from the config, and that silently corrupt output if dropped:
//
//   * A per-head RMS norm with no learned weight is applied to Q after wq_b.
//   * The cached KV row goes through an E4M3 QAT round trip on its unrotated part, and is then
//     stored fp16-rounded.
//   * The attention output is INVERSE-RoPE'd before the grouped output projection, because V is
//     the full latent including the rotated tail.
//   * Hyper-connections wrap every sublayer: the residual is four lanes wide, and the reduce /
//     expand weights are recomputed per token from the lanes themselves.

#include "model.h"
#include "threadpool.h"
#include "engine/instrument.h"
#include "gpu/attention_gpu.h"          // kAttnBatch: the prefill attention sub-batch
#include "quant/source_dtypes.h"

#include "ui/log.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <utility>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace aff {

namespace {

// A missing tensor yields null rather than failing the load: a container truncated with --layers
// is legitimate, and every consumer below is null-tolerant.
struct Binder {
  const AffReader& r;
  const Model::DenseOps* dops = nullptr;
  const float* f32(const std::string& n) const {
    const AffTensorEntry* e = r.find_tensor(n);
    return e ? reinterpret_cast<const float*>(r.tensor_data(*e)) : nullptr;
  }
  const uint16_t* bf16(const std::string& n) const {
    const AffTensorEntry* e = r.find_tensor(n);
    if (!e) return nullptr;
    // Reinterpreting a non-bf16 tensor as bf16 is exactly how a container full of FP8 produced NaN
    // logits with no error anywhere. Refuse instead.
    if (e->desc.quant != AFF_BF16) {
      aff::ui::fatal("fatal: tensor %s is %s, not bf16 — engine and container disagree\n",
                   n.c_str(), quant_name((AffQuant)e->desc.quant));
      std::abort();
    }
    return reinterpret_cast<const uint16_t*>(r.tensor_data(*e));
  }
  // A dense weight in whatever the container stores. The quant tag travels with the pointer so the
  // kernel dispatches on what is actually there rather than on what the engine assumed.
  DenseW dense(const std::string& n) const {
    DenseW w;
    const AffTensorEntry* e = r.find_tensor(n);
    if (!e) return w;
    w.data = r.tensor_data(*e);
    w.quant = e->desc.quant;
    if (w.quant == AFF_FP8_E4M3) w.scales = r.tensor_scales(*e);
    else if (w.quant != AFF_BF16) {
      aff::ui::fatal("fatal: tensor %s has quant %s, which no dense kernel decodes\n",
                   n.c_str(), quant_name((AffQuant)e->desc.quant));
      std::abort();
    }
    // Hand it to the GPU as we bind it. A weight that fails to register keeps gpu = -1 and falls
    // back to the CPU kernel, so running out of VRAM degrades rather than fails.
    if (dops && dops->reg)
      w.gpu = dops->reg(dops->ctx, n.c_str(), w.data, w.scales, w.quant,
                        e->desc.rows, e->desc.cols);
    return w;
  }
};

inline float sigmoid(float z) noexcept { return 1.0f / (1.0f + std::exp(-z)); }

// The compressor rotates the trailing 64 of the row it emits whatever the model's rope_dim is, and
// the derived YaRN corner frequencies follow from that same 64. Shared by the decode and prefill
// call sites so the two cannot drift.
constexpr uint32_t kCompressorRot = 64;

// AFF_PROFILE=1. Read once; the branch is perfectly predicted either way.
inline bool profiling() noexcept {
  static const bool on = getenv("AFF_PROFILE") != nullptr;
  return on;
}

// ---- the hybrid census (AFF_PROFILE) ----------------------------------------------------------
//
// What a hybrid dispatch has to be sized for is NOT "47% of experts are resident". Two different
// numbers matter and they are not the same one:
//
//   the H-index      per token, how many of its six are in VRAM (H6 down to H0). A property of
//                    the routing, not of the engine.
//   distinct missing per LAYER, over the whole batch. A batched dispatch reads an expert's weights
//                    once however many of the batch chose it, so this — not the slot count — is
//                    what a stream has to move, and it is what the overlap window has to hide.
//
// A speculative block verifies six positions at once, so the two differ by however much the six
// tokens agree with each other. Guessing that factor is exactly the mistake this exists to avoid:
// at one extreme six tokens share one expert set and a layer misses at most six distinct, at the
// other they share nothing and it misses thirty-six.
struct HybridCensus {
  uint64_t calls = 0, tokens = 0;
  uint64_t hits[16] = {};          // per token: how many of its k were resident
  uint64_t slots = 0, slots_miss = 0;
  uint64_t distinct = 0, distinct_miss = 0;
  uint64_t worst_miss = 0;
  ~HybridCensus() {
    if (!calls) return;
    aff::ui::out("\nhybrid census: %llu layer-dispatches, %.2f tokens each\n",
                (unsigned long long)calls, (double)tokens / (double)calls);
    aff::ui::out("  slots %.1f a layer, %.1f of them missing (%.1f%%)\n",
                (double)slots / calls, (double)slots_miss / calls,
                100.0 * slots_miss / std::max<uint64_t>(1, slots));
    aff::ui::out("  DISTINCT experts %.1f a layer, %.1f of them missing (%.1f%%), worst %llu\n",
                (double)distinct / calls, (double)distinct_miss / calls,
                100.0 * distinct_miss / std::max<uint64_t>(1, distinct),
                (unsigned long long)worst_miss);
    aff::ui::out("  H-index (per token, resident of k):");
    for (uint32_t h = 0; h < 8; ++h)
      if (hits[h]) aff::ui::out("  H%u %.1f%%", h, 100.0 * hits[h] / std::max<uint64_t>(1, tokens));
    aff::ui::out("\n");
  }
};
inline HybridCensus& hybrid_census() { static HybridCensus c; return c; }

// ---- everything a DSpark stage and an ordinary layer bind the same way -------------------------
//
// Which is all of it except the expert pool, the compressor, the indexer and the hash table. ONE
// body: a second copy is a second place to forget when a tensor is added, and a draft stage
// silently missing an hc projection does not fail — it produces plausible drafts that are never
// accepted, which reads as "speculation does not pay on this model".
//
// Returns the name of the first required tensor that is absent, or null.
const char* bind_layer_common(const Binder& b, const ModelConfig& cfg, const Model::DenseOps& dops,
                              const AffReader& r, const std::string& p, LayerWeights& w,
                              std::vector<std::vector<uint16_t>>* router_bf16) {
  w.attn_norm  = b.f32(p + "attn_norm");
  w.ffn_norm   = b.f32(p + "ffn_norm");
  w.attn_sinks = b.f32(p + "attn_sinks");
  // bf16, not f32. The gate is 256x4096 = 4 MiB a layer in f32 and decode reads all of it every
  // token on rank 0 alone; the output is a softmax then a top-6, so what has to survive is the
  // ORDER of the top few logits, and bf16 weights against an f32 activation and accumulator keep
  // far more than the router's own logit gaps need.
  // A container written before the gate went bf16 stores it as f32, and the binder aborts on a
  // dtype mismatch rather than reinterpreting one — correctly. Convert instead of refusing: the
  // rounding here is bit-for-bit what the quantiser now does at write time, so an old container
  // and a new one differ ONLY in the expert format, which is what an A/B between them is for.
  if (const AffTensorEntry* rg = r.find_tensor(p + "ffn_gate_inp");
      rg && rg->desc.quant == AFF_F32) {
    const float* src = reinterpret_cast<const float*>(r.tensor_data(*rg));
    const size_t n = (size_t)cfg.n_expert * cfg.n_embd;
    router_bf16->emplace_back(n);
    uint16_t* dst = router_bf16->back().data();
    for (size_t i = 0; i < n; ++i) dst[i] = float_to_bf16(src[i]);
    w.router_w = dst;
  } else {
    w.router_w = b.bf16(p + "ffn_gate_inp");
  }
  w.router_b   = b.f32(p + "ffn_gate_bias");
  w.wq_a       = b.dense(p + "attn_q_a");
  w.wq_b       = b.dense(p + "attn_q_b");
  w.wkv        = b.dense(p + "attn_kv");
  w.wo_a       = b.dense(p + "attn_out_a");
  w.wo_b       = b.dense(p + "attn_out_b");
  w.q_norm     = b.f32(p + "attn_q_norm");
  w.kv_norm    = b.f32(p + "attn_kv_norm");
  w.shexp_gate = b.dense(p + "shexp_gate");
  w.shexp_up   = b.dense(p + "shexp_up");
  w.shexp_down = b.dense(p + "shexp_down");
  w.hc_attn_fn    = b.dense(p + "hc_attn_fn");
  w.hc_attn_base  = b.f32(p + "hc_attn_base");
  w.hc_attn_scale = b.f32(p + "hc_attn_scale");
  w.hc_ffn_fn     = b.dense(p + "hc_ffn_fn");
  w.hc_ffn_base   = b.f32(p + "hc_ffn_base");
  w.hc_ffn_scale  = b.f32(p + "hc_ffn_scale");

  // Missing weights must fail the load, not degrade the output. Every null pointer here has a
  // "plausible" fallback in the forward pass (skip the projection, touch only HC lane 0), and
  // those fallbacks produce believable-looking logits from an incomplete container — the same
  // failure mode that let a container with zero written experts look healthy.
  const std::pair<const void*, const char*> required[] = {
    {w.attn_norm, "attn_norm"},   {w.ffn_norm, "ffn_norm"},   {w.attn_sinks, "attn_sinks"},
    {w.wq_a.data, "attn_q_a"},    {w.wq_b.data, "attn_q_b"},  {w.wkv.data, "attn_kv"},
    {w.wo_a.data, "attn_out_a"},  {w.wo_b.data, "attn_out_b"},
    {w.q_norm, "attn_q_norm"},    {w.kv_norm, "attn_kv_norm"},
    {w.router_w, "ffn_gate_inp"},
    {w.shexp_gate.data, "shexp_gate"}, {w.shexp_up.data, "shexp_up"},
    {w.shexp_down.data, "shexp_down"},
    {w.hc_attn_fn.data, "hc_attn_fn"}, {w.hc_attn_base, "hc_attn_base"},
    {w.hc_attn_scale, "hc_attn_scale"},
    {w.hc_ffn_fn.data, "hc_ffn_fn"},   {w.hc_ffn_base, "hc_ffn_base"},
    {w.hc_ffn_scale, "hc_ffn_scale"},
  };
  for (const auto& [ptr, name] : required) if (!ptr) return name;

  if (dops.reg_vec) {
    const uint32_t HCN = cfg.hc_mult, NC = 2 * HCN + HCN * HCN;
    w.q_norm_gpu   = dops.reg_vec(dops.ctx, w.q_norm, cfg.q_lora_rank);
    w.kv_norm_gpu  = dops.reg_vec(dops.ctx, w.kv_norm, cfg.head_dim);
    w.attn_norm_gpu = dops.reg_vec(dops.ctx, w.attn_norm, cfg.n_embd);
    w.ffn_norm_gpu  = dops.reg_vec(dops.ctx, w.ffn_norm, cfg.n_embd);
    w.hc_attn_scale_gpu = dops.reg_vec(dops.ctx, w.hc_attn_scale, 3);
    w.hc_attn_base_gpu  = dops.reg_vec(dops.ctx, w.hc_attn_base, NC);
    w.hc_ffn_scale_gpu  = dops.reg_vec(dops.ctx, w.hc_ffn_scale, 3);
    w.hc_ffn_base_gpu   = dops.reg_vec(dops.ctx, w.hc_ffn_base, NC);
  }
  // The router gate goes to VRAM if there is room. A -1 back means there was not, and the host
  // matvec is still correct — this is the one dense tensor whose device copy is optional. Hash
  // layers too: they do not RANK by the logits, but they still weight by them, and doing that on
  // the host was the largest stall in the prefill.
  if (dops.reg_router)
    w.router_gpu = dops.reg_router(dops.ctx, w.router_w, w.router_b, cfg.n_expert, cfg.n_embd);
  return nullptr;
}

// The lightning indexer's scoring loop: score[c] = sum_h w[h] * relu(q[h] . K[c]), over `n`
// compressed rows of `head_dim` floats each. See the call site for why this is worth hand-writing.
//
// Vectorised four heads at a time. Four independent accumulator chains hide the FMA latency, and
// the four lane-sums are folded together with one 4x4 transpose (`unpacklo`/`movelh`) instead of
// four separate horizontal reduces — at head_dim 128 each head is only eight FMAs, so a horizontal
// sum per head costs about as much as the arithmetic it is reducing, and that is most of what a
// naive vectorisation would leave on the table.
//
// Threaded by ROW: score[c] depends only on row c, so workers never share an output and the sum
// order inside a row is identical whatever the thread count. Same numbers on 1 thread and on 32.
//
// The SCORES are not bit-identical to the scalar loop and cannot be: this reassociates the sum and
// `_mm512_fmadd_ps` is a fused multiply-add, which the reference deliberately is not
// (-ffp-contract=off). What comes out of the indexer is not a score, it is the top-k MASK over
// them, so a perturbation this far below the model's own quantisation error changes nothing unless
// it moves a row across the selection boundary — and whether it does depends on how contested the
// selection is. A narrow selection is bit-identical end to end; a wide, contested one does swap
// rows. So the check that matters is end-to-end continuation agreement, not the scores.
inline void indexer_scores(const float* __restrict q, const float* __restrict w,
                           const float* __restrict K, uint32_t n_head, uint32_t head_dim,
                           uint32_t n, float* __restrict out) {
#if defined(__AVX512F__)
  if ((head_dim % 16) == 0 && (n_head % 4) == 0) {
    // A row is n_head*head_dim/16 = 512 FMAs here, so a few dozen rows already outweigh what a
    // parallel region costs to open and close; below that it runs serially. The caller only reaches
    // this once n_comp passes index_topk, so a short-context token pays nothing here.
    parallel_for(n, 64, [&](uint64_t lo, uint64_t hi) {
      const uint32_t nv = head_dim / 16;
      for (uint64_t c = lo; c < hi; ++c) {
        const float* __restrict k = K + (size_t)c * head_dim;
        __m128 acc = _mm_setzero_ps();
        for (uint32_t h = 0; h < n_head; h += 4) {
          const float* q0 = q + (size_t)(h + 0) * head_dim;
          const float* q1 = q + (size_t)(h + 1) * head_dim;
          const float* q2 = q + (size_t)(h + 2) * head_dim;
          const float* q3 = q + (size_t)(h + 3) * head_dim;
          __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
          __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
          for (uint32_t v = 0; v < nv; ++v) {
            const __m512 kv = _mm512_loadu_ps(k + v * 16);
            a0 = _mm512_fmadd_ps(_mm512_loadu_ps(q0 + v * 16), kv, a0);
            a1 = _mm512_fmadd_ps(_mm512_loadu_ps(q1 + v * 16), kv, a1);
            a2 = _mm512_fmadd_ps(_mm512_loadu_ps(q2 + v * 16), kv, a2);
            a3 = _mm512_fmadd_ps(_mm512_loadu_ps(q3 + v * 16), kv, a3);
          }
          // Four lane-sums into lanes 0..3 of one vector: pairwise adds down to 128-bit lanes,
          // then a 4x4 transpose of what is left.
          const __m128 s0 = _mm_add_ps(_mm_add_ps(_mm512_extractf32x4_ps(a0, 0),
                                                  _mm512_extractf32x4_ps(a0, 1)),
                                       _mm_add_ps(_mm512_extractf32x4_ps(a0, 2),
                                                  _mm512_extractf32x4_ps(a0, 3)));
          const __m128 s1 = _mm_add_ps(_mm_add_ps(_mm512_extractf32x4_ps(a1, 0),
                                                  _mm512_extractf32x4_ps(a1, 1)),
                                       _mm_add_ps(_mm512_extractf32x4_ps(a1, 2),
                                                  _mm512_extractf32x4_ps(a1, 3)));
          const __m128 s2 = _mm_add_ps(_mm_add_ps(_mm512_extractf32x4_ps(a2, 0),
                                                  _mm512_extractf32x4_ps(a2, 1)),
                                       _mm_add_ps(_mm512_extractf32x4_ps(a2, 2),
                                                  _mm512_extractf32x4_ps(a2, 3)));
          const __m128 s3 = _mm_add_ps(_mm_add_ps(_mm512_extractf32x4_ps(a3, 0),
                                                  _mm512_extractf32x4_ps(a3, 1)),
                                       _mm_add_ps(_mm512_extractf32x4_ps(a3, 2),
                                                  _mm512_extractf32x4_ps(a3, 3)));
          __m128 t0 = _mm_unpacklo_ps(s0, s1), t1 = _mm_unpackhi_ps(s0, s1);
          __m128 t2 = _mm_unpacklo_ps(s2, s3), t3 = _mm_unpackhi_ps(s2, s3);
          const __m128 d = _mm_add_ps(_mm_add_ps(_mm_movelh_ps(t0, t2), _mm_movehl_ps(t2, t0)),
                                      _mm_add_ps(_mm_movelh_ps(t1, t3), _mm_movehl_ps(t3, t1)));
          // relu, then weight. max() rather than a branch: the scalar version took 64 data-dependent
          // and thoroughly unpredictable branches per row, which was its own cost on top of the
          // arithmetic. Adding an exact zero for a rejected head is the same sum.
          acc = _mm_add_ps(acc, _mm_mul_ps(_mm_max_ps(d, _mm_setzero_ps()), _mm_loadu_ps(w + h)));
        }
        alignas(16) float lanes[4];
        _mm_store_ps(lanes, acc);
        out[c] = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
      }
    });
    return;
  }
#endif
  for (uint32_t c = 0; c < n; ++c) {
    const float* k = K + (size_t)c * head_dim;
    float acc = 0.0f;
    for (uint32_t h = 0; h < n_head; ++h) {
      const float* qh = q + (size_t)h * head_dim;
      float d = 0.0f;
      for (uint32_t i = 0; i < head_dim; ++i) d += qh[i] * k[i];
      if (d > 0.0f) acc += d * w[h];
    }
    out[c] = acc;
  }
}

} // namespace

// ------------------------------------------------------------------------------------------
// Load
// ------------------------------------------------------------------------------------------

bool Model::load(const std::string& aff_path, std::string* err) {
  if (!aff_.open(aff_path, err)) return false;
  if (!parse_model_config(std::string(aff_.meta()), &cfg_, err)) return false;

  // The codebook is fitted data, not a constant: it must come out of the container or every
  // decoded weight is wrong.
  {
    const AffTensorEntry* e = aff_.find_tensor("__codebook");
    if (!e) { if (err) *err = "container has no __codebook tensor"; return false; }
    const float* v = reinterpret_cast<const float*>(aff_.tensor_data(*e));
    const uint64_t n = e->desc.rows * e->desc.cols;
    // How to READ this tensor depends on the expert format, and every layer shares one by
    // construction, so layer 0's gate answers for the file. Pair codes store 4 x 32 x 2 floats.
    // 4-wide codes store 4 x 1024 packed E4M3 quads, one dword an entry, because that is exactly
    // what the GPU LUT holds and building 4096 entries from floats in every block is not
    // affordable — unpacking here gives the CPU path the same values the GPU decodes, bit for bit.
    const uint32_t q = aff_.layer_count() ? (uint32_t)aff_.layer(0).gate.quant : 0u;
    const QuantSpec spec = expert_quant_spec(q);
    if (spec.pair == 4) {
      const uint64_t want = (uint64_t)spec.variants() * spec.lut_size();
      if (n != want) {
        if (err) *err = "__codebook is " + std::to_string(n) + " dwords, expected " +
                        std::to_string(want) + " for a 4-wide codebook";
        return false;
      }
      unpack_codebook_e4m3_quads(reinterpret_cast<const uint32_t*>(v), spec.variants(),
                                 spec.lut_size(), &cb_);
    } else {
      cb_.variants = 4; cb_.k = 32; cb_.pair = 2;
      cb_.v.assign(v, v + n);
    }
  }

  Binder b{aff_, &dops_};
  embed_         = b.bf16("token_embd");
  head_          = b.dense("output");
  out_norm_      = b.f32("output_norm");
  hc_head_fn_    = b.dense("hc_head_fn");
  hc_head_base_  = b.f32("hc_head_base");
  hc_head_scale_ = b.f32("hc_head_scale");
  {
    // V4.1 has no head mixer. `Transformer.forward` collapses the lanes with the mix the LAST
    // layer's FFN produced (`h = layer.hc_pre(h, pre_mix)`) and goes straight to the final norm, so
    // the three hc_head_* tensors do not exist in the checkpoint and cannot be required here.
    const std::pair<const void*, const char*> req[] = {
      {embed_, "token_embd"}, {head_.data, "output"}, {out_norm_, "output_norm"},
      {cfg_.v41 ? (const void*)1 : hc_head_fn_.data, "hc_head_fn"},
      {cfg_.v41 ? (const void*)1 : hc_head_base_, "hc_head_base"},
      {cfg_.v41 ? (const void*)1 : hc_head_scale_, "hc_head_scale"},
    };
    for (const auto& [ptr, name] : req)
      if (!ptr) { if (err) *err = std::string("container is missing ") + name; return false; }
  }
  // The block's head epilogue runs on the card — the same three kernels the draft's does — so its
  // three vectors need handles for the same reason the draft's do: a container mapping reaches a
  // kernel as an unmapped address. `hc_head_fn_` already has one; b.dense registers as it binds.
  if (dops_.reg_vec) {
    out_norm_gpu_ = dops_.reg_vec(dops_.ctx, out_norm_, cfg_.n_embd);
    // Absent on V4.1, where the head has no mixer of its own.
    if (hc_head_base_)  hc_head_base_gpu_  = dops_.reg_vec(dops_.ctx, hc_head_base_, cfg_.hc_mult);
    if (hc_head_scale_) hc_head_scale_gpu_ = dops_.reg_vec(dops_.ctx, hc_head_scale_, 1);
  }

  // A container may hold fewer layers than the config declares (--layers during bring-up).
  cfg_.n_layer = (uint32_t)std::min<uint64_t>(cfg_.n_layer, aff_.layer_count());
  if (max_layers_) cfg_.n_layer = std::min(cfg_.n_layer, max_layers_);
  if (index_topk_override_) cfg_.index_topk = index_topk_override_;
  layers_.resize(cfg_.n_layer);

  for (uint32_t l = 0; l < cfg_.n_layer; ++l) {
    LayerWeights& w = layers_[l];
    const std::string p = "blk." + std::to_string(l) + ".";
    w.experts    = &aff_.layer(l);
    w.ratio      = cfg_.ratio_for(l);
    if (const char* miss = bind_layer_common(b, cfg_, dops_, aff_, p, w, &router_bf16_)) {
      if (err) *err = "layer " + std::to_string(l) + " is missing " + miss;
      return false;
    }
    if (dops_.kv_sinks && w.attn_sinks) dops_.kv_sinks(dops_.ctx, l, w.attn_sinks, cfg_.n_head);
    w.comp_wkv   = b.dense(p + "comp_wkv");
    w.comp_wgate = b.dense(p + "comp_wgate");
    w.comp_ape   = b.f32(p + "comp_ape");
    w.comp_norm  = b.f32(p + "comp_norm");
    w.idx_wq_b       = b.dense(p + "idx_wq_b");
    w.idx_proj       = b.dense(p + "idx_proj");
    w.idx_comp_wkv   = b.dense(p + "idx_comp_wkv");
    w.idx_comp_wgate = b.dense(p + "idx_comp_wgate");
    w.idx_comp_ape   = b.f32(p + "idx_comp_ape");
    w.idx_comp_norm  = b.f32(p + "idx_comp_norm");
    // Compressed layers rotate with the long-context base under YaRN frequency interpolation;
    // uncompressed layers disable YaRN entirely and use the base theta.
    if (w.ratio != 0) {
      w.rope.freq_base  = cfg_.compress_rope_theta;
      w.rope.freq_scale = cfg_.rope_factor > 0.0f ? 1.0f / cfg_.rope_factor : 1.0f;
      w.rope.ext_factor = cfg_.rope_factor > 1.0f ? 1.0f : 0.0f;
      w.rope.orig_ctx   = cfg_.rope_orig_ctx;
      w.rope.beta_fast  = cfg_.rope_beta_fast;
      w.rope.beta_slow  = cfg_.rope_beta_slow;
    } else {
      w.rope.freq_base = cfg_.rope_theta;
    }

    // V4.1: the indexer keys come from the layer's own projection (kv-source layers only), the
    // routing gains an image-span bias, and two layers carry the Engram projections. All optional
    // per layer — the checks below are what make "absent where it should be present" fatal.
    if (cfg_.v41) {
      w.idx_wk      = b.dense(p + "idx_wk");
      w.idx_k_norm  = b.f32(p + "idx_k_norm");
      w.router_b_vl = b.f32(p + "ffn_gate_bias_vl");
      w.engram_wkv  = b.dense(p + "engram_wkv");
      w.engram_q    = b.f32(p + "engram_q");
      w.engram_k    = b.f32(p + "engram_k");
    }

    // Missing weights must fail the load, not degrade the output — see the note on the same check
    // over the dense weights above.
    if (cfg_.v41) {
      // A compressing layer that is NOT a kv source is legal here and carries no compressor: it
      // reads the source layer's compressed KV. `wgate` follows ratio 2 (the ratio-1 source has
      // none), so it is required only where the checkpoint put it.
      if (cfg_.is_kv_source(l) && (!w.comp_wkv || !w.comp_norm || !w.idx_wk || !w.idx_k_norm)) {
        if (err) *err = "layer " + std::to_string(l) + " is a kv-source layer with no compressor / indexer keys";
        return false;
      }
      if (cfg_.is_index_source(l) && (!w.idx_wq_b || !w.idx_proj)) {
        if (err) *err = "layer " + std::to_string(l) + " is an index-source layer with no indexer queries";
        return false;
      }
      if (cfg_.is_engram_layer(l) && (!w.engram_wkv || !w.engram_q || !w.engram_k)) {
        if (err) *err = "layer " + std::to_string(l) + " is an engram layer with no engram weights";
        return false;
      }
      if (!w.router_b) {
        if (err) *err = "layer " + std::to_string(l) + " is missing ffn_gate_bias";
        return false;
      }
      // Present in every shipped V4.1 layer. Missing, the router would silently use the text bias
      // inside image spans — invisible in a text-only run and wrong in a VL one.
      if (!w.router_b_vl) {
        if (err) *err = "layer " + std::to_string(l) + " is missing ffn_gate_bias_vl";
        return false;
      }
    } else {
      if (w.ratio && (!w.comp_wkv || !w.comp_wgate || !w.comp_ape || !w.comp_norm)) {
        if (err) *err = "layer " + std::to_string(l) + " has ratio " + std::to_string(w.ratio) +
                        " but no compressor weights";
        return false;
      }
      if (w.ratio == 4 && (!w.idx_wq_b || !w.idx_proj || !w.idx_comp_wkv || !w.idx_comp_wgate ||
                           !w.idx_comp_ape || !w.idx_comp_norm)) {
        if (err) *err = "layer " + std::to_string(l) + " is a CSA layer with no indexer weights";
        return false;
      }
      // The aux-loss-free bias exists on non-hash layers only, and it changes which experts run.
      if (!cfg_.uses_hash_routing(l) && !w.router_b) {
        if (err) *err = "layer " + std::to_string(l) + " is missing ffn_gate_bias";
        return false;
      }
    }

    if (cfg_.uses_hash_routing(l)) {
      const AffTensorEntry* e = aff_.find_tensor(p + "ffn_gate_hash");
      if (!e) { if (err) *err = "layer " + std::to_string(l) + " is a hash layer with no ffn_gate_hash"; return false; }
      {
        const float* src = reinterpret_cast<const float*>(aff_.tensor_data(*e));
        const uint64_t n = e->desc.rows * e->desc.cols;
        w.hash_table.resize(n);
        for (uint64_t i = 0; i < n; ++i) w.hash_table[i] = (int32_t)src[i];
      }
    }
  }

  // ---- who owns each layer's compressed KV ------------------------------------------------------
  //
  // V4.1: the nearest kv-source at or before the layer. The reference calls this sharing a cache
  // "with the layers that share its ratio", and the source is always the first of its run, so
  // nearest-preceding and first-of-run are the same layer. V4: every layer owns its own, and the
  // map is the identity, so the device path is unchanged for it.
  kv_owner_.resize(cfg_.n_layer);
  for (uint32_t l = 0; l < cfg_.n_layer; ++l) {
    uint32_t owner = l;
    if (cfg_.v41 && cfg_.ratio_for(l) && !cfg_.is_kv_source(l)) {
      owner = UINT32_MAX;
      for (uint32_t k = l + 1; k-- > 0;)
        if (cfg_.is_kv_source(k) && cfg_.ratio_for(k) == cfg_.ratio_for(l)) { owner = k; break; }
      if (owner == UINT32_MAX) {
        if (err) *err = "layer " + std::to_string(l) + " compresses but no kv-source precedes it";
        return false;
      }
    }
    kv_owner_[l] = owner;
    layers_[l].kv_owner = owner;
  }
  if (dops_.set_kv_owner) dops_.set_kv_owner(dops_.ctx, kv_owner_.data(), (uint32_t)kv_owner_.size());

  // ---- ...and who owns each layer's top-k -------------------------------------------------------
  //
  // One level up, and a DIFFERENT map: V4.1 has eight index sources against four kv sources, so a
  // layer's keys and its admissions can come from two different layers (24, 28, 32 and 36 all score
  // layer 20's keys with their own queries). The reference is `Attention._compress_topk_idxs`: an
  // index source publishes, everyone after it reuses, until the next one publishes.
  //
  // The first compressing layer is always an index source — layer 2 is in both lists — so the slot
  // is written before it is read. Checked rather than assumed: a consumer that found no source
  // would read a mask belonging to no layer, which attends to a plausible wrong set rather than
  // failing.
  idx_owner_.resize(cfg_.n_layer);
  for (uint32_t l = 0; l < cfg_.n_layer; ++l) {
    uint32_t owner = l;
    if (cfg_.v41 && cfg_.ratio_for(l) && !cfg_.is_index_source(l)) {
      owner = UINT32_MAX;
      for (uint32_t k = l + 1; k-- > 0;)
        if (cfg_.is_index_source(k)) { owner = k; break; }
      if (owner == UINT32_MAX) {
        if (err) *err = "layer " + std::to_string(l) + " compresses but no index-source precedes it";
        return false;
      }
    }
    idx_owner_[l] = owner;
  }
  if (dops_.set_idx_owner)
    dops_.set_idx_owner(dops_.ctx, idx_owner_.data(), (uint32_t)idx_owner_.size());
  if (dops_.set_shift_pre) dops_.set_shift_pre(dops_.ctx, cfg_.v41);
  return true;
}

// ---- the DSpark draft, from its own container --------------------------------------------------
//
// A companion file and not part of the model, because under forced residency the text the engine
// produces is a function of which experts fit in VRAM: folding 7 GiB of draft weights into the main
// container would move every fingerprint the A/Bs are taken against.
//
// Every failure below is fatal to the caller. There is no fallback and there must not be one: a
// stage missing a projection does not crash — it emits token ids that are never accepted, decode
// still produces correct text at about the speed it had without the draft, and the run reports a
// number that reads as an honest negative result about speculation. That is the one failure mode
// nobody would go looking for, so the load refuses instead.
bool Model::load_dspark(const std::string& path, std::string* err) {
  dspark_ = DsparkWeights{};
  if (!cfg_.dspark_block) { if (err) *err = "config has no dspark_block_size"; return false; }
  if (cfg_.dspark_taps.empty()) { if (err) *err = "config has no dspark_target_layer_ids"; return false; }
  for (int32_t t : cfg_.dspark_taps)
    if (t < 0 || (uint32_t)t >= cfg_.n_layer) {
      if (err) *err = "dspark_target_layer_ids names layer " + std::to_string(t) +
                      ", which is outside the model's " + std::to_string(cfg_.n_layer);
      return false;
    }
  if (!dspark_aff_.open(path, err)) return false;

  // One global codebook decodes every routed expert in the engine, so the two containers have to
  // agree on it byte for byte. aff-quantize --dspark copies it rather than refitting; this is the
  // other half of that contract, and it is checked because a codebook mismatch is not a crash —
  // it is 768 experts of plausible noise.
  {
    const AffTensorEntry* a = aff_.find_tensor("__codebook");
    const AffTensorEntry* c = dspark_aff_.find_tensor("__codebook");
    if (!a || !c) { if (err) *err = "one of the containers has no __codebook"; return false; }
    const uint64_t n = a->desc.rows * a->desc.cols;
    if (c->desc.rows * c->desc.cols != n ||
        std::memcmp(aff_.tensor_data(*a), dspark_aff_.tensor_data(*c), n * 4) != 0) {
      if (err) *err = "the draft container's __codebook differs from the model's; re-run "
                      "aff-quantize --dspark --codebook <the model .aff>";
      return false;
    }
  }

  const uint32_t n_stage = (uint32_t)dspark_aff_.layer_count();
  if (!n_stage) { if (err) *err = "draft container has no stages"; return false; }
  Binder b{dspark_aff_, &dops_};
  DsparkWeights d;
  d.stage.resize(n_stage);
  for (uint32_t s = 0; s < n_stage; ++s) {
    LayerWeights& w = d.stage[s];
    const std::string p = "mtp." + std::to_string(s) + ".";
    w.experts = &dspark_aff_.layer(s);
    w.ratio   = 0;                       // DSparkAttention asserts it; aff-quantize checked it
    if (const char* miss = bind_layer_common(b, cfg_, dops_, dspark_aff_, p, w, &router_bf16_)) {
      if (err) *err = "draft stage " + std::to_string(s) + " is missing " + miss;
      return false;
    }
    // The stages get KV slots after the model's layers, and their sinks have to follow them there.
    // Missing, a_sink[43..45] stays null and the kernel simply attends without a sink — no crash,
    // no warning, and a draft nobody accepts.
    if (dops_.kv_sinks && w.attn_sinks)
      dops_.kv_sinks(dops_.ctx, cfg_.n_layer + s, w.attn_sinks, cfg_.n_head);
    // Uncompressed, so the base theta with YaRN off — the same rule the model's ratio-0 layers use.
    w.rope.freq_base = cfg_.rope_theta;
    if (!w.router_b) { if (err) *err = "draft stage " + std::to_string(s) + " is missing ffn_gate_bias"; return false; }
  }
  d.main_proj  = b.dense("mtp_main_proj");
  d.main_norm  = b.f32("mtp_main_norm");
  if (d.main_norm && dops_.reg_vec)
    d.main_norm_gpu = dops_.reg_vec(dops_.ctx, d.main_norm, cfg_.n_embd);
  d.norm       = b.f32("mtp_norm");
  d.hc_head_fn = b.dense("mtp_hc_head_fn");
  d.hc_head_base  = b.f32("mtp_hc_head_base");
  d.hc_head_scale = b.f32("mtp_hc_head_scale");
  // The head's collapse runs on the card, so its three vectors need handles for the same reason
  // main_norm does: a container mapping reaches a kernel as an unmapped address.
  if (dops_.reg_vec) {
    if (d.norm) d.norm_gpu = dops_.reg_vec(dops_.ctx, d.norm, cfg_.n_embd);
    if (d.hc_head_base) d.hc_head_base_gpu = dops_.reg_vec(dops_.ctx, d.hc_head_base, cfg_.hc_mult);
    if (d.hc_head_scale) d.hc_head_scale_gpu = dops_.reg_vec(dops_.ctx, d.hc_head_scale, 1);
  }
  d.markov_w1  = b.dense("mtp_markov_w1");
  d.markov_w2  = b.dense("mtp_markov_w2");
  d.confidence = b.bf16("mtp_confidence");
  {
    const std::pair<const void*, const char*> required[] = {
      {d.main_proj.data, "mtp_main_proj"}, {d.main_norm, "mtp_main_norm"}, {d.norm, "mtp_norm"},
      {cfg_.v41 ? (const void*)1 : d.hc_head_fn.data, "mtp_hc_head_fn"},
      {cfg_.v41 ? (const void*)1 : d.hc_head_base, "mtp_hc_head_base"},
      {cfg_.v41 ? (const void*)1 : d.hc_head_scale, "mtp_hc_head_scale"},
      {d.markov_w1.data, "mtp_markov_w1"}, {d.markov_w2.data, "mtp_markov_w2"},
      {d.confidence, "mtp_confidence"},
    };
    for (const auto& [ptr, name] : required)
      if (!ptr) { if (err) *err = std::string("draft container is missing ") + name; return false; }
  }
  // The projection's width is the tap count times n_embd, and getting that wrong reads the wrong
  // slice of a concatenation rather than failing — so it is checked against the config that named
  // the taps, not against the container that stored it.
  if (const AffTensorEntry* e = dspark_aff_.find_tensor("mtp_main_proj")) {
    const uint64_t want = (uint64_t)cfg_.n_embd * cfg_.dspark_taps.size();
    if (e->desc.cols != want) {
      if (err) *err = "mtp_main_proj is " + std::to_string(e->desc.cols) + " wide, but " +
                      std::to_string(cfg_.dspark_taps.size()) + " taps x " +
                      std::to_string(cfg_.n_embd) + " is " + std::to_string(want);
      return false;
    }
  }
  dspark_ = std::move(d);
  return true;
}

// Nothing here is sized by the context window any more — see LayerState. The device allocators are
// what grow with --kv-size, and they are the ones that should.
void Model::init_state(SeqState* s, uint64_t max_pos) const {
  s->layer.assign(cfg_.n_layer, LayerState{});
  s->hc.assign((size_t)cfg_.hc_mult * cfg_.n_embd, 0.0f);
  s->pos = 0;
  // amdnas-fixes init-state compress_reset: the device-side compressor/indexer history rings were never cleared between
  // requests (the op existed, nothing called it). A request's first `hist` positions pool over slots the previous request's
  // tail wrote; if one of those rows went bad, every later request inherits it. Clear them with the host state.
  if (bops_.compress_reset && bops_.ctx) (void)bops_.compress_reset(bops_.ctx);
  for (uint32_t l = 0; l < cfg_.n_layer; ++l) {
    const uint32_t ratio = cfg_.ratio_for(l);
    if (ratio) s->layer[l].comp_rows = max_pos / ratio + 2;
  }
}

// Unwind to `keep` positions after a speculative block whose tail was rejected.
//
// Only the counts move, and that is a property of the two caches rather than a shortcut:
//
//   * The raw ring is addressed by position. The rejected positions are keep..pos-1; the next block
//     starts at exactly `keep` and is at least as long as the rejected run, so every stale slot is
//     overwritten by a real token before anything reads it. A query at position q inside that block
//     reads [q-sliding+1, q], and every slot in (keep-1, q] was written by the block itself.
//   * The compressor's cross-chunk state is a RING of raw projections indexed by
//     `position & (hist-1)` (see gpu/compressor_gpu.h) and not an accumulator, so the same argument
//     covers it. A row pooled at position q reads positions q-hist+1..q: those at or below keep-1
//     are the committed values, and those above were re-written by the new block.
//
// What is NOT recoverable that way is how much of each cache is real, which is what attention is
// told per token — so that is what this fixes.
void Model::rollback(SeqState* s, uint32_t keep) const {
  if (!s || keep > s->pos) return;
  for (uint32_t l = 0; l < cfg_.n_layer; ++l) {
    LayerState& st = s->layer[l];
    const uint32_t ratio = cfg_.ratio_for(l);
    st.n_raw = std::min(keep, cfg_.sliding);
    st.raw_head = cfg_.sliding ? keep % cfg_.sliding : 0u;
    if (ratio) {
      // One row per `ratio` positions: emitted at p when (p+1) % ratio == 0, so `keep` positions
      // have produced keep/ratio of them. min() because the cache can have been capacity-capped.
      st.n_comp = std::min(st.n_comp, keep / ratio);
      if (ratio == 4) st.n_idx_comp = std::min(st.n_idx_comp, keep / ratio);
    }
  }
  s->pos = keep;
}

void Model::restore_state(SeqState* s, uint32_t pos) const {
  if (!s || s->layer.size() < cfg_.n_layer) return;
  for (uint32_t l = 0; l < cfg_.n_layer; ++l) {
    LayerState& st = s->layer[l];
    const uint32_t ratio = cfg_.ratio_for(l);
    st.n_raw = std::min(pos, cfg_.sliding);
    st.raw_head = cfg_.sliding ? pos % cfg_.sliding : 0u;
    // The same rule the compressor emits by: a row lands at p when (p+1) % ratio == 0, so `pos`
    // positions have produced pos/ratio of them.
    if (ratio) {
      st.n_comp = pos / ratio;
      if (ratio == 4) st.n_idx_comp = pos / ratio;
    }
  }
  s->pos = pos;
}

ExpertMatrixView Model::expert_view(uint32_t layer, uint32_t expert, int which) const {
  const AffLayerDesc& L = aff_.layer(layer);
  const ExpertView v = aff_.expert(layer, expert);
  const MatrixView& m = which == 0 ? v.gate : (which == 1 ? v.up : v.down);
  const AffMatrixDesc& d = which == 0 ? L.gate : (which == 1 ? L.up : L.down);
  ExpertMatrixView e;
  e.data = m.data; e.scales = m.scales; e.rscales = m.rscales;
  e.rows = d.rows; e.cols = d.cols;
  // From the container, not a literal: the file decides whether these codes are 5-bit pairs or
  // 10-bit quads, and reading one as the other is in-bounds and completely wrong.
  e.spec = expert_quant_spec(d.quant);
  return e;
}

// ------------------------------------------------------------------------------------------
// AFF_FORCE_RESIDENT — see the comment on set_resident_pred
// ------------------------------------------------------------------------------------------

bool Model::force_resident_routing() {
  static const bool on = [] {
    const char* v = getenv("AFF_FORCE_RESIDENT");
    return v && *v && *v != '0';
  }();
  return on;
}

void Model::force_resident(uint32_t layer, const float* logits, uint32_t* sel, float* w,
                           uint32_t* n) const {
  if (!res_ || !*n || !force_resident_routing()) return;
  bool all = true;
  for (uint32_t k = 0; k < *n; ++k)
    if (!res_(res_ctx_, layer, sel[k])) { all = false; break; }
  if (all) return;

  if (logits) {
    // Re-rank rather than substitute, so the weights are still route_topk's own renormalisation
    // over whatever set wins. A degenerate weight vector would change the numerics downstream and
    // therefore the timing, which is the one thing this flag must not do.
    //
    // This repeats route_topk's arithmetic instead of calling it with masked logits, and that is
    // deliberate: the aux-loss-free `bias` is added AFTER sqrtsoftplus, so a -inf logit still
    // scores 0 and a non-resident expert with a positive bias would out-rank a resident one with a
    // real score. The mask belongs on the selection value.
    const float* bias = layers_[layer].router_b;
    const uint32_t k = std::min<uint32_t>(8, std::min(cfg_.n_expert_used, cfg_.n_expert));
    uint32_t best[8] = {0};
    float    bsel[8];
    uint32_t got = 0;
    for (uint32_t e = 0; e < cfg_.n_expert; ++e) {
      if (!res_(res_ctx_, layer, e)) continue;
      const float sv = sqrtsoftplus(logits[e]) + (bias ? bias[e] : 0.0f);
      if (got == k && sv <= bsel[got - 1]) continue;      // route_topk's tie-break: lower id wins
      uint32_t at = got < k ? got : k - 1;
      while (at > 0 && sv > bsel[at - 1]) { bsel[at] = bsel[at - 1]; best[at] = best[at - 1]; --at; }
      bsel[at] = sv; best[at] = e;
      if (got < k) ++got;
    }
    if (got == k) {
      double sum = 0.0;
      float score[8];
      for (uint32_t i = 0; i < k; ++i) { score[i] = sqrtsoftplus(logits[best[i]]); sum += score[i]; }
      if (sum < 6.103515625e-5) sum = 6.103515625e-5;
      const double inv = cfg_.norm_topk_prob ? 1.0 / sum : 1.0;
      for (uint32_t i = 0; i < k; ++i) {
        sel[i] = best[i];
        w[i] = (float)(score[i] * inv * cfg_.routed_scaling);
      }
      *n = k;
      return;
    }
    // Fewer resident experts than the layer selects. Fall through to the scan, which will
    // duplicate — a layer that thin is not what this flag is for and is worth not hiding.
  }

  // Hash-routed layers have no logits to re-rank. Walk forward from the id that missed, which is
  // deterministic and keeps the selection a function of the token.
  for (uint32_t k = 0; k < *n; ++k) {
    if (res_(res_ctx_, layer, sel[k])) continue;
    for (uint32_t step = 1; step <= cfg_.n_expert; ++step) {
      const uint32_t cand = (sel[k] + step) % cfg_.n_expert;
      if (!res_(res_ctx_, layer, cand)) continue;
      bool taken = false;
      for (uint32_t j = 0; j < *n; ++j) if (j != k && sel[j] == cand) { taken = true; break; }
      if (taken) continue;
      sel[k] = cand;
      break;
    }
  }
}

// ------------------------------------------------------------------------------------------
// Forward
// ------------------------------------------------------------------------------------------

// Closes a phase: wall time into `.t`, and the drain nanoseconds accumulated since the last
// boundary into `.wait`. The second half is what says whether a row of the table is code to optimise
// or a queue to go and look behind — without it, a phase named for host work can be almost entirely
// a stream synchronise. See PhaseProfile::Ph and engine/instrument.h.
#define AFF_PH(field) do { if (prof_on) { \
    const auto _n = std::chrono::steady_clock::now(); \
    const uint64_t _w = drain_ns_now(); \
    prof_.field.t += std::chrono::duration<double>(_n - _ph).count(); \
    prof_.field.wait += (double)(_w - _phw) * 1e-9; \
    _ph = _n; _phw = _w; } } while (0)

// Declares both clocks a phase boundary needs. Two separate `auto x = ...` lines drifted apart
// once already when a new forward path was added and only copied the first.
#define AFF_PH_BEGIN() \
    auto _ph = std::chrono::steady_clock::now(); \
    uint64_t _phw = drain_ns_now(); \
    (void)_phw

void Model::forward_token(uint32_t token_id, SeqState* s,
                          ExpertFetch fetch, void* fetch_ctx,
                          std::vector<float>* logits, bool routed_experts) const {
  const bool prof_on = profiling();
  // The 4-lane residual state lives in VRAM for the whole token when the device can host it, so a
  // sublayer need not make a host round trip at all. `norm` still comes back once per sublayer: the
  // router, the expert dispatch, the compressor and the lightning indexer all still read it there.
  const bool dev_hc = dops_.hc_ready && dops_.hc_ready(dops_.ctx) && dops_.hc_pre && dops_.hc_post;
  AFF_PH_BEGIN();
  // ---- THE HOST LANES HAVE TO STILL MEAN SOMETHING ----------------------------------------------
  //
  // This path carries the residual through `s->hc` on the host; the block path leaves it in VRAM and
  // does not maintain the host copy. Continuing here on a state a block last touched would read the
  // lanes of whatever position that block happened to end on — and after a partial acceptance the
  // sequence has already rewound past it. That produced a fluent, wrong continuation and no error,
  // which is the worst of the three possible outcomes.
  //
  // Fatal rather than a resync: the two paths are meant to be alternatives, and a caller that
  // reaches here has a bug that a silent fetch would hide.
  if (!s->hc_host) {
    aff::ui::fatal("fatal: forward_token on a sequence whose hidden state a BLOCK left in "
                 "VRAM. The host lanes are stale — and after a partial acceptance they belong to a "
                 "position the sequence has rewound past. Run one path or the other.\n");
    std::abort();
  }
  const uint32_t E = cfg_.n_embd, NH = cfg_.n_head, HC = cfg_.hc_mult;
  const AttnConfig ac = cfg_.attn();
  const uint32_t W = ac.qk_width(), ROT = cfg_.rope_dim;
  const uint32_t QR = cfg_.q_lora_rank, OR = cfg_.o_lora_rank;
  const uint32_t IHD = cfg_.index_head_dim, INH = cfg_.index_n_heads;
  const uint32_t pos = s->pos;

  // Every HC lane starts as the token embedding.
  if (embed_) {
    std::vector<float> e(E);
    bf16_dequant(embed_ + (size_t)token_id * E, E, e.data());
    for (uint32_t h = 0; h < HC; ++h) std::memcpy(s->hc.data() + (size_t)h * E, e.data(), E * 4);
    if (dev_hc) dops_.hc_seed(dops_.ctx, e.data(), E, HC);
  }

  std::vector<float> cur(E), norm(E), qa(QR), qr_norm(QR), q((size_t)NH * W), kv(W), raw(W);
  std::vector<float> comp_kv(2 * W), comp_sc(2 * W);
  // Separate from comp_kv/comp_sc, not aliased onto them: all four matvecs are issued in one batch,
  // so both compressors' outputs are live at the same time.
  std::vector<float> idx_kv(2 * IHD), idx_sc(2 * IHD);
  std::vector<float> heads((size_t)NH * W), low((size_t)8 * OR), block(E), hc_next(s->hc.size());
  std::vector<float> gate(cfg_.moe_inter), up(cfg_.moe_inter), act(cfg_.moe_inter), sh(E);
  std::vector<uint32_t> sel(cfg_.n_expert_used);
  std::vector<float> sw(cfg_.n_expert_used), rlog(cfg_.n_expert);

  for (uint32_t l = 0; l < cfg_.n_layer; ++l) {
    const LayerWeights& w = layers_[l];
    LayerState& st = s->layer[l];
    HcControl hcc;

    AFF_PH(hyper);
    // Every host read of `norm` in this layer goes through here first. `norm` is computed on device
    // and its copy back is left in flight (see hc_pre's `defer`); this drains it. Idempotent and
    // free once the copy has landed, so it can be called defensively — the cost of getting it wrong
    // is a stale activation, which reads as fluent text and gets blamed on quantisation.
    auto norm_ready = [&] {
      if (dev_hc && dops_.norm_collect) dops_.norm_collect(dops_.ctx, norm.data(), E);
    };

    // ---- attention sublayer ------------------------------------------------------------------
    // Does anything on the HOST read `norm` in this sublayer? Under dev_hc the answer is almost
    // always no — attn_q, attn_kv and the compressor all take the device copy — and the exception
    // is the lightning indexer, which still has a host matvec (`w.idx_proj`). The indexer only runs
    // once the compressed cache outgrows index_topk, so at short context hc_pre's copy-back and its
    // stream synchronise deliver a vector nobody reads. That synchronise, twice a layer for every
    // layer, is what makes hc_pre the most expensive device entry point in the engine. `+ 1` because
    // the compressor may append a row before the indexer looks, which is the same margin
    // `want_qrnorm` uses below.
    // V4.1 runs an indexer on its index-source layers, whatever their ratio — `ratio == 4` was V4's
    // way of naming the same set and is false for every V4.1 layer. `is_index_source` is the
    // identity-preserving form: on V4 it IS `ratio == 4`.
    const bool idx_may_run = cfg_.is_index_source(l) && w.idx_wq_b &&
                             cfg_.index_topk < st.n_comp + 1u;
    const bool host_norm = !dev_hc || idx_may_run;
    if (dev_hc) {
      // A refusal here means `norm` was never written, and everything downstream would then read a
      // stale buffer and produce fluent nonsense. There is no correct way to continue.
      if (!dops_.hc_pre(dops_.ctx, w.hc_attn_fn.gpu, w.hc_attn_scale_gpu, w.hc_attn_base_gpu,
                        w.attn_norm_gpu, E, HC, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, cfg_.rms_eps,
                        host_norm ? norm.data() : nullptr,
                        dops_.norm_collect != nullptr)) {
        aff::ui::fatal("fatal: device hidden state is active but hc_pre refused at layer %u "
                     "— `norm` would be stale\n", l);
        std::abort();
      }
    } else {
      if (w.hc_attn_fn) {
        hc_control((const uint16_t*)w.hc_attn_fn.data, w.hc_attn_scale, w.hc_attn_base, s->hc.data(), E, HC,
                   cfg_.hc_sinkhorn_iters, cfg_.hc_eps, cfg_.rms_eps, &hcc);
        hc_reduce(s->hc.data(), hcc.pre, E, HC, cur.data());
      } else {
        std::memcpy(cur.data(), s->hc.data(), E * 4);
      }
      rms_norm(cur.data(), w.attn_norm, E, cfg_.rms_eps, norm.data());
    }

    // Q: low-rank pair with a learned norm between, then a weightless per-head norm.
    //
    // Fused, this is one round trip for two matvecs, two norms and the RoPE. Unfused it is two
    // round trips plus three host passes over 32768 floats, and the passes are not the expensive
    // part — the trips are.
    const RopeDerived rd = rope_derive(w.rope, ROT);
    bool q_fused = false, kv_fused = false;
    // The lightning indexer is the ONLY consumer of qr_norm, and it runs only once the compressed
    // cache outgrows index_topk — n_comp is pos/4, so below ~2048 positions it never runs at all.
    // Asking for qr_norm unconditionally forced a device synchronise per layer per token to deliver
    // a vector nobody read. `+ 1` because the compressor may append a row later this sublayer.
    const bool want_qrnorm = idx_may_run;
    // The device owns the compressor's pooling window once anything has used the device path, so
    // decode has to use it too — a host-side pool here would run over a state that prefill stopped
    // updating. Both halves move together: the indexer's keys are the compressor's output.
    //
    // "This layer BUILDS index keys", which is not the same question as "this layer indexes". V4:
    // its own second compressor, on every ratio-4 layer. V4.1: `wk` applied to the KV compressor's
    // pre-RoPE latent, on kv-source layers only — four of them against eight index sources.
    const bool has_idx = cfg_.v41
                             ? (cfg_.is_kv_source(l) && w.idx_wk && w.idx_k_norm)
                             : (w.ratio == 4 && w.idx_comp_wkv && w.idx_comp_wgate && w.idx_proj);
    // `comp_wgate` is not required: V4.1's ratio-1 layers pool a single position and ship no gate,
    // and the device compressor takes h_sc = -1 for that (DenseW leaves .gpu at -1 when unbound).
    const bool dev_comp = dops_.compress_one && dops_.indexer_one && w.ratio && w.comp_wkv &&
                          (w.comp_wgate || cfg_.v41) && dev_hc && cfg_.is_kv_source(l);
    // Whether attn_q should leave qr_norm in VRAM for the indexer. On V4 an indexing layer always
    // compresses too, so `dev_comp` came for free; on V4.1 it does not — layers 24, 28, 32 and 36
    // index without owning a compressor or a key cache, and gating this on dev_comp would send
    // their queries to the host and then refuse to score them.
    const bool dev_index = want_qrnorm && st.n_comp && dev_hc && dops_.indexer_one &&
                           (cfg_.v41 || (dev_comp && has_idx));
    if (dops_.want_qrn_dev) dops_.want_qrn_dev(dops_.ctx, dev_index);

    // ---- one launch for every matvec in this sublayer that reads `norm` ------------------------
    //
    // wq_a, wkv and the compressor's four are mutually independent and all read the same vector;
    // the first two are just the opening link of the attn_q and attn_kv chains. The job list is
    // built HERE rather than at the compressor below, because the batch has to be issued before
    // the two chains that consume half of it. `comp_collect` further down finishes the other half.
    const uint32_t cw = ((w.ratio == 4) ? 2u : 1u) * W;
    const uint32_t iw = 2u * IHD;
    int32_t ch[4]; uint64_t crows[4]; float* cy[4]; uint32_t cn = 0;
    // NOT when the device compressor runs. `compress_one` issues these same four matvecs on device
    // and reads the results there; the only host consumer of `cy` is the fallback below, which
    // dev_comp skips. Building the list anyway fires the `mv_multi` further down every layer of
    // every token — four matvecs, a D2H and a full stream drain — for values nothing then reads.
    if (!dev_comp && w.ratio && w.comp_wkv) {
      ch[cn] = w.comp_wkv.gpu;   crows[cn] = cw; cy[cn] = comp_kv.data(); ++cn;
      ch[cn] = w.comp_wgate.gpu; crows[cn] = cw; cy[cn] = comp_sc.data(); ++cn;
      if (w.ratio == 4 && w.idx_comp_wkv) {
        ch[cn] = w.idx_comp_wkv.gpu;   crows[cn] = iw; cy[cn] = idx_kv.data(); ++cn;
        ch[cn] = w.idx_comp_wgate.gpu; crows[cn] = iw; cy[cn] = idx_sc.data(); ++cn;
      }
    }
    // Everything here runs on ONE stream, so what the batch changes is the ORDER: the compressor's
    // four matvecs are the only ones in this sublayer that nothing downstream waits for, so putting
    // them in the same launch as wq_a puts them ahead of the q chain, which is the critical path.
    // Nothing when the device compressor runs: it issues its own batch, on device, and attn_pre's
    // half of the deal is a copy to the host that nobody would read.
    const uint32_t pre_cn = dev_comp ? 0u : cn;
    bool pre_fused = false;
    if (dev_hc && dops_.attn_pre && w.wq_a && w.wq_b && w.wkv &&
        dops_.attn_q && dops_.attn_kv)
      pre_fused = dops_.attn_pre(dops_.ctx, w.wq_a.gpu, w.wkv.gpu, QR, W, ch, crows, E, pre_cn, cy);

    if (w.wq_a && w.wq_b && dops_.attn_q)
      q_fused = dops_.attn_q(dops_.ctx, w.wq_a.gpu, w.wq_b.gpu, w.q_norm_gpu,
                             dev_hc ? nullptr : norm.data(), E, QR,
                             NH, W, ROT, pos, rd, cfg_.rms_eps, q.data(),
                             (want_qrnorm && !dev_index) ? qr_norm.data() : nullptr);
    // kv_out is null when the device owns the cache: the latent is QAT'd, f16-rounded and encoded
    // in place by kv_commit below, so it never crosses the bus at all.
    const bool kv_on_device = dops_.kv_commit && dops_.attend;
    if (w.wkv && dops_.attn_kv)
      kv_fused = dops_.attn_kv(dops_.ctx, w.wkv.gpu, w.kv_norm_gpu, dev_hc ? nullptr : norm.data(),
                               E, W, ROT, pos,
                               rd, cfg_.rms_eps, kv_on_device ? nullptr : kv.data());

    if (!q_fused && w.wq_a && w.wq_b) {
      norm_ready();
      dense_mv(w.wq_a, norm.data(), QR, E, qa.data());
      rms_norm(qa.data(), w.q_norm, QR, cfg_.rms_eps, qr_norm.data());
      dense_mv(w.wq_b, qr_norm.data(), (uint64_t)NH * W, QR, q.data());
      head_rms_norm_inplace(q.data(), NH, W, cfg_.rms_eps);
    }
    if (!kv_fused && w.wkv) {                      // ONE shared latent, serving all 64 heads
      norm_ready();
      dense_mv(w.wkv, norm.data(), W, E, raw.data());
      rms_norm(raw.data(), w.kv_norm, W, cfg_.rms_eps, kv.data());
    }

    if (!q_fused)  rope_tail(q.data(), NH, W, ROT, pos, w.rope, false);
    if (!kv_fused) rope_tail(kv.data(), 1, W, ROT, pos, w.rope, false);
    if (!(kv_fused && kv_on_device)) {
      aff::ui::fatal("fatal: layer %u did not commit its latent to the device KV cache "
                   "(fused=%d on_device=%d)\n", l, (int)kv_fused, (int)kv_on_device);
      std::abort();
    }
    {                                              // push into the sliding raw window
      // The device ring is WIDER than `sliding` so prefill can commit a whole sub-batch before any
      // of it attends, so it is indexed by absolute position. Only the counts live here.
      dops_.kv_commit(dops_.ctx, l, pos, W, ROT);
      st.raw_head = (st.raw_head + 1) % cfg_.sliding;
      if (st.n_raw < cfg_.sliding) ++st.n_raw;
    }

    AFF_PH(attn_proj);
    // ---- compressors: one row emitted per `ratio` positions -----------------------------------
    //
    // All FOUR of these matvecs read `norm` and nothing else, and none of them feeds another. They
    // are issued as ONE round trip: four separate calls mean four host synchronisations a layer, for
    // weights that take a fraction of that to read. On this device a sync that lets the queue drain
    // costs more than the weights do — see gpu/keepalive.h for why an idle dispatch path is
    // expensive.
    bool comp_fused = false;
    // Waits for whatever the batch left in flight, and aborts rather than continue if either
    // attention chain failed to take the projection it was handed — a chain that skipped its first
    // matvec and then read stale scratch is fluent, wrong text.
    if (pre_fused) comp_fused = dops_.comp_collect(dops_.ctx) && pre_cn > 0;
    if (!comp_fused && dops_.mv_multi && cn)
      comp_fused = dops_.mv_multi(dops_.ctx, ch, crows, E, cn, dev_hc ? nullptr : norm.data(), cy);
    // Every fallback below this point reads `norm` on the host, and when host_norm is false that
    // buffer holds the PREVIOUS layer's values. Wrong output from a stale activation looks fluent
    // and would be attributed to quantisation, so refuse loudly instead. In practice this cannot
    // fire — dev_hc requires can_fuse(), which is what these three test — but "cannot fire" is
    // exactly the assumption that produced the dangling-hook and device-static bugs.
    if (!host_norm &&
        (!q_fused || !kv_fused || (!dev_comp && w.ratio && w.comp_wkv && !comp_fused))) {
      aff::ui::fatal("fatal: layer %u fell back to a host matvec while `norm` lives only in "
                   "VRAM (q=%d kv=%d comp=%d)\n", l, (int)q_fused, (int)kv_fused, (int)comp_fused);
      std::abort();
    }
    if (dev_comp) {
      // The whole compressor for this one position, on device — its own five matvecs, the pool,
      // the RoPE, the QAT and the cache write, with the pooling window read from and written back
      // to VRAM. See gpu/compressor_gpu.h.
      const uint32_t n_out =
          ((pos + 1) % w.ratio == 0 && st.n_comp + 1 <= st.comp_rows) ? 1u : 0u;
      CompressArgs ca;
      ca.layer = l; ca.n = 1; ca.pos0 = pos; ca.ratio = w.ratio; ca.n_out = n_out;
      ca.h_kv = w.comp_wkv.gpu; ca.h_sc = w.comp_wgate.gpu;
      if (has_idx && cfg_.v41) {
        // V4.1: one projection off the latent this same call is about to produce. No second
        // compressor, and no head weights here — those belong to the index-source layers, which
        // are not this set, and indexer_one issues them itself.
        ca.h_idx_wk = w.idx_wk.gpu; ca.idx_k_norm = w.idx_k_norm;
      } else if (has_idx) {
        ca.h_ikv = w.idx_comp_wkv.gpu; ca.h_isc = w.idx_comp_wgate.gpu;
        ca.h_iproj = w.idx_proj.gpu;
        ca.idx_heads = INH;
      }
      ca.ape = w.comp_ape;      ca.norm = w.comp_norm;
      ca.iape = w.idx_comp_ape; ca.inorm = w.idx_comp_norm;
      ca.width = W; ca.idx_dim = IHD; ca.n_embd = E;
      ca.n_rot = kCompressorRot; ca.rope = rope_derive(w.rope, kCompressorRot);
      ca.slot0 = st.n_comp; ca.idx_slot0 = st.n_idx_comp;
      ca.iscale = 1.0f / std::sqrt((float)(IHD * INH));
      ca.eps = cfg_.rms_eps;
      if (!dops_.compress_one(dops_.ctx, ca)) {
        aff::ui::fatal("fatal: layer %u could not run the device compressor, and the host "
                     "copy of its pooling window has not been updated since prefill\n", l);
        std::abort();
      }
      st.n_comp += n_out;
      if (has_idx) st.n_idx_comp += n_out;
      // The layers reading this cache must see the same row count: they do not compress, so their
      // own counter would stay at zero and attention would ignore every compressed row.
      for (uint32_t c = l + 1; c < cfg_.n_layer && kv_owner_[c] == l; ++c) {
        s->layer[c].n_comp = st.n_comp;
        s->layer[c].n_idx_comp = st.n_idx_comp;
      }
    }

    AFF_PH(compressor);
    // ---- lightning indexer: which compressed rows this query may attend to --------------------
    //
    // On V4 this runs where the compressor just ran. On V4.1 it runs on the INDEX-SOURCE layers,
    // which are a superset of the kv sources — so the gate is `is_index_source` (inside
    // idx_may_run) and not `dev_comp`. The layers between two sources run nothing here and attend
    // under the mask the earlier source left, which is what `set_idx_owner` told the device.
    const bool run_idx = idx_may_run && st.n_comp && cfg_.index_topk < st.n_comp &&
                         (cfg_.v41 || (dev_comp && has_idx));
    if (run_idx) {
      // On device, straight into the mask buffer attend() already reads. The query projection reads
      // qr_norm, which attn_q left in VRAM because want_qrn_dev asked it to.
      IndexArgs ia;
      ia.layer = l; ia.n = 1;
      ia.h_wq_b = w.idx_wq_b.gpu; ia.qr = QR;
      ia.n_head = INH; ia.dim = IHD;
      ia.n_keys = std::min<uint32_t>(st.n_idx_comp, st.n_comp);
      ia.n_mask = st.n_comp; ia.topk = cfg_.index_topk;
      ia.pos0 = pos; ia.n_rot = ROT; ia.rope = rope_derive(w.rope, ROT);
      if (cfg_.v41) {
        ia.h_proj = w.idx_proj.gpu; ia.n_embd = E;
        ia.scale = 1.0f / std::sqrt((float)(IHD * INH));
        ia.hadamard = false;          // V4.1's fp4_act_quant does not rotate — see indexer_gpu.h
      }
      if (!dops_.indexer_one(dops_.ctx, ia)) return;
    }

    AFF_PH(indexer);
    // ---- attention over (raw sliding window ++ permitted compressed rows) ---------------------
    // One kernel for both CSA and HCA: the key list does not care which half a row came from,
    // which is exactly why the two "algorithms" are one implementation.
    const float kq = 1.0f / std::sqrt((float)W);
    // The whole key set lives in VRAM, so this is one launch over 64 blocks instead of 64 heads x
    // ~640 keys x 512 dims twice on the CPU.
    bool attn_done = false;
    // When the fused tail will run, the heads stay in VRAM and never come back to the host. The
    // fused_ok query is load-bearing: without it a multi-rank split left the heads on device and
    // then refused to consume them, and the fallback below rotated whatever `heads` held last.
    const bool tail_fused = dops_.attn_out && dops_.fused_ok && dops_.fused_ok(dops_.ctx)
                            && w.wo_a && w.wo_b && w.wo_a.gpu >= 0 && w.wo_b.gpu >= 0;
    // The mask is on device: indexer_one wrote it straight into the buffer attend() reads.
    if (w.attn_sinks && dops_.attend)
      attn_done = dops_.attend(dops_.ctx, l, q_fused ? nullptr : q.data(), nullptr,
                               pos, st.n_comp, kq, tail_fused ? nullptr : heads.data());
    if (w.attn_sinks && !attn_done) {
      aff::ui::fatal("fatal: attention did not run on layer %u — the keys live only in "
                   "VRAM\n", l);
      std::abort();
    }

    AFF_PH(attention);
    // V carried the rotated tail, so undo the rotation before projecting out.
    std::fill(block.begin(), block.end(), 0.0f);
    bool out_done = false;
    if (attn_done && tail_fused)
      out_done = dops_.attn_out(dops_.ctx, w.wo_a.gpu, w.wo_b.gpu, NH, W, ROT, pos,
                                rope_derive(w.rope, ROT), 8, OR, E,
                                dev_hc ? nullptr : block.data());
    if (!out_done) {
      rope_tail(heads.data(), NH, W, ROT, pos, w.rope, true);
      if (w.wo_a && w.wo_b) {
        // Grouped: each of 8 groups maps its 8 heads to a 1024-rank vector, then all 8 ranks are
        // projected back to the model width together.
        dense_mvg(w.wo_a, heads.data(), 8, (uint64_t)(NH / 8) * W, OR, low.data());
        dense_mv(w.wo_b, low.data(), E, (uint64_t)8 * OR, block.data());
      }
    }

    if (dev_hc && out_done) {
      // wo_b is split along its INPUT, so each card holds a partial of the attention contribution.
      // This is the sublayer's one exchange and it has to happen before hc_post reads the block.
      if (dops_.block_reduce) dops_.block_reduce(dops_.ctx, E);
      dops_.hc_post(dops_.ctx, w.hc_attn_fn.data != nullptr, E, HC);
    } else if (w.hc_attn_fn) {
      hc_expand(hc_next.data(), block.data(), s->hc.data(), hcc.post, hcc.comb, E, HC);
      s->hc.swap(hc_next);
    } else {
      for (uint32_t d = 0; d < E; ++d) s->hc[d] += block[d];
    }

    AFF_PH(attn_out);
    // ---- FFN sublayer ------------------------------------------------------------------------
    // DEFERRED. `norm` is computed on device and the copy back is left in flight; norm_ready()
    // below drains it. On the common path — a non-hash layer with the gate resident — the device
    // router runs next on the same stream and synchronises on its own way out, so by the time
    // anything on the host reads `norm` the copy has long landed and the wait is free.
    //
    // Same question the attention sublayer asks above, for the FFN's host readers: the hash layers
    // re-weight their selection with `norm`, and so does anything that falls off the device router.
    // With every routed expert resident and the routing done on device there is nobody. Every
    // consumer below goes through norm_ready() first and that issues the copy on demand, so this
    // predicate decides a COST and not a correctness — being wrong here costs a stall, not a stale
    // activation.
    const bool host_norm_ffn = !dev_hc || !res_ || cfg_.uses_hash_routing(l);
    if (dev_hc) {
      if (!dops_.hc_pre(dops_.ctx, w.hc_ffn_fn.gpu, w.hc_ffn_scale_gpu, w.hc_ffn_base_gpu,
                        w.ffn_norm_gpu, E, HC, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, cfg_.rms_eps,
                        host_norm_ffn ? norm.data() : nullptr,
                        dops_.norm_collect != nullptr)) {
        aff::ui::fatal("fatal: device hidden state is active but hc_pre refused at layer %u "
                     "— `norm` would be stale\n", l);
        std::abort();
      }
    } else {
      if (w.hc_ffn_fn) {
        hc_control((const uint16_t*)w.hc_ffn_fn.data, w.hc_ffn_scale, w.hc_ffn_base, s->hc.data(), E, HC,
                   cfg_.hc_sinkhorn_iters, cfg_.hc_eps, cfg_.rms_eps, &hcc);
        hc_reduce(s->hc.data(), hcc.pre, E, HC, cur.data());
        // V4.1's head has no mixer: it collapses with what the last layer's FFN produced here.
        if (cfg_.v41 && l + 1 == cfg_.n_layer) {
          std::memcpy(s->last_ffn_pre, hcc.pre, sizeof(s->last_ffn_pre));
          s->last_ffn_pre_valid = true;
        }
      } else {
        std::memcpy(cur.data(), s->hc.data(), E * 4);
      }
      rms_norm(cur.data(), w.ffn_norm, E, cfg_.rms_eps, norm.data());
    }

    // Routing. Layers 0-2 look their experts up by token id — no gate matvec, and the choice is
    // known before the forward pass starts, which is a free prefetch oracle.
    uint32_t n_sel = 0;

    // The host always knows the selection here. What remains of forward_token is the CPU-only path
    // the offline tools use: the fp8 shards are preshuffled for the W8A8 GEMM, so a device build
    // aborts in the dense matvec long before it reaches this point, and DSpark's block verify
    // replaced single-token decode on the serving path.
    if (cfg_.uses_hash_routing(l) && !w.hash_table.empty()) {
      const int32_t* row = w.hash_table.data() + (size_t)token_id * cfg_.n_expert_used;
      for (uint32_t k = 0; k < cfg_.n_expert_used; ++k) { sel[k] = (uint32_t)row[k]; sw[k] = 1.0f; }
      n_sel = cfg_.n_expert_used;
      force_resident(l, nullptr, sel.data(), sw.data(), &n_sel);
      if (w.router_w) {                            // hash layers still weight by router probability
        norm_ready();
        float sum = 0.0f;
        for (uint32_t k = 0; k < n_sel; ++k) {
          const uint16_t* rw = w.router_w + (size_t)sel[k] * E;
          float acc = 0.0f;
          for (uint32_t d = 0; d < E; ++d) acc += bf16_to_float(rw[d]) * norm[d];
          sw[k] = sqrtsoftplus(acc);
          sum += sw[k];
        }
        if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
        if (cfg_.norm_topk_prob)
          for (uint32_t k = 0; k < n_sel; ++k) sw[k] /= sum;
        for (uint32_t k = 0; k < n_sel; ++k) sw[k] *= cfg_.routed_scaling;
      }
    } else if (w.router_w) {
      // On device when the gate is resident: `norm` is already in VRAM under dev_hc, so nothing
      // goes up and 1 KB of logits comes back. On the host this read 4.2 MB per layer out of the
      // same DRAM the expert kernel needs.
      const bool gpu = dops_.router && w.router_gpu >= 0 &&
                       dops_.router(dops_.ctx, w.router_gpu, dev_hc ? nullptr : norm.data(),
                                    cfg_.n_expert, E, rlog.data());
      if (!gpu) {
        norm_ready();
        parallel_for(cfg_.n_expert, 64, [&](uint64_t lo, uint64_t hi) {
          for (uint64_t e = lo; e < hi; ++e) {
            const uint16_t* rw = w.router_w + (size_t)e * E;
            float acc = 0.0f;
            for (uint32_t d = 0; d < E; ++d) acc += bf16_to_float(rw[d]) * norm[d];
            rlog[e] = acc;
          }
        });
      }
      RouterOut ro;
      route_topk(rlog.data(), w.router_b, cfg_.n_expert, cfg_.n_expert_used,
                 cfg_.norm_topk_prob, cfg_.routed_scaling, &ro);
      n_sel = ro.n;
      for (uint32_t k = 0; k < n_sel; ++k) { sel[k] = ro.expert[k]; sw[k] = ro.weight[k]; }
      force_resident(l, rlog.data(), sel.data(), sw.data(), &n_sel);
    }

    // BEFORE the shared expert is issued, not after: norm_collect synchronises the stream, and the
    // shared expert is deliberately left in flight so the routed experts' CPU work overlaps it.
    // Collecting afterwards waits for that whole chain and throws the overlap away.
    //
    // It looks free here, because on the common path the device router has just drained this same
    // stream. That is circular: norm_collect is free because the router drained, and the router's
    // drain looked necessary because it fed the host's top-k. Removing only one of them makes the
    // other absorb the identical wait — one stall wearing two names. Removing BOTH is what pays, so
    // the question is never which to remove but whether the data is wanted at all.
    //
    // No predicate means residency is unknown, so assume the host path can run and collect.
    bool host_wants_norm = !dev_hc || !res_;
    for (uint32_t k = 0; k < n_sel && !host_wants_norm; ++k)
      host_wants_norm = !res_(res_ctx_, l, sel[k]);
    if (host_wants_norm) norm_ready();

    AFF_PH(router);
    std::fill(block.begin(), block.end(), 0.0f);
    // The device block accumulates from two independent sources this sublayer — the shared expert
    // and, when every routed expert is resident, the routed set — so it has to be zero before
    // either runs. It already is: hc_post drained it at the end of the previous sublayer and leaves
    // it zeroed, which is a store in a kernel that visits those elements anyway rather than a 16 KiB
    // memset of its own. See the comment above hc_expand_reg_kernel.

    // ---- the shared expert is ISSUED FIRST and collected last ----------------------------------
    //
    // It depends only on `norm`, exactly as the routed experts do, and neither depends on the other.
    // Whatever share of the routed set is not VRAM-resident runs on the CPU, so issuing the shared
    // expert before that and collecting it after is the only window in the sublayer where the card
    // and the host have genuinely independent work. Running it after the join spends that window
    // queueing.
    //
    // It matters more than its own cost: an idle dispatch path is expensive here (gpu/keepalive.h),
    // so keeping work in flight across the CPU share is worth more than the work itself.
    bool shexp_fused = false, shexp_deferred = false;
    if (w.shexp_gate && dops_.shexp && dops_.shexp_collect) {
      shexp_fused = dops_.shexp(dops_.ctx, w.shexp_gate.gpu, w.shexp_up.gpu, w.shexp_down.gpu,
                                dev_hc ? nullptr : norm.data(), E, cfg_.moe_inter,
                                cfg_.swiglu_limit, dev_hc ? nullptr : sh.data(), /*defer=*/true);
      shexp_deferred = shexp_fused;
    }

    // One engine. The routed experts run on the host kernel, out of the container's own mapping,
    // and the contribution lands in the host `block` like every other host-side term.
    for (uint32_t k = 0; k < (routed_experts ? n_sel : 0u); ++k) {
      const uint32_t e = sel[k];
      const ExpertMatrixView mg = fetch ? fetch(fetch_ctx, l, e, 0) : expert_view(l, e, 0);
      const ExpertMatrixView mu = fetch ? fetch(fetch_ctx, l, e, 1) : expert_view(l, e, 1);
      const ExpertMatrixView md = fetch ? fetch(fetch_ctx, l, e, 2) : expert_view(l, e, 2);
      std::fill(gate.begin(), gate.end(), 0.0f);
      std::fill(up.begin(), up.end(), 0.0f);
      expert_gemv_scaled_mt(mg, cb_, norm.data(), 1.0f, gate.data());
      expert_gemv_scaled_mt(mu, cb_, norm.data(), 1.0f, up.data());
      swiglu(gate.data(), up.data(), cfg_.moe_inter, cfg_.swiglu_limit, act.data());
      expert_gemv_scaled_mt(md, cb_, act.data(), sw[k], block.data());
    }

    AFF_PH(ffn_routed);
    if (w.shexp_gate) {                            // shared expert: dense, on every token
      // Collect the chain issued above. Under dev_hc it accumulated straight into the device block
      // and there is nothing to bring back; otherwise `sh` lands from the pinned buffer here.
      if (shexp_deferred) {
        dops_.shexp_collect(dops_.ctx, dev_hc ? nullptr : sh.data(), E);
      } else {
        // One device-resident chain when the GPU can take it: gate, up, SwiGLU and down for a
        // single round trip. The unfused form below is three round trips plus a host SwiGLU, and
        // those trips cost more than the arithmetic between them.
        shexp_fused = dops_.shexp &&
            dops_.shexp(dops_.ctx, w.shexp_gate.gpu, w.shexp_up.gpu, w.shexp_down.gpu,
                        dev_hc ? nullptr : norm.data(), E, cfg_.moe_inter, cfg_.swiglu_limit,
                        dev_hc ? nullptr : sh.data(), /*defer=*/false);
        if (!shexp_fused) {
          norm_ready();
          dense_mv(w.shexp_gate, norm.data(), cfg_.moe_inter, E, gate.data());
          dense_mv(w.shexp_up, norm.data(), cfg_.moe_inter, E, up.data());
          swiglu(gate.data(), up.data(), cfg_.moe_inter, cfg_.swiglu_limit, act.data());
          dense_mv(w.shexp_down, act.data(), E, cfg_.moe_inter, sh.data());
        }
      }
      if (!(dev_hc && shexp_fused)) for (uint32_t d = 0; d < E; ++d) block[d] += sh[d];
    }
    // The routed experts always run on the host here, so the block always carries something.
    if (dev_hc) dops_.block_add(dops_.ctx, block.data(), E);
    AFF_PH(ffn_dense);

    if (dev_hc) {
      // Every `down` in this sublayer — the shared expert's and every VRAM-resident routed
      // expert's — is split along its input, so every card's block is a partial. One exchange folds
      // them together with the host tier's contribution, which block_add put on rank 0 alone.
      if (dops_.block_reduce) dops_.block_reduce(dops_.ctx, E);
      dops_.hc_post(dops_.ctx, w.hc_ffn_fn.data != nullptr, E, HC);
    } else if (w.hc_ffn_fn) {
      hc_expand(hc_next.data(), block.data(), s->hc.data(), hcc.post, hcc.comb, E, HC);
      s->hc.swap(hc_next);
    } else {
      for (uint32_t d = 0; d < E; ++d) s->hc[d] += block[d];
    }
    AFF_PH(hyper);
  }

  ++s->pos;                                        // one position per TOKEN, not per layer

  // The head still collapses the lanes on the host, so read them back — ONCE per token, not per
  // layer, which is the whole point of keeping them on device in between.
  if (dev_hc) dops_.hc_read(dops_.ctx, s->hc.data(), E, HC);
  // ---- head: collapse the HC lanes, norm, project to the vocabulary --------------------------
  std::vector<float> embd(E);
  if (cfg_.v41) {
    // `Transformer.forward`: h = layer.hc_pre(h, pre_mix) with the mix the last layer's FFN
    // returned, then the final norm. Nothing recomputes a head mix.
    if (!s->last_ffn_pre_valid) {
      aff::ui::fatal("fatal: v4.1 head has no lane mix — the last layer's FFN did not run on the "
                     "host path. The device epilogue is not ported yet.\n");
      std::abort();
    }
    hc_reduce(s->hc.data(), s->last_ffn_pre, E, HC, embd.data());
  } else if (hc_head_fn_ && hc_head_scale_ && hc_head_base_) {
    std::vector<float> flat((size_t)HC * E), pre(HC);
    rms_norm_noweight(s->hc.data(), (uint64_t)HC * E, cfg_.rms_eps, flat.data());
    matvec_bf16((const uint16_t*)hc_head_fn_.data, flat.data(), HC, (uint64_t)HC * E, pre.data());
    for (uint32_t i = 0; i < HC; ++i)
      pre[i] = sigmoid(pre[i] * hc_head_scale_[0] + hc_head_base_[i]) + cfg_.hc_eps;
    hc_reduce(s->hc.data(), pre.data(), E, HC, embd.data());
  } else {
    std::memcpy(embd.data(), s->hc.data(), E * 4);
  }
  rms_norm(embd.data(), out_norm_, E, cfg_.rms_eps, norm.data());

  logits->assign(cfg_.vocab, 0.0f);
  if (head_) dense_mv(head_, norm.data(), cfg_.vocab, E, logits->data());
  AFF_PH(head);
  if (prof_on) ++prof_.tokens;
}

// ------------------------------------------------------------------------------------------
// Batched prefill
// ------------------------------------------------------------------------------------------
//
// The same layer schedule as forward_token, run over a chunk of tokens at once. Read the two side
// by side: every step below is the plural of a step above, with three exceptions, and the
// exceptions are the whole reason this is not simply a wider forward_token.
//
//   * The KV ring, the compressor window and attention's causal mask are ORDERED inside the chunk.
//     Token b must not see a row token b+1 wrote, so those three stay a loop over b. They cost no
//     synchronise: the commit and the attention are kernel launches on one stream and the host work
//     between them (the compressor pool, the indexer's top-k) is CPU-only.
//   * `norm` comes back to the host once per sublayer instead of once per token, because the routed
//     expert dispatch still runs there.
//   * The vocabulary head runs for the LAST token only. Prefill otherwise computes a full logit
//     vector for every token in the chunk and throws all but the last away.
//
// Everything else — the hyper-connection control, all six projections that read `norm`, the router
// gate, the shared expert, the output projection — is `nb` independent problems and becomes a GEMM.
bool Model::forward_prefill(const uint32_t* ids, uint32_t n, SeqState* s,
                            ExpertFetch fetch, void* fetch_ctx,
                            std::vector<float>* logits, bool routed_experts,
                            BlockOut* blk) const {
  if (!bops_.cap || !bops_.begin || !embed_ || !n) return false;
  const uint32_t cap = bops_.cap(bops_.ctx);
  if (cap < 2) return false;
  // A block is ONE chunk by construction: its head runs for every position, and across two chunks
  // the first chunk's would be overwritten. Six tokens against an arena sized in the hundreds, so
  // this is a caller bug rather than a configuration.
  if (blk && !bops_.hc_tap) return false;
  if (blk && blk->per_position_head && (n > cap || !bops_.head || head_.gpu < 0)) return false;
  // One check for the whole run rather than a refusal in the middle of a chunk: a per-layer op that
  // gave up halfway would leave the KV cache holding a partial chunk with no way to unwind it.
  for (uint32_t l = 0; l < cfg_.n_layer; ++l) {
    const LayerWeights& w = layers_[l];
    if (!w.wq_a || !w.wq_b || !w.wkv || !w.wo_a || !w.wo_b || !w.attn_sinks) return false;
    if (w.wq_a.gpu < 0 || w.wq_b.gpu < 0 || w.wkv.gpu < 0 || w.wo_a.gpu < 0 || w.wo_b.gpu < 0)
      return false;
  }

  const bool prof_on = profiling();
  AFF_PH_BEGIN();
  const uint32_t E = cfg_.n_embd, NH = cfg_.n_head, HC = cfg_.hc_mult;
  const AttnConfig ac = cfg_.attn();
  const uint32_t W = ac.qk_width(), ROT = cfg_.rope_dim;
  const uint32_t QR = cfg_.q_lora_rank, OR = cfg_.o_lora_rank;
  const uint32_t IHD = cfg_.index_head_dim, INH = cfg_.index_n_heads;
  const uint32_t NE = cfg_.n_expert;

  // `min(cap, n)`, not `cap`. Everything below is sized by this, and a chunk that never fills is a
  // per-CALL allocation of the difference — a six-token speculative block would otherwise pay for a
  // whole cap's worth of value-initialised host buffers every block. The chunking is unchanged for
  // prefill, since a prompt shorter than the cap was already one chunk, so this moves nothing about
  // what runs, only what is reserved to run it.
  const uint32_t chunk = std::min(cap, n);
  if (chunk < 2 && n > 1) return false;

  std::vector<float> emb((size_t)chunk * E);
  std::vector<float> normb((size_t)chunk * E);       // the FFN sublayer's host expert tier, lazily
  std::vector<float> blockh((size_t)chunk * E), rlogb((size_t)chunk * NE);
  // The chunk's whole routing, [live][n_expert_used], so the batched pass can group by expert.
  std::vector<uint32_t> selb, nselb;
  std::vector<float> swb;
  std::vector<uint8_t> handled;
  std::vector<float> iqb((size_t)chunk * INH * IHD);
  // Per-token compressed-row counts for a sub-batch's worth of attention.
  std::vector<uint32_t> ncompb;
  std::vector<uint8_t> residentb;         // AFF_FORCE_RESIDENT's mask, rebuilt per layer
  std::vector<uint32_t> sel(cfg_.n_expert_used);
  std::vector<float> sw(cfg_.n_expert_used);

  // Which slot of the tap plane each layer writes, or -1. Resolved once rather than searched per
  // layer: dspark_taps is three entries and the loop is 43 layers a chunk.
  const uint32_t NT = (uint32_t)cfg_.dspark_taps.size();
  std::vector<int32_t> tap_at;
  uint32_t tap_pos0 = 0, tap_rows = 0;
  if (blk) {
    blk->greedy.assign(blk->per_position_head ? n : 0u, 0u);
    // Sized here rather than by the caller: `n` is this call's token count and the caller knows only
    // what it asked for. `query` defaults to "nothing proposed", which is right for the last
    // position of a block and for a prefill chunk.
    if (blk->sample) {
      blk->draws.assign(n, PosDraw{});
      if (blk->query.size() < n) blk->query.resize(n, 0xFFFFFFFFu);
      if (blk->uniforms.size() < (size_t)2 * n) blk->uniforms.resize((size_t)2 * n, 0.5f);
    }
    blk->tap_rows = 0;
    if (NT && dspark_.ready()) {
      tap_at.assign(cfg_.n_layer, -1);
      // The tap fires at the END of a layer, so the plane holds that layer's OUTPUT. V4.1's draft
      // is conditioned on the attention INPUT of its target layers -- `Transformer.forward` appends
      // `h.mean(dim=2)` before calling the layer, and the reference says so in as many words -- and
      // the input of layer t is the output of t-1. Tapping t itself would hand the draft a hidden
      // state one layer further on than the one it was trained against, which costs acceptance
      // rather than correctness and is therefore invisible except as a draft nobody accepts.
      for (uint32_t i = 0; i < NT; ++i) {
        const int32_t t = cfg_.dspark_taps[i];
        const int32_t at = cfg_.v41 ? t - 1 : t;
        if (at >= 0 && (uint32_t)at < cfg_.n_layer) tap_at[(uint32_t)at] = (int32_t)i;
      }
      tap_rows = (blk->tap_keep && blk->tap_keep < n) ? blk->tap_keep : n;
      tap_pos0 = s->pos + (n - tap_rows);           // absolute position of the plane's row 0
      blk->tap_pos0 = tap_pos0;
      blk->tap_rows = tap_rows;
    }
  }
  // Index into the tap plane for the token at chunk offset `base + b`, or -1 when it fell out of the
  // trailing window the caller asked to keep.
  const uint32_t tap_first = tap_rows ? (n - tap_rows) : n;

  for (uint32_t base = 0; base < n; base += chunk) {
    const uint32_t live = std::min(chunk, n - base);
    // Before the chunk, not after: a display should name what is being worked on, and the last
    // chunk's completion is this function returning.
    if (pp_) pp_(pp_ctx_, base, n);
    const uint32_t pos0 = s->pos;
    for (uint32_t b = 0; b < live; ++b)
      bf16_dequant(embed_ + (size_t)ids[base + b] * E, E, emb.data() + (size_t)b * E);
    // Not `return false`: every reason this can refuse was checked once before the loop, and by
    // the time base > 0 the KV cache already holds the earlier chunks — the caller's per-token
    // retry would then re-append them. A refusal here is a bug, and the shape of the bug is a
    // prefill that quietly runs at decode speed.
    if (!bops_.begin(bops_.ctx, live, pos0, emb.data(), E, HC)) {
      aff::ui::fatal("fatal: batched prefill refused %u tokens at position %u\n", live, pos0);
      std::abort();
    }
    AFF_PH(hyper);

    for (uint32_t l = 0; l < cfg_.n_layer; ++l) {
      const LayerWeights& w = layers_[l];
      LayerState& st = s->layer[l];

      // ---- attention sublayer --------------------------------------------------------------
      // The indexer engages once the compressed cache outgrows index_topk. Within a chunk that can
      // become true partway through, so the test is against the count the LAST token will see;
      // asking for qr_norm and `norm` when no token needs them costs a copy per layer.
      // `is_index_source`, not `ratio == 4`: the two name the same set on V4, and only the first
      // names anything at all on V4.1. Guarded on w.ratio because the divisor below is the layer's.
      const bool idx_may_run = w.ratio && cfg_.is_index_source(l) && w.idx_wq_b &&
                               cfg_.index_topk < st.n_comp + live / w.ratio + 1u;
      // Null, not `normb`: neither host buffer this sublayer could ask for has a reader. The
      // compressors take `norm` on device and the indexer's query projection takes `qr_norm` there,
      // and nothing between here and the FFN sublayer looks at a host copy — the FFN sublayer
      // fetches `norm` for itself, lazily, and only when a host expert tier actually exists. Asking
      // is not a copy but a stage-out plus a drain, on every ratio-4 layer of the chunk.
      if (!bops_.hc_pre(bops_.ctx, w.hc_attn_fn.gpu, w.hc_attn_scale_gpu, w.hc_attn_base_gpu,
                        w.attn_norm_gpu, E, HC, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, cfg_.rms_eps,
                        nullptr))
        return false;
      AFF_PH(hyper);

      const RopeDerived rd = rope_derive(w.rope, ROT);
      if (!bops_.attn_qkv(bops_.ctx, w.wq_a.gpu, w.wq_b.gpu, w.q_norm_gpu, w.wkv.gpu, w.kv_norm_gpu,
                          E, QR, NH, W, ROT, rd, cfg_.rms_eps, nullptr))
        return false;
      AFF_PH(attn_proj);

      // ---- the compressors, on device, for the whole chunk ------------------------------------
      //
      // The pool is a window over eight positions and not an accumulator, so every emitted row is
      // independent and the whole chunk is two kernels — see gpu/compressor_gpu.h. What is left on
      // the host is the ROW COUNT, which is arithmetic: one row per `ratio` positions, and the
      // per-token prefix of it that attention needs. The ratio-128 HCA layers go the same way; they
      // simply have no indexer half, so their pool is one window instead of two.
      // Whether this layer BUILDS index keys — see the decode path for why that is a different
      // question from whether it indexes.
      const bool has_idx = cfg_.v41 ? (cfg_.is_kv_source(l) && w.idx_wk && w.idx_k_norm)
                                    : (w.ratio == 4 && w.idx_comp_wkv && w.idx_comp_wgate);
      const bool dev_comp = bops_.compress && w.ratio && w.comp_wkv && (w.comp_wgate || cfg_.v41) &&
                            cfg_.is_kv_source(l);
      uint32_t n_out = 0;
      if (w.ratio) {
        const uint64_t cap = st.comp_rows;
        for (uint32_t b = 0; b < live; ++b)
          if ((pos0 + b + 1) % w.ratio == 0 && st.n_comp + n_out + 1 <= cap) ++n_out;
      }
      if (dev_comp) {
        CompressArgs ca;
        ca.layer = l; ca.n = live; ca.pos0 = pos0; ca.ratio = w.ratio; ca.n_out = n_out;
        ca.h_kv = w.comp_wkv.gpu; ca.h_sc = w.comp_wgate.gpu;
        if (has_idx && cfg_.v41) {
          // See the decode path: V4.1's keys are one projection off the latent, and the head
          // weights move to indexer_q, which runs on the index sources rather than on these.
          ca.h_idx_wk = w.idx_wk.gpu; ca.idx_k_norm = w.idx_k_norm;
        } else if (has_idx) {
          ca.h_ikv = w.idx_comp_wkv.gpu; ca.h_isc = w.idx_comp_wgate.gpu;
        }
        ca.h_iproj =
            (!cfg_.v41 && has_idx && idx_may_run && w.idx_proj) ? w.idx_proj.gpu : -1;
        ca.ape = w.comp_ape;   ca.norm = w.comp_norm;
        ca.iape = w.idx_comp_ape; ca.inorm = w.idx_comp_norm;
        ca.width = W; ca.idx_dim = IHD; ca.n_embd = E;
        // kCompressorRot, not ROT — see its definition.
        ca.n_rot = kCompressorRot; ca.rope = rope_derive(w.rope, kCompressorRot);
        ca.slot0 = st.n_comp; ca.idx_slot0 = st.n_idx_comp;
        ca.idx_heads = ca.h_iproj >= 0 ? INH : 0u;
        ca.iscale = 1.0f / std::sqrt((float)(IHD * INH));
        ca.eps = cfg_.rms_eps;
        if (!bops_.compress(bops_.ctx, ca)) return false;
      } else if (w.ratio && cfg_.is_kv_source(l)) {
        aff::ui::fatal("fatal: layer %u has a compressor and the device cannot run it\n", l);
        std::abort();
      }
      if (idx_may_run) {
        // The indexer's query projection reads qr_norm, not `norm` — a different activation, so a
        // second batch. It stays ON DEVICE: nothing on the host reads it, and draining it back only
        // to upload it again is a round trip per layer per chunk.
        const int32_t h = w.idx_wq_b.gpu;
        const uint64_t r = (uint64_t)INH * IHD;
        if (bops_.indexer_q) {
          // V4.1's head weights ride along here. This hook runs on exactly the index-source layers,
          // which is the set that needs them; V4's compressor above already issued its own, so it
          // passes -1 and this is the call it always was.
          const int32_t hp = (cfg_.v41 && w.idx_proj) ? w.idx_proj.gpu : -1;
          if (!bops_.indexer_q(bops_.ctx, h, r, QR, INH, IHD, hp, E,
                               1.0f / std::sqrt((float)(IHD * INH))))
            return false;
        } else {
          float* y = iqb.data();
          if (!bops_.mv_host(bops_.ctx, 1, &h, &r, QR, 1, &y)) return false;
        }
      }
      AFF_PH(compressor);

      // ---- the ordered part ------------------------------------------------------------------
      //
      // TWO PASSES, and the split is the point. What is genuinely sequential in position is the
      // compressor's pooling state and the indexer's top-k over it — both host, both cheap. The
      // raw KV ring and attention are NOT: token p's window is a function of p alone. So pass one
      // walks the chunk on the host, and pass two commits and attends a whole sub-batch per launch.
      //
      // Doing that needs the ring to be wider than the sliding window, or committing the sub-batch
      // would destroy rows its own early tokens still need — see attn_ring_slots().
      const float kq = 1.0f / std::sqrt((float)W);
      const uint32_t mstride = (uint32_t)st.comp_rows;
      ncompb.assign(live, 0u);
      // One launch for the whole chunk once pass one has walked it, rather than a host scoring pass
      // per token. The per-token counts go with it: the compressed cache grows inside a chunk, so
      // each token sees its own prefix of it.
      bool idx_run = false;
      uint32_t idx_keys = 0;
      const uint32_t n_comp_end = st.n_comp + n_out;   // what the device just emitted, in total
      const bool shares_kv = w.ratio && w.kv_owner != l;
      for (uint32_t b = 0; b < live; ++b) {
        const uint32_t pos = pos0 + b;
        {                                            // the host's own copy of the sliding window
          st.raw_head = (st.raw_head + 1) % cfg_.sliding;
          if (st.n_raw < cfg_.sliding) ++st.n_raw;
        }
        if (dev_comp) {
          // The rows themselves were emitted by `bops_.compress` above; all that is left is the
          // count, and the per-token prefix of it, because the cache grows inside the chunk.
          if ((pos + 1) % w.ratio == 0 && st.n_comp < n_comp_end) {
            ++st.n_comp;
            if (has_idx) ++st.n_idx_comp;
          }
        } else if (shares_kv && (pos + 1) % w.ratio == 0 &&
                   st.n_comp < s->layer[w.kv_owner].n_comp) {
          // V4.1's index keys are 1:1 with the compressed rows — one `wk` projection per emitted
          // latent — so a reader's key prefix is its row prefix. Without this the index-source
          // layers that are NOT kv sources (24, 28, 32, 36) would carry n_idx_comp == 0 and score
          // against an empty key set, which reads as "the indexer admitted nothing" and attends to
          // the sliding window alone.
          if (cfg_.v41) ++st.n_idx_comp;
          // A layer that READS another's cache still needs its own per-token prefix: the rows exist
          // (the source emitted them earlier in this same chunk, so its total is already final),
          // but this layer never ran a compressor to count them. Same growth rule, bounded by what
          // the owner actually holds.
          ++st.n_comp;
        }

        ncompb[b] = st.n_comp;
        // ---- the indexer, on device -----------------------------------------------------------
        //
        // What is left on the host here is the per-token count. The SCORING is O(INH*IHD*n_comp)
        // and it moved to the device, which also produces the admission mask straight into the
        // attention kernel's own buffer, so neither the scores nor the mask cross the bus.
        if (idx_may_run && st.n_comp) {
          if (st.n_comp > mstride) return false;      // the mask plane was sized from comp capacity
          // No rope_tail and no indexer_qat here: both moved to the device, where theta is computed
          // once per (token, dim) instead of once per (token, head, dim), and the transpose that
          // lands the head weights in VRAM folds the 1/sqrt(IHD*INH) in.
          idx_keys = std::min<uint32_t>(st.n_idx_comp, st.n_comp);
          idx_run = true;
        }
      }
      if (idx_run) {
        IndexArgs ia;
        ia.layer = l; ia.n = live; ia.n_comp = ncompb.data();
        ia.h_wq_b = w.idx_wq_b.gpu; ia.qr = QR;
        ia.n_head = INH; ia.dim = IHD;
        ia.n_keys = idx_keys; ia.topk = cfg_.index_topk;
        ia.pos0 = pos0; ia.n_rot = ROT; ia.rope = rope_derive(w.rope, ROT);
        // n_mask is derived from ncompb on the device side, which already walks it for the
        // per-token counts. Queries and head weights are in VRAM: indexer_q put them there.
        ia.hadamard = !cfg_.v41;      // see the decode path
        if (!bops_.indexer(bops_.ctx, ia)) return false;
      }
      // Splits the bucket: everything above is pass one, the sequential HOST walk, and everything
      // below is the batched device work. They were charged to one timer and read as "attention".
      AFF_PH(indexer);

      // Pass two: the ring, and attention. kAttnBatch tokens per launch, and the ring is wide
      // enough that committing all of them first cannot overwrite a row one of them still needs.
      for (uint32_t g0 = 0; g0 < live; g0 += kAttnBatch) {
        const uint32_t gn = std::min<uint32_t>(kAttnBatch, live - g0);
        if (!bops_.kv_commit(bops_.ctx, l, g0, gn, pos0 + g0, W, ROT)) return false;
        // Null `allowed`: the indexer left the admission mask on device, in the buffer this reads.
        if (!bops_.attend(bops_.ctx, l, g0, gn, pos0 + g0, nullptr, mstride, ncompb.data() + g0, kq))
          return false;
      }
      AFF_PH(attention);

      if (!bops_.attn_out(bops_.ctx, w.wo_a.gpu, w.wo_b.gpu, NH, W, ROT, rd, 8, OR, E)) return false;
      // See the single-token path: wo_b is Col, so every card holds a partial of the chunk's block.
      if (bops_.block_reduce && !bops_.block_reduce(bops_.ctx, E)) return false;
      if (!bops_.hc_post(bops_.ctx, w.hc_attn_fn.data != nullptr, E, HC)) return false;
      AFF_PH(attn_out);

      // ---- FFN sublayer ----------------------------------------------------------------------
      // `norm` comes back only when something on the host will read it. That is hash routing, which
      // weights by router_w . norm, and the expert tier for whatever did not fit in VRAM — and the
      // second of those is not known until after the dispatch, so it is fetched there.
      const bool hashed = cfg_.uses_hash_routing(l) && !w.hash_table.empty();
      const bool dev_route = !hashed && w.router_w && w.router_gpu >= 0 && bops_.router_topk &&
                             bops_.norm_fetch;
      // A hash layer's SELECTION is a table lookup and belongs on the host; its WEIGHTING is a
      // `router_w[e] . norm` dot per selected expert and does not. Left on the host it is the single
      // largest stall in the whole prefill — it needs the entire chunk's `norm` back and then runs
      // live*k dots single-threaded, with the card idle throughout.
      const bool dev_hash = hashed && routed_experts && w.router_w && w.router_gpu >= 0 &&
                            bops_.router_hash && bops_.norm_fetch;
      bool norm_live = !dev_route && !dev_hash;
      if (!bops_.hc_pre(bops_.ctx, w.hc_ffn_fn.gpu, w.hc_ffn_scale_gpu, w.hc_ffn_base_gpu,
                        w.ffn_norm_gpu, E, HC, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, cfg_.rms_eps,
                        norm_live ? normb.data() : nullptr))
        return false;
      if (!dev_route && !hashed && w.router_w && w.router_gpu >= 0) {
        if (!bops_.router(bops_.ctx, w.router_gpu, NE, E, rlogb.data())) return false;
      }
      AFF_PH(router);

      if (w.shexp_gate &&
          !bops_.shexp(bops_.ctx, w.shexp_gate.gpu, w.shexp_up.gpu, w.shexp_down.gpu, E,
                       cfg_.moe_inter, cfg_.swiglu_limit))
        return false;
      AFF_PH(ffn_dense);

      // Route every token BEFORE dispatching any of them: the batched pass groups by expert, which
      // it cannot do while the selections are still arriving one token at a time.
      const uint32_t KS = cfg_.n_expert_used;
      selb.resize((size_t)live * KS); swb.resize((size_t)live * KS);
      handled.resize((size_t)live * KS); nselb.assign(live, 0u);
      // On device when it can be: the logits are already there, and only the selection comes back
      // instead of the whole chunk's `norm`.
      //
      // Does anything on the host still want that selection? Under a device-built dispatch nothing
      // on the fast path does — the plan kernel reads the router's own device buffer and `handled`
      // comes back all-set — so the readback and the drain behind it are skipped. What does want it
      // are diagnostics, and each forces the readback rather than reading a stale buffer:
      // AFF_PROFILE's hybrid census walks the selection, and the host expert arm below needs
      // `nselb` to decide it has nothing to do.
      //
      // Asked ONCE for the layer, above both routing paths, because it is a property of the layer
      // and not of how its experts were chosen.
      const bool dev_plan = ddp_ && ddp_(ddp_ctx_, l) && !prof_on;
      if (dev_route && routed_experts) {
        const uint8_t* res = nullptr;
        if (force_resident_routing() && res_) {
          if (residentb.size() < (size_t)NE) residentb.resize(NE);
          for (uint32_t e = 0; e < NE; ++e) residentb[e] = res_(res_ctx_, l, e) ? 1 : 0;
          res = residentb.data();
        }
        if (!bops_.router_topk(bops_.ctx, w.router_gpu, NE, E, KS, w.router_b, res,
                               cfg_.norm_topk_prob ? 1 : 0, cfg_.routed_scaling,
                               dev_plan ? nullptr : selb.data(),
                               dev_plan ? nullptr : swb.data(),
                               dev_plan ? nullptr : nselb.data()))
          return false;
        // Stale otherwise, and `any_host` below reads it. All-set is the truth when the device
        // build ran — it takes every slot the router named — and it is also what makes the host
        // expert arm unreachable, which is the point: two engines that disagree make a lottery.
        if (dev_plan) std::fill(nselb.begin(), nselb.begin() + live, 0u);
      }
      // The hash layers' host half: a table lookup and the residency walk, both O(k) a token. What
      // is NOT here is the weighting — `bops_.router_hash` does that on the card out of the gate
      // GEMM's own logits, so `norm` never comes back.
      if (dev_hash) {
        for (uint32_t b = 0; b < live; ++b) {
          const int32_t* row = w.hash_table.data() + (size_t)ids[base + b] * cfg_.n_expert_used;
          uint32_t n_sel = cfg_.n_expert_used;
          for (uint32_t k = 0; k < n_sel; ++k) { sel[k] = (uint32_t)row[k]; sw[k] = 1.0f; }
          force_resident(l, nullptr, sel.data(), sw.data(), &n_sel);
          nselb[b] = n_sel;
          for (uint32_t j = 0; j < n_sel; ++j) selb[(size_t)b * KS + j] = sel[j];
        }
        // Null `swb` under a device-built dispatch, exactly as the top-k path above: the weights are
        // read only by the host expert arm, which that dispatch makes unreachable, and passing them
        // back costs a stream drain per hash layer.
        if (!bops_.router_hash(bops_.ctx, w.router_gpu, NE, E, KS, selb.data(), nselb.data(),
                               cfg_.norm_topk_prob ? 1 : 0, cfg_.routed_scaling,
                               dev_plan ? nullptr : swb.data()))
          return false;
      }
      for (uint32_t b = 0; b < live && routed_experts && !dev_route && !dev_hash; ++b) {
        const float* xb = normb.data() + (size_t)b * E;
        uint32_t n_sel = 0;
        if (hashed) {
          const int32_t* row = w.hash_table.data() + (size_t)ids[base + b] * cfg_.n_expert_used;
          for (uint32_t k = 0; k < cfg_.n_expert_used; ++k) { sel[k] = (uint32_t)row[k]; sw[k] = 1.0f; }
          n_sel = cfg_.n_expert_used;
          force_resident(l, nullptr, sel.data(), sw.data(), &n_sel);
          if (w.router_w) {                          // hash layers still weight by router probability
            float sum = 0.0f;
            for (uint32_t k = 0; k < n_sel; ++k) {
              const uint16_t* rw = w.router_w + (size_t)sel[k] * E;
              float acc = 0.0f;
              for (uint32_t d = 0; d < E; ++d) acc += bf16_to_float(rw[d]) * xb[d];
              sw[k] = sqrtsoftplus(acc);
              sum += sw[k];
            }
            if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
            if (cfg_.norm_topk_prob) for (uint32_t k = 0; k < n_sel; ++k) sw[k] /= sum;
            for (uint32_t k = 0; k < n_sel; ++k) sw[k] *= cfg_.routed_scaling;
          }
        } else if (w.router_w) {
          RouterOut ro;
          route_topk(rlogb.data() + (size_t)b * NE, w.router_b, NE, cfg_.n_expert_used,
                     cfg_.norm_topk_prob, cfg_.routed_scaling, &ro);
          n_sel = ro.n;
          for (uint32_t k = 0; k < n_sel; ++k) { sel[k] = ro.expert[k]; sw[k] = ro.weight[k]; }
          force_resident(l, rlogb.data() + (size_t)b * NE, sel.data(), sw.data(), &n_sel);
        }
        (void)xb;
        nselb[b] = n_sel;
        for (uint32_t j = 0; j < n_sel; ++j) {
          selb[(size_t)b * KS + j] = sel[j];
          swb[(size_t)b * KS + j] = sw[j];
        }
      }
      AFF_PH(route);

      // The batched pass first: it reads the chunk arena in VRAM and accumulates straight into the
      // device block, so it needs no host activation and returns no host contribution. What it
      // could not take -- experts on the far card, experts not resident at all -- it marks unhandled
      // and the per-token loop below picks up exactly those.
      std::fill(handled.begin(), handled.begin() + (size_t)live * KS, (uint8_t)0);
      if (ffnb_ && routed_experts)
        ffnb_(ffnb_ctx_, l, selb.data(), swb.data(), live, KS, handled.data());
      // `res_`, NOT `handled`. `handled` is the answer the dispatch itself gave, and the hybrid path
      // takes non-resident experts too, so it reports everything taken whatever the residency. The
      // H-index is a property of PLACEMENT and has to be asked of placement.
      if (prof_on && routed_experts && ffnb_ && res_) {
        HybridCensus& C = hybrid_census();
        static thread_local std::vector<uint8_t> seen, seenm;
        if (seen.size() != NE) { seen.assign(NE, 0u); seenm.assign(NE, 0u); }
        uint32_t nd = 0, ndm = 0;
        ++C.calls; C.tokens += live;
        for (uint32_t b = 0; b < live; ++b) {
          uint32_t h = 0;
          for (uint32_t j = 0; j < nselb[b]; ++j) {
            const size_t i = (size_t)b * KS + j;
            const uint32_t e = selb[i];
            const bool res = e < NE && res_(res_ctx_, l, e);
            ++C.slots;
            if (res) ++h; else ++C.slots_miss;
            if (e < NE && !seen[e]) { seen[e] = 1; ++nd; }
            if (e < NE && !res && !seenm[e]) { seenm[e] = 1; ++ndm; }
          }
          ++C.hits[h < 15 ? h : 15];
        }
        C.distinct += nd; C.distinct_miss += ndm;
        if (ndm > C.worst_miss) C.worst_miss = ndm;
        for (uint32_t b = 0; b < live; ++b)
          for (uint32_t j = 0; j < nselb[b]; ++j)
            if (selb[(size_t)b * KS + j] < NE) seen[selb[(size_t)b * KS + j]] = seenm[selb[(size_t)b * KS + j]] = 0;
      }
      AFF_PH(dispatch);

      // Whatever the batched pass could not take runs on the host or over the bus, and both read
      // `norm`. Only now is it known whether anything will, so only now is it fetched — and when
      // NOTHING is left, none of the rest of this happens either.
      //
      // block_add is the reason that matters: it memsets a [nb][n_embd] arena, copies live*n_embd
      // floats into it out of PAGEABLE memory, transposes, adds, and then DRAINS because `src` is a
      // caller local. With every expert resident every byte of that upload is zero.
      bool any_host = false;
      for (size_t i = 0; i < (size_t)live * KS && !any_host; ++i)
        any_host = routed_experts && (i % KS) < nselb[i / KS] && !handled[i];
      // FATAL, not a fallback. With `ffnb_` wired the dispatch has two homes for an expert's bytes
      // — this card's slab, or the registered host pool the tile points at — and a pair it declined
      // has neither, so there is nothing left to compute it from. Running it on the host instead is
      // two engines computing the same expert: the AVX-512 kernel and the expert GEMM sum 4096
      // terms in different orders, so which one took an expert moves the logits in the last bits,
      // and 43 layers of argmax turn that into a different token from the same binary and prompt.
      //
      // The only cause is an expert on the SSD TIER, where `pool_shard_dev` returns null. Under
      // static placement that tier is a budget leftover — whatever did not fit in RAM, which nothing
      // is expected to route to — so reaching here means the RAM budget was set too low, and
      // stopping is the honest answer.
      //
      // THIS BECOMES WRONG UNDER THE PLACEMENT ENGINE, which populates that tier on purpose with
      // the experts it predicts are least likely to be routed to. Being routed to one then stops
      // being a configuration error and becomes an expected, priced event: stream the expert in
      // from disk and discard it, no admission. That path does not exist yet — it wants a third
      // membership pass in run_batch_device over a pinned staging ring, NOT a host compute arm.
      // Until it exists, this abort is what keeps the tier from being entered silently.
      if (any_host && ffnb_) {
        aff::ui::fatal("fatal: layer %u left routed (token, expert) pairs unhandled — they are "
                     "on the ssd tier and the batched dispatch cannot stream from disk yet. "
                     "Raise --host-pool-mib.\n", l);
        std::abort();
      }
      if (any_host) {
        // No batched dispatch at all: this is the CPU-only build, and the routed experts run on the
        // same host kernel forward_token uses. One engine, so no divergence to have.
        if (!norm_live) {
          if (!bops_.norm_fetch(bops_.ctx, normb.data())) return false;
          norm_live = true;
        }
        std::fill(blockh.begin(), blockh.begin() + (size_t)live * E, 0.0f);
        std::vector<float> hg(cfg_.moe_inter), hu(cfg_.moe_inter), ha(cfg_.moe_inter);
        for (uint32_t b = 0; b < live; ++b) {
          const float* xb = normb.data() + (size_t)b * E;
          float* ob = blockh.data() + (size_t)b * E;
          for (uint32_t j = 0; j < nselb[b]; ++j) {
            if (handled[(size_t)b * KS + j]) continue;
            const uint32_t e = selb[(size_t)b * KS + j];
            const ExpertMatrixView mg = fetch ? fetch(fetch_ctx, l, e, 0) : expert_view(l, e, 0);
            const ExpertMatrixView mu = fetch ? fetch(fetch_ctx, l, e, 1) : expert_view(l, e, 1);
            const ExpertMatrixView md = fetch ? fetch(fetch_ctx, l, e, 2) : expert_view(l, e, 2);
            std::fill(hg.begin(), hg.end(), 0.0f);
            std::fill(hu.begin(), hu.end(), 0.0f);
            expert_gemv_scaled_mt(mg, cb_, xb, 1.0f, hg.data());
            expert_gemv_scaled_mt(mu, cb_, xb, 1.0f, hu.data());
            swiglu(hg.data(), hu.data(), cfg_.moe_inter, cfg_.swiglu_limit, ha.data());
            expert_gemv_scaled_mt(md, cb_, ha.data(), swb[(size_t)b * KS + j], ob);
          }
        }
        if (!bops_.block_add(bops_.ctx, blockh.data(), E)) return false;
      }
      AFF_PH(ffn_routed);

      // Every `down` in this sublayer is Col, so every card's block is a partial; block_add put the
      // host tier's whole contribution on rank 0 alone so this sums it in exactly once.
      if (bops_.block_reduce && !bops_.block_reduce(bops_.ctx, E)) return false;
      if (!bops_.hc_post(bops_.ctx, w.hc_ffn_fn.data != nullptr, E, HC)) return false;

      // ---- the DSpark tap --------------------------------------------------------------------
      // `main_hiddens.append(h.mean(dim=2))` in the reference, taken AFTER the layer's second
      // hc_post — the layer's output, not either sublayer's — and concatenated in tap order.
      //
      // Straight into the card's own tap plane: the windowing is the kernel's addressing rather than
      // a per-token memcpy out of a staging buffer, and `b_lo`/`row0` are what select the trailing
      // window inside this chunk.
      if (!tap_at.empty() && tap_at[l] >= 0 && base + live > tap_first) {
        const uint32_t b_lo = base >= tap_first ? 0u : tap_first - base;
        if (!bops_.hc_tap(bops_.ctx, (uint32_t)tap_at[l], NT, b_lo, live - b_lo,
                          base + b_lo - tap_first, tap_rows, E, HC))
          return false;
      }
      AFF_PH(hyper);
    }

    s->pos += live;
    if (prof_on) prof_.tokens += live;

    // ---- the head ----------------------------------------------------------------------------
    // A block wants every position's winner; prefill wants the last position's logits. The collapse
    // and the norm are the same host arithmetic either way — 16384 floats a token — and only the
    // vocabulary projection differs, because 529 MB a card read six times is not the same as read
    // once over six columns.
    if (blk && blk->per_position_head) {
      // ---- ON THE CARD, for every position at once ---------------------------------------------
      //
      // What this replaces ran per position on the host: hc_read (a stream drain and 64 KiB down),
      // an rms_norm over HC*E floats, a [HC][HC*E] matvec, a sigmoid, the lane reduce and a second
      // rms_norm — then `live * n_embd` floats back up for the vocabulary GEMM. Six positions a
      // block, so six round trips for arithmetic over a state that was already in VRAM and whose
      // result was going straight back to it.
      //
      // The kernels are the draft's, unmodified: `draft_collapse` is exactly this epilogue, and it
      // has been the draft's live path all along. The handles differ, nothing else does.
      //
      // NOT bit-identical to the host epilogue: rms_norm_noweight sums 16384 squares serially and
      // the kernel sums them as a tree, so the norm's last bits differ. This is the last arithmetic
      // before the vocabulary head rather than something 43 layers of routing amplify, so the argmax
      // absorbs it — but that is a property of the input, not a guarantee, and a change to what
      // feeds this has to be re-checked against the host path rather than assumed.
      if (cfg_.v41) {
        // V4.1's head has no mixer: it collapses with what the last layer's FFN produced, which is
        // still the device's `b_pre` — nothing between that hc_pre and here writes it.
        if (!bops_.collapse_pre || out_norm_gpu_ < 0) {
          aff::ui::fatal("fatal: the v4.1 head epilogue needs collapse_pre and the output norm on "
                         "the card (norm handle %d)\n", out_norm_gpu_);
          std::abort();
        }
        if (!bops_.collapse_pre(bops_.ctx, out_norm_gpu_, E, HC, cfg_.hc_eps, cfg_.rms_eps)) {
          ui::err("block head: the v4.1 lane collapse refused\n");
          return false;
        }
        s->hc_host = false;
      } else {
      if (!bops_.collapse || hc_head_fn_.gpu < 0 || hc_head_scale_gpu_ < 0 ||
          hc_head_base_gpu_ < 0 || out_norm_gpu_ < 0) {
        aff::ui::fatal("fatal: the block head's epilogue runs on the card and one of its four "
                     "handles is missing (fn %d, scale %d, base %d, norm %d). Refusing rather than "
                     "running the host copy: they round differently and 43 layers of argmax turn "
                     "that into different text.\n",
                     hc_head_fn_.gpu, hc_head_scale_gpu_, hc_head_base_gpu_, out_norm_gpu_);
        std::abort();
      }
      if (!bops_.collapse(bops_.ctx, hc_head_fn_.gpu, hc_head_scale_gpu_, hc_head_base_gpu_,
                          out_norm_gpu_, E, HC, cfg_.hc_eps, cfg_.rms_eps)) {
        ui::err("block head: the lane collapse refused\n");
        return false;
      }
      // The host lanes are no longer maintained across a block; see SeqState::hc_host.
      s->hc_host = false;
      }
      // Sampling and the argmax are the same head; only what comes back differs. A caller that
      // asked to sample and reached a build without the hook gets a refusal rather than a silent
      // greedy turn, because those are different models. `nullptr` for the activation: it is
      // already on the card, in both layouts, and neither head needs to stage it.
      if (blk->sample) {
        if (!bops_.head_sample) { ui::err("block head: no sampling hook in this build\n"); return false; }
        if (!bops_.head_sample(bops_.ctx, head_.gpu, cfg_.vocab, E, live, nullptr,
                               blk->query.data() + base, blk->uniforms.data() + 2 * base,
                               blk->temperature, blk->top_p, blk->top_k, blk->min_p,
                               blk->draws.data() + base)) {
          ui::err("block head: the sampling head refused %u positions at %u\n", live, base);
          return false;
        }
      } else if (!bops_.head(bops_.ctx, head_.gpu, cfg_.vocab, E, live, nullptr,
                             blk->greedy.data() + base)) {
        ui::err("block head: the argmax head refused %u positions at %u\n", live, base);
        return false;
      }
      AFF_PH(head);
    } else if (logits && base + live >= n) {
      // ---- THE LAST CHUNK'S LOGITS, AND ONLY IF SOMEONE ASKED FOR THEM ---------------------------
      //
      // `logits` is an optional out-parameter, and a caller can take neither the per-position head
      // above nor a logits pointer: the prefix cache splits a prefill in two so a checkpoint can be
      // published at a block boundary, and the first half's logits are nobody's — the second half
      // produces the ones that matter. Without the null check that is a dereference of nothing.
      //
      // Skipping it is not just a guard, it is the right thing: what follows is `hc_read` — which
      // DRAINS — plus a host rms_norm and a host `dense_mv` over the whole 129280 x 4096 head. That
      // is the epilogue the block path moved onto the card precisely because it is expensive. Doing
      // it for a caller who discards the result was pure cost.
      if (!bops_.hc_read(bops_.ctx, live - 1, s->hc.data(), E, HC)) return false;
      std::vector<float> embd(E), norm(E);
      if (hc_head_fn_ && hc_head_scale_ && hc_head_base_) {
        std::vector<float> flat((size_t)HC * E), pre(HC);
        rms_norm_noweight(s->hc.data(), (uint64_t)HC * E, cfg_.rms_eps, flat.data());
        matvec_bf16((const uint16_t*)hc_head_fn_.data, flat.data(), HC, (uint64_t)HC * E,
                    pre.data());
        for (uint32_t i = 0; i < HC; ++i)
          pre[i] = sigmoid(pre[i] * hc_head_scale_[0] + hc_head_base_[i]) + cfg_.hc_eps;
        hc_reduce(s->hc.data(), pre.data(), E, HC, embd.data());
      } else {
        std::memcpy(embd.data(), s->hc.data(), E * 4);
      }
      rms_norm(embd.data(), out_norm_, E, cfg_.rms_eps, norm.data());
      logits->assign(cfg_.vocab, 0.0f);
      if (head_) dense_mv(head_, norm.data(), cfg_.vocab, E, logits->data());
      AFF_PH(head);
    }
  }
  (void)fetch; (void)fetch_ctx;
  return true;
}

// ------------------------------------------------------------------------------------------
// The DSpark draft
// ------------------------------------------------------------------------------------------
//
// Read this next to `forward_prefill`: a stage IS a layer, and every step below that has a
// counterpart there is the same call with the stage's handles. Four things differ, and they are the
// whole of what makes it a draft rather than three more layers of the model.
//
//   * The RESIDUAL STREAM carries almost nothing. Every position is seeded with the embedding of
//     the noise token, except position 0, which gets the one token the target has already decided.
//     The draft is not continuing the model's hidden state; it is re-deriving a continuation from
//     the target's own summary of it.
//   * That summary — `main_x` — enters through the KEYS. Each stage projects it with its own wkv
//     into its own sliding-window ring, one row per real position, and the draft's queries attend
//     to it. So the conditioning is 128 positions of history, not one vector.
//   * The window is FLAT and not causal: the block's tokens are predicted jointly, so every one of
//     them sees all the others. See mqa_attend_batch_hip.
//   * The block is made autoregressive afterwards, by the Markov head, on the logits rather than in
//     the hidden state. That is what buys `block` tokens for one pass instead of `block` passes.
// The tap is already in VRAM on every card — batch_hc_tap put it there as the target ran — so this
// hands the projection a ROW INDEX rather than a pointer, and nothing crosses the bus.
bool Model::dspark_seed(uint32_t n, uint32_t pos0) const {
  const DsparkWeights& D = dspark_;
  if (!D.ready() || !n) return false;
  if (!bops_.draft_proj || !bops_.draft_kv) return false;
  const uint32_t E = cfg_.n_embd, W = cfg_.attn().qk_width(), ROT = cfg_.rope_dim;
  const uint32_t NT = (uint32_t)cfg_.dspark_taps.size(), NS = (uint32_t)D.stage.size();
  const bool prof_on = profiling();
  AFF_PH_BEGIN();
  // In kNarrowTok-sized runs, because that is how wide the draft's planes are. Steady state is one
  // run of at most block+1; the prompt's tail is `sliding` positions and takes eight of them, once.
  for (uint32_t o = 0; o < n; o += kSpecTok) {
    const uint32_t nn = std::min<uint32_t>(kSpecTok, n - o);
    if (!bops_.draft_proj(bops_.ctx, D.main_proj.gpu, D.main_norm_gpu, E, (uint64_t)E * NT, nn, o,
                          cfg_.rms_eps))
      return false;
    for (uint32_t st = 0; st < NS; ++st) {
      const LayerWeights& w = D.stage[st];
      if (!bops_.draft_kv(bops_.ctx, cfg_.n_layer + st, w.wkv.gpu, w.kv_norm_gpu, nn, pos0 + o, E,
                          W, ROT, rope_derive(w.rope, ROT), cfg_.rms_eps))
        return false;
    }
  }
  AFF_PH(draft_kv);
  return true;
}

bool Model::dspark_draft(uint32_t first_id, SeqState* s, uint32_t* out) const {
  const DsparkWeights& D = dspark_;
  if (!D.ready() || !s || !out || !embed_) return false;
  if (!bops_.begin || !bops_.draft_attend || !bops_.draft_head || !bops_.draft_collapse ||
      !drops_.ffn_batch)
    return false;
  const uint32_t E = cfg_.n_embd, NH = cfg_.n_head, HC = cfg_.hc_mult;
  const AttnConfig ac = cfg_.attn();
  const uint32_t W = ac.qk_width(), ROT = cfg_.rope_dim;
  const uint32_t QR = cfg_.q_lora_rank, OR = cfg_.o_lora_rank;
  // The draft's own pool, which V4.1 sizes below the target's (128 of 384, 3 active of 6). V4's
  // draft shares the target's counts, and dspark_n_expert is 0 there, so this is the same value.
  const uint32_t NE = cfg_.dspark_n_expert ? cfg_.dspark_n_expert : cfg_.n_expert;
  const uint32_t KS = cfg_.dspark_n_expert_used ? cfg_.dspark_n_expert_used : cfg_.n_expert_used;
  const uint32_t NS = (uint32_t)D.stage.size(), B = cfg_.dspark_block;
  const uint32_t P = s->pos;                       // the block's first position
  if (!P || !B) return false;
  const bool prof_on = profiling();
  AFF_PH_BEGIN();

  // ---- the block's own input -----------------------------------------------------------------
  //
  // Positions 1..B-1 are ALL the noise token, every block, for the life of the run — that is what
  // the draft was trained on and dspark_block_size does not vary. So they are dequantised once and
  // only row 0, the token the target just committed, is rebuilt. The buffer is a member for the
  // same reason: it is what `begin` uploads, and a caller-local one is why that upload has to drain.
  if (dspark_emb_.size() != (size_t)B * E) {
    dspark_emb_.assign((size_t)B * E, 0.0f);
    for (uint32_t b = 1; b < B; ++b)
      bf16_dequant(embed_ + (size_t)cfg_.dspark_noise_token * E, E,
                   dspark_emb_.data() + (size_t)b * E);
  }
  bf16_dequant(embed_ + (size_t)first_id * E, E, dspark_emb_.data());
  if (const char* e = std::getenv("AFF_SLOT_DEBUG")) { (void)e;
    uint64_t v = 1469598103934665603ull;
    for (size_t i = 0; i < (size_t)E; ++i) { uint32_t bits; std::memcpy(&bits, &dspark_emb_[i], 4); v ^= bits; v *= 1099511628211ull; }
    aff::ui::err("slot-hash dspark_emb row0 first_id=%u P=%u hash=%016llx\n",
                 first_id, P, (unsigned long long)v);
  }
  if (!bops_.begin(bops_.ctx, B, P, dspark_emb_.data(), E, HC)) return false;
  AFF_PH(draft_seed);

  // The key set, computed once: the target's last `sliding` real positions plus the block itself.
  // `P - 1` is the last position the ring has a main row for, so the history is [lo, P-1] and the
  // block adds [P, P+B-1] — contiguous, which is what lets it be one slot run.
  const uint32_t hist = std::min(cfg_.sliding, P);
  const uint32_t win_lo = P - hist;
  const float kq = 1.0f / std::sqrt((float)W);

  std::vector<uint32_t> selb((size_t)B * KS), nselb(B);
  std::vector<float> swb((size_t)B * KS);
  std::vector<uint8_t> handled((size_t)B * KS);

  for (uint32_t st = 0; st < NS; ++st) {
    const LayerWeights& w = D.stage[st];
    const uint32_t layer = cfg_.n_layer + st;
    const RopeDerived rd = rope_derive(w.rope, ROT);

    // ---- attention -------------------------------------------------------------------------
    if (!bops_.hc_pre(bops_.ctx, w.hc_attn_fn.gpu, w.hc_attn_scale_gpu, w.hc_attn_base_gpu,
                      w.attn_norm_gpu, E, HC, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, cfg_.rms_eps,
                      nullptr))
      return false;
    if (!bops_.attn_qkv(bops_.ctx, w.wq_a.gpu, w.wq_b.gpu, w.q_norm_gpu, w.wkv.gpu, w.kv_norm_gpu,
                        E, QR, NH, W, ROT, rd, cfg_.rms_eps, nullptr))
      return false;
    // The block's own KV, at its own positions. It overwrites whatever the LAST block drafted
    // there, which is exactly right: those positions were either accepted — and then the loop
    // above has just written the target's real row over them — or rejected, and stale.
    if (!bops_.kv_commit(bops_.ctx, layer, 0, B, P, W, ROT)) return false;
    if (!bops_.draft_attend(bops_.ctx, layer, B, P, win_lo, kq)) return false;
    if (!bops_.attn_out(bops_.ctx, w.wo_a.gpu, w.wo_b.gpu, NH, W, ROT, rd, 8, OR, E)) return false;
    if (bops_.block_reduce && !bops_.block_reduce(bops_.ctx, E)) return false;
    if (!bops_.hc_post(bops_.ctx, w.hc_attn_fn.data != nullptr, E, HC)) return false;
    AFF_PH(draft_attn);

    // ---- FFN -------------------------------------------------------------------------------
    // Stage `st` is layer n_layer + st, which is past n_hash_layer, so it routes on the gate like
    // any ordinary layer. No residency mask: the draft's experts are all resident or the run does
    // not start.
    if (!bops_.hc_pre(bops_.ctx, w.hc_ffn_fn.gpu, w.hc_ffn_scale_gpu, w.hc_ffn_base_gpu,
                      w.ffn_norm_gpu, E, HC, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, cfg_.rms_eps,
                      nullptr))
      return false;
    if (w.shexp_gate &&
        !bops_.shexp(bops_.ctx, w.shexp_gate.gpu, w.shexp_up.gpu, w.shexp_down.gpu, E,
                     cfg_.moe_inter, cfg_.swiglu_limit))
      return false;
    // The draft's dispatch is built on the card for exactly the reason the target's is, and the
    // predicate has to be the DRAFT placement's: the two latch on their own first dispatch, and the
    // draft's is 43 layers behind the target's. Asking the target's would skip a readback the
    // draft's dispatch still wanted, one block before the draft's own table was live.
    //
    // No `prof_on` term, unlike the target's: the hybrid census walks the target's selection and
    // does not reach here. The draft has no hybrid tier to take a census of — an expert it cannot
    // address is refused at load.
    const bool dev_plan = drops_.dev_plan && drops_.dev_plan(drops_.ctx, st);
    if (!bops_.router_topk(bops_.ctx, w.router_gpu, NE, E, KS, w.router_b, nullptr,
                           cfg_.norm_topk_prob ? 1 : 0, cfg_.routed_scaling,
                           dev_plan ? nullptr : selb.data(),
                           dev_plan ? nullptr : swb.data(),
                           dev_plan ? nullptr : nselb.data()))
      return false;
    AFF_PH(draft_route);
    std::fill(handled.begin(), handled.end(), (uint8_t)0);
    drops_.ffn_batch(drops_.ctx, st, selb.data(), swb.data(), B, KS, handled.data());
    // ---- the guard, and where it moves to under the device build -------------------------------
    //
    // Not a fallback: there is nowhere for an unhandled pair to go. The draft's expert pool is a
    // separate container with no host tier, so it means placement did not take every expert and the
    // draft would quietly be running a different model.
    //
    // The device build cannot answer per slot — nothing about the layer's shape comes back, which is
    // the entire point — so it counts the same fact into `MoePlanStatus::bad` and `run_batch_device`
    // aborts on it behind the issue frontier. The check is not weaker, it is later: a slot with no
    // address is a load-time configuration error that cannot appear mid-run, and the draft refuses
    // to start with one. `nselb` is stale here under the device build, so the loop must not run at
    // all rather than read it.
    if (!dev_plan)
      for (uint32_t b = 0; b < B; ++b)
        for (uint32_t j = 0; j < nselb[b]; ++j)
          if (!handled[(size_t)b * KS + j]) {
            aff::ui::fatal("fatal: draft stage %u expert %u for token %u is not resident\n",
                         st, selb[(size_t)b * KS + j], b);
            std::abort();
          }
    if (bops_.block_reduce && !bops_.block_reduce(bops_.ctx, E)) return false;
    if (!bops_.hc_post(bops_.ctx, w.hc_ffn_fn.data != nullptr, E, HC)) return false;
    AFF_PH(draft_ffn);
  }

  // ---- the draft's head ----------------------------------------------------------------------
  // The stage's OWN hc_head and norm, and the TARGET's vocabulary projection — shared exactly as
  // `Transformer.__init__` shares it.
  //
  // On the card, in three kernels. batch_hc_head_narrow_hip transcribes the host epilogue still
  // standing at the end of forward_prefill, which per token copies the whole [n_hc][n_embd] control
  // plane back behind a stream drain, rms_norms 16384 floats, runs a 4 x 16384 bf16 matvec, gates,
  // collapses and norms, then uploads the result again — five round trips and 640 KB each way for
  // data that never had to leave.
  //
  // Not bit-identical to it — the sum of squares is a tree rather than a serial walk, and the mix
  // scale multiplies the four dots rather than their 16384 inputs. That cannot reach the output:
  // the accept rule is `draft[j] == argmax(target[j])`, so a draft whose last bits moved proposes a
  // different token and the target either agrees with it or does not. Only acceptance moves, and
  // acceptance is a property of the prompt to begin with.
  // `DSparkBlock.forward_head` collapses with `pre_mix` exactly as the model's head does, so on
  // V4.1 the draft has no head mixer either and reuses what its last stage's FFN computed.
  if (cfg_.v41) {
    if (!bops_.collapse_pre || D.norm_gpu < 0) {
      aff::ui::fatal("fatal: the v4.1 draft head needs collapse_pre and its norm on the device "
                     "(norm handle %d)\n", D.norm_gpu);
      std::abort();
    }
    if (!bops_.collapse_pre(bops_.ctx, D.norm_gpu, E, HC, cfg_.hc_eps, cfg_.rms_eps)) return false;
  } else {
    if (!bops_.draft_collapse || D.hc_head_fn.gpu < 0 || D.hc_head_scale_gpu < 0 ||
        D.hc_head_base_gpu < 0 || D.norm_gpu < 0) {
      aff::ui::fatal("fatal: the draft's head collapse is not bound on the device\n");
      std::abort();
    }
    if (!bops_.draft_collapse(bops_.ctx, D.hc_head_fn.gpu, D.hc_head_scale_gpu, D.hc_head_base_gpu,
                              D.norm_gpu, E, HC, cfg_.hc_eps, cfg_.rms_eps))
      return false;
  }
  AFF_PH(draft_ep);
  if (!bops_.draft_head(bops_.ctx, head_.gpu, cfg_.vocab, E, B, nullptr, D.markov_w1.gpu,
                        D.markov_w2.gpu, cfg_.dspark_markov_rank, first_id, out))
    return false;
  AFF_PH(draft_head);
  return true;
}

}  // namespace aff
