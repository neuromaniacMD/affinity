#include "ops.h"
#include "threadpool.h"
#include "quant/source_dtypes.h"
#if defined(__AVX512F__)
#include <immintrin.h>
#endif
#include <vector>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>

// After <immintrin.h>: the variants are written directly in AVX-512 intrinsics.
#include "matvec_bf16.inc"
#include "matvec_fp8.inc"

#include "ui/log.h"
namespace aff {

void rms_norm(const float* x, const float* weight, uint64_t n, float eps, float* out) noexcept {
  double ss = 0.0;
  for (uint64_t i = 0; i < n; ++i) ss += (double)x[i] * x[i];
  const float inv = (float)(1.0 / std::sqrt(ss / (double)n + eps));
  for (uint64_t i = 0; i < n; ++i) out[i] = x[i] * inv * (weight ? weight[i] : 1.0f);
}

float sqrtsoftplus(float x) noexcept {
  // softplus with the standard large-x guard, then sqrt. Non-negative by construction, which is
  // what lets the router treat it as a score rather than a logit.
  const float sp = (x > 20.0f) ? x : std::log1p(std::exp(x));
  return std::sqrt(sp);
}

void softmax_inplace(float* x, uint64_t n) noexcept {
  if (!n) return;
  float m = x[0];
  for (uint64_t i = 1; i < n; ++i) m = std::max(m, x[i]);
  double s = 0.0;
  for (uint64_t i = 0; i < n; ++i) { x[i] = std::exp(x[i] - m); s += x[i]; }
  const float inv = (float)(1.0 / (s > 0 ? s : 1.0));
  for (uint64_t i = 0; i < n; ++i) x[i] *= inv;
}

void swiglu(const float* gate, const float* up, uint64_t n, float limit, float* out) noexcept {
  for (uint64_t i = 0; i < n; ++i) {
    // The two clamps are NOT the same shape (DeepSeek-V4-Flash inference/model.py, Expert.forward):
    //   gate = clamp(gate, max=limit)          one-sided
    //   up   = clamp(up, -limit, +limit)       two-sided
    // Leaving `up` unclamped lets a single outlier activation dominate the expert's output.
    float g = gate[i], u = up[i];
    if (limit > 0.0f) {
      g = std::min(g, limit);
      u = std::min(limit, std::max(-limit, u));
    }
    out[i] = (g / (1.0f + std::exp(-g))) * u;
  }
}

void route_topk(const float* logits, const float* bias, uint32_t n_expert, uint32_t top_k,
                bool norm_topk_prob, float routed_scaling, RouterOut* out) noexcept {
  if (top_k > kMaxTopK) {
    aff::ui::fatal("fatal: route_topk asked for %u experts; RouterOut carries %u\n",
                 top_k, kMaxTopK);
    std::abort();
  }
  std::vector<float> score(n_expert), sel(n_expert);
  for (uint32_t e = 0; e < n_expert; ++e) {
    score[e] = sqrtsoftplus(logits[e]);
    // The aux-loss-free correction biases WHICH experts win, but never how much they contribute.
    sel[e] = score[e] + (bias ? bias[e] : 0.0f);
  }
  std::vector<uint32_t> idx(n_expert);
  std::iota(idx.begin(), idx.end(), 0u);
  const uint32_t k = std::min(top_k, n_expert);
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [&](uint32_t a, uint32_t b) {
                      if (sel[a] != sel[b]) return sel[a] > sel[b];
                      return a < b;                       // deterministic tie-break
                    });
  double sum = 0.0;
  for (uint32_t i = 0; i < k; ++i) sum += score[idx[i]];
  if (sum < 6.103515625e-5) sum = 6.103515625e-5;   // fp16 min-normal floor, as in the reference
  const double inv = norm_topk_prob ? 1.0 / sum : 1.0;
  out->n = k;
  for (uint32_t i = 0; i < k; ++i) {
    out->expert[i] = idx[i];
    out->weight[i] = (float)(score[idx[i]] * inv * routed_scaling);
  }
}

void route_hash(const int32_t* table, uint32_t vocab, uint32_t top_k, uint32_t token_id,
                RouterOut* out) noexcept {
  if (top_k > kMaxTopK) {
    aff::ui::fatal("fatal: route_hash asked for %u experts; RouterOut carries %u\n",
                 top_k, kMaxTopK);
    std::abort();
  }
  out->n = 0;
  if (token_id >= vocab) return;
  for (uint32_t i = 0; i < top_k; ++i) {
    out->expert[i] = (uint32_t)table[(size_t)i * vocab + token_id];
    out->weight[i] = 1.0f / (float)top_k;
    out->n = i + 1;
  }
}

void compress_pool(const CompressorConfig& c, const float* raw, uint64_t n_raw,
                   const float* score_w, const float* ape,
                   float* out, uint64_t* n_out) noexcept {
  const uint32_t W = c.width, m = c.ratio;
  const uint64_t nc = (n_raw + m - 1) / m;
  *n_out = nc;
  // Each compressed row pools a window of `m` (or 2m when overlapping) raw rows. The weight is a
  // SOFTMAX over a learned projection of each row plus a positional embedding — not a mean.
  const uint32_t span = (c.overlap && m > 1) ? m * 2 : m;
  std::vector<float> logit(span);
  for (uint64_t b = 0; b < nc; ++b) {
    const uint64_t start = b * m;
    const uint64_t lo = (c.overlap && start >= m) ? start - m : start;
    const uint64_t hi = std::min<uint64_t>(start + m, n_raw);
    const uint64_t cnt = hi - lo;
    if (cnt == 0) { std::memset(out + b * W, 0, W * sizeof(float)); continue; }
    for (uint64_t i = 0; i < cnt; ++i) {
      const float* r = raw + (lo + i) * W;
      double s = 0.0;
      for (uint32_t d = 0; d < W; ++d) s += (double)score_w[d] * r[d];
      logit[i] = (float)s + (ape ? ape[std::min<uint64_t>(i, span - 1)] : 0.0f);
    }
    softmax_inplace(logit.data(), cnt);
    float* o = out + b * W;
    std::memset(o, 0, W * sizeof(float));
    for (uint64_t i = 0; i < cnt; ++i) {
      const float* r = raw + (lo + i) * W;
      const float wgt = logit[i];
      for (uint32_t d = 0; d < W; ++d) o[d] += wgt * r[d];
    }
  }
}

void indexer_scores(const IndexerConfig& c, const float* q, const float* w,
                    const float* keys, uint64_t n_keys, float* scores) noexcept {
  const uint32_t H = c.n_head, D = c.head_dim;
  for (uint64_t s = 0; s < n_keys; ++s) {
    const float* k = keys + s * D;
    double acc = 0.0;
    for (uint32_t h = 0; h < H; ++h) {
      const float* qh = q + (size_t)h * D;
      float dot = 0.0f;
      for (uint32_t d = 0; d < D; ++d) dot += qh[d] * k[d];
      // ReLU then weighted sum. No softmax anywhere — the indexer produces a ranking signal,
      // not a distribution.
      acc += (double)w[h] * (dot > 0.0f ? dot : 0.0f);
    }
    scores[s] = (float)acc;
  }
}

void top_k_select(const float* scores, uint64_t n, uint32_t k, uint64_t* out_idx,
                  uint64_t* n_out) noexcept {
  const uint64_t kk = std::min<uint64_t>(k, n);
  *n_out = kk;
  if (!kk) return;
  std::vector<uint64_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0ull);
  std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                    [&](uint64_t a, uint64_t b) {
                      if (scores[a] != scores[b]) return scores[a] > scores[b];
                      return a < b;
                    });
  // Selected rows are attended in position order, so sort the winners by index.
  std::sort(idx.begin(), idx.begin() + kk);
  for (uint64_t i = 0; i < kk; ++i) out_idx[i] = idx[i];
}

void sinkhorn(float* m, uint32_t n, uint32_t iters, float eps) noexcept {
  // Iteration 0 is row-only; the alternating row/column sweeps start at iteration 1. Running a
  // column pass first changes the fixed point it converges to.
  for (uint32_t it = 0; it < iters; ++it) {
    for (uint32_t r = 0; r < n; ++r) {
      double s = eps;
      for (uint32_t c = 0; c < n; ++c) s += m[(size_t)r * n + c];
      const float inv = (float)(1.0 / s);
      for (uint32_t c = 0; c < n; ++c) m[(size_t)r * n + c] *= inv;
    }
    if (it == 0) continue;                       // first pass is rows only
    for (uint32_t c = 0; c < n; ++c) {
      double s = eps;
      for (uint32_t r = 0; r < n; ++r) s += m[(size_t)r * n + c];
      const float inv = (float)(1.0 / s);
      for (uint32_t r = 0; r < n; ++r) m[(size_t)r * n + c] *= inv;
    }
  }
}

void hc_mix(const HcConfig& c, const float* lanes, uint32_t width,
            const float* fn, const float* scale, float* out) noexcept {
  const uint32_t L = c.mult;
  std::vector<float> mix((size_t)L * L);
  for (uint32_t i = 0; i < L * L; ++i) mix[i] = std::exp(fn[i] * (scale ? scale[0] : 1.0f));
  sinkhorn(mix.data(), L, c.sinkhorn_iters, c.eps);
  for (uint32_t o = 0; o < L; ++o) {
    float* d = out + (size_t)o * width;
    std::memset(d, 0, width * sizeof(float));
    for (uint32_t i = 0; i < L; ++i) {
      const float wgt = mix[(size_t)o * L + i];
      const float* s = lanes + (size_t)i * width;
      for (uint32_t x = 0; x < width; ++x) d[x] += wgt * s[x];
    }
  }
}


// ==============================================================================================
// Exact DeepSeek-V4-Flash primitives (see the header for why each one is here).
// ==============================================================================================

namespace {

// e4m3fn: 4-bit exponent, 3-bit mantissa, no infinities, max 448. Built once and searched, which
// is exact and cheap next to the matvecs around it.
const std::vector<float>& e4m3_values() {
  static const std::vector<float> v = [] {
    std::vector<float> t;
    for (uint32_t c = 0; c < 128; ++c) {           // sign bit excluded; table is |value|
      const uint32_t e = (c >> 3) & 0xF, m = c & 0x7;
      if (e == 0xF && m == 0x7) continue;          // NaN in e4m3fn
      t.push_back(e == 0 ? std::ldexp((float)m, -9) : std::ldexp(1.0f + (float)m / 8.0f, (int)e - 7));
    }
    std::sort(t.begin(), t.end());
    return t;
  }();
  return v;
}

float e4m3_round(float x) noexcept {
  const float sign = x < 0.0f ? -1.0f : 1.0f;
  const float a = std::fabs(x);
  const std::vector<float>& t = e4m3_values();
  auto it = std::lower_bound(t.begin(), t.end(), a);
  if (it == t.end()) return sign * t.back();
  if (it == t.begin()) return sign * *it;
  const float hi = *it, lo = *(it - 1);
  return sign * ((a - lo) <= (hi - a) ? lo : hi);
}

float yarn_ramp(float low, float high, int i) noexcept {
  const float y = ((float)i / 2.0f - low) / std::max(0.001f, high - low);
  return 1.0f - std::min(1.0f, std::max(0.0f, y));
}
float yarn_corr_dim(int n_dims, uint32_t n_ctx_orig, float n_rot_target, float base) noexcept {
  return (float)n_dims * std::log((float)n_ctx_orig / (n_rot_target * 6.2831853071795864f)) /
         (2.0f * std::log(base));
}

} // namespace

// Variant selection for A/B measurement only; 2 is the default. See matvec_bf16.inc for why this
// kernel matters more than the expert kernel does.
static int g_mv_variant = 2;
void matvec_bf16_variant(int v) { g_mv_variant = v; }
int  matvec_bf16_variant() { return g_mv_variant; }

void matvec_bf16(const uint16_t* W, const float* x, uint64_t rows, uint64_t cols,
                 float* out) noexcept {
  if (g_mv_variant == 1) matvec_bf16_v1(W, x, rows, cols, out);
  else                   matvec_bf16_v2(W, x, rows, cols, out);
}


// 512 rows is roughly where a pool round trip (~2-5 us) stops dominating a bf16 row sweep.
constexpr uint64_t kMtRowThreshold = 512;

void matvec_bf16_mt(const uint16_t* W, const float* x, uint64_t rows, uint64_t cols, float* out) {
  parallel_for(rows, kMtRowThreshold, [&](uint64_t lo, uint64_t hi) {
    matvec_bf16(W + lo * cols, x, hi - lo, cols, out + lo);
  });
}

// ---- dense dispatch ---------------------------------------------------------------------------

void matvec_dense(const DenseW& w, const float* x, uint64_t rows, uint64_t cols,
                  float* out) noexcept {
  if (w.quant == AFF_FP8_E4M3)
    matvec_fp8_rows((const uint8_t*)w.data, w.scales, 0, rows, cols, x, out);
  else
    matvec_bf16((const uint16_t*)w.data, x, rows, cols, out);
}

void matvec_dense_mt(const DenseW& w, const float* x, uint64_t rows, uint64_t cols, float* out) {
  if (w.quant != AFF_FP8_E4M3) { matvec_bf16_mt((const uint16_t*)w.data, x, rows, cols, out); return; }
  // Pass the ABSOLUTE row offset, not a shifted base pointer: the scale tile is chosen by row index,
  // so a thread that started its own numbering at zero would read another tile's scale.
  parallel_for(rows, kMtRowThreshold, [&](uint64_t lo, uint64_t hi) {
    matvec_fp8_rows((const uint8_t*)w.data, w.scales, lo, hi - lo, cols, x, out + lo);
  });
}

void matvec_dense_grouped_mt(const DenseW& w, const float* x, uint32_t n_groups,
                             uint64_t group_dim, uint64_t rank, float* out) {
  if (w.quant != AFF_FP8_E4M3) {
    matvec_bf16_grouped_mt((const uint16_t*)w.data, x, n_groups, group_dim, rank, out);
    return;
  }
  const uint64_t total = (uint64_t)n_groups * rank;
  parallel_for(total, kMtRowThreshold, [&](uint64_t lo, uint64_t hi) {
    for (uint64_t i = lo; i < hi; ++i) {
      const uint64_t g = i / rank;
      matvec_fp8_rows((const uint8_t*)w.data, w.scales, i, 1, group_dim, x + g * group_dim, out + i);
    }
  });
}

void matvec_bf16_grouped_mt(const uint16_t* W, const float* x, uint32_t n_groups,
                            uint64_t group_dim, uint64_t rank, float* out) {
  // Parallelise over the flattened (group, row) space so 8 groups of 1024 rows still spread over
  // more than 8 workers.
  parallel_for((uint64_t)n_groups * rank, kMtRowThreshold, [&](uint64_t lo, uint64_t hi) {
    for (uint64_t i = lo; i < hi; ++i) {
      const uint64_t g = i / rank, r = i % rank;
      matvec_bf16(W + (g * rank + r) * group_dim, x + g * group_dim, 1, group_dim, out + i);
    }
  });
}

void rms_norm_noweight(const float* x, uint64_t n, float eps, float* out) noexcept {
  double ss = 0.0;
  for (uint64_t i = 0; i < n; ++i) ss += (double)x[i] * x[i];
  const float inv = 1.0f / std::sqrt((float)(ss / (double)n) + eps);
  for (uint64_t i = 0; i < n; ++i) out[i] = x[i] * inv;
}

void head_rms_norm_inplace(float* x, uint32_t n_head, uint32_t head_dim, float eps) noexcept {
  for (uint32_t h = 0; h < n_head; ++h) {
    float* p = x + (size_t)h * head_dim;
    rms_norm_noweight(p, head_dim, eps, p);
  }
}

RopeDerived rope_derive(const RopeParams& p, uint32_t n_rot) noexcept {
  RopeDerived d{};
  d.freq_scale = p.freq_scale;
  d.theta_scale = std::pow(p.freq_base, -2.0f / (float)n_rot);
  d.ext_factor = p.ext_factor;
  if (p.ext_factor != 0.0f) {
    d.lo = std::max(0.0f, std::floor(yarn_corr_dim((int)n_rot, p.orig_ctx, p.beta_fast, p.freq_base)));
    d.hi = std::min((float)(n_rot - 1),
                    std::ceil(yarn_corr_dim((int)n_rot, p.orig_ctx, p.beta_slow, p.freq_base)));
  }
  return d;
}

void rope_tail(float* x, uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
               uint32_t pos, const RopeParams& p, bool inverse) noexcept {
  const uint32_t n_nope = head_dim - n_rot;
  const float theta_scale = std::pow(p.freq_base, -2.0f / (float)n_rot);
  const float sin_sign = inverse ? -1.0f : 1.0f;
  float lo = 0.0f, hi = 0.0f;
  if (p.ext_factor != 0.0f) {
    lo = std::max(0.0f, std::floor(yarn_corr_dim((int)n_rot, p.orig_ctx, p.beta_fast, p.freq_base)));
    hi = std::min((float)(n_rot - 1),
                  std::ceil(yarn_corr_dim((int)n_rot, p.orig_ctx, p.beta_slow, p.freq_base)));
  }
  for (uint32_t h = 0; h < n_head; ++h) {
    float* t = x + (size_t)h * head_dim + n_nope;
    float theta_extrap = (float)pos;
    for (uint32_t i = 0; i < n_rot; i += 2) {
      float theta = p.freq_scale * theta_extrap;
      if (p.ext_factor != 0.0f) {
        // YaRN blends interpolated and extrapolated FREQUENCIES. It does not touch magnitude:
        // precompute_freqs_cis builds polar(ones, freqs), so |e^{i.theta}| is always 1. ds4 applies
        // an mscale and then cancels it with a reciprocal attn_factor; the official code has no
        // such term, so there is nothing to cancel.
        const float mix = yarn_ramp(lo, hi, (int)i) * p.ext_factor;
        theta = theta * (1.0f - mix) + theta_extrap * mix;
      }
      const float c = std::cos(theta), sn = sin_sign * std::sin(theta);
      const float x0 = t[i], x1 = t[i + 1];
      t[i] = x0 * c - x1 * sn;
      t[i + 1] = x0 * sn + x1 * c;
      theta_extrap *= theta_scale;
    }
  }
}

void fp8_kv_qat(float* x, uint32_t head_dim, uint32_t n_rot) noexcept {
  const uint32_t n_nope = head_dim - n_rot;
  for (uint32_t off = 0; off + 64 <= n_nope; off += 64) {
    float amax = 0.0f;
    for (uint32_t i = 0; i < 64; ++i) amax = std::max(amax, std::fabs(x[off + i]));
    if (amax < 1e-4f) amax = 1e-4f;
    const float scale = std::ldexp(1.0f, (int)std::ceil(std::log2(amax / 448.0f)));
    for (uint32_t i = 0; i < 64; ++i) {
      const float v = std::min(448.0f, std::max(-448.0f, x[off + i] / scale));
      x[off + i] = e4m3_round(v) * scale;
    }
  }
}


void hc_control(const uint16_t* fn, const float* scale, const float* base,
                const float* residual_hc, uint32_t n_embd, uint32_t n_hc,
                uint32_t iters, float hc_eps, float rms_eps, HcControl* out) noexcept {
  const uint64_t hc_dim = (uint64_t)n_embd * n_hc;
  const uint32_t n_out = 2 * n_hc + n_hc * n_hc;
  std::vector<float> flat(hc_dim), mix(n_out);
  rms_norm_noweight(residual_hc, hc_dim, rms_eps, flat.data());
  matvec_bf16(fn, flat.data(), n_out, hc_dim, mix.data());

  for (uint32_t i = 0; i < n_hc; ++i) {
    const float z = mix[i] * scale[0] + base[i];
    out->pre[i] = 1.0f / (1.0f + std::exp(-z)) + hc_eps;
  }
  for (uint32_t i = 0; i < n_hc; ++i) {
    const uint32_t o = n_hc + i;
    const float z = mix[o] * scale[1] + base[o];
    out->post[i] = 2.0f / (1.0f + std::exp(-z));
  }

  // Combine matrix, indexed [src + dst*n_hc]: softmax over src, then Sinkhorn. The first pass is
  // NOT the same as the loop body — it normalises rows (via the softmax) then columns once.
  float* c = out->comb;
  for (uint32_t dst = 0; dst < n_hc; ++dst) {
    float rmax = -1e30f;
    for (uint32_t src = 0; src < n_hc; ++src) {
      const uint32_t idx = src + dst * n_hc;
      c[idx] = mix[2 * n_hc + idx] * scale[2] + base[2 * n_hc + idx];
      rmax = std::max(rmax, c[idx]);
    }
    float rsum = 0.0f;
    for (uint32_t src = 0; src < n_hc; ++src) {
      const uint32_t idx = src + dst * n_hc;
      c[idx] = std::exp(c[idx] - rmax); rsum += c[idx];
    }
    for (uint32_t src = 0; src < n_hc; ++src) c[src + dst * n_hc] = c[src + dst * n_hc] / rsum + hc_eps;
  }
  for (uint32_t src = 0; src < n_hc; ++src) {
    float sum = 0.0f;
    for (uint32_t dst = 0; dst < n_hc; ++dst) sum += c[src + dst * n_hc];
    const float inv = 1.0f / (sum + hc_eps);
    for (uint32_t dst = 0; dst < n_hc; ++dst) c[src + dst * n_hc] *= inv;
  }
  for (uint32_t it = 1; it < iters; ++it) {
    for (uint32_t dst = 0; dst < n_hc; ++dst) {
      float sum = 0.0f;
      for (uint32_t src = 0; src < n_hc; ++src) sum += c[src + dst * n_hc];
      const float inv = 1.0f / (sum + hc_eps);
      for (uint32_t src = 0; src < n_hc; ++src) c[src + dst * n_hc] *= inv;
    }
    for (uint32_t src = 0; src < n_hc; ++src) {
      float sum = 0.0f;
      for (uint32_t dst = 0; dst < n_hc; ++dst) sum += c[src + dst * n_hc];
      const float inv = 1.0f / (sum + hc_eps);
      for (uint32_t dst = 0; dst < n_hc; ++dst) c[src + dst * n_hc] *= inv;
    }
  }
}

void hc_reduce(const float* residual_hc, const float* pre, uint32_t n_embd, uint32_t n_hc,
               float* out) noexcept {
  for (uint32_t d = 0; d < n_embd; ++d) {
    float acc = 0.0f;
    for (uint32_t h = 0; h < n_hc; ++h) acc += residual_hc[(size_t)h * n_embd + d] * pre[h];
    out[d] = acc;
  }
}

void hc_expand(float* out_hc, const float* block_out, const float* residual_hc,
               const float* post, const float* comb, uint32_t n_embd, uint32_t n_hc) noexcept {
  for (uint32_t dst = 0; dst < n_hc; ++dst) {
    float* o = out_hc + (size_t)dst * n_embd;
    const float g = post[dst];
    for (uint32_t d = 0; d < n_embd; ++d) o[d] = block_out[d] * g;
    for (uint32_t src = 0; src < n_hc; ++src) {
      const float w = comb[dst + src * n_hc];
      const float* r = residual_hc + (size_t)src * n_embd;
      for (uint32_t d = 0; d < n_embd; ++d) o[d] += w * r[d];
    }
  }
}

} // namespace aff
