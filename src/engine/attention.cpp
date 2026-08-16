#include "attention.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace aff {

float AttnConfig::eff_scale() const {
  return scale > 0.0f ? scale : 1.0f / std::sqrt((float)qk_width());
}

void rope_partial(float* v, const AttnConfig& c, uint64_t pos) noexcept {
  // Only the trailing rope_dim is rotated; the leading head_dim passes through untouched.
  float* r = v + c.head_dim;
  const uint32_t half = c.rope_dim / 2;
  for (uint32_t i = 0; i < half; ++i) {
    const float freq = 1.0f / std::pow(c.rope_theta, (float)(2 * i) / (float)c.rope_dim);
    const float ang = (float)pos * freq;
    const float cs = std::cos(ang), sn = std::sin(ang);
    const float a = r[i], b = r[i + half];
    r[i]        = a * cs - b * sn;
    r[i + half] = a * sn + b * cs;
  }
}

void attention_mqa(const AttnConfig& c, const float* q, const KvCache& kv,
                   const uint64_t* keys, uint64_t n_keys,
                   const float* sinks, float* out) noexcept {
  const uint32_t W = c.qk_width(), D = c.qk_width();
  const float sc = c.eff_scale();

  for (uint32_t h = 0; h < c.n_head; ++h) {
    const float* qh = q + (size_t)h * W;
    float* oh = out + (size_t)h * D;

    // Sinks SEED the running max and denominator. The sink contributes exp(0)=1 to `l` but
    // nothing to `acc` — it has no value vector — so attention weights can sum to < 1.
    float m = sinks ? sinks[h] : -INFINITY;
    float l = sinks ? 1.0f : 0.0f;
    std::memset(oh, 0, D * sizeof(float));

    for (uint64_t j = 0; j < n_keys; ++j) {
      const float* kvp = kv.at(keys[j]);
      // One shared latent serves every head: the score uses all qk_width dims, the value is the
      // leading head_dim of the SAME vector. No per-head K/V is ever materialised.
      float dot = 0.0f;
      for (uint32_t d = 0; d < W; ++d) dot += qh[d] * kvp[d];
      const float s = dot * sc;

      const float mn = std::max(m, s);
      const float corr = (m == -INFINITY) ? 0.0f : std::exp(m - mn);
      const float p = std::exp(s - mn);
      l = l * corr + p;
      for (uint32_t d = 0; d < D; ++d) oh[d] = oh[d] * corr + p * kvp[d];
      m = mn;
    }
    const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    for (uint32_t d = 0; d < D; ++d) oh[d] *= inv;
  }
}

void attention_mqa_reference(const AttnConfig& c, const float* q, const KvCache& kv,
                             const uint64_t* keys, uint64_t n_keys,
                             const float* sinks, float* out) noexcept {
  const uint32_t W = c.qk_width(), D = c.qk_width();
  const float sc = c.eff_scale();
  std::vector<float> logits(n_keys);

  for (uint32_t h = 0; h < c.n_head; ++h) {
    const float* qh = q + (size_t)h * W;
    float* oh = out + (size_t)h * D;
    float mx = sinks ? sinks[h] : -INFINITY;
    for (uint64_t j = 0; j < n_keys; ++j) {
      const float* kvp = kv.at(keys[j]);
      double dot = 0.0;
      for (uint32_t d = 0; d < W; ++d) dot += (double)qh[d] * kvp[d];
      logits[j] = (float)(dot * sc);
      mx = std::max(mx, logits[j]);
    }
    // Denominator includes the sink term exp(sink - mx); the numerator does not.
    double den = sinks ? std::exp((double)sinks[h] - mx) : 0.0;
    for (uint64_t j = 0; j < n_keys; ++j) den += std::exp((double)logits[j] - mx);
    std::vector<double> acc(D, 0.0);
    for (uint64_t j = 0; j < n_keys; ++j) {
      const double p = std::exp((double)logits[j] - mx);
      const float* kvp = kv.at(keys[j]);
      for (uint32_t d = 0; d < D; ++d) acc[d] += p * kvp[d];
    }
    for (uint32_t d = 0; d < D; ++d) oh[d] = (float)(acc[d] / (den > 0 ? den : 1.0));
  }
}

void build_key_list(const AttnConfig& c, uint64_t cur_pos, uint32_t ratio,
                    const uint64_t* selected, uint64_t n_selected,
                    std::vector<uint64_t>* keys) {
  keys->clear();
  if (ratio == 0) {                       // layers 0-1: no compression, plain causal
    for (uint64_t p = 0; p <= cur_pos; ++p) keys->push_back(p);
    return;
  }
  // Raw sliding window of the most recent `sliding` positions ...
  const uint64_t lo = (cur_pos >= c.sliding) ? cur_pos - c.sliding + 1 : 0;
  for (uint64_t p = lo; p <= cur_pos; ++p) keys->push_back(p);
  // ... then the compressed rows chosen for this query (indexer top-k at ratio 4, all of them at
  // ratio 128 where the compressed cache is short enough to attend densely).
  for (uint64_t i = 0; i < n_selected; ++i) keys->push_back(selected[i]);
}

} // namespace aff
