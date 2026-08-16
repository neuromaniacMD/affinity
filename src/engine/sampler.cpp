#include "sampler.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace aff {

double Sampler::next_uniform() {
  uint64_t x = (rng_ += 0x9E3779B97F4A7C15ull);
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  x ^= x >> 31;
  return (double)(x >> 11) * (1.0 / 9007199254740992.0);
}

uint32_t Sampler::argmax(const std::vector<float>& logits) const {
  return (uint32_t)(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

uint32_t Sampler::sample(std::vector<float>& logits, const std::vector<uint32_t>& history) {
  const uint32_t n = (uint32_t)logits.size();
  if (!n) return 0;

  if (cfg_.repeat_penalty != 1.0f && cfg_.repeat_last_n > 0) {
    const size_t start = history.size() > cfg_.repeat_last_n ? history.size() - cfg_.repeat_last_n : 0;
    for (size_t i = start; i < history.size(); ++i) {
      const uint32_t t = history[i];
      if (t >= n) continue;
      // Positive logits are divided, negative multiplied — otherwise the penalty would *raise*
      // the score of already-negative tokens.
      logits[t] = (logits[t] > 0) ? logits[t] / cfg_.repeat_penalty
                                  : logits[t] * cfg_.repeat_penalty;
    }
  }

  if (cfg_.temperature <= 0.0f) return argmax(logits);
  const float inv_t = 1.0f / cfg_.temperature;
  for (uint32_t i = 0; i < n; ++i) logits[i] *= inv_t;

  std::vector<uint32_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0u);
  uint32_t keep = n;
  if (cfg_.top_k > 0 && cfg_.top_k < n) {
    std::partial_sort(idx.begin(), idx.begin() + cfg_.top_k, idx.end(),
                      [&](uint32_t a, uint32_t b) { return logits[a] > logits[b]; });
    keep = cfg_.top_k;
  } else {
    std::sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) { return logits[a] > logits[b]; });
  }

  const float mx = logits[idx[0]];
  std::vector<double> p(keep);
  double sum = 0.0;
  for (uint32_t i = 0; i < keep; ++i) { p[i] = std::exp((double)logits[idx[i]] - mx); sum += p[i]; }
  for (uint32_t i = 0; i < keep; ++i) p[i] /= sum;

  if (cfg_.min_p > 0.0f) {                     // relative floor against the top token
    const double floor_p = cfg_.min_p * p[0];
    uint32_t k = 0;
    while (k < keep && p[k] >= floor_p) ++k;
    keep = std::max(1u, k);
  }
  if (cfg_.top_p < 1.0f) {                     // nucleus
    double cum = 0.0; uint32_t k = 0;
    while (k < keep) { cum += p[k]; ++k; if (cum >= cfg_.top_p) break; }
    keep = std::max(1u, k);
  }

  double norm = 0.0;
  for (uint32_t i = 0; i < keep; ++i) norm += p[i];
  double r = next_uniform() * norm, acc = 0.0;
  for (uint32_t i = 0; i < keep; ++i) { acc += p[i]; if (acc >= r) return idx[i]; }
  return idx[keep - 1];
}

} // namespace aff
