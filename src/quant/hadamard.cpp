#include "hadamard.h"

#include <cmath>
#include <cstring>

namespace aff {

void fwht(float* x, uint64_t n) noexcept {
  if (!is_pow2(n)) return;
  for (uint64_t len = 1; len < n; len <<= 1) {
    for (uint64_t i = 0; i < n; i += (len << 1)) {
      for (uint64_t j = i; j < i + len; ++j) {
        const float a = x[j], b = x[j + len];
        x[j]       = a + b;
        x[j + len] = a - b;
      }
    }
  }
}

void fwht_normalized(float* x, uint64_t n) noexcept {
  if (!is_pow2(n)) return;
  fwht(x, n);
  const float inv = 1.0f / std::sqrt((float)n);
  for (uint64_t i = 0; i < n; ++i) x[i] *= inv;
}

void random_signs(int8_t* s, uint64_t n, uint64_t seed) noexcept {
  // splitmix64 — deterministic across platforms, no <random> state to serialise.
  uint64_t z = seed + 0x9E3779B97F4A7C15ull;
  for (uint64_t i = 0; i < n; ++i) {
    uint64_t x = (z += 0x9E3779B97F4A7C15ull);
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    x =  x ^ (x >> 31);
    s[i] = (x & 1) ? 1 : -1;
  }
}

void randomized_hadamard_rows(float* x, uint64_t rows, uint64_t n,
                              const int8_t* signs, bool normalized) noexcept {
  for (uint64_t r = 0; r < rows; ++r) {
    float* row = x + r * n;
    for (uint64_t i = 0; i < n; ++i) row[i] *= (float)signs[i];
    if (normalized) fwht_normalized(row, n);
    else            fwht(row, n);
  }
}

IncoherenceStats incoherence_stats(const float* x, uint64_t n, uint64_t block) noexcept {
  IncoherenceStats s;
  if (n == 0) return s;

  double sum = 0, sq = 0, amax = 0;
  for (uint64_t i = 0; i < n; ++i) {
    const double v = x[i];
    sum += v; sq += v * v;
    const double a = std::fabs(v);
    if (a > amax) amax = a;
  }
  const double mean = sum / (double)n;
  s.rms  = std::sqrt(sq / (double)n);
  s.amax = amax;
  s.max_over_rms = s.rms > 0 ? amax / s.rms : 0;

  // excess-free kurtosis about the mean
  double m2 = 0, m4 = 0;
  for (uint64_t i = 0; i < n; ++i) {
    const double d = x[i] - mean;
    m2 += d * d; m4 += d * d * d * d;
  }
  m2 /= (double)n; m4 /= (double)n;
  s.kurtosis = (m2 > 0) ? m4 / (m2 * m2) : 0;

  if (block > 0) {
    double acc = 0;
    uint64_t nb = 0;
    for (uint64_t b = 0; b + block <= n; b += block) {
      double bmax = 0, bsq = 0;
      for (uint64_t i = 0; i < block; ++i) {
        const double v = x[b + i];
        bsq += v * v;
        const double a = std::fabs(v);
        if (a > bmax) bmax = a;
      }
      const double brms = std::sqrt(bsq / (double)block);
      if (brms > 0) { acc += bmax / brms; ++nb; }
    }
    s.mean_block_max_over_rms = nb ? acc / (double)nb : 0;
  }
  return s;
}

} // namespace aff
