#include "blockquant.h"

#if defined(__AVX512F__)
#include <immintrin.h>
#endif
#include "source_dtypes.h"
#include "format/aff_format.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace aff {

namespace {
// A plain variable so test_quant can flip it mid-process and compare the two search paths.
bool g_use_simd = true;
} // namespace

QuantSpec expert_quant_spec(uint32_t q) {
  // block 16 | 10-bit QUAD codes | 3-bit pow2 scale | sign | rot2 | 1 variant bit = 2.875 bpw.
  // TWO variants, not four, and the reason is LDS: 1024 packed quads is 4 KB a table, and the
  // expert GEMM's staging leaves room for 8 KB before occupancy drops a block. See the FMT table
  // in expert_kernel.hip.
  if (q == (uint32_t)AFF_Q2P875_Q4) return QuantSpec{16, 10, 3, 4, 1, false, true, 0, 1, 1};
  // block 16 | 5-bit PAIR codes | 4-bit integer scale | 2 variant bits = the same 2.875 bpw.
  return QuantSpec{16, 5, 4, 2, 2, false};
}

void blockquant_use_simd(bool on) { g_use_simd = on; }
bool blockquant_use_simd() { return g_use_simd; }

namespace {

inline uint64_t splitmix(uint64_t& z) {
  uint64_t x = (z += 0x9E3779B97F4A7C15ull);
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

void pack_bits(std::vector<uint8_t>& dst, uint64_t index, uint32_t bits, uint32_t value) {
  const uint64_t bitpos = index * bits;
  uint64_t byte = bitpos >> 3;
  uint32_t off = static_cast<uint32_t>(bitpos & 7), remaining = bits;
  while (remaining > 0) {
    const uint32_t take = std::min(remaining, 8u - off);
    const uint32_t mask = ((1u << take) - 1u);
    dst[byte] = static_cast<uint8_t>((dst[byte] & ~(mask << off)) | ((value & mask) << off));
    value >>= take; remaining -= take; ++byte; off = 0;
  }
}

uint32_t unpack_bits(const std::vector<uint8_t>& src, uint64_t index, uint32_t bits) {
  const uint64_t bitpos = index * bits;
  uint64_t byte = bitpos >> 3;
  uint32_t off = static_cast<uint32_t>(bitpos & 7), remaining = bits, shift = 0, out = 0;
  while (remaining > 0) {
    const uint32_t take = std::min(remaining, 8u - off);
    const uint32_t mask = ((1u << take) - 1u);
    out |= ((src[byte] >> off) & mask) << shift;
    shift += take; remaining -= take; ++byte; off = 0;
  }
  return out;
}

// Widest tuple the encoder will code as one symbol. 4 is the interesting one beyond a pair: a
// 4-wide code is a whole dword of E4M3 in the kernel's LUT, which is one LDS read where a pair
// takes two.
constexpr uint32_t kMaxPair = 8;

// Squared distance from a sample (scalar or pair) to a codebook entry, optionally weighted.
inline double code_dist(const float* x, const float* e, uint32_t pair, double wa, double wb) {
  if (pair == 1) { const double d = (double)x[0] - e[0]; return wa * d * d; }
  if (pair == 2) {
    const double dx = (double)x[0] - e[0], dy = (double)x[1] - e[1];
    return wa * dx * dx + wb * dy * dy;
  }
  // Wider tuples carry ONE importance weight for the whole tuple — a per-column weight inside a
  // 4-wide code cannot change the argmin independently of its neighbours anyway.
  double s = 0;
  for (uint32_t p = 0; p < pair; ++p) { const double d = (double)x[p] - e[p]; s += d * d; }
  return wa * s;
}

// ---------------------------------------------------------------------------------------------
// SIMD nearest-neighbour over a 32-entry 2-D codebook.
//
// The scale search dominates quantisation: ~36 (scale, variant) candidates x 8 pairs x 32 entries
// is ~576 distance evaluations PER WEIGHT, and every one of them was a scalar double-precision
// expression. Two observations make it much cheaper:
//
//   * The search only needs the MINIMUM DISTANCE, never the code — best_code's return value is
//     discarded there. Argmin costs a compare-and-select chain; a min does not.
//   * 32 entries is exactly two 512-bit registers of floats. Deinterleaving the codebook into lo
//     and hi planes once per matrix turns the whole inner loop into ~8 vector ops.
//
// float, not double: this is an argmin/min over candidates, and the accumulated block error only
// has to ORDER candidates correctly. The chosen scale and code are identical either way.
// ---------------------------------------------------------------------------------------------
#if defined(__AVX512F__)
struct CbPlanes {                  // per variant: 32 lo values and 32 hi values, as 2x16 lanes
  __m512 lo0, lo1, hi0, hi1;
};

inline void build_planes(const Codebook& cb, std::vector<CbPlanes>* out) {
  out->resize(cb.variants);
  for (uint32_t v = 0; v < cb.variants; ++v) {
    alignas(64) float L[32], H[32];
    for (uint32_t j = 0; j < 32; ++j) { L[j] = cb.entry(v, j)[0]; H[j] = cb.entry(v, j)[1]; }
    (*out)[v].lo0 = _mm512_load_ps(L);
    (*out)[v].lo1 = _mm512_load_ps(L + 16);
    (*out)[v].hi0 = _mm512_load_ps(H);
    (*out)[v].hi1 = _mm512_load_ps(H + 16);
  }
}

// Smallest weighted squared distance from (t0,t1) to any of the 32 entries.
inline float min_dist_x32(const CbPlanes& p, float t0, float t1, float wa, float wb) {
  const __m512 v0 = _mm512_set1_ps(t0), v1 = _mm512_set1_ps(t1);
  const __m512  va = _mm512_set1_ps(wa), vb = _mm512_set1_ps(wb);
  const __m512 d0 = _mm512_sub_ps(p.lo0, v0), d1 = _mm512_sub_ps(p.lo1, v0);
  const __m512 e0 = _mm512_sub_ps(p.hi0, v1), e1 = _mm512_sub_ps(p.hi1, v1);
  const __m512 s0 = _mm512_fmadd_ps(_mm512_mul_ps(d0, d0), va, _mm512_mul_ps(_mm512_mul_ps(e0, e0), vb));
  const __m512 s1 = _mm512_fmadd_ps(_mm512_mul_ps(d1, d1), va, _mm512_mul_ps(_mm512_mul_ps(e1, e1), vb));
  return _mm512_reduce_min_ps(_mm512_min_ps(s0, s1));
}

// Same, but returns the winning index — needed once per pair at encode time, not in the search.
inline uint32_t argmin_x32(const CbPlanes& p, float t0, float t1, float wa, float wb) {
  const __m512 v0 = _mm512_set1_ps(t0), v1 = _mm512_set1_ps(t1);
  const __m512 va = _mm512_set1_ps(wa), vb = _mm512_set1_ps(wb);
  const __m512 d0 = _mm512_sub_ps(p.lo0, v0), d1 = _mm512_sub_ps(p.lo1, v0);
  const __m512 e0 = _mm512_sub_ps(p.hi0, v1), e1 = _mm512_sub_ps(p.hi1, v1);
  const __m512 s0 = _mm512_fmadd_ps(_mm512_mul_ps(d0, d0), va, _mm512_mul_ps(_mm512_mul_ps(e0, e0), vb));
  const __m512 s1 = _mm512_fmadd_ps(_mm512_mul_ps(d1, d1), va, _mm512_mul_ps(_mm512_mul_ps(e1, e1), vb));
  const float m0 = _mm512_reduce_min_ps(s0), m1 = _mm512_reduce_min_ps(s1);
  if (m0 <= m1) return (uint32_t)__builtin_ctz(_mm512_cmp_ps_mask(s0, _mm512_set1_ps(m0), _CMP_EQ_OQ));
  return 16u + (uint32_t)__builtin_ctz(_mm512_cmp_ps_mask(s1, _mm512_set1_ps(m1), _CMP_EQ_OQ));
}
#endif

// ---------------------------------------------------------------------------------------------
// Wide codes (pair >= 4)
//
// `best_code` walks the interleaved codebook in doubles, which at k=1024 is 16 KB of strided loads
// per query. These take the same argmin over a per-dimension PLANE in floats, which the compiler
// turns into 16-wide FMA on Zen 4 — and, per `code_dist` above, a wide tuple carries one importance
// weight for the whole tuple, so the weight cannot change the argmin and never enters here.
// ---------------------------------------------------------------------------------------------

// soa is [dim][code] for ONE variant: soa[d * k + c].
inline uint32_t wide_argmin(const float* soa, uint32_t k, uint32_t pair, const float* x,
                            float* out_d) {
  float bd = std::numeric_limits<float>::max();
  uint32_t best = 0;
  for (uint32_t c = 0; c < k; ++c) {
    float s = 0;
    for (uint32_t p = 0; p < pair; ++p) { const float d = soa[p * k + c] - x[p]; s += d * d; }
    if (s < bd) { bd = s; best = c; }
  }
  if (out_d) *out_d = bd;
  return best;
}

// Same, over an explicit candidate list.
inline uint32_t wide_argmin_list(const float* soa, uint32_t k, uint32_t pair, const float* x,
                                 const uint16_t* cand, uint32_t n, float* out_d) {
  float bd = std::numeric_limits<float>::max();
  uint32_t best = 0;
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t c = cand[i];
    float s = 0;
    for (uint32_t p = 0; p < pair; ++p) { const float d = soa[p * k + c] - x[p]; s += d * d; }
    if (s < bd) { bd = s; best = c; }
  }
  if (out_d) *out_d = bd;
  return best;
}

// Deinterleaves one variant of `cb` into the plane form the two above want.
void wide_planes(const Codebook& cb, uint32_t var, std::vector<float>* out) {
  out->assign((size_t)cb.pair * cb.k, 0.0f);
  for (uint32_t c = 0; c < cb.k; ++c)
    for (uint32_t p = 0; p < cb.pair; ++p) (*out)[(size_t)p * cb.k + c] = cb.entry(var, c)[p];
}

inline uint32_t best_code(const float* x, const Codebook& cb, uint32_t var,
                          double wa, double wb, double* out_d) {
  uint32_t best = 0;
  double bd = std::numeric_limits<double>::max();
  for (uint32_t j = 0; j < cb.k; ++j) {
    const double d = code_dist(x, cb.entry(var, j), cb.pair, wa, wb);
    if (d < bd) { bd = d; best = j; }
  }
  if (out_d) *out_d = bd;
  return best;
}

// Lloyd/k-means passes over the blocks assigned to `var`, refitting its centroids in place.
void refit_variant(Codebook& cb, uint32_t var,
                   const float* samples, const float* weights,
                   const std::vector<uint32_t>& blocks,
                   uint64_t items_per_block, uint32_t pair, uint32_t iters) {
  const uint32_t k = cb.k;
  float* cent = &cb.v[(size_t)var * k * pair];
  std::vector<double> acc((size_t)k * pair), cnt(k);
  std::vector<float> soa;
  for (uint32_t it = 0; it < iters; ++it) {
    std::fill(acc.begin(), acc.end(), 0.0);
    std::fill(cnt.begin(), cnt.end(), 0.0);
    if (pair >= 4) wide_planes(cb, var, &soa);
    for (uint32_t b : blocks) {
      const uint64_t base = (uint64_t)b * items_per_block;
      for (uint64_t i = 0; i < items_per_block; ++i) {
        const float* x = samples + (base + i) * pair;
        const double wt = weights ? (double)weights[base + i] : 1.0;
        // THE SYMMETRY GROUP BELONGS IN THE ENCODER, NOT IN THE FIT. Folding it into the Lloyd
        // assignment — matching a sample to the best (centroid, TRANSFORM) pair and accumulating
        // the transformed sample — is the textbook quotient k-means, and it is what a codebook
        // whose images are free "should" be fitted against. It is a clear regression, and seeding
        // k-means++ inside a fundamental domain to match is worse still. Fitting on the raw
        // symmetric samples leaves the codebook roughly symmetric, and the encoder's group search
        // then only ever improves on the image the fit already had.
        const uint32_t c = pair >= 4 ? wide_argmin(soa.data(), k, pair, x, nullptr)
                                     : best_code(x, cb, var, wt, wt, nullptr);
        for (uint32_t p = 0; p < pair; ++p) acc[(size_t)c * pair + p] += (double)x[p] * wt;
        cnt[c] += wt;
      }
    }
    double shift = 0;
    for (uint32_t c = 0; c < k; ++c) {
      if (cnt[c] <= 0) continue;
      for (uint32_t p = 0; p < pair; ++p) {
        const float nv = (float)(acc[(size_t)c * pair + p] / cnt[c]);
        shift += std::fabs(nv - cent[(size_t)c * pair + p]);
        cent[(size_t)c * pair + p] = nv;
      }
    }
    if (shift < 1e-7) break;
  }
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Codebook fitting
// ---------------------------------------------------------------------------------------------

Codebook fit_codebook(const float* samples, uint64_t n_samples,
                      const QuantSpec& spec, const float* weights,
                      uint32_t iters, uint64_t seed) {
  const uint32_t pair = spec.pair;
  const uint32_t k = spec.lut_size();
  const uint32_t nvar = spec.variants();
  const uint64_t items_per_block = spec.block / pair;
  const uint64_t n_items = n_samples / pair;

  Codebook cb;
  cb.pair = pair; cb.variants = nvar; cb.k = k;
  cb.v.assign((size_t)nvar * k * pair, 0.0f);
  if (n_items == 0 || k == 0 || items_per_block == 0) return cb;

  // ---- variant 0 ------------------------------------------------------------------------------
  if (pair == 1) {
    std::vector<float> sorted(samples, samples + n_samples);
    std::sort(sorted.begin(), sorted.end());
    for (uint32_t i = 0; i < k; ++i)
      cb.v[i] = sorted[std::min<uint64_t>(n_samples - 1, (uint64_t)(((i + 0.5) / k) * n_samples))];
  } else {
    // k-means++, with d2 carried forward rather than recomputed. Recomputing every chosen centroid
    // for every candidate is O(n k^2), which is invisible at k=32 and is 34 G distance evaluations
    // at k=1024. The running minimum is the same number by definition — d2[i] is the distance to
    // the NEAREST chosen centroid, and adding one centroid can only lower it.
    uint64_t z = seed;
    for (uint32_t p = 0; p < pair; ++p) cb.v[p] = samples[p];
    std::vector<double> d2(n_items, std::numeric_limits<double>::max());
    for (uint32_t c = 1; c < k; ++c) {
      const float* last = cb.entry(0, c - 1);
      double tot = 0;
      for (uint64_t i = 0; i < n_items; ++i) {
        const double d = code_dist(samples + i * pair, last, pair, 1.0, 1.0);
        if (d < d2[i]) d2[i] = d;
        tot += d2[i];
      }
      double target = (double)(splitmix(z) % 1000000) / 1e6 * tot, cum = 0;
      uint64_t pick = n_items - 1;
      for (uint64_t i = 0; i < n_items; ++i) { cum += d2[i]; if (cum >= target) { pick = i; break; } }
      for (uint32_t p = 0; p < pair; ++p)
        cb.v[(size_t)c * pair + p] = samples[pick * pair + p];
    }
  }

  const uint64_t n_blocks = n_items / items_per_block;
  if (n_blocks == 0) return cb;
  std::vector<uint32_t> all(n_blocks);
  for (uint64_t b = 0; b < n_blocks; ++b) all[b] = (uint32_t)b;
  refit_variant(cb, 0, samples, weights, all, items_per_block, pair, iters);
  if (pair == 1) std::sort(cb.v.begin(), cb.v.begin() + k);

  if (nvar == 1) return cb;

  // ---- extra variants: dither, then block-level EM ---------------------------------------------
  //
  // ik_llama's IQ4_KS uses a FIXED half-quantum offset for its second grid. We seed with that idea
  // but then let the data move both grids: each block picks its better variant (E-step), each
  // variant refits on the blocks that chose it (M-step).
  double gap = 0;
  if (pair == 1 && k > 1) {
    for (uint32_t i = 1; i < k; ++i) gap += cb.v[i] - cb.v[i - 1];
    gap /= (k - 1);
  } else {
    double sd = 0;
    for (uint64_t i = 0; i < n_samples; ++i) sd += (double)samples[i] * samples[i];
    gap = std::sqrt(sd / (double)n_samples) / std::sqrt((double)k);
  }
  // Under a POWER-OF-TWO block scale the variants have a second job, and it is the more valuable
  // one. An exponent-only scale is a grid an octave apart, so the block scale is badly quantised
  // where the shipped integer scale is nearly exact — and that gap is the whole of the pow2 form's
  // error regression. Seeding the variants as FRACTIONAL OCTAVES of the same fit makes (exponent,
  // variant) a single finer scale grid, and the EM below is still free to move them somewhere
  // better if shape pays more than scale.
  auto seed_variant = [&](uint32_t v, double mult) {
    for (uint32_t c = 0; c < k; ++c)
      for (uint32_t p = 0; p < pair; ++p)
        cb.v[((size_t)v * k + c) * pair + p] =
            cb.entry(0, c)[p] + (float)(gap * mult * (p == 0 ? 1.0 : -1.0));
  };
  auto seed_octave = [&](uint32_t v) {
    const float m = (float)std::exp2((double)v / (double)nvar);
    for (uint32_t c = 0; c < k; ++c)
      for (uint32_t p = 0; p < pair; ++p)
        cb.v[((size_t)v * k + c) * pair + p] = cb.entry(0, c)[p] * m;
  };
  for (uint32_t v = 1; v < nvar; ++v) {
    if (spec.pow2_block_scale) seed_octave(v); else seed_variant(v, 0.5 * v);
  }

  std::vector<std::vector<uint32_t>> members(nvar);
  std::vector<size_t> prev(nvar, (size_t)-1);
  std::vector<std::vector<float>> soa(nvar);
  const uint32_t em_rounds = 12u;
  for (uint32_t em = 0; em < em_rounds; ++em) {
    for (auto& m : members) m.clear();
    if (pair >= 4)
      for (uint32_t v = 0; v < nvar; ++v) wide_planes(cb, v, &soa[v]);
    for (uint64_t b = 0; b < n_blocks; ++b) {
      const uint64_t base = b * items_per_block;
      uint32_t bestv = 0; double bestE = std::numeric_limits<double>::max();
      for (uint32_t v = 0; v < nvar; ++v) {
        double e = 0;
        for (uint64_t i = 0; i < items_per_block; ++i) {
          const double wt = weights ? (double)weights[base + i] : 1.0;
          if (pair >= 4) {
            float d;
            wide_argmin(soa[v].data(), k, pair, samples + (base + i) * pair, &d);
            e += (double)d * wt;
          } else {
            double d; best_code(samples + (base + i) * pair, cb, v, wt, wt, &d);
            e += d;
          }
        }
        if (e < bestE) { bestE = e; bestv = v; }
      }
      members[bestv].push_back((uint32_t)b);
    }
    // An empty variant is re-seeded with a fresh dither rather than left degenerate — otherwise
    // it can never win a block again.
    for (uint32_t v = 0; v < nvar; ++v) {
      if (members[v].empty()) seed_variant(v, 0.25 * (v + 1));
      else refit_variant(cb, v, samples, weights, members[v], items_per_block, pair, 6);
    }
    std::vector<size_t> now(nvar);
    for (uint32_t v = 0; v < nvar; ++v) now[v] = members[v].size();
    if (now == prev) break;
    prev = now;
  }
  if (pair == 1)
    for (uint32_t v = 0; v < nvar; ++v)
      std::sort(cb.v.begin() + (size_t)v * k, cb.v.begin() + (size_t)(v + 1) * k);
  return cb;
}

void collect_normalized_samples(const float* w, uint64_t rows, uint64_t cols,
                                const QuantSpec& spec, std::vector<float>* out,
                                uint64_t max_samples,
                                const float* col_importance,
                                std::vector<float>* sample_weights) {
  out->clear();
  if (sample_weights) sample_weights->clear();
  const uint64_t nblk_row = cols / spec.block;
  const uint64_t total = rows * cols;
  const uint64_t stride_rows = std::max<uint64_t>(1, total / std::max<uint64_t>(1, max_samples));

  for (uint64_t r = 0; r < rows; r += stride_rows) {
    const float* row = w + r * cols;
    float rmax = 0;
    for (uint64_t c = 0; c < cols; ++c) rmax = std::max(rmax, std::fabs(row[c]));
    if (rmax <= 0) continue;
    for (uint64_t b = 0; b < nblk_row; ++b) {
      const float* blk = row + b * spec.block;
      float bmax = 0;
      for (uint32_t i = 0; i < spec.block; ++i) bmax = std::max(bmax, std::fabs(blk[i]));
      if (bmax <= 0) continue;
      for (uint32_t i = 0; i < spec.block; ++i) out->push_back(blk[i] / bmax);
      if (sample_weights) {
        if (spec.pair == 1) {
          for (uint32_t i = 0; i < spec.block; ++i)
            sample_weights->push_back(col_importance ? col_importance[b * spec.block + i] : 1.0f);
        } else {
          for (uint32_t i = 0; i < spec.block; i += spec.pair) {
            float s = 0;
            for (uint32_t p = 0; p < spec.pair; ++p)
              s += col_importance ? col_importance[b * spec.block + i + p] : 1.0f;
            sample_weights->push_back(s / (float)spec.pair);
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// What the GPU can actually hold, and how to search it quickly
// ---------------------------------------------------------------------------------------------

namespace {

// Exhaustive nearest over the 256 E4M3 codes. This runs 16384 times in a whole quantise, so there
// is nothing to gain from a rounding rule that has to be argued about instead of read.
struct E4M3Table {
  float v[256];
  E4M3Table() { for (int i = 0; i < 256; ++i) v[i] = fp8_e4m3_to_float((uint8_t)i); }
};
const E4M3Table& e4m3_table() { static const E4M3Table t; return t; }

uint8_t e4m3_nearest(float x, bool pow2_scale) {
  const E4M3Table& t = e4m3_table();
  int best = 0;
  float bd = std::numeric_limits<float>::max();
  for (int i = 0; i < 256; ++i) {
    if ((i & 0x7f) == 0x7f) continue;                        // NaN
    if (pow2_scale) {
      // Subnormal (E=0) has no leading one, so adding to the exponent field is simply wrong. At the
      // top the constraint is NaN, not overflow: E4M3-fn spells NaN 0x7F, so an entry at E=8 that
      // takes the largest block-scale add (7) and happens to carry mantissa 7 becomes a NaN and
      // poisons a whole WMMA accumulator. E<=7 puts the sum at 14 and makes that unreachable. It
      // costs nothing real — the fit is on blk/bmax, so entries live in [-1,1] and the octave
      // variants stretch that to at most 1.68, inside the 1.875 this leaves.
      const int e = (i >> 3) & 0xf;
      if (e == 0 || e > 7) continue;
    }
    const float d = std::fabs(t.v[i] - x);
    if (d < bd) { bd = d; best = i; }
  }
  return (uint8_t)best;
}

} // namespace

void project_codebook_e4m3(Codebook* cb, bool pow2_scale) {
  if (!cb) return;
  for (float& f : cb->v) f = e4m3_table().v[e4m3_nearest(f, pow2_scale)];
}

std::vector<uint32_t> pack_codebook_e4m3_quads(const Codebook& cb) {
  std::vector<uint32_t> out;
  if (cb.pair != 4) return out;
  out.assign((size_t)cb.variants * cb.k, 0u);
  for (uint32_t v = 0; v < cb.variants; ++v)
    for (uint32_t c = 0; c < cb.k; ++c) {
      const float* e = cb.entry(v, c);
      uint32_t w = 0;
      for (uint32_t p = 0; p < 4; ++p) w |= (uint32_t)e4m3_nearest(e[p], true) << (8 * p);
      out[(size_t)v * cb.k + c] = w;
    }
  return out;
}

void unpack_codebook_e4m3_quads(const uint32_t* packed, uint32_t variants, uint32_t k,
                                Codebook* out) {
  out->pair = 4; out->variants = variants; out->k = k;
  out->v.assign((size_t)variants * k * 4, 0.0f);
  for (uint32_t v = 0; v < variants; ++v)
    for (uint32_t c = 0; c < k; ++c) {
      const uint32_t w = packed[(size_t)v * k + c];
      for (uint32_t p = 0; p < 4; ++p)
        out->v[((size_t)v * k + c) * 4 + p] = e4m3_table().v[(w >> (8 * p)) & 0xffu];
    }
}

// A cell lists every entry that can be nearest for SOME point in it: keep entry c when its closest
// corner is no farther than the closest entry's FARTHEST corner. That is exact, and because the
// squared distance separates over dimensions the whole build is four adds per (cell, entry) rather
// than twenty operations — which is what takes 390625 cells from a minute to under a second.
void build_codebook_search(const Codebook& cb, CodebookSearch* out) {
  *out = CodebookSearch{};
  if (cb.pair != 4 || cb.k == 0 || cb.k > 65535 || cb.variants == 0) return;
  const uint32_t K = cb.k, V = cb.variants;
  constexpr int G = CodebookSearch::kCells;
  const float w = 2.0f * CodebookSearch::kRange / (float)G;

  out->pair = 4; out->k = K; out->variants = V;
  out->ent.assign((size_t)V * 4 * K, 0.0f);
  for (uint32_t v = 0; v < V; ++v)
    for (uint32_t c = 0; c < K; ++c)
      for (uint32_t d = 0; d < 4; ++d) out->ent[((size_t)v * 4 + d) * K + c] = cb.entry(v, c)[d];

  const size_t ncell = (size_t)G * G * G * G;
  out->start.assign((size_t)V * (ncell + 1), 0u);
  out->cand.clear();
  out->cand.reserve(ncell * 4);

  std::vector<float> dmin((size_t)4 * G * K), dmax((size_t)4 * G * K);
  std::vector<float> amin(K), amax(K), bmin(K), bmax(K);
  for (uint32_t v = 0; v < V; ++v) {
    for (uint32_t d = 0; d < 4; ++d)
      for (int i = 0; i < G; ++i) {
        const float a = -CodebookSearch::kRange + (float)i * w, b = a + w;
        for (uint32_t c = 0; c < K; ++c) {
          const float e = out->ent[((size_t)v * 4 + d) * K + c];
          const float lo = e < a ? a - e : (e > b ? e - b : 0.0f);
          const float hi = std::max(std::fabs(e - a), std::fabs(e - b));
          dmin[((size_t)d * G + i) * K + c] = lo * lo;
          dmax[((size_t)d * G + i) * K + c] = hi * hi;
        }
      }
    uint32_t* st = &out->start[(size_t)v * (ncell + 1)];
    size_t cell = 0;
    for (int i0 = 0; i0 < G; ++i0)
      for (int i1 = 0; i1 < G; ++i1) {
        const float* n0 = &dmin[(size_t)(0 * G + i0) * K], *x0 = &dmax[(size_t)(0 * G + i0) * K];
        const float* n1 = &dmin[(size_t)(1 * G + i1) * K], *x1 = &dmax[(size_t)(1 * G + i1) * K];
        for (uint32_t c = 0; c < K; ++c) { amin[c] = n0[c] + n1[c]; amax[c] = x0[c] + x1[c]; }
        for (int i2 = 0; i2 < G; ++i2) {
          const float* n2 = &dmin[(size_t)(2 * G + i2) * K], *x2 = &dmax[(size_t)(2 * G + i2) * K];
          for (uint32_t c = 0; c < K; ++c) { bmin[c] = amin[c] + n2[c]; bmax[c] = amax[c] + x2[c]; }
          for (int i3 = 0; i3 < G; ++i3, ++cell) {
            const float* n3 = &dmin[(size_t)(3 * G + i3) * K], *x3 = &dmax[(size_t)(3 * G + i3) * K];
            st[cell] = (uint32_t)out->cand.size();
            float r = std::numeric_limits<float>::max();
            for (uint32_t c = 0; c < K; ++c) r = std::min(r, bmax[c] + x3[c]);
            for (uint32_t c = 0; c < K; ++c)
              if (bmin[c] + n3[c] <= r) out->cand.push_back((uint16_t)c);
          }
        }
      }
    st[ncell] = (uint32_t)out->cand.size();
    // uint32 offsets. 390625 cells x 4 variants at a plausible list length is a few million; a
    // codebook degenerate enough to overflow would make the grid useless anyway, so refuse it
    // rather than wrap.
    if (out->cand.size() > 0xF0000000ull) { *out = CodebookSearch{}; return; }
  }
}

namespace {

// Exact nearest entry of variant `var` to the quad `x`. Out of grid range — which the scale search
// reaches only when a block's own maximum is more than an octave from the exponent under test —
// falls back to the full scan, still exact.
inline uint32_t quad_nn(const CodebookSearch& s, uint32_t var, const float* x, float* out_d) {
  constexpr int G = CodebookSearch::kCells;
  const float inv = (float)G / (2.0f * CodebookSearch::kRange);
  const float* soa = &s.ent[(size_t)var * 4 * s.k];
  int idx[4];
  for (int d = 0; d < 4; ++d) {
    const int i = (int)((x[d] + CodebookSearch::kRange) * inv);
    if (i < 0 || i >= G) return wide_argmin(soa, s.k, 4, x, out_d);
    idx[d] = i;
  }
  const size_t ncell = (size_t)G * G * G * G;
  const uint32_t* st = &s.start[(size_t)var * (ncell + 1)];
  const size_t cell = (((size_t)idx[0] * G + idx[1]) * G + idx[2]) * G + idx[3];
  const uint32_t a = st[cell], b = st[cell + 1];
  return wide_argmin_list(soa, s.k, 4, x, &s.cand[a], b - a, out_d);
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Quantise / dequantise
// ---------------------------------------------------------------------------------------------

bool quantize_matrix(const float* w, uint64_t rows, uint64_t cols,
                     const QuantSpec& spec, const Codebook& cb,
                     QuantizedMatrix* out, std::string* err,
                     const float* col_importance, const CodebookSearch* search) {
  auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
  if (spec.block == 0 || cols % spec.block != 0) return fail("cols must be a multiple of block");
  if (spec.pair != cb.pair) return fail("codebook pair-ness does not match spec");
  if (spec.pair == 0 || spec.block % spec.pair != 0) return fail("block must be a multiple of pair");
  if (cb.k != spec.lut_size()) return fail("codebook size != 2^code_bits");
  if (cb.variants != spec.variants()) return fail("codebook variants != 2^variant_bits");
  // A mantissa needs at least one exponent bit left over, and it means nothing without pow2.
  if (spec.scale_mant_bits && !spec.pow2_block_scale) return fail("scale_mant_bits needs pow2");
  if (spec.scale_mant_bits >= spec.scale_bits) return fail("scale_mant_bits >= scale_bits");

  if (spec.sign_bits > 1) return fail("sign_bits must be 0 or 1");
  if (spec.rot_bits > 1) return fail("rot_bits must be 0 or 1");
  // Only the 4-wide path implements it; the AVX-512 pair search below has no rotation.
  if (spec.rot_bits && spec.pair != 4) return fail("rot_bits is only implemented at pair 4");

  const uint32_t pair = spec.pair;
  const uint64_t nblk_row = cols / spec.block;
  const uint64_t codes_row = cols / pair;
  const uint64_t items_per_block = spec.block / pair;
  const uint32_t smax = (1u << spec.scale_bits) - 1u;
  const uint32_t fbits = spec.block_field_bits();
  const uint32_t vshift = spec.scale_bits + spec.sign_bits + spec.rot_bits;
  const uint32_t ngrp = spec.transforms();

  // Wide codes: the grid index if the caller built one, a plain SoA scan if not. Both are exact, so
  // an absent index costs time and nothing else — which is what makes it safe for a test to omit.
  std::vector<std::vector<float>> soa;
  const bool indexed = pair >= 4 && search && search->ready() && search->k == cb.k &&
                       search->variants == cb.variants;
  if (pair >= 4 && !indexed) {
    soa.resize(cb.variants);
    for (uint32_t v = 0; v < cb.variants; ++v) wide_planes(cb, v, &soa[v]);
  }
  auto quad_min = [&](uint32_t v, const float* x, float* d) -> uint32_t {
    return indexed ? quad_nn(*search, v, x, d) : wide_argmin(soa[v].data(), cb.k, pair, x, d);
  };
  constexpr int full_scale = 0;                      // pow2 exponent window; widening it buys nothing

  out->rows = rows; out->cols = cols; out->spec = spec;
  out->data.assign((rows * codes_row * spec.code_bits + 7) / 8, 0);
  out->scales.assign((rows * nblk_row * fbits + 7) / 8, 0);
  out->rscales.assign(rows * (spec.row_scale_f32 ? 4 : 2), 0);

  uint64_t used_alt = 0, nblocks_total = 0;

#if defined(__AVX512F__)
  // The fast path covers the pair-2 32-entry formats only; everything else, the shipped 4-wide one
  // included, keeps the scalar code. So a container's bytes do not depend on the host CPU, and
  // correctness never depends on the vectorisation.
  const bool simd = g_use_simd && (pair == 2 && cb.k == 32 && items_per_block == 8);
  std::vector<CbPlanes> planes;
  if (simd) build_planes(cb, &planes);
#endif

  for (uint64_t r = 0; r < rows; ++r) {
    const float* row = w + r * cols;
    float rmax = 0;
    for (uint64_t c = 0; c < cols; ++c) rmax = std::max(rmax, std::fabs(row[c]));
    // The pow2 BIAS lives here, not in the scale code. `block_scale_from_code` returns a bare 2^e
    // so that a GPU kernel can apply it by adding e to a packed E4M3 byte's exponent field and
    // nothing else; the row scale carries the offset that makes the largest code reproduce rmax.
    // Both halves of that are load-bearing: with the bias dropped from the code and NOT folded in
    // here, every block scale is 2^bias too large, the search picks exponent 0 for everything, the
    // variant split collapses and the error rises by more than half.
    const int pbias = block_scale_bias(spec);
    float rscale = (rmax > 0) ? (spec.pow2_block_scale ? std::ldexp(rmax, -pbias)
                                                       : rmax / (float)smax)
                              : 0.0f;
    if (spec.row_scale_f32) std::memcpy(&out->rscales[r * 4], &rscale, 4);
    else {
      // Search against the value that will be STORED, not the one that was computed. bf16 keeps 8
      // mantissa bits, so the two differ by up to 0.4% — small, and free to be rid of.
      const uint16_t h = float_to_bf16(rscale);
      std::memcpy(&out->rscales[r * 2], &h, 2);
      rscale = bf16_to_float(h);
    }

    for (uint64_t b = 0; b < nblk_row; ++b) {
      const float* blk = row + b * spec.block;
      float bmax = 0;
      for (uint32_t i = 0; i < spec.block; ++i) bmax = std::max(bmax, std::fabs(blk[i]));
      ++nblocks_total;

      uint32_t best_s = 0, best_v = 0, best_g = 0;
      double best_err = std::numeric_limits<double>::max();
      if (bmax > 0 && rscale > 0) {
        // The codebook was fitted on blk/bmax, so the natural scale is bmax itself. Search
        // integer scales around it JOINTLY with the variant, scoring true reconstruction error.
        // Search window must be RELATIVE, not a fixed number of integers: rscale = rmax/smax, so
        // a wider scale field makes `ideal` proportionally larger and a fixed +-4 window would
        // cover a narrower fraction of the range. That would silently bias any comparison between
        // configs with different scale_bits.
        // Power-of-two codes are few enough to score exhaustively — 2^scale_bits candidates, 4 at
        // the width that matters here — and the relative window below is meaningless for them
        // because the code is an exponent, not a multiple.
        const double ideal = (double)bmax / (double)rscale;
        const double win = std::max(4.0, ideal * 0.06);
        int lo = spec.pow2_block_scale ? 0 : std::max(1, (int)std::floor(ideal - win));
        int hi = spec.pow2_block_scale ? (int)smax
                                       : std::min<int>(smax, (int)std::ceil(ideal + win));
        // A power-of-two exponent within ONE OCTAVE of the block's own maximum. The codebook was
        // fitted on blk/bmax, so its support is about [-1,1]: two octaves low clips the block, two
        // high wastes most of the table, and scoring either costs exactly as much as the exponent
        // that wins. Variants add at most a fractional octave (see the octave seeding in
        // fit_codebook), so the window still covers their optimum.
        if (spec.pow2_block_scale && !full_scale) {
          uint32_t mid = 0;
          double bd = std::numeric_limits<double>::max();
          for (uint32_t s = 0; s <= smax; ++s) {
            const double d = std::fabs((double)block_scale_from_code(s, spec) - ideal);
            if (d < bd) { bd = d; mid = s; }
          }
          const int step = 1 << spec.scale_mant_bits;
          lo = std::max(0, (int)mid - step);
          hi = std::min<int>(smax, (int)mid + step);
        }
        for (uint32_t v = 0; v < cb.variants; ++v) {
          for (int s = lo; s <= hi; ++s) {
            const float bs = rscale * block_scale_from_code((uint32_t)s, spec);
            if (bs <= 0) continue;
            // Reciprocal once per candidate: the original divided every element by bs, 16 divides
            // per candidate where one reciprocal and 16 multiplies do the same job.
            const float inv = 1.0f / bs;
            for (uint32_t g = 0; g < ngrp; ++g) {
              const float sg = (g & 1u) ? -inv : inv;
              const uint32_t rh = (g & 2u) ? pair / 2u : 0u;
              double e = 0;
#if defined(__AVX512F__)
            if (simd) {
              float acc = 0.0f;
              for (uint64_t i = 0; i < 8; ++i) {
                const uint64_t c0 = b * spec.block + i * 2;
                const float wa = col_importance ? col_importance[c0] : 1.0f;
                const float wb = col_importance ? col_importance[c0 + 1] : wa;
                acc += min_dist_x32(planes[v], blk[i * 2] * sg, blk[i * 2 + 1] * sg, wa, wb);
              }
              e = (double)acc * (double)bs * (double)bs;
            } else
#endif
            if (pair >= 4) {
              // ONE importance weight for the whole tuple — it scales the tuple's error and cannot
              // reorder the codes within it, so the nearest entry is the unweighted one and the
              // grid index above is valid whether or not an imatrix is present.
              float acc = 0.0f;
              for (uint64_t i = 0; i < items_per_block; ++i) {
                float t[kMaxPair] = {0};
                for (uint32_t p = 0; p < pair; ++p) t[p] = blk[i * pair + (p + rh) % pair] * sg;
                float wt = 1.0f;
                if (col_importance) {
                  const uint64_t c0 = b * spec.block + i * pair;
                  wt = 0.0f;
                  for (uint32_t p = 0; p < pair; ++p) wt += col_importance[c0 + p];
                  wt /= (float)pair;
                }
                float d; quad_min(v, t, &d);
                acc += d * wt;
              }
              e = (double)acc * (double)bs * (double)bs;
            } else {
              for (uint64_t i = 0; i < items_per_block; ++i) {
                float t[kMaxPair] = {0};
                for (uint32_t p = 0; p < pair; ++p) t[p] = blk[i * pair + (p + rh) % pair] * sg;
                const uint64_t c0 = b * spec.block + i * pair;
                const double wa = col_importance ? col_importance[c0] : 1.0;
                const double wb = (pair == 2 && col_importance) ? col_importance[c0 + 1] : wa;
                double d; best_code(t, cb, v, wa, wb, &d);
                e += d * (double)bs * (double)bs;
              }
            }
              if (e < best_err) { best_err = e; best_s = (uint32_t)s; best_v = v; best_g = g; }
            }
          }
        }
      }
      if (best_v != 0) ++used_alt;
      pack_bits(out->scales, r * nblk_row + b, fbits,
                (best_v << vshift) | (best_g << spec.scale_bits) | best_s);

      const float mag = rscale * block_scale_from_code(best_s, spec);
      const float bs = (best_g & 1u) ? -mag : mag;      // bit 0 of the transform negates the block
      const uint32_t best_rh = (best_g & 2u) ? pair / 2u : 0u;
      const uint64_t code_base = r * codes_row + b * items_per_block;
      if (mag <= 0) {
        for (uint64_t i = 0; i < items_per_block; ++i)
          pack_bits(out->data, code_base + i, spec.code_bits, 0);
        continue;
      }
      for (uint64_t i = 0; i < items_per_block; ++i) {
        float t[kMaxPair] = {0};
        for (uint32_t p = 0; p < pair; ++p) t[p] = blk[i * pair + (p + best_rh) % pair] / bs;
        const uint64_t c0 = b * spec.block + i * pair;
        const double wa = col_importance ? col_importance[c0] : 1.0;
        const double wb = (pair == 2 && col_importance) ? col_importance[c0 + 1] : wa;
        pack_bits(out->data, code_base + i, spec.code_bits,
                  pair >= 4 ? quad_min(best_v, t, nullptr)
                            : best_code(t, cb, best_v, wa, wb, nullptr));
      }
    }
  }
  out->variant_split = nblocks_total ? (double)used_alt / (double)nblocks_total : 0.0;
  return true;
}

bool dequantize_matrix(const QuantizedMatrix& q, const Codebook& cb, float* out,
                       std::string* err) {
  auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
  const QuantSpec& spec = q.spec;
  if (cb.k != spec.lut_size()) return fail("codebook size mismatch");
  if (cb.variants != spec.variants()) return fail("codebook variant mismatch");

  const uint32_t pair = spec.pair;
  const uint64_t nblk_row = q.cols / spec.block;
  const uint64_t codes_row = q.cols / pair;
  const uint64_t items_per_block = spec.block / pair;
  const uint32_t fbits = spec.block_field_bits();
  const uint32_t smask = (1u << spec.scale_bits) - 1u;

  for (uint64_t r = 0; r < q.rows; ++r) {
    float rscale;
    if (spec.row_scale_f32) std::memcpy(&rscale, &q.rscales[r * 4], 4);
    else { uint16_t h; std::memcpy(&h, &q.rscales[r * 2], 2); rscale = bf16_to_float(h); }

    float* orow = out + r * q.cols;
    for (uint64_t b = 0; b < nblk_row; ++b) {
      const uint32_t field = unpack_bits(q.scales, r * nblk_row + b, fbits);
      const uint32_t s = field & smask;
      const uint32_t g = (field >> spec.scale_bits) & (spec.transforms() - 1u);
      const uint32_t v = std::min(field >> (spec.scale_bits + spec.sign_bits + spec.rot_bits),
                                  cb.variants - 1);
      const float bs = rscale * block_scale_from_code(s, spec) * ((g & 1u) ? -1.0f : 1.0f);
      const uint32_t rh = (g & 2u) ? spec.pair / 2u : 0u;
      const uint64_t code_base = r * codes_row + b * items_per_block;
      float* ob = orow + b * spec.block;
      for (uint64_t i = 0; i < items_per_block; ++i) {
        const uint32_t c = unpack_bits(q.data, code_base + i, spec.code_bits);
        const float* e = cb.entry(v, c);
        // The rotation applies to the RECONSTRUCTION, so position p takes entry (p + rh) % pair —
        // the same permutation the kernel gets from a funnel shift of the packed dword.
        for (uint32_t p = 0; p < pair; ++p) ob[i * pair + p] = bs * e[(p + rh) % pair];
      }
    }
  }
  return true;
}

QuantError measure_error(const float* ref, const float* got,
                         uint64_t rows, uint64_t cols, uint32_t probes, uint64_t seed,
                         const float* col_importance) {
  QuantError e;
  double num = 0, den = 0, mx = 0;
  const uint64_t n = rows * cols;
  for (uint64_t i = 0; i < n; ++i) {
    const double d = (double)ref[i] - (double)got[i];
    num += d * d; den += (double)ref[i] * (double)ref[i];
    mx = std::max(mx, std::fabs(d));
  }
  e.rel_fro = den > 0 ? std::sqrt(num / den) : 0;
  e.rmse = std::sqrt(num / (double)n);
  e.max_abs = mx;

  if (col_importance) {
    double wn = 0, wd = 0;
    for (uint64_t r = 0; r < rows; ++r)
      for (uint64_t c = 0; c < cols; ++c) {
        const uint64_t i = r * cols + c;
        const double d = (double)ref[i] - (double)got[i];
        wn += col_importance[c] * d * d;
        wd += col_importance[c] * (double)ref[i] * (double)ref[i];
      }
    e.rel_fro_w = wd > 0 ? std::sqrt(wn / wd) : 0;
  }

  uint64_t z = seed;
  std::vector<float> x(cols);
  double onum = 0, oden = 0;
  for (uint32_t p = 0; p < probes; ++p) {
    for (uint64_t c = 0; c < cols; ++c) {
      const double u1 = ((splitmix(z) >> 11) + 1) * (1.0 / 9007199254740992.0);
      const double u2 = ((splitmix(z) >> 11) + 1) * (1.0 / 9007199254740992.0);
      x[c] = (float)(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
    }
    for (uint64_t r = 0; r < rows; ++r) {
      double a = 0, b = 0;
      const float* rr = ref + r * cols;
      const float* gg = got + r * cols;
      for (uint64_t c = 0; c < cols; ++c) { a += (double)rr[c] * x[c]; b += (double)gg[c] * x[c]; }
      onum += (a - b) * (a - b); oden += a * a;
    }
  }
  e.rel_out = oden > 0 ? std::sqrt(onum / oden) : 0;
  return e;
}

} // namespace aff
