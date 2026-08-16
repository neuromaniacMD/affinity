#include "evalharness.h"
#include "source_dtypes.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace aff {

namespace {

inline uint64_t splitmix(uint64_t& z) {
  uint64_t x = (z += 0x9E3779B97F4A7C15ull);
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
inline double gauss(uint64_t& z) {
  const double u1 = ((splitmix(z) >> 11) + 1) * (1.0 / 9007199254740992.0);
  const double u2 = ((splitmix(z) >> 11) + 1) * (1.0 / 9007199254740992.0);
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

void pack_bits(std::vector<uint8_t>& dst, uint64_t index, uint32_t bits, uint32_t value) {
  const uint64_t bitpos = index * bits;
  uint64_t byte = bitpos >> 3;
  uint32_t off = static_cast<uint32_t>(bitpos & 7), rem = bits;
  while (rem > 0) {
    const uint32_t take = std::min(rem, 8u - off);
    const uint32_t mask = ((1u << take) - 1u);
    dst[byte] = static_cast<uint8_t>((dst[byte] & ~(mask << off)) | ((value & mask) << off));
    value >>= take; rem -= take; ++byte; off = 0;
  }
}

inline double cd(const float* x, const float* e, uint32_t pair) {
  if (pair == 1) { const double d = (double)x[0] - e[0]; return d * d; }
  const double dx = (double)x[0] - e[0], dy = (double)x[1] - e[1];
  return dx * dx + dy * dy;
}

} // namespace

Activations make_synthetic_activations(uint64_t d_in, uint64_t n_tok, double corr, uint64_t seed) {
  Activations a;
  a.d_in = d_in; a.n_tok = n_tok;
  a.x.assign(n_tok * d_in, 0.0f);
  uint64_t z = seed;

  // Per-channel scale with a power-law tail plus a handful of large outlier channels — the
  // structure that makes real activation covariance far from diagonal.
  std::vector<double> chan(d_in);
  for (uint64_t j = 0; j < d_in; ++j) chan[j] = 1.0 / std::sqrt(1.0 + 0.002 * (double)j);
  for (int k = 0; k < 8; ++k) chan[splitmix(z) % d_in] *= 12.0;

  for (uint64_t t = 0; t < n_tok; ++t) {
    float* row = a.x.data() + t * d_in;
    // AR(1) across channels gives a Toeplitz-ish covariance with real off-diagonal mass.
    double prev = gauss(z);
    for (uint64_t j = 0; j < d_in; ++j) {
      const double e = gauss(z);
      prev = corr * prev + std::sqrt(std::max(0.0, 1.0 - corr * corr)) * e;
      row[j] = (float)(prev * chan[j]);
    }
  }
  return a;
}

std::vector<double> gram_matrix(const Activations& a, double damp) {
  const uint64_t d = a.d_in;
  std::vector<double> H((size_t)d * d, 0.0);
  for (uint64_t t = 0; t < a.n_tok; ++t) {
    const float* x = a.token(t);
    for (uint64_t i = 0; i < d; ++i) {
      const double xi = x[i];
      if (xi == 0.0) continue;
      double* Hi = H.data() + (size_t)i * d;
      for (uint64_t j = 0; j < d; ++j) Hi[j] += xi * x[j];
    }
  }
  double md = 0;
  for (uint64_t i = 0; i < d; ++i) md += H[(size_t)i * d + i];
  md /= (double)d;
  const double lam = damp * (md > 0 ? md : 1.0);
  for (uint64_t i = 0; i < d; ++i) H[(size_t)i * d + i] += lam;
  return H;
}

double output_error(const float* ref, const float* got,
                    uint64_t rows, uint64_t cols, const Activations& a) {
  if (a.d_in != cols || a.n_tok == 0) return -1.0;
  double num = 0, den = 0;
  std::vector<double> yr(rows), yg(rows);
  for (uint64_t t = 0; t < a.n_tok; ++t) {
    const float* x = a.token(t);
    for (uint64_t r = 0; r < rows; ++r) {
      const float* wr = ref + r * cols;
      const float* gr = got + r * cols;
      double ar = 0, ag = 0;
      for (uint64_t c = 0; c < cols; ++c) { ar += (double)wr[c] * x[c]; ag += (double)gr[c] * x[c]; }
      const double d = ar - ag;
      num += d * d; den += ar * ar;
    }
  }
  return den > 0 ? std::sqrt(num / den) : 0.0;
}

// ---------------------------------------------------------------------------------------------
// LDLQ
// ---------------------------------------------------------------------------------------------

bool quantize_matrix_ldlq(const float* w, uint64_t rows, uint64_t cols,
                          const QuantSpec& spec, const Codebook& cb,
                          const std::vector<double>& H,
                          QuantizedMatrix* out, std::string* err,
                          const LdlqOptions& opt) {
  auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
  if (H.size() != (size_t)cols * cols) return fail("Hessian must be cols x cols");
  if (spec.pair != cb.pair) return fail("codebook pair-ness mismatch");
  if (cb.k != spec.lut_size()) return fail("codebook size mismatch");
  if (cols % spec.block != 0) return fail("cols must be a multiple of block");

  const uint32_t pair = spec.pair;
  const uint64_t nblk_row = cols / spec.block;
  const uint64_t codes_row = cols / pair;
  const uint64_t items_per_block = spec.block / pair;
  const uint32_t smax = (1u << spec.scale_bits) - 1u;
  const uint32_t fbits = spec.block_field_bits();

  out->rows = rows; out->cols = cols; out->spec = spec;
  out->data.assign((rows * codes_row * spec.code_bits + 7) / 8, 0);
  out->scales.assign((rows * nblk_row * fbits + 7) / 8, 0);
  out->rscales.assign(rows * (spec.row_scale_f32 ? 4 : 2), 0);

  // --- Reverse Cholesky: H = U D U^T with U unit upper-triangular ------------------------------
  //
  // TRAP 1: a plain lower Cholesky diverges here. GPTQ processes columns in order and needs the
  // factor that expresses column j's influence on columns > j, which is the UDU^T orientation.
  const uint64_t n = cols;
  std::vector<double> U((size_t)n * n, 0.0), Dg(n, 0.0);
  {
    // Standard UDU^T via backward recursion.
    for (int64_t j = (int64_t)n - 1; j >= 0; --j) {
      double d = H[(size_t)j * n + j];
      for (int64_t k = j + 1; k < (int64_t)n; ++k) d -= Dg[k] * U[(size_t)j * n + k] * U[(size_t)j * n + k];
      if (d <= 1e-12) d = 1e-12;
      Dg[j] = d;
      for (int64_t i = j - 1; i >= 0; --i) {
        double s = H[(size_t)i * n + j];
        for (int64_t k = j + 1; k < (int64_t)n; ++k)
          s -= Dg[k] * U[(size_t)i * n + k] * U[(size_t)j * n + k];
        U[(size_t)i * n + j] = s / d;
      }
      U[(size_t)j * n + j] = 1.0;
    }
  }

  std::vector<double> wrow(cols), err_acc(cols);
  std::vector<float> qrow(cols);

  for (uint64_t r = 0; r < rows; ++r) {
    const float* src = w + r * cols;
    for (uint64_t c = 0; c < cols; ++c) wrow[c] = src[c];
    std::fill(err_acc.begin(), err_acc.end(), 0.0);

    // Row scale from the ORIGINAL row (feedback must not move the scale).
    float rmax = 0;
    for (uint64_t c = 0; c < cols; ++c) rmax = std::max(rmax, std::fabs(src[c]));
    const float rscale = (rmax > 0) ? rmax / (float)smax : 0.0f;
    if (spec.row_scale_f32) std::memcpy(&out->rscales[r * 4], &rscale, 4);
    else { const uint16_t h = float_to_bf16(rscale); std::memcpy(&out->rscales[r * 2], &h, 2); }

    for (uint64_t b = 0; b < nblk_row; ++b) {
      const uint64_t c0 = b * spec.block;
      // Block scale + variant chosen on the ORIGINAL block, so feedback cannot destabilise it.
      float bmax = 0;
      for (uint32_t i = 0; i < spec.block; ++i) bmax = std::max(bmax, std::fabs(src[c0 + i]));
      uint32_t best_s = 0, best_v = 0;
      if (bmax > 0 && rscale > 0) {
        const double ideal = (double)bmax / (double)rscale;
        const double win = std::max(4.0, ideal * 0.06);
        const int lo = std::max(1, (int)std::floor(ideal - win));
        const int hi = std::min<int>(smax, (int)std::ceil(ideal + win));
        double be = std::numeric_limits<double>::max();
        for (uint32_t v = 0; v < cb.variants; ++v)
          for (int s = lo; s <= hi; ++s) {
            const float bs = rscale * (float)s;
            if (bs <= 0) continue;
            double e = 0;
            for (uint64_t i = 0; i < items_per_block; ++i) {
              float t[2] = {0, 0};
              for (uint32_t p = 0; p < pair; ++p) t[p] = src[c0 + i * pair + p] / bs;
              double bd = std::numeric_limits<double>::max();
              for (uint32_t j = 0; j < cb.k; ++j) bd = std::min(bd, cd(t, cb.entry(v, j), pair));
              e += bd * (double)bs * (double)bs;
            }
            if (e < be) { be = e; best_s = (uint32_t)s; best_v = v; }
          }
      }
      pack_bits(out->scales, r * nblk_row + b, fbits, (best_v << spec.scale_bits) | best_s);
      const float bs = rscale * (float)best_s;

      for (uint64_t i = 0; i < items_per_block; ++i) {
        const uint64_t j0 = c0 + i * pair;
        // Quantise the ERROR-CORRECTED value...
        float t[2] = {0, 0};
        if (bs > 0)
          for (uint32_t p = 0; p < pair; ++p)
            t[p] = (float)((wrow[j0 + p] + err_acc[j0 + p]) / bs);

        uint32_t code = 0;
        if (bs > 0) {
          double bd = std::numeric_limits<double>::max();
          for (uint32_t j = 0; j < cb.k; ++j) {
            const double d = cd(t, cb.entry(best_v, j), pair);
            if (d < bd) { bd = d; code = j; }
          }
        }
        pack_bits(out->data, r * codes_row + j0 / pair, spec.code_bits, code);

        // ...but measure the residual against the ORIGINAL W.
        // TRAP 2: using the corrected value here makes the feedback compound and blow up.
        for (uint32_t p = 0; p < pair; ++p) {
          const uint64_t j = j0 + p;
          const double rec = (bs > 0) ? (double)bs * cb.entry(best_v, code)[p] : 0.0;
          const double e = wrow[j] - rec;
          // Push the residual into the not-yet-quantised columns via U.
          const double* Uj = U.data() + (size_t)j * n;
          for (uint64_t k = j + 1; k < n; ++k) err_acc[k] += e * Uj[k];
        }
      }
    }
  }
  (void)opt;
  return true;
}

} // namespace aff
