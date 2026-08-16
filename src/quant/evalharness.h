// Activation-based quantisation evaluation, and error-feedback rounding (LDLQ / GPTQ).
//
// WHY THIS EXISTS:
//
// The diagonal-imatrix metric `rel_fro_w` is *provably minimised* by nearest-neighbour rounding —
// h_j > 0 is a constant multiplier within a column, so argmin_c h_j(w-c)^2 == argmin_c (w-c)^2.
// It therefore cannot evaluate any method that deliberately makes non-nearest choices: LDLQ with a
// full Hessian scores worse on rel_fro_w while cutting true output error by more than half.
//
// The true objective is
//     ||(W - W') X||_F / ||W X||_F      with X = real input activations [d_in, n_tok]
// which equals tr(E H E^T) for H = X X^T, but is computed directly from X — exact, and cheaper
// to store than H whenever n_tok < d_in.
//
// Activations come from a calibration pass over the real model (ds4 hook at ds4.c:10761,
// `layer_routed_moe_one`). Until those exist, `make_synthetic_activations` produces X with a
// realistic correlated spectrum, which is enough to validate that LDLQ works at all — an
// uncorrelated X would make LDLQ provably a no-op.

#pragma once

#include "blockquant.h"

#include <cstdint>
#include <string>
#include <vector>

namespace aff {

// Column-major-ish activation block: `n_tok` samples of a `d_in`-dimensional input,
// stored row-major as [n_tok][d_in] so a token is contiguous.
struct Activations {
  std::vector<float> x;
  uint64_t n_tok = 0;
  uint64_t d_in  = 0;
  const float* token(uint64_t t) const { return x.data() + t * d_in; }
};

// Synthetic activations with a power-law covariance spectrum plus a few outlier channels —
// the structure real transformer activations actually have. `corr` in [0,1) sets how strongly
// neighbouring channels co-vary; corr=0 gives independent channels, where LDLQ is a provable
// no-op and the diagonal metric is exactly right.
Activations make_synthetic_activations(uint64_t d_in, uint64_t n_tok, double corr,
                                       uint64_t seed = 1234);

// H = X^T X  (d_in x d_in), plus `damp` * mean(diag) on the diagonal.
// GPTQ/LDLQ need this positive-definite; damping is what makes a rank-deficient X usable.
std::vector<double> gram_matrix(const Activations& a, double damp = 0.01);

// True output error: ||(W - W') X^T||_F / ||W X^T||_F, computed directly from activations.
// This is the metric that ranks error-compensating methods correctly.
double output_error(const float* ref, const float* got,
                    uint64_t rows, uint64_t cols, const Activations& a);

// ---------------------------------------------------------------------------------------------
// LDLQ / GPTQ-style error-feedback rounding
// ---------------------------------------------------------------------------------------------
//
// Quantises columns in order; after fixing column j, the residual (w_j - w'_j) is pushed into the
// not-yet-quantised columns weighted by H's off-diagonal structure. With a DIAGONAL H this
// degenerates exactly to nearest-neighbour (QuIP states this outright), so it is only worth
// running with a genuine full Hessian.
//
// Two traps that are easy to get wrong, both guarded in the implementation:
//   * the factorisation must be the reverse-Cholesky (UDU^T) orientation; a plain cholesky(H)
//     diverges;
//   * the error term must use the ORIGINAL W, not the already-corrected value.
struct LdlqOptions {
  double damp = 0.01;      // diagonal damping as a fraction of mean(diag(H))
  bool   verbose = false;
};

// Same contract as quantize_matrix, but rounds with error feedback against `H`.
// `H` must be `cols x cols`, row-major, as returned by gram_matrix().
bool quantize_matrix_ldlq(const float* w, uint64_t rows, uint64_t cols,
                          const QuantSpec& spec, const Codebook& cb,
                          const std::vector<double>& H,
                          QuantizedMatrix* out, std::string* err,
                          const LdlqOptions& opt = {});

} // namespace aff
