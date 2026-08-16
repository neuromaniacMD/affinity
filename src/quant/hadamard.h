// Fast Walsh-Hadamard transform and incoherence processing (QuIP#-style).
//
// Why: block-scaled quantisation error is driven by the ratio max|w| / rms(w) inside each block.
// An orthogonal random rotation spreads outliers across dimensions, pushing the distribution
// toward Gaussian and shrinking that ratio. Because H is orthogonal and (normalised) involutive,
//
//     W x = (W Hᵀ)(H x)
//
// the weights can be rotated once, offline, and the activation rotated online per layer.
//
// Cost at inference: n log2(n) adds. For DeepSeek-V4-Flash that is 4096*12 + 2048*11 ≈ 72k ops
// per token per layer, against ~25M MACs for one expert — i.e. free.
//
// Caveat for this project: the checkpoint is ALREADY MXFP4. Rotation cannot undo DeepSeek's
// quantisation error, only reduce the error we add on top. Whether that is worth the runtime
// FWHT is an empirical question — see tools/aff_quantlab.cpp.

#pragma once

#include <cstdint>
#include <cstddef>

namespace aff {

// In-place FWHT over `n` contiguous floats. `n` must be a power of two.
// Unnormalised: applying twice scales by n. Use fwht_normalized for an involution.
void fwht(float* x, uint64_t n) noexcept;

// In-place, scaled by 1/sqrt(n) so that applying it twice is the identity.
void fwht_normalized(float* x, uint64_t n) noexcept;

inline bool is_pow2(uint64_t n) noexcept { return n && (n & (n - 1)) == 0; }

// A deterministic random sign vector, used to build a randomised Hadamard transform
// H_s = H * diag(s). Randomisation matters: a plain Hadamard has structure that can align
// badly with weight structure. `seed` must match between the offline weight rotation and the
// online activation rotation.
void random_signs(int8_t* s, uint64_t n, uint64_t seed) noexcept;

// Applies diag(s) then FWHT, in place, per row.
void randomized_hadamard_rows(float* x, uint64_t rows, uint64_t n,
                              const int8_t* signs, bool normalized = true) noexcept;

// --- diagnostics -------------------------------------------------------------------------------
// These are what actually predict block-quantisation error.
struct IncoherenceStats {
  double rms = 0;
  double amax = 0;
  double max_over_rms = 0;   // the number that matters for block scaling
  double kurtosis = 0;       // 3.0 for a Gaussian; higher means heavier tails
  double mean_block_max_over_rms = 0;  // averaged over blocks of `block` elements
};
IncoherenceStats incoherence_stats(const float* x, uint64_t n, uint64_t block = 32) noexcept;

} // namespace aff
