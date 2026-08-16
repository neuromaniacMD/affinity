// CPU expert kernel — the DRAM-bound path that sets decode speed.
//
// Computes  y += scale * (W · x)  for one AFF_Q2P875 expert matrix, fusing dequantisation into
// the GEMV so weights are touched exactly once. This is the hot loop the whole engine is built
// around.
//
// IT IS COMPUTE-BOUND, NOT BANDWIDTH-BOUND. Reading the same cold pool two ways at the same thread
// counts, a plain streaming sum is an order of magnitude faster per core than this kernel, and the
// all-core plateau sits below what the memory system delivers: the plateau is an instruction
// ceiling.
//
// The consequence for anyone changing this file: operation count is the only thing that moves it.
// Prefetching, layout and thread placement have nothing to recover.
//
// FORMAT — per matrix, three separate planes:
//
//   data     5 bits per PAIR of weights, packed contiguously   (2.5 bits/weight)
//   scales   6 bits per 16-weight block: [variant:2][scale:4]
//   rscales  one bf16 per row
//
//   w[2i], w[2i+1]  =  rscale[row] * blockscale * LUT[variant][code]
//
// Decode is a 2-D table lookup into a 4×32×2 float codebook (1 KB — L1-resident), never a memory
// gather. There is no "free" decode budget hiding behind a DRAM wall: the kernel never reaches one.

#pragma once

#include "quant/blockquant.h"

#include <cstdint>
#include <cstddef>

namespace aff {

// A quantised expert matrix as it sits in the .aff file — pointers into mmap'd/huge-page memory,
// never copied.
struct ExpertMatrixView {
  const uint8_t* data    = nullptr;   // packed pair codes
  const uint8_t* scales  = nullptr;   // packed [variant:2][scale:4] per block
  const uint8_t* rscales = nullptr;   // bf16 per row
  uint64_t rows = 0, cols = 0;
  QuantSpec spec{};
};

// y[0..rows) += W · x[0..cols)
//
// `cb` must be the codebook this matrix was quantised with (4 variants × 32 entries × 2 floats).
// `x` is fp32. Accumulates into y so several experts can be summed without a temporary.
void expert_gemv(const ExpertMatrixView& m, const Codebook& cb,
                 const float* x, float* y) noexcept;

// Same, scaled by `w` — the router weight. Fused so the output is touched once per expert.
void expert_gemv_scaled(const ExpertMatrixView& m, const Codebook& cb,
                        const float* x, float w, float* y) noexcept;

// Multi-threaded over ROWS. Rows are independent and each writes only its own y[r], so this needs
// no reduction and no locking. This is the op that dominates a decode step: six experts x three
// matrices x 8.4 M weights is ~150 M MACs per layer against ~140 M for everything else combined.
//
// ONE MATRIX PER PARALLEL REGION, which is why the hybrid path does not use this. A region is not
// free to open and close — tens of microseconds — and a layer's non-resident share is three per
// expert, so at that granularity the dispatch overhead alone is milliseconds a token.
// `StaticPlacement::run` slices the rows itself
// with `expert_row_slice` and puts the whole set in two regions instead.
void expert_gemv_scaled_mt(const ExpertMatrixView& m, const Codebook& cb,
                           const float* x, float w, float* y);

// Rows [lo, hi) of `m` as a matrix in its own right.
//
// Every plane is row-major with a fixed stride, so a row range is a contiguous byte range and no
// bit offset has to be re-derived. Row-aligned by construction for the locked format: 10240 code
// bits and 1536 field bits per row, both whole bytes, so a row offset never lands mid-byte.
ExpertMatrixView expert_row_slice(const ExpertMatrixView& m, uint64_t lo, uint64_t hi) noexcept;

// Reference implementation: dequantise to fp32, then a plain GEMV. Slow, obviously correct,
// used to validate the fused path.
void expert_gemv_reference(const ExpertMatrixView& m, const Codebook& cb,
                           const float* x, float* y) noexcept;

// CPU kernel variant, for A/B measurement only. 1 is the scalar reference; 2 adds a second
// accumulator and one field load per block pair; 3-5 decode the bit-packed codes and fields with
// AVX-512 VBMI multishift. 4 is the shipped default.
void cpu_kernel_variant(int v);
int  cpu_kernel_variant();

// Builds a view over a QuantizedMatrix produced by the quantiser (for tests and the offline path).
ExpertMatrixView view_of(const QuantizedMatrix& q) noexcept;

// Bytes of weight data this matrix streams per full pass.
uint64_t expert_stream_bytes(const ExpertMatrixView& m) noexcept;

} // namespace aff
