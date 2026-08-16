#pragma once
// WMMA fragments for gfx1201 (RDNA4, wave32). One place, so no kernel hand-rolls the builtin.
//
// ---- the lane->element mapping, which AMD does not document -------------------------------------
//
// ROCm#6025 asked for it and was closed with no answer. `rdna4-gfx1201.md` §3.4 determined it
// empirically against a CPU reference with ASYMMETRIC matrices; `tests/test_wmma.hip` is that test,
// and it is the reason to trust anything below. With lane `l`, `c = l % 16`, `h = l / 16`:
//
//   A, row-major A[m][k]     lane l holds A[c][h*8 + j],  j = 0..7
//   B, row-major B[k][n]     lane l holds B[h*8 + j][c],  j = 0..7
//   C/D accumulator          acc[j] IS D[h*8 + j][c]
//
// The accumulator is COLUMN-distributed: a lane owns one column and eight consecutive rows, lanes
// 0-15 rows 0-7 and lanes 16-31 rows 8-15. The intuitive "a lane owns a row" is wrong, produces a
// silently transposed tile, with no compile or runtime error.
//
// IDENTITY, DIAGONAL AND SYMMETRIC TEST MATRICES CANNOT TELL THE TWO APART — both give 0 errors.
// Any test of fragment code here must use asymmetric operands. That trap has its own case in
// `tests/test_wmma.hip`.
//
// ---- the K labelling ---------------------------------------------------------------------------
//
// The canonical hardware K-order is not contiguous — lanes 0-15 really hold
// k = 0,1,2,3,8,9,10,11. Calling slot j of lane-half h "k = 8h + j" as above is a permutation of K
// applied IDENTICALLY to A and B, and D = sum_k A[m][k] B[k][n] is invariant under that, so the
// product is bit-identical. Two rules keep it that way:
//
//   1. Both operands must use the labelling in this header. If one ever comes from
//      GLOBAL_LOAD_TR_B128 (which emits the canonical order) and the other from a plain load, the K
//      labels do not line up and the result is silently wrong.
//   2. The sparse path may NOT use this labelling: SWMMAC's index bits are tied to the hardware K
//      groups. Anything using swmmac must use the canonical order instead.
//
// ---- hazards -----------------------------------------------------------------------------------
//
// §7.12.1: a WMMA whose A, B or index overlaps the PREVIOUS WMMA's D needs at least one
// independent VALU instruction in between — "required for correct function", not a perf hint. The
// compiler inserts it for these intrinsics; hand-written asm must not skip it. Feeding D straight
// back as C of the same opcode is the fast path and is exempt, which is why the kernels here keep
// one long accumulator chain of a single opcode rather than mixing WMMA types in the inner loop.
//
// And WMMA does NOT co-issue with VALU on RDNA4, for every opcode. Dequantisation
// and address math inside a wave are serial with its matrix work; the only way to hide them is to
// have other waves resident, which is the argument for staying at or under 96 VGPRs.

#include <cstdint>

namespace aff {
namespace wmma {

// A/B operand widths are the RDNA4 ones — half of RDNA3's, because RDNA4 dropped the cross-half
// replication. Passing a gfx11-width vector compiles for gfx1100 and fails here, which is the
// intended way to find out you used the unsuffixed builtin.
typedef short s8v __attribute__((ext_vector_type(8)));  // bf16 A or B, 4 VGPRs
typedef int   i2v __attribute__((ext_vector_type(2)));  // fp8/bf8/int8 A or B, 2 VGPRs
typedef float f8v __attribute__((ext_vector_type(8)));  // f32 accumulator, 8 VGPRs
typedef int   i8v __attribute__((ext_vector_type(8)));  // i32 accumulator, 8 VGPRs

constexpr uint32_t kM = 16, kN = 16, kK = 16;

// ---- the mma itself ----------------------------------------------------------------------------
//
// Always the _gfx12-suffixed builtin. The unsuffixed names are the RDNA3 forms with different
// operand widths; they do not compile for gfx1201, which is the good outcome.

__device__ __forceinline__ f8v mma_bf16(s8v a, s8v b, f8v c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}
__device__ __forceinline__ f8v mma_fp8(i2v a, i2v b, f8v c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}
// Signed both sides, no clamp: the accumulator is i32 and K is 16, so int8 x int8 x 16 peaks at
// 2^21 and cannot overflow. Clamping would cost nothing but says something untrue about the range.
__device__ __forceinline__ i8v mma_i8(i2v a, i2v b, i8v c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
}

// ---- element conversion ------------------------------------------------------------------------

// Round-to-nearest-even, not truncation. Truncating is one instruction cheaper and biases every
// weight toward zero; on a 2.875-bpw model the dequantised values are already the error budget.
__device__ __forceinline__ uint16_t f32_to_bf16(float f) {
  const uint32_t u = __float_as_uint(f);
  if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu)) return (uint16_t)((u >> 16) | 0x40u);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float bf16_to_f32(uint16_t h) {
  return __uint_as_float((uint32_t)h << 16);
}

// ---- fragment indexing -------------------------------------------------------------------------
//
// Spelled out rather than inlined into the loaders because kernels that stage through LDS need the
// indices themselves to compute a swizzle.

__device__ __forceinline__ uint32_t frag_c(uint32_t lane) { return lane & 15u; }
__device__ __forceinline__ uint32_t frag_k0(uint32_t lane) { return (lane >> 4) * 8u; }
// Row of D that accumulator slot `j` holds; the column is frag_c(lane).
__device__ __forceinline__ uint32_t acc_row(uint32_t lane, uint32_t j) {
  return (lane >> 4) * 8u + j;
}

__device__ __forceinline__ f8v acc_zero() {
  f8v z;
#pragma unroll
  for (int j = 0; j < 8; ++j) z[j] = 0.0f;
  return z;
}

// ---- loads from a row-major tile ---------------------------------------------------------------
//
// The reference forms: correct, unswizzled, and what the test checks against. A kernel that stages
// through LDS should use frag_c/frag_k0 directly and swizzle, because `load_b_bf16` below walks a
// column and is 16 separate rows of LDS.
//
// The engine's dense GEMM is Y[row][tok] = sum_c W[row][c] * X[c][tok], so W lands on A with
// lda = cols and X lands on B with ldb = nb, both already row-major. No transpose anywhere.

__device__ __forceinline__ s8v load_a_bf16(const uint16_t* __restrict__ a, uint32_t lda,
                                           uint32_t lane) {
  const uint32_t c = frag_c(lane), k0 = frag_k0(lane);
  s8v f;
#pragma unroll
  for (int j = 0; j < 8; ++j) f[j] = (short)a[(size_t)c * lda + k0 + j];
  return f;
}

__device__ __forceinline__ s8v load_b_bf16(const uint16_t* __restrict__ b, uint32_t ldb,
                                           uint32_t lane) {
  const uint32_t c = frag_c(lane), k0 = frag_k0(lane);
  s8v f;
#pragma unroll
  for (int j = 0; j < 8; ++j) f[j] = (short)b[(size_t)(k0 + j) * ldb + c];
  return f;
}

// fp8 operands are 8 bytes in 2 VGPRs, same (c, k0+j) mapping, one byte per j.
__device__ __forceinline__ i2v load_a_fp8(const uint8_t* __restrict__ a, uint32_t lda,
                                          uint32_t lane) {
  const uint32_t c = frag_c(lane), k0 = frag_k0(lane);
  uint32_t w[2] = {0, 0};
#pragma unroll
  for (int j = 0; j < 8; ++j) w[j >> 2] |= (uint32_t)a[(size_t)c * lda + k0 + j] << (8 * (j & 3));
  i2v f;
  f[0] = (int)w[0];
  f[1] = (int)w[1];
  return f;
}

__device__ __forceinline__ i2v load_b_fp8(const uint8_t* __restrict__ b, uint32_t ldb,
                                          uint32_t lane) {
  const uint32_t c = frag_c(lane), k0 = frag_k0(lane);
  uint32_t w[2] = {0, 0};
#pragma unroll
  for (int j = 0; j < 8; ++j) w[j >> 2] |= (uint32_t)b[(size_t)(k0 + j) * ldb + c] << (8 * (j & 3));
  i2v f;
  f[0] = (int)w[0];
  f[1] = (int)w[1];
  return f;
}

// ---- accumulator store -------------------------------------------------------------------------

__device__ __forceinline__ void store_acc(f8v d, float* __restrict__ out, uint32_t ldd,
                                          uint32_t lane) {
  const uint32_t c = frag_c(lane);
#pragma unroll
  for (int j = 0; j < 8; ++j) out[(size_t)acc_row(lane, j) * ldd + c] = d[j];
}

}  // namespace wmma
}  // namespace aff
