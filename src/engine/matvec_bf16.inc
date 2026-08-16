// bf16 dense matvec — the hottest kernel in the engine by BYTES MOVED. Storage is not traffic: only
// six of 256 experts are read per token, while EVERY dense weight is, so the dense side is several
// times the routed-expert traffic despite being a fraction of the container.
//
// Single-threaded this kernel already runs at 83-90% of the single-thread DRAM wall. Unlike the
// expert kernel, which unpacks 2.875-bit codes through a codebook and is genuinely decode-limited,
// this one only shifts bf16 left by 16 — there is no arithmetic to hide behind, so no arithmetic
// change can buy anything. Variant 2's four accumulators are the default because the spread is
// tighter, not because they are faster.
//
// THE LEVER IS THE FORMAT, NOT THE KERNEL — and not by quantising further. Most of these dense
// parameters ship as F8_E4M3 with a per-128x128 UE8M0 scale, and dequantising them to bf16 at
// quantise time doubles the largest traffic source in the engine for nothing. Storing FP8 as FP8 is
// bit-exact with the checkpoint and halves it, and the VRAM it returns is residency budget.

#pragma once

namespace aff {

inline void matvec_bf16_v1(const uint16_t* W, const float* x, uint64_t rows, uint64_t cols,
                           float* out) noexcept {
  for (uint64_t r = 0; r < rows; ++r) {
    const uint16_t* w = W + r * cols;
    float acc = 0.0f;
#if defined(__AVX512F__)
    __m512 a = _mm512_setzero_ps();
    uint64_t c = 0;
    for (; c + 16 <= cols; c += 16) {
      const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + c));
      const __m512 wv = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
      a = _mm512_fmadd_ps(wv, _mm512_loadu_ps(x + c), a);
    }
    acc = _mm512_reduce_add_ps(a);
    for (; c < cols; ++c) acc += bf16_to_float(w[c]) * x[c];
#else
    for (uint64_t c = 0; c < cols; ++c) acc += bf16_to_float(w[c]) * x[c];
#endif
    out[r] = acc;
  }
}

inline void matvec_bf16_v2(const uint16_t* W, const float* x, uint64_t rows, uint64_t cols,
                           float* out) noexcept {
#if defined(__AVX512F__)
  for (uint64_t r = 0; r < rows; ++r) {
    const uint16_t* w = W + r * cols;
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    uint64_t c = 0;
    // Four chains, 64 columns per iteration. bf16 -> f32 is a 16-bit left shift, so the widening
    // costs one shuffle and one shift and never touches a conversion unit.
    for (; c + 64 <= cols; c += 64) {
      const __m256i h0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + c));
      const __m256i h1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + c + 16));
      const __m256i h2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + c + 32));
      const __m256i h3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + c + 48));
      a0 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h0), 16)),
                           _mm512_loadu_ps(x + c), a0);
      a1 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h1), 16)),
                           _mm512_loadu_ps(x + c + 16), a1);
      a2 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h2), 16)),
                           _mm512_loadu_ps(x + c + 32), a2);
      a3 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h3), 16)),
                           _mm512_loadu_ps(x + c + 48), a3);
    }
    for (; c + 16 <= cols; c += 16) {
      const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + c));
      a0 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16)),
                           _mm512_loadu_ps(x + c), a0);
    }
    float acc = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
    for (; c < cols; ++c) acc += bf16_to_float(w[c]) * x[c];
    out[r] = acc;
  }
#else
  matvec_bf16_v1(W, x, rows, cols, out);
#endif
}

} // namespace aff
