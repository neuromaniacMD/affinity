// FP8-E4M3 dense matvec — the checkpoint's own dense format, decoded in the kernel.
//
// 92% of the dense parameters ship as F8_E4M3 with one UE8M0 scale per 128x128 tile. The quantiser
// used to dequantise them to bf16 and store bf16, which doubled the single largest traffic source
// in the engine (12.60 GB/token dense against 2.33 for the routed experts) and bought nothing: the
// model was TRAINED in this format, so bf16 carries no extra information. Storing the bytes
// verbatim is bit-exact with the checkpoint and halves that traffic.
//
// The container holds exactly what the safetensors held, no repacking:
//   data   row-major [rows, cols] E4M3, one byte per weight
//   scales row-major [rows/128, cols/128] E8M0, one byte per tile
//
// The scale is constant across a 128-wide run of a row, so it multiplies ONCE per tile rather than
// per weight — the inner loop is a plain dot product and the scale is a single FMA at the end of it.
// That is why the tile is the unit here and not the individual weight.
//
// E4M3 -> f32 is pure bit manipulation, no table:
//   sign        b & 0x80          -> bit 31
//   exponent    (b >> 3) & 0x0F   -> bias 7, rebias to 127 by adding 120
//   mantissa    b & 0x07          -> the top 3 bits of the f32 mantissa, shift left 20
// Subnormals (exponent 0) are NOT representable that way — the implicit leading bit is absent, so
// the value is mantissa * 2^-9. They are handled with a select rather than a branch.
//
// This is the OCP "fn" variant: no infinities, and 0x7F/0xFF are NaN. The code above maps those to
// +-480 instead. That is deliberate: trained weights contain no NaN, and paying a compare per
// element to reproduce a value that cannot occur would cost more than it can ever be worth.

#pragma once

namespace aff {

// One f32 per E4M3 code. Built once; used by the scalar path and by the tail of the vector path.
struct Fp8Lut {
  float v[256];
  Fp8Lut() noexcept { for (int i = 0; i < 256; ++i) v[i] = fp8_e4m3_to_float((uint8_t)i); }
};
inline const Fp8Lut& fp8_lut() noexcept { static const Fp8Lut t; return t; }

// `W`/`S` address the FULL matrix; `row0` is the absolute index of the first row to compute, which
// is what selects the scale tile row. Splitting by rows across threads therefore needs no
// renumbering — the caller passes its own row offset and the scales stay correct.
inline void matvec_fp8_rows(const uint8_t* W, const uint8_t* S, uint64_t row0, uint64_t nrows,
                            uint64_t cols, const float* x, float* out) noexcept {
  const uint64_t sc_cols = (cols + 127) / 128;
  const float* lut = fp8_lut().v;

  for (uint64_t r = 0; r < nrows; ++r) {
    const uint8_t* w = W + (row0 + r) * cols;
    const uint8_t* srow = S + ((row0 + r) >> 7) * sc_cols;
    float acc = 0.0f;

    for (uint64_t t = 0; t < sc_cols; ++t) {
      const uint64_t c0 = t * 128;
      const uint64_t c1 = std::min<uint64_t>(c0 + 128, cols);
      const float scale = e8m0_to_float(srow[t]);
      float sub = 0.0f;
      uint64_t c = c0;
#if defined(__AVX512F__)
      // Two chains over the 128-wide tile. The decode is integer-only, so it issues on ports the
      // FMAs are not using.
      const __m512i k_sign = _mm512_set1_epi32(0x80);
      const __m512i k_exp  = _mm512_set1_epi32(0x78);
      const __m512i k_man  = _mm512_set1_epi32(0x07);
      __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
      for (; c + 32 <= c1; c += 32) {
        for (int h = 0; h < 2; ++h) {
          const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + c + h * 16));
          const __m512i b = _mm512_cvtepu8_epi32(raw);
          const __m512i sign = _mm512_slli_epi32(_mm512_and_si512(b, k_sign), 24);
          const __m512i ex   = _mm512_srli_epi32(_mm512_and_si512(b, k_exp), 3);
          const __m512i mn   = _mm512_and_si512(b, k_man);
          // Normal: (ex + 120) << 23 | mn << 20. Bias 7 -> 127 is +120.
          const __m512i nrm = _mm512_or_si512(_mm512_slli_epi32(_mm512_add_epi32(ex, _mm512_set1_epi32(120)), 23),
                                              _mm512_slli_epi32(mn, 20));
          // Subnormal: mantissa * 2^-9. Covers zero as well, since mn == 0 there.
          const __m512 sbn = _mm512_mul_ps(_mm512_cvtepi32_ps(mn), _mm512_set1_ps(1.0f / 512.0f));
          const __mmask16 is_sub = _mm512_cmpeq_epi32_mask(ex, _mm512_setzero_si512());
          const __m512 mag = _mm512_mask_blend_ps(is_sub, _mm512_castsi512_ps(nrm), sbn);
          const __m512 wv = _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(mag), sign));
          if (h == 0) a0 = _mm512_fmadd_ps(wv, _mm512_loadu_ps(x + c), a0);
          else        a1 = _mm512_fmadd_ps(wv, _mm512_loadu_ps(x + c + 16), a1);
        }
      }
      sub = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
#endif
      for (; c < c1; ++c) sub += lut[w[c]] * x[c];
      acc += sub * scale;
    }
    out[r] = acc;
  }
}

} // namespace aff
