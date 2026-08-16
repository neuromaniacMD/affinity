#include "expert_kernel.h"
#include "engine/threadpool.h"
#include "quant/source_dtypes.h"

#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace aff {

// CPU kernel variant. 1 is the scalar reference; 2 adds two accumulators and one field load per
// block pair; 4 is the default and cuts instructions with AVX512_VBMI, bit-identical to v2. 3 and 5
// are intermediate points kept as controls rather than as attributions — both are unstable against
// code layout, so the only claim this file makes is v2 -> v4. All are retained so the set stays
// comparable after a change to the surrounding code (`aff-kbench`).
//
// The kernel is COMPUTE-bound, not memory-bound (see the header), which is why every variant below
// chases operation count and nothing else.
//
// The shipped variant. `aff-kbench` sets others through cpu_kernel_variant() to compare them.
static int g_cpu_variant = 4;
void cpu_kernel_variant(int v) { g_cpu_variant = v; }
int  cpu_kernel_variant() { return g_cpu_variant; }

namespace {

// Codes are 5 bits and therefore not byte-aligned. Rather than a bit-extract per pair, we walk a
// running 64-bit window: for block=16 there are 8 pairs = 40 bits, so one 64-bit load per block
// covers it with room to spare. This keeps the inner loop free of shifts across byte boundaries.
inline uint64_t load_bits(const uint8_t* p, uint64_t bitpos) noexcept {
  uint64_t v;
  std::memcpy(&v, p + (bitpos >> 3), sizeof(v));
  return v >> (bitpos & 7);
}

inline float rscale_of(const uint8_t* rscales, uint64_t row, bool f32) noexcept {
  if (f32) { float v; std::memcpy(&v, rscales + row * 4, 4); return v; }
  uint16_t h; std::memcpy(&h, rscales + row * 2, 2);
  return bf16_to_float(h);
}

} // namespace

ExpertMatrixView view_of(const QuantizedMatrix& q) noexcept {
  ExpertMatrixView v;
  v.data = q.data.data();
  v.scales = q.scales.data();
  v.rscales = q.rscales.data();
  v.rows = q.rows; v.cols = q.cols; v.spec = q.spec;
  return v;
}

uint64_t expert_stream_bytes(const ExpertMatrixView& m) noexcept {
  const QuantSpec& s = m.spec;
  const uint64_t codes = m.rows * (m.cols / s.pair) * s.code_bits;
  const uint64_t blocks = m.rows * (m.cols / s.block) * s.block_field_bits();
  return (codes + 7) / 8 + (blocks + 7) / 8 + m.rows * (s.row_scale_f32 ? 4 : 2);
}

void expert_gemv_reference(const ExpertMatrixView& m, const Codebook& cb,
                           const float* x, float* y) noexcept {
  const QuantSpec& s = m.spec;
  const uint64_t nblk = m.cols / s.block;
  const uint64_t ipb = s.block / s.pair;           // pairs per block
  const uint32_t smask = (1u << s.scale_bits) - 1u;
  const uint64_t codes_row = m.cols / s.pair;

  for (uint64_t r = 0; r < m.rows; ++r) {
    const float rs = rscale_of(m.rscales, r, s.row_scale_f32);
    double acc = 0.0;
    for (uint64_t b = 0; b < nblk; ++b) {
      const uint64_t fbit = (r * nblk + b) * s.block_field_bits();
      const uint32_t field = (uint32_t)(load_bits(m.scales, fbit) & ((1u << s.block_field_bits()) - 1u));
      // LSB-first: scale, sign, rotation, variant — the same order the GPU reads. Spelled out
      // rather than assumed because both the integer-scale pair format and the pow2 quad format
      // with its two symmetry bits reach here, and getting the shift wrong is not a crash: it
      // reads the rotation bit as the variant and returns numbers with no relationship to the
      // weights at all.
      const uint32_t g = (field >> s.scale_bits) & (s.transforms() - 1u);
      const uint32_t var = field >> (s.scale_bits + s.sign_bits + s.rot_bits);
      const float bs = rs * block_scale_from_code(field & smask, s) * ((g & 1u) ? -1.0f : 1.0f);
      const uint32_t rh = (g & 2u) ? s.pair / 2u : 0u;
      const float* lut = cb.entry(var < cb.variants ? var : 0, 0);
      for (uint64_t i = 0; i < ipb; ++i) {
        const uint64_t cbit = (r * codes_row + b * ipb + i) * s.code_bits;
        const uint32_t code = (uint32_t)(load_bits(m.data, cbit) & ((1u << s.code_bits) - 1u));
        const float* e = lut + (size_t)code * s.pair;
        const uint64_t c0 = b * s.block + i * s.pair;
        // The rotation is on the RECONSTRUCTION: position p takes entry (p + rh) % pair, the same
        // permutation the kernel gets from one funnel shift of the packed dword.
        for (uint32_t p = 0; p < s.pair; ++p)
          acc += (double)(bs * e[(p + rh) % s.pair]) * x[c0 + p];
      }
    }
    y[r] += (float)acc;
  }
}

void expert_gemv(const ExpertMatrixView& m, const Codebook& cb,
                 const float* x, float* y) noexcept {
  expert_gemv_scaled(m, cb, x, 1.0f, y);
}

ExpertMatrixView expert_row_slice(const ExpertMatrixView& m, uint64_t lo, uint64_t hi) noexcept {
  ExpertMatrixView s = m;
  s.rows = hi - lo;
  s.rscales = m.rscales + lo * (m.spec.row_scale_f32 ? 4 : 2);
  const uint64_t codes_row_bits = m.cols / m.spec.pair * m.spec.code_bits;
  const uint64_t field_row_bits = m.cols / m.spec.block * m.spec.block_field_bits();
  s.data    = m.data + lo * codes_row_bits / 8;
  s.scales  = m.scales + lo * field_row_bits / 8;
  return s;
}

void expert_gemv_scaled_mt(const ExpertMatrixView& m, const Codebook& cb,
                           const float* x, float w, float* y) {
  parallel_for(m.rows, 256, [&](uint64_t lo, uint64_t hi) {
    expert_gemv_scaled(expert_row_slice(m, lo, hi), cb, x, w, y + lo);
  });
}

void expert_gemv_scaled(const ExpertMatrixView& m, const Codebook& cb,
                        const float* x, float w, float* y) noexcept {
  const QuantSpec& s = m.spec;
  // Fast path is specialised for the locked format (D11): block 16, pair 2, 5-bit codes.
  // Anything else falls back to the reference — correctness never depends on the fast path.
  const bool fast = (s.block == 16 && s.pair == 2 && s.code_bits == 5 &&
                     s.block_field_bits() <= 8 && cb.pair == 2 && cb.k == 32 &&
                     (m.cols % 32) == 0);
  if (!fast) {
    std::vector<float> tmp(m.rows, 0.0f);
    expert_gemv_reference(m, cb, x, tmp.data());
    for (uint64_t r = 0; r < m.rows; ++r) y[r] += w * tmp[r];
    return;
  }

  const uint64_t nblk  = m.cols / 16;
  const uint32_t smask = (1u << s.scale_bits) - 1u;
  const uint32_t fmask = (1u << s.block_field_bits()) - 1u;
  const uint64_t fbits = s.block_field_bits();
  const uint64_t codes_row = m.cols / 2;
  const uint64_t npair = m.cols / 2;

  // Split the activation into even/odd planes ONCE per matrix. The codebook is 2-D (a code
  // decodes a PAIR), so with x pre-split the pair components line up with contiguous lanes and
  // no cross-lane interleave is needed in the inner loop. Cost is one pass over `x` (4096 floats)
  // against ~2.9 MiB of weights — free.
  std::vector<float> xe(npair), xo(npair);
  for (uint64_t i = 0; i < npair; ++i) { xe[i] = x[2 * i]; xo[i] = x[2 * i + 1]; }

#if defined(__AVX512F__)
  // Codebook as lo/hi planes, 32 entries each, split across two 512-bit registers per plane so
  // _mm512_permutex2var_ps gives a true 32-entry lookup producing 16 results per instruction.
  __m512 lo0[4], lo1[4], hi0[4], hi1[4];
  const __m512i vshift = _mm512_setr_epi64(0, 5, 10, 15, 20, 25, 30, 35);
  const __m512i vmask  = _mm512_set1_epi64(31);
  for (uint32_t v = 0; v < 4; ++v) {
    alignas(64) float L[32], H[32];
    const uint32_t vv = v < cb.variants ? v : 0;
    for (uint32_t c = 0; c < 32; ++c) { L[c] = cb.entry(vv, c)[0]; H[c] = cb.entry(vv, c)[1]; }
    lo0[v] = _mm512_load_ps(L);      lo1[v] = _mm512_load_ps(L + 16);
    hi0[v] = _mm512_load_ps(H);      hi1[v] = _mm512_load_ps(H + 16);
  }

  // The block-field decode stays INLINE. Field extraction is a large share of kernel time, but
  // hoisting it into a per-row buffer only relocates that cost and adds a buffer round trip, filled
  // scalar or SIMD.

  // Two accumulators, and ONE field load per block pair.
  //
  //   * fA and fB sit 6 bits apart, so one 64-bit window already contains both. The second
  //     load_bits was pure duplication.
  //   * `acc` was a single serial FMA chain across all 128 block pairs of a row. At one thread the
  //     kernel is nowhere near the DRAM wall, so that latency is exposed.
  if (g_cpu_variant == 2) {
   for (uint64_t r = 0; r < m.rows; ++r) {
    const float rs = rscale_of(m.rscales, r, s.row_scale_f32) * w;
    const uint64_t cbase = r * codes_row * 5;
    const uint64_t fbase = r * nblk * fbits;
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    uint64_t b = 0;
    // Manually unrolled, no loop-carried branch. A 2-iteration inner loop with
    // `if (half == 0) acc0 else acc1` leaves the compiler a real branch and is slower; what pays is
    // the unrolled shape, not the second accumulator on its own.
    for (; b + 3 < nblk; b += 4) {
      const uint64_t fw0 = load_bits(m.scales, fbase + b * fbits);
      const uint64_t fw1 = load_bits(m.scales, fbase + (b + 2) * fbits);
      const uint32_t f0 = (uint32_t)(fw0 & fmask), f1 = (uint32_t)((fw0 >> fbits) & fmask);
      const uint32_t f2 = (uint32_t)(fw1 & fmask), f3 = (uint32_t)((fw1 >> fbits) & fmask);
      const float bs0 = rs * (float)(f0 & smask), bs1 = rs * (float)(f1 & smask);
      const float bs2 = rs * (float)(f2 & smask), bs3 = rs * (float)(f3 & smask);
      const __mmask16 hiHalf = 0xFF00;

      if (bs0 != 0.0f || bs1 != 0.0f) {
        const uint32_t vA = f0 >> s.scale_bits, vB = f1 >> s.scale_bits;
        const uint64_t w0 = load_bits(m.data, cbase + b * 8 * 5);
        const uint64_t w1 = load_bits(m.data, cbase + (b + 1) * 8 * 5);
        const __m256i c0 = _mm512_cvtepi64_epi32(
            _mm512_and_si512(_mm512_srlv_epi64(_mm512_set1_epi64((long long)w0), vshift), vmask));
        const __m256i c1 = _mm512_cvtepi64_epi32(
            _mm512_and_si512(_mm512_srlv_epi64(_mm512_set1_epi64((long long)w1), vshift), vmask));
        const __m512i vidx = _mm512_inserti64x4(_mm512_castsi256_si512(c0), c1, 1);
        const __m512 vlo = _mm512_mask_blend_ps(hiHalf,
            _mm512_permutex2var_ps(lo0[vA], vidx, lo1[vA]),
            _mm512_permutex2var_ps(lo0[vB], vidx, lo1[vB]));
        const __m512 vhi = _mm512_mask_blend_ps(hiHalf,
            _mm512_permutex2var_ps(hi0[vA], vidx, hi1[vA]),
            _mm512_permutex2var_ps(hi0[vB], vidx, hi1[vB]));
        const __m512 vs = _mm512_mask_blend_ps(hiHalf, _mm512_set1_ps(bs0), _mm512_set1_ps(bs1));
        acc0 = _mm512_fmadd_ps(vs,
            _mm512_fmadd_ps(vlo, _mm512_loadu_ps(xe.data() + b * 8),
                            _mm512_mul_ps(vhi, _mm512_loadu_ps(xo.data() + b * 8))), acc0);
      }
      if (bs2 != 0.0f || bs3 != 0.0f) {
        const uint32_t vA = f2 >> s.scale_bits, vB = f3 >> s.scale_bits;
        const uint64_t w0 = load_bits(m.data, cbase + (b + 2) * 8 * 5);
        const uint64_t w1 = load_bits(m.data, cbase + (b + 3) * 8 * 5);
        const __m256i c0 = _mm512_cvtepi64_epi32(
            _mm512_and_si512(_mm512_srlv_epi64(_mm512_set1_epi64((long long)w0), vshift), vmask));
        const __m256i c1 = _mm512_cvtepi64_epi32(
            _mm512_and_si512(_mm512_srlv_epi64(_mm512_set1_epi64((long long)w1), vshift), vmask));
        const __m512i vidx = _mm512_inserti64x4(_mm512_castsi256_si512(c0), c1, 1);
        const __m512 vlo = _mm512_mask_blend_ps(hiHalf,
            _mm512_permutex2var_ps(lo0[vA], vidx, lo1[vA]),
            _mm512_permutex2var_ps(lo0[vB], vidx, lo1[vB]));
        const __m512 vhi = _mm512_mask_blend_ps(hiHalf,
            _mm512_permutex2var_ps(hi0[vA], vidx, hi1[vA]),
            _mm512_permutex2var_ps(hi0[vB], vidx, hi1[vB]));
        const __m512 vs = _mm512_mask_blend_ps(hiHalf, _mm512_set1_ps(bs2), _mm512_set1_ps(bs3));
        acc1 = _mm512_fmadd_ps(vs,
            _mm512_fmadd_ps(vlo, _mm512_loadu_ps(xe.data() + (b + 2) * 8),
                            _mm512_mul_ps(vhi, _mm512_loadu_ps(xo.data() + (b + 2) * 8))), acc1);
      }
    }
    for (; b + 1 < nblk; b += 2) {
      const uint64_t fw = load_bits(m.scales, fbase + b * fbits);
      const uint32_t fA = (uint32_t)(fw & fmask), fB = (uint32_t)((fw >> fbits) & fmask);
      const float bsA = rs * (float)(fA & smask), bsB = rs * (float)(fB & smask);
      if (bsA == 0.0f && bsB == 0.0f) continue;
      const uint32_t vA = fA >> s.scale_bits, vB = fB >> s.scale_bits;
      const uint64_t w0 = load_bits(m.data, cbase + b * 8 * 5);
      const uint64_t w1 = load_bits(m.data, cbase + (b + 1) * 8 * 5);
      const __m256i c0 = _mm512_cvtepi64_epi32(
          _mm512_and_si512(_mm512_srlv_epi64(_mm512_set1_epi64((long long)w0), vshift), vmask));
      const __m256i c1 = _mm512_cvtepi64_epi32(
          _mm512_and_si512(_mm512_srlv_epi64(_mm512_set1_epi64((long long)w1), vshift), vmask));
      const __m512i vidx = _mm512_inserti64x4(_mm512_castsi256_si512(c0), c1, 1);
      const __mmask16 hiHalf = 0xFF00;
      const __m512 vlo = _mm512_mask_blend_ps(hiHalf,
          _mm512_permutex2var_ps(lo0[vA], vidx, lo1[vA]),
          _mm512_permutex2var_ps(lo0[vB], vidx, lo1[vB]));
      const __m512 vhi = _mm512_mask_blend_ps(hiHalf,
          _mm512_permutex2var_ps(hi0[vA], vidx, hi1[vA]),
          _mm512_permutex2var_ps(hi0[vB], vidx, hi1[vB]));
      const __m512 vs = _mm512_mask_blend_ps(hiHalf, _mm512_set1_ps(bsA), _mm512_set1_ps(bsB));
      acc0 = _mm512_fmadd_ps(vs, _mm512_fmadd_ps(vlo, _mm512_loadu_ps(xe.data() + b * 8),
                                                 _mm512_mul_ps(vhi, _mm512_loadu_ps(xo.data() + b * 8))),
                             acc0);
    }
    y[r] += _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
   }
   return;
  }

#if defined(__AVX512VBMI__)
  // ---- variants 3 and 4: the same arithmetic, fewer instructions to reach it -------------------
  //
  // v2 is issue-bound rather than memory-bound, so the only lever is asking for less. Two places
  // pay for a bit-packed format with general-purpose code, and `vpmultishiftqb` — eight
  // arbitrarily-aligned 8-bit fields out of a qword in one instruction — replaces both:
  //
  //   CODES (v3). Sixteen 5-bit codes are 80 bits: qword 0 holds the first eight and qword 1 the
  //     same bytes five further along, so one control vector of {0,5,...,35} twice lifts all of
  //     them, in place of two windows, two shifts, two masks and a lane insert.
  //   FIELDS (v4). Four 6-bit block fields are 24 bits, lifted together instead of decoded with
  //     four rounds of mask/shift/convert. Only the two variant INDICES come back to scalar
  //     registers, because they index the codebook table registers.
  //
  // v4 is also branchless. v2 skips a block pair whose scales are both zero, which is worth having
  // given how many QAT blocks are all-zero — but it is a data-dependent branch in the innermost
  // loop, and the vectorised field decode has already done the work by the time the test could be
  // made.
  //
  // ALIGNMENT, which is what makes any of this legal: `cbase` is r*(cols/2)*5 bits and cols is a
  // multiple of 64 here, so `cbase` is a whole number of bytes; b*40 bits is byte-aligned for every
  // b. `fbase` is r*(cols/16)*6 bits, a whole number of bytes when cols is a multiple of 64, and
  // b*6 is byte-aligned for b a multiple of 4 — which is the step of the main loop. The extra
  // `cols % 64` condition below is exactly that requirement and nothing more; a matrix that fails
  // it falls through to the v1 loop below, which reads through `load_bits` and does not care.
  if (g_cpu_variant >= 3 && g_cpu_variant <= 5 && (m.cols % 64) == 0) {
   // Codes: output bytes 0-7 take bits {0,5,...,35} of qword 0, bytes 8-15 the same of qword 1.
   const __m512i kCodeCtl = _mm512_broadcast_i32x4(
       _mm_setr_epi8(0, 5, 10, 15, 20, 25, 30, 35, 0, 5, 10, 15, 20, 25, 30, 35));
   // Fields: four 6-bit fields at bits {0,6,12,18} of qword 0.
   const __m512i kFieldCtl = _mm512_castsi128_si512(
       _mm_setr_epi8(0, 6, 12, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
   const __m512i kCode31 = _mm512_set1_epi32(31);
   const __m512i kFmask  = _mm512_set1_epi32((int)fmask);
   const __m512i kSmask  = _mm512_set1_epi32((int)smask);
   // Broadcast lane 0 of a 4-scale vector over the low half and lane 1 over the high half, and
   // likewise lanes 2 and 3 for the second block pair.
   const __m512i kBcast01 = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
   const __m512i kBcast23 = _mm512_setr_epi32(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
   const __mmask16 hiHalf = 0xFF00;
   const bool v4 = g_cpu_variant == 4;

   // 16 codes as dword indices, from a byte-aligned group start.
   auto codes16 = [&](const uint8_t* p) {
     uint64_t w0, w1;
     std::memcpy(&w0, p, 8);
     std::memcpy(&w1, p + 5, 8);
     const __m512i d = _mm512_castsi128_si512(_mm_set_epi64x((long long)w1, (long long)w0));
     return _mm512_and_si512(
         _mm512_cvtepu8_epi32(_mm512_castsi512_si128(_mm512_multishift_epi64_epi8(kCodeCtl, d))),
         kCode31);
   };
   // acc += vs * (LUT_lo[code]*xe + LUT_hi[code]*xo), lanes 0-7 under variant vA and 8-15 under vB.
   //
   // Unconditional for v3 and v4, exactly as v2 does it, so that v3 isolates the code-extraction
   // change and nothing else. v5 adds back the `vA == vB` shortcut v1 had: four permutes become two
   // whenever adjacent blocks pick the same grid, at the price of a data-dependent branch in the
   // innermost loop. Whether it pays is a property of the QUANTISER's variant choices, so it can
   // only be judged on real container bytes and not on the synthetic set.
   const bool v5 = g_cpu_variant == 5;
   auto accum = [&](__m512 acc, __m512i vidx, uint32_t vA, uint32_t vB, __m512 vs, uint64_t xoff) {
     __m512 vlo, vhi;
     if (v5 && vA == vB) {
       vlo = _mm512_permutex2var_ps(lo0[vA], vidx, lo1[vA]);
       vhi = _mm512_permutex2var_ps(hi0[vA], vidx, hi1[vA]);
     } else {
       vlo = _mm512_mask_blend_ps(hiHalf,
           _mm512_permutex2var_ps(lo0[vA], vidx, lo1[vA]),
           _mm512_permutex2var_ps(lo0[vB], vidx, lo1[vB]));
       vhi = _mm512_mask_blend_ps(hiHalf,
           _mm512_permutex2var_ps(hi0[vA], vidx, hi1[vA]),
           _mm512_permutex2var_ps(hi0[vB], vidx, hi1[vB]));
     }
     return _mm512_fmadd_ps(vs,
         _mm512_fmadd_ps(vlo, _mm512_loadu_ps(xe.data() + xoff),
                         _mm512_mul_ps(vhi, _mm512_loadu_ps(xo.data() + xoff))), acc);
   };

   for (uint64_t r = 0; r < m.rows; ++r) {
    const float rs = rscale_of(m.rscales, r, s.row_scale_f32) * w;
    const uint64_t cbase = r * codes_row * 5;
    const uint64_t fbase = r * nblk * fbits;
    const uint8_t* cp = m.data + (cbase >> 3);
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    uint64_t b = 0;
    for (; b + 3 < nblk; b += 4) {
      uint32_t vr[4];
      __m512 vs01, vs23;
      if (v4) {
        uint64_t fw;
        std::memcpy(&fw, m.scales + ((fbase + b * fbits) >> 3), 8);
        const __m512i f4 = _mm512_and_si512(
            _mm512_cvtepu8_epi32(_mm512_castsi512_si128(_mm512_multishift_epi64_epi8(
                kFieldCtl, _mm512_castsi128_si512(_mm_cvtsi64_si128((long long)fw))))),
            kFmask);
        const __m512 bs4 = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_and_si512(f4, kSmask)), _mm512_set1_ps(rs));
        alignas(16) uint32_t vtmp[4];
        _mm_store_si128((__m128i*)vtmp,
                        _mm512_castsi512_si128(_mm512_srli_epi32(f4, (int)s.scale_bits)));
        vr[0] = vtmp[0]; vr[1] = vtmp[1]; vr[2] = vtmp[2]; vr[3] = vtmp[3];
        vs01 = _mm512_permutexvar_ps(kBcast01, bs4);
        vs23 = _mm512_permutexvar_ps(kBcast23, bs4);
      } else {
        const uint64_t fw0 = load_bits(m.scales, fbase + b * fbits);
        const uint64_t fw1 = load_bits(m.scales, fbase + (b + 2) * fbits);
        const uint32_t f0 = (uint32_t)(fw0 & fmask), f1 = (uint32_t)((fw0 >> fbits) & fmask);
        const uint32_t f2 = (uint32_t)(fw1 & fmask), f3 = (uint32_t)((fw1 >> fbits) & fmask);
        const float b0 = rs * (float)(f0 & smask), b1 = rs * (float)(f1 & smask);
        const float b2 = rs * (float)(f2 & smask), b3 = rs * (float)(f3 & smask);
        vr[0] = f0 >> s.scale_bits; vr[1] = f1 >> s.scale_bits;
        vr[2] = f2 >> s.scale_bits; vr[3] = f3 >> s.scale_bits;
        if (b0 == 0.0f && b1 == 0.0f && b2 == 0.0f && b3 == 0.0f) continue;
        vs01 = _mm512_mask_blend_ps(hiHalf, _mm512_set1_ps(b0), _mm512_set1_ps(b1));
        vs23 = _mm512_mask_blend_ps(hiHalf, _mm512_set1_ps(b2), _mm512_set1_ps(b3));
      }
      acc0 = accum(acc0, codes16(cp + b * 5), vr[0], vr[1], vs01, b * 8);
      acc1 = accum(acc1, codes16(cp + (b + 2) * 5), vr[2], vr[3], vs23, (b + 2) * 8);
    }
    // Tail, for a block count that is not a multiple of four. Never runs on the locked geometry
    // (256 and 128 blocks), and goes through `load_bits` because b*6 is no longer byte-aligned.
    for (; b + 1 < nblk; b += 2) {
      const uint64_t fw = load_bits(m.scales, fbase + b * fbits);
      const uint32_t fA = (uint32_t)(fw & fmask), fB = (uint32_t)((fw >> fbits) & fmask);
      const float bsA = rs * (float)(fA & smask), bsB = rs * (float)(fB & smask);
      if (bsA == 0.0f && bsB == 0.0f) continue;
      const __m512 vs = _mm512_mask_blend_ps(hiHalf, _mm512_set1_ps(bsA), _mm512_set1_ps(bsB));
      acc0 = accum(acc0, codes16(cp + b * 5), fA >> s.scale_bits, fB >> s.scale_bits, vs, b * 8);
    }
    y[r] += _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
   }
   return;
  }
#endif  // __AVX512VBMI__

  for (uint64_t r = 0; r < m.rows; ++r) {
    const float rs = rscale_of(m.rscales, r, s.row_scale_f32) * w;
    const uint64_t cbase = r * codes_row * 5;
    const uint64_t fbase = r * nblk * fbits;

    __m512 acc = _mm512_setzero_ps();

    // Two blocks (32 weights, 16 codes) per iteration to fill a 512-bit register.
    for (uint64_t b = 0; b + 1 < nblk; b += 2) {
      const uint32_t fA = (uint32_t)(load_bits(m.scales, fbase + b * fbits) & fmask);
      const uint32_t fB = (uint32_t)(load_bits(m.scales, fbase + (b + 1) * fbits) & fmask);
      const float bsA = rs * (float)(fA & smask);
      const float bsB = rs * (float)(fB & smask);
      if (bsA == 0.0f && bsB == 0.0f) continue;        // all-zero blocks: ~11.7% of QAT weights
      const uint32_t vA = fA >> s.scale_bits, vB = fB >> s.scale_bits;

      // 16 five-bit codes = 80 bits. Extract entirely in SIMD: broadcasting each 64-bit word to
      // eight lanes and variable-shifting by {0,5,...,35} yields eight codes per word with no
      // round trip through memory. The scalar version writes 16 uint32s and reloads them as a
      // vector, which costs a store-to-load forwarding stall every iteration and caps the kernel at
      // half this rate.
      const uint64_t w0 = load_bits(m.data, cbase + b * 8 * 5);
      const uint64_t w1 = load_bits(m.data, cbase + (b + 1) * 8 * 5);
      const __m256i c0 = _mm512_cvtepi64_epi32(
          _mm512_and_si512(_mm512_srlv_epi64(_mm512_set1_epi64((long long)w0), vshift), vmask));
      const __m256i c1 = _mm512_cvtepi64_epi32(
          _mm512_and_si512(_mm512_srlv_epi64(_mm512_set1_epi64((long long)w1), vshift), vmask));
      const __m512i vidx = _mm512_inserti64x4(_mm512_castsi256_si512(c0), c1, 1);

      // Permute under BOTH variants, then blend: low 8 lanes take block A, high 8 take block B.
      // Adjacent blocks often pick the same grid; when they do, half the permutes disappear.
      const __mmask16 hiHalf = 0xFF00;
      __m512 vlo, vhi;
      if (vA == vB) {
        vlo = _mm512_permutex2var_ps(lo0[vA], vidx, lo1[vA]);
        vhi = _mm512_permutex2var_ps(hi0[vA], vidx, hi1[vA]);
      } else {
        vlo = _mm512_mask_blend_ps(hiHalf,
            _mm512_permutex2var_ps(lo0[vA], vidx, lo1[vA]),
            _mm512_permutex2var_ps(lo0[vB], vidx, lo1[vB]));
        vhi = _mm512_mask_blend_ps(hiHalf,
            _mm512_permutex2var_ps(hi0[vA], vidx, hi1[vA]),
            _mm512_permutex2var_ps(hi0[vB], vidx, hi1[vB]));
      }

      const __m512 vs = _mm512_mask_blend_ps(hiHalf, _mm512_set1_ps(bsA), _mm512_set1_ps(bsB));
      const __m512 xev = _mm512_loadu_ps(xe.data() + b * 8);
      const __m512 xod = _mm512_loadu_ps(xo.data() + b * 8);
      // acc += vs * (vlo*xe + vhi*xo)
      acc = _mm512_fmadd_ps(vs, _mm512_fmadd_ps(vlo, xev, _mm512_mul_ps(vhi, xod)), acc);
    }
    y[r] += _mm512_reduce_add_ps(acc);
  }
#else
  for (uint64_t r = 0; r < m.rows; ++r) {
    const float rs = rscale_of(m.rscales, r, s.row_scale_f32) * w;
    const uint64_t cbase = r * codes_row * 5;
    const uint64_t fbase = r * nblk * fbits;
    float acc = 0.0f;
    for (uint64_t b = 0; b < nblk; ++b) {
      const uint32_t field = (uint32_t)(load_bits(m.scales, fbase + b * fbits) & fmask);
      const float bs = rs * (float)(field & smask);
      if (bs == 0.0f) continue;
      const float* lut = cb.v.data() + (size_t)((field >> s.scale_bits) < cb.variants
                                                ? (field >> s.scale_bits) : 0) * cb.k * 2;
      uint64_t bits = load_bits(m.data, cbase + b * 8 * 5);
      float part = 0.0f;
      for (int i = 0; i < 8; ++i) {
        const float* e = lut + (size_t)((bits >> (5 * i)) & 31u) * 2;
        part += e[0] * xe[b * 8 + i] + e[1] * xo[b * 8 + i];
      }
      acc += bs * part;
    }
    y[r] += acc;
  }
#endif
}

} // namespace aff
