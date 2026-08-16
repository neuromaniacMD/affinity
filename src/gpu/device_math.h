// The numeric primitives that MUST have exactly one definition on device.
//
// These are not here for tidiness. Each one is a place where a second copy would be a second
// opinion about what a number means, and both opinions would produce fluent output: a RoPE that
// disagrees with `rope_tail` by one term of the YaRN mix is a plausible attention pattern rather
// than a crash, and two byte-identical E4M3 decoders stay identical only until one is edited.
//
// Host-side counterparts live in engine/ops.cpp and are checked against these by aff-gpucheck.

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace aff {

// E4M3 -> f32 by bit manipulation, no table: rebias the 4-bit exponent by +120 into f32's 8-bit
// field and place the 3 mantissa bits at 20. Exponent 0 is subnormal (no implicit leading 1) and
// becomes mantissa * 2^-9, selected without a branch.
// ONE INSTRUCTION on gfx1201: v_cvt_f32_fp8. The software form below is ~11 VALU ops and is kept
// as the fallback for a target without the builtin.
//
// Checked exhaustively over all 256 byte values against that fallback. They agree on 254 of them.
// The two that differ are 0x7F and 0xFF, which are E4M3-fn's NaN encodings: the hardware returns
// NaN, as the format says, and the software form returns +-480 — an accident of never testing
// exponent 15, not a decision. Nothing in this engine writes those bytes (the encoder clamps to
// +-448, which is 0x7E, and the expert codebook is clamped to exponent <= 7), so the change is
// inert on real data and turns a silent 480 into a visible NaN if that ever stops being true.
__device__ __forceinline__ float e4m3_decode(uint32_t b) {
#if __has_builtin(__builtin_amdgcn_cvt_f32_fp8)
  return __builtin_amdgcn_cvt_f32_fp8(b, 0);
#else
  const uint32_t sign = (b & 0x80u) << 24;
  const uint32_t ex   = (b >> 3) & 0x0Fu;
  const uint32_t mn   = b & 0x07u;
  const float nrm = __uint_as_float(((ex + 120u) << 23) | (mn << 20));
  const float sub = (float)mn * (1.0f / 512.0f);
  return __uint_as_float(__float_as_uint(ex ? nrm : sub) | sign);
#endif
}

// The same decode with the SUBNORMAL CASE REMOVED, for callers that can prove exponent != 0.
//
// PRECONDITION: the exponent field of `b` is 1..14. Feeding it a subnormal returns
// 2^-7 * 1.MMM instead of MMM * 2^-9 — a wrong number, not a NaN, so nothing downstream notices.
//
// The routed-expert codebook satisfies this by construction and always has: project_codebook_e4m3
// clamps entries to [2^-6, 1.875], i.e. stored exponent 1..7, precisely so the block scale can be
// applied as an ADD to that field (below 2^-6 the leading mantissa bit is implicit-zero and an
// exponent add is simply wrong). The largest scale add is 7, so the decoded exponent is 1..14 and
// 15 — E4M3-fn's NaN — is unreachable too. static_placement checks the uploaded table against this
// at load, so a container that violated it would fail loudly rather than decode quietly wrong.
//
// Why it is worth a second function: the ISA. The general decode above compiles to ~11 VALU ops,
// of which four (a v_cvt_f32_ubyte0, a v_mul_f32, a v_cmp_eq_u32 and a v_cndmask_b32) exist only
// for the subnormal branch. The expert gemv issues 64 of these per column group per lane against
// 64 FMAs — ~700 VALU instructions for 64 multiply-adds — so it is ALU-bound on this function and
// nothing else. Here the whole thing is a shift, an add and an or:
//
//   (b & 0x7F) << 20   puts EEEE in bits 23-26 and MMM in 20-22, which is where f32 wants them
//   + (120 << 23)      biases the exponent; it cannot carry into the mantissa, and EEEE <= 14
//                      cannot carry out of the exponent field either
//   | (b & 0x80) << 24 the sign
__device__ __forceinline__ float e4m3_decode_norm(uint32_t b) {
  const uint32_t v = ((b & 0x7Fu) << 20) + 0x3C000000u;
  return __uint_as_float(v | ((b & 0x80u) << 24));
}

typedef float aff_f32x2 __attribute__((ext_vector_type(2)));

// TWO E4M3 bytes of a packed dword to two f32, in ONE instruction — `v_cvt_pk_f32_fp8`, which
// gfx1201 has because RDNA4 carries the FP8 WMMA path. `hi` selects bytes 2,3 over bytes 0,1.
//
// It is the same number as e4m3_decode_norm above over the whole exponent 1..7 range this codebook
// uses, checked exhaustively over all 256 byte values including signed ones.
// Where they differ is outside that range and in the hardware's favour: it handles subnormals
// correctly, which the software fast path does not. So this is strictly the safer of the two as
// well as much cheaper — four bytes go from a chain of VALU ops to two instructions.
//
// The fallback exists for a target without the builtin; it is bit-identical inside the precondition.
// HI is a template parameter because the builtin's word select must be a constant integer.
template <bool HI>
__device__ __forceinline__ aff_f32x2 e4m3_decode_pk(uint32_t q) {
#if __has_builtin(__builtin_amdgcn_cvt_pk_f32_fp8)
  return __builtin_amdgcn_cvt_pk_f32_fp8(q, HI);
#else
  constexpr uint32_t s = HI ? 16u : 0u;
  aff_f32x2 r;
  r.x = e4m3_decode_norm((q >> s) & 0xFFu);
  r.y = e4m3_decode_norm((q >> (s + 8u)) & 0xFFu);
  return r;
#endif
}

// f32 -> E4M3 byte, the mirror of attention_gpu.hip's host e4m3_encode. Written out rather than
// shared with the host copy because the host one uses libm and this one must not.
__device__ __forceinline__ uint8_t e4m3_enc(float f) {
  const uint32_t sign = (__float_as_uint(f) >> 24) & 0x80u;
  float a = fabsf(f);
  if (!(a > 0.0f)) return (uint8_t)sign;
  if (a > 448.0f) a = 448.0f;
  if (a >= 0x1p-6f) {
    const uint32_t u = __float_as_uint(a);
    int e = (int)((u >> 23) & 0xFFu) - 127;
    uint32_t mm = (u & 0x7FFFFFu) >> 20;
    const uint32_t rem = u & 0xFFFFFu, half = 1u << 19;
    if (rem > half || (rem == half && (mm & 1u))) { if (++mm == 8u) { mm = 0; ++e; } }
    if (e > 8) { e = 8; mm = 7; }
    return (uint8_t)(sign | ((uint32_t)(e + 7) << 3) | mm);
  }
  // rintf, not `+ 0.5f`: the host uses lrintf, and round-half-up against round-half-even disagrees
  // on exact halves. Two encoders for one format only stay one format if they round alike.
  uint32_t mm = (uint32_t)rintf(a * 512.0f);        // subnormal: m * 2^-9
  if (mm > 7u) mm = 7u;
  return (uint8_t)(sign | mm);
}

// Nearest of {0, .5, 1, 1.5, 2, 3, 4, 6}, ties to the EVEN TABLE INDEX — the rule
// tools/ref_forward.py's e2m1_round follows, and not the same rule as ties-to-even-mantissa.
__device__ __forceinline__ float e2m1_round_d(float x) {
  const float a = fminf(fabsf(x), 6.0f);
  float r;
  if (a <= 0.25f) r = 0.0f;
  else if (a < 0.75f) r = 0.5f;
  else if (a <= 1.25f) r = 1.0f;
  else if (a < 1.75f) r = 1.5f;
  else if (a <= 2.5f) r = 2.0f;
  else if (a < 3.5f) r = 3.0f;
  else if (a <= 5.0f) r = 4.0f;
  else r = 6.0f;
  return x < 0.0f ? -r : r;
}

__device__ __forceinline__ float yarn_ramp_d(float low, float high, int i) {
  const float y = ((float)i / 2.0f - low) / fmaxf(0.001f, high - low);
  return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

// Just the angle. The host loop walks theta_extrap by repeated multiplication; here it is
// pos * theta_scale^k, the same value in one step instead of k of them. Split out so a caller that
// writes the rotated pair SOMEWHERE ELSE — the inverse RoPE folded into attn_out's matvec writes
// LDS, not back over its input — computes it from the same body. Two call sites that agree by
// inspection stay agreeing only until one is edited.
__device__ __forceinline__ void rope_cs(uint32_t k, uint32_t pos, float freq_scale,
                                        float theta_scale, float ext_factor, float lo, float hi,
                                        float sin_sign, float* c, float* sn) {
  const uint32_t i = k * 2;
  const float theta_extrap = (float)pos * __powf(theta_scale, (float)k);
  float theta = freq_scale * theta_extrap;
  if (ext_factor != 0.0f) {
    const float mix = yarn_ramp_d(lo, hi, (int)i) * ext_factor;
    theta = theta * (1.0f - mix) + theta_extrap * mix;
  }
  *c = __cosf(theta);
  *sn = sin_sign * __sinf(theta);
}

// One rotated PAIR, `stride` floats apart. `stride` exists for the batched prefill path, where
// activations are stored dim-major ([dim][nb]) so that a wave's lanes each own a different token.
// Every single-token caller passes 1 and the compiler folds the multiply away.
__device__ __forceinline__ void rope_pair_strided(float* __restrict__ p, uint32_t k, uint32_t stride,
                                                  uint32_t pos, float freq_scale, float theta_scale,
                                                  float ext_factor, float lo, float hi,
                                                  float sin_sign) {
  const uint32_t i = k * 2;
  float c, sn;
  rope_cs(k, pos, freq_scale, theta_scale, ext_factor, lo, hi, sin_sign, &c, &sn);
  const size_t a = (size_t)i * stride, b = (size_t)(i + 1) * stride;
  const float x0 = p[a], x1 = p[b];
  p[a] = x0 * c - x1 * sn;
  p[b] = x0 * sn + x1 * c;
}

__device__ __forceinline__ void rope_pair(float* __restrict__ p, uint32_t k, uint32_t pos,
                                          float freq_scale, float theta_scale, float ext_factor,
                                          float lo, float hi, float sin_sign) {
  rope_pair_strided(p, k, 1u, pos, freq_scale, theta_scale, ext_factor, lo, hi, sin_sign);
}

} // namespace aff
