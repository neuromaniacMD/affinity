// Decoders for the dtypes DeepSeek-V4-Flash actually ships.
//
// Determined by inspecting the real checkpoint, with `aff-probe`:
//
//   routed experts   I8 nibble-packed + F8_E8M0 scale per 32   -> MXFP4, exactly 4.25 bpw
//                    w1/w3: [inter, hidden/2] I8, [inter, hidden/32] scale
//                    w2:    [hidden, inter/2] I8, [hidden, inter/32] scale
//   attention,       F8_E4M3 + F8_E8M0 scale per 128x128 block
//   shared experts
//   routers, norms   BF16
//   sinks, hc_*      F32
//
// Everything decodes to float32; requantisation to the target IQ_K format happens downstream.

#pragma once

#include <cstdint>
#include <cstddef>

namespace aff {

// --- scalar element decoders ------------------------------------------------------------------

// E8M0: a bare 8-bit exponent, value = 2^(e-127). 0xFF is NaN. The OCP microscaling scale type.
float e8m0_to_float(uint8_t e) noexcept;

// E4M3 (OCP "fn" variant: no infinities, 0xFF/0x7F are NaN).
float fp8_e4m3_to_float(uint8_t v) noexcept;

// E2M1, the 4-bit element of MXFP4. Grid: +-{0, 0.5, 1, 1.5, 2, 3, 4, 6}.
float fp4_e2m1_to_float(uint8_t nibble) noexcept;

inline float bf16_to_float(uint16_t v) noexcept {
  const uint32_t u = static_cast<uint32_t>(v) << 16;
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}

inline uint16_t float_to_bf16(float f) noexcept {
  uint32_t u;
  __builtin_memcpy(&u, &f, 4);
  // round-to-nearest-even
  const uint32_t r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16;
  return static_cast<uint16_t>(r);
}

// --- nibble order -------------------------------------------------------------------------------
//
// CONFIRMED low-nibble-first, from DeepSeek's own inference/convert.py:
//
//     x = x.view(torch.uint8)
//     low  = x & 0x0F
//     high = (x >> 4) & 0x0F
//     x = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(2)
//
// stack(..., dim=-1).flatten() interleaves low into element 0 and high into element 1.
// This also matches torch's `float4_e2m1fn_x2` and the OCP MX spec.
enum class NibbleOrder { LowFirst, HighFirst };

// --- block decoders -----------------------------------------------------------------------------

// MXFP4 -> float32. `packed` holds `rows * (cols/2)` bytes; `scales` holds `rows * (cols/32)`
// E8M0 bytes. Output is `rows * cols` floats, row-major.
// Returns false if the shapes are inconsistent.
bool mxfp4_dequant(const uint8_t* packed, const uint8_t* scales,
                   uint64_t rows, uint64_t cols,
                   float* out, NibbleOrder order = NibbleOrder::LowFirst) noexcept;

// FP8-E4M3 with a per-(128x128) block E8M0 scale -> float32.
// `weight` is rows*cols E4M3 bytes; `scales` is ceil(rows/BS) * ceil(cols/BS) E8M0 bytes.
bool fp8_block_dequant(const uint8_t* weight, const uint8_t* scales,
                       uint64_t rows, uint64_t cols,
                       uint64_t block_rows, uint64_t block_cols,
                       float* out) noexcept;

void bf16_dequant(const uint16_t* src, uint64_t n, float* out) noexcept;

} // namespace aff
