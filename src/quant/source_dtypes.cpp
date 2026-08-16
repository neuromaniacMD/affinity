#include "source_dtypes.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace aff {

namespace {

// 256-entry LUT for E4M3, built once. Cheaper and clearer than bit-twiddling per element.
struct Fp8E4M3Table {
  float v[256];
  Fp8E4M3Table() {
    for (int i = 0; i < 256; ++i) {
      const uint8_t b = static_cast<uint8_t>(i);
      const int sign = (b >> 7) & 1;
      const int exp  = (b >> 3) & 0xF;
      const int mant = b & 0x7;
      float f;
      if (exp == 0) {
        // subnormal: 2^-6 * (mant/8)
        f = std::ldexp(static_cast<float>(mant) / 8.0f, -6);
      } else if (exp == 0xF && mant == 0x7) {
        f = std::numeric_limits<float>::quiet_NaN();   // OCP e4m3fn: no inf, S1111111 is NaN
      } else {
        f = std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, exp - 7);
      }
      v[i] = sign ? -f : f;
    }
  }
};
const Fp8E4M3Table g_e4m3;

// E2M1 grid, 16 entries.
constexpr float kFp4[16] = {
   0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

struct E8M0Table {
  float v[256];
  E8M0Table() {
    for (int i = 0; i < 256; ++i)
      v[i] = (i == 0xFF) ? std::numeric_limits<float>::quiet_NaN()
                         : std::ldexp(1.0f, i - 127);
  }
};
const E8M0Table g_e8m0;

} // namespace

float e8m0_to_float(uint8_t e) noexcept { return g_e8m0.v[e]; }
float fp8_e4m3_to_float(uint8_t v) noexcept { return g_e4m3.v[v]; }
float fp4_e2m1_to_float(uint8_t nibble) noexcept { return kFp4[nibble & 0xF]; }

bool mxfp4_dequant(const uint8_t* packed, const uint8_t* scales,
                   uint64_t rows, uint64_t cols,
                   float* out, NibbleOrder order) noexcept {
  constexpr uint64_t kBlock = 32;
  if (!packed || !scales || !out) return false;
  if (cols % kBlock != 0) return false;          // MXFP4 requires whole 32-element blocks
  const uint64_t nblk = cols / kBlock;
  const uint64_t packed_row = cols / 2;

  for (uint64_t r = 0; r < rows; ++r) {
    const uint8_t* pr = packed + r * packed_row;
    const uint8_t* sr = scales + r * nblk;
    float*         orow = out + r * cols;

    for (uint64_t b = 0; b < nblk; ++b) {
      const float s = g_e8m0.v[sr[b]];
      const uint8_t* pb = pr + b * (kBlock / 2);
      float* ob = orow + b * kBlock;
      for (uint64_t i = 0; i < kBlock / 2; ++i) {
        const uint8_t byte = pb[i];
        const uint8_t lo = byte & 0x0F;
        const uint8_t hi = (byte >> 4) & 0x0F;
        if (order == NibbleOrder::LowFirst) {
          ob[2 * i + 0] = kFp4[lo] * s;
          ob[2 * i + 1] = kFp4[hi] * s;
        } else {
          ob[2 * i + 0] = kFp4[hi] * s;
          ob[2 * i + 1] = kFp4[lo] * s;
        }
      }
    }
  }
  return true;
}

bool fp8_block_dequant(const uint8_t* weight, const uint8_t* scales,
                       uint64_t rows, uint64_t cols,
                       uint64_t block_rows, uint64_t block_cols,
                       float* out) noexcept {
  if (!weight || !scales || !out || block_rows == 0 || block_cols == 0) return false;
  const uint64_t sr_n = (rows + block_rows - 1) / block_rows;
  const uint64_t sc_n = (cols + block_cols - 1) / block_cols;

  for (uint64_t r = 0; r < rows; ++r) {
    const uint8_t* wr = weight + r * cols;
    float*         orow = out + r * cols;
    const uint8_t* srow = scales + (r / block_rows) * sc_n;
    for (uint64_t c = 0; c < cols; ++c)
      orow[c] = g_e4m3.v[wr[c]] * g_e8m0.v[srow[c / block_cols]];
  }
  (void)sr_n;
  return true;
}

void bf16_dequant(const uint16_t* src, uint64_t n, float* out) noexcept {
  for (uint64_t i = 0; i < n; ++i) out[i] = bf16_to_float(src[i]);
}

} // namespace aff
