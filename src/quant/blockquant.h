// Generic non-uniform block quantiser — the write side of the .aff pipeline.
//
// Structure follows the IQ_K family, but the codebooks are
// FITTED to this model's own weights rather than borrowed. ik_llama.cpp's grids were tuned against
// bf16-trained weights; DeepSeek-V4-Flash's experts are QAT-trained MXFP4, a materially different
// distribution (values cluster on 16 grid points rather than filling a Gaussian). Owning the
// quantiser (D2) is what allows the grid to be fitted to the distribution actually present.
//
// Layout per matrix, matching AffMatrixDesc's three sections:
//
//   row scale     float (or fp16) per ROW                      -> rscale
//   block field   (scale_bits + variant_bits) per block        -> scale
//   codes         code_bits per weight (or per PAIR)           -> data
//
//   w[i] ~= rscale[row] * blockscale * LUT[variant][code[i]]
//
// Two mechanisms beyond a plain scalar quantiser:
//
//  * PAIR CODING (pair=2): one code per weight PAIR against a 2-D codebook. Genuine vector
//    quantisation, and how IQ2_KL reaches 2.69 bpw while still decoding with a byte shuffle.
//
//  * GRID-SHIFT BIT (variant_bits=1): each block picks between two codebooks. ik_llama's IQ4_KS
//    uses a fixed half-quantum dither for this; we instead FIT both grids jointly by block-level
//    EM, so the data chooses the pair. Costs variant_bits/block = 0.031 bpw at block 32.

#pragma once

#include <cstdint>
#include <vector>
#include <string>

namespace aff {

struct QuantSpec {
  uint32_t block        = 32;  // weights per block scale
  uint32_t code_bits    = 4;   // bits per code
  uint32_t scale_bits   = 7;   // bits per block scale
  uint32_t pair         = 1;   // 1 = scalar LUT, 2 = 2-D LUT (one code per weight pair)
  uint32_t variant_bits = 0;   // 0 = one codebook; 1 = grid-shift bit (2 codebooks); 2 = 4
  bool     row_scale_f32 = true;
  // Block scale is a POWER OF TWO rather than an integer multiple of the row scale. Costs nothing
  // in bits and is what lets a GPU kernel apply the scale to a packed dword of E4M3 with one
  // integer add to each byte's exponent field, instead of carrying it in the LUT index. With a
  // narrow scale field it is not even a concession: 2 bits of integer scale is {1,2,3}, a 3x
  // range, where 2 bits of exponent is 8x.
  bool     pow2_block_scale = false;
  // How many LOW bits of the scale field are a mantissa rather than exponent, under
  // pow2_block_scale: scale = (1 + m/2^mb) * 2^(e - bias). This is the knob that trades LDS for
  // scale resolution, and it exists because the pure exponent form is worse on error than the
  // shipped integer scale.
  //
  // The kernel bakes the MANTISSA into its LUT, multiplying the table by 2^mb, and applies only
  // the exponent as a byte-wise add. So the cost is exactly 2^mb: at 4 variants and 32 pair codes
  // the table is 256 B at mb=0, 1 KB at mb=2, 2 KB at mb=3 — and 1638 B is the ceiling that keeps
  // five waves a SIMD in expert_gemm_wmma_kernel. mb=2 is the largest that fits.
  uint32_t scale_mant_bits = 0;
  // One bit per block that NEGATES every weight it reconstructs. A 4-wide codebook fitted to a
  // symmetric source spends half its entries on the negatives of the other half; this bit buys that
  // half back, so 1024 stored entries cover 2048 quads. It is free in the container — the 6-bit
  // field of AFF_Q2P875 has a bit left over once the scale is 3 bits and the variant 2 — and one
  // `^ 0x80808080` in the kernel, on the same packed dword the block scale already adds to.
  uint32_t sign_bits = 0;
  // One bit per block that ROTATES the reconstructed tuple by half its width — (a,b,c,d) also
  // spells (c,d,a,b). Only meaningful at pair == 4. Weights within a row have no positional
  // structure, so a codebook fitted to them holds both orderings of every tuple it holds at all,
  // and this bit buys one of those copies back. In the kernel it is a funnel shift of the packed
  // dword; here it is an index permutation. Same argument as `sign_bits`, different symmetry.
  uint32_t rot_bits = 0;

  uint32_t lut_size()         const { return 1u << code_bits; }
  uint32_t variants()         const { return 1u << variant_bits; }
  // How many symmetry transforms a block may pick between: sign, rotation, or both.
  uint32_t transforms()       const { return 1u << (sign_bits + rot_bits); }
  // LSB-first: scale, sign, rotation, variant. The GPU kernel reads the same order.
  uint32_t block_field_bits() const {
    return scale_bits + sign_bits + rot_bits + variant_bits;
  }

  // Exact bits per weight across both scale tiers; the per-row term is negligible.
  double bpw() const {
    return double(code_bits) / double(pair) + double(block_field_bits()) / double(block);
  }
};

// A fitted codebook, possibly with several variants selected per block.
// Values are stored as [variant][code][pair].
struct Codebook {
  std::vector<float> v;
  uint32_t pair = 1;
  uint32_t variants = 1;
  uint32_t k = 0;              // codes per variant

  const float* entry(uint32_t var, uint32_t code) const {
    return &v[((size_t)var * k + code) * pair];
  }
  uint32_t size() const { return k; }
};

// Forces the scalar codebook search, for A/B against the SIMD path. Runtime-settable rather than
// an env read cached in a static: a cached read cannot be flipped inside one process, which made
// an equivalence test silently compare the SIMD path against itself and report 0% difference.
void blockquant_use_simd(bool on);
bool blockquant_use_simd();

// The QuantSpec a container's routed-expert quant id means. `aff_quant` is a AffQuant, as
// AffMatrixDesc::quant holds it. ONE definition: the literal used to be spelled out at three call
// sites in the engine, which is three places to forget when a format is added — and getting it
// wrong is not a crash, it is a matrix decoded with the wrong field split.
QuantSpec expert_quant_spec(uint32_t aff_quant);

// The block scale a field's scale code stands for, in units of the ROW scale.
//   linear : the code IS the integer multiple.
//   pow2   : scale = (1 + m/2^mb) * 2^e, where the low `scale_mant_bits` of the code are m. There
//            is no bias here — it lives in the row scale, so that a GPU kernel can apply this by
//            adding `e` to the exponent field of a packed E4M3 byte and nothing else.
inline float block_scale_from_code(uint32_t s, const QuantSpec& sp) {
  if (!sp.pow2_block_scale) return static_cast<float>(s);
  const uint32_t mb = sp.scale_mant_bits;
  const uint32_t m = s & ((1u << mb) - 1u), e = s >> mb;
  float v = 1.0f + static_cast<float>(m) / static_cast<float>(1u << mb);
  return v * static_cast<float>(1ull << e);
}


// The power-of-two bias that `block_scale_from_code` does NOT apply: the row scale carries it, so
// that the largest scale code reproduces the row maximum exactly.
inline int block_scale_bias(const QuantSpec& sp) {
  return sp.pow2_block_scale ? (int)((1u << (sp.scale_bits - sp.scale_mant_bits)) - 1u) : 0;
}

// --- codebook fitting ---------------------------------------------------------------------------

// Fits `spec.variants()` codebooks jointly. With one variant this is plain Lloyd-Max (1-D) or
// k-means (2-D). With more, blocks are alternately assigned to their best-fitting variant and each
// variant is refitted on its assigned blocks — a block-level EM, initialised by dithering the
// single-variant solution so the variants start distinguishable.
//
// `samples` are per-block-normalised values, laid out block-contiguously (spec.block per block).
// `weights` (optional) supplies imatrix importance, one per sample for pair=1, one per PAIR
// for pair=2.
Codebook fit_codebook(const float* samples, uint64_t n_samples,
                      const QuantSpec& spec,
                      const float* weights = nullptr,
                      uint32_t iters = 50, uint64_t seed = 1);

// Gathers per-block-normalised samples from a matrix, for fitting.
// If `col_importance` (length cols) is given, `sample_weights` receives the matching per-sample
// importance so the codebook fit can be weighted.
void collect_normalized_samples(const float* w, uint64_t rows, uint64_t cols,
                                const QuantSpec& spec, std::vector<float>* out,
                                uint64_t max_samples = 1u << 22,
                                const float* col_importance = nullptr,
                                std::vector<float>* sample_weights = nullptr);

// --- what the GPU can actually hold ---------------------------------------------------------------
//
// For the 4-wide formats the container stores the codebook ALREADY PACKED as E4M3 quads, one dword
// per entry, because the kernel cannot afford to build 4096 entries from floats in every block. So
// the encoder must search against the packed values and not against the fitted floats — otherwise
// the rounding is an error source nothing ever measured. `project_codebook_e4m3` rounds the fit onto
// exactly what will be stored, in place.
//
// Under `pow2_scale` the block scale is an ADD to each byte's exponent field, which constrains the
// entries further: below 2^-6 an E4M3 is subnormal, where the leading mantissa bit is implicit-zero
// and an exponent add is simply wrong; at the top, an entry that takes the largest add (7) into
// exponent 15 with mantissa 7 spells NaN and poisons a WMMA accumulator. Magnitudes are clamped
// into [2^-6, 1.875], which the fit does not reach anyway.
void project_codebook_e4m3(Codebook* cb, bool pow2_scale);

// [variant][code] -> one dword of four E4M3 bytes, low byte first. Only meaningful at pair == 4.
std::vector<uint32_t> pack_codebook_e4m3_quads(const Codebook& cb);
void unpack_codebook_e4m3_quads(const uint32_t* packed, uint32_t variants, uint32_t k,
                                Codebook* out);

// Prebuilt nearest-neighbour acceleration for a wide codebook. Brute force over 1024 entries is
// 1024 distance evaluations a quad and the model has 69 billion quads; this makes it ~4. Build ONCE
// per codebook — it costs about a second, and a full quantise calls quantize_matrix 33024 times.
// Not needed (and ignored) for pair <= 2, where 32 entries fit in two AVX-512 registers.
struct CodebookSearch {
  // A uniform grid over [-kRange, kRange]^pair. Each cell lists every entry that can be the nearest
  // for SOME point in it — computed from the cell's corner distances, so the answer is exact. A
  // query outside the grid falls back to brute force, which the scale search reaches rarely: it
  // only tries exponents within one octave of the block's own maximum.
  static constexpr float kRange = 2.5f;
  static constexpr int   kCells = 25;
  uint32_t pair = 0, k = 0, variants = 0;
  std::vector<float>    ent;      // [variant][dim][code] — SoA, for the brute-force fallback
  std::vector<uint32_t> start;    // [variant][cell] -> offset into cand; one extra per variant
  std::vector<uint16_t> cand;
  bool ready() const { return pair == 4 && !cand.empty(); }
};

void build_codebook_search(const Codebook& cb, CodebookSearch* out);

// --- quantise / dequantise ----------------------------------------------------------------------

struct QuantizedMatrix {
  std::vector<uint8_t> data;     // packed codes
  std::vector<uint8_t> scales;   // packed (variant<<scale_bits)|scale per block
  std::vector<uint8_t> rscales;  // per-row scales (float or bf16 per spec)
  uint64_t rows = 0, cols = 0;
  QuantSpec spec;
  // Diagnostics
  double variant_split = 0;      // fraction of blocks choosing variant != 0
};

// `col_importance` (optional, length cols) is the imatrix weight E[x_j^2], normalised to mean 1.
// It changes the block-scale search and, for pair coding, the code choice. It does NOT change
// scalar code choice — importance is constant within a column, so the argmin is unaffected.
bool quantize_matrix(const float* w, uint64_t rows, uint64_t cols,
                     const QuantSpec& spec, const Codebook& cb,
                     QuantizedMatrix* out, std::string* err,
                     const float* col_importance = nullptr,
                     const CodebookSearch* search = nullptr);

bool dequantize_matrix(const QuantizedMatrix& q, const Codebook& cb, float* out,
                       std::string* err);

// --- error metrics --------------------------------------------------------------------------------

struct QuantError {
  double rel_fro = 0;      // ||W - W'||_F / ||W||_F
  double rel_out = 0;      // E||(W-W')x|| / E||Wx|| over random isotropic x
  double max_abs = 0;
  double rmse = 0;
  double rel_fro_w = 0;    // imatrix-weighted Frobenius error — the output-error metric when a
                           // measured activation distribution is available
};

QuantError measure_error(const float* ref, const float* got,
                         uint64_t rows, uint64_t cols, uint32_t probes = 8, uint64_t seed = 7,
                         const float* col_importance = nullptr);

} // namespace aff
