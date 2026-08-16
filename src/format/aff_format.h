// affinity-engine — .aff container on-disk format
//
// Design rationale:
//
//  * EXPERT-ORIENTED, not tensor-oriented. GGUF stores one tensor per entry; our hot path is
//    "give me expert (layer, id) as one contiguous range I can DMA in a single transfer".
//    So an expert's gate/up/down live adjacent, and addressing is O(1) arithmetic — no lookup
//    table walk on the critical path.
//  * FIXED STRIDE PER LAYER. All experts within a layer share a format and size, so
//        addr = pool + layer.base + id * layer.stride
//    Format may still vary *between* layers (early/late layers are more error-sensitive).
//  * SCALES SPLIT FROM WEIGHTS (SoA). Quantised data and scale arrays are separate blocks so the
//    weight stream is dense and 512 B-coalescable on GPU and pure-sequential on CPU.
//  * ALIGNMENT. The expert pool starts 2 MiB-aligned so it can be mapped with huge pages; each
//    expert is 4 KiB-aligned so a transfer never straddles a page boundary needlessly.
//    Sub-blocks inside an expert are 256 B-aligned — the measured gfx1201 cacheline.
//
// Little-endian only (x86-64 / RDNA4). Not portable by design — see requirement R2.

#pragma once

#include <cstdint>
#include <cstddef>

namespace aff {

// ---------------------------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------------------------

inline constexpr char     kMagic[8]        = {'A','F','F','I','N','I','T','Y'};
inline constexpr uint32_t kVersion         = 1;

inline constexpr uint64_t kPoolAlign       = 2u << 20;   // 2 MiB — huge-page mappable
inline constexpr uint64_t kExpertAlign     = 4096;       // page
inline constexpr uint64_t kBlockAlign      = 256;        // gfx1201 cacheline (measured)

inline constexpr uint32_t kMaxTensorName   = 128;

// File-level flags
enum AffFlags : uint32_t {
  AFF_FLAG_NONE          = 0,
  AFF_FLAG_HAS_PROFILE   = 1u << 0,   // expert-popularity profile present
};

// ---------------------------------------------------------------------------------------------
// Quantisation formats
//
// `bits_num/bits_den` express bits-per-weight as an exact rational so sizes are computed with
// integer math only — no float rounding anywhere in the layout path.
// ---------------------------------------------------------------------------------------------

enum AffQuant : uint32_t {
  AFF_F32       = 0,
  AFF_F16       = 1,
  AFF_BF16      = 2,
  AFF_FP8_E4M3  = 3,
  AFF_Q8_0      = 4,   // 8-bit + per-block fp16 scale

  // IQ_K family (ik_llama.cpp lineage, MIT). Decode is a single byte-shuffle to int8 —
  // fits comfortably inside the measured CPU budget.
  AFF_IQ2_KL    = 16,  // ~2.69 bpw — capacity lever
  AFF_IQ3_KS    = 17,  // ~3.19 bpw — CPU-resident expert default
  AFF_IQ4_KS    = 18,  // ~4.25 bpw
  AFF_IQ5_KS    = 19,  // ~5.25 bpw — error-sensitive tensors

  // Our own format. Not an IQ_K variant: codes address a fitted 2-D
  // codebook a PAIR of weights at a time, and each 16-weight block carries a 4-bit scale plus a
  // 2-bit codebook-variant selector.
  AFF_Q2P875    = 32,  // 2.875 bpw exactly — the locked routed-expert format
  // The SAME 2.875 bpw and byte-for-byte the same container geometry: 80 bytes of codes and 12 of
  // block fields per 256 weights. What changes is what a code addresses — 10 bits over a
  // 1024-entry FOUR-dimensional codebook instead of 5 bits over a 32-entry two-dimensional one —
  // and how the 6-bit field splits: 3 bits of power-of-two scale (an add to each E4M3 byte's
  // exponent), 1 bit that negates the block, 1 bit that rotates the quad by two bytes, 1 bit of
  // codebook variant, against the old 4-bit integer scale + 2-bit variant. Two variants and not
  // four, because a 1024-entry quad table is 4 KB and the expert GEMM's staging leaves room for
  // 8 KB before occupancy drops a block. That halves the LDS traffic of the decode — one read per
  // FOUR weights where the pair form takes one per two. Because the geometry is identical,
  // placement, streaming, the reader and every size calculation are untouched; only the decode
  // differs.
  AFF_Q2P875_Q4 = 33,
};

// The IQ_K family uses a 256-weight SUPER-BLOCK with sub-block scales every 32 weights, plus a
// single scale PER ROW. Three separate quantities, so three fields.
//
// Note a deliberate deviation from ik_llama.cpp: upstream interleaves the sub-block scales inside
// the 136-byte block struct. We store them as a separate SoA array so the weight stream is dense
// and 512 B-coalescable. Same bits, different arrangement.
struct AffQuantInfo {
  AffQuant    id;
  const char* name;
  uint32_t    block;           // weights per super-block
  uint32_t    data_bytes;      // packed weight data per super-block (quants only)
  uint32_t    scale_bytes;     // sub-block scale bytes per super-block
  uint32_t    row_scale_bytes; // scale bytes per matrix ROW (0 if none)
};

// Single source of truth for format geometry. The bpw figures are verified against the ik_llama.cpp
// definitions the IQ_K entries are taken from, and `tests/test_aff_format.cpp` re-derives them.
inline constexpr AffQuantInfo kQuantTable[] = {
  { AFF_F32,      "f32",        1,   4,  0, 0 },
  { AFF_F16,      "f16",        1,   2,  0, 0 },
  { AFF_BF16,     "bf16",       1,   2,  0, 0 },
  // FP8 as the CHECKPOINT ships it: E4M3 bytes plus one UE8M0 scale per 128x128 tile. The
  // "super-block" is therefore a whole tile (16384 weights, 16384 data bytes, 1 scale byte), which
  // is 8.0005 bpw. Storing this verbatim is bit-exact with the source, where dequantising it to
  // bf16 would double the largest traffic source in the engine for no quality gain.
  { AFF_FP8_E4M3, "fp8_e4m3", 16384, 16384, 1, 0 },
  { AFF_Q8_0,     "q8_0",      32,  32,  2, 0 },
  // (data + scale) * 8 / 256 == bpw:
  { AFF_IQ2_KL,   "iq2_kl",   256,  80,  6, 2 },  // (80+6)*8/256 = 2.6875 bpw, + row fp16
  { AFF_IQ3_KS,   "iq3_ks",   256,  96,  6, 2 },  // (96+6)*8/256 = 3.1875 bpw, + row fp16
  { AFF_IQ4_KS,   "iq4_ks",   256, 128,  8, 4 },  // (128+8)*8/256 = 4.25 bpw, + row float
  { AFF_IQ5_KS,   "iq5_ks",   256, 160,  8, 4 },  // (160+8)*8/256 = 5.25 bpw, + row float
  // 5 bits per PAIR = 2.5 bpw of codes, plus 6 bits per 16-weight block = 12 bytes per 256.
  { AFF_Q2P875,   "q2p875",   256,  80, 12, 2 },  // (80+12)*8/256 = 2.875 bpw, + row bf16
  // Identical geometry on purpose — see the enum. 64 codes of 10 bits is the same 80 bytes that 128
  // codes of 5 bits are, so every offset, stride and slab size in the engine is unchanged.
  { AFF_Q2P875_Q4,"q2p875q4", 256,  80, 12, 2 },  // (80+12)*8/256 = 2.875 bpw, + row bf16
};

const AffQuantInfo* quant_info(AffQuant q) noexcept;
const char*         quant_name(AffQuant q) noexcept;

// Bytes of packed weight data for `n` weights in format `q` (quants only).
uint64_t quant_data_bytes(AffQuant q, uint64_t n) noexcept;
// Bytes of sub-block scale data for `n` weights in format `q`.
uint64_t quant_scale_bytes(AffQuant q, uint64_t n) noexcept;
// Bytes of per-row scale data for a matrix with `rows` rows.
uint64_t quant_row_scale_bytes(AffQuant q, uint64_t rows) noexcept;
// Exact bits-per-weight including both scale tiers, for a [rows x cols] matrix.
double   quant_bpw(AffQuant q, uint64_t rows, uint64_t cols) noexcept;

inline uint64_t align_up(uint64_t v, uint64_t a) noexcept { return (v + a - 1) / a * a; }

// ---------------------------------------------------------------------------------------------
// On-disk structures. All POD, explicitly sized, natural alignment, static_assert'd below.
// ---------------------------------------------------------------------------------------------

// One matrix inside an expert (gate / up / down), or a standalone non-expert tensor.
// Offsets are relative to the start of the containing object (the expert, or the tensor blob).
struct AffMatrixDesc {
  uint64_t rows;
  uint64_t cols;          // logical shape; row-major, `cols` is the reduction dim for gate/up
  uint32_t quant;         // AffQuant
  uint32_t _pad0;
  uint64_t data_off;      // packed weight data (quants only)
  uint64_t data_size;
  uint64_t scale_off;     // sub-block scales, SoA — separate from data for coalescing
  uint64_t scale_size;
  uint64_t rscale_off;    // per-row scales (fp16 or float, per AffQuantInfo::row_scale_bytes)
  uint64_t rscale_size;
};

// Per-layer expert geometry. All experts in a layer are identical in shape and format,
// which is what makes addressing O(1).
struct AffLayerDesc {
  uint32_t      layer;
  uint32_t      n_experts;
  uint64_t      base_off;     // byte offset of expert 0, relative to expert pool start
  uint64_t      stride;       // bytes between consecutive experts (already aligned)
  AffMatrixDesc gate;
  AffMatrixDesc up;
  AffMatrixDesc down;
};

// A non-expert tensor (embeddings, attention, norms, routers, shared expert, lm_head, MTP).
struct AffTensorEntry {
  char          name[kMaxTensorName];
  AffMatrixDesc desc;
  uint64_t      blob_off;     // offset of this tensor's blob, relative to tensor-data section
};

struct AffHeader {
  char     magic[8];
  uint32_t version;
  uint32_t flags;

  uint64_t meta_off;          // JSON metadata blob (model config, tokenizer ref, quant recipe)
  uint64_t meta_size;

  uint64_t tensor_dir_off;    // array of AffTensorEntry
  uint64_t tensor_count;
  uint64_t tensor_data_off;   // base for AffTensorEntry::blob_off
  uint64_t tensor_data_size;

  uint64_t layer_dir_off;     // array of AffLayerDesc
  uint64_t layer_count;

  uint64_t expert_pool_off;   // 2 MiB-aligned; base for AffLayerDesc::base_off
  uint64_t expert_pool_size;

  uint64_t profile_off;       // optional expert-popularity profile (float32 per (layer,expert))
  uint64_t profile_size;

  uint64_t file_size;

  // Sized so the header stays 192 bytes: this absorbs a retired checksum field, so containers
  // written before it was dropped still parse.
  uint8_t  reserved[72];
};

static_assert(sizeof(AffMatrixDesc)  == 72,  "AffMatrixDesc layout changed");
static_assert(sizeof(AffLayerDesc)   == 240, "AffLayerDesc layout changed");
static_assert(sizeof(AffTensorEntry) == 208, "AffTensorEntry layout changed");
static_assert(sizeof(AffHeader)      == 192, "AffHeader layout changed");
static_assert(alignof(AffHeader)     == 8,   "AffHeader alignment changed");

// ---------------------------------------------------------------------------------------------
// Layout helper — computes an expert's internal layout for a given geometry.
// Shared by writer and reader so the two can never disagree.
// ---------------------------------------------------------------------------------------------

struct ExpertGeometry {
  uint64_t hidden;             // model hidden size
  uint64_t intermediate;       // moe_intermediate_size
  AffQuant gate_q, up_q, down_q;
};

// Fills gate/up/down descs and returns the aligned per-expert stride in bytes.
uint64_t compute_expert_layout(const ExpertGeometry& g,
                               AffMatrixDesc& gate,
                               AffMatrixDesc& up,
                               AffMatrixDesc& down) noexcept;

} // namespace aff
