// Shared-KV MQA attention on device. See attention_gpu.hip for the semantics that matter.
#pragma once

#include <cstdint>

namespace aff {

// KV cache element format. The cache is per layer and bounded (sliding window + indexer-admitted
// compressed rows), but at long context the compressed half dominates VRAM, and VRAM is expert
// residency — so this is a real lever, not a micro-optimisation.
//
//   MIXED  E4M3 for the nope part, f16 for the rope tail. 1.18 B/elem.  DEFAULT.
//   FP8    E4M3 throughout + one f32 scale per 64, i.e. 1.0625 B/elem.
//   F32    4 B/elem. Kept only as a reference, and it is WIDER than the data — see below.
//
// What the model actually produces, per cached row, is not one format but two. `ref_forward.py`:
//
//     kv[: W - RD] = fp8_act_quant(kv[: W - RD], 64)      # nope part, E4M3 on a per-64 pow2 scale
//     st["raw"][...] = kv.astype(np.float16)              # then the WHOLE row down to f16
//
// So the leading 448 dims carry 4 significant bits and the trailing 64 carry 11. MIXED stores each
// half at exactly its own precision and is therefore BIT-EXACT with the reference row: the E4M3
// bytes are the ones fp8_act_quant already chose, and the f16 tail is the one np.float16 chose.
//
// FP8 gets the nope part right for the same reason and then re-quantises the rope tail, which the
// model never asked for. That second lossy step is not free: against the numpy oracle it costs more
// than two orders of magnitude of relative error, for an eighth of the row, and leaving the tail
// alone fixes it for an eighth of the bytes.
//
// F32 is wider than the data and still not more faithful than MIXED: every cached row has been
// an fp16 round trip, so f32 merely spends more bytes storing zeroes in the bottom mantissa bits. It
// is here to answer "is the cache the problem", not to be run.
//
//   FP8R   E4M3 over the whole row on ONE pow2 scale, padded to 528 B. 1.03 B/elem, the smallest
//          of the four, and the only one the flash WMMA kernel can read.
//
// FP8R exists because of a constraint in the matrix core, not in the data. The PV product is
// O[head][dim] = sum_key P[head][key] * s(key, dimgroup) * b[key][dim]; a scale depending on both
// the reduction index and an output index cannot be folded out of a matrix multiply on either side,
// while a scale depending only on the key folds straight into P. Since the group scales are powers
// of two and E4M3 is a floating-point format, collapsing eight of them into one is exact except in
// dim groups more than ~6 binades below the row's own maximum. Measured against an f32 reference,
// the cache sits at plain E4M3 error, i.e. the collapse costs nothing measurable.
//
// bf16 is deliberately NOT offered. Its 8 mantissa bits are fewer than the f16 round-trip every
// cached row has already been through, at the same 2 bytes — strictly worse than f16 on both axes.
enum class KvDtype { F32 = 0, FP8 = 2, MIXED = 3, FP8R = 4 };

inline const char* kv_dtype_name(KvDtype d) {
  switch (d) {
    case KvDtype::F32:   return "f32";
    case KvDtype::FP8:   return "fp8";
    case KvDtype::MIXED: return "mixed";
    case KvDtype::FP8R:  return "fp8r";
  }
  return "?";
}
// Parses a --kv-dtype value. Returns false on an unknown name and leaves `out` alone, so a typo is
// a usage error rather than a silent switch to some other cache format.
inline bool kv_dtype_parse(const char* s, KvDtype* out) {
  if (!s) return false;
  if (!__builtin_strcmp(s, "f32"))   { *out = KvDtype::F32;   return true; }
  if (!__builtin_strcmp(s, "fp8"))   { *out = KvDtype::FP8;   return true; }
  if (!__builtin_strcmp(s, "mixed")) { *out = KvDtype::MIXED; return true; }
  if (!__builtin_strcmp(s, "fp8r"))  { *out = KvDtype::FP8R;  return true; }
  return false;
}

constexpr uint32_t kKvScaleGroup = 64;   // elements per FP8 scale; matches fp8_kv_qat

// Bytes one cached row occupies, scales and padding included. `n_rot` is the width of the rope tail
// and is ignored by every format but MIXED.
//
// MIXED lays a row out nope-then-tail-then-scales:
//     [0, N)            N   bytes  E4M3
//     [N, N + 2R)       2R  bytes  f16
//     [N + 2R, ...)     4N/64      f32 scales, one per 64 nope elements
// with the total rounded up to 4 so that row r+1's scales stay aligned. For W=512, R=64 that is
// 448 + 128 + 28 = 604 -> 604, already a multiple of 4.
inline uint64_t kv_row_bytes(KvDtype d, uint32_t width, uint32_t n_rot) {
  switch (d) {
    case KvDtype::F32: return (uint64_t)width * 4;
    case KvDtype::FP8: return (uint64_t)width + (uint64_t)(width / kKvScaleGroup) * 4;
    // 512 data bytes then one f32 scale, rounded to 16 so that every row — and every 16-dim window
    // inside it — is 8-byte aligned. The WMMA operand loads are 64- and 128-bit and a 516-byte
    // stride would misalign every odd row.
    case KvDtype::FP8R: return ((uint64_t)width + 4 + 15) & ~(uint64_t)15;
    case KvDtype::MIXED: {
      const uint32_t n_nope = width - n_rot;
      const uint64_t b = (uint64_t)n_nope + (uint64_t)n_rot * 2 +
                         (uint64_t)(n_nope / kKvScaleGroup) * 4;
      return (b + 3) & ~(uint64_t)3;
    }
  }
  return 0;
}

// Converts one f32 row into the cache format, into `dst` (kv_row_bytes wide). Host side: a row is
// 512 values and is written once, so this is not worth a kernel launch.
void kv_encode_row(KvDtype d, const float* src, uint32_t width, uint32_t n_rot, void* dst);

void kv_store_hip(KvDtype d, const float* kv, void* slot, uint32_t width, uint32_t n_rot,
                  void* stream);

// ---- the raw ring, and why it is wider than the sliding window --------------------------------
//
// Prefill attends a SUB-BATCH of tokens in one launch, which means every row in the sub-batch is
// committed before any of them attends. With a ring of exactly `sliding` slots that is wrong:
// a 1024-token chunk laps a 128-slot ring eight times, so by the time token 5 attends, its window
// has been overwritten by tokens 133 onwards and it sees the future. The failure is silent — the
// text stays fluent.
//
// Widening the ring to `sliding + kAttnBatch - 1` makes it exactly impossible. Token at position p
// needs slots for positions p-sliding+1 .. p; a commit at p+j (j < kAttnBatch) lands on the same
// slot only if i + j == ring for some i < sliding, and i + j <= (sliding-1) + (kAttnBatch-1) =
// ring - 1. One slot of slack short of a collision, by construction.
//
// The slot rule is then ABSOLUTE: position q lives at q % ring_slots, at every batch size and in
// decode as well. That is what removes the old `raw_head` cursor rather than complicating it.
constexpr uint32_t kAttnBatch = 256;
inline uint32_t attn_ring_slots(uint32_t sliding) { return sliding + kAttnBatch - 1; }

// `q` is [n_head, width], already RoPE'd. `raw` is the ring and `comp` is [n_comp] rows, each
// kv_row_bytes wide; both are attended over as one key set — the kernel does not care which half a
// row came from, which is why CSA and HCA are one implementation. The raw keys are the `n_raw`
// slots starting at `raw0`, wrapping at `ring`. `allowed` is an n_comp byte mask (null = all).
// `out` is [n_head, width] f32, still rotated: the caller applies the inverse RoPE.
void mqa_attend_hip(KvDtype dt, const float* q, const void* raw, const void* comp,
                    const uint8_t* allowed, const float* sinks, float* out, uint32_t n_head,
                    uint32_t raw0, uint32_t n_raw, uint32_t ring, uint32_t n_comp, uint32_t width,
                    uint32_t n_rot, float scale, void* stream);

// The same attention for `n` consecutive tokens in ONE launch — grid is (n_head, n), and the whole
// point is that it is one launch: per token per layer, dispatch and ramp cost several times the
// kernel itself, and a prefill chunk issues tens of thousands of them.
//
// `q` and `out` are the chunk's TOKEN-MAJOR buffers, [nb][n_head * width]; token `b0 + t` of the
// chunk is at position `pos0 + t`, and each token's raw window follows from that position alone.
// The kernel indexes off `b0` and `n_head` and never needs `nb` itself; bounding b0 + n against it
// is the caller's.
// `n_comp_v` is a device array of `n` per-token compressed-row counts — they differ inside a
// sub-batch because the compressor emits a row every `ratio` positions. `allowed` is a device
// [n][max_comp] mask plane or null; when any token in the sub-batch has a mask, every row must be
// filled, because there is no per-token "no mask" flag.
// `flat_n` is DSpark's window and 0 everywhere else. Normally each token's key window follows from
// its own position — [pos-sliding+1, pos] — which is what makes the kernel causal without a mask.
// The draft's is not causal and not per token: its `block` queries all see the same set, the
// target's last `sliding` positions PLUS all `block` of each other, because a drafted block is
// predicted jointly rather than left to right. So the window is handed in whole: `flat_n` slots
// from `flat_raw0`, identical for every token, and nothing else about the kernel changes.
void mqa_attend_batch_hip(KvDtype dt, const void* q, const void* raw, const void* comp,
                          const uint8_t* allowed, const uint32_t* n_comp_v, const float* sinks,
                          float* out, uint32_t b0, uint32_t n, uint32_t nb, uint32_t pos0,
                          uint32_t sliding, uint32_t ring, uint32_t n_comp_max, uint32_t max_comp,
                          uint32_t n_head, uint32_t width, uint32_t n_rot, float scale,
                          void* stream, bool q_bf16 = false, uint32_t flat_n = 0,
                          uint32_t flat_raw0 = 0);

// QAT + f16 round + cache encode for `n` consecutive tokens, one block per token, ONE launch.
// `kv` is the chunk's DIM-MAJOR [width][nb] latent buffer; token `b0 + t` goes to ring slot
// `(pos0 + t) % ring`. This is kv_qat_hip and kv_store_hip fused: as a per-token pair they moved a
// 2 KiB column through a scratch buffer, once per token per layer.
void kv_commit_batch_hip(KvDtype dt, const float* kv, uint32_t nb, uint32_t b0, uint32_t n,
                         uint8_t* raw, uint32_t pos0, uint32_t ring, uint32_t width,
                         uint32_t n_rot, void* stream);

// The same commit for ONE token, with the learned RMSNorm and the RoPE tail folded in as a
// prologue: `latent` is the RAW wkv output, not the normalised one. Unfused, decode runs those two
// steps as a single-block kernel writing d_u and then reads d_u straight back — two launches for a
// couple of KiB. Bit-identical to rms_norm_rope_kernel followed by kv_commit_batch_hip;
// the commit half is literally the same code, reached with Norm=true. Returns false having launched
// nothing if the shape is not the one the prologue assumes.
bool kv_norm_commit_hip(KvDtype dt, const float* latent, const float* nw, float eps, uint8_t* raw,
                        uint32_t pos, uint32_t ring, uint32_t width, uint32_t n_rot,
                        float freq_scale, float theta_scale, float ext_factor, float lo, float hi,
                        void* stream);

} // namespace aff
