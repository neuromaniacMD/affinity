// Flash-style fp8 WMMA attention for the shared-KV (MQA) geometry.
//
// The kernel in `attention_gpu.hip` keeps the whole score row in LDS — `heads * n_keys` floats — so
// its LDS grows with context and past a few thousand keys it does not run at all: a launch failure,
// and well under a TFLOP/s below that. This is the replacement: online softmax, so LDS is O(tile)
// and independent of context.
//
// ---- the cache format this kernel reads --------------------------------------------------------
//
// One row is 512 E4M3 bytes then one f32 power-of-two scale, padded to `kFlashRowBytes` = 528 so
// that every row — and every 16-dim window inside it — is 8-byte aligned, which the 64-bit operand
// loads require. That is SMALLER than MIXED's 604 and FP8's 544.
//
// ONE scale per row rather than one per 64 dims, and that is a deliberate choice, not a shortcut.
// The PV product is O[head][dim] = sum_key P[head][key] * s(key, dimgroup) * b[key][dim]. A scale
// that depends on both the reduction index and an output index cannot be folded out of a matrix
// multiply on either side; a scale that depends only on the key folds straight into P. Since the
// group scales are powers of two and E4M3 is a floating-point format, collapsing them costs
// precision only in dim groups more than ~6 binades below the row's own maximum, which measures as
// nothing against an f32 reference.
#pragma once

#include <cstdint>

namespace aff {

constexpr uint32_t kFlashWidth = 512;      // qk head dim this kernel is compiled for
constexpr uint32_t kFlashRowBytes = 528;   // 512 E4M3 + f32 scale, padded for 8-byte operand loads

// ---- the key split -----------------------------------------------------------------------------
//
// A block serves NT*16 heads of ONE token and walks EVERY key, so the grid is (tokens, head tiles).
// Prefill has a chunk of tokens and fills the device; DECODE HAS ONE, and with a card's share of
// the heads that is a grid of exactly one workgroup — the whole attention sublayer on one CU,
// reading at a few percent of the card's bandwidth. It is CU-starved, not bandwidth-starved.
//
// So slice the KEYS: gridDim.z blocks each walk a disjoint contiguous run of them, and a combine
// pass merges the partial softmaxes the way the online update already does within a block —
// max, rescale, add. Total bytes read are unchanged, which is what makes this the split that pays
// and the head split the one that does not (see attn_flash.hip).
//
// Each slice writes an UNNORMALISED numerator plus its own (m, l), and the arena below holds them.
// It is preallocated at startup and never grows: the split is only taken when the natural grid is
// small, so `tok_cap` is a handful of tokens rather than a prefill chunk.
struct FlashSplit {
  float* o = nullptr;        // [slots][tok_cap][n_head][kFlashWidth] numerators, unnormalised
  float* ml = nullptr;       // [slots][tok_cap][n_head][2]           running max, denominator
  uint32_t tok_cap = 0;      // tokens the arena holds; the split is skipped for a launch above it
  uint32_t slots = 0;        // slices the arena holds
};

// Bytes the arena needs for `tok` tokens, `heads` heads a card and `slots` slices.
uint64_t flash_split_bytes(uint32_t tok, uint32_t heads, uint32_t slots, uint64_t* ml_bytes);

// f32 row -> one FP8R cache row. Host side; a row is written once and read n_head times.
void kv_encode_fp8r_row(const float* src, uint8_t* dst);

// Attention for `n` consecutive tokens in one launch, over an FP8R cache.
//
// `q` and `out` are the chunk's TOKEN-MAJOR [nb][n_head * width] buffers, of which this launch
// covers rows b0..b0+n. `nb` is that stride; the kernel indexes off `b0` and `n_head` and never
// needs it, and the caller is the one that has to bound b0 + n against it. Q is quantised to E4M3
// inside the kernel with one scale per head, so nothing upstream has to change to feed it, and
// `q_bf16` says whether that source is bf16 (prefill, where Q is 8.4 MB a launch) or f32 (decode).
// `raw` is the sliding-window ring and `comp` the compressed rows, both `kFlashRowBytes` apart and
// attended as one key set. `allowed` is a device [n][max_comp] mask plane or null.
//
// The compressed-row count comes EITHER from `n_comp_v`, a device array with one entry a token, OR
// — when that is null — from the scalar `n_comp1`, which every token then shares. Prefill needs the
// array because its tokens legitimately differ. Decode does not, and the array costs it a
// `hipMemsetD32Async` of one word before every launch — a blit kernel and a launch boundary per
// layer per rank, to store a single integer. A scalar travels inside the dispatch packet, so it
// needs no buffer, no upload and nothing to synchronise on.

// `adm`, when non-null, is the indexer's ASCENDING list of admitted compressed rows for each token,
// `adm_stride` entries apart — and then `n_comp_v` counts THOSE, not the whole compressed cache.
// It is the difference between reading every candidate row and reading the at most index_topk that
// survive: 16 MB against 270 KB at 123K context. `allowed` must be null when it is given; the two
// say the same thing, and the list says it in the form that saves the read.
void mqa_attend_flash_hip(const void* q, const void* raw, const void* comp,
                          const uint8_t* allowed, const uint32_t* n_comp_v, uint32_t n_comp1,
                          const float* sinks,
                          float* out, uint32_t b0, uint32_t n, uint32_t nb, uint32_t pos0,
                          uint32_t sliding, uint32_t ring, uint32_t max_comp, uint32_t n_head,
                          float scale, void* stream, bool q_bf16 = false,
                          const FlashSplit* split = nullptr, uint32_t n_comp_hint = 0,
                          const uint32_t* adm = nullptr, uint32_t adm_stride = 0);

}  // namespace aff
