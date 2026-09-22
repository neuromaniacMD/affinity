// The lightning indexer on device.
//
// This is the single largest cost in long-prompt prefill, and it is why prefill throughput falls off
// with prompt length: the indexer engages only once the compressed cache outgrows `index_topk`, so a
// short prompt never pays it at all.
//
//   scores[t][s] = sum_h w[h] * relu(dot(q[t][h], k[s]))
//
// The work is quadratic in context, so it cannot stay on the host at any context worth having.
//
// ---- why it is a WMMA kernel and not a matvec ---------------------------------------------------
//
// It is the attention QK product with a different reduction: M = tokens, N = keys, K = the index
// dims. A is Q token-major and B is K key-major, both with the reduction index contiguous, so
// NEITHER operand needs a transpose — unlike the PV half of attention, which needed the hardware
// transpose load.
//
// The ReLU is the whole difficulty. It sits between the per-head product and the sum over heads, so
// the 64 heads cannot collapse into one accumulator chain: each head's 16x16 tile has to complete,
// be rectified, scaled and added. That costs about one ReLU and one FMA per mma.
//
// ---- the scales, and why one per token ----------------------------------------------------------
//
// Q carries ONE f32 scale per token, not one per (token, head), and K one per key. ReLU commutes
// with a positive scale, so
//
//   out[t][s] = qs[t] * ks[s] * sum_h w[h] * relu(raw_h[t][s])
//
// and both scales leave the head loop entirely — the epilogue is one multiply by a per-row value and
// one by a per-column value. A per-head scale would instead need eight lookups indexed by the
// accumulator row inside every head iteration. E4M3 spans ~18 binades, so one scale over a token's
// 8192 values costs nothing measurable; `bench/indexer.hip` checks that against an f32 reference.
#pragma once

#include <cstdint>

namespace aff {

constexpr uint32_t kIdxDim = 128;        // index_head_dim; the kernel is compiled for it
constexpr uint32_t kIdxTokTile = 16;     // tokens a block
constexpr uint32_t kIdxKeys = 128;       // keys a block

// `q8` is [nb][n_head][kIdxDim] E4M3 with `qs` one f32 a token; `k8` is [n_keys][kIdxDim] E4M3 with
// `ks` one f32 a key; `hw` is [nb][n_head], the indexer's own per-token projection. `out` is
// [nb][n_keys] f32, row-major.
//
// `n_head` must be a multiple of 8 and `n_keys` may be anything — the tail is masked on store.
void indexer_scores_hip(const uint8_t* q8, const float* qs, const uint8_t* k8, const float* ks,
                        const float* hw, float* out, uint32_t nb, uint32_t n_keys,
                        uint32_t n_head, void* stream);

// RoPE tail + the indexer's QAT (an E2M1 round trip per 32, behind an optional 128-wide Hadamard)
// over the whole chunk's queries, in place. `q` is [n][n_head][kIdxDim] f32 and token t sits at
// position pos0 + t.
//
// On the host this cost more than the scoring did, because `rope_tail` recomputes theta inside the
// head loop when theta depends only on (pos, i).
//
// `hadamard` is V4's rotation and must match what wrote the KEYS — the two are quantised
// independently and their scores only mean anything in a shared basis. V4 rotates both (here and
// in compress_kernel<1>); V4.1 rotates neither (here and in idx_key_kernel).
void idx_rope_qat_hip(float* q, uint32_t n, uint32_t n_head, uint32_t n_rot, uint32_t pos0,
                      float freq_scale, float theta_scale, float ext_factor, float lo, float hi,
                      void* stream, bool hadamard = true);

// f32 [n][dim] -> E4M3 [n][dim] + one f32 scale a row, on device. `dim` is n_head*kIdxDim for a
// query token and kIdxDim for a key.
void idx_quant_rows_hip(const float* src, uint8_t* dst, float* scale, uint32_t n, uint32_t dim,
                        void* stream);

// Top-k of each token's score row into an admission mask, `mstride` bytes a row.
//
// Matches `top_k_select` exactly: score descending, ties broken by the LOWER index. That is not
// cosmetic — the golden fixture reproduces a greedy continuation, and a different tie rule admits a
// different set of rows.
// `nkeys` is a per-token count (null = `stride` for every token): the compressed cache grows inside
// a chunk, so each token sees its own prefix. A token with fewer keys than `k` admits all of them.
//
// `mstride` is the PLANE's row pitch and `mlen` the prefix of each row that anyone will read — the
// attention's own compressed-row count, which can exceed the indexer's key count when the two
// compressors' capacities diverge, so the rows in between must read as "not admitted". They are two
// numbers because they are two different facts: the pitch is sized from the cache's CAPACITY and at
// a 1M cache that is 262146 bytes a row, where the prefix is what the context has actually filled.

// Device scratch for the multi-block select. Preallocated at startup and ZEROED once: every pass
// leaves the histogram, the arrival counter and the tie list back at zero, so no call has to clear
// them.
struct TopkScratch {
  uint32_t* hist = nullptr;    // [tok_cap][256]     bucket counts
  uint32_t* state = nullptr;   // [tok_cap][4]       prefix, need, gt, arrivals
  uint32_t* tie = nullptr;     // [tok_cap][1+1024]  count, then the indices tied at the threshold
  uint32_t tok_cap = 0;
};

// Bytes the scratch needs for `tok` tokens: the return value is the histogram's.
uint64_t idx_topk_scratch_bytes(uint32_t tok, uint64_t* state_bytes, uint64_t* tie_bytes);

// `sp` null, or a token count too large to be worth spreading, takes the single-block kernel — the
// one prefill wants, since a 1536-token chunk is already 1536 blocks.
// `adm` (optional) receives each token's admitted rows as an ASCENDING list, `adm_stride` entries
// a token, with the count in `admn[t]`. That is what attention actually wants: at most index_topk
// of a context's worth of candidate rows survive, and walking the rest to give them a weight of
// zero is the single largest read in a deep-context token. Null, or a stride below `k`, keeps the
// mask alone.
void idx_topk_mask_hip(const float* scores, uint8_t* mask, const uint32_t* nkeys, uint32_t nb,
                       uint32_t stride, uint32_t mstride, uint32_t mlen, uint32_t k, void* stream,
                       const TopkScratch* sp = nullptr, uint32_t* adm = nullptr,
                       uint32_t* admn = nullptr, uint32_t adm_stride = 0);

// f32 [n][dim] -> E4M3 [n][dim] + one f32 scale a row. Host side, for the bench and for rows the
// engine still builds on the host.
void idx_encode_rows(const float* src, uint8_t* dst, float* scale, uint32_t n, uint32_t dim);

}  // namespace aff
