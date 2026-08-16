// The sliding-window compressor on device.
//
// This was the last sequential host walk in the batched prefill, and a large share of its wall
// clock: one step per token per layer on the CPU, fed by draining the projections out of VRAM, and
// one kernel launch per emitted row on the way back.
//
// ---- why it looked sequential and is not ---------------------------------------------------------
//
// The host form streamed positions through a rolling state and emitted a row only on a ratio
// boundary, which reads as a state machine that has to be walked in order. It is not one. At
// ratio 4 the emitted row for group g is a softmax pool over exactly EIGHT raw positions — the four
// of group g-1 at column j, and the four of group g at column head_dim+j — and nothing else. The
// "state" is a window, not an accumulator, so row g depends on positions 4(g-1) .. 4g+3 and on no
// earlier row. Every row of a chunk can therefore be computed at once, one block each.
//
// The ratio-128 layers are the same shape with one window instead of two.
//
// What crosses a chunk boundary is only that window. It lives here as a RING indexed by
// `position & (hist-1)`, so there is no shifting to serialise: the compressor reads that slot for
// any position before the chunk, and a second launch writes the chunk's last `hist` positions into
// their own slots afterwards. A position before the sequence starts maps to a slot its own writer
// has not reached yet, which is why the reset value (score -1e30, kv 0) is still there to be read.
//
// ---- one kernel, the whole tail --------------------------------------------------------------
//
// pool -> rms_norm -> rope_tail -> QAT -> encode is one block here, and the row is written in its
// final cache form: the comp half as FP8R straight into `a_comp`, the indexer half as E4M3 plus one
// f32 scale a row, which is exactly what `indexer_scores_hip` wants. Nothing round-trips through a
// host mirror — see LayerState.
#pragma once

#include <cstdint>

#include "engine/ops.h"

namespace aff {

// Ring slots of raw projection kept across chunks, indexed by `position & (hist-1)` — so it has to
// be a power of two, which every shipped ratio is.
//
// At ratio 4 a row pools over its own group AND the one before it, so the window is 2*ratio wide
// and the projection carries both halves in one 2*head_dim row. At every other ratio only the
// current group contributes and the projection is head_dim wide. That asymmetry is the whole
// difference between the two compressors.
constexpr uint32_t comp_hist(uint32_t ratio) { return ratio == 4u ? 8u : ratio; }
constexpr uint32_t comp_width(uint32_t ratio, uint32_t head_dim) {
  return (ratio == 4u ? 2u : 1u) * head_dim;
}

// What one compressor needs for one layer for one chunk. Two of these run per layer: the KV
// compressor (head_dim = qk_width, output FP8R into the attention cache) and the indexer's
// (head_dim = 128, output E4M3 + a row scale).
struct CompressJob {
  const float* kv = nullptr;      // [n][comp_width] token-major, this chunk's projections
  const float* sc = nullptr;
  float* hist_kv = nullptr;       // [comp_hist(ratio)][comp_width(ratio, head_dim)], the ring
  float* hist_sc = nullptr;
  const float* ape = nullptr;     // [ratio][comp_width], added to the score before pooling
  const float* norm = nullptr;    // [head_dim]
  uint8_t* out = nullptr;         // row `slot0 + g` at `out + (slot0+g)*out_stride`
  float* out_scale = nullptr;     // FP8R keeps its scale inside the row; the indexer's is here
  uint64_t out_stride = 0;
  uint32_t slot0 = 0;
  uint32_t head_dim = 0;
};

// `n_out` rows for positions pos0 .. pos0+n-1: every p in that range with (p+1) % ratio == 0.
// `mode` 0 is the KV compressor (per-64 E4M3 QAT, then an FP8R row), 1 the indexer's (128-wide
// Hadamard, an E2M1 round trip per 32, then an E4M3 row). `ratio` and `head_dim` must both be
// powers of two, head_dim at least 64 — every shipped layer is. Mode 1 additionally requires
// head_dim exactly 128, and says so rather than silently normalising by the wrong factor.
void compressor_rows_hip(int mode, const CompressJob& j, uint32_t n, uint32_t pos0, uint32_t ratio,
                         uint32_t n_out, uint32_t n_rot, RopeDerived rope, float rms_eps,
                         void* stream);


// score = -1e30, kv = 0: the empty-state values, so a position before the sequence begins
// contributes weight zero rather than garbage.
// The two halves in ONE launch — see hist2_kernel. `kv_b` null runs only the first, so a layer
// without an index compressor takes the same path.
void compressor_hist2_hip(const float* kv_a, const float* sc_a, float* hkv_a, float* hsc_a,
                          uint32_t width_a, const float* kv_b, const float* sc_b, float* hkv_b,
                          float* hsc_b, uint32_t width_b, uint32_t n, uint32_t pos0, uint32_t hist,
                          void* stream);

void compressor_hist_reset_hip(float* hist_kv, float* hist_sc, uint32_t width, uint32_t hist,
                               void* stream);

}  // namespace aff
