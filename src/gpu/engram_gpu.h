// Engram's gate and its write into the hidden state, on device.
//
// The lookup and the `wkv` projection happen before this: `wkv` turns the 24 looked-up rows into
// one key per hc copy plus a shared value, and what is left is a gate — how well that key matches
// the stream it is about to be added to — and the add itself.
//
//   rstd  = rsqrt(mean(h[c]^2) + eps) * rsqrt(mean(key[c]^2) + eps)
//   dot   = sum_i h[c][i] * w[c][i] * key[c][i] * rstd / sqrt(dim)
//   gate  = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot))       <- signed sqrt, per the reference
//   h[c] += gate * value                                         <- ONE value, all four copies
//
// `w` is `q_weight * k_weight`, which the reference only ever uses as that product, so the two are
// multiplied once at load and never again.
//
// ---- why one thread a token ----------------------------------------------------------------------
//
// Both the hidden state and the `wkv` output are DIM-major, `[row][nb]`: that is the layout the
// batched GEMM writes and the one every other consumer here reads. A reduction over `dim` with one
// thread per element would therefore stride by `nb` and coalesce nothing. One thread per TOKEN
// walking `dim` in a loop reads consecutive addresses across the warp instead, which is the same
// arithmetic and a full-width load. At decode nb is 1 and the distinction is moot.
#pragma once

#include <cstdint>

namespace aff {

// `hc` is [n_hc][n_embd][nb], f32 when `hc_bf16` is false (decode) and bf16 when true (prefill's
// lanes are bf16 — see Impl::Dev::b_hc). `kv` is [(n_hc+1)*n_embd][nb] f32, the wkv output: the
// first n_hc bands are the per-copy keys and the last is the shared value. `w` is [n_hc][n_embd]
// f32. The state is updated in place.
void engram_gate_hip(void* hc, bool hc_bf16, const float* kv, const float* w, uint32_t n_embd,
                     uint32_t n_hc, uint32_t nb, uint32_t hc_stride, float eps, void* stream);

}  // namespace aff
