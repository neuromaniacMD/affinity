// Sampling on the device, for the speculative verify block.
//
// WHY IT IS HERE AND NOT ON THE HOST. The block head leaves a [vocab][nb] plane in VRAM, and the
// vocabulary is Row-split, so a host sampler would need every card's band brought back — and then a
// softmax and a nucleus cut over the whole vocabulary per position, on the dispatch thread, in the
// middle of the decode loop. Sampling where the logits already are costs one gather of the other
// cards' bands and nothing else.
//
// WHY IT RETURNS THREE THINGS. Greedy speculation accepts a drafted token when it equals the
// target's argmax. That rule is only correct at temperature 0. With the target sampling, the correct
// rule is rejection sampling — and with a greedy draft (`draft_sample_method: greedy`, which is what
// the model card's vLLM line specifies) the draft's distribution is a point mass, so a proposal is
// accepted with probability p(x) under the target's own filtered distribution. On rejection the
// emitted token must come from the residual, which for a point-mass proposal is the same
// distribution with the proposed token removed and the rest renormalised.
//
// So one pass produces everything a position can need: the accept probability, a draw for the
// bonus/all-accepted case, and a draw for the rejection case. The host then walks the positions with
// its own uniforms and never has to come back.
//
// Consequence worth stating plainly: acceptance FALLS relative to greedy-against-greedy, because a
// proposal the target agrees with is still only accepted with probability p(x) rather than always.
// That is the cost of emitting the target's true distribution, not a regression.

#pragma once

#include <cstdint>

namespace aff {

// The model card's recommendation is the default: temperature 1.0, top_p 0.95 for agentic work
// (1.0 otherwise). top_k and min_p are off, as `generation_config.json` leaves them.
struct SampleParams {
  float    temperature = 1.0f;
  float    top_p       = 0.95f;
  uint32_t top_k       = 0;      // 0 disables
  float    min_p       = 0.0f;   // keep p >= min_p * p_max
};

struct SpecDraw {
  uint32_t tok;       // a draw from the filtered distribution
  uint32_t tok_excl;  // a draw with `query` removed and the remainder renormalised
  float    p_query;   // filtered probability of `query` — the accept probability
};

// One position a block. `logits` is the head's plane; `row_stride`/`col_stride` describe it exactly
// as batch_argmax_hip's do, so either layout the head produces is readable without a transpose.
//
// `d_query[i]` is the token the draft proposed for position i, or 0xFFFFFFFF where there is none
// (the last position of a block, which has no proposal to accept). `d_u` holds two uniforms a
// position, [0,1), used for `tok` and `tok_excl`; the accept draw stays on the host, which is where
// the seed lives.
//
// `vocab` is the FULL vocabulary — the caller gathers the other bands first, because a nucleus over
// half a distribution is not a nucleus.
bool batch_sample_hip(const float* logits, uint32_t vocab, uint32_t n,
                      uint32_t row_stride, uint32_t col_stride,
                      const uint32_t* d_query, const float* d_u, SampleParams p,
                      SpecDraw* d_out, void* stream);

} // namespace aff
