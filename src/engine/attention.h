// DeepSeek-V4-Flash attention.
//
// Corrected against antirez/ds4's working implementation. Two things the config and tech report
// make look more complicated than they are:
//
//  * CSA and HCA are not two algorithms. There is ONE attention kernel operating over
//        raw_SWA(128)  ++  compressed_rows
//    and layers differ only in compression ratio: 4 on even layers (with a lightning indexer +
//    top-k over the compressed rows), 128 on odd layers (dense over a very short compressed
//    cache), and layers 0–1 uncompressed.
//
//  * Shared-KV MQA, not MLA. `num_key_value_heads = 1`, and there is no `kv_b_proj`: V3's
//    absorbed-weight trick ships baked into the checkpoint. There is exactly ONE 512-wide latent
//    per position (448 unrotated + 64 rotated), shared by all 64 query heads. Per-head K/V are
//    never materialised.
//
//  * The latent is also the value. V is the FULL 512, rope tail included — not the leading
//    448. The rotation is undone afterwards by re-applying RoPE to the attention output with the
//    sine negated, before the grouped output projection. Taking only the unrotated part instead
//    drops 64 of 512 value dimensions and silently degrades every layer.
//
// Attention sinks are per-head scalars that SEED the online softmax rather than acting as an
// extra key: m0 = sink, l0 = 1.0. That lets the attention weights sum to less than one, which is
// the entire point — a "nowhere to look" escape valve. Getting this wrong (adding the sink as a
// pseudo-key with a value vector) silently changes the output.
//
// RoPE is PARTIAL: only the last `rope_dim` of the 576 are rotated; the leading 512 are untouched.

#pragma once

#include <cstdint>
#include <vector>

namespace aff {

struct AttnConfig {
  uint32_t n_head    = 64;     // query heads
  uint32_t head_dim  = 512;    // "nope" part, shared KV latent width
  uint32_t rope_dim  = 64;     // rotated tail; total per-head query width = head_dim + rope_dim
  uint32_t sliding   = 128;    // raw (uncompressed) recent window
  float    rope_theta = 10000.0f;
  float    scale      = 0.0f;  // 0 => 1/sqrt(head_dim + rope_dim)

  uint32_t qk_width() const { return head_dim + rope_dim; }
  float    eff_scale() const;
};

// One position's shared KV latent: [head_dim nope | rope_dim rope], contiguous.
// A whole cache is `n_pos * qk_width()` floats — one latent per position, NOT per head.
struct KvCache {
  std::vector<float> latent;
  uint32_t width = 0;
  uint64_t n_pos = 0;
  const float* at(uint64_t p) const { return latent.data() + p * width; }
  float*       at(uint64_t p)       { return latent.data() + p * width; }
};

// In-place partial RoPE over the trailing `rope_dim` of one `qk_width()`-wide vector.
void rope_partial(float* v, const AttnConfig& c, uint64_t pos) noexcept;

// Online-softmax attention for one query position.
//
//   q     : n_head * qk_width()   (already RoPE'd)
//   kv    : the shared latent cache
//   sinks : n_head per-head sink logits, or null for none
//   out   : n_head * qk_width()   (still rotated; the caller applies the inverse RoPE)
//
// `keys` lists the cache positions to attend over — the concatenation of the raw sliding window
// and whichever compressed rows the indexer selected. The kernel does not care which is which,
// which is exactly why CSA and HCA share one implementation.
void attention_mqa(const AttnConfig& c, const float* q, const KvCache& kv,
                   const uint64_t* keys, uint64_t n_keys,
                   const float* sinks, float* out) noexcept;

// Reference: materialise the full softmax then weight. Numerically straightforward, O(n_keys)
// memory, used to validate the online path (which must match it to float rounding).
void attention_mqa_reference(const AttnConfig& c, const float* q, const KvCache& kv,
                             const uint64_t* keys, uint64_t n_keys,
                             const float* sinks, float* out) noexcept;

// Builds the key list for a layer: the last `sliding` raw positions, then the selected
// compressed rows. `ratio` is 0 (uncompressed), 4 (CSA) or 128 (HCA).
void build_key_list(const AttnConfig& c, uint64_t cur_pos, uint32_t ratio,
                    const uint64_t* selected, uint64_t n_selected,
                    std::vector<uint64_t>* keys);

} // namespace aff
