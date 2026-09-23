// Engram: n-gram hash lookups written into the residual stream between layers.
//
// DeepSeek-V4.1 adds, at layers 1 and 14, a lookup keyed on the n-grams ending at each position. The
// tables are the two largest tensors in the checkpoint by an order of magnitude -- 384 million rows
// of 256 FP8 each, ~91.6 GiB apiece -- and they stay in the checkpoint: the container carries only
// `engram_q`, `engram_k` and `engram_wkv`, and the engine mmaps the rest. See `EngramTables`.
//
// ---- why the constants come from a file ----------------------------------------------------------
//
// The hash must match `engram.py` exactly. A mismatch does not fail: it reads a different row of a
// 384-million-row table, so the model keeps running and adds noise to its own residual stream. Two
// of the three inputs are unreasonable to reproduce here --
//
//   - the compressed token map, which runs the HF `tokenizers` Rust normalizer stack (NFKC, NFD,
//     strip accents, lowercase, whitespace regexes) over all 129,280 ids so that " The", "the" and
//     "THE" hash alike;
//   - the multipliers, which are numpy PCG64 `default_rng(10007 * layer_id).integers(...)`, i.e.
//     Lemire rejection over that exact bit stream;
//
// -- and both are deterministic and small once evaluated. `tools/v41/engram_prep.py` imports the
// reference's own functions and writes them down; this reads that file. The prep tool asserts the
// compressed vocab against config.json, because every multiplier is derived from it.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace aff {

// The layout and constants of the n-gram hash, as `tools/v41/engram_prep.py` emits them.
struct EngramConsts {
  static constexpr int32_t kDead = -1;   // a position that may not take part in an n-gram

  uint32_t n_layer = 0;                  // engram layers (2)
  uint32_t max_ngram = 0;                // 4: the 2-, 3- and 4-grams ending at a position
  uint32_t n_heads = 0;                  // 8
  uint32_t head_dim = 0;                 // 256, the table's row width
  uint32_t n_cols = 0;                   // (max_ngram-1) * n_heads = 24 lookups a position a layer
  uint32_t vocab = 0;                    // tokenizer vocab, the token_map length
  uint32_t cvocab = 0;                   // compressed vocab; every multiplier derives from it
  uint32_t pad = 0;                      // the pad id ALREADY mapped through token_map
  std::vector<uint64_t> rows;            // [n_layer]                 table rows
  std::vector<int64_t>  mult;            // [n_layer][max_ngram]
  std::vector<int64_t>  primes;          // [n_layer][max_ngram-1][n_heads]   bucket modulus
  std::vector<int64_t>  offsets;         // [n_layer][n_cols]         where each bucket range starts
  std::vector<int32_t>  token_map;       // [vocab]
  // Where each layer's table actually lives, so the engine mmaps two byte ranges instead of
  // parsing safetensors for the sake of two numbers.
  struct Table { std::string shard; uint64_t w_off = 0, s_off = 0, w_bytes = 0, s_bytes = 0; };
  std::vector<Table> tables;             // [n_layer]

  bool load(const std::string& path, std::string* err);
  // Which engram layers these are, in the order their hash columns are laid out. Read from the
  // model config rather than the file, which carries only the count.
  int64_t prime(uint32_t l, uint32_t ng, uint32_t h) const {
    return primes[((size_t)l * (max_ngram - 1u) + ng) * n_heads + h];
  }
};

// The rolling per-position state: which compressed id each position carries, so that a decode step
// can look back into what prefill wrote. `NgramHashState` in the reference, and the same cache.
//
// Look-back stops at the start of the sequence and at any DEAD position (an image span), so an
// n-gram never spans one; a blocked slot takes the pad id, and once blocked every longer look-back
// stays blocked.
class EngramHash {
public:
  // `hint` sizes the look-back cache initially; it grows on demand, so it is not a bound.
  void init(const EngramConsts* c, uint64_t hint);
  void reset() { filled_ = 0; }

  // Appends `n` token ids starting at absolute position `pos0` and writes their hash ids.
  // `out` is [n][n_layer][n_cols], row-major, and must hold n * n_layer * n_cols int64.
  // `alive` is optional, one byte a token, 0 marking a position that takes no part in an n-gram.
  void push(const uint32_t* ids, uint32_t n, uint64_t pos0, int64_t* out,
            const uint8_t* alive = nullptr);

  bool ready() const { return c_ != nullptr; }

  // Writes the compressed ids of `n` tokens starting at absolute position `pos0` into the look-back
  // WITHOUT hashing them. For a sequence that resumes mid-stream from a checkpoint: the suffix
  // prefill hashes n-grams that reach up to max_ngram-1 positions back into the restored prefix,
  // and those slots otherwise hold whatever request last ran here.
  void seed(const uint32_t* ids, uint32_t n, uint64_t pos0);
  uint32_t lookback() const { return c_ && c_->max_ngram ? c_->max_ngram - 1u : 0u; }

private:
  void grow(uint64_t end);
  const EngramConsts* c_ = nullptr;
  std::vector<int32_t> cache_;           // compressed id per position, kDead where blocked
  uint64_t filled_ = 0;
};

// The tables themselves, mmapped out of the checkpoint shards.
//
// 91.6 GiB a layer, and the two of them together are larger than this box's RAM — so they are never
// read whole. A position touches `n_cols` rows of `head_dim` FP8 plus their E8M0 scales, which is
// 24 x 256 B of table and 24 x 8 B of scale: demand paging reads the pages under those rows and
// nothing else. That is the whole reason the tables stay in the checkpoint instead of being folded
// into the container, where the expert pool would have to make room for them.
class EngramTables {
public:
  ~EngramTables();
  // `dir` is the checkpoint directory the shards sit in. Maps every layer's two ranges.
  bool open(const EngramConsts* c, const std::string& dir, std::string* err);
  bool ready() const { return c_ != nullptr; }

  // The dequantised lookup for ONE position: `n_cols` rows of `head_dim`, concatenated in column
  // order, which is the flattened activation `engram_wkv` consumes.
  // `out` holds n_cols * head_dim floats. Row ids outside the table come back as zeros, which is
  // what the reference's own bounds mask produces.
  void gather(uint32_t layer, const int64_t* ids, float* out) const;

  uint64_t bytes() const { return bytes_; }

private:
  const EngramConsts* c_ = nullptr;
  struct Map { const uint8_t* w = nullptr; const uint8_t* s = nullptr; void* wbase = nullptr;
               void* sbase = nullptr; size_t wlen = 0, slen = 0; uint64_t rows = 0; };
  std::vector<Map> m_;
  uint64_t bytes_ = 0;
};

}  // namespace aff
