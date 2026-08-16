// Byte-level BPE tokenizer, loading a HuggingFace tokenizer.json.
//
// DeepSeek-V4 uses byte-level BPE with a GPT-2-style byte<->unicode mapping. Merges are applied
// by rank; ties are broken by leftmost position.
//
// Pre-tokenisation matters as much as the merges. DeepSeek-V4-Flash ships a three-stage Sequence:
// isolate runs of 1-3 digits, isolate CJK runs, then split on a GPT-4-style pattern. Feeding the
// merge table whitespace-delimited words instead produces ids the model never saw in training —
// the text still round-trips, so the damage is invisible until generation quality is measured.
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace aff {

class Tokenizer {
public:
  bool load(const std::string& tokenizer_json_path, std::string* err);
  std::vector<uint32_t> encode(const std::string& text, bool add_bos = false) const;
  std::string decode(const std::vector<uint32_t>& ids) const;
  std::string decode_one(uint32_t id) const;
  uint32_t vocab_size() const { return (uint32_t)id_to_tok_.size(); }
  int32_t bos() const { return bos_; }
  int32_t eos() const { return eos_; }

private:
  std::unordered_map<std::string, uint32_t> tok_to_id_;
  std::vector<std::string> id_to_tok_;
  std::unordered_map<std::string, uint32_t> merge_rank_;   // "a b" -> rank
  std::unordered_map<uint32_t, int> special_;
  int32_t bos_ = -1, eos_ = -1;
  std::vector<std::string> byte_to_unicode_;               // 256 entries
  std::unordered_map<std::string, uint8_t> unicode_to_byte_;
  void build_byte_maps();
  std::vector<std::string> bpe(const std::string& piece) const;
  std::vector<std::string> pretokenize(const std::string& text) const;
  // added_tokens, longest-match-first so multi-byte markers like <｜User｜> map to their own id
  // instead of being byte-BPE'd into a dozen pieces.
  std::vector<std::pair<std::string, uint32_t>> added_;
};

} // namespace aff
