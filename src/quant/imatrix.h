// GGUF-format importance-matrix reader.
//
// An imatrix records, per weight tensor, the mean squared activation of each INPUT column:
//     imp[j] = E[ x_j^2 ]
// which turns the quantiser's objective from plain L2 into the weighted form
//     minimise  sum_i sum_j imp[j] * (W_ij - W'_ij)^2
// i.e. spend precision on the columns the model actually drives hard. This is what makes
// sub-3-bit quantisation usable in practice.
//
// Layout (llama.cpp GGUF imatrix): two tensors per weight,
//     <name>.in_sum2   f32, [cols]           or [cols, n_expert] for MoE tensors
//     <name>.counts    f32, [1]              or [1, n_expert]
// with mean importance = in_sum2 / counts.
//
// For DeepSeek-V4-Flash the MoE entries are PER EXPERT, e.g.
//     blk.7.ffn_gate_exps.weight.in_sum2   [4096, 256]
//     blk.7.ffn_gate_exps.weight.counts    [1, 256]
// so we get an independent importance vector for every (layer, expert) — and the counts double
// as a direct measurement of expert activation frequency, which is what the placement engine's
// starting profile is built from.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace aff {

struct ImatrixEntry {
  std::string    name;      // without the .in_sum2 / .counts suffix
  const float*   sum2 = nullptr;
  const float*   counts = nullptr;
  uint64_t       cols = 0;      // importance values per expert
  uint64_t       n_expert = 1;  // 1 for dense tensors
};

class ImatrixFile {
public:
  ImatrixFile() = default;
  ~ImatrixFile();
  ImatrixFile(const ImatrixFile&) = delete;
  ImatrixFile& operator=(const ImatrixFile&) = delete;

  bool open(const std::string& path, std::string* err);
  void close();

  const ImatrixEntry* find(const std::string& name) const;
  const std::vector<ImatrixEntry>& entries() const { return entries_; }
  const std::string& dataset() const { return dataset_; }
  uint32_t chunk_count() const { return chunk_count_; }
  uint32_t chunk_size() const { return chunk_size_; }

  // Mean importance for one expert, normalised by its call count and rescaled to mean 1.0 so
  // that weighted and unweighted error are directly comparable. Returns false if absent.
  bool importance(const std::string& name, uint64_t expert, std::vector<float>* out) const;

  // Raw activation count for one expert — the expert-popularity signal.
  double count(const std::string& name, uint64_t expert) const;

  // --- name mapping: DeepSeek safetensors -> GGUF imatrix -------------------------------------
  // layers.L.ffn.experts.E.w1  -> blk.L.ffn_gate_exps.weight   (gate)
  // layers.L.ffn.experts.E.w3  -> blk.L.ffn_up_exps.weight     (up)
  // layers.L.ffn.experts.E.w2  -> blk.L.ffn_down_exps.weight   (down)
  // Returns "" if the tensor has no imatrix counterpart.
  static std::string gguf_name_for_expert(uint32_t layer, const char* which /* "w1","w2","w3" */);

private:
  int       fd_ = -1;
  uint8_t*  map_ = nullptr;
  uint64_t  map_size_ = 0;
  std::string dataset_;
  uint32_t  chunk_count_ = 0, chunk_size_ = 0;
  std::vector<ImatrixEntry> entries_;
  std::unordered_map<std::string, size_t> index_;
};

} // namespace aff
