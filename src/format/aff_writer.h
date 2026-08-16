// affinity-engine — .aff container writer
//
// Layout is computed entirely up front from the model config, then the file is sized with
// ftruncate() and filled with random-access pwrite(). That is what allows shard-at-a-time
// quantisation: safetensors shards deliver tensors and experts in arbitrary order, and each one
// is written straight to its final offset with no staging, no rewrite, and no second pass.

#pragma once

#include "aff_format.h"

#include <string>
#include <vector>
#include <unordered_map>

namespace aff {

struct TensorDecl {
  std::string name;
  uint64_t    rows = 0;
  uint64_t    cols = 0;
  AffQuant    quant = AFF_F16;
};

struct LayerDecl {
  uint32_t       n_experts = 0;
  ExpertGeometry geom{};
};

// One matrix's payload. Any pointer may be null when the corresponding size is 0.
struct MatrixData {
  const void* data    = nullptr;  uint64_t data_size   = 0;  // quants
  const void* scales  = nullptr;  uint64_t scale_size  = 0;  // sub-block scales
  const void* rscales = nullptr;  uint64_t rscale_size = 0;  // per-row scales
};

// One expert's payload.
struct ExpertData {
  MatrixData gate, up, down;
};

class AffWriter {
public:
  AffWriter() = default;
  ~AffWriter();
  AffWriter(const AffWriter&) = delete;
  AffWriter& operator=(const AffWriter&) = delete;

  // Computes the full layout, sizes the file, and writes the directories.
  // `meta_json` carries model config / tokenizer reference / quant recipe.
  bool begin(const std::string& path,
             const std::string& meta_json,
             const std::vector<LayerDecl>& layers,
             const std::vector<TensorDecl>& tensors,
             std::string* err);

  // Non-expert tensor. Sizes must match what was declared in begin().
  bool write_tensor(const std::string& name, const MatrixData& m, std::string* err);

  // One expert, addressed by (layer, id). Safe to call in any order, from any thread
  // (pwrite to disjoint ranges), provided each (layer,id) is written at most once.
  bool write_expert(uint32_t layer, uint32_t expert, const ExpertData& d, std::string* err);

  // Optional expert-popularity profile: `layer_count * n_experts` floats, layer-major.
  // Consumed by the residency manager for static placement.
  bool write_profile(const float* popularity, uint64_t count, std::string* err);

  // Flushes directories + header. Must be called; the file is invalid without it.
  bool finalize(std::string* err);

  uint64_t file_size() const { return file_size_; }
  // Number of experts still unwritten — a completeness check for the quantiser.
  uint64_t missing_experts() const;

private:
  int      fd_ = -1;
  std::string path_;
  std::string meta_;
  AffHeader   hdr_{};

  std::vector<AffLayerDesc>   layers_;
  std::vector<AffTensorEntry> tensors_;
  std::unordered_map<std::string, size_t> tensor_index_;

  std::vector<bool> expert_written_;   // flat (layer, expert) presence bitmap
  std::vector<uint64_t> layer_expert_base_; // index into expert_written_ per layer
  std::vector<bool> tensor_written_;

  uint64_t file_size_ = 0;
  bool     finalized_ = false;

  bool pwrite_all(const void* buf, uint64_t n, uint64_t off, std::string* err);
};

} // namespace aff
