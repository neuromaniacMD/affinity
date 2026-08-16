// affinity-engine — .aff container reader.
//
// One MAP_SHARED mapping of the whole file, and O(1) arithmetic addressing into the expert pool.

#pragma once

#include "aff_format.h"

#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>

namespace aff {

// Resolved pointers into one expert's payload.
struct MatrixView {
  const uint8_t* data    = nullptr;
  const uint8_t* scales  = nullptr;   // sub-block scales
  const uint8_t* rscales = nullptr;   // per-row scales
};

struct ExpertView {
  MatrixView gate, up, down;
  uint64_t   bytes = 0;               // stride: the contiguous span of this expert
};

class AffReader {
public:
  AffReader() = default;
  ~AffReader();
  AffReader(const AffReader&) = delete;
  AffReader& operator=(const AffReader&) = delete;

  bool open(const std::string& path, std::string* err);
  void close();

  const AffHeader& header() const { return *hdr_; }
  std::string_view meta() const { return meta_; }

  uint64_t layer_count() const { return hdr_ ? hdr_->layer_count : 0; }
  const AffLayerDesc& layer(uint32_t i) const { return layers_[i]; }

  // O(1) — pure arithmetic, no table walk. This is on the residency manager's hot path.
  const uint8_t* expert_ptr(uint32_t layer, uint32_t expert) const {
    const AffLayerDesc& L = layers_[layer];
    return expert_pool_ + L.base_off + static_cast<uint64_t>(expert) * L.stride;
  }
  ExpertView expert(uint32_t layer, uint32_t expert) const;

  const AffTensorEntry* find_tensor(const std::string& name) const;

  // The tensor directory as an array. `find_tensor` answers "is this one here"; this answers
  // "what is ALL of it", which is the question anything summing bytes over the container has —
  // ask the container rather than hardcoding a per-token total that goes stale when a weight
  // changes format.
  uint64_t tensor_count() const { return hdr_ ? hdr_->tensor_count : 0; }
  const AffTensorEntry& tensor_at(uint64_t i) const {
    return reinterpret_cast<const AffTensorEntry*>(base_ + hdr_->tensor_dir_off)[i];
  }
  const uint8_t* tensor_data(const AffTensorEntry& e) const {
    return base_ + hdr_->tensor_data_off + e.blob_off + e.desc.data_off;
  }
  const uint8_t* tensor_scales(const AffTensorEntry& e) const {
    return e.desc.scale_size ? base_ + hdr_->tensor_data_off + e.blob_off + e.desc.scale_off
                             : nullptr;
  }
  const uint8_t* tensor_rscales(const AffTensorEntry& e) const {
    return e.desc.rscale_size ? base_ + hdr_->tensor_data_off + e.blob_off + e.desc.rscale_off
                              : nullptr;
  }

  // Expert-popularity profile, or null when absent. layer-major, layer_count * n_experts floats.
  //
  // ---- ONE ACCESSOR, AND THE OVERRIDE GOES INSIDE IT ---------------------------------------------
  //
  // Two places rank experts by this — the VRAM slab (`static_placement.hip`) and the host pool's
  // fill order (`host_pool.cpp`) — and they must agree, because the pool holds the COMPLEMENT of
  // what the slab took. Ranked by two different profiles they would not partition the expert set;
  // some experts would be in both and some in neither, and the second kind is an addressless expert,
  // which costs its whole layer the device-built dispatch. So `--expert-profile` is resolved here
  // rather than at either call site, and there is deliberately no way to ask for the embedded one
  // once an override is set.
  //
  // The override exists because the shipped profile is derived from the IMATRIX — the calibration
  // counts `aff-quantize` already had — and imatrix popularity is a poor predictor of decode
  // routing, ranking barely better than a balanced split where an oracle is far ahead of both.
  // `aff-profile` fits one from real `--route-trace` records instead. It stays a side file and a
  // flag rather than being written into the container so both rankings are reachable from one
  // binary; a profile that proves out belongs in the container.
  const float* profile() const {
    if (prof_override_) return prof_override_;
    return (hdr_->flags & AFF_FLAG_HAS_PROFILE)
             ? reinterpret_cast<const float*>(base_ + hdr_->profile_off) : nullptr;
  }
  // `count` must be layer_count * n_experts; a mismatch is rejected rather than truncated, because a
  // short profile ranks the tail it does not cover as zero and that is a silent placement change.
  bool set_profile_override(const float* p, uint64_t count, std::string* err);

  // Page-cache hints for the expert pool.
  //
  // These matter more than they look. Uploading the resident set to VRAM READS it through the page
  // cache, so the cache ends up holding a duplicate of exactly what the GPUs already have — and
  // evicts the non-resident experts, which are the only ones the CPU fallback path ever touches.
  // Every fallback expert then becomes a disk read with the CPU idle waiting for it. The
  // non-resident remainder is what should be cached, and it is the part that fits.
  void advise_expert(uint32_t layer, uint32_t expert, bool want) const;

  // ---- give the container's page cache back once nothing needs it ------------------------------
  //
  // The container is one MAP_SHARED mapping. At load, every byte of it is read once — into VRAM for
  // the resident experts and into the registered host pool for the rest — and after that nothing
  // reads it again unless an expert is on the SSD tier. The page cache it leaves behind is dead
  // weight the size of the container.
  //
  // It is not tidiness. With the pool sized to the machine the engine decodes with almost nothing
  // free and gigabytes of page cache, so every host allocation and every first touch goes through
  // direct reclaim. The visible cost is not a slowdown but VARIANCE: the same prefill on the same
  // binary can differ by an order of magnitude, and in the slow runs the time is host-side rather
  // than in the stream drains.
  //
  // Returns how many bytes the kernel was asked to drop. POSIX_FADV_DONTNEED rather than madvise:
  // this has to drop the page cache, not merely unmap the pages from this process, and the shared
  // mapping would leave the duplicate in place. Correct-but-slower if something does read the
  // container afterwards — it faults back in from disk, which is exactly what the SSD tier is.
  uint64_t release_cache() const;

  // Byte offset of an expert IN THE FILE, and its size. An expert is stored contiguously, so a
  // loader can pread the whole thing in one request rather than walking the mapping.
  uint64_t expert_file_offset(uint32_t layer, uint32_t expert) const {
    const AffLayerDesc& L = layers_[layer];
    return hdr_->expert_pool_off + L.base_off + (uint64_t)expert * L.stride;
  }
  int fd() const { return fd_; }

  uint64_t expert_pool_size() const { return hdr_ ? hdr_->expert_pool_size : 0; }

private:
  int            fd_    = -1;
  uint8_t*       map_   = nullptr;   // whole-file mapping
  uint64_t       map_size_ = 0;
  const uint8_t* base_  = nullptr;   // == map_
  const AffHeader* hdr_ = nullptr;

  const AffLayerDesc* layers_ = nullptr;
  std::string_view    meta_;
  std::unordered_map<std::string, const AffTensorEntry*> tensor_index_;

  const uint8_t* expert_pool_ = nullptr;   // into map_

  // Owned, because the caller's buffer is a local in main() and placement is ranked long after it
  // would have gone out of scope. Null unless --expert-profile was given.
  std::vector<float> prof_own_;
  const float*       prof_override_ = nullptr;
};

} // namespace aff
