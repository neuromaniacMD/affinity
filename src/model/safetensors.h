// Minimal safetensors reader.
//
// Format: [8-byte LE header length][JSON header][raw tensor data]
// The JSON maps tensor name -> {dtype, shape, data_offsets:[begin,end]}, where offsets are
// relative to the end of the header. There is also a "__metadata__" key which is not a tensor.
//
// We mmap the shard and hand out pointers — no copying. A 46-shard model is opened one shard at
// a time by the quantiser, so peak RSS stays bounded regardless of model size.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace aff {

enum class StDType {
  Unknown,
  BF16, F16, F32, F64,
  I8, U8, I16, I32, I64,
  F8_E4M3, F8_E5M2, F8_E8M0,
  BOOL,
};

StDType     st_dtype_from_string(std::string_view s) noexcept;
const char* st_dtype_name(StDType d) noexcept;
uint32_t    st_dtype_size(StDType d) noexcept;   // bytes per element, 0 if unknown

struct StTensor {
  std::string          name;
  StDType              dtype = StDType::Unknown;
  std::vector<int64_t> shape;
  uint64_t             offset = 0;   // relative to data section start
  uint64_t             nbytes = 0;
  const uint8_t*       data   = nullptr;

  uint64_t numel() const {
    uint64_t n = 1;
    for (int64_t d : shape) n *= static_cast<uint64_t>(d);
    return shape.empty() ? 0 : n;
  }
};

class SafeTensorsFile {
public:
  SafeTensorsFile() = default;
  ~SafeTensorsFile();
  SafeTensorsFile(const SafeTensorsFile&) = delete;
  SafeTensorsFile& operator=(const SafeTensorsFile&) = delete;

  bool open(const std::string& path, std::string* err);
  void close();

  const StTensor* find(const std::string& name) const;
  const std::vector<StTensor>& tensors() const { return tensors_; }
  const std::string& metadata() const { return metadata_; }
  const std::string& path() const { return path_; }

private:
  int         fd_ = -1;
  uint8_t*    map_ = nullptr;
  uint64_t    map_size_ = 0;
  std::string path_;
  std::string metadata_;
  std::vector<StTensor> tensors_;
  std::unordered_map<std::string, size_t> index_;
};

// Reads model.safetensors.index.json -> {tensor name: shard filename}.
bool load_safetensors_index(const std::string& index_path,
                            std::unordered_map<std::string, std::string>* weight_map,
                            std::string* err);

} // namespace aff
