#include "aff_writer.h"

#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>

namespace aff {

namespace {
void set_err(std::string* err, const std::string& msg) { if (err) *err = msg; }
std::string errno_str(const char* what) {
  return std::string(what) + ": " + std::strerror(errno);
}
} // namespace

AffWriter::~AffWriter() {
  if (fd_ >= 0) ::close(fd_);
}

bool AffWriter::pwrite_all(const void* buf, uint64_t n, uint64_t off, std::string* err) {
  const uint8_t* p = static_cast<const uint8_t*>(buf);
  while (n > 0) {
    ssize_t w = ::pwrite(fd_, p, n, static_cast<off_t>(off));
    if (w < 0) {
      if (errno == EINTR) continue;
      set_err(err, errno_str("pwrite"));
      return false;
    }
    if (w == 0) { set_err(err, "pwrite returned 0"); return false; }
    p += w; n -= static_cast<uint64_t>(w); off += static_cast<uint64_t>(w);
  }
  return true;
}

bool AffWriter::begin(const std::string& path,
                      const std::string& meta_json,
                      const std::vector<LayerDecl>& layers,
                      const std::vector<TensorDecl>& tensors,
                      std::string* err) {
  if (fd_ >= 0) { set_err(err, "begin() called twice"); return false; }
  if (layers.empty()) { set_err(err, "no layers declared"); return false; }

  path_ = path;
  meta_ = meta_json;

  // ---- compute the full layout up front -----------------------------------------------------
  uint64_t off = sizeof(AffHeader);

  const uint64_t meta_off = off;
  off = align_up(off + meta_.size(), 8);

  const uint64_t tensor_dir_off = off;
  off += static_cast<uint64_t>(tensors.size()) * sizeof(AffTensorEntry);

  const uint64_t layer_dir_off = off;
  off += static_cast<uint64_t>(layers.size()) * sizeof(AffLayerDesc);

  // Profile region is reserved unconditionally; the flag records whether it was filled.
  uint64_t total_experts = 0;
  for (const auto& L : layers) total_experts += L.n_experts;
  const uint64_t profile_off  = off;
  const uint64_t profile_size = total_experts * sizeof(float);
  off = align_up(off + profile_size, kBlockAlign);

  // ---- non-expert tensor blobs --------------------------------------------------------------
  const uint64_t tensor_data_off = off;
  tensors_.resize(tensors.size());
  uint64_t blob = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    const TensorDecl& t = tensors[i];
    if (t.name.size() >= kMaxTensorName) {
      set_err(err, "tensor name too long: " + t.name);
      return false;
    }
    if (!quant_info(t.quant)) {
      set_err(err, "unknown quant for tensor " + t.name);
      return false;
    }
    AffTensorEntry& e = tensors_[i];
    std::memset(&e, 0, sizeof(e));
    std::memcpy(e.name, t.name.data(), t.name.size());

    const uint64_t n = t.rows * t.cols;
    e.desc.rows        = t.rows;
    e.desc.cols        = t.cols;
    e.desc.quant       = static_cast<uint32_t>(t.quant);
    e.desc.data_size   = quant_data_bytes(t.quant, n);
    e.desc.scale_size  = quant_scale_bytes(t.quant, n);
    e.desc.rscale_size = quant_row_scale_bytes(t.quant, t.rows);
    e.desc.data_off    = 0;
    e.desc.scale_off   = align_up(e.desc.data_size, kBlockAlign);
    e.desc.rscale_off  = align_up(e.desc.scale_off + e.desc.scale_size, kBlockAlign);
    e.blob_off         = blob;

    blob = align_up(blob + e.desc.rscale_off + e.desc.rscale_size, kBlockAlign);
    tensor_index_[t.name] = i;
  }
  const uint64_t tensor_data_size = blob;
  off += tensor_data_size;

  // ---- expert pool, 2 MiB-aligned so it can be mapped with huge pages ------------------------
  off = align_up(off, kPoolAlign);
  const uint64_t expert_pool_off = off;

  layers_.resize(layers.size());
  layer_expert_base_.resize(layers.size());
  uint64_t pool = 0;
  uint64_t expert_slot = 0;
  for (size_t i = 0; i < layers.size(); ++i) {
    const LayerDecl& L = layers[i];
    if (L.n_experts == 0) { set_err(err, "layer with zero experts"); return false; }
    AffLayerDesc& d = layers_[i];
    std::memset(&d, 0, sizeof(d));
    d.layer     = static_cast<uint32_t>(i);
    d.n_experts = L.n_experts;
    d.stride    = compute_expert_layout(L.geom, d.gate, d.up, d.down);
    if (d.stride == 0) { set_err(err, "bad expert geometry in layer " + std::to_string(i)); return false; }
    d.base_off  = pool;
    pool += static_cast<uint64_t>(L.n_experts) * d.stride;

    layer_expert_base_[i] = expert_slot;
    expert_slot += L.n_experts;
  }
  const uint64_t expert_pool_size = pool;
  off = expert_pool_off + expert_pool_size;

  file_size_ = off;

  // ---- header -------------------------------------------------------------------------------
  std::memset(&hdr_, 0, sizeof(hdr_));
  std::memcpy(hdr_.magic, kMagic, sizeof(kMagic));
  hdr_.version          = kVersion;
  hdr_.flags            = AFF_FLAG_NONE;
  hdr_.meta_off         = meta_off;
  hdr_.meta_size        = meta_.size();
  hdr_.tensor_dir_off   = tensor_dir_off;
  hdr_.tensor_count     = tensors.size();
  hdr_.tensor_data_off  = tensor_data_off;
  hdr_.tensor_data_size = tensor_data_size;
  hdr_.layer_dir_off    = layer_dir_off;
  hdr_.layer_count      = layers.size();
  hdr_.expert_pool_off  = expert_pool_off;
  hdr_.expert_pool_size = expert_pool_size;
  hdr_.profile_off      = profile_off;
  hdr_.profile_size     = profile_size;
  hdr_.file_size        = file_size_;

  expert_written_.assign(total_experts, false);
  tensor_written_.assign(tensors.size(), false);

  // ---- create and size the file -------------------------------------------------------------
  fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (fd_ < 0) { set_err(err, errno_str("open")); return false; }
  if (::ftruncate(fd_, static_cast<off_t>(file_size_)) != 0) {
    set_err(err, errno_str("ftruncate"));
    return false;
  }
  // Header is rewritten at finalize(); write a placeholder now so a truncated file is detectable.
  return pwrite_all(&hdr_, sizeof(hdr_), 0, err);
}

bool AffWriter::write_tensor(const std::string& name, const MatrixData& m, std::string* err) {
  if (fd_ < 0) { set_err(err, "writer not open"); return false; }
  auto it = tensor_index_.find(name);
  if (it == tensor_index_.end()) { set_err(err, "undeclared tensor: " + name); return false; }
  AffTensorEntry& e = tensors_[it->second];

  auto mismatch = [&](const char* what, uint64_t got, uint64_t want) {
    set_err(err, "tensor " + name + ": " + what + " size " + std::to_string(got) +
                 " != declared " + std::to_string(want));
    return false;
  };
  if (m.data_size   != e.desc.data_size)   return mismatch("data",      m.data_size,   e.desc.data_size);
  if (m.scale_size  != e.desc.scale_size)  return mismatch("scale",     m.scale_size,  e.desc.scale_size);
  if (m.rscale_size != e.desc.rscale_size) return mismatch("row-scale", m.rscale_size, e.desc.rscale_size);

  const uint64_t base = hdr_.tensor_data_off + e.blob_off;
  struct Part { const void* p; uint64_t off, size; };
  const Part parts[3] = {
    { m.data,    e.desc.data_off,   m.data_size   },
    { m.scales,  e.desc.scale_off,  m.scale_size  },
    { m.rscales, e.desc.rscale_off, m.rscale_size },
  };
  for (const Part& part : parts) {
    if (part.size == 0) continue;
    if (!part.p) { set_err(err, "tensor " + name + ": missing payload section"); return false; }
    if (!pwrite_all(part.p, part.size, base + part.off, err)) return false;
  }

  tensor_written_[it->second] = true;
  return true;
}

bool AffWriter::write_expert(uint32_t layer, uint32_t expert,
                             const ExpertData& d, std::string* err) {
  if (fd_ < 0) { set_err(err, "writer not open"); return false; }
  if (layer >= layers_.size()) { set_err(err, "layer out of range"); return false; }
  const AffLayerDesc& L = layers_[layer];
  if (expert >= L.n_experts) { set_err(err, "expert out of range"); return false; }

  const uint64_t base = hdr_.expert_pool_off + L.base_off +
                        static_cast<uint64_t>(expert) * L.stride;

  struct Part { const void* p; uint64_t off, want, got; const char* what; };
  const Part parts[9] = {
    { d.gate.data,    L.gate.data_off,   L.gate.data_size,   d.gate.data_size,   "gate data"  },
    { d.up.data,      L.up.data_off,     L.up.data_size,     d.up.data_size,     "up data"    },
    { d.down.data,    L.down.data_off,   L.down.data_size,   d.down.data_size,   "down data"  },
    { d.gate.scales,  L.gate.scale_off,  L.gate.scale_size,  d.gate.scale_size,  "gate scale" },
    { d.up.scales,    L.up.scale_off,    L.up.scale_size,    d.up.scale_size,    "up scale"   },
    { d.down.scales,  L.down.scale_off,  L.down.scale_size,  d.down.scale_size,  "down scale" },
    { d.gate.rscales, L.gate.rscale_off, L.gate.rscale_size, d.gate.rscale_size, "gate rscale"},
    { d.up.rscales,   L.up.rscale_off,   L.up.rscale_size,   d.up.rscale_size,   "up rscale"  },
    { d.down.rscales, L.down.rscale_off, L.down.rscale_size, d.down.rscale_size, "down rscale"},
  };
  for (const Part& part : parts) {
    if (part.want == 0) continue;
    if (part.got != part.want) {
      set_err(err, std::string(part.what) + ": size " + std::to_string(part.got) +
                   " != expected " + std::to_string(part.want));
      return false;
    }
    if (!part.p) { set_err(err, std::string("missing expert section: ") + part.what); return false; }
    if (!pwrite_all(part.p, part.want, base + part.off, err)) return false;
  }

  expert_written_[layer_expert_base_[layer] + expert] = true;
  return true;
}

bool AffWriter::write_profile(const float* popularity, uint64_t count, std::string* err) {
  if (fd_ < 0) { set_err(err, "writer not open"); return false; }
  if (count * sizeof(float) != hdr_.profile_size) {
    set_err(err, "profile size mismatch: expected " +
                 std::to_string(hdr_.profile_size / sizeof(float)) + " entries");
    return false;
  }
  if (!pwrite_all(popularity, hdr_.profile_size, hdr_.profile_off, err)) return false;
  hdr_.flags |= AFF_FLAG_HAS_PROFILE;
  return true;
}

uint64_t AffWriter::missing_experts() const {
  uint64_t n = 0;
  for (bool b : expert_written_) if (!b) ++n;
  return n;
}

bool AffWriter::finalize(std::string* err) {
  if (fd_ < 0) { set_err(err, "writer not open"); return false; }
  if (finalized_) return true;

  if (!tensors_.empty() &&
      !pwrite_all(tensors_.data(),
                  tensors_.size() * sizeof(AffTensorEntry),
                  hdr_.tensor_dir_off, err)) return false;

  if (!pwrite_all(layers_.data(),
                  layers_.size() * sizeof(AffLayerDesc),
                  hdr_.layer_dir_off, err)) return false;

  if (!meta_.empty() && !pwrite_all(meta_.data(), meta_.size(), hdr_.meta_off, err)) return false;

  // Header last: its presence with a valid magic means the file is complete.
  if (!pwrite_all(&hdr_, sizeof(hdr_), 0, err)) return false;

  if (::fsync(fd_) != 0) { set_err(err, errno_str("fsync")); return false; }
  ::close(fd_);
  fd_ = -1;
  finalized_ = true;
  return true;
}

} // namespace aff
