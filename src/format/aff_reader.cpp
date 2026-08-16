#include "aff_reader.h"

#include <cstdio>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstring>
#include <cerrno>

namespace aff {

namespace {
void set_err(std::string* err, const std::string& m) { if (err) *err = m; }
std::string errno_str(const char* what) {
  return std::string(what) + ": " + std::strerror(errno);
}
} // namespace

AffReader::~AffReader() { close(); }

void AffReader::close() {
  if (map_)  { ::munmap(map_, map_size_);   map_  = nullptr; map_size_  = 0; }
  if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
  hdr_ = nullptr; base_ = nullptr; layers_ = nullptr;
  expert_pool_ = nullptr;
  meta_ = {};
  tensor_index_.clear();
}

bool AffReader::open(const std::string& path, std::string* err) {
  close();

  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) { set_err(err, errno_str("open")); return false; }

  struct stat st{};
  if (::fstat(fd_, &st) != 0) { set_err(err, errno_str("fstat")); close(); return false; }
  map_size_ = static_cast<uint64_t>(st.st_size);
  if (map_size_ < sizeof(AffHeader)) {
    set_err(err, "file too small to contain a header"); close(); return false;
  }

  void* m = ::mmap(nullptr, map_size_, PROT_READ, MAP_SHARED, fd_, 0);
  if (m == MAP_FAILED) { set_err(err, errno_str("mmap")); close(); return false; }
  map_ = static_cast<uint8_t*>(m);
  base_ = map_;
  hdr_ = reinterpret_cast<const AffHeader*>(base_);

  // ---- validation ---------------------------------------------------------------------------
  if (std::memcmp(hdr_->magic, kMagic, sizeof(kMagic)) != 0) {
    set_err(err, "bad magic — not a .aff file"); close(); return false;
  }
  if (hdr_->version != kVersion) {
    set_err(err, "unsupported .aff version " + std::to_string(hdr_->version) +
                 " (expected " + std::to_string(kVersion) + ")");
    close(); return false;
  }
  if (hdr_->file_size != map_size_) {
    set_err(err, "truncated file: header says " + std::to_string(hdr_->file_size) +
                 " bytes, found " + std::to_string(map_size_));
    close(); return false;
  }

  auto in_bounds = [&](uint64_t off, uint64_t size) {
    return off <= map_size_ && size <= map_size_ - off;
  };
  if (!in_bounds(hdr_->meta_off, hdr_->meta_size) ||
      !in_bounds(hdr_->layer_dir_off, hdr_->layer_count * sizeof(AffLayerDesc)) ||
      !in_bounds(hdr_->tensor_dir_off, hdr_->tensor_count * sizeof(AffTensorEntry)) ||
      !in_bounds(hdr_->tensor_data_off, hdr_->tensor_data_size) ||
      !in_bounds(hdr_->expert_pool_off, hdr_->expert_pool_size) ||
      !in_bounds(hdr_->profile_off, hdr_->profile_size)) {
    set_err(err, "corrupt header: a section lies outside the file"); close(); return false;
  }

  meta_ = std::string_view(reinterpret_cast<const char*>(base_ + hdr_->meta_off),
                           hdr_->meta_size);
  layers_ = reinterpret_cast<const AffLayerDesc*>(base_ + hdr_->layer_dir_off);

  // Validate each layer's experts actually fit in the declared pool.
  for (uint64_t i = 0; i < hdr_->layer_count; ++i) {
    const AffLayerDesc& L = layers_[i];
    const uint64_t span = L.base_off + static_cast<uint64_t>(L.n_experts) * L.stride;
    if (L.stride == 0 || span > hdr_->expert_pool_size) {
      set_err(err, "corrupt layer descriptor " + std::to_string(i));
      close(); return false;
    }
  }

  const auto* tents = reinterpret_cast<const AffTensorEntry*>(base_ + hdr_->tensor_dir_off);
  for (uint64_t i = 0; i < hdr_->tensor_count; ++i) {
    // Names are written zero-padded; guarantee termination before constructing a std::string.
    char name[kMaxTensorName + 1];
    std::memcpy(name, tents[i].name, kMaxTensorName);
    name[kMaxTensorName] = '\0';
    tensor_index_[name] = &tents[i];
  }

  expert_pool_ = base_ + hdr_->expert_pool_off;
  return true;
}

void AffReader::advise_expert(uint32_t layer, uint32_t expert, bool want) const {
  if (fd_ < 0 || !hdr_) return;
  const AffLayerDesc& L = layers_[layer];
  const uint64_t off = hdr_->expert_pool_off + L.base_off + (uint64_t)expert * L.stride;
  // POSIX_FADV_DONTNEED, not madvise: this must drop the PAGE CACHE, not merely unmap the pages
  // from this process. madvise on a shared file mapping would leave the duplicate in place.
  (void)posix_fadvise(fd_, (off_t)off, (off_t)L.stride,
                      want ? POSIX_FADV_WILLNEED : POSIX_FADV_DONTNEED);
}

uint64_t AffReader::release_cache() const {
  if (fd_ < 0 || !map_size_) return 0;
  // Two different things, and the first one is not the one it looks like.
  //
  // POSIX_FADV_DONTNEED returns the UNMAPPED page cache, which is most of the file and shows up as
  // MemFree rising. It cannot touch a page this process still has a PTE for.
  //
  // MADV_DONTNEED was supposed to be what dropped those PTEs, and on this mapping it does not: it
  // returns success and leaves everything the dense and draft loaders faulted in by reading the
  // mapping resident until exit. So the VMA is replaced with a fresh one over the same address
  // instead. MAP_FIXED atomically drops every PTE in the range, and because the new mapping is the
  // same file at the same offset with the same protection, every pointer into it stays valid and
  // simply re-faults on the next read. That matters because the embedding table lives here and is
  // read every token; it comes back lazily instead of all at once.
  //
  // Worth the care because that residue is what decides how large a pool hipHostRegister will take
  // — the band in static_placement.hip depends on what else the process has mapped.
  void* m = ::mmap(map_, map_size_, PROT_READ, MAP_SHARED | MAP_FIXED, fd_, 0);
  if (m == MAP_FAILED)
    std::fprintf(stderr, "page cache: could not remap %.1f GiB to drop its page tables: %s\n",
                 map_size_ / 1073741824.0, std::strerror(errno));
  (void)::posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED);
  return map_size_;
}

ExpertView AffReader::expert(uint32_t layer, uint32_t expert) const {
  const AffLayerDesc& L = layers_[layer];
  const uint8_t* p = expert_ptr(layer, expert);
  auto view = [p](const AffMatrixDesc& m) {
    MatrixView v;
    v.data    = m.data_size   ? p + m.data_off   : nullptr;
    v.scales  = m.scale_size  ? p + m.scale_off  : nullptr;
    v.rscales = m.rscale_size ? p + m.rscale_off : nullptr;
    return v;
  };
  ExpertView v;
  v.gate  = view(L.gate);
  v.up    = view(L.up);
  v.down  = view(L.down);
  v.bytes = L.stride;
  return v;
}

const AffTensorEntry* AffReader::find_tensor(const std::string& name) const {
  auto it = tensor_index_.find(name);
  return it == tensor_index_.end() ? nullptr : it->second;
}

bool AffReader::set_profile_override(const float* p, uint64_t count, std::string* err) {
  if (!hdr_ || !layers_ || !hdr_->layer_count) { if (err) *err = "container not open"; return false; }
  // From the LAYER DIRECTORY, not the header: `n_experts` is per-layer geometry (AffLayerDesc) and
  // the header carries no expert count at all. Every layer is identical in expert count by
  // construction — that is what makes expert addressing O(1) — so layer 0 answers for the file, the
  // same way the wire format is latched from layer 0's gate.
  const uint32_t n_expert = layers_[0].n_experts;
  const uint64_t want = hdr_->layer_count * (uint64_t)n_expert;
  if (count != want) {
    if (err) *err = "expert profile has " + std::to_string(count) + " entries, this container needs "
                    + std::to_string(want) + " (" + std::to_string(hdr_->layer_count) + " layers x "
                    + std::to_string(n_expert) + " experts)";
    return false;
  }
  // Copied rather than aliased: see the note on `prof_own_`. Cheap — 11008 floats is 43 KiB.
  prof_own_.assign(p, p + count);
  prof_override_ = prof_own_.data();
  return true;
}

} // namespace aff
