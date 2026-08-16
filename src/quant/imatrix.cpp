#include "imatrix.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstring>
#include <cerrno>
#include <cstdio>

namespace aff {

namespace {

void set_err(std::string* e, const std::string& m) { if (e) *e = m; }

// GGUF metadata value types
enum : uint32_t {
  GT_U8=0, GT_I8=1, GT_U16=2, GT_I16=3, GT_U32=4, GT_I32=5, GT_F32=6,
  GT_BOOL=7, GT_STR=8, GT_ARR=9, GT_U64=10, GT_I64=11, GT_F64=12,
};
constexpr uint32_t GGML_TYPE_F32 = 0;

struct Cursor {
  const uint8_t* p;
  const uint8_t* end;
  bool ok = true;

  bool take(uint64_t n, const uint8_t** out) {
    if (!ok || (uint64_t)(end - p) < n) { ok = false; return false; }
    *out = p; p += n; return true;
  }
  template <typename T> bool pod(T* v) {
    const uint8_t* q;
    if (!take(sizeof(T), &q)) return false;
    std::memcpy(v, q, sizeof(T));
    return true;
  }
  bool str(std::string* s) {
    uint64_t n;
    if (!pod(&n)) return false;
    const uint8_t* q;
    if (!take(n, &q)) return false;
    s->assign(reinterpret_cast<const char*>(q), n);
    return true;
  }
  bool skip_value(uint32_t t) {
    switch (t) {
      case GT_U8: case GT_I8: case GT_BOOL: return take(1, &p) || true ? (p -= 0, true) : false;
      default: break;
    }
    const uint8_t* q;
    switch (t) {
      case GT_U8: case GT_I8: case GT_BOOL: return take(1, &q);
      case GT_U16: case GT_I16: return take(2, &q);
      case GT_U32: case GT_I32: case GT_F32: return take(4, &q);
      case GT_U64: case GT_I64: case GT_F64: return take(8, &q);
      case GT_STR: { std::string s; return str(&s); }
      case GT_ARR: {
        uint32_t et; uint64_t n;
        if (!pod(&et) || !pod(&n)) return false;
        for (uint64_t i = 0; i < n && ok; ++i) if (!skip_value(et)) return false;
        return ok;
      }
      default: ok = false; return false;
    }
  }
};

const char* suffix_of(const std::string& n, const char* suf) {
  const size_t ls = std::strlen(suf);
  if (n.size() > ls && n.compare(n.size() - ls, ls, suf) == 0) return suf;
  return nullptr;
}

} // namespace

ImatrixFile::~ImatrixFile() { close(); }

void ImatrixFile::close() {
  if (map_) { ::munmap(map_, map_size_); map_ = nullptr; map_size_ = 0; }
  if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
  entries_.clear(); index_.clear(); dataset_.clear();
}

bool ImatrixFile::open(const std::string& path, std::string* err) {
  close();
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) { set_err(err, "open " + path + ": " + std::strerror(errno)); return false; }
  struct stat st{};
  if (::fstat(fd_, &st) != 0) { set_err(err, "fstat"); close(); return false; }
  map_size_ = (uint64_t)st.st_size;
  void* m = ::mmap(nullptr, map_size_, PROT_READ, MAP_SHARED, fd_, 0);
  if (m == MAP_FAILED) { set_err(err, "mmap"); close(); return false; }
  map_ = (uint8_t*)m;

  Cursor c{map_, map_ + map_size_};
  const uint8_t* magic;
  if (!c.take(4, &magic) || std::memcmp(magic, "GGUF", 4) != 0) {
    set_err(err, "not a GGUF file"); close(); return false;
  }
  uint32_t version = 0; uint64_t ntensor = 0, nkv = 0;
  if (!c.pod(&version) || !c.pod(&ntensor) || !c.pod(&nkv)) {
    set_err(err, "truncated GGUF header"); close(); return false;
  }
  if (version != 3) {
    set_err(err, "unsupported GGUF version " + std::to_string(version)); close(); return false;
  }

  uint32_t alignment = 32;
  for (uint64_t i = 0; i < nkv; ++i) {
    std::string k; uint32_t t;
    if (!c.str(&k) || !c.pod(&t)) { set_err(err, "bad metadata"); close(); return false; }
    if (k == "general.alignment" && t == GT_U32) { c.pod(&alignment); }
    else if (k == "imatrix.chunk_count" && t == GT_U32) { c.pod(&chunk_count_); }
    else if (k == "imatrix.chunk_size" && t == GT_U32) { c.pod(&chunk_size_); }
    else if (k == "imatrix.datasets" && t == GT_ARR) {
      // Both reads CHECKED. Unchecked, a file truncated inside this header leaves `n` holding
      // whatever was on the stack and the loop below runs that many times — so a corrupt input
      // became an arbitrary read rather than an error message.
      uint32_t et = 0; uint64_t n = 0;
      if (!c.pod(&et) || !c.pod(&n)) { set_err(err, "truncated dataset array"); close(); return false; }
      for (uint64_t j = 0; j < n; ++j) {
        std::string s;
        if (et == GT_STR) { c.str(&s); if (j == 0) dataset_ = s; }
        else c.skip_value(et);
      }
    } else if (!c.skip_value(t)) { set_err(err, "bad metadata value for " + k); close(); return false; }
  }

  struct Info { std::string name; std::vector<uint64_t> dims; uint32_t type; uint64_t off; };
  std::vector<Info> infos;
  infos.reserve(ntensor);
  for (uint64_t i = 0; i < ntensor; ++i) {
    Info in;
    uint32_t nd = 0;
    if (!c.str(&in.name) || !c.pod(&nd)) { set_err(err, "bad tensor info"); close(); return false; }
    in.dims.resize(nd);
    for (uint32_t d = 0; d < nd; ++d) if (!c.pod(&in.dims[d])) { set_err(err, "bad dims"); close(); return false; }
    if (!c.pod(&in.type) || !c.pod(&in.off)) { set_err(err, "bad tensor info"); close(); return false; }
    infos.push_back(std::move(in));
  }
  if (!c.ok) { set_err(err, "truncated GGUF"); close(); return false; }

  const uint64_t here = (uint64_t)(c.p - map_);
  const uint64_t data_start = (here + alignment - 1) / alignment * alignment;

  // Pair up <name>.in_sum2 with <name>.counts.
  std::unordered_map<std::string, size_t> byname;
  for (const Info& in : infos) {
    const bool is_sum = suffix_of(in.name, ".in_sum2") != nullptr;
    const bool is_cnt = suffix_of(in.name, ".counts") != nullptr;
    if (!is_sum && !is_cnt) continue;
    if (in.type != GGML_TYPE_F32) continue;

    const std::string base = in.name.substr(0, in.name.size() - (is_sum ? 8 : 7));
    uint64_t n = 1;
    for (uint64_t d : in.dims) n *= d;
    if (data_start + in.off + n * 4 > map_size_) {
      set_err(err, "tensor " + in.name + " out of bounds"); close(); return false;
    }
    const float* ptr = reinterpret_cast<const float*>(map_ + data_start + in.off);

    auto it = byname.find(base);
    if (it == byname.end()) {
      ImatrixEntry e;
      e.name = base;
      byname[base] = entries_.size();
      entries_.push_back(std::move(e));
      it = byname.find(base);
    }
    ImatrixEntry& e = entries_[it->second];
    if (is_sum) {
      e.sum2 = ptr;
      e.cols = in.dims.empty() ? 0 : in.dims[0];
      e.n_expert = in.dims.size() > 1 ? in.dims[1] : 1;
    } else {
      e.counts = ptr;
    }
  }
  index_ = std::move(byname);
  return true;
}

const ImatrixEntry* ImatrixFile::find(const std::string& name) const {
  auto it = index_.find(name);
  return it == index_.end() ? nullptr : &entries_[it->second];
}

double ImatrixFile::count(const std::string& name, uint64_t expert) const {
  const ImatrixEntry* e = find(name);
  if (!e || !e->counts || expert >= e->n_expert) return 0.0;
  return e->counts[expert];
}

bool ImatrixFile::importance(const std::string& name, uint64_t expert,
                             std::vector<float>* out) const {
  const ImatrixEntry* e = find(name);
  if (!e || !e->sum2 || expert >= e->n_expert || e->cols == 0) return false;
  const float* src = e->sum2 + expert * e->cols;
  const double n = (e->counts && e->counts[expert] > 0) ? e->counts[expert] : 1.0;

  out->resize(e->cols);
  double mean = 0;
  for (uint64_t i = 0; i < e->cols; ++i) {
    const double v = (double)src[i] / n;
    (*out)[i] = (float)v;
    mean += v;
  }
  mean /= (double)e->cols;
  if (mean <= 0) { std::fill(out->begin(), out->end(), 1.0f); return true; }
  // Rescale to mean 1 so weighted error stays comparable to unweighted error.
  const float inv = (float)(1.0 / mean);
  for (uint64_t i = 0; i < e->cols; ++i) (*out)[i] *= inv;
  return true;
}

std::string ImatrixFile::gguf_name_for_expert(uint32_t layer, const char* which) {
  const char* g = nullptr;
  if (std::strcmp(which, "w1") == 0) g = "ffn_gate_exps";
  else if (std::strcmp(which, "w3") == 0) g = "ffn_up_exps";
  else if (std::strcmp(which, "w2") == 0) g = "ffn_down_exps";
  else return "";
  return "blk." + std::to_string(layer) + "." + g + ".weight";
}

} // namespace aff
