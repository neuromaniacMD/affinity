#include "safetensors.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstring>
#include <cerrno>
#include <cstdio>
#include <cstdlib>

namespace aff {

namespace {
void set_err(std::string* e, const std::string& m) { if (e) *e = m; }

// ---------------------------------------------------------------------------------------------
// A tiny non-allocating JSON scanner. safetensors headers are machine-generated and flat enough
// that a full JSON parser is not warranted; but the scanner is strict about structure so a
// malformed header fails loudly rather than silently mis-parsing.
// ---------------------------------------------------------------------------------------------
struct Json {
  const char* p;
  const char* end;
  bool ok = true;

  void ws() { while (p < end && (*p==' '||*p=='\t'||*p=='\n'||*p=='\r')) ++p; }
  bool lit(char c) { ws(); if (p < end && *p == c) { ++p; return true; } return false; }
  bool peek(char c) { ws(); return p < end && *p == c; }

  bool str(std::string* out) {
    ws();
    if (p >= end || *p != '"') { ok = false; return false; }
    ++p;
    out->clear();
    while (p < end && *p != '"') {
      if (*p == '\\' && p + 1 < end) {
        ++p;
        switch (*p) {
          case 'n': out->push_back('\n'); break;
          case 't': out->push_back('\t'); break;
          case 'r': out->push_back('\r'); break;
          case 'b': out->push_back('\b'); break;
          case 'f': out->push_back('\f'); break;
          case 'u': {  // keep the escape verbatim; safetensors names are ASCII in practice
            for (int i = 0; i < 4 && p + 1 < end; ++i) out->push_back(*++p);
            break;
          }
          default: out->push_back(*p);
        }
        ++p;
      } else {
        out->push_back(*p++);
      }
    }
    if (p >= end) { ok = false; return false; }
    ++p;
    return true;
  }

  bool num(int64_t* out) {
    ws();
    char* e = nullptr;
    const long long v = std::strtoll(p, &e, 10);
    if (e == p) { ok = false; return false; }
    p = e;
    *out = v;
    return true;
  }

  // Skips any value (object, array, string, number, literal).
  void skip_value() {
    ws();
    if (p >= end) { ok = false; return; }
    if (*p == '{' || *p == '[') {
      const char open = *p, close = (open == '{') ? '}' : ']';
      int depth = 0;
      while (p < end) {
        if (*p == '"') { std::string tmp; str(&tmp); if (!ok) return; continue; }
        if (*p == open) ++depth;
        else if (*p == close) { if (--depth == 0) { ++p; return; } }
        ++p;
      }
      ok = false;
    } else if (*p == '"') {
      std::string tmp; str(&tmp);
    } else {
      while (p < end && *p != ',' && *p != '}' && *p != ']') ++p;
    }
  }
};
} // namespace

StDType st_dtype_from_string(std::string_view s) noexcept {
  if (s == "BF16")    return StDType::BF16;
  if (s == "F16")     return StDType::F16;
  if (s == "F32")     return StDType::F32;
  if (s == "F64")     return StDType::F64;
  if (s == "I8")      return StDType::I8;
  if (s == "U8")      return StDType::U8;
  if (s == "I16")     return StDType::I16;
  if (s == "I32")     return StDType::I32;
  if (s == "I64")     return StDType::I64;
  if (s == "F8_E4M3") return StDType::F8_E4M3;
  if (s == "F8_E5M2") return StDType::F8_E5M2;
  if (s == "F8_E8M0") return StDType::F8_E8M0;
  if (s == "BOOL")    return StDType::BOOL;
  return StDType::Unknown;
}

const char* st_dtype_name(StDType d) noexcept {
  switch (d) {
    case StDType::BF16: return "BF16";
    case StDType::F16: return "F16";
    case StDType::F32: return "F32";
    case StDType::F64: return "F64";
    case StDType::I8: return "I8";
    case StDType::U8: return "U8";
    case StDType::I16: return "I16";
    case StDType::I32: return "I32";
    case StDType::I64: return "I64";
    case StDType::F8_E4M3: return "F8_E4M3";
    case StDType::F8_E5M2: return "F8_E5M2";
    case StDType::F8_E8M0: return "F8_E8M0";
    case StDType::BOOL: return "BOOL";
    default: return "Unknown";
  }
}

uint32_t st_dtype_size(StDType d) noexcept {
  switch (d) {
    case StDType::F64: case StDType::I64: return 8;
    case StDType::F32: case StDType::I32: return 4;
    case StDType::BF16: case StDType::F16: case StDType::I16: return 2;
    case StDType::I8: case StDType::U8: case StDType::BOOL:
    case StDType::F8_E4M3: case StDType::F8_E5M2: case StDType::F8_E8M0: return 1;
    default: return 0;
  }
}

SafeTensorsFile::~SafeTensorsFile() { close(); }

void SafeTensorsFile::close() {
  if (map_) { ::munmap(map_, map_size_); map_ = nullptr; map_size_ = 0; }
  if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
  tensors_.clear();
  index_.clear();
  metadata_.clear();
}

bool SafeTensorsFile::open(const std::string& path, std::string* err) {
  close();
  path_ = path;

  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) { set_err(err, "open " + path + ": " + std::strerror(errno)); return false; }

  struct stat st{};
  if (::fstat(fd_, &st) != 0) { set_err(err, "fstat: " + std::string(std::strerror(errno))); close(); return false; }
  map_size_ = static_cast<uint64_t>(st.st_size);
  if (map_size_ < 8) { set_err(err, "file too small"); close(); return false; }

  void* m = ::mmap(nullptr, map_size_, PROT_READ, MAP_SHARED, fd_, 0);
  if (m == MAP_FAILED) { set_err(err, "mmap: " + std::string(std::strerror(errno))); close(); return false; }
  map_ = static_cast<uint8_t*>(m);
  // Header is walked once, then tensor bodies are read sequentially — tell the kernel.
  ::madvise(map_, map_size_, MADV_SEQUENTIAL);

  uint64_t hdr_len = 0;
  std::memcpy(&hdr_len, map_, 8);
  if (hdr_len == 0 || hdr_len > map_size_ - 8) {
    set_err(err, "invalid header length " + std::to_string(hdr_len)); close(); return false;
  }
  const uint8_t* data_base = map_ + 8 + hdr_len;
  const uint64_t data_size = map_size_ - 8 - hdr_len;

  Json j{reinterpret_cast<const char*>(map_ + 8), reinterpret_cast<const char*>(map_ + 8 + hdr_len)};
  if (!j.lit('{')) { set_err(err, "header is not a JSON object"); close(); return false; }

  if (!j.peek('}')) {
    do {
      std::string key;
      if (!j.str(&key)) { set_err(err, "malformed header: expected key"); close(); return false; }
      if (!j.lit(':')) { set_err(err, "malformed header: expected ':'"); close(); return false; }

      if (key == "__metadata__") {
        const char* s = j.p; j.ws(); const char* b = j.p;
        j.skip_value();
        metadata_.assign(b, static_cast<size_t>(j.p - b));
        (void)s;
        continue;
      }

      StTensor t;
      t.name = key;
      if (!j.lit('{')) { set_err(err, "malformed tensor entry for " + key); close(); return false; }
      do {
        std::string field;
        if (!j.str(&field)) { set_err(err, "malformed field in " + key); close(); return false; }
        if (!j.lit(':')) { set_err(err, "expected ':' in " + key); close(); return false; }
        if (field == "dtype") {
          std::string dt;
          if (!j.str(&dt)) { set_err(err, "bad dtype in " + key); close(); return false; }
          t.dtype = st_dtype_from_string(dt);
          if (t.dtype == StDType::Unknown) {
            set_err(err, "unsupported dtype '" + dt + "' for " + key); close(); return false;
          }
        } else if (field == "shape") {
          if (!j.lit('[')) { set_err(err, "bad shape in " + key); close(); return false; }
          if (!j.peek(']')) {
            do { int64_t v; if (!j.num(&v)) { set_err(err, "bad shape value in " + key); close(); return false; }
                 t.shape.push_back(v); } while (j.lit(','));
          }
          if (!j.lit(']')) { set_err(err, "unterminated shape in " + key); close(); return false; }
        } else if (field == "data_offsets") {
          int64_t a = 0, b = 0;
          if (!j.lit('[') || !j.num(&a) || !j.lit(',') || !j.num(&b) || !j.lit(']')) {
            set_err(err, "bad data_offsets in " + key); close(); return false;
          }
          t.offset = static_cast<uint64_t>(a);
          t.nbytes = static_cast<uint64_t>(b - a);
        } else {
          j.skip_value();
        }
        if (!j.ok) { set_err(err, "header parse failed in " + key); close(); return false; }
      } while (j.lit(','));
      if (!j.lit('}')) { set_err(err, "unterminated tensor entry for " + key); close(); return false; }

      if (t.offset + t.nbytes > data_size) {
        set_err(err, "tensor " + key + " extends past end of file"); close(); return false;
      }
      const uint64_t expect = t.numel() * st_dtype_size(t.dtype);
      if (expect != t.nbytes) {
        set_err(err, "tensor " + key + ": shape implies " + std::to_string(expect) +
                     " bytes but header says " + std::to_string(t.nbytes));
        close(); return false;
      }
      t.data = data_base + t.offset;
      index_[t.name] = tensors_.size();
      tensors_.push_back(std::move(t));
    } while (j.lit(','));
  }
  if (!j.lit('}')) { set_err(err, "unterminated header object"); close(); return false; }
  return true;
}

const StTensor* SafeTensorsFile::find(const std::string& name) const {
  auto it = index_.find(name);
  return it == index_.end() ? nullptr : &tensors_[it->second];
}

bool load_safetensors_index(const std::string& index_path,
                            std::unordered_map<std::string, std::string>* weight_map,
                            std::string* err) {
  FILE* f = std::fopen(index_path.c_str(), "rb");
  if (!f) { set_err(err, "open " + index_path + ": " + std::strerror(errno)); return false; }
  std::string buf;
  char tmp[65536];
  size_t n;
  while ((n = std::fread(tmp, 1, sizeof(tmp), f)) > 0) buf.append(tmp, n);
  std::fclose(f);

  // Locate "weight_map": { ... } and read its flat string->string pairs.
  const std::string key = "\"weight_map\"";
  const size_t k = buf.find(key);
  if (k == std::string::npos) { set_err(err, "no weight_map in " + index_path); return false; }
  const size_t brace = buf.find('{', k + key.size());
  if (brace == std::string::npos) { set_err(err, "malformed weight_map"); return false; }

  Json j{buf.data() + brace, buf.data() + buf.size()};
  if (!j.lit('{')) { set_err(err, "malformed weight_map"); return false; }
  weight_map->clear();
  if (!j.peek('}')) {
    do {
      std::string name, shard;
      if (!j.str(&name) || !j.lit(':') || !j.str(&shard)) {
        set_err(err, "malformed weight_map entry"); return false;
      }
      (*weight_map)[name] = shard;
    } while (j.lit(','));
  }
  return true;
}

} // namespace aff
