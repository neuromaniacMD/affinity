#include "engine/route_trace.h"

#include <cstring>

namespace aff {

namespace {
// 4 MiB. Sized so a whole 512-token run's worth of dispatches is a handful of writes and the
// buffer is allocated once, at open, like every other buffer in the engine.
constexpr size_t kBuf = 4u << 20;
} // namespace

RouteTrace& route_trace() {
  static RouteTrace t;
  return t;
}

bool RouteTrace::open(const std::string& path, uint32_t n_layer, uint32_t n_expert, uint32_t k) {
  close();
  f_ = std::fopen(path.c_str(), "wb");
  if (!f_) return false;
  buf_.resize(kBuf);
  used_ = 0;
  seq_ = 0;
  char hdr[128];
  const int n = std::snprintf(hdr, sizeof hdr, "# aff-route-trace 1 layers=%u experts=%u k=%u\n",
                              n_layer, n_expert, k);
  put(hdr, (size_t)n);
  return true;
}

void RouteTrace::put(const char* s, size_t n) {
  if (!f_) return;
  if (used_ + n > buf_.size()) flush();
  if (n > buf_.size()) { std::fwrite(s, 1, n, f_); return; }
  std::memcpy(buf_.data() + used_, s, n);
  used_ += n;
}

void RouteTrace::gate(uint32_t handle, const uint32_t* q, uint32_t n) {
  if (!f_) return;
  put("G ", 2);
  putu(handle);
  for (uint32_t i = 0; i < n; ++i) putu(q[i]);
  put("\n", 1);
}

void RouteTrace::putu(uint32_t v) {
  // A hand-rolled itoa rather than snprintf: this runs 43 times a layer dispatch times the token
  // count, and snprintf's format parsing is most of the cost of writing a small integer.
  char t[12];
  int i = 0;
  do { t[i++] = (char)('0' + v % 10u); v /= 10u; } while (v);
  char out[13];
  int j = 0;
  while (i) out[j++] = t[--i];
  out[j++] = ' ';
  put(out, (size_t)j);
}

void RouteTrace::flush() {
  if (!f_ || !used_) return;
  std::fwrite(buf_.data(), 1, used_, f_);
  used_ = 0;
}

void RouteTrace::resident(uint32_t layer, const uint32_t* experts, uint32_t n) {
  if (!f_) return;
  put("R ", 2);
  putu(layer);
  for (uint32_t i = 0; i < n; ++i) putu(experts[i]);
  put("\n", 1);
}

void RouteTrace::dispatch(char tag, uint32_t layer, uint32_t nb_tok, const uint32_t* sel,
                          uint32_t n_sel, const uint32_t* miss, uint32_t n_miss) {
  if (!f_) return;
  const char pre[2] = {tag, ' '};
  put(pre, 2);
  putu((uint32_t)seq_++);
  putu(layer);
  putu(nb_tok);
  for (uint32_t i = 0; i < n_sel; ++i) putu(sel[i]);
  put("| ", 2);
  for (uint32_t i = 0; i < n_miss; ++i) putu(miss[i]);
  put("\n", 1);
}

void RouteTrace::close() {
  if (!f_) return;
  flush();
  std::fclose(f_);
  f_ = nullptr;
}

} // namespace aff
