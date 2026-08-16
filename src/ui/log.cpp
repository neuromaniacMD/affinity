#include "ui/log.h"

#include <cstdarg>
#include <cstdio>
#include <string>
#include <vector>

namespace aff::ui {
namespace {

Sink g_sink = nullptr;
void* g_ctx = nullptr;
Teardown g_teardown = nullptr;

// vsnprintf twice rather than a fixed buffer: the load banners interpolate paths and driver error
// strings, and a truncated diagnostic is worse than none.
std::string format(const char* fmt, va_list ap) {
  va_list copy;
  va_copy(copy, ap);
  char stack[512];
  const int n = std::vsnprintf(stack, sizeof(stack), fmt, copy);
  va_end(copy);
  if (n < 0) return {};
  if ((size_t)n < sizeof(stack)) return std::string(stack, (size_t)n);
  std::vector<char> heap((size_t)n + 1);
  std::vsnprintf(heap.data(), heap.size(), fmt, ap);
  return std::string(heap.data(), (size_t)n);
}

}  // namespace

void set_sink(Sink s, void* ctx) {
  g_sink = s;
  g_ctx = ctx;
}

bool dashboard_active() { return g_sink != nullptr; }

void out(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  if (!g_sink) {
    std::vprintf(fmt, ap);
    va_end(ap);
    std::fflush(stdout);
    return;
  }
  const std::string s = format(fmt, ap);
  va_end(ap);
  g_sink(g_ctx, Level::Info, s);
}

void err(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  if (!g_sink) {
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    return;
  }
  const std::string s = format(fmt, ap);
  va_end(ap);
  g_sink(g_ctx, Level::Note, s);
}

void set_teardown(Teardown t) { g_teardown = t; }

void fatal(const char* fmt, ...) {
  // Order matters: the teardown clears the sink, so the write below goes to a restored terminal
  // whichever mode the process was in.
  if (g_teardown) g_teardown();
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(stderr, fmt, ap);
  va_end(ap);
  std::fflush(stderr);
}

void tok(std::string_view piece) {
  if (!g_sink) {
    std::fwrite(piece.data(), 1, piece.size(), stdout);
    std::fflush(stdout);
    return;
  }
  g_sink(g_ctx, Level::Token, piece);
}

}  // namespace aff::ui
