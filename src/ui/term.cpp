#include "ui/term.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <termios.h>
#include <sys/ioctl.h>
#include <unistd.h>

namespace aff::ui {
namespace {

// ---- what a signal has to be able to undo -------------------------------------------------------
//
// Restoring the screen is three escape sequences and a tcsetattr, and it has to happen on paths
// that do not run destructors: the engine aborts deliberately in a dozen places, and an abort in
// the alternate screen leaves a terminal with no cursor and no echo, which the operator has to fix
// with `reset`. So the state a handler needs lives here, at file scope, and the handler touches
// nothing else.
volatile sig_atomic_t g_winch = 0;
volatile sig_atomic_t g_quit = 0;
volatile sig_atomic_t g_active = 0;
struct termios g_saved_tio;
volatile sig_atomic_t g_saved_tio_ok = 0;

constexpr char kRestore[] = "\033[0m\033[?25h\033[?1049l";

void restore_now() {
  if (!g_active) return;
  g_active = 0;
  // write(2), not fputs: stdio locks, and a handler that takes the stdout lock while the
  // interrupted thread holds it deadlocks the process as it is trying to die cleanly.
  ssize_t n = ::write(STDOUT_FILENO, kRestore, sizeof(kRestore) - 1);
  (void)n;
  if (g_saved_tio_ok) ::tcsetattr(STDIN_FILENO, TCSANOW, &g_saved_tio);
}

void on_winch(int) { g_winch = 1; }

void on_quit(int sig) {
  // The first interrupt asks the run to stop, so the engine can finish its block and print its
  // summary. A second one means the operator has waited long enough.
  if (g_quit) { restore_now(); ::signal(sig, SIG_DFL); ::raise(sig); return; }
  g_quit = 1;
}

void on_fatal(int sig) {
  restore_now();
  ::signal(sig, SIG_DFL);
  ::raise(sig);
}

// xterm-256 for terminals that do not advertise truecolor. The cube is 6x6x6 from 16, then 24
// greys; a near-grey colour quantises visibly better in the grey ramp than in the cube, so it is
// first.
int to256(Rgb c) {
  const int r = (int)(c >> 16 & 255), g = (int)(c >> 8 & 255), b = (int)(c & 255);
  const int mx = r > g ? (r > b ? r : b) : (g > b ? g : b);
  const int mn = r < g ? (r < b ? r : b) : (g < b ? g : b);
  if (mx - mn < 10) {
    const int lvl = (r + g + b) / 3;
    if (lvl < 8) return 16;
    if (lvl > 248) return 231;
    return 232 + (lvl - 8) * 24 / 240;
  }
  auto q = [](int v) { return v < 48 ? 0 : v < 114 ? 1 : (v - 35) / 40; };
  return 16 + 36 * q(r) + 6 * q(g) + q(b);
}

void append_uint(std::string* s, unsigned v) {
  char b[12];
  int n = 0;
  do { b[n++] = char('0' + v % 10); v /= 10; } while (v);
  while (n) s->push_back(b[--n]);
}

void append_sgr_color(std::string* s, Rgb c, bool fg, bool truecolor) {
  s->append(fg ? "\033[38;" : "\033[48;");
  if (truecolor) {
    s->append("2;");
    append_uint(s, c >> 16 & 255); s->push_back(';');
    append_uint(s, c >> 8 & 255);  s->push_back(';');
    append_uint(s, c & 255);
  } else {
    s->append("5;");
    append_uint(s, (unsigned)to256(c));
  }
  s->push_back('m');
}

void append_utf8(std::string* s, char32_t c) {
  if (c < 0x80) { s->push_back((char)c); return; }
  if (c < 0x800) {
    s->push_back((char)(0xC0 | c >> 6));
    s->push_back((char)(0x80 | (c & 0x3F)));
    return;
  }
  if (c < 0x10000) {
    s->push_back((char)(0xE0 | c >> 12));
    s->push_back((char)(0x80 | (c >> 6 & 0x3F)));
    s->push_back((char)(0x80 | (c & 0x3F)));
    return;
  }
  s->push_back((char)(0xF0 | c >> 18));
  s->push_back((char)(0x80 | (c >> 12 & 0x3F)));
  s->push_back((char)(0x80 | (c >> 6 & 0x3F)));
  s->push_back((char)(0x80 | (c & 0x3F)));
}

}  // namespace

// ---- colour -------------------------------------------------------------------------------------

Rgb mix(Rgb a, Rgb b, double t) {
  if (t <= 0) return a;
  if (t >= 1) return b;
  auto ch = [&](int sh) {
    const double x = (double)(a >> sh & 255), y = (double)(b >> sh & 255);
    return (int)(x + (y - x) * t + 0.5);
  };
  return rgb(ch(16), ch(8), ch(0));
}

Rgb ramp(const Rgb* stops, int n, double t) {
  if (n <= 0) return 0;
  if (n == 1 || t <= 0) return stops[0];
  if (t >= 1) return stops[n - 1];
  const double p = t * (double)(n - 1);
  const int i = (int)p;
  return mix(stops[i], stops[i + 1], p - (double)i);
}

// ---- text ---------------------------------------------------------------------------------------

int char_width(char32_t c) {
  if (c == 0) return 0;
  if (c < 32 || (c >= 0x7F && c < 0xA0)) return 0;
  // Combining marks, the ranges a model's output realistically reaches.
  if ((c >= 0x0300 && c <= 0x036F) || (c >= 0x1AB0 && c <= 0x1AFF) ||
      (c >= 0x20D0 && c <= 0x20FF) || (c >= 0xFE00 && c <= 0xFE0F) ||
      (c >= 0xFE20 && c <= 0xFE2F)) return 0;
  // East Asian Wide and Fullwidth, Hangul, and the emoji planes.
  if ((c >= 0x1100 && c <= 0x115F) || (c >= 0x2E80 && c <= 0xA4CF) ||
      (c >= 0xA960 && c <= 0xA97F) || (c >= 0xAC00 && c <= 0xD7A3) ||
      (c >= 0xF900 && c <= 0xFAFF) || (c >= 0xFE30 && c <= 0xFE6F) ||
      (c >= 0xFF00 && c <= 0xFF60) || (c >= 0xFFE0 && c <= 0xFFE6) ||
      (c >= 0x1F300 && c <= 0x1F64F) || (c >= 0x1F900 && c <= 0x1F9FF) ||
      (c >= 0x20000 && c <= 0x3FFFD)) return 2;
  return 1;
}

char32_t utf8_next(std::string_view s, size_t* i) {
  const size_t n = s.size();
  size_t p = *i;
  if (p >= n) { *i = n; return 0; }
  const unsigned char c0 = (unsigned char)s[p];
  int len = c0 < 0x80 ? 1 : (c0 & 0xE0) == 0xC0 ? 2 : (c0 & 0xF0) == 0xE0 ? 3
                        : (c0 & 0xF8) == 0xF0 ? 4 : 0;
  if (len == 0 || p + (size_t)len > n) { *i = p + 1; return 0xFFFD; }
  char32_t cp = len == 1 ? c0 : (char32_t)(c0 & (0xFF >> (len + 1)));
  for (int k = 1; k < len; ++k) {
    const unsigned char ck = (unsigned char)s[p + (size_t)k];
    if ((ck & 0xC0) != 0x80) { *i = p + 1; return 0xFFFD; }
    cp = cp << 6 | (ck & 0x3F);
  }
  *i = p + (size_t)len;
  return cp;
}

// ---- canvas -------------------------------------------------------------------------------------

void Canvas::resize(int w, int h) {
  if (w == w_ && h == h_) return;
  w_ = w > 0 ? w : 0;
  h_ = h > 0 ? h : 0;
  cur_.assign((size_t)w_ * (size_t)h_, Cell{});
  prev_.assign((size_t)w_ * (size_t)h_, Cell{});
  dirty_all_ = true;
}

void Canvas::clear(Rgb bg) {
  for (auto& c : cur_) c = Cell{U' ', rgb(200, 200, 200), bg, 0};
}

void Canvas::put(int x, int y, char32_t ch, Rgb fg, Rgb bg, uint8_t attr) {
  if (x < 0 || y < 0 || x >= w_ || y >= h_) return;
  Cell& c = cur_[(size_t)y * (size_t)w_ + (size_t)x];
  c.ch = ch;
  if (fg != kKeep) c.fg = fg;
  if (bg != kKeep) c.bg = bg;
  c.attr = attr;
}

int Canvas::text(int x, int y, std::string_view s, Rgb fg, Rgb bg, uint8_t attr) {
  size_t i = 0;
  while (i < s.size() && x < w_) {
    const char32_t cp = utf8_next(s, &i);
    if (!cp) continue;
    const int cw = char_width(cp);
    if (cw == 0) continue;
    if (x + cw > w_) break;
    put(x, y, cp, fg, bg, attr);
    // The trailing half of a double-width glyph must not keep whatever was under it, or the diff
    // will later repaint that column and split the glyph.
    if (cw == 2) put(x + 1, y, U'\0', fg, bg, attr);
    x += cw;
  }
  return x;
}

int Canvas::text_clip(int x, int y, int max_w, std::string_view s, Rgb fg, Rgb bg, uint8_t attr) {
  if (max_w <= 0) return x;
  const int end = x + max_w < w_ ? x + max_w : w_;
  size_t i = 0;
  while (i < s.size() && x < end && x < w_) {
    const size_t start = i;
    const char32_t cp = utf8_next(s, &i);
    if (!cp) continue;
    const int cw = char_width(cp);
    if (cw == 0) continue;
    if (x + cw > end) {
      // More text than room: the last cell says so rather than cutting mid-word and reading as if
      // the value itself were short.
      if (start < s.size()) put(end - 1, y, U'…', fg, bg, attr);
      return end;
    }
    put(x, y, cp, fg, bg, attr);
    if (cw == 2) put(x + 1, y, U'\0', fg, bg, attr);
    x += cw;
  }
  return x;
}

void Canvas::fill(int x, int y, int w, int h, char32_t ch, Rgb fg, Rgb bg) {
  for (int j = y; j < y + h; ++j)
    for (int i = x; i < x + w; ++i) put(i, j, ch, fg, bg, 0);
}

void Canvas::flush(std::string* out) {
  out->clear();
  if (w_ <= 0 || h_ <= 0) return;
  // A style that cannot occur, so the first cell always emits its SGR.
  Rgb pf = 0xDEADBEEF, pb = 0xDEADBEEF;
  uint8_t pa = 0xFF;
  int cx = -1, cy = -1;

  for (int y = 0; y < h_; ++y) {
    for (int x = 0; x < w_; ++x) {
      const size_t k = (size_t)y * (size_t)w_ + (size_t)x;
      const Cell& c = cur_[k];
      if (!dirty_all_) {
        const Cell& p = prev_[k];
        if (c.ch == p.ch && c.fg == p.fg && c.bg == p.bg && c.attr == p.attr) continue;
      }
      // The second half of a double-width glyph is not addressable — the terminal advanced over it
      // when the first half was written.
      if (c.ch == U'\0') continue;
      if (cy != y || cx != x) {
        out->append("\033[");
        append_uint(out, (unsigned)(y + 1));
        out->push_back(';');
        append_uint(out, (unsigned)(x + 1));
        out->push_back('H');
        cy = y;
        cx = x;
      }
      if (c.attr != pa) {
        // Attributes have no individual "off", so a change resets and re-states the colour.
        out->append("\033[0m");
        if (c.attr & kBold) out->append("\033[1m");
        if (c.attr & kDim) out->append("\033[2m");
        if (c.attr & kUnderline) out->append("\033[4m");
        pa = c.attr;
        pf = pb = 0xDEADBEEF;
      }
      if (c.fg != pf) { append_sgr_color(out, c.fg, true, truecolor_); pf = c.fg; }
      if (c.bg != pb) { append_sgr_color(out, c.bg, false, truecolor_); pb = c.bg; }
      append_utf8(out, c.ch);
      cx += char_width(c.ch);
    }
  }
  out->append("\033[0m");
  prev_ = cur_;
  dirty_all_ = false;
}

// ---- terminal -----------------------------------------------------------------------------------

bool Terminal::is_tty() { return ::isatty(STDOUT_FILENO) == 1; }

bool Terminal::truecolor() {
  if (const char* c = std::getenv("COLORTERM"))
    if (std::strstr(c, "truecolor") || std::strstr(c, "24bit")) return true;
  // Everything this is likely to run under either sets COLORTERM or is a modern xterm; the 256
  // colour path is a correct rendering rather than a broken one, so guessing wrong is cosmetic.
  const char* t = std::getenv("TERM");
  return t && (std::strstr(t, "256color") || std::strstr(t, "kitty") ||
               std::strstr(t, "alacritty"));
}

bool Terminal::enter() {
  if (active_) return true;
  if (!is_tty()) return false;

  if (::isatty(STDIN_FILENO) == 1 && ::tcgetattr(STDIN_FILENO, &g_saved_tio) == 0) {
    g_saved_tio_ok = 1;
    struct termios t = g_saved_tio;
    // Keys arrive as they are typed and are not echoed into the frame. ISIG stays ON: Ctrl-C must
    // still raise SIGINT, because that is how a run is stopped and the handler below is what makes
    // stopping clean.
    t.c_lflag &= (tcflag_t)~(ICANON | ECHO);
    t.c_cc[VMIN] = 0;
    t.c_cc[VTIME] = 0;
    if (::tcsetattr(STDIN_FILENO, TCSANOW, &t) == 0) raw_ = true;
  }

  struct sigaction sa{};
  sa.sa_handler = on_winch;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESTART;
  ::sigaction(SIGWINCH, &sa, nullptr);

  sa.sa_handler = on_quit;
  ::sigaction(SIGINT, &sa, nullptr);
  ::sigaction(SIGTERM, &sa, nullptr);

  // SIGHUP is deliberately left at its default. Killing the ssh that started a remote run is how
  // this engine is stopped from another machine, and a handler that only sets a flag would turn
  // that into a process nobody can reach.
  sa.sa_handler = on_fatal;
  sa.sa_flags = SA_RESETHAND;
  ::sigaction(SIGSEGV, &sa, nullptr);
  ::sigaction(SIGABRT, &sa, nullptr);
  ::sigaction(SIGBUS, &sa, nullptr);
  ::sigaction(SIGFPE, &sa, nullptr);
  ::sigaction(SIGILL, &sa, nullptr);

  const char* seq = "\033[?1049h\033[?25l\033[2J";
  if (::write(STDOUT_FILENO, seq, std::strlen(seq)) < 0) return false;
  active_ = true;
  g_active = 1;
  return true;
}

void Terminal::leave() {
  if (!active_) return;
  active_ = false;
  restore_now();
  raw_ = false;
}

Terminal::~Terminal() { leave(); }

void Terminal::size(int* w, int* h) const {
  struct winsize ws{};
  if (::ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0 && ws.ws_row > 0) {
    *w = ws.ws_col;
    *h = ws.ws_row;
    return;
  }
  *w = 80;
  *h = 24;
}

bool Terminal::take_resize() {
  if (!g_winch) return false;
  g_winch = 0;
  return true;
}

bool Terminal::quit_requested() { return g_quit != 0; }

int Terminal::read_key() {
  if (!raw_) return -1;
  unsigned char c = 0;
  const ssize_t n = ::read(STDIN_FILENO, &c, 1);
  if (n != 1) return -1;
  return (int)c;
}

}  // namespace aff::ui
