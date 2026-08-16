// A terminal the dashboard draws into: a cell grid, a diffing flush, and the mode switching that
// has to be undone on every exit path including a signal.
//
// No curses. The dependency would be one more thing to have on a box that is otherwise ROCm and a
// compiler, and the subset used here — alternate screen, truecolor SGR, absolute positioning — is
// stable across every terminal this runs on. What curses would buy is the terminfo lookup, and the
// engine already refuses to start on hardware it does not recognise; a terminal that cannot do
// 24-bit colour gets the 256-colour path below rather than a query.
//
// DRAWING IS INTO A GRID, NOT INTO A STREAM. Panels overlap in the layout and are drawn in an order
// nobody should have to think about, so a panel writes cells and the frame is serialised once at
// the end. That also makes the flush a diff against the previous frame, which is what keeps this
// usable over ssh: a full 200x60 repaint is ~40 KB, and a steady-state frame changes a few hundred
// cells.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace aff::ui {

// 0x00RRGGBB. The high byte is a tag rather than alpha — kKeep leaves whatever the cell already
// had, which is what lets a foreground write land on a panel background it does not know.
using Rgb = uint32_t;
inline constexpr Rgb kKeep = 0xFF000000u;

inline constexpr Rgb rgb(int r, int g, int b) {
  return (Rgb)((r & 255) << 16 | (g & 255) << 8 | (b & 255));
}

// Linear interpolation in sRGB space. Not perceptually uniform, and deliberately so: the gradients
// here are read as "more" and "less" rather than compared against each other, and a Lab ramp costs
// a cube root per cell.
Rgb mix(Rgb a, Rgb b, double t);

// A gradient through N stops, t in [0,1].
Rgb ramp(const Rgb* stops, int n, double t);

enum Attr : uint8_t { kBold = 1, kDim = 2, kUnderline = 4 };

struct Cell {
  char32_t ch = U' ';
  Rgb fg = rgb(200, 200, 200);
  Rgb bg = 0;
  uint8_t attr = 0;
};

// Columns a codepoint occupies. Zero for combining marks and controls, two for the CJK/emoji
// ranges, one otherwise. A local table rather than wcwidth(3) because that answers according to
// LC_CTYPE, and the engine never calls setlocale — under the default "C" locale wcwidth reports -1
// for everything above ASCII, which would collapse the output pane the first time a model emits an
// em dash.
int char_width(char32_t c);

// Decode one UTF-8 sequence. Returns the codepoint and advances `i`; invalid bytes decode to
// U+FFFD and advance one, so a partial token at the end of a stream cannot spin.
char32_t utf8_next(std::string_view s, size_t* i);

class Canvas {
 public:
  void resize(int w, int h);
  int w() const { return w_; }
  int h() const { return h_; }

  void clear(Rgb bg);
  void put(int x, int y, char32_t ch, Rgb fg, Rgb bg = kKeep, uint8_t attr = 0);
  // Returns the column after the last one written, so callers can chain runs of different colour.
  int text(int x, int y, std::string_view utf8, Rgb fg, Rgb bg = kKeep, uint8_t attr = 0);
  // As text(), but stops at `max_w` columns and writes an ellipsis in the last one when it had to.
  int text_clip(int x, int y, int max_w, std::string_view utf8, Rgb fg, Rgb bg = kKeep,
                uint8_t attr = 0);
  void fill(int x, int y, int w, int h, char32_t ch, Rgb fg, Rgb bg);

  // Serialise the difference against the last flush into `out` as one ANSI batch. The caller writes
  // it in a single write(2): two writes per frame is how a terminal ends up showing half of one.
  void flush(std::string* out);
  // Force the next flush to emit every cell. Needed after a resize and after entering the alternate
  // screen, where the terminal's contents are not what the previous frame left.
  void invalidate() { dirty_all_ = true; }

  // Emit 24-bit SGR rather than the 256-colour cube. Set once from the environment.
  void set_truecolor(bool t) { truecolor_ = t; }

 private:
  int w_ = 0, h_ = 0;
  std::vector<Cell> cur_, prev_;
  bool dirty_all_ = true;
  bool truecolor_ = true;
};

// Mode switching, size, and the keyboard. One instance; the destructor restores, and so does the
// signal path: a process that dies inside the alternate screen leaves a terminal with no cursor.
class Terminal {
 public:
  ~Terminal();

  // False when stdout is not a terminal — the caller then has no dashboard, which is the rule that
  // keeps piped output byte-identical to what it has always been.
  static bool is_tty();
  static bool truecolor();

  bool enter();
  void leave();

  // Current size, or 80x24 when the ioctl fails (a pty that has not been sized yet).
  void size(int* w, int* h) const;
  // True once per SIGWINCH; clears the flag.
  bool take_resize();
  // A quit was requested by SIGINT/SIGTERM. Sticky.
  static bool quit_requested();
  // Next key, or -1 when none is buffered. Only ever non-negative when stdin is a terminal.
  int read_key();

 private:
  bool active_ = false;
  bool raw_ = false;
};

}  // namespace aff::ui
