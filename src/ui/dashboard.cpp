#include "ui/dashboard.h"

#include "ui/log.h"
#include "ui/term.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace aff::ui {
namespace {

// ---- palette ------------------------------------------------------------------------------------
//
// Dark, low-chroma frame with the colour reserved for quantities. Three families carry all the
// meaning: green is VRAM-resident, amber is host RAM, red is disk — the same order as the tiers,
// so a panel is read the same way as the heat plane without a legend.
constexpr Rgb kBg      = rgb(0x0d, 0x11, 0x17);
constexpr Rgb kPanel   = rgb(0x11, 0x16, 0x1e);
constexpr Rgb kFrame   = rgb(0x27, 0x31, 0x3d);
constexpr Rgb kFrameHi = rgb(0x3d, 0x4c, 0x5e);
constexpr Rgb kText    = rgb(0xc9, 0xd4, 0xe3);
constexpr Rgb kMuted   = rgb(0x6b, 0x7a, 0x8d);
constexpr Rgb kFaint   = rgb(0x44, 0x50, 0x5e);
constexpr Rgb kCyan    = rgb(0x4d, 0xd0, 0xe1);
constexpr Rgb kAmber   = rgb(0xff, 0xb7, 0x4d);
constexpr Rgb kGreen   = rgb(0x66, 0xbb, 0x6a);
constexpr Rgb kRed     = rgb(0xef, 0x53, 0x50);
constexpr Rgb kViolet  = rgb(0xb3, 0x88, 0xff);

// Bottom-to-top gradient for the meters and the graphs.
constexpr Rgb kMeter[3] = {rgb(0x2e, 0x7d, 0xff), rgb(0x4d, 0xd0, 0xe1), rgb(0x9c, 0xff, 0xa0)};
constexpr Rgb kHot[4]   = {rgb(0x1e, 0x88, 0xe5), rgb(0x66, 0xbb, 0x6a), rgb(0xff, 0xb7, 0x4d),
                           rgb(0xef, 0x53, 0x50)};

// One ramp per tier: darkest is "this expert exists and is idle", brightest is "it is being used
// constantly". A bright amber cell is the one that costs a link read on every block.
//
// Five stops, not three, and the three ramps share a lightness envelope so the tiers differ by hue
// and the heat reads the same way in each. Three stops put the whole middle of the distribution on
// one interpolation leg, which is where the desaturated ochre and sage came from: most of an eleven
// thousand cell plane rendered as a wash rather than as a colour. The dark stop of each ramp keeps
// its tier's hue — an idle pooled expert still has to be findable — but sits close enough to the
// panel that the eye reads the plane's shape first and its temperature second.
constexpr int kTierStops   = 5;
constexpr Rgb kVramRamp[kTierStops] = {rgb(0x14, 0x20, 0x1a), rgb(0x1a, 0x33, 0x26),
                                       rgb(0x20, 0x6b, 0x41), rgb(0x35, 0xb5, 0x66),
                                       rgb(0x7d, 0xf0, 0xa8)};
constexpr Rgb kPoolRamp[kTierStops] = {rgb(0x24, 0x1a, 0x12), rgb(0x40, 0x26, 0x12),
                                       rgb(0x8a, 0x4a, 0x10), rgb(0xe0, 0x91, 0x2a),
                                       rgb(0xff, 0xd7, 0x82)};
constexpr Rgb kSsdRamp[kTierStops]  = {rgb(0x2e, 0x10, 0x16), rgb(0x7a, 0x18, 0x30),
                                       rgb(0xd0, 0x2a, 0x4a), rgb(0xff, 0x64, 0x78),
                                       rgb(0xff, 0xc0, 0xc8)};

// ---- small formatting ---------------------------------------------------------------------------

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
  char b[256];
  va_list ap;
  va_start(ap, f);
  const int n = std::vsnprintf(b, sizeof(b), f, ap);
  va_end(ap);
  return n > 0 ? std::string(b, (size_t)std::min<int>(n, sizeof(b) - 1)) : std::string();
}

std::string commas(uint64_t v) {
  std::string s = std::to_string(v);
  for (int i = (int)s.size() - 3; i > 0; i -= 3) s.insert((size_t)i, " ");
  return s;
}

std::string hhmmss(double sec) {
  if (sec < 0) sec = 0;
  const int t = (int)sec;
  return fmt("%02d:%02d:%02d", t / 3600, t / 60 % 60, t % 60);
}

std::string rate_bytes(double bps) {
  if (bps >= 1e9) return fmt("%.2f GB/s", bps / 1e9);
  if (bps >= 1e6) return fmt("%.0f MB/s", bps / 1e6);
  if (bps >= 1e3) return fmt("%.0f kB/s", bps / 1e3);
  return fmt("%.0f B/s", bps);
}

// ---- history ------------------------------------------------------------------------------------

class Ring {
 public:
  explicit Ring(size_t n = 512) : v_(n, 0.0f) {}
  void push(float x) {
    v_[head_] = x;
    head_ = (head_ + 1) % v_.size();
    if (n_ < v_.size()) ++n_;
  }
  size_t size() const { return n_; }
  // Most recent `k` samples, oldest first.
  void tail(size_t k, std::vector<float>* out) const {
    if (k > n_) k = n_;
    out->resize(k);
    for (size_t i = 0; i < k; ++i)
      out->at(i) = v_[(head_ + v_.size() - k + i) % v_.size()];
  }
  float max() const {
    float m = 0;
    for (size_t i = 0; i < n_; ++i) m = std::max(m, v_[i]);
    return m;
  }
  float last() const { return n_ ? v_[(head_ + v_.size() - 1) % v_.size()] : 0.0f; }

 private:
  std::vector<float> v_;
  size_t head_ = 0, n_ = 0;
};

// ---- drawing primitives -------------------------------------------------------------------------

void box(Canvas& c, int x, int y, int w, int h, std::string_view title, Rgb accent,
         std::string_view right = {}) {
  if (w < 2 || h < 2) return;
  const Rgb f = kFrame;
  c.put(x, y, U'╭', f, kPanel);
  c.put(x + w - 1, y, U'╮', f, kPanel);
  c.put(x, y + h - 1, U'╰', f, kPanel);
  c.put(x + w - 1, y + h - 1, U'╯', f, kPanel);
  for (int i = 1; i < w - 1; ++i) {
    c.put(x + i, y, U'─', f, kPanel);
    c.put(x + i, y + h - 1, U'─', f, kPanel);
  }
  for (int j = 1; j < h - 1; ++j) {
    c.put(x, y + j, U'│', f, kPanel);
    c.put(x + w - 1, y + j, U'│', f, kPanel);
    for (int i = 1; i < w - 1; ++i) c.put(x + i, y + j, U' ', kText, kPanel);
  }
  if (!title.empty() && w > 6) {
    int tx = x + 2;
    c.put(tx - 1, y, U' ', f, kPanel);
    tx = c.text_clip(tx, y, w - 6, title, accent, kPanel, kBold);
    c.put(tx, y, U' ', f, kPanel);
  }
  if (!right.empty() && w > (int)right.size() + 8) {
    const int rx = x + w - 2 - (int)right.size();
    c.put(rx - 1, y, U' ', f, kPanel);
    c.text_clip(rx, y, (int)right.size(), right, kMuted, kPanel);
    c.put(x + w - 2, y, U' ', f, kPanel);
  }
}

// A meter with eighth-block resolution in the final cell, so a 20-cell bar reads 160 steps rather
// than 20 — at these widths the difference between 41% and 44% residency is otherwise invisible.
void hbar(Canvas& c, int x, int y, int w, double frac, const Rgb* stops, int nstops,
          Rgb track = kFaint) {
  if (w <= 0) return;
  frac = frac < 0 ? 0 : frac > 1 ? 1 : frac;
  const double cells = frac * (double)w;
  const int full = (int)cells;
  const int part = (int)((cells - (double)full) * 8.0);
  static const char32_t kEighth[8] = {U' ', U'▏', U'▎', U'▍',
                                      U'▌', U'▋', U'▊', U'▉'};
  for (int i = 0; i < w; ++i) {
    const Rgb col = ramp(stops, nstops, w > 1 ? (double)i / (double)(w - 1) : 0.0);
    if (i < full) c.put(x + i, y, U'█', col, kPanel);
    else if (i == full && part > 0) c.put(x + i, y, kEighth[part], col, kPanel);
    else c.put(x + i, y, U'░', track, kPanel);
  }
}

// Braille area graph: two data columns and four rows of dots per cell, filled from the baseline so
// the shape reads at three rows high. `vmax` of zero autoscales to the data.
void graph(Canvas& c, int x, int y, int w, int h, const std::vector<float>& v, float vmax,
           const Rgb* stops, int nstops) {
  if (w <= 0 || h <= 0) return;
  const int cols = w * 2, rows = h * 4;
  if (vmax <= 0) { for (float f : v) vmax = std::max(vmax, f); }
  if (vmax <= 0) vmax = 1;

  // Right-aligned: the newest sample is always in the last column, so the graph does not slide
  // sideways while the history is still filling.
  std::vector<int> lvl((size_t)cols, -1);
  const int have = (int)v.size();
  for (int i = 0; i < have && i < cols; ++i) {
    const float s = v[(size_t)(have - 1 - i)];
    const double t = std::max(0.0, std::min(1.0, (double)s / (double)vmax));
    lvl[(size_t)(cols - 1 - i)] = (int)std::lround(t * (double)rows);
  }
  static const uint8_t kDot[4][2] = {{0x01, 0x08}, {0x02, 0x10}, {0x04, 0x20}, {0x40, 0x80}};
  for (int cy = 0; cy < h; ++cy) {
    // Vertical gradient rather than per-sample colour: the eye reads height off the colour before
    // it reads it off the shape, which is the whole reason a three-row graph works at all.
    const Rgb col = ramp(stops, nstops, h > 1 ? (double)(h - 1 - cy) / (double)(h - 1) : 1.0);
    for (int cx = 0; cx < w; ++cx) {
      uint8_t mask = 0;
      for (int sub = 0; sub < 2; ++sub) {
        const int L = lvl[(size_t)(cx * 2 + sub)];
        if (L < 0) continue;
        for (int r = 0; r < 4; ++r) {
          const int dot_from_bottom = rows - (cy * 4 + r);
          if (dot_from_bottom <= L) mask |= kDot[r][sub];
        }
      }
      if (mask) c.put(x + cx, y + cy, (char32_t)(0x2800 + mask), col, kPanel);
    }
  }
}

// The expert plane. Layers down, experts across, two layers per text row via the upper half block —
// 43 layers in 22 rows, which fits a pane that also has to hold everything else.
void heatplane(Canvas& c, int x, int y, int w, int h, const std::vector<float>& heat,
               const std::vector<uint8_t>& tier, int L, int E) {
  if (w <= 0 || h <= 0 || L <= 0 || E <= 0) return;
  if ((int)tier.size() < L * E) return;
  // Static placement keeps no heat plane, and there is nothing dishonest about saying so by drawing
  // every cell at one brightness: the map is then purely where the bytes are, which is the whole of
  // what that configuration decides.
  const bool has_heat = (int)heat.size() >= L * E;

  float hmax = 0;
  for (float f : heat) hmax = std::max(hmax, f);
  if (hmax <= 0) hmax = 1;

  const int half_rows = h * 2;
  auto cell = [&](int lrow, int col, Rgb* out) -> bool {
    // Each screen cell covers a rectangle of the plane. Heat aggregates by MAX and the tier by the
    // most expensive one present: a cell holding one pooled expert among fifteen resident ones is
    // a cell that costs a link read, and an average would hide exactly that.
    const int l0 = (int)((int64_t)lrow * L / half_rows);
    int l1 = (int)((int64_t)(lrow + 1) * L / half_rows);
    if (l1 <= l0) l1 = l0 + 1;
    if (l0 >= L) return false;
    const int e0 = (int)((int64_t)col * E / w);
    int e1 = (int)((int64_t)(col + 1) * E / w);
    if (e1 <= e0) e1 = e0 + 1;
    float best = 0;
    uint8_t worst = kTierVram;
    for (int l = l0; l < l1 && l < L; ++l)
      for (int e = e0; e < e1 && e < E; ++e) {
        const size_t k = (size_t)l * (size_t)E + (size_t)e;
        if (has_heat) best = std::max(best, heat[k]);
        worst = std::max(worst, tier[k]);
      }
    // Compressed rather than linear: the heat distribution is long-tailed, and on a linear ramp
    // every expert but the few hottest renders as the darkest step. The exponent is above a square
    // root because five stops already resolve the low end — sqrt on top of them lifts the whole
    // plane into the bright half and throws away the contrast the stops were added for.
    const double t =
        has_heat ? std::pow(std::min(1.0, (double)best / (double)hmax), 0.7) : 0.5;
    const Rgb* r = worst == kTierSsd ? kSsdRamp : worst == kTierPool ? kPoolRamp : kVramRamp;
    *out = ramp(r, kTierStops, t);
    return true;
  };

  for (int cy = 0; cy < h; ++cy)
    for (int cx = 0; cx < w; ++cx) {
      Rgb up = kPanel, dn = kPanel;
      const bool a = cell(cy * 2, cx, &up);
      const bool b = cell(cy * 2 + 1, cx, &dn);
      if (!a && !b) continue;
      c.put(x + cx, y + cy, U'▀', a ? up : kPanel, b ? dn : kPanel);
    }
}

// ---- the console pane ---------------------------------------------------------------------------

struct Line {
  Level lvl = Level::Info;
  std::string s;
};

Rgb level_colour(Level l) {
  switch (l) {
    case Level::Note: return kMuted;
    case Level::Warn: return kAmber;
    case Level::Error: return kRed;
    case Level::Token: return kText;
    default: return rgb(0x93, 0xa4, 0xb8);
  }
}

}  // namespace

// ---- mode ---------------------------------------------------------------------------------------

bool parse_ui_mode(std::string_view s, UiMode* out, std::string* err) {
  if (s == "auto") { *out = UiMode::Auto; return true; }
  if (s == "dash" || s == "dashboard") { *out = UiMode::Dash; return true; }
  if (s == "plain" || s == "off") { *out = UiMode::Plain; return true; }
  if (err) *err = "unknown --ui '" + std::string(s) + "' (want auto, dash or plain)";
  return false;
}

const char* stage_name(Stage s) {
  switch (s) {
    case Stage::Init:        return "starting";
    case Stage::LoadDense:   return "loading dense weights";
    case Stage::LoadExperts: return "placing experts";
    case Stage::LoadDraft:   return "loading draft";
    case Stage::LoadPool:    return "filling host pool";
    case Stage::Ready:       return "ready";
    case Stage::Prefill:     return "prefill";
    case Stage::Decode:      return "decode";
    case Stage::Serve:       return "serving";
    case Stage::Done:        return "done";
  }
  return "?";
}

// ---- impl ---------------------------------------------------------------------------------------

struct Dashboard::Impl {
  Terminal term;
  Canvas canvas;
  Probe probe;
  std::vector<DeviceInfo> pending;
  std::vector<DeviceSample> dsamp;

  std::thread th;
  std::atomic<bool> stop{false};
  std::atomic<bool> quit{false};
  std::mutex log_m;
  std::deque<Line> log;
  bool token_open = false;

  Snapshot snap;
  uint64_t snap_seq = 0;
  std::chrono::steady_clock::time_point t0;

  Ring g_tok, g_block, g_link;
  std::vector<Ring> g_busy;
  uint64_t last_link_bytes = 0;
  double last_link_t = 0;
  int tab = 0;   // 0 experts, 1 layers, 2 timing, 3 link
  std::string frame;

  void push(Level l, std::string_view s) {
    std::lock_guard<std::mutex> lk(log_m);
    if (l == Level::Token) {
      // Tokens arrive as fragments and must appear as fragments — waiting for a newline would make
      // the pane lag the model by a whole sentence.
      size_t i = 0;
      while (i <= s.size()) {
        const size_t nl = s.find('\n', i);
        const std::string_view part = s.substr(i, nl == std::string_view::npos ? s.size() - i
                                                                              : nl - i);
        if (!token_open) { log.push_back({Level::Token, std::string()}); token_open = true; }
        log.back().s.append(part);
        if (nl == std::string_view::npos) break;
        token_open = false;
        i = nl + 1;
      }
    } else {
      token_open = false;
      // Engine banners are multi-line printf calls; one screen line each keeps the wrapping honest.
      size_t i = 0;
      while (i <= s.size()) {
        const size_t nl = s.find('\n', i);
        const std::string_view part = s.substr(i, nl == std::string_view::npos ? s.size() - i
                                                                              : nl - i);
        if (!part.empty()) log.push_back({l, std::string(part)});
        if (nl == std::string_view::npos) break;
        i = nl + 1;
      }
    }
    while (log.size() > 600) log.pop_front();
  }

  void run(Bus* bus);
  void render(const Snapshot& s, double up);
  void panel_header(const Snapshot& s, int x, int y, int w, double up);
  void panel_device(int x, int y, int w, int h, size_t i);
  void panel_host(const Snapshot& s, int x, int y, int w, int h);
  void panel_speed(const Snapshot& s, int x, int y, int w, int h);
  void panel_spec(const Snapshot& s, int x, int y, int w, int h);
  void panel_tab(const Snapshot& s, int x, int y, int w, int h);
  void panel_console(int x, int y, int w, int h);
  void panel_footer(int y, int w);
};

Dashboard::Dashboard() : im_(new Impl) {}
Dashboard::~Dashboard() { stop(); }

void Dashboard::add_device(const DeviceInfo& d) { im_->pending.push_back(d); }

bool Dashboard::quit_requested() const {
  return im_->quit.load(std::memory_order_relaxed) || Terminal::quit_requested();
}

void Dashboard::stage(Stage s, std::string detail, double frac) {
  bus_.publish_sync([&](Snapshot& w) {
    w.stage = s;
    w.detail = std::move(detail);
    w.stage_frac = frac;
  });
}

bool Dashboard::start(UiMode mode) {
  if (running_.load()) return true;
  if (mode == UiMode::Plain) return false;
  if (!Terminal::is_tty()) return false;
  if (!im_->term.enter()) return false;

  for (const DeviceInfo& d : im_->pending) im_->probe.add_device(d);
  im_->g_busy.assign(im_->probe.devices(), Ring(512));
  im_->canvas.set_truecolor(Terminal::truecolor());
  im_->t0 = std::chrono::steady_clock::now();
  running_.store(true);

  set_sink([](void* ctx, Level l, std::string_view s) {
    static_cast<Impl*>(ctx)->push(l, s);
  }, im_.get());
  set_teardown([] { dashboard().stop(); });

  im_->th = std::thread([this] { im_->run(&bus_); });
  return true;
}

void Dashboard::stop() {
  if (!running_.exchange(false)) return;
  set_sink(nullptr, nullptr);
  im_->stop.store(true);
  if (im_->th.joinable()) im_->th.join();
  im_->term.leave();
  // The terminal is back; anything the pane still held would otherwise be lost with the alternate
  // screen, and the load banners are exactly what an operator wants to keep after a run. Each line
  // goes back to the stream it came from, so `2>/dev/null` still selects the same half it always
  // did — the dashboard changes where output is drawn, never which stream it belongs to.
  std::lock_guard<std::mutex> lk(im_->log_m);
  for (const Line& l : im_->log) {
    if (l.lvl == Level::Token) continue;
    std::FILE* to = l.lvl == Level::Info ? stdout : stderr;
    std::fprintf(to, "%s\n", l.s.c_str());
  }
  std::fflush(stdout);
  im_->log.clear();
}

Dashboard& dashboard() {
  // Never destroyed: ui::log may be called from a static destructor in another translation unit,
  // and the sink pointer has to stay valid for as long as anything can log.
  static Dashboard* d = new Dashboard();
  return *d;
}

// ---- the loop -----------------------------------------------------------------------------------

void Dashboard::Impl::run(Bus* bus) {
  using clock = std::chrono::steady_clock;
  // Fast enough that gpu_busy_percent is an average rather than one instantaneous bit, slow enough
  // that a frame is not being drawn while the previous one is still on the wire.
  constexpr auto kProbe = std::chrono::milliseconds(20);
  constexpr auto kFrameEvery = std::chrono::milliseconds(100);

  auto next_frame = clock::now();
  while (!stop.load(std::memory_order_relaxed)) {
    probe.tick();
    std::this_thread::sleep_for(kProbe);
    if (clock::now() < next_frame) continue;
    next_frame += kFrameEvery;
    if (next_frame < clock::now()) next_frame = clock::now() + kFrameEvery;

    // Bounded: a key held down, or an escape sequence arriving as its bytes, must not let the
    // reader starve the frame it is being read for.
    for (int guard = 0, k; guard < 32 && (k = term.read_key()) >= 0; ++guard) {
      if (k == 'q' || k == 'Q') quit.store(true);
      else if (k >= '1' && k <= '4') tab = k - '1';
      else if (k == '\t') tab = (tab + 1) % 4;
    }
    if (term.take_resize()) canvas.invalidate();

    probe.roll(&dsamp);
    for (size_t i = 0; i < dsamp.size() && i < g_busy.size(); ++i)
      g_busy[i].push(dsamp[i].busy > 0 ? (float)dsamp[i].busy : 0.0f);

    uint64_t seq = 0;
    bus->read(&snap, &seq);
    const double up = std::chrono::duration<double>(clock::now() - t0).count();

    if (seq != snap_seq) {
      snap_seq = seq;
      g_tok.push((float)snap.decode_tok_s);
      g_block.push((float)snap.ms_per_block);
    }
    // The link rate is a difference, so it is computed here rather than published: the engine
    // counts bytes, and a rate needs two observations the engine has no reason to keep.
    if (up > last_link_t + 0.2) {
      const uint64_t b = snap.h2d_bytes + snap.d2h_bytes;
      const double dt = up - last_link_t;
      if (last_link_bytes && b >= last_link_bytes)
        g_link.push((float)((double)(b - last_link_bytes) / dt));
      else if (last_link_bytes)
        g_link.push(0.0f);
      last_link_bytes = b;
      last_link_t = up;
    }

    render(snap, up);
    canvas.flush(&frame);
    // One write for the whole frame. Two is how a terminal ends up showing half of one.
    if (!frame.empty()) {
      std::fwrite(frame.data(), 1, frame.size(), stdout);
      std::fflush(stdout);
    }
  }
}

// ---- layout -------------------------------------------------------------------------------------

void Dashboard::Impl::render(const Snapshot& s, double up) {
  int W = 0, H = 0;
  term.size(&W, &H);
  canvas.resize(W, H);
  canvas.clear(kBg);

  if (W < 72 || H < 18) {
    const std::string m = fmt("terminal is %dx%d; the dashboard needs 72x18", W, H);
    canvas.text(std::max(0, (W - (int)m.size()) / 2), H / 2, m, kAmber, kBg);
    return;
  }

  int y = 0;
  panel_header(s, 0, y, W, up);
  y += 4;

  // Devices side by side while each still gets 30 columns; below that they stack, and below that
  // the host panel goes to a single line inside the last card's box.
  const int ndev = (int)std::max<size_t>(probe.devices(), 0);
  const int cards = ndev + 1;                       // +1 for the host
  const int dev_h = 6;
  // Below this each card's panel is narrower than its own meters, so they stack instead. The
  // threshold is the width at which the busy bar still has room to be read as a bar.
  if (cards > 0 && W / cards >= 34) {
    const int cw = W / cards;
    for (int i = 0; i < ndev; ++i) panel_device(i * cw, y, cw, dev_h, (size_t)i);
    panel_host(s, ndev * cw, y, W - ndev * cw, dev_h);
  } else {
    for (int i = 0; i < ndev; ++i) panel_device(0, y + i * dev_h, W, dev_h, (size_t)i);
    panel_host(s, 0, y + ndev * dev_h, W, dev_h);
    y += (ndev) * dev_h;
  }
  y += dev_h;

  const int rest = H - y - 1;
  int graph_h = 0, console_h = 0;
  if (rest >= 26) { graph_h = 9; console_h = 8; }
  else if (rest >= 20) { graph_h = 8; console_h = 6; }
  else if (rest >= 14) { graph_h = 0; console_h = 6; }
  else { graph_h = 0; console_h = std::max(4, rest - 6); }

  if (graph_h) {
    const int half = W / 2;
    panel_speed(s, 0, y, half, graph_h);
    panel_spec(s, half, y, W - half, graph_h);
    y += graph_h;
  }

  const int tab_h = H - 1 - console_h - y;
  if (tab_h >= 4) {
    panel_tab(s, 0, y, W, tab_h);
    y += tab_h;
  }
  panel_console(0, y, W, H - 1 - y);
  panel_footer(H - 1, W);
}

// ---- panels -------------------------------------------------------------------------------------

void Dashboard::Impl::panel_header(const Snapshot& s, int x, int y, int w, double up) {
  box(canvas, x, y, w, 4, "AFFINITY", kCyan, hhmmss(up));

  std::string id = s.model.empty() ? std::string("no container") : s.model;
  if (s.n_layer) {
    id += fmt("  ·  %uL × %uE top-%u", s.n_layer, s.n_expert, s.top_k);
    if (!s.quant.empty()) id += "  ·  " + s.quant;
    if (s.ranks) id += fmt("  ·  TP%u", s.ranks);
  }
  canvas.text_clip(x + 2, y + 1, w - 4, id, kText, kPanel);

  // Line two is the state of the run, and its left third is the stage — the one field that says
  // whether a blank throughput number means "not yet" or "stopped".
  int cx = x + 2;
  const bool busy = s.stage != Stage::Ready && s.stage != Stage::Done;
  cx = canvas.text(cx, y + 2, stage_name(s.stage), busy ? kAmber : kGreen, kPanel, kBold);
  if (!s.detail.empty()) {
    cx = canvas.text(cx, y + 2, " ", kText, kPanel);
    cx = canvas.text_clip(cx, y + 2, 28, s.detail, kMuted, kPanel);
  }
  if (s.stage_frac >= 0 && cx + 14 < x + w - 2) {
    hbar(canvas, cx + 1, y + 2, 12, s.stage_frac, kMeter, 3);
    cx += 14;
  }

  auto stat = [&](int at, const char* label, const std::string& v, Rgb col) {
    if (at + 8 > x + w - 2) return;
    int p = canvas.text(at, y + 2, label, kMuted, kPanel);
    canvas.text_clip(p + 1, y + 2, x + w - 2 - p, v, col, kPanel, kBold);
  };
  const int right = x + w - 2;
  int slot = std::max(cx + 2, right - 62);
  stat(slot, "tok/s", s.decode_tok_s > 0 ? fmt("%.2f", s.decode_tok_s)
                                         : (s.prefill_tok_s > 0 ? fmt("%.0f", s.prefill_tok_s)
                                                                : std::string("—")),
       s.decode_tok_s > 0 ? kCyan : kMuted);
  slot += 14;
  stat(slot, "ms/blk", s.ms_per_block > 0 ? fmt("%.1f", s.ms_per_block) : std::string("—"),
       kViolet);
  slot += 14;
  stat(slot, "accept", s.drafted ? fmt("%.1f%%", 100.0 * (double)s.accepted / (double)s.drafted)
                                 : std::string("—"),
       kGreen);
  slot += 15;
  stat(slot, "ctx", s.kv_capacity ? fmt("%s/%s", commas(s.pos).c_str(),
                                        commas(s.kv_capacity).c_str())
                                  : commas(s.pos), kAmber);
}

void Dashboard::Impl::panel_device(int x, int y, int w, int h, size_t i) {
  const DeviceInfo& d = probe.info(i);
  DeviceSample v{};
  if (i < dsamp.size()) v = dsamp[i];
  box(canvas, x, y, w, h, fmt("GPU%zu %s", i, d.name.c_str()), kCyan, d.arch);
  const int ix = x + 2, iw = w - 4;
  if (iw < 14) return;

  // A graph AND a meter for the same quantity, because they answer different questions: the meter
  // is "is it busy now", the graph is "has it been idle in a way a mean would hide".
  const int gw = std::min(std::max(iw / 4, 6), 16);
  std::vector<float> hist;
  if (i < g_busy.size()) g_busy[i].tail((size_t)gw * 2, &hist);
  graph(canvas, ix, y + 1, gw, 2, hist, 100.0f, kMeter, 3);

  const int rx = x + w - 2;              // one past the last writable interior column
  const int lx = ix + gw + 1;
  const int bx = lx + 5;
  const int bw = std::max(4, rx - bx - 12);

  canvas.text(lx, y + 1, "busy", kMuted, kPanel);
  if (v.busy >= 0) {
    hbar(canvas, bx, y + 1, bw, v.busy / 100.0, kMeter, 3);
    canvas.text_clip(bx + bw + 1, y + 1, rx - bx - bw - 1, fmt("%3.0f%%", v.busy), kText, kPanel,
                     kBold);
  } else {
    canvas.text_clip(bx, y + 1, rx - bx, "no amdgpu sysfs node", kMuted, kPanel);
  }

  const double used = v.vram_used_gib >= 0 ? v.vram_used_gib : 0.0;
  const double tot = d.vram_total_gib > 0 ? d.vram_total_gib : 1.0;
  static const Rgb kVram[3] = {kGreen, kAmber, kRed};
  canvas.text(lx, y + 2, "vram", kMuted, kPanel);
  hbar(canvas, bx, y + 2, bw, used / tot, kVram, 3);
  canvas.text_clip(bx + bw + 1, y + 2, rx - bx - bw - 1, fmt("%.1f/%.0f", used, d.vram_total_gib),
                   kText, kPanel);

  std::string foot;
  auto add = [&](const std::string& piece) {
    if (!foot.empty()) foot += "  ·  ";
    foot += piece;
  };
  if (v.sclk_mhz > 0) add(fmt("%.0f MHz", v.sclk_mhz));
  if (v.temp_c > 0) add(fmt("%.0f °C", v.temp_c));
  if (v.power_w > 0) add(fmt("%.0f W", v.power_w));
  if (d.cus) add(fmt("%d CU", d.cus));
  if (h >= 5) canvas.text_clip(ix, y + h - 2, iw, foot, kMuted, kPanel);
}

void Dashboard::Impl::panel_host(const Snapshot& s, int x, int y, int w, int h) {
  box(canvas, x, y, w, h, "Host", kAmber);
  const int ix = x + 2, iw = w - 4;
  if (iw < 10) return;
  const HostSample& hs = probe.host();
  const int bw = std::max(4, iw - 20);

  static const Rgb kPool[3] = {kAmber, kAmber, kRed};
  const double tot = hs.mem_total_gib > 0 ? hs.mem_total_gib : 1.0;
  canvas.text(ix, y + 1, "pool", kMuted, kPanel);
  hbar(canvas, ix + 5, y + 1, bw, s.pool_gib / tot, kPool, 3);
  canvas.text_clip(ix + 6 + bw, y + 1, iw - bw - 6, fmt("%.1f GiB", s.pool_gib), kText, kPanel);

  static const Rgb kRss[3] = {kGreen, kAmber, kRed};
  canvas.text(ix, y + 2, "rss ", kMuted, kPanel);
  hbar(canvas, ix + 5, y + 2, bw, hs.rss_gib / tot, kRss, 3);
  canvas.text_clip(ix + 6 + bw, y + 2, iw - bw - 6, fmt("%.1f GiB", hs.rss_gib), kText, kPanel);

  canvas.text_clip(ix, y + 3, iw,
                   fmt("%.1f of %.0f GiB free  ·  peak %.1f", hs.mem_avail_gib,
                       hs.mem_total_gib, hs.rss_peak_gib), kMuted, kPanel);
}

void Dashboard::Impl::panel_speed(const Snapshot& s, int x, int y, int w, int h) {
  const bool dec = s.decode_tok_s > 0 || s.stage == Stage::Decode;
  const float peak = std::max(g_tok.max(), 0.001f);
  box(canvas, x, y, w, h, dec ? "Decode" : "Throughput", kCyan,
      dec ? fmt("peak %.2f tok/s", peak) : std::string());
  const int ix = x + 2, iw = w - 4, gh = h - 4;
  if (iw < 10 || gh < 1) return;

  std::vector<float> hist;
  g_tok.tail((size_t)iw * 2, &hist);
  graph(canvas, ix, y + 1, iw, gh, hist, 0.0f, kMeter, 3);

  const int ly = y + 1 + gh;
  int cx = canvas.text(ix, ly, "now ", kMuted, kPanel);
  cx = canvas.text(cx, ly, s.decode_tok_s > 0 ? fmt("%.2f tok/s", s.decode_tok_s)
                                              : std::string("—"), kCyan, kPanel, kBold);
  cx = canvas.text(cx + 2, ly, "block ", kMuted, kPanel);
  cx = canvas.text(cx, ly, s.ms_per_block > 0 ? fmt("%.1f ms", s.ms_per_block)
                                              : std::string("—"), kViolet, kPanel, kBold);
  if (s.prefill_tok_s > 0) {
    cx = canvas.text(cx + 2, ly, "prefill ", kMuted, kPanel);
    canvas.text_clip(cx, ly, x + w - 2 - cx, fmt("%.0f tok/s", s.prefill_tok_s), kAmber, kPanel);
  }
  canvas.text_clip(ix, ly + 1, iw,
                   fmt("%s generated of %s  ·  %s prompt", commas(s.generated).c_str(),
                       s.target ? commas(s.target).c_str() : "∞",
                       commas(s.prompt_tokens).c_str()), kMuted, kPanel);
}

void Dashboard::Impl::panel_spec(const Snapshot& s, int x, int y, int w, int h) {
  const double acc = s.drafted ? (double)s.accepted / (double)s.drafted : 0.0;
  box(canvas, x, y, w, h, "Speculation", kGreen,
      s.spec_width ? fmt("block %u", s.spec_width) : std::string("off"));
  const int ix = x + 2, iw = w - 4;
  if (iw < 12) return;

  if (!s.spec_width) {
    canvas.text(ix, y + 2, "draft disabled — every token is a full forward pass", kMuted,
                kPanel);
    return;
  }
  const int bw = std::max(6, iw - 16);
  // Marginal, not cumulative: acceptance stops at the first miss, so position 4's share of the
  // total is conditioned on 0..3 being right while this is not. It is the number that says whether
  // the block is too long.
  const uint32_t n = std::min<uint32_t>(s.pos_valid, (uint32_t)std::max(0, h - 4));
  for (uint32_t j = 0; j < n; ++j) {
    canvas.text(ix, y + 1 + (int)j, fmt("pos %u", j), kMuted, kPanel);
    static const Rgb kAcc[3] = {kRed, kAmber, kGreen};
    hbar(canvas, ix + 6, y + 1 + (int)j, bw, s.pos_rate[j], kAcc, 3);
    canvas.text(ix + 7 + bw, y + 1 + (int)j, fmt("%3.0f%%", 100.0 * s.pos_rate[j]), kText, kPanel);
  }
  const int ly = y + h - 2;
  canvas.text_clip(ix, ly, iw,
                   fmt("%s blocks  ·  %s drafted  ·  %s accepted (%.1f%%, %.2f a block)",
                       commas(s.blocks).c_str(), commas(s.drafted).c_str(),
                       commas(s.accepted).c_str(), 100.0 * acc,
                       s.blocks ? (double)(s.accepted + s.blocks) / (double)s.blocks : 0.0),
                   kMuted, kPanel);
}

void Dashboard::Impl::panel_tab(const Snapshot& s, int x, int y, int w, int h) {
  static const char* kNames[4] = {"Experts", "Layers", "Timing", "Link"};
  std::string title;
  for (int i = 0; i < 4; ++i) {
    title += i ? "  " : "";
    title += fmt("%d %s", i + 1, kNames[i]);
  }
  const int ix = x + 2, iw = w - 4, ih = h - 2;
  box(canvas, x, y, w, h, "", kCyan);
  // The tab strip is drawn into the frame rather than under it: it is chrome, and a row of it would
  // cost an eighth of the pane at the heights this runs at.
  int tx = x + 2;
  for (int i = 0; i < 4; ++i) {
    const bool on = i == tab;
    canvas.put(tx++, y, U' ', kFrame, kPanel);
    tx = canvas.text(tx, y, fmt("%d", i + 1), on ? kBg : kMuted, on ? kCyan : kPanel, kBold);
    canvas.put(tx++, y, U' ', on ? kBg : kMuted, on ? kCyan : kPanel);
    tx = canvas.text(tx, y, kNames[i], on ? kBg : kMuted, on ? kCyan : kPanel, on ? kBold : 0);
    canvas.put(tx++, y, U' ', kFrame, on ? kCyan : kPanel);
  }
  if (iw < 12 || ih < 2) return;

  if (tab == 0) {
    if (s.tier.empty() || !s.n_layer) {
      canvas.text(ix, y + 1, "placement has not reported a plane yet", kMuted, kPanel);
      return;
    }
    const int legend = 1;
    heatplane(canvas, ix, y + 1, iw, ih - legend, s.heat, s.tier, (int)s.n_layer, (int)s.n_expert);
    const int ly = y + ih;
    int cx = canvas.text(ix, ly, "layer 0", kMuted, kPanel);
    cx = canvas.text(cx + 1, ly, "→", kFaint, kPanel);
    cx = canvas.text(cx + 1, ly, fmt("%u", s.n_layer - 1), kMuted, kPanel);
    // Each swatch is the tier's ramp in miniature rather than one block of its brightest stop: the
    // plane encodes two things in one colour, and a three-step swatch says so without a sentence.
    auto swatch = [&](const Rgb* r, const char* label) {
      for (int i = 0; i < 3; ++i)
        cx = canvas.text(cx, ly, "█", ramp(r, kTierStops, 0.15 + 0.425 * i), kPanel);
      cx = canvas.text(cx, ly, label, kMuted, kPanel);
    };
    cx += 3;
    swatch(kVramRamp, " vram  ");
    swatch(kPoolRamp, " host ram  ");
    swatch(kSsdRamp, " ssd");
    canvas.text_clip(cx + 3, ly, x + w - 2 - cx - 3,
                     s.heat.empty() ? "static placement: no heat plane, so only the tier is shown"
                                    : "brightness is how often the router picked it",
                     kFaint, kPanel);
    return;
  }

  if (tab == 1) {
    if (s.resident_per_layer.empty()) {
      canvas.text(ix, y + 1, "placement has not reported a per-layer census", kMuted, kPanel);
      return;
    }
    // Residency per layer, which is the distribution the placement engine exists to flatten: a
    // layer starved of slots misses on nearly every token however good the ranking inside it is.
    //
    // A bar and a percentage do not say what is being counted, and this pane has no other context to
    // infer it from, so the units are spelled out on a header row and each bar is labelled with the
    // count it draws rather than with a share of an unnamed whole.
    const int L = (int)s.resident_per_layer.size();
    canvas.text_clip(ix, y + 1, iw,
                     fmt("experts resident in VRAM, per layer  ·  %d layers  ·  %u experts each",
                         L, s.n_expert),
                     kFaint, kPanel);
    const int cols = ih - 1;                 // one layer per row, columns of `per` layers
    if (cols < 1) return;
    const int per = (L + cols - 1) / std::max(1, cols);
    const int colw = iw / std::max(1, per);
    for (int i = 0; i < L; ++i) {
      const int c = i / cols, r = i % cols;
      const int px = ix + c * colw, py = y + 2 + r;
      if (colw < 24 || px + colw > x + w - 2) break;
      double f = s.n_expert ? (double)s.resident_per_layer[(size_t)i] / (double)s.n_expert : 0.0;
      f = f < 0 ? 0 : f > 1 ? 1 : f;
      canvas.text(px, py, fmt("L%-2d", i), kFaint, kPanel);
      static const Rgb kRes[3] = {kRed, kAmber, kGreen};
      // Two columns of gutter at the right: without it the count sits against the next column's
      // layer label and reads as belonging to it.
      hbar(canvas, px + 4, py, colw - 17, f, kRes, 3);
      canvas.text_clip(px + colw - 12, py, 10,
                       fmt("%4u %3.0f%%", s.resident_per_layer[(size_t)i], 100.0 * f), kMuted,
                       kPanel);
    }
    return;
  }

  if (tab == 2) {
    if (s.phases.empty()) {
      canvas.text(ix, y + 1, "phase timing needs AFF_PROFILE=1", kMuted, kPanel);
      return;
    }
    double tot = 0;
    for (const PhaseRow& p : s.phases) tot += p.ms;
    if (tot <= 0) tot = 1;
    canvas.text(ix, y + 1, "phase", kFaint, kPanel);
    canvas.text(ix + 20, y + 1, "ms/token", kFaint, kPanel);
    canvas.text(ix + 32, y + 1, "share", kFaint, kPanel);
    canvas.text(ix + iw - 10, y + 1, "of it wait", kFaint, kPanel);
    const int rows = std::min((int)s.phases.size(), ih - 2);
    for (int i = 0; i < rows; ++i) {
      const PhaseRow& p = s.phases[(size_t)i];
      const int py = y + 2 + i;
      canvas.text_clip(ix, py, 18, p.name, kText, kPanel);
      canvas.text(ix + 20, py, fmt("%8.3f", p.ms), kCyan, kPanel);
      const int bw = std::max(4, iw - 46);
      static const Rgb kSh[3] = {kCyan, kViolet, kAmber};
      hbar(canvas, ix + 32, py, bw, p.ms / tot, kSh, 3);
      // A phase that is nearly all wait is not a phase to make faster; it is a phase whose contents
      // to go and find. That distinction is the reason this column exists.
      canvas.text(ix + iw - 10, py, fmt("%6.0f%%", 100.0 * p.wait_frac),
                  p.wait_frac > 0.5 ? kAmber : kMuted, kPanel);
    }
    return;
  }

  // Link. This is the placement engine's OWN traffic — what the mover spent moving experts between
  // VRAM and RAM. The hybrid tier's streaming rides the same links and is deliberately not added
  // in: a missing expert is never copied anywhere, the GEMM reads it in place, and there is no
  // host-side count of it to add. A number mixing a real count with an estimate would be neither.
  // The graph is unlabelled otherwise: the tab strip says "Link", which names the hardware and not
  // the quantity, and a bare area plot of bytes a second is indistinguishable from the decode one
  // above it. The caption carries what is being counted and the axis it is drawn against.
  int cx0 = canvas.text(ix, y + 1, "mover traffic", kText, kPanel, kBold);
  cx0 = canvas.text(cx0, y + 1, "  PCIe bytes/s, VRAM ⇄ host RAM", kFaint, kPanel);
  const std::string peak = fmt("full scale %s", rate_bytes(g_link.max()).c_str());
  if ((int)peak.size() + cx0 + 2 < ix + iw)
    canvas.text(ix + iw - (int)peak.size(), y + 1, peak, kMuted, kPanel);

  const int gh = std::max(2, ih - 6);
  std::vector<float> hist;
  g_link.tail((size_t)iw * 2, &hist);
  graph(canvas, ix, y + 2, iw, gh, hist, 0.0f, kHot, 4);
  int ly = y + 2 + gh;
  int cx1 = canvas.text(ix, ly, fmt("now %s  ·  peak %s", rate_bytes(g_link.last()).c_str(),
                                    rate_bytes(g_link.max()).c_str()), kText, kPanel);
  canvas.text_clip(cx1 + 2, ly, ix + iw - cx1 - 2,
                   "·  excludes the hybrid tier, which the GEMM reads in place", kFaint, kPanel);
  ++ly;
  canvas.text_clip(ix, ly, iw,
                   fmt("placement  %s promotions  ·  %s demotions  ·  %s readmits  "
                       "·  %.2f GiB h2d / %.2f GiB d2h",
                       commas(s.promotions).c_str(), commas(s.demotions).c_str(),
                       commas(s.readmits).c_str(), (double)s.h2d_bytes / 1073741824.0,
                       (double)s.d2h_bytes / 1073741824.0), kMuted, kPanel);
  ++ly;
  const double rf = s.experts_total ? (double)s.resident / (double)s.experts_total : 0.0;
  int cx = canvas.text(ix, ly, "resident ", kMuted, kPanel);
  static const Rgb kRes[3] = {kRed, kAmber, kGreen};
  hbar(canvas, cx, ly, std::max(6, iw / 3), rf, kRes, 3);
  cx += std::max(6, iw / 3) + 1;
  std::string res = fmt("%s / %s experts (%.1f%%)", commas(s.resident).c_str(),
                        commas(s.experts_total).c_str(), 100.0 * rf);
  // Non-zero means dispatch blocked on a pread and a transform before it could launch. It is a
  // correctness path, not a tier — a run that leans on it is a run to re-budget.
  if (s.ssd_staged) res += fmt("  ·  %s ssd reads", commas(s.ssd_staged).c_str());
  canvas.text_clip(cx, ly, x + w - 2 - cx, res, kText, kPanel);
}

void Dashboard::Impl::panel_console(int x, int y, int w, int h) {
  if (h < 3) return;
  box(canvas, x, y, w, h, "Console", kViolet);
  const int ix = x + 2, iw = w - 4, ih = h - 2;
  if (iw < 8 || ih < 1) return;

  // Wrap from the newest line backwards and stop as soon as the pane is full: the alternative is
  // wrapping six hundred lines every frame to display the last eight of them.
  std::vector<std::pair<Rgb, std::string>> rows;
  {
    std::lock_guard<std::mutex> lk(log_m);
    for (auto it = log.rbegin(); it != log.rend() && (int)rows.size() < ih; ++it) {
      std::vector<std::string> wrapped;
      std::string cur;
      int col = 0;
      size_t i = 0;
      while (i < it->s.size()) {
        const size_t start = i;
        const char32_t cp = utf8_next(it->s, &i);
        const int cw = char_width(cp);
        if (col + cw > iw) { wrapped.push_back(cur); cur.clear(); col = 0; }
        cur.append(it->s, start, i - start);
        col += cw;
      }
      wrapped.push_back(cur);
      const Rgb col_rgb = level_colour(it->lvl);
      for (auto w_it = wrapped.rbegin(); w_it != wrapped.rend() && (int)rows.size() < ih; ++w_it)
        rows.emplace_back(col_rgb, *w_it);
    }
  }
  for (int i = 0; i < (int)rows.size(); ++i) {
    const int py = y + ih - i;
    if (py <= y) break;
    canvas.text_clip(ix, py, iw, rows[(size_t)i].second, rows[(size_t)i].first, kPanel);
  }
}

void Dashboard::Impl::panel_footer(int y, int w) {
  canvas.fill(0, y, w, 1, U' ', kMuted, kBg);
  int cx = 1;
  auto key = [&](const char* k, const char* what) {
    cx = canvas.text(cx, y, k, kBg, kCyan, kBold);
    cx = canvas.text(cx, y, " ", kMuted, kBg);
    cx = canvas.text(cx, y, what, kMuted, kBg);
    cx += 2;
  };
  key(" 1-4 ", "pane");
  key(" tab ", "next");
  key(" q ", "stop after this block");
}

}  // namespace aff::ui
