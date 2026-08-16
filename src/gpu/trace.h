#pragma once
// Per-entry-point device timing without rocprof and without draining the stream.
//
// Two columns, and they answer different questions:
//   sec[]  host time inside the scope. Issue is async, so this is where the host WAITS; a slot can
//          read very large and own none of it.
//   dev[]  device time from an event pair recorded on the stream and read back a call later, once
//          it has completed. No drain, no added synchronisation.
//
// The device path is sampled through a ring of kRing pairs per slot, collected with hipEventQuery
// and never hipEventSynchronize. A single blocking pair per slot stalls the host at every scope
// entry, and a stalled host changes how backlogged the stream is, which is the very thing the event
// pair measures. With a ring, a call goes unmeasured when the queue is more than kRing behind; the
// ones measured are consecutive.
//
// Reading a trace:
//   - Slots kDeepFirst and up are nested inside another slot, and an event pair is not free —
//     adding sub-slots can inflate the enclosing scope substantially. Read a deep run as ratios
//     between its sub-slots, never as absolute time against an untraced run.
//   - A slot inside a `for (d : devices)` loop fires twice a call and records both on card 0's
//     stream, so the second pair times a gap: the slot total is right, its us/call is half.
//   - Events record on card 0's stream only. A slot the two cards run differently (`reduce`)
//     measures whichever card reaches the barrier first, not what it cost.

#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace aff {
namespace trace {

// One accessor, parsed by value: AFF_TRACE=dense[,events][,deep][,sync][,evsync].
//   dense   the host column, and the master switch
//   events  adds the device column
//   deep    the nested slots (kDeepFirst and up)
//   sync    drain the stream before each scope stops its clock, charging every entry point its own
//           device time. The total is inflated by construction — read shares, not absolutes.
//   evsync  blocking collect, as a control for the sampled path above.
struct Flags {
  bool on = false, events = false, deep = false, sync_all = false, ev_sync = false;
};

inline const Flags& flags() {
  static const Flags f = [] {
    Flags r;
    const char* v = std::getenv("AFF_TRACE");
    if (!v || !*v) return r;
    auto has = [v](const char* k) {
      const size_t n = std::strlen(k);
      for (const char* p = v; (p = std::strstr(p, k)) != nullptr; p += n) {
        const bool lo = p == v || p[-1] == ',';
        const bool hi = p[n] == '\0' || p[n] == ',';
        if (lo && hi) return true;
      }
      return false;
    };
    r.on = true;                              // any value at all turns the host column on
    r.events = has("events");
    r.deep = has("deep");
    r.sync_all = has("sync");
    r.ev_sync = has("evsync");
    return r;
  }();
  return f;
}

struct DenseTrace {
  // 0-19 single-token (decode) entry points, 20-32 batched prefill, 33 up nested splits of a group.
  // A slot added past the end of the name table writes over `calls[0]` and over `on`, which turns
  // tracing off after 255 calls; the static_assert makes that a compile error instead.
  static constexpr int kN = 52;
  static constexpr int kDeepFirst = 33;      // b_gate, b_topk, hc_mv, hc_fk and the group splits
  static const char* name(int i) {
    static const char* n[] = {"mv", "attn_pre", "shexp", "attn_q", "attn_kv", "attn_run", "attn_out",
                              "hc_pre", "hc_post", "kv_raw", "kv_comp", "kv_commit",
                              "router", "norm_coll", "pre_mv", "pre_rtr", "norm_wait", "norm_cpy",
                              "comp_coll", "reduce",
                              "b_begin", "b_hc_pre", "b_qkv", "b_mv_host", "b_kv_commit",
                              "b_attend", "b_attn_out", "b_hc_post", "b_router", "b_shexp",
                              "b_reduce", "b_rtopk", "b_compress", "b_gate", "b_topk",
                              "hc_mv", "hc_fk",
                              "b_hc_mv", "b_hc_fk", "b_q_qa", "b_q_qb", "b_q_rope",
                              "b_ao_pre", "b_ao_woa", "b_ao_wob",
                              "q_qa", "q_norm", "q_qb", "q_rope",
                              "ao_pre", "ao_woa", "ao_wob"};
    static_assert(sizeof(n) / sizeof(n[0]) == kN, "a trace slot has no name");
    return n[i];
  }
  double sec[kN] = {};
  uint64_t calls[kN] = {};

  static constexpr int kRing = 8;
  hipEvent_t ev[kN][kRing][2] = {};
  bool pend[kN][kRing] = {};
  uint32_t head[kN] = {};
  double dev[kN] = {};
  uint64_t dcalls[kN] = {};                  // pairs actually collected; the per-call divisor
  bool on       = flags().on;
  bool events   = flags().events;
  bool ev_sync  = flags().ev_sync;
  bool deep     = flags().deep;
  bool sync_all = flags().sync_all;
  hipStream_t stream = nullptr;

  // Fold in ring entry r of slot i if it has completed. `block` waits for it; the measured path
  // never passes true, so a pair still in flight stays outstanding at the cost of one query. Only
  // collected pairs advance dcalls, so dev/dcalls is unbiased over the subset measured.
  void collect(int i, int r, bool block) {
    if (!pend[i][r]) return;
    if (!block && hipEventQuery(ev[i][r][1]) != hipSuccess) return;
    pend[i][r] = false;
    float ms = 0.0f;
    if (hipEventSynchronize(ev[i][r][1]) == hipSuccess &&
        hipEventElapsedTime(&ms, ev[i][r][0], ev[i][r][1]) == hipSuccess) {
      dev[i] += (double)ms * 1e-3;
      ++dcalls[i];
    }
  }

  double dev_us(int i) const { return dcalls[i] ? 1e6 * dev[i] / (double)dcalls[i] : 0.0; }
  double dev_ms(int i) const { return 1e-3 * dev_us(i) * (double)calls[i]; }

  // Eagerly, on the card whose stream will record them: an event belongs to whichever device was
  // current at creation and may only be recorded on a stream of that device. Creating them lazily
  // inside a scope dies with "invalid resource handle" on the first scope that runs under rank 1.
  void bind(hipStream_t s) {
    stream = s;
    if (!events) return;
    for (int i = 0; i < kN; ++i)
      for (int r = 0; r < kRing; ++r) {
        (void)hipEventCreate(&ev[i][r][0]);
        (void)hipEventCreate(&ev[i][r][1]);
      }
  }

  ~DenseTrace() {
    if (!on) return;
    double tot = 0, dtot = 0;
    for (int i = 0; i < kN; ++i) {
      for (int r = 0; r < kRing; ++r) collect(i, r, true);
      tot += sec[i];
      dtot += dev_ms(i);
    }
    std::fprintf(stderr, "\ndense-gpu trace (%.3f s host-blocked", tot);
    if (events) std::fprintf(stderr, ", %.3f s device est", 1e-3 * dtot);
    std::fprintf(stderr, ")\n");
    std::fprintf(stderr, "  %-10s %10s %12s %12s", "op", "calls", "host ms", "us/call");
    if (events) std::fprintf(stderr, " %12s %12s %8s", "DEVICE ms", "us/call", "samp");
    std::fprintf(stderr, "\n");
    for (int i = 0; i < kN; ++i) {
      if (!calls[i]) continue;
      std::fprintf(stderr, "  %-10s %10llu %12.2f %12.2f", name(i),
                   (unsigned long long)calls[i], 1000.0 * sec[i],
                   1e6 * sec[i] / (double)calls[i]);
      if (events)
        std::fprintf(stderr, " %12.2f %12.2f %8llu", dev_ms(i), dev_us(i),
                     (unsigned long long)dcalls[i]);
      std::fprintf(stderr, "\n");
    }
  }
};

// `inline`, so every translation unit that traces shares one instance: two copies would split the
// table and each would print a partial run.
inline DenseTrace g_trace;

// Scoped: charges the enclosing block to slot `i`.
struct TraceScope {
  int i;
  int r = -1;                                // ring entry this call was measured into, or -1
  std::chrono::steady_clock::time_point t0;
  explicit TraceScope(int slot) : i(slot) {
    if (!g_trace.on) return;
    if (i >= DenseTrace::kDeepFirst && !g_trace.deep) { i = -1; return; }
    if (g_trace.events && g_trace.stream) {
      // Free the ring entry about to be reused. A query, not a wait: a pair still in flight leaves
      // the call unmeasured rather than stalling the host to measure it.
      const uint32_t h = g_trace.head[i];
      if (g_trace.ev[i][h][0]) {
        g_trace.collect(i, (int)h, g_trace.ev_sync);
        if (!g_trace.pend[i][h]) {
          r = (int)h;
          (void)hipEventRecord(g_trace.ev[i][h][0], g_trace.stream);
        }
      }
    }
    t0 = std::chrono::steady_clock::now();
  }
  ~TraceScope() {
    if (!g_trace.on || i < 0) return;
    if (g_trace.sync_all && g_trace.stream) (void)hipStreamSynchronize(g_trace.stream);
    g_trace.sec[i] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ++g_trace.calls[i];
    if (r >= 0) {
      (void)hipEventRecord(g_trace.ev[i][r][1], g_trace.stream);
      g_trace.pend[i][r] = true;
      g_trace.head[i] = (g_trace.head[i] + 1u) % (uint32_t)DenseTrace::kRing;
    }
  }
};

// The routed-expert path needs a per-card table: the two cards run the same slots on different
// streams, and an event pair whose halves land on two streams times something that never happened.
// Card 0's rows and the dense table above are one timeline — same stream, and their sum is that
// stream's busy time.
//
// The host column is nearly useless here: every stage but the one sync is an async launch, so
// `sec[]` is launch overhead and the blocking call absorbs the rest. The asymmetry is worth seeing
// (it is how a starved stream shows up) but the deciding number is the device column.
struct XTrace {
  // x_bucket is pure host work — the counting pass, the slot lists, the tile table — recorded with
  // no stream and contributing nothing to the device column. It is in the table because a host
  // stage between two async groups is what starves a stream. Slot 7 is decode: run_device()'s one
  // expert_ffn_hip a card a layer.
  //
  // Slots 8 and 9 are the hybrid stream, gate/up and down of the experts this card does not hold,
  // streamed from the registered host pool by the GEMM itself. They run concurrently with slots 3
  // and 5, so their time is not additive: read h_gu against x_gu, and a hybrid dispatch is free
  // exactly when h_gu is the smaller.
  static constexpr int kN = 10;
  static constexpr int kMaxDev = 8;
  static const char* name(int i) {
    static const char* n[] = {"x_bucket", "x_h2d", "x_quant", "x_gu",
                              "x_swiglu", "x_down", "x_scatter", "d_ffn", "h_gu", "h_down"};
    static_assert(sizeof(n) / sizeof(n[0]) == kN, "a routed trace slot has no name");
    return n[i];
  }
  bool on = flags().on;
  bool events = flags().events;
  double sec[kMaxDev][kN] = {};
  uint64_t calls[kMaxDev][kN] = {};
  static constexpr int kRing = 8;
  hipEvent_t ev[kMaxDev][kN][kRing][2] = {};
  bool pend[kMaxDev][kN][kRing] = {};
  uint32_t head[kMaxDev][kN] = {};
  double dev[kMaxDev][kN] = {};
  uint64_t dcalls[kMaxDev][kN] = {};
  hipStream_t stream[kMaxDev][kN] = {};
  // Slot totals that are not time: the work each call was given. A stage's device time means
  // nothing without its token count, and `taken` varies several-fold per layer per card. `padded`
  // is columns multiplied against columns asked for — a tile runs its bucket's widest member and
  // zero-fills the rest.
  uint64_t taken[kMaxDev] = {}, tiles[kMaxDev] = {};
  uint64_t padded[kMaxDev] = {};
  // Of `tiles`, how many named the host pool instead of the slab — the H-index as the dispatch
  // itself sees it, and the divisor for every hybrid number.
  uint64_t hyb_tiles[kMaxDev] = {};

  void collect(int d, int i, int r, bool block) {
    if (!pend[d][i][r]) return;
    if (!block && hipEventQuery(ev[d][i][r][1]) != hipSuccess) return;
    pend[d][i][r] = false;
    float ms = 0.0f;
    if (hipEventSynchronize(ev[d][i][r][1]) == hipSuccess &&
        hipEventElapsedTime(&ms, ev[d][i][r][0], ev[d][i][r][1]) == hipSuccess) {
      dev[d][i] += (double)ms * 1e-3;
      ++dcalls[d][i];
    }
  }
  double dev_us(int d, int i) const {
    return dcalls[d][i] ? 1e6 * dev[d][i] / (double)dcalls[d][i] : 0.0;
  }
  double dev_ms(int d, int i) const { return 1e-3 * dev_us(d, i) * (double)calls[d][i]; }

  ~XTrace() {
    if (!on) return;
    for (int d = 0; d < kMaxDev; ++d)
      for (int i = 0; i < kN; ++i)
        for (int r = 0; r < kRing; ++r) collect(d, i, r, true);
    for (int d = 0; d < kMaxDev; ++d) {
      // Any slot, not slot 0: x_bucket is entered only by the batched path, so guarding on it
      // printed nothing for a decode-only run, which is the run slot 7 exists for.
      uint64_t any = 0;
      for (int i = 0; i < kN; ++i) any += calls[d][i];
      if (!any) continue;
      double tot = 0, dtot = 0;
      for (int i = 0; i < kN; ++i) { tot += sec[d][i]; dtot += dev_ms(d, i); }
      dtot *= 1e-3;
      std::fprintf(stderr, "\nrouted-expert trace, card %d (%.3f s host-blocked", d, tot);
      if (events) std::fprintf(stderr, ", %.3f s device est", dtot);
      if (tiles[d])
        // `hyb` is what h_gu and h_down actually carried. Without it their GB/s comes off an
        // assumed miss count rather than off the tiles the dispatch issued, which is a rate for a
        // census figure and not for the work.
        std::fprintf(stderr, "; %llu calls, %llu slots, %llu tiles (%llu hybrid), %.1f slots/tile, "
                     "%.2fx padded",
                     (unsigned long long)calls[d][0], (unsigned long long)taken[d],
                     (unsigned long long)tiles[d], (unsigned long long)hyb_tiles[d],
                     (double)taken[d] / (double)tiles[d],
                     taken[d] ? (double)padded[d] / (double)taken[d] : 0.0);
      std::fprintf(stderr, ")\n");
      std::fprintf(stderr, "  %-10s %10s %12s %12s", "op", "calls", "host ms", "us/call");
      if (events) std::fprintf(stderr, " %12s %12s %8s", "DEVICE ms", "us/call", "share");
      std::fprintf(stderr, "\n");
      for (int i = 0; i < kN; ++i) {
        if (!calls[d][i]) continue;
        std::fprintf(stderr, "  %-10s %10llu %12.2f %12.2f", name(i),
                     (unsigned long long)calls[d][i], 1000.0 * sec[d][i],
                     1e6 * sec[d][i] / (double)calls[d][i]);
        if (events) {
          std::fprintf(stderr, " %12.2f %12.2f", dev_ms(d, i), dev_us(d, i));
          if (dtot > 0) std::fprintf(stderr, " %7.1f%%", 0.1 * dev_ms(d, i) / dtot);
        }
        std::fprintf(stderr, "\n");
      }
    }
  }
};
inline XTrace g_xt;

struct XScope {
  int d, i;
  int r = -1;                                  // ring entry this call was measured into, or -1
  std::chrono::steady_clock::time_point t0;
  XScope(int dev_idx, int slot, hipStream_t st) : d(dev_idx), i(slot) {
    if (!g_xt.on) { i = -1; return; }
    if (g_xt.events && st) {
      const uint32_t h = g_xt.head[d][i];
      if (!g_xt.ev[d][i][h][0]) {              // lazy: the caller has already set the device
        (void)hipEventCreate(&g_xt.ev[d][i][h][0]);
        (void)hipEventCreate(&g_xt.ev[d][i][h][1]);
      }
      g_xt.collect(d, i, (int)h, false);       // a query, never a wait — see XTrace::kRing
      if (!g_xt.pend[d][i][h]) {
        r = (int)h;
        g_xt.stream[d][i] = st;
        (void)hipEventRecord(g_xt.ev[d][i][h][0], st);
      }
    }
    t0 = std::chrono::steady_clock::now();
  }
  // The host build has no lexical block of its own — `taken`, the tile table and the slot lists are
  // declared inside it and read after — so that slot stops by hand rather than at scope exit.
  // Idempotent; the destructor is the only other caller.
  void stop() {
    if (i < 0) return;
    g_xt.sec[d][i] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ++g_xt.calls[d][i];
    if (r >= 0 && g_xt.stream[d][i]) {
      (void)hipEventRecord(g_xt.ev[d][i][r][1], g_xt.stream[d][i]);
      g_xt.pend[d][i][r] = true;
      g_xt.head[d][i] = (g_xt.head[d][i] + 1u) % (uint32_t)XTrace::kRing;
      r = -1;
    }
    i = -1;
  }
  ~XScope() { stop(); }
};

}  // namespace trace
}  // namespace aff

// Every call site passes a literal, so the range check is free and turns a slot added past the end
// of the name table into a compile error rather than a corrupted counter.
#define AFF_TRACE(slot)                                                                     \
  static_assert((slot) >= 0 && (slot) < ::aff::trace::DenseTrace::kN, "trace slot out of range"); \
  ::aff::trace::TraceScope _trace_scope_(slot)

#define AFF_XTRACE(dev, slot, st)                                                           \
  static_assert((slot) >= 0 && (slot) < ::aff::trace::XTrace::kN, "routed slot out of range"); \
  ::aff::trace::XScope _xscope_##slot(dev, slot, st)
