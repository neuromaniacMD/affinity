// The live view of a run: devices, throughput, speculation, the expert plane, and where the time
// goes — redrawn in place instead of scrolled past.
//
// IT NEVER RUNS UNLESS STDOUT IS A TERMINAL. Everything the bench harness reads is a line on stdout
// or stderr, and the rule that keeps those bytes identical is that the dashboard and the plain
// sinks are mutually exclusive and the choice is made by isatty(3). `--ui plain` forces the lines
// on a terminal; `--ui dash` is an error when stdout is a pipe rather than a silent downgrade —
// a flag that quietly does nothing is the failure this project has spent the most time on.
//
// ONE THREAD, AND IT ONLY READS. The render thread owns the terminal, samples sysfs and /proc, and
// takes a copy of the engine's Snapshot when the dispatch thread has published one. It makes no HIP
// calls and holds no engine object. The dispatch thread's entire share of the cost is a try_lock
// and a copy of the heat plane once a block.

#pragma once

#include "ui/sysprobe.h"
#include "ui/telemetry.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace aff::ui {

enum class UiMode : uint8_t { Auto, Dash, Plain };

// Parses --ui. Returns false and fills `err` on an unrecognised value.
bool parse_ui_mode(std::string_view s, UiMode* out, std::string* err);

class Dashboard {
 public:
  Dashboard();
  ~Dashboard();
  Dashboard(const Dashboard&) = delete;
  Dashboard& operator=(const Dashboard&) = delete;

  // Registered before start(); the driver is the only thing that knows the cards' PCI addresses,
  // and the probe refuses to guess a card index.
  void add_device(const DeviceInfo& d);

  // False when there is no terminal to draw on. Installs the ui::log sink on success.
  bool start(UiMode mode);
  // Restores the terminal and puts ui::log back on stdout/stderr. Idempotent, and called from the
  // destructor so an early return out of main cannot leave the screen switched.
  void stop();
  bool running() const { return running_.load(std::memory_order_relaxed); }

  // Where the dispatch thread publishes. Safe to call whether or not the dashboard is running: with
  // no reader the copy is still cheap, and the driver then has one code path.
  Bus& bus() { return bus_; }

  // A stage change is published synchronously — it is what the header reads, it happens a handful
  // of times a run, and a dropped one leaves the screen describing the wrong phase indefinitely.
  void stage(Stage s, std::string detail = {}, double frac = -1);

  // Ctrl-C, or `q`. The decode loop polls this to stop at a block boundary instead of dying inside
  // one, so the run still prints its fingerprint and its counters.
  bool quit_requested() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> im_;
  Bus bus_;
  std::atomic<bool> running_{false};
};

// Process-wide, because ui::log's sink is a process-wide function pointer and there is exactly one
// terminal. Constructed on first use and never destroyed early — the engine's teardown order is not
// something a log call should have to know about.
Dashboard& dashboard();

}  // namespace aff::ui
