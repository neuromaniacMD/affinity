// The one place the engine's informational output goes.
//
// Two sinks, chosen once at startup and never mixed. Without a dashboard `out` and `err` are
// std::printf and std::fprintf(stderr) and emit exactly the bytes they always have — the bench
// scripts parse that stream, and a reworked banner is a harness that silently reports an empty
// column. With a dashboard they become lines in its console pane, because a stray write into the
// alternate screen lands wherever the cursor happened to be.
//
// The split between the two streams is preserved either way: stdout carries what a run IS, stderr
// what it had to say about getting there. `aff --help | head` and `2>/dev/null` both keep working.

#pragma once

#include <cstdint>
#include <string_view>

namespace aff::ui {

enum class Level : uint8_t { Info, Note, Warn, Error, Token };

// printf-style, no trailing newline required — one call is one line.
void out(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void err(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
// A generated token. Never buffered by line: partial words have to appear as they are produced or
// the pane lags the model by a sentence.
void tok(std::string_view piece);

// A message the process is about to die on: an abort, or a return that ends the run.
//
// It tears the dashboard down FIRST and then writes to stderr, because a diagnostic that goes into
// a pane and is then destroyed with the alternate screen is worse than no diagnostic at all — the
// operator gets a restored terminal, a non-zero status and no reason. Every print that precedes a
// std::abort() or a fatal return belongs here rather than in err().
void fatal(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

// Restores the terminal and replays whatever the pane still holds. Installed by Dashboard::start().
using Teardown = void (*)();
void set_teardown(Teardown t);

// True while a dashboard is drawing. Callers use it to skip output that only makes sense as a
// scrolling log — a per-position table that the dashboard already draws as bars, say.
bool dashboard_active();

// Installed by Dashboard::start() and cleared by stop(). Null restores the printf sinks.
using Sink = void (*)(void* ctx, Level, std::string_view);
void set_sink(Sink s, void* ctx);

}  // namespace aff::ui
