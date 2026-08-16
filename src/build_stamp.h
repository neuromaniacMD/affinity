// What binary am I actually running?
//
// This exists because measurements from a STALE BINARY have twice been reported here as real.
// Once a broken `sed` left a source file unparseable, the build failed, the harness kept going and
// four sweep points were four runs of the previous build. Once `bench/fp8_gemm.hip` was not in
// CMakeLists at all, so `cmake --build` never touched it and a binary from an earlier day answered
// an environment variable it had never been compiled to read — two "configurations" that were the
// same code twice, and a conclusion drawn from the difference between them.
//
// Neither failure is detectable from the numbers. Both are trivially detectable from a stamp.
//
// WHAT THIS DOES AND DOES NOT COVER. `__DATE__`/`__TIME__` is the compile time of THIS translation
// unit only. A change to src/gpu/*.hip relinks the binary without recompiling a bench's own .cpp, so
// the stamp does NOT move and must not be read as "this binary predates my edit". What covers that
// is the binary's mtime, which bench/measure.sh checks against src/ and prints alongside the stamp.
//
// The stamp is still the signal that catches the worst case — a binary the build system never
// touched at all, whose compile time is hours or days old — and it stays in the transcript, so a
// stale reading is diagnosable after the fact rather than only while someone is watching.
//
// `AFF_BUILD_ID` is provenance from `git describe` at CONFIGURE time; it does not move on an edit.
#pragma once

#include <cstdio>

#ifndef AFF_BUILD_ID
#define AFF_BUILD_ID "unknown"
#endif

namespace aff {

// First line of every tool and bench. Goes to stderr so it never lands in piped result tables.
inline void print_build_stamp(const char* who) {
  std::fprintf(stderr, "[build] %s  %s  compiled %s %s\n", who, AFF_BUILD_ID, __DATE__, __TIME__);
}

}  // namespace aff
