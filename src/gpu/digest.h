// The permanent cross-rank check: do the cards still hold the same bytes?
//
// Tensor parallel makes several buffers replicated by construction — the normed activation, the
// accumulator after its crossing, the hyper-connection lanes — and the engine relies on that
// everywhere. It has been false: an FP8 all-reduce that quantised only the peer's half left the two
// cards' KV rows disagreeing, and nothing caught it, because every check in the tree compared a card
// against a REFERENCE rather than against another card. A reference check cannot see a
// symmetric-looking asymmetry; only the cards compared to each other can.
//
// Every rank is compared against rank 0, which keeps the check complete as ranks are added: the
// buffer is meant to be one replicated thing, so agreeing with rank 0 is agreeing with each other.

#pragma once

#include <cstddef>
#include <cstdint>

namespace aff {

// It is a SAMPLER, not a barrier. Every kXrEvery-th call each card fingerprints the buffer on its own
// stream into pinned host memory and publishes a release-stamped sequence number; the host compares
// the slots whose stamps have all landed and skips the ones that have not. Nothing waits for
// anything — an instrument that needed a drain could not live on the path it is watching — and a
// mismatch aborts, because there is no correct way to continue from cards that have parted.
//
// `rank` is the rank index and `ranks` the count; the comparison runs once per visit, on the last
// rank, after every card's launch has been issued.
void xrank_check(uint32_t rank, uint32_t ranks, const void* p, size_t bytes, void* stream);

// AFF_NORM_TRACE=1: one magnitude a visit, per rank, on stderr. Diff two runs at DIFFERENT RANK
// COUNTS and the first line that parts names the sublayer that went wrong — which the hash above
// cannot do, because two rank counts never agree bitwise. Synchronises, so it is a diagnostic and
// never a measurement.
void xrank_trace(uint32_t rank, const char* tag, const void* p, size_t bytes, void* stream);

// How many samples have been compared, for the one-line report at exit. A check that reports zero
// checks is a failed check, not a passed one.
uint64_t xrank_compared();

} // namespace aff
