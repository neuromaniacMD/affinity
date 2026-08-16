// The routing trace: what the router actually selected, layer dispatch by layer dispatch.
//
// WHY THIS EXISTS. Every placement strategy after `heat` is a prediction about the NEXT dispatch,
// and a prediction can only be designed against a record of what actually happened. The engine's
// unit is a LAYER DISPATCH over a block of ~6 verify tokens, and the missing-expert cost is per
// DISTINCT expert in that dispatch. An offline replay that runs the model itself — one token per
// step, no DSpark — does not have the same statistics, and a strategy tuned on the wrong unit is
// tuned on the wrong thing.
//
// So this records the engine's own unit, from the one place that has it: the top of
// `StaticPlacement::run_batch_device`, before residency has been consulted.
//
// WHAT IS IN A RECORD, and why each field is separable:
//
//   * the raw selection (`nb_tok * k` expert ids) — placement-INDEPENDENT ground truth. This is
//     what the router chose; it does not know or care where the bytes were. Every policy question
//     ("what would LRU have done", "how well does the previous block predict this one") is
//     answerable from this column alone, offline, with no card.
//   * the DISTINCT experts that missed — placement-DEPENDENT, and the thing being predicted. It is
//     recorded rather than recomputed because it is the engine's own answer, including whatever
//     the mover had published by then, and an offline replay that disagrees with it is a bug in the
//     replay.
//
// Both are needed: the first says what a perfect policy could have done, the second says what this
// one did.
//
// The file also carries the residency map as it stood at load, so a replay is self-contained: an
// offline simulator can start from exactly the state the engine started from rather than from a
// reconstruction of it.
//
// OFF BY DEFAULT AND OFF THE HOT PATH. Writing is a memcpy into a preallocated buffer and a
// `fwrite` when it fills — no allocation after `open`, in keeping with the rest of the engine. It
// is a CLI flag (`--route-trace FILE`), not an environment variable, because those are spent.

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace aff {

// A text format, deliberately. It is ~25 bytes a dispatch-token, so a 512-token run is about 1 MB
// and ten prompts are ten — small enough that the convenience of `awk` on it beats the size, and
// the analysis that matters is a python script that has to be quick to change.
//
//   # aff-route-trace 1 layers=43 experts=256 k=6
//   R <layer> <e0> <e1> ...          the resident set at load, one line a layer
//   D <seq> <layer> <nb_tok> <sel...> | <miss...>      the TARGET model
//   S <seq> <layer> <nb_tok> <sel...> | <miss...>      the DSpark draft
//   G <handle> <q0> <q1> ... <q255>  the router's ranking value for EVERY expert (see below)
//
// `seq` counts dispatches from zero across the whole run, so `seq / layers` is the forward index
// and a layer wrapping to 0 is a block boundary. Prefill chunks are in the same stream and are told
// apart by `nb_tok` — a chunk is hundreds of tokens, a DSpark block is six.
//
// WHY THE GATE LINE EXISTS, AND WHAT IT BUYS THAT `sel` CANNOT. Everything derivable from `sel` is
// a statistic of what has ALREADY been selected, and an expert that has never been selected is
// invisible to all of it — count zero, no age, no history. Offline replay says that is exactly where
// the headroom is: a perfect out-of-sample estimate of each expert's future rate beats the shipped
// policy on BOTH misses and moves, while every estimator built from past use alone is worse than
// shipping. The router already computes a value for all 256 experts every block and throws away the
// 250 it does not pick, and an expert sitting at rank seven every block is the one thing past use
// cannot see. So this records the router's own ranking value, `sqrt(softplus(logit)) + bias`,
// summed over the block's tokens and quantised to a per-line scale.
//
// It is keyed by ROUTER HANDLE rather than by layer, because that is what the router knows. A
// reader must align it against the `D` line's selection — the top-k of a `G` line has to BE that
// line's `sel`, and if it is not, the two are not the same dispatch and nothing built on them means
// anything.
//
// THE TWO LETTERS ARE NOT COSMETIC. The draft is its own `StaticPlacement` over its own container,
// and its stages are the low layers of a DIFFERENT model — so without the tag they collide with the
// target's and a reader silently mixes two unrelated expert sets. The draft is fully resident and
// never misses, so nothing about placement depends on its rows; they are recorded anyway because
// "what did the drafter route to" is a candidate FEATURE for predicting what the target will route
// to, and it is free to keep.
class RouteTrace {
public:
  bool open(const std::string& path, uint32_t n_layer, uint32_t n_expert, uint32_t k);
  bool on() const { return f_ != nullptr; }

  // The residency map at load. Called once per layer, before any dispatch.
  void resident(uint32_t layer, const uint32_t* experts, uint32_t n);

  // One layer dispatch. `sel` is nb_tok*k entries in token-major order; `miss` is the distinct
  // experts this dispatch had to stream, ascending. `tag` is 'D' for the target model and 'S' for
  // the DSpark draft — see the format note above for why they must not be merged.
  void dispatch(char tag, uint32_t layer, uint32_t nb_tok, const uint32_t* sel, uint32_t n_sel,
                const uint32_t* miss, uint32_t n_miss);

  // The router's ranking value for every expert of one dispatch, already quantised by the caller to
  // 0..1000 against the line's own maximum. Written before the matching `D` line.
  void gate(uint32_t handle, const uint32_t* q, uint32_t n);

  void close();
  ~RouteTrace() { close(); }

  uint64_t records() const { return seq_; }

private:
  void put(const char* s, size_t n);
  void putu(uint32_t v);
  void flush();

  std::FILE* f_ = nullptr;
  std::vector<char> buf_;
  size_t used_ = 0;
  uint64_t seq_ = 0;
};

// The one instance, because the dispatch that writes it is reached from three call sites and
// threading a pointer through all of them buys nothing. Null until `--route-trace` opens it.
RouteTrace& route_trace();

} // namespace aff
