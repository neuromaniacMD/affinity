// The placement engine's STRATEGY: which expert should be in VRAM, decided from what the router
// actually did — the "heat" strategy.
//
// Deliberately pure host code with no HIP in it and no knowledge of slabs, pools or streams. It
// sees three things — a tier per expert, a heat per expert, and a tick — and answers one question:
// "swap these two." Everything about how the bytes get there lives in gpu/static_placement.hip.
// That split is what makes this testable on a machine with no card (tests/test_expert_heat.cpp).
//
// ---- WHY LFU-WITH-DECAY AND NOT SOMETHING CLEVERER --------------------------------------------
//
// An offline replay of a real routing trace against cache policies sets the shape of this file.
// Runtime placement beats static profile placement by a wide margin, so it is worth building.
// LFU-with-decay and LRU score the same, so the decay constant is not where the wins are and this
// does not need to be subtle.
//
// Belady is NOT barely above either — it is six points of hit rate above, at fewer moves — but
// almost none of that is reachable, and the reason is worth stating precisely because it decides
// what a predictive strategy could ever be. Split the oracle in half and let it see only a disjoint
// sample of the future, and it keeps 3.5 of those points; the rest is knowledge of the realised
// future. Of what remains, every statistic of PAST USE captures none — an expert never selected has
// count zero, no age and no history, and 71% of the space is in that state. The router's own gate
// value does separate those cases, and a scorer built on it is worth a few percent of the link
// budget — which is the one direction here that has not been refuted by measurement.
//
// ---- THE RANKING IS GLOBAL, NOT PER LAYER, AND THAT IS THE POINT ------------------------------
//
// Static placement ranks every expert by the container's profile and takes the top slice, which
// gives layers wildly different resident counts. The H-index census then comes out badly skewed —
// a hit rate below what choosing at random would give at the same residency, which is the signature
// of capacity sitting in the wrong layers. A layer with few resident experts misses on nearly every
// token no matter how good the ranking inside it is.
//
// So a swap here is (promote X, demote Y) with X and Y in DIFFERENT layers whenever that is what
// the heat says. A slab slot is layer-agnostic — the offsets are uniform — so cross-layer costs
// nothing, and a per-layer strategy could never repair the imbalance above.
//
// ---- WHAT IT DOES NOT DO ----------------------------------------------------------------------
//
// It never moves anything to or from the SSD tier. Banishing an expert to disk is a bet that it
// will not be needed, paid at a full pread every time the bet is wrong, and a frequency counter has
// no forward-looking confidence to justify one. That is the predictive strategy's job, and the
// dispatch path for it does not exist yet either.

#pragma once

#include <cstdint>
#include <vector>

namespace aff {

// Where an expert's bytes live. Mirrored here rather than read out of StaticPlacement, so this
// object stays pure — the engine tells it when a move publishes, and nothing else writes it.
enum class ExpertTier : uint8_t { Vram = 0, Pool = 1, Ssd = 2 };

struct HeatParams {
  // Applied to a layer's whole heat plane each time that layer is dispatched. Every layer is
  // dispatched exactly once per forward, so all planes decay in lockstep and heats stay comparable
  // ACROSS layers — which the global ranking above depends on.
  //
  // 0.98 gives a half-life of a few dozen dispatches, which at one dispatch a layer a block is on
  // the order of a couple of hundred tokens of memory: long enough to survive a paragraph, short
  // enough to follow a change of subject. A sweep over it separates no better than the spread within
  // one setting, which is what the offline replay predicts.
  float decay = 0.98f;
  // Promote only when the candidate beats the victim by this ratio. Without it the two ends of the
  // ranking trade places on noise and the engine spends the whole link on churn that changes
  // nothing. This is the only defence against thrash, because there is no admission filter — see
  // the note on `min_gain`.
  float hysteresis = 1.25f;
  // ...and this is the additive half of the same guard, in units of one activation. It is what
  // stops a swap when the victim's heat is ZERO, where any ratio test passes trivially. Early on
  // that is most of the resident set, so without this the first tick after warmup would try to
  // swap the entire cache.
  //
  // The economics look one-directional — a promotion costs an H2D plus a D2H and only pays if the
  // expert is selected again, so a tighter bar spends less — and they are not, because a promotion
  // DECLINED comes back as a miss, and a miss is a shard on the same budget. Total link traffic is
  // therefore U-SHAPED in this knob; this value sits at the bottom of the curve, and tightening past
  // it gives up hit rate and buys no traffic back. `aff-predict --sweep` re-derives it.
  float min_gain = 1.5f;
  // Dispatches to watch before moving anything, and it is 0 because THE GUARD ALREADY DOES THIS
  // JOB. The original reason for a warmup was that heats start at zero, so an early ranking is
  // noise ordered by expert id — but `min_gain` means a promotion needs `hot >= cold*hysteresis +
  // 1.5`, and against a zero-heat victim that is two tokens agreeing in one dispatch. That is
  // evidence, not noise, whatever the tick count is.
  //
  // So a warmup only postpones convergence. The corpus replay reaches the same steady-state hit
  // rate at every setting and simply takes proportionally longer to get there, which is the proof
  // that this knob does nothing but delay.
  uint32_t warmup_ticks = 0;
};

class ExpertHeat {
public:
  void init(uint32_t n_layer, uint32_t n_expert, const HeatParams& p);
  bool ready() const { return n_expert_ != 0; }

  // ---- the tier map, written only by the engine when a move PUBLISHES --------------------------
  //
  // Not when it is issued. Between issue and publish both copies of the bytes are valid and the
  // dispatch still reads the old one, so the strategy must agree with dispatch about where an
  // expert is or it will pick the same one twice. `busy` is what excludes an in-flight expert
  // instead.
  void set_tier(uint32_t idx, ExpertTier t);
  ExpertTier tier(uint32_t idx) const { return tier_[idx]; }
  void set_busy(uint32_t idx, bool b) { busy_[idx] = b ? 1u : 0u; }

  // One layer dispatch: decay that layer's plane, then credit the DISTINCT experts it selected.
  // Distinct, not per slot — the dispatch reads an expert's weights once however many of the block's
  // tokens chose it, so a repeat inside a block costs nothing and must not be paid for twice.
  //
  // `counts` is how many of the block's tokens chose each, and it is the CREDIT rather than a flat
  // 1.0. Cost is per distinct expert but the probability of needing it again is not: over the trace
  // corpus, an expert several of a block's tokens chose is far likelier to be chosen again soon
  // than one a single token chose, and most selections are single-token. Weighting by it buys hit
  // rate for nothing — it moves no extra bytes, it only ranks better.
  //
  // Null means one each, which is what the tests want and what a caller with no counts to hand
  // should get.
  void observe(uint32_t layer, const uint32_t* experts, uint32_t n,
               const uint32_t* counts = nullptr);

  uint64_t ticks() const { return ticks_; }
  bool warm() const { return ticks_ >= params_.warmup_ticks; }

  // The best swap available, or false when the guards refuse. Both halves are returned together
  // even though the engine issues them as two independent transfers, because the DECISION is a
  // comparison between them: promoting without the matching demotion would grow the resident set.
  bool pick(uint32_t* promote, uint32_t* demote) const;

  // ---- THE DEMAND ARM: the same heat plane, a different trigger ---------------------------------
  //
  // `pick` promotes the globally hottest POOLED expert, and that is the wrong question. Replaying a
  // real trace (`affinity --route-trace`, then `aff-predict`) with the move budget and both guards
  // removed — so the strategy may swap the global extremes as often as it likes — barely moves the
  // miss count. The strategy was never moving too little; the ranking simply does not name the
  // expert that is about to be needed. An expert becomes "hottest pooled" only after it has been
  // paid for several times, and by then the misses have already been paid.
  //
  // A miss, by contrast, is direct evidence: this expert was needed NOW, and the same expert is
  // selected again in the next block far more often than a frequency ranking predicts. So admission
  // is triggered by the miss and only the VICTIM is chosen by heat, which is what the plane is
  // actually good at.
  //
  // `note_missing` hands the dispatch's misses over; `pick_demand` then drains that list. It is not
  // const — the list is consumed, so the engine cannot issue the same transfer twice in one tick.
  void note_missing(uint32_t layer, const uint32_t* experts, uint32_t n);
  bool pick_demand(uint32_t* promote, uint32_t* demote);

  float heat(uint32_t idx) const { return heat_[idx]; }
  // How many of each layer's experts are in VRAM. The whole diagnosis above is one histogram of
  // this, so the engine prints it at exit and this is where it comes from.
  void resident_per_layer(std::vector<uint32_t>* out) const;

private:
  HeatParams params_;
  uint32_t n_layer_ = 0, n_expert_ = 0;
  uint64_t ticks_ = 0;
  std::vector<float> heat_;
  std::vector<ExpertTier> tier_;
  std::vector<uint8_t> busy_;
  // ---- THE TIE-BREAK, and it is not cosmetic ---------------------------------------------------
  //
  // Most resident experts are cold — the resident set answers a far smaller share of activations
  // than its size — so `heat == 0.0f` exactly is the common case at both ends of the ranking, and
  // whatever breaks that tie IS the policy for most of a run. Breaking it by index order strips the
  // low-numbered layers first, which replaces the skew this engine exists to fix with a different
  // one. tests/test_expert_heat.cpp covers it.
  //
  // So ties go to the layer that can most afford it: take the victim from the layer with the MOST
  // resident experts, and promote into the one with the fewest. Maintained here rather than
  // recomputed, because `pick` runs once a layer dispatch.
  std::vector<uint32_t> res_count_;
  // This dispatch's misses, as global indices, consumed by `pick_demand`. Reserved at init to
  // n_expert so a tick never allocates — a layer cannot miss on more experts than it has.
  std::vector<uint32_t> demand_;
};

} // namespace aff
