#include "engine/expert_heat.h"

#include <algorithm>

namespace aff {

void ExpertHeat::init(uint32_t n_layer, uint32_t n_expert, const HeatParams& p) {
  params_ = p;
  n_layer_ = n_layer;
  n_expert_ = n_expert;
  ticks_ = 0;
  const size_t n = (size_t)n_layer * n_expert;
  heat_.assign(n, 0.0f);
  tier_.assign(n, ExpertTier::Ssd);
  busy_.assign(n, 0u);
  res_count_.assign(n_layer, 0u);
  demand_.clear();
  demand_.reserve(n_expert);
}

void ExpertHeat::set_tier(uint32_t idx, ExpertTier t) {
  const uint32_t l = idx / n_expert_;
  if (tier_[idx] == ExpertTier::Vram && t != ExpertTier::Vram) --res_count_[l];
  else if (tier_[idx] != ExpertTier::Vram && t == ExpertTier::Vram) ++res_count_[l];
  tier_[idx] = t;
}

void ExpertHeat::observe(uint32_t layer, const uint32_t* experts, uint32_t n,
                         const uint32_t* counts) {
  if (layer >= n_layer_) return;
  float* row = heat_.data() + (size_t)layer * n_expert_;
  // Decay first, then credit. The other order would give this dispatch's experts a decayed count,
  // which makes `min_gain = 1.0` mean 1/decay activations rather than one and is the kind of
  // off-by-a-constant that only shows up as a threshold that does not do what it says.
  for (uint32_t e = 0; e < n_expert_; ++e) row[e] *= params_.decay;
  for (uint32_t i = 0; i < n; ++i)
    if (experts[i] < n_expert_) row[experts[i]] += counts ? (float)counts[i] : 1.0f;
  ++ticks_;
}

// A FULL SCAN of the whole heat plane, called once per move rather than once per tick — several
// scans a layer dispatch, which looks like something to fold into one pass keeping the top-N of each
// end. It is not: quadrupling the move budget quadruples the scanning and costs nothing, because the
// host ends every layer blocked in a stream drain and this runs inside that slack. Folding the scans
// would buy nothing and cost the property that makes this easy to reason about — that picking N
// pairs is exactly picking one pair N times.
bool ExpertHeat::pick(uint32_t* promote, uint32_t* demote) const {
  if (!warm() || !promote || !demote) return false;
  const size_t n = heat_.size();
  // One pass, both ends. `busy` excludes an expert with a move in flight — its tier still says
  // where dispatch will look, which is the old tier, so without this the same expert is picked
  // every tick until its move retires and the engine issues the same transfer several times.
  uint32_t hot = UINT32_MAX, cold = UINT32_MAX;
  float hot_h = 0.0f, cold_h = 0.0f;
  uint32_t hot_res = 0, cold_res = 0;             // resident count of the candidate's layer
  for (size_t i = 0; i < n; ++i) {
    if (busy_[i]) continue;
    const float h = heat_[i];
    const uint32_t rc = res_count_[i / n_expert_];
    if (tier_[i] == ExpertTier::Pool) {
      // Hotter wins; on an exact tie, the layer with the FEWEST resident experts wins, because
      // that is the layer whose tokens are missing most often.
      if (hot == UINT32_MAX || h > hot_h || (h == hot_h && rc < hot_res)) {
        hot = (uint32_t)i; hot_h = h; hot_res = rc;
      }
    } else if (tier_[i] == ExpertTier::Vram) {
      // Colder wins; on an exact tie, take from the layer holding the MOST, which is the one that
      // can spare it. Both halves of this are the same statement about where capacity should sit.
      if (cold == UINT32_MAX || h < cold_h || (h == cold_h && rc > cold_res)) {
        cold = (uint32_t)i; cold_h = h; cold_res = rc;
      }
    }
    // ExpertTier::Ssd is skipped on purpose: reaching one needs a streaming dispatch path that
    // does not exist yet, and a frequency counter is not evidence enough to put an expert there.
    // Both directions are the predictive strategy's business.
  }
  if (hot == UINT32_MAX || cold == UINT32_MAX) return false;
  // BOTH guards, and they answer different failures. The ratio stops a swap that trades one warm
  // expert for another; the additive floor stops the degenerate case where the victim's heat is
  // zero, which every ratio passes and which describes most of the resident set right after warmup.
  if (hot_h < cold_h * params_.hysteresis + params_.min_gain) return false;
  *promote = hot;
  *demote = cold;
  return true;
}

void ExpertHeat::note_missing(uint32_t layer, const uint32_t* experts, uint32_t n) {
  demand_.clear();
  if (layer >= n_layer_) return;
  // In the order the dispatch found them, which is ascending by expert id. The engine takes them
  // from the FRONT so the list is served in that order rather than reversed; with a budget below
  // the miss count the choice is arbitrary either way, and being deterministic is what makes an
  // A/B reproducible.
  for (uint32_t i = 0; i < n; ++i)
    if (experts[i] < n_expert_) demand_.push_back((uint32_t)layer * n_expert_ + experts[i]);
}

bool ExpertHeat::pick_demand(uint32_t* promote, uint32_t* demote) {
  if (!warm() || !promote || !demote) return false;
  const size_t n = heat_.size();
  while (!demand_.empty()) {
    const uint32_t idx = demand_.front();
    demand_.erase(demand_.begin());
    // Not `Vram`: between issue and publish a promoted expert is still read out of the pool, so the
    // dispatch reports it as missing for one more layer. Its `busy` flag is what stops it being
    // admitted twice. Not `Ssd` either — there is no dispatch path for one, so promoting out of it
    // would name bytes nothing can read.
    if (busy_[idx] || tier_[idx] != ExpertTier::Pool) continue;
    uint32_t cold = UINT32_MAX, cold_res = 0;
    float cold_h = 0.0f;
    for (size_t i = 0; i < n; ++i) {
      if (busy_[i] || tier_[i] != ExpertTier::Vram) continue;
      const float h = heat_[i];
      const uint32_t rc = res_count_[i / n_expert_];
      // Identical to `pick`'s victim half, tie-break included: on an exact tie take from the layer
      // holding the MOST resident experts. At the residencies this runs at `heat == 0.0f` is the
      // common case, so whatever breaks that tie IS the policy for most of a run (§4.1).
      if (cold == UINT32_MAX || h < cold_h || (h == cold_h && rc > cold_res)) {
        cold = (uint32_t)i; cold_h = h; cold_res = rc;
      }
    }
    if (cold == UINT32_MAX) return false;
    // ---- the guard, and it is doing a DIFFERENT job here than in `pick` -------------------------
    //
    // In `pick` the guard stops two ends of a ranking trading places on noise. Here the candidate is
    // not a ranking's guess, it is an expert that was demonstrably needed — so the question is not
    // "is this the right expert" but "is one sighting worth two shards".
    //
    // TWO shards, and that is the whole reason a guard is needed at all. The two PCIe directions
    // share one budget, so the demotion's write-back is charged at exactly the rate a miss is
    // streamed at: a swap costs a promotion plus a demotion against a miss's single shard.
    //
    // Unguarded, this trigger cuts missing experts a dispatch, RAISES total link traffic, and is
    // slower end to end. The guard is what buys the miss reduction without the traffic.
    if (heat_[idx] < cold_h * params_.hysteresis + params_.min_gain) continue;
    *promote = idx;
    *demote = cold;
    return true;
  }
  return false;
}

void ExpertHeat::resident_per_layer(std::vector<uint32_t>* out) const {
  if (!out) return;
  out->assign(n_layer_, 0u);
  for (uint32_t l = 0; l < n_layer_; ++l) {
    uint32_t n = 0;
    for (uint32_t e = 0; e < n_expert_; ++e)
      if (tier_[(size_t)l * n_expert_ + e] == ExpertTier::Vram) ++n;
    (*out)[l] = n;
  }
}

} // namespace aff
