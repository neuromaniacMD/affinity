// What the dashboard draws, and the seam it reads it through.
//
// Two kinds of number end up on the screen and they are gathered completely differently.
//
//   ENGINE STATE — heat, residency, acceptance, phase times. Owned by the dispatch thread and
//   written on the hot path, so the UI thread may not touch it. The dispatch thread copies it into
//   a Snapshot when it is between blocks, and the UI thread renders whatever the last copy was.
//
//   MACHINE STATE — VRAM, GPU busy, clocks, host RAM. Owned by the driver and the kernel, and
//   readable by anyone. The UI thread samples it itself, at its own rate, so it stays live while
//   the engine is inside a long load or a prefill chunk and publishes nothing.
//
// Keeping the two apart is what makes the dashboard cost the engine one try_lock and one memcpy a
// block. It is also why a stalled engine still shows moving GPU meters: that is a diagnosis, not a
// glitch, and a dashboard that froze with the engine would hide it.
//
// This header deliberately includes nothing from engine/ or gpu/. The UI links into the CPU-only
// build too, and a dependency here would drag HIP into it.

#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace aff::ui {

inline constexpr int kMaxSpecPos = 16;

// Where the run is. Drives which panels carry live numbers and which are still blank.
enum class Stage : uint8_t {
  Init, LoadDense, LoadExperts, LoadDraft, LoadPool, Ready, Prefill, Decode, Serve, Done
};
const char* stage_name(Stage s);

// Mirrors engine/expert_heat.h's ExpertTier without including it — the UI must not depend on the
// placement engine's headers, and a tier is one byte either way. Kept in the same order so the
// publisher's cast is a cast and not a translation table.
enum : uint8_t { kTierVram = 0, kTierPool = 1, kTierSsd = 2, kTierUnknown = 3 };

struct PhaseRow {
  const char* name = "";
  double ms = 0;         // per token
  double wait_frac = 0;  // of `ms`, the share the host spent blocked on a card
};

struct Snapshot {
  // ---- identity -------------------------------------------------------------------------------
  std::string model;        // container stem
  std::string quant;        // routed-expert format, e.g. "q2.875 quad"
  uint32_t n_layer = 0, n_expert = 0, top_k = 0, n_embd = 0, vocab = 0;
  uint32_t spec_width = 0;  // dspark_block; 0 when speculation is off
  uint32_t ranks = 0;

  // ---- where the run is -----------------------------------------------------------------------
  Stage stage = Stage::Init;
  std::string detail;       // free text under the stage, e.g. "layer 21/43"
  double stage_frac = -1;   // 0..1, or negative when the step has no measurable extent

  // ---- throughput -----------------------------------------------------------------------------
  double prefill_tok_s = 0, decode_tok_s = 0, ms_per_block = 0;
  uint64_t prompt_tokens = 0, generated = 0, target = 0;
  uint64_t pos = 0, kv_capacity = 0;
  double elapsed_s = 0;

  // ---- speculation ----------------------------------------------------------------------------
  uint64_t blocks = 0, drafted = 0, accepted = 0;
  double pos_rate[kMaxSpecPos] = {};   // marginal hit rate per drafted position
  uint32_t pos_valid = 0;

  // ---- placement ------------------------------------------------------------------------------
  uint64_t resident = 0, experts_total = 0;
  uint64_t promotions = 0, demotions = 0, readmits = 0, ssd_staged = 0;
  uint64_t refused = 0, starved_slab = 0, starved_pool = 0;
  uint64_t h2d_bytes = 0, d2h_bytes = 0;
  std::vector<uint32_t> resident_per_layer;

  // ---- the heat plane, n_layer * n_expert -----------------------------------------------------
  //
  // Both arrays or neither. `heat` is the raw decayed count, whose scale moves with the decay
  // constant, so the UI normalises per frame rather than against a fixed maximum.
  std::vector<float> heat;
  std::vector<uint8_t> tier;

  // ---- where the time goes --------------------------------------------------------------------
  std::vector<PhaseRow> phases;
  double blocked_ms_per_token = 0;
  uint64_t drain_count = 0;
  double drain_max_ms = 0;

  // ---- host -----------------------------------------------------------------------------------
  double rss_gib = 0, pool_gib = 0;
};

// The handoff. One writer (the dispatch thread), one reader (the UI thread).
//
// `publish` takes the lock with try_lock and gives up when the reader has it. A missed frame is a
// frame the UI redraws from the previous snapshot, which is invisible at ten a second — and the
// alternative is a blocking primitive on the thread whose whole job is keeping the cards' queues
// full. The reader does block, because it has nothing else to do.
class Bus {
 public:
  template <class F>
  bool publish(F&& fill) {
    std::unique_lock<std::mutex> lk(m_, std::try_to_lock);
    if (!lk.owns_lock()) return false;
    fill(w_);
    ++seq_;
    return true;
  }
  // Unconditional variant for the paths that must not be dropped: a stage change, and the final
  // state before the dashboard is torn down.
  template <class F>
  void publish_sync(F&& fill) {
    std::lock_guard<std::mutex> lk(m_);
    fill(w_);
    ++seq_;
  }
  void read(Snapshot* dst, uint64_t* seq) const {
    std::lock_guard<std::mutex> lk(m_);
    *dst = w_;
    *seq = seq_;
  }

 private:
  mutable std::mutex m_;
  Snapshot w_;
  uint64_t seq_ = 0;
};

}  // namespace aff::ui
