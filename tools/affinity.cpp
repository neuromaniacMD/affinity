// affinity — CLI entry point: serve an OpenAI-compatible endpoint from a .aff model.
#include "build_stamp.h"
#include <mutex>

#include "engine/model.h"
#include "engine/instrument.h"
#include "engine/route_trace.h"
#include "engine/sampler.h"
#include "engine/tokenizer.h"
#ifdef AFF_WITH_HIP
#include "gpu/digest.h"
#include "gpu/static_placement.h"
#include "gpu/dense_gpu.h"
#include "gpu/keepalive.h"
#endif
#include "server/server.h"
#include "engine/chat_encode.h"
#include "engine/prefix_cache.h"

#include "ui/dashboard.h"
#include "ui/log.h"
#include <atomic>
#include <cstdarg>
#include <thread>
#include <random>
#include <cstdio>
#include <csignal>
#include <functional>
#include <unistd.h>
#include <sys/statvfs.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <string>

using namespace aff;

static volatile std::sig_atomic_t g_stop = 0;

// A line assembled from pieces, for the few places that build one in a loop. ui::out takes a whole
// line at a time — a dashboard pane cannot show half of one — so the loop builds a string instead of
// printing fragments.
static std::string sfmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
static std::string sfmt(const char* f, ...) {
  char b[512];
  va_list ap;
  va_start(ap, f);
  const int n = std::vsnprintf(b, sizeof b, f, ap);
  va_end(ap);
  return n > 0 ? std::string(b, (size_t)std::min<int>(n, (int)sizeof b - 1)) : std::string();
}

// <｜end▁of▁sentence｜> in DeepSeek-V4-Flash's tokenizer, and eos_token_id in its config.json.
constexpr uint32_t kEosToken = 1;

// VmHWM, the high-water mark: the pool is populated in one pass and never shrinks, so the peak is
// the number that has to fit, not the resident set at any later moment. 0 if /proc is unreadable.
static double peak_rss_gib() {
  std::FILE* f = std::fopen("/proc/self/status", "r");
  if (!f) return 0.0;
  char line[256];
  double gib = 0.0;
  while (std::fgets(line, sizeof line, f)) {
    unsigned long long kb = 0;
    if (std::sscanf(line, "VmHWM: %llu kB", &kb) == 1) { gib = (double)kb / 1048576.0; break; }
  }
  std::fclose(f);
  return gib;
}

// A path with its final extension removed, whatever that extension is. Not `rfind(".aff")`: the
// container is identified by its magic and not by its name, so a path reached through a symlink or
// a copy with another suffix loads fine — and then a companion path derived by stripping a literal
// ".aff" finds nothing at all, which reads as a checkpoint with no draft rather than as a name this
// code could not parse.
static std::string drop_ext(const std::string& p) {
  const size_t slash = p.find_last_of('/');
  const size_t dot = p.find_last_of('.');
  if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) return p;
  return p.substr(0, dot);
}

// The phase breakdown's rows, in report order. ONE definition: the end-of-run table and the
// dashboard's timing pane both read it, and two copies drift the first time a phase is added.
static std::vector<std::pair<const char*, Model::PhaseProfile::Ph>>
phase_rows(const Model::PhaseProfile& p) {
  return {
    {"embed+hyper-conn", p.hyper}, {"attn projections", p.attn_proj},
    {"compressor", p.compressor},  {"lightning indexer", p.indexer},
    {"attention (softmax+AV)", p.attention}, {"attn out (inv rope, wo_a/b, hc)", p.attn_out},
    {"router (gate matvec+topk)", p.router},
    {"ffn dense (shared expert)", p.ffn_dense},
    // Named for the drain, not the top-k. Labelled "routing (host top-k)" this row reads as most
    // of a decode token and gets believed. The host top-k is a narrow argmax over a few tokens;
    // the number is batch_router_topk's hipStreamSynchronize, which cannot return until the
    // layer's whole device chain has — and that chain is mostly the hybrid tier streaming missing
    // experts over PCIe. The wait column beside it says so on every run.
    {"route + drain (waits for layer)", p.route}, {"expert dispatch (device)", p.dispatch},
    {"ffn routed experts", p.ffn_routed}, {"head (vocab)", p.head},
    // The draft, separate: a different model at a different batch size, and folding it into the
    // target's buckets makes both unreadable.
    {"dspark main_x + draft kv", p.draft_kv}, {"dspark attn (3 stages)", p.draft_attn},
    {"dspark ffn (3 stages)", p.draft_ffn}, {"dspark head + markov", p.draft_head},
    {"dspark block seed (emb h2d)", p.draft_seed},
    {"dspark router drain (3x d2h)", p.draft_route},
    {"dspark head epilogue (host)", p.draft_ep},
  };
}

#ifdef AFF_WITH_HIP
// ---- the VRAM ledger ----------------------------------------------------------------------------
//
// Per-card VRAM, priced by differencing the driver's own free-memory count across each load phase
// rather than by summing what the engine asked for. Those are different numbers: the allocator
// rounds, and the runtime takes a reserve of its own before the engine allocates anything.
//
// `report` prints an `unattributed` row on purpose. A residual folded into the rows that ARE
// counted stops being findable.
struct VramLedger {
  std::vector<std::pair<std::string, std::vector<double>>> rows;   // label -> per-card GiB
  std::vector<aff::VramSample> last, first;

  void begin() { last = first = aff::vram_sample(); }
  void mark(const char* label) {
    std::vector<aff::VramSample> now = aff::vram_sample();
    if (now.size() != last.size()) { last = now; return; }
    std::vector<double> d(now.size());
    for (size_t i = 0; i < now.size(); ++i) d[i] = (now[i].used_gib - last[i].used_gib) * 1024.0;
    // A phase that moved less than a MiB on every card does not earn a row.
    for (double v : d) if (v > 1.0 || v < -1.0) { rows.emplace_back(label, d); break; }
    last = now;
  }
  void report() const {
    const std::vector<aff::VramSample> now = aff::vram_sample();
    if (now.empty()) return;
    auto line = [&](const char* label, const std::vector<double>& mib) {
      std::string s = sfmt("  %-30s", label);
      for (double v : mib) s += sfmt("%10.0f", v);
      aff::ui::out("%s\n", s.c_str());
    };
    aff::ui::out("\nVRAM per card, MiB, measured by differencing the driver's free count per phase:\n");
    std::vector<double> sum(now.size(), 0.0);
    for (const auto& r : rows) {
      line(r.first.c_str(), r.second);
      for (size_t i = 0; i < r.second.size() && i < sum.size(); ++i) sum[i] += r.second[i];
    }
    // The runtime's context exists before the first sample, so no phase can be differenced
    // against it and it is named separately.
    std::vector<double> base(now.size()), gap(now.size()), used(now.size()), tot(now.size());
    for (size_t i = 0; i < now.size(); ++i) {
      base[i] = (i < first.size() ? first[i].used_gib : 0.0) * 1024.0;
      sum[i] += base[i];
      used[i] = now[i].used_gib * 1024.0;
      gap[i]  = used[i] - sum[i];
      tot[i]  = now[i].total_gib * 1024.0;
    }
    line("runtime, before any phase", base);
    line("unattributed", gap);
    line("= in use", used);
    line("  card total", tot);
  }
};
#endif

#ifdef AFF_WITH_HIP
// The prefix cache's view of the cards. It moves opaque rows and never learns the compress
// schedule, so these two are the whole of the coupling between it and the GPU.
static bool pc_save(void* ctx, uint8_t part, uint32_t layer, uint32_t rank, uint64_t row0,
                    uint64_t nrows, void* dst) {
  return ((DenseGpu*)ctx)->kv_part_save((DenseGpu::KvPart)part, layer, rank, row0, nrows, dst);
}
static bool pc_load(void* ctx, uint8_t part, uint32_t layer, uint32_t rank, uint64_t row0,
                    uint64_t nrows, const void* src) {
  return ((DenseGpu*)ctx)->kv_part_load((DenseGpu::KvPart)part, layer, rank, row0, nrows, src);
}
#endif

// `aff-profile`'s output: a magic, two dimensions, then n_layer*n_expert float32 layer-major. The
// dimensions are in the file rather than inferred from its length because the failure they prevent
// is silent — a profile of the wrong shape still ranks, just wrongly, and the run would look like a
// placement regression rather than a mistyped path. The reader checks the count against the
// container separately, so a profile fitted for another model is caught too.
static bool load_expert_profile(const std::string& path, std::vector<float>* out, std::string* err) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) { *err = "cannot open"; return false; }
  struct { char magic[8]; uint32_t n_layer, n_expert; } h{};
  if (std::fread(&h, sizeof h, 1, f) != 1) { std::fclose(f); *err = "truncated header"; return false; }
  if (std::memcmp(h.magic, "affprof1", 8) != 0) {
    std::fclose(f); *err = "not a aff-profile file (bad magic)"; return false;
  }
  const size_t n = (size_t)h.n_layer * h.n_expert;
  if (!n) { std::fclose(f); *err = "zero-sized profile"; return false; }
  out->resize(n);
  const bool ok = std::fread(out->data(), sizeof(float), n, f) == n;
  std::fclose(f);
  if (!ok) { *err = "truncated body"; return false; }
  return true;
}

int main(int argc, char** argv) {
  aff::print_build_stamp("affinity");
  // dspark_path empty means "<model>.dspark.aff if present". Naming one explicitly makes a load
  // failure fatal: a run that asked for the draft and silently did not get it measures the wrong
  // thing while looking healthy.
  std::string model_path, tok_path, dspark_path, host = "127.0.0.1";
  // Off does NOT mean single-token decode — that path aborts on a preshuffled fp8 shard (see
  // no_ps in dense_gpu.hip), because the W8A8 GEMM is the only reader of that layout. It means the
  // same block verify at width 1: one token a cycle instead of 1+B, through the same code.
  //
  // The two arms are not bit-comparable and are not meant to be. The block head picks its kernel by
  // width, so width 1 takes the multi-x matvec and width B the GEMM, and they round differently; the
  // draft's VRAM is also expert residency the target does not get. Placement alone does not move the
  // text, but the head kernel does.
  //
  // The draft pays for its VRAM everywhere except an answer too short to amortise one block.
  // Acceptance is a property of the prompt, so a tok/s figure from any single prompt is not a speed
  // number for a code change — ms a cycle is. See bench/spec.sh.
  bool dspark_on = true;
  // Which tier the draft's experts live in. The draft has the same two tiers the target does, so
  // `ram` pools its slab and streams it over PCIe through the same GEMM, handing the VRAM to the
  // target's slab instead.
  //
  // `vram` by default because `ram` costs more a block than it saves. The residency it hands the
  // target is real, but the target already hits nearly everything it selects while a pooled draft
  // misses all of it, every block, so the bus gains less than it pays. The balance is a property of
  // a machine's PCIe-to-residency ratio rather than of the idea, so re-measure it before assuming
  // the default is right elsewhere.
  bool dspark_ram = false;
  uint16_t port = 8080;
  // Per card, and a CAP rather than a target: StaticPlacement::init takes min(this, free VRAM minus
  // a fixed reserve), so the default is deliberately larger than any card here and the free-VRAM
  // term is what binds. Lower it only to leave room for something else on the card.
  double gpu_budget_mib = 1048576.0;
  // Host RAM held for the experts that did not fit in VRAM. Negative sizes it from MemAvailable at
  // load; 0 disables the pool and reads the container's mapping instead. See engine/host_pool.h.
  double host_pool_mib = -1.0;
  // How much of MemAvailable the auto-sizer refuses to take.
  //
  // Every MiB of it is a MiB the expert pool does not get, and whatever the pool cannot hold falls
  // to the SSD tier and is re-read on each activation. It is not covering the engine's own
  // allocations — those are already resident when the pool is sized, so MemAvailable has excluded
  // them — it is deciding how much of the machine one process should take.
  //
  // ---- a floor, not a tax, and the SSD tier is why it is set this low --------------------------
  //
  // HostPool sizes its allocation by DEMAND — `n_pool = min(budget / stride, want.size())` — so
  // while the complement fits, lowering this frees nothing to anyone and costs nothing to anybody.
  // It binds in exactly one case: the complement does not fit, and then every MiB of it is experts
  // on disk.
  //
  // The cost of that is not proportional to the shortfall, which is what makes the knob sharp. A
  // layer holding even ONE addressless expert cannot use the device-built dispatch at all — a kernel
  // cannot read an expert off disk — so the whole layer reverts to the host build and the drain
  // behind it. A shortfall of a few tenths of one per cent of the expert set, spread thinly, can
  // therefore take half the engine's layer-dispatches off the device path.
  //
  // It trades against cold prefill, which is faster at a larger reserve: a smaller reserve is a
  // larger pool, and the pool has to be filled and registered before the first token. The default
  // buys the steady state at the cold start's expense.
  //
  // Raise --host-pool-reserve-mib on the command line for a machine that shares its RAM, and read
  // the pool's own line at startup — it prints the exact shortfall and the value that would close
  // it.
  double host_pool_reserve_mib = 1536.0;
  // VRAM a card the expert slab must leave alone; 0 derives it from the card.
  double gpu_headroom_mib = 0.0;
  // Per-card heartbeat interval, microseconds. Off by default: a clear win in the microbenchmark
  // and a small loss in the engine (gpu/keepalive.h). Kept because the balance tips if the CPU
  // expert share grows.
  double keepalive_us = 0.0;
  // How many GPUs to run across. 0 means every supported card the machine has, which is what a
  // dedicated box wants. Set it lower to leave cards for something else; HIP_VISIBLE_DEVICES is the
  // way to choose WHICH, since this only takes a prefix of what the runtime enumerates.
  //
  // One card is supported and runs the whole model on it, with no collective at all. It is not the
  // design point — what tensor parallel buys is dividing the dense weight read every token — and on
  // a card that cannot hold the experts it spills them to the host pool and the SSD tier.
  size_t kGpuCount = 0;
  // Positions the KV cache is sized for. 1M is the model's maximum and the default, and since the
  // cache became lazily backed it is very nearly free: --kv-size reserves address space, and only
  // --kv-commit below spends VRAM. Lowering it no longer buys residency.
  // Independent sequences the engine holds device state for. 1 is the historical engine.
  // --kv-size is PER SLOT, so N slots cost N times the KV: the budget is total positions.
  uint32_t kSlots = 1;
  uint64_t kMaxKvPositions = 1048576;
  // Positions the compressed cache is BACKED with at load. The rest of --kv-size is reserved
  // address space that costs no VRAM until the sequence reaches it, and the difference goes to the
  // expert slab: at the shipped 1M that is 621 more resident experts, worth 11.5% of decode cold
  // and 3.7% once the placement engine has converged.
  // Grown from the placement thread on a watermark, never on the dispatch path.
  uint64_t kKvCommitPositions = 65536;
  // How far ahead of the sequence the grower keeps the cache backed. Large enough that it fires
  // roughly once every 3700 blocks at decode's ~4.4 positions a block; small values exist so the
  // growth path can be exercised in a short run instead of only by a 65k-token prompt.
  uint64_t kKvMarginPositions = 16384;
  // A floor under the server's per-request generation budget, in tokens.
  //
  // `max_tokens` is the caller's, and the engine spends it on reasoning and answer together (see
  // the block above the budget). A reasoning turn that needs more than the caller allowed returns
  // finish_reason "length" with no answer at all, which is correct and is also useless to an
  // operator whose client hardcodes a small number it will not send. This raises such a request to
  // the floor; it never lowers one, and the context still bounds it. 0 honours the caller exactly.
  uint32_t max_tokens_floor = 256000;
  // The prefix cache. Off unless a directory is named, because it writes gigabytes and where it
  // writes them is not something to guess at.
  PrefixCache::Options pcache;
  // Hash the live attention state once the prompt is in, and print it. Two runs of the same
  // conversation must agree whether the state came from a prefill or from the store — that is the
  // one claim the whole design rests on, and nothing else can check it from outside.
  bool pcache_verify = false;
  std::string prompt;
  bool one_shot = false;
  int n_predict = 32;
  int dump_topk = 0;
  std::string token_list;
  // Ids that override the sampler, cycling if shorter than -n. A measurement tool: it makes decode
  // throughput a property of the code rather than of the continuation.
  std::string force_list;
  bool no_routed = false;
  bool ignore_eos = false;
  // One-shot sampling. Greedy by default because it is reproducible, but temperature 0 degenerates
  // and changes which experts route, so a throughput or acceptance figure should be taken at real
  // sampler settings with a fixed seed instead.
  SamplerConfig sample_cfg;
  sample_cfg.temperature = 0.0f;
  // Whether any of them was given, so that passing none leaves the decode on its argmax path
  // rather than on the sampler at temperature 0. The two agree on the token but not on the code
  // that picks it, and every recorded fingerprint here was taken on the former.
  bool sampling_set = false;
  std::string logits_out;
  // Oracle knobs, for tools/ref_forward.py: at the shipped index_topk the lightning indexer needs
  // ~2048 positions and all 43 layers before it engages. See Model::set_index_topk.
  uint32_t index_topk = 0, max_layers = 0;
  bool dump_tokens = false;
  // Tokens per batched-prefill chunk; 0 disables the batched path. Rounded up to a multiple of 128.
  //
  // Bigger amortises the routed-expert weights over more tokens, and costs VRAM twice: the dense
  // activation arena, and the routed arena that placement allocates before it sizes the expert
  // slabs. So the chunk trades prefill throughput against how many experts stay resident, and the
  // best value is the largest one whose complement still fits the host pool.
  // The default is the largest chunk whose complement still fit the host pool on the machine this
  // was tuned on; re-sweep it on a machine with a different VRAM-to-RAM split.
  uint32_t prefill_chunk = 1408;
  bool echo_prompt = true;
  // The live dashboard, on when stdout is a terminal and off when it is not. That condition is what
  // keeps a piped run byte-identical to what it has always been — every bench script parses these
  // streams, and a reworked banner is a harness reporting an empty column.
  aff::ui::UiMode ui_mode = aff::ui::UiMode::Auto;
#ifdef AFF_WITH_HIP
  // Placement. `heat` is the default, and by a wide margin on measurement. `static` decides at load
  // and adds nothing to any tick, which makes it the control arm rather than a configuration anyone
  // should run.
  PlacementEngineConfig pcfg;
  pcfg.strategy = PlacementEngineConfig::Strategy::Heat;
  // Taken from the free VRAM left after the slabs rather than from residency, so turning the engine
  // on costs no hit rate. It also bounds how many transfers can be in flight, which is why it is not
  // larger.
  uint32_t placement_shadow = 32;
  std::string route_trace_path;
  // Fitted by `aff-profile` from --route-trace records; overrides the container's imatrix-derived
  // profile for BOTH consumers. See AffReader::profile().
  std::string expert_profile_path;
  // Repeat the prefill in-process and report the steady state. See the block that uses it.
  uint32_t prefill_reps = 1;
  // A second prompt to rotate in, so convergence is measured against a union rather than one set.
  std::string prefill_alt;
#endif
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto nxt = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "-m" || a == "--model") model_path = nxt();
    else if (a == "--tokenizer") tok_path = nxt();
    else if (a == "--host") host = nxt();
    else if (a == "--port") port = (uint16_t)std::strtoul(nxt(), nullptr, 10);
    else if (a == "--max-tokens-floor") max_tokens_floor = (uint32_t)std::strtoul(nxt(), nullptr, 10);
    else if (a == "-p" || a == "--prompt") { prompt = nxt(); one_shot = true; }
    else if (a == "--temp") { sample_cfg.temperature = std::strtof(nxt(), nullptr); sampling_set = true; }
    else if (a == "--top-k") { sample_cfg.top_k = (uint32_t)std::strtoul(nxt(), nullptr, 10); sampling_set = true; }
    else if (a == "--top-p") { sample_cfg.top_p = std::strtof(nxt(), nullptr); sampling_set = true; }
    else if (a == "--min-p") { sample_cfg.min_p = std::strtof(nxt(), nullptr); sampling_set = true; }
    else if (a == "--seed") { sample_cfg.seed = std::strtoull(nxt(), nullptr, 10); sampling_set = true; }
    // ARG_MAX stops `-p "$(cat file)"` at a few hundred KiB — ~30K tokens, well short of what the
    // cache is sized for — so a deep-context prompt has to come from a file.
    else if (a == "--prompt-file") {
      const char* pf = nxt();
      std::FILE* f = std::fopen(pf, "rb");
      if (!f) { aff::ui::err("cannot open prompt file %s\n", pf); return 1; }
      std::string buf;
      char chunk[65536];
      size_t got;
      while ((got = std::fread(chunk, 1, sizeof chunk, f)) > 0) buf.append(chunk, got);
      std::fclose(f);
      prompt = buf;
      one_shot = true;
    }
    else if (a == "-n" || a == "--n-predict") n_predict = (int)std::strtol(nxt(), nullptr, 10);
    else if (a == "--dump-logits") dump_topk = (int)std::strtol(nxt(), nullptr, 10);
    else if (a == "--tokens") token_list = nxt();
    else if (a == "--force-tokens") force_list = nxt();
    else if (a == "--no-routed-experts") no_routed = true;
    else if (a == "--ignore-eos") ignore_eos = true;
    else if (a == "--gpu-mib") gpu_budget_mib = std::strtod(nxt(), nullptr);
    else if (a == "--host-pool-mib") host_pool_mib = std::strtod(nxt(), nullptr);
    else if (a == "--dspark") {
      const std::string v = nxt();
      if (v == "on") dspark_on = true;
      else if (v == "off") dspark_on = false;
      else { aff::ui::err("unknown --dspark '%s' (want on or off; the container path is "
                          "--dspark-model)\n", v.c_str()); return 2; }
    }
    else if (a == "--dspark-model") dspark_path = nxt();
    else if (a == "--dspark-placement") {
      const std::string v = nxt();
      if (v == "ram") dspark_ram = true;
      else if (v == "vram") dspark_ram = false;
      else { aff::ui::err("unknown --dspark-placement '%s' (want ram or vram)\n", v.c_str());
             return 2; }
    }
    else if (a == "--host-pool-reserve-mib") host_pool_reserve_mib = std::strtod(nxt(), nullptr);
    else if (a == "--gpu-headroom-mib") gpu_headroom_mib = std::strtod(nxt(), nullptr);
    else if (a == "--keepalive-us") keepalive_us = std::strtod(nxt(), nullptr);
    else if (a == "--gpus") kGpuCount = (size_t)std::strtoull(nxt(), nullptr, 10);
    else if (a == "--slots") kSlots = (uint32_t)std::strtoul(nxt(), nullptr, 10);
    else if (a == "--kv-size") kMaxKvPositions = std::strtoull(nxt(), nullptr, 10);
    else if (a == "--kv-commit") kKvCommitPositions = std::strtoull(nxt(), nullptr, 10);
    else if (a == "--kv-margin") kKvMarginPositions = std::strtoull(nxt(), nullptr, 10);
    else if (a == "--prefix-cache-dir") pcache.dir = nxt();
    else if (a == "--prefix-cache-gib")
      pcache.budget_bytes = (uint64_t)(std::strtod(nxt(), nullptr) * 1073741824.0);
    else if (a == "--prefix-cache-block")
      pcache.block_tokens = (uint32_t)std::strtoul(nxt(), nullptr, 10);
    else if (a == "--persistent-prefix-cache") pcache.persistent = true;
    else if (a == "--prefix-cache-verify") pcache_verify = true;
    else if (a == "--logits-out") logits_out = nxt();
    else if (a == "--index-topk") index_topk = (uint32_t)std::strtoul(nxt(), nullptr, 10);
    else if (a == "--max-layers") max_layers = (uint32_t)std::strtoul(nxt(), nullptr, 10);
    else if (a == "--dump-tokens") dump_tokens = true;
    else if (a == "--no-echo") echo_prompt = false;
    else if (a == "--ui") {
      std::string uerr;
      if (!aff::ui::parse_ui_mode(nxt(), &ui_mode, &uerr)) {
        std::fprintf(stderr, "error: %s\n", uerr.c_str());
        return 2;
      }
    }
    else if (a == "--prefill-chunk-size")
      prefill_chunk = (uint32_t)std::strtoul(nxt(), nullptr, 10);
#ifdef AFF_WITH_HIP
    // One accessor per flag, each parsing its value rather than testing for presence: a flag an A/B
    // sets on both sides has to be read the same way by everything that consults it.
    else if (a == "--placement-strategy") {
      const std::string v = nxt();
      if (v == "static") pcfg.strategy = PlacementEngineConfig::Strategy::Static;
      else if (v == "heat") pcfg.strategy = PlacementEngineConfig::Strategy::Heat;
      else if (v == "demand") pcfg.strategy = PlacementEngineConfig::Strategy::Demand;
      else { aff::ui::err("unknown --placement-strategy '%s' (want static, heat or "
                          "demand)\n", v.c_str()); return 2; }
    }
    else if (a == "--placement-moves") pcfg.moves_per_tick = (uint32_t)std::strtoul(nxt(), nullptr, 10);
    else if (a == "--placement-decay") pcfg.heat.decay = (float)std::strtod(nxt(), nullptr);
    else if (a == "--placement-hysteresis") pcfg.heat.hysteresis = (float)std::strtod(nxt(), nullptr);
    else if (a == "--placement-min-gain") pcfg.heat.min_gain = (float)std::strtod(nxt(), nullptr);
    else if (a == "--placement-shadow") placement_shadow = (uint32_t)std::strtoul(nxt(), nullptr, 10);
    else if (a == "--placement-lead") pcfg.max_lead_blocks = (uint32_t)std::strtoul(nxt(), nullptr, 10);
    else if (a == "--route-trace") route_trace_path = nxt();
    else if (a == "--expert-profile") expert_profile_path = nxt();
    else if (a == "--prefill-reps") prefill_reps = (uint32_t)std::strtoul(nxt(), nullptr, 10);
    else if (a == "--prefill-alt") prefill_alt = nxt();
#endif
    else if (a == "-h" || a == "--help") {
      aff::ui::out(
        "affinity — hybrid CPU+GPU MoE inference\n"
        "\n"
        "Model and memory\n"
        "  -m, --model FILE            .aff model container (required)\n"
        "      --tokenizer FILE        tokenizer.json\n"
        "      --dspark on|off         speculative decode via the DSpark draft (default on). Off\n"
        "                              runs the same block verify one token wide and does not load\n"
        "                              the draft, handing its VRAM back to the routed experts. The\n"
        "                              arms differ in residency and in head kernel, so the text\n"
        "                              moves between them.\n"
        "      --dspark-model FILE     draft container (default <model>.dspark.aff)\n"
        "      --dspark-placement T    where the draft's experts live: vram (default) or ram. `ram`\n"
        "                              pools the slab and streams it over PCIe through the same GEMM\n"
        "                              the target's non-resident experts use, handing the VRAM to\n"
        "                              the target's slab. It is a trade, not a saving: the draft\n"
        "                              reads its experts several times a block, so what it streams\n"
        "                              can cost more than the VRAM it hands back.\n"
        "      --gpu-mib M             cap on VRAM per card for the expert slabs. The default is\n"
        "                              larger than any card, so free VRAM is what binds; lower it\n"
        "                              only to leave room for something else on the card.\n"
        "      --host-pool-mib M       RAM held for the non-VRAM experts; 0 disables the pool and\n"
        "                              reads the mapping instead (default: from MemAvailable)\n"
        "      --gpu-headroom-mib M    VRAM a card the expert slab leaves alone; 0 (default)\n"
        "                              derives it from the card. Too small and the slab's own\n"
        "                              allocation fails -- free VRAM and one contiguous block of\n"
        "                              that size are not the same quantity.\n"
        "      --host-pool-reserve-mib M  RAM left free when auto-sizing (default 1536). Every\n"
        "                              MiB of it is a MiB the expert pool does not get, and what\n"
        "                              does not fit the pool falls to the SSD tier. Raise it to\n"
        "                              share the machine with other work.\n"
        "                              A layer holding even one expert on the SSD tier reverts to\n"
        "                              the HOST dispatch and the drain behind it, so a small\n"
        "                              shortfall can cost a large share of the device-built\n"
        "                              dispatch. The pool says how short it is at load.\n"
        "      --gpus N                GPUs to run across (default 0 = every supported card).\n"
        "                              HIP_VISIBLE_DEVICES chooses WHICH; this takes a prefix.\n"
        "      --kv-size N             KV cache positions (default 1048576, the model's maximum).\n"
        "                              Address space, not VRAM — the cache is backed lazily, so\n"
        "                              lowering this does not buy residency. --kv-commit does.\n"
        "      --slots N               independent sequences to hold device state for (default 1).\n"
        "                              --kv-size is PER SLOT, so N slots cost N times the KV and\n"
        "                              the budget is TOTAL positions: 4 slots of 128K costs what\n"
        "                              one slot of 512K does. Above 1 requires --kv-commit ==\n"
        "                              --kv-size (a slot is fully backed; the grower is per-\n"
        "                              sequence and does not run).\n"
        "      --kv-commit N           positions BACKED with VRAM at load (default 65536). The\n"
        "                              rest of --kv-size costs no VRAM until the sequence reaches\n"
        "                              it, and the difference goes to the expert slab: 621 more\n"
        "                              resident experts, worth 11.5%% of decode cold and 3.7%%\n"
        "                              once placement has converged.\n"
        "      --kv-margin N           positions the grower keeps backed ahead of the sequence\n"
        "                              (default 16384). It runs on its own thread, never on the\n"
        "                              dispatch path.\n"
        "\n"
        "Prefix cache\n"
        "      --prefix-cache-dir D    keep the attention state for served prefixes under D, on the\n"
        "                              SSD. Off unless given. An agent session re-sends its whole\n"
        "                              transcript every turn, and without this the engine re-prefills\n"
        "                              every earlier turn every time. D must be on a filesystem that\n"
        "                              supports O_DIRECT.\n"
        "      --prefix-cache-gib N    how much of D to use. The default sizes itself from the free\n"
        "                              space on D and caps at 64. Blocks are shared between\n"
        "                              conversations; checkpoints are not, and take a quarter.\n"
        "      --prefix-cache-block N  tokens a block (default 512). Must divide evenly by the\n"
        "                              largest compress ratio. Larger reads faster and shares less.\n"
        "      --persistent-prefix-cache\n"
        "                              keep the store across restarts. Without it the files are\n"
        "                              unlinked at startup and the space returns when the process\n"
        "                              exits, however it exits. NOTE that a persistent store holds\n"
        "                              conversation content on disk, unencrypted, until it is\n"
        "                              evicted.\n"
        "      --prefix-cache-verify   print a hash of the attention state once each prompt is in.\n"
        "                              Two runs of the same conversation must print the same hash\n"
        "                              whether the state was prefilled or restored. Reads the whole\n"
        "                              live cache off the cards, so it is a check and not a mode to\n"
        "                              serve in.\n"
        "\n"
        "Serving\n"
        "      --host ADDR             bind address (default 127.0.0.1)\n"
        "      --port N                bind port (default 8080)\n"
        "      --max-tokens-floor N    raise a request's max_tokens to at least N (default\n"
        "                              256000, 0 honours the caller). Reasoning and answer share\n"
        "                              one budget, so a client that hardcodes a small max_tokens\n"
        "                              gets a truncated reasoning block and no answer.\n"
        "\n"
        "One-shot generation\n"
        "  -p, --prompt TEXT           generate once to stdout instead of serving\n"
        "      --prompt-file F         read the prompt from F; needed past ~30K tokens, where\n"
        "                              -p \"$(cat F)\" dies on ARG_MAX\n"
        "  -n, --n-predict N           tokens to generate (default 32)\n"
        "      --ignore-eos            keep generating past EOS, so -n N really is N tokens\n"
        "      --no-echo               do not echo the prompt\n"
        "      --ui auto|dash|plain    the live dashboard: devices, throughput, acceptance, the\n"
        "                              expert plane and where the time goes, redrawn in place.\n"
        "                              `auto` draws it when stdout is a terminal and prints the\n"
        "                              ordinary line output when it is not, so a piped or scripted\n"
        "                              run is unchanged. `dash` fails rather than downgrade.\n"
        "                              Keys: 1-4 or tab switch the lower pane, q stops after the\n"
        "                              current block.\n"
        "      --temp F                sampling temperature (default 0, greedy). Greedy is\n"
        "                              reproducible but degenerates, and it changes which experts\n"
        "                              route — take throughput and acceptance figures at real\n"
        "                              settings with --seed instead.\n"
        "      --top-k N               keep only the N highest-probability tokens (0 = off)\n"
        "      --top-p F               nucleus sampling threshold (default 1.0)\n"
        "      --min-p F               keep tokens scoring at least F times the top (default 0)\n"
        "      --seed N                RNG seed. Reproducible only under --placement-strategy\n"
        "                              static: the mover changes residency, and residency sets the\n"
        "                              summation order.\n"
        "\n"
        "Performance\n"
        "      --prefill-chunk-size N  tokens per batched prefill chunk (default 1408, 0 = off),\n"
        "                              rounded up to a multiple of 128. Bigger amortises the\n"
        "                              routed experts further but leaves fewer of them resident.\n"
        "      --placement-strategy S  routed-expert placement: heat (default; LFU with decay,\n"
        "                              promotes the hottest pooled expert), demand (promotes one\n"
        "                              the dispatch just streamed) or static (nothing moves, the\n"
        "                              control arm). Demand cuts misses a dispatch but spends more\n"
        "                              link doing it: a swap costs two shards and a miss costs one.\n"
        "      --placement-moves N     swaps started per layer dispatch (default 8; above 8 the\n"
        "                              guard binds before the budget does)\n"
        "      --placement-decay F     heat decay per dispatch (default 0.98, half-life ~200\n"
        "                              tokens)\n"
        "      --placement-hysteresis F  promote only at this heat ratio (default 1.25)\n"
        "      --placement-min-gain F  ...and by at least this many activations (default 1.5).\n"
        "                              Link traffic is U-shaped in this and the default is near the\n"
        "                              bottom; raising it moves more bytes, not fewer.\n"
        "      --placement-lead N      hold the host within N layer dispatches of the cards\n"
        "                              (default 2; 0 = unbounded). A freed slot returns to the free\n"
        "                              list only when its quarantine event signals, and that event\n"
        "                              is on the compute stream — so an unbounded lead means the\n"
        "                              mover runs out of slots to move into and issues a quarter of\n"
        "                              what it should. Worth 2.6-28.6%% of decode. Decode only.\n"
        "      --placement-shadow N    spare slab slots for in-flight promotions (default 32),\n"
        "                              taken from free VRAM rather than from residency\n"
        "      --keepalive-us U        per-card heartbeat interval; 0 = off (default). See\n"
        "                              src/gpu/keepalive.h\n"
        "\n"
        "Diagnostics\n"
        "      --prefill-reps N        re-run the prefill N times in ONE process and report the\n"
        "                              median and spread, discarding the first as warm-up. Timing\n"
        "                              one prefill behind the cold load folds the load's own\n"
        "                              variance into the result; this takes the load out.\n"
        "                              THE PLACEMENT ENGINE CONVERGES ACROSS REPS, so a\n"
        "                              one-prefill-per-process number is a COLD-placement number\n"
        "                              and understates a long-lived server. Quote the plateau.\n"
        "                              Prints a logits checksum per rep: constant under\n"
        "                              --placement-strategy static (the reset is complete), and\n"
        "                              drifting under heat (the mover is working).\n"
        "      --prefill-alt FILE      rotate a second prompt through --prefill-reps, so\n"
        "                              convergence is measured against a union of two expert sets\n"
        "                              rather than the single set one prompt converges onto.\n"
        "      --route-trace FILE      record every layer dispatch: the router's selection and the\n"
        "                              experts placement had to stream. Off the hot path, ~1 MB per\n"
        "                              512 tokens. The record format is documented at the top of\n"
        "                              src/engine/route_trace.h.\n"
        "                              Forces the HOST-built dispatch, since it records host-side\n"
        "                              facts, so its tok/s is not the shipping rate.\n"
        "      --expert-profile FILE   use this expert-popularity profile instead of the\n"
        "                              container's, for both the VRAM slab and the host pool.\n"
        "                              A magic, two dimensions, then n_layer*n_expert float32,\n"
        "                              layer-major. The container's own is derived from the\n"
        "                              imatrix, which ranks weights by importance rather than by\n"
        "                              how often the router picks them.\n"
        "      --dump-logits K         print the top-K logits after --tokens and exit\n"
        "      --tokens a,b,c          explicit token ids for --dump-logits\n"
        "      --logits-out FILE       also write the full float32 logit vector\n"
        "      --dump-tokens           print the token ids for -p and exit (feeds --tokens)\n"
        "      --force-tokens a,b,c    decode these ids instead of the sampler's, cycling as\n"
        "                              needed. Decode speed depends on which experts each token\n"
        "                              routes to, so two builds that generate different text report\n"
        "                              different tok/s for reasons unrelated to either.\n"
        "      --no-routed-experts     zero the routed experts to isolate the dense path. THE\n"
        "                              OUTPUT IS NOT THE MODEL'S — most of the FFN is gone.\n"
        "      --index-topk K          override the indexer's top-k. Oracle use only: the shipped\n"
        "                              512 needs ~2048 positions before the indexer engages, which\n"
        "                              no numpy reference reaches. Pass the same K to\n"
        "                              ref_forward.py --index-topk or the comparison is meaningless.\n"
        "      --max-layers N          stop after N layers; pairs with ref_forward.py --layers.\n"
        "                              THE OUTPUT IS NOT THE MODEL'S — a truncated network is\n"
        "                              fluent and wrong.\n"
        "\n"
        "Environment\n"
        "      AFF_PROFILE=1           print the per-phase breakdown for prefill and decode\n"
        "      AFF_BLOCK=1             print the host blocking table alone, without AFF_PROFILE's\n"
        "                              readback (which would force the drain it measures)\n"
        "      AFF_FORCE_RESIDENT=1    restrict routing to VRAM-resident experts. The output is\n"
        "                              wrong on purpose; this measures the all-resident ceiling.\n"
        "      AFF_TRACE=dense[,events][,deep][,sync][,evsync]  per-entry-point device timing\n"
        "      AFF_MOVER_DUP=K         re-issue each mover shard copy K times (default 1). Same\n"
        "                              placement decisions and same text, K times the mover's link\n"
        "                              bytes — which is how what a move costs gets measured\n"
        "                              without changing what the engine decides.\n"
        "      AFF_MOVER_DUP_DIR=both|h2d|d2h   which direction AFF_MOVER_DUP duplicates (default\n"
        "                              both). Promotions read host RAM, demotions write it;\n"
        "                              duplicating both at once measures their sum and cannot say\n"
        "                              whether the two directions share one PCIe budget.\n"
        "      AFF_LOGICAL_RANKS=N     run N tensor-parallel ranks over however many cards are\n"
        "                              present, dealt round-robin. A CORRECTNESS harness for\n"
        "                              rank counts the machine cannot supply — the ranks then\n"
        "                              share a card, so timings from it mean nothing, and each\n"
        "                              still pays its own VRAM. Needs GPU_MAX_HW_QUEUES above\n"
        "                              the ranks-per-card, or the collective deadlocks.\n"
        "      AFF_XRANK_INJECT=N      perturb one rank's cross-rank digest at sample N, so the\n"
        "                              check that compares the two cards against each other can be\n"
        "                              shown to fire. A check reporting zero checks is a failed\n"
        "                              check.\n"
        "\n"
        "Serves /v1/chat/completions, /v1/completions and /v1/models.\n");
      return 0;
    }
    // Anything unrecognised is fatal. Silently ignoring it means a mistyped flag runs the default
    // and says nothing, which in an A/B measures one arm twice.
    else {
      aff::ui::err("error: unrecognised argument '%s'. Try --help.\n", a.c_str());
      return 2;
    }
  }
  if (model_path.empty()) { aff::ui::err("error: -m/--model is required\n"); return 2; }

  std::string err;
  // Before the model, because it needs nothing but the tokenizer and the container is 101 GiB.
  if (dump_tokens) {
    Tokenizer t;
    if (!t.load(tok_path, &err)) { aff::ui::err("tokenizer: %s\n", err.c_str()); return 1; }
    const std::vector<uint32_t> ids = t.encode(prompt, true);
    std::string csv;
    for (size_t i = 0; i < ids.size(); ++i) csv += sfmt("%s%u", i ? "," : "", ids[i]);
    aff::ui::out("%s\n", csv.c_str());
    return 0;
  }
  Model model;
  if (index_topk) model.set_index_topk(index_topk);
  if (max_layers) model.set_max_layers(max_layers);
#ifdef AFF_WITH_HIP
  // Dense goes to VRAM first; placement sizes the expert slabs from what is left. Reversed,
  // placement claims nearly all of it and every dense tensor falls back to the CPU.
  DenseGpu dense_gpu;
  // The devices backing ranks 0..n-1, resolved by dense_gpu.init and handed to both
  // StaticPlacements so every tier agrees on which card a rank is.
  std::vector<int> rank_plan;
  PrefixCache prefix_cache;
  // Kept past the setup block because --prefix-cache-verify hashes exactly these, in this order.
  std::vector<KvRegion> pcache_regions;
  VramLedger vram;
  {
    std::string derr;
    // Bounds for the staging buffers: the widest dense row (vocab) and the widest input (8*o_lora).
    // The ledger's baseline is taken BEFORE this: hipMalloc here is the first thing that brings up
    // a device context, and the context is hundreds of MiB that belongs to no phase.
    vram.begin();
    if (dense_gpu.init(129280 + 4096, 32768, &derr, kGpuCount)) {
      model.set_dense_ops(dense_gpu.ops());
      // Placement's slabs must land on the cards the dense ranks read from, in the same order. It
      // would otherwise enumerate for itself, which agrees only as long as nothing has capped or
      // reordered the plan. Both StaticPlacements are declared further down, so the plan is carried
      // to them rather than applied here.
      rank_plan.resize(dense_gpu.ranks());
      for (size_t r = 0; r < rank_plan.size(); ++r) rank_plan[r] = dense_gpu.device(r);
      // ---- WHERE THE RANK COUNT ACTUALLY STOPS -----------------------------------------------
      //
      // Past two cards the engine runs to completion and emits DEGENERATE TEXT — a handful of tokens
      // repeated. It is not the expert split, which is checked byte-for-byte and arithmetically at
      // four ways by tests/test_expert_shard.hip; not the collective, which is bit-identical across
      // ranks at 1, 3, 4 and 8; not the flash kernel, which matches its reference at every per-card
      // head count; and not numerical drift, which is LARGER between one card and two — both of
      // which are correct — than between two and four. The cause is not known.
      //
      // Refused here rather than left to be found in the output, because fluent-and-wrong is the
      // failure mode this project can least afford.
      const char* allow = std::getenv("AFF_ALLOW_NRANK");
      if (dense_gpu.ranks() > 2 && !(allow && *allow && *allow != '0')) {
        aff::ui::err("fatal: %zu GPUs were found and this engine supports 1 or 2. The rank count is "
                     "a runtime value throughout, and the expert split, the collective and "
                     "attention are all tested past two — but the engine as a whole emits "
                     "degenerate text at more than two ranks and the cause is not yet known.\n"
                     "Use --gpus 2 (or --gpus 1), or HIP_VISIBLE_DEVICES to choose which.\n"
                     "AFF_ALLOW_NRANK=1 runs it anyway, for working ON the fault. The output is "
                     "wrong; nothing measured under it means anything.\n",
                     dense_gpu.ranks());
        return 1;
      }
      aff::ui::out("tensor parallel across %zu GPU%s\n", dense_gpu.ranks(),
                   dense_gpu.ranks() == 1 ? "" : "s");
    } else {
      // Fatal, not a warning. Without the dense path the whole forward pass falls to the CPU
      // reference, which is orders of magnitude slower — it looks like a working engine having a
      // bad day rather than a configuration that cannot work. Say what was found and stop.
      aff::ui::err("fatal: %s\n", derr.c_str());
      aff::ui::err("Check `rocminfo | grep gfx` against the architecture this binary was built "
                   "for.\n");
      return 1;
    }
    vram.mark("hip context + staging");
  }
#endif

  // ---- the live view ---------------------------------------------------------------------------
  //
  // Started here and not earlier: the cards' PCI addresses are what the sysfs probe resolves its
  // nodes from, and asking for them before the runtime has a context would take the VRAM ledger's
  // baseline with it. Nothing prints between the context coming up and this point.
  //
  // From here on every banner goes through ui::log, which draws it into the console pane while the
  // dashboard is up and prints exactly the bytes it always did when it is not.
  aff::ui::Dashboard& dash = aff::ui::dashboard();
#ifdef AFF_WITH_HIP
  for (const GpuDeviceDesc& g : gpu_devices())
    dash.add_device({g.name, g.bus_id, g.arch, g.cus, g.vram_total_gib});
#endif
  if (!dash.start(ui_mode) && ui_mode == aff::ui::UiMode::Dash) {
    // Not a silent downgrade. A flag that quietly does nothing is indistinguishable from one that
    // worked, which is the failure this project has spent the most time on.
    std::fprintf(stderr, "error: --ui dash needs stdout to be a terminal.\n");
    return 2;
  }
  // Every fatal path out of main is a `return`, and there are dozens. One guard rather than a
  // teardown at each: without it a configuration error prints into a pane that is then destroyed
  // with the alternate screen, and the operator gets a restored terminal, a non-zero status and no
  // reason. Idempotent, so the ordinary path's explicit stop() before the summary still wins.
  // Joined before anything it touches is destroyed. It maps VRAM on its own schedule, so leaving
  // it running into teardown is a use-after-free with a driver call in it.
  struct GrowGuard {
    aff::DenseGpu* g;
    ~GrowGuard() { if (g) g->kv_autogrow_stop(); }
  };
  struct DashGuard {
    ~DashGuard() { aff::ui::dashboard().stop(); }
  } dash_guard;
  GrowGuard grow_guard{nullptr};
  dash.stage(aff::ui::Stage::LoadDense, "dense weights");
  if (!model.load(model_path, &err)) { aff::ui::err("load: %s\n", err.c_str()); return 1; }
#ifdef AFF_WITH_HIP
  vram.mark("dense weights");
#endif
  const ModelConfig& c = model.config();
  aff::ui::out("model: %u layers, %u experts (top-%u), hidden %u, vocab %u\n",
              c.n_layer, c.n_expert, c.n_expert_used, c.n_embd, c.vocab);
  {
    // The geometry never changes after this, so the header reads it once.
    const std::string stem = model_path.substr(model_path.find_last_of('/') + 1);
    dash.bus().publish_sync([&](aff::ui::Snapshot& w) {
      w.model = drop_ext(stem);
      w.n_layer = c.n_layer;
      w.n_expert = c.n_expert;
      w.top_k = c.n_expert_used;
      w.n_embd = c.n_embd;
      w.vocab = c.vocab;
      w.spec_width = dspark_on ? c.dspark_block : 0u;
      w.experts_total = (uint64_t)c.n_layer * c.n_expert;
      const AffReader& r = model.reader();
      if (r.layer_count()) w.quant = quant_name((AffQuant)r.layer(0).gate.quant);
#ifdef AFF_WITH_HIP
      w.ranks = (uint32_t)dense_gpu.ranks();
#endif
    });
  }

  // Required whenever the checkpoint declares one, with no fallback path. A draft that failed to
  // bind does not crash — it emits ids that are never accepted, so decode still produces correct
  // text at roughly its old speed, and the only symptom is a number reading "speculation does not
  // pay on this model". That is indistinguishable from a real result. Exiting keeps it distinct.
  if (c.dspark_block && dspark_on) {
    const std::string dsp = dspark_path.empty()
        ? drop_ext(model_path) + ".dspark.aff" : dspark_path;
    std::string derr;
    if (!model.load_dspark(dsp, &derr)) {
      aff::ui::err("dspark: %s\n       (%s)\n"
                           "       the checkpoint declares dspark_block_size=%u, so the draft is\n"
                           "       required. Write it with:\n"
                           "         aff-quantize --src <ckpt> --out %s --dspark --codebook %s\n",
                   derr.c_str(), dsp.c_str(), c.dspark_block, dsp.c_str(), model_path.c_str());
      return 1;
    }
    const DsparkWeights& d = model.dspark();
    std::string taps;
    for (int32_t t : c.dspark_taps) taps += sfmt(" %d", t);
    aff::ui::out("dspark: %zu stages, block %u, taps%s, %.2f GiB\n", d.stage.size(),
                 c.dspark_block, taps.c_str(), model.dspark_bytes() / 1073741824.0);
#ifdef AFF_WITH_HIP
    vram.mark("draft dense weights");
#endif
  }

  Tokenizer tok;
  if (!tok_path.empty() && !tok.load(tok_path, &err))
    aff::ui::err("warning: tokenizer: %s\n", err.c_str());


#ifdef AFF_WITH_HIP
  {
    // Compressed rows per layer, by the same rule Model::init_state uses: a layer at ratio r emits
    // a row every r positions, ratio 0 emits none. Only 21 of 43 layers are ratio 4, and sizing all
    // of them for it costs 2.76 GiB a card at a 1M cache.
    std::string kerr;
    // The draft's stages get KV slots immediately after the model's layers. They are ratio 0, so
    // they cost one sliding window each and no compressed cache; sharing the array is what lets the
    // draft call kv_commit and attend unchanged.
    const uint32_t n_stage = (uint32_t)model.dspark().stage.size();
    std::vector<uint32_t> comp_rows(c.n_layer + n_stage, 0u);
    uint32_t max_comp = 0;
    uint64_t comp_total = 0;
    for (uint32_t l = 0; l < c.n_layer; ++l) {
      const uint32_t r = c.ratio_for(l);
      comp_rows[l] = r ? (uint32_t)(kMaxKvPositions / r + 2) : 0u;
      max_comp = std::max(max_comp, comp_rows[l]);
      comp_total += comp_rows[l];
    }
    std::string herr;
    if (!dense_gpu.hc_init(c.n_embd, c.hc_mult, &herr))
      aff::ui::out("hidden state: on host (%s)\n", herr.c_str());
    // FP8R is the only layout the flash attention kernel reads; the enum's other arms exist for
    // aff-verify's reference paths, not for a runtime choice.
    if (kSlots > 1) {
      if (kKvCommitPositions != kMaxKvPositions) {
        aff::ui::fatal("--slots %u needs --kv-commit == --kv-size (%llu != %llu): a slot is backed "
                       "in full at load, because the KV grower tracks one sequence.\n",
                       kSlots, (unsigned long long)kKvCommitPositions,
                       (unsigned long long)kMaxKvPositions);
        return 1;
      }
      if (!dense_gpu.set_n_slots(kSlots)) {
        aff::ui::fatal("--slots %u out of range (1..16)\n", kSlots);
        return 1;
      }
      aff::ui::out("slots: %u sequences, %llu positions each (%llu total)\n", kSlots,
                   (unsigned long long)kMaxKvPositions,
                   (unsigned long long)kMaxKvPositions * kSlots);
    }
    // The prefix cache holds single-sequence design (a shared writer thread, stage pool and index)
    // and poisons draft acceptance for one slot under two interleaved sequences — a concurrency bug
    // isolated but not yet fixed (repro: SLOTS-STAGE3 findings). Correct behaviour beats a fast
    // wrong one, so multi-slot serving runs WITHOUT it. Single-slot is unaffected and keeps it.
    if (kSlots > 1 && !pcache.dir.empty()) {
      aff::ui::err("prefix cache: disabled under --slots %u (single-sequence design; multi-slot "
                   "poisons draft acceptance — see SLOTS-STAGE3). Single-slot keeps it.\n", kSlots);
      pcache.dir.clear();
      pcache_verify = false;
    }
    if (dense_gpu.attn_init(c.n_layer + n_stage, c.n_head, c.head_dim, c.rope_dim, c.sliding,
                            comp_rows.data(), KvDtype::FP8R, kMaxKvPositions, kKvCommitPositions,
                            &kerr))
      aff::ui::out("kv cache: %llu positions reserved, %llu backed with %.2f GiB of VRAM "
                  "(%u raw/layer, widest %u of %llu comp rows)\n",
                  (unsigned long long)kMaxKvPositions, (unsigned long long)kKvCommitPositions,
                  dense_gpu.stats().kv_bytes / 1073741824.0,
                  c.sliding, max_comp, (unsigned long long)comp_total);
    else
      aff::ui::out("kv cache: on host (%s)\n", kerr.c_str());
    // The grower runs from here until shutdown. Margin 16384 positions: decode advances ~4.4 a
    // block, so it fires about once every 3700 blocks and always long before the position that
    // needs the memory.
    dense_gpu.kv_autogrow_start(kKvMarginPositions);
    grow_guard.g = &dense_gpu;
    vram.mark("kv cache");
    // After attn_init, because the batch arena sizes its mask staging from the compressed cache;
    // before StaticPlacement, because ~170 MiB of activations must come out of the VRAM the expert
    // slabs are sized against rather than out from under them.
    std::string berr;
    if (prefill_chunk > 0) {
      // max_mid is the widest intermediate that is not the query block: 8*o_lora_rank.
      if (dense_gpu.batch_init(prefill_chunk, c.n_embd, c.hc_mult, c.n_head, c.head_dim,
                               8 * c.o_lora_rank, c.vocab, &berr)) {
        model.set_batch_ops(dense_gpu.batch_ops());
        aff::ui::out("prefill: batched, up to %u tokens per chunk\n", dense_gpu.batch_cap());
      } else {
        aff::ui::out("prefill: one token at a time (%s)\n", berr.c_str());
      }
      vram.mark("prefill arena");
    }
    // ---- refuse to run degraded ------------------------------------------------------------------
    //
    // A dense tensor that did not fit stays on the CPU, and the engine then comes up "successfully":
    // the KV cache falls to the host, prefill drops to one token at a time, and the first request dies
    // somewhere unrelated — the observed shape was `invalid device ordinal` from a placement whose
    // device list never got built. Hours can go into that before anyone scrolls back to the load log.
    //
    // The usual cause is another affinity still holding VRAM, because a process exits before the
    // driver releases its memory. So the message says that, and the run stops here where the reason is
    // still on screen. This is the no-fallback rule the rest of the engine follows: a half-working
    // accelerator reads as an honest negative result, and it is not one.
    if (dense_gpu.stats().cpu_fallbacks) {
      size_t freeb = 0, totalb = 0;
      const std::vector<aff::VramSample> vs = aff::vram_sample();
      if (!vs.empty()) { freeb = (size_t)((vs[0].total_gib - vs[0].used_gib) * 1073741824.0);
                         totalb = (size_t)(vs[0].total_gib * 1073741824.0); }
      aff::ui::err("\nfatal: %llu dense tensors did not fit and were left on the CPU, so this engine "
                   "cannot run.\n"
                   "       card 0 has %.0f MiB free of %.0f MiB.\n"
                   "       The usual cause is another affinity still holding VRAM: a process exits "
                   "before\n"
                   "       the driver releases its memory, so a restart that only waits for the "
                   "process\n"
                   "       comes up starved. Check with `rocm-smi --showpids` and wait for the memory "
                   "to\n"
                   "       come back, not just for the pid to go.\n",
                   (unsigned long long)dense_gpu.stats().cpu_fallbacks,
                   freeb / 1048576.0, totalb / 1048576.0);
      return 1;
    }

    // The draft's arena, before placement like everything else. Fatal rather than a print: every
    // other path here has a slower thing to fall back to, and this one does not.
    if (n_stage) {
      std::string derr;
      if (!dense_gpu.draft_init(c.n_embd, c.n_embd * (uint32_t)c.dspark_taps.size(), c.vocab,
                                c.dspark_markov_rank, &derr)) {
        aff::ui::err("dspark: draft arena: %s\n", derr.c_str());
        return 1;
      }
      vram.mark("draft arena");
    }
    // Last, and still before StaticPlacement: the compressor's history and the indexer's key and
    // score planes otherwise appear on first use, mid-prefill, long after placement has handed the
    // leftover VRAM to expert shards. At 1M that is 2.56 GiB arriving after the fact.
    std::vector<uint32_t> ratios(c.n_layer);
    for (uint32_t l = 0; l < c.n_layer; ++l) ratios[l] = c.ratio_for(l);
    std::string rerr;
    if (!dense_gpu.reserve_runtime(c.n_layer, ratios.data(), c.head_dim, c.index_head_dim,
                                   c.index_n_heads, &rerr))
      aff::ui::out("runtime reservation: %s — expert placement will over-commit\n", rerr.c_str());
    vram.mark("compressor + indexer");

    // ---- the prefix cache ------------------------------------------------------------------------
    //
    // Last, because it enumerates the buffers everything above has just allocated. Nothing here
    // costs VRAM: the store is on the SSD and the staging arenas are host memory.
    if (!pcache.dir.empty() || pcache_verify) {
      std::vector<KvRegion>& regions = pcache_regions;
      auto add = [&](DenseGpu::KvPart part, uint32_t layer, uint32_t ratio) {
        uint64_t rb = 0, rows = 0;
        if (!dense_gpu.kv_part_geometry(part, layer, &rb, &rows)) return;
        KvRegion r;
        r.part = (uint8_t)part; r.layer = layer; r.row_bytes = rb; r.rows = rows; r.ratio = ratio;
        regions.push_back(r);
      };
      // A fixed order, because it is also the order of the bytes in a stored block. The draft's
      // stages are in the raw sweep: they share a_raw, so saving it saves the draft's conditioning
      // and there is nothing separate to keep.
      for (uint32_t l = 0; l < c.n_layer + n_stage; ++l) add(DenseGpu::KvPart::Raw, l, 0);
      for (uint32_t l = 0; l < c.n_layer; ++l) {
        const uint32_t r = c.ratio_for(l);
        if (!r) continue;
        add(DenseGpu::KvPart::Comp, l, r);
        add(DenseGpu::KvPart::Idx8, l, r);
        add(DenseGpu::KvPart::IdxScale, l, r);
        add(DenseGpu::KvPart::Hist, l, 0);
        add(DenseGpu::KvPart::IdxHist, l, 0);
      }
      // Everything that decides the bytes. --kv-size is in it even though it only sizes the
      // buffers: a store is cheap to rebuild and a store read under the wrong geometry is not
      // detectable from its contents.
      uint64_t fp = 1469598103934665603ull;
      auto mix = [&](uint64_t v) { for (int i = 0; i < 8; ++i) { fp ^= (v >> (8*i)) & 0xFF; fp *= 1099511628211ull; } };
      mix((uint64_t)KvDtype::FP8R); mix(c.n_layer); mix(n_stage); mix(c.head_dim);
      mix(c.rope_dim); mix(c.sliding); mix(c.n_head); mix(c.index_head_dim);
      mix(kMaxKvPositions); mix(pcache.block_tokens); mix(model.reader().header().file_size);
      for (uint32_t l = 0; l < c.n_layer; ++l) mix(c.ratio_for(l));

      KvIo pio; pio.save = pc_save; pio.load = pc_load; pio.ctx = &dense_gpu;
      // Size the default from what the filesystem actually has rather than assuming a large SSD:
      // the store is preallocated, so a fixed default either fails to open or fills the disk.
      // A quarter of the free space, capped, and never more than is there.
      if (!pcache.budget_bytes && !pcache.dir.empty()) {
        struct statvfs vfs {};
        if (statvfs(pcache.dir.c_str(), &vfs) == 0) {
          const uint64_t freeb = (uint64_t)vfs.f_bavail * (uint64_t)vfs.f_frsize;
          pcache.budget_bytes = std::min<uint64_t>(freeb / 4, 64ull << 30);
        } else {
          pcache.budget_bytes = 8ull << 30;
        }
        aff::ui::err("prefix cache: sized to %.1f GiB from free space on %s; set "
                     "--prefix-cache-gib to choose.\n",
                     pcache.budget_bytes / 1073741824.0, pcache.dir.c_str());
      }
      pcache.ranks = (uint32_t)dense_gpu.ranks();
      std::string cerr_;
      if (pcache.dir.empty()) {
        // --prefix-cache-verify alone: the regions are what it hashes, and there is no store.
      } else if (!prefix_cache.open(pcache, regions, pio, fp, &cerr_)) {
        aff::ui::err("fatal: prefix cache: %s\n", cerr_.c_str());
        return 1;
      }
      else aff::ui::out("prefix cache: %s, %.1f GiB, %u-token blocks (%.2f MiB each, %u slots), "
                  "checkpoint %.2f MiB (%u slots)%s\n",
                  pcache.dir.c_str(), pcache.budget_bytes / 1073741824.0, pcache.block_tokens,
                  prefix_cache.block_bytes() / 1048576.0, prefix_cache.block_slots(),
                  prefix_cache.ckpt_bytes() / 1048576.0, prefix_cache.ckpt_slots(),
                  pcache.persistent ? ", persistent" : "");
    }
  }
  aff::ui::out("dense on gpu: %llu tensors, %.2f GiB across %zu ranks (%.2f GiB/card) in %.1f s\n",
              (unsigned long long)dense_gpu.stats().tensors,
              dense_gpu.stats().bytes / 1073741824.0, dense_gpu.stats().ranks,
              dense_gpu.stats().bytes / 1073741824.0 / std::max<size_t>(1, dense_gpu.stats().ranks),
              dense_gpu.stats().upload_seconds);
  // The draft's expert pool, before the target's. It is its own container and so its own
  // StaticPlacement, and it goes first because the target's placement is what gives way — it is
  // sized from whatever VRAM is left.
  //
  // By default every draft expert is VRAM-resident. `--dspark-placement ram` pools the slab
  // instead, which is the same two-tier arrangement the target runs: a missing expert's tile points
  // into registered RAM and the GEMM streams it over PCIe. It is a trade, not a saving — see the
  // note on the flag.
  //
  // This is also why the draft moves the target's fingerprint: under AFF_FORCE_RESIDENT the text is
  // a function of which experts fit, and the draft takes some of them. So the A/B for speculation is
  // `--dspark off` against `--dspark on` in one binary with the draft loaded both times, never
  // against a build that never heard of it.
  //
  // Declared before `draft_placement` so it is destroyed after it, which set_host_pool requires.
  HostExpertPool draft_pool;
  StaticPlacement draft_placement;
  struct DrCtx { StaticPlacement* c; const ModelConfig* cfg; DenseGpu* g; };
  DrCtx drctx{nullptr, nullptr, nullptr};
  bool draft_wants_pool = false;
  if (c.dspark_block && dspark_on) {
    const uint32_t n_stage = (uint32_t)model.dspark().stage.size();
    // What the slab needs plus a little, never greedy: everything it takes is an expert the target
    // does not get.
    const double shard_gib = (double)model.dspark_reader().layer(0).stride /
                             (double)std::max<size_t>(1, dense_gpu.ranks()) / 1073741824.0;
    const double need = shard_gib * c.n_expert * n_stage * 1.05;
    // Under `ram` the slab is not empty but token: a placement with no slots at all cannot
    // initialise, and this is the smallest one that can. Those few experts are under a percent of
    // the draft and the pool holds the rest.
    constexpr uint32_t kMinDraftSlots = 8;
    const double want = dspark_ram ? shard_gib * kMinDraftSlots : need;
    draft_wants_pool = dspark_ram;
    std::string derr;
    draft_placement.set_rank_plan(rank_plan.data(), rank_plan.size());
    draft_placement.set_route_trace_tag('S');
    if (!draft_placement.init(&model.dspark_reader(), model.codebook(), want, n_stage, c.n_expert,
                              kSpecTok, c.n_expert_used, &derr)) {
      aff::ui::err("dspark: draft expert placement: %s\n", derr.c_str());
      return 1;
    }
    if (!draft_wants_pool &&
        draft_placement.stats().resident_experts != draft_placement.stats().total_experts) {
      aff::ui::err("dspark: only %llu of %llu draft experts fit in VRAM (%.1f GiB taken).\n"
                   "        Under --dspark-placement vram the draft has no host tier, so a partial\n"
                   "        slab is not a slower draft, it is a different one. Free VRAM, lower\n"
                   "        --kv-size, or ask for the tier with --dspark-placement ram.\n",
                   (unsigned long long)draft_placement.stats().resident_experts,
                   (unsigned long long)draft_placement.stats().total_experts,
                   draft_placement.stats().vram_used_gib);
      return 1;
    }
    vram.mark("draft expert slab");
    aff::ui::out("dspark: %llu/%llu draft experts resident, %.2f GiB VRAM in %.1f s\n",
                (unsigned long long)draft_placement.stats().resident_experts,
                (unsigned long long)draft_placement.stats().total_experts,
                draft_placement.stats().vram_used_gib, draft_placement.stats().load_seconds);
    drctx = {&draft_placement, &c, &dense_gpu};
    Model::DraftOps dops;
    dops.ctx = &drctx;
    dops.ffn_batch = [](void* v, uint32_t stage, const uint32_t* sel, const float* w,
                        uint32_t nb_tok, uint32_t k, uint8_t* handled) -> uint32_t {
      DrCtx* f = (DrCtx*)v;
      const float* d_x[kMaxRanks]; float* d_blk[kMaxRanks]; void* st[kMaxRanks];
      StaticPlacement::RouteDev rt[kMaxRanks];
      for (size_t r = 0; r < f->g->ranks(); ++r) {
        d_x[r] = f->g->batch_norm_dev(r); d_blk[r] = f->g->batch_block_dev(r);
        st[r] = f->g->stream(r);
        if (!d_x[r] || !d_blk[r]) return 0;
        // The SAME ring the target's dispatch reads, because it is the same router: the draft's
        // stages call `batch_router_topk`, which publishes every selection it computes. `route_dev`
        // hands back the slot of the LAST publish, and the draft's dispatch is the next thing the
        // host issues after the draft's own top-k, so that slot is this stage's.
        const DenseGpu::RouteDev d = f->g->route_dev(r);
        rt[r].sel = d.sel; rt[r].wt = d.wt; rt[r].ready = d.ready;
      }
      return f->c->run_batch_device(stage, sel, w, nb_tok, k, d_x, d_blk, f->g->batch_nb(),
                                    f->cfg->n_embd, f->cfg->moe_inter, f->cfg->swiglu_limit, st,
                                    handled, rt);
    };
    dops.dev_plan = [](void* v, uint32_t stage) {
      return ((DrCtx*)v)->c->device_dispatch(stage);
    };
    model.set_draft_ops(dops);
  }

  // Declared before `placement` and destroyed after it, which StaticPlacement::set_host_pool
  // requires ("must outlive this"). It is BUILT after placement, ~50 lines below, because it needs
  // placement's residency predicate — but locals destruct in reverse declaration order, so
  // declaring it there would destroy it first. The placement engine's mover writes into the pool
  // for the life of the run, so the ordering is load-bearing rather than tidy.
  HostExpertPool host_pool;

  StaticPlacement placement;
  std::string cerr;
  // Must outlive every forward pass: the model reads it on every expert FFN.
  struct FCtx { StaticPlacement* c; const ModelConfig* cfg; DenseGpu* g; };
  FCtx fctx{nullptr, nullptr, nullptr};
  // Before init(), because the shadow slots are part of the slab allocation. Reserved only for the
  // moving strategies: the static arm has to be byte-for-byte the old configuration, or the A/B
  // compares two amounts of free VRAM as well as two placements.
  if (pcfg.strategy != PlacementEngineConfig::Strategy::Static)
    placement.reserve_shadow_slots(placement_shadow);
  placement.set_driver_headroom_mib(gpu_headroom_mib);
  // ---- the expert profile override, and it must land BEFORE both consumers --------------------
  //
  // `placement.init` ranks the whole expert set to choose the VRAM residents, and `host_pool.init`
  // ranks it again to choose the pool's fill order. They partition the set between them, so they
  // must see the SAME profile — which is why the override lives behind `AffReader::profile()`
  // rather than being passed to either. Here is simply the last point before the first of them
  // runs. Fatal on a bad file: falling through to the container's profile would run the baseline
  // arm under the flag's name, which is the one failure an A/B cannot detect.
  if (!expert_profile_path.empty()) {
    std::vector<float> prof;
    std::string perr;
    if (!load_expert_profile(expert_profile_path, &prof, &perr)) {
      aff::ui::err("fatal: --expert-profile %s: %s\n", expert_profile_path.c_str(),
                   perr.c_str());
      return 1;
    }
    if (!model.reader_mut().set_profile_override(prof.data(), prof.size(), &perr)) {
      aff::ui::err("fatal: --expert-profile %s: %s\n", expert_profile_path.c_str(),
                   perr.c_str());
      return 1;
    }
    aff::ui::out("expert profile: %zu entries from %s, overriding the container's\n",
                prof.size(), expert_profile_path.c_str());
  }
  dash.stage(aff::ui::Stage::LoadExperts, "uploading the hottest experts");
  // The chunk has to reach placement: the routed arena scales with it, placement allocates that
  // arena before it sizes its slabs, and a chunk it never heard about aborts the run mid-prefill.
  placement.set_rank_plan(rank_plan.data(), rank_plan.size());
  if (placement.init(&model.reader(), model.codebook(), gpu_budget_mib / 1024.0, c.n_layer,
                     c.n_expert,
                     dense_gpu.batch_cap(), c.n_expert_used, &cerr)) {
    aff::ui::out("static placement: %llu/%llu experts resident (%.1f%%), %.1f GiB VRAM in %.1f s"
                " (%.2f GB/s)\n",
                (unsigned long long)placement.stats().resident_experts,
                (unsigned long long)placement.stats().total_experts,
                100.0 * (double)placement.stats().resident_experts / (double)placement.stats().total_experts,
                placement.stats().vram_used_gib, placement.stats().load_seconds,
                placement.stats().vram_used_gib * 1.0737 / std::max(0.001, placement.stats().load_seconds));
    vram.mark("routed expert slab");
    fctx = {&placement, &c, &dense_gpu};
    // Prefill and DSpark verify both reach the routed experts through this hook. It reads the chunk
    // arena and accumulates into the chunk block, both already in VRAM on each card, so it takes
    // nothing from the host and returns nothing to it. Anything it declines is fatal in the model.
    model.set_expert_ffn_batch([](void* v, uint32_t layer, const uint32_t* sel, const float* w,
                                  uint32_t nb_tok, uint32_t k, uint8_t* handled) -> uint32_t {
      FCtx* f = (FCtx*)v;
      const float* d_x[kMaxRanks]; float* d_blk[kMaxRanks]; void* st[kMaxRanks];
      StaticPlacement::RouteDev rt[kMaxRanks];
      for (size_t r = 0; r < f->g->ranks(); ++r) {
        d_x[r] = f->g->batch_norm_dev(r); d_blk[r] = f->g->batch_block_dev(r);
        st[r] = f->g->stream(r);
        if (!d_x[r] || !d_blk[r]) return 0;
        // The selection as card r can read it. Null unless the peer mirror was built, in which case
        // the dispatch keeps the host build — see run_batch_device's refusal banner.
        const DenseGpu::RouteDev d = f->g->route_dev(r);
        rt[r].sel = d.sel; rt[r].wt = d.wt; rt[r].ready = d.ready;
      }
      return f->c->run_batch_device(layer, sel, w, nb_tok, k, d_x, d_blk, f->g->batch_nb(),
                                    f->cfg->n_embd, f->cfg->moe_inter, f->cfg->swiglu_limit, st,
                                    handled, rt);
    }, &fctx);
    model.set_resident_pred([](void* v, uint32_t l, uint32_t e) {
      return ((StaticPlacement*)v)->resident(l, e);
    }, &placement);
    // The router asks this before it decides whether to bring the selection back to the host. The
    // dispatch asks the same function, so they cannot disagree about who owns it.
    model.set_device_dispatch_pred([](void* v, uint32_t l) {
      return ((StaticPlacement*)v)->device_dispatch(l);
    }, &placement);
    if (Model::force_resident_routing())
      aff::ui::out("AFF_FORCE_RESIDENT: routing restricted to VRAM-resident experts. "
                  "THE OUTPUT IS WRONG ON PURPOSE — this measures the all-resident ceiling.\n");
  } else {
    aff::ui::out("static placement: disabled (%s)\n", cerr.c_str());
  }

  // Same treatment as AFF_FORCE_RESIDENT: a flag that changes the arithmetic says so at runtime,
  // or its output gets quoted as the model's.
  if (no_routed)
    aff::ui::out("--no-routed-experts: the routed experts are zeroed. THE OUTPUT IS NOT THE "
                "MODEL'S — this isolates the dense path.\n");
  if (max_layers)
    aff::ui::out("--max-layers %u: the network is truncated. THE OUTPUT IS NOT THE MODEL'S.\n",
                max_layers);

  // Give the container's page cache back. Every tier has taken its copy by now — VRAM slabs, the
  // draft's slabs, the registered pool — and nothing reads the mapping again: the pool preads into
  // its own registered pages and an SSD-tier expert preads on demand.
  //
  // Two things: the unmapped page cache (most of the 108 GiB, seen as MemFree rising) and this
  // process's own page tables over the mapping, which take a MAP_FIXED remap rather than
  // MADV_DONTNEED — see AffReader::release_cache. The second is what decides how large a pool
  // hipHostRegister will take.
  //
  // It must come before placement_engine_init. With the cache still held the process sits close
  // enough to the machine's RAM that a larger prefill arena tips it into a SIGSEGV inside
  // libamdhip64 that reads as neither an OOM nor a crash.
  {
    auto meminfo = [](const char* key) -> double {
      std::FILE* f = std::fopen("/proc/meminfo", "r");
      if (!f) return 0.0;
      char line[256];
      double v = 0;
      const size_t n = std::strlen(key);
      while (std::fgets(line, sizeof line, f))
        if (!std::strncmp(line, key, n)) { v = std::strtod(line + n + 1, nullptr) / 1048576.0; break; }
      std::fclose(f);
      return v;
    };
    const double f0 = meminfo("MemFree:"), c0 = meminfo("Cached:");
    const uint64_t asked = model.reader().release_cache() + model.dspark_reader().release_cache();
    const double f1 = meminfo("MemFree:"), c1 = meminfo("Cached:");
    // stderr, not stdout: stdout is block-buffered to a pipe, so if the process crashes next this
    // line is lost and the transcript shows the pool registering and then nothing, so a crash after
    // the release reads as one before it.
    if (asked)
      aff::ui::err("page cache: released %.1f GiB of container mapping — free %.1f -> %.1f GiB, "
                   "cached %.1f -> %.1f GiB\n",
                   asked / 1073741824.0, f0, f1, c0, c1);
  }

  // The draft's host tier, BEFORE the target's. Both size themselves from MemAvailable, so whichever
  // runs first gets its bytes and the other adapts; the draft's claim is the small, fixed one and
  // the target's is "whatever is left", so this order is the one that cannot fail.
  //
  // Sized to cover the whole complement exactly, plus a staging ring. A draft dispatch selects at
  // most `dspark_block * n_expert_used` distinct experts, so a ring that size cannot be exhausted by
  // one — which matters because a staging slot is only released by the events of the layer that
  // claimed it, and a layer wanting more slots than exist cannot wait its way out.
  if (draft_wants_pool && draft_placement.enabled()) {
    const uint64_t missing = draft_placement.stats().total_experts -
                             draft_placement.stats().resident_experts;
    const uint64_t ring = (uint64_t)c.dspark_block * c.n_expert_used;
    const uint64_t stride = draft_placement.pool_out_stride();
    std::string perr;
    if (draft_pool.init(&model.dspark_reader(), (missing + ring) * stride,
                        [](void* v, uint32_t l, uint32_t e) {
                          return !((StaticPlacement*)v)->resident(l, e);
                        }, &draft_placement, (uint32_t)model.dspark().stage.size(), c.n_expert,
                        &perr, &StaticPlacement::pool_tf, &draft_placement, stride, ring,
                        (uint64_t)(host_pool_reserve_mib * 1048576.0)) &&
        draft_placement.set_host_pool(&draft_pool)) {
      const auto& dp = draft_pool.stats();
      aff::ui::out("dspark: %llu draft experts streamed from %.2f GiB of RAM (%llu on the ssd tier)\n",
                  (unsigned long long)dp.pooled, dp.bytes / 1073741824.0,
                  (unsigned long long)dp.ssd);
      if (dp.ssd) {
        // Not survivable: the draft's dispatch aborts on an expert it cannot reach, and an SSD read
        // in the drafting loop would cost more than the block it is drafting.
        aff::ui::err("dspark: %llu draft experts reached neither VRAM nor RAM. Free host memory,\n"
                     "        or run --dspark-placement vram.\n",
                     (unsigned long long)dp.ssd);
        return 1;
      }
    } else {
      aff::ui::err("dspark: draft host tier: %s\n",
                   perr.empty() ? "hipHostRegister declined the pool" : perr.c_str());
      return 1;
    }
  }

  if (placement.enabled() && host_pool_mib != 0.0) {
    uint64_t budget = 0;
    if (host_pool_mib > 0) {
      budget = (uint64_t)(host_pool_mib * 1048576.0);
    } else {
      // What the kernel says is available now, minus a reserve. MemAvailable already counts
      // reclaimable page cache, which is exactly the memory to convert into an owned pool; the
      // reserve is not optional because the machine brings up other work too.
      uint64_t avail_kb = 0;
      if (FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, f))
          if (std::sscanf(line, "MemAvailable: %llu kB", (unsigned long long*)&avail_kb) == 1) break;
        std::fclose(f);
      }
      const uint64_t reserve = (uint64_t)(host_pool_reserve_mib * 1048576.0);
      const uint64_t avail = avail_kb * 1024ull;
      budget = avail > reserve ? avail - reserve : 0;
    }
    std::string perr;
    // The pool holds DEVICE bytes, not container bytes. A missing expert is never copied anywhere:
    // its tile points into this pool and the GEMM streams the weights over PCIe as it computes
    // rather than stalling. That only works if the contents match what a VRAM slab would
    // hold — colgroup-major, sharded the same way — so placement's permutation runs on the pool's
    // reader threads and the slot stride is placement's, not the container's.
    // The SSD tier's staging ring. It must hold one LAYER's worth of staged experts at once — a
    // slot is only released by the events its layer's launches record, so a layer that needs more
    // slots than exist cannot wait its way out. `n_expert` is the hard bound on distinct experts a
    // single layer can select, which makes the ring exhaustion-proof by construction.
    //
    // The slots come off the pool's own budget, so they cost that many experts of residency. That
    // is the price of the tier being dispatchable at all, and it is only paid when the pool cannot
    // cover the complement — which the report at exit names explicitly.
    dash.stage(aff::ui::Stage::LoadPool, "reading the non-resident experts into RAM");
    if (host_pool.init(&model.reader(), budget,
                       [](void* v, uint32_t l, uint32_t e) {
                         return !((StaticPlacement*)v)->resident(l, e);
                       }, &placement, c.n_layer, c.n_expert, &perr,
                       &StaticPlacement::pool_tf, &placement, placement.pool_out_stride(),
                       c.n_expert, (uint64_t)(host_pool_reserve_mib * 1048576.0))) {
      const auto& hp = host_pool.stats();
      aff::ui::out("host pool: %llu experts in %.1f GiB of RAM in %.1f s (%.2f GB/s)"
                  ", %llu on the ssd tier, %.0f%% on 2 MiB pages\n",
                  (unsigned long long)hp.pooled, hp.bytes / 1073741824.0, hp.load_seconds,
                  hp.load_gbps, (unsigned long long)hp.ssd,
                  hp.bytes ? 100.0 * (double)hp.thp_bytes / (double)hp.bytes : 0.0);
      placement.set_host_pool(&host_pool);
    } else {
      aff::ui::out("host pool: disabled (%s) — the cpu fallback reads the mapping\n", perr.c_str());
    }
  }

  // The placement engine last: it needs both tiers final. Fatal on failure rather than falling back
  // to static — asking for --placement heat and silently getting static reads as "runtime placement
  // is not worth it", which is the same failure mode the dspark loader refuses.
  {
    std::string eerr;
    if (!placement.placement_engine_init(&host_pool, pcfg, &eerr)) {
      aff::ui::err("placement engine: %s\n", eerr.c_str());
      return 1;
    }
  }

  // Load is over; both budgets are final. Printed together because they are coupled: the KV cache
  // and the expert slabs share VRAM, and what does not fit there has to live in host RAM.
  vram.mark("placement engine (shadow slots)");
  vram.report();
  aff::ui::out("host RAM: %.2f GiB expert pool + %.2f GiB everything else (peak so far %.2f GiB)\n",
              host_pool.stats().bytes / 1073741824.0,
              std::max(0.0, peak_rss_gib() - host_pool.stats().bytes / 1073741824.0),
              peak_rss_gib());

  // ---- what the dashboard reads -----------------------------------------------------------------
  //
  // Called from the DISPATCH THREAD and nowhere else. The heat plane and the placement tables are
  // written by the mover on this thread, so this is the only thread allowed to read them; the bus
  // takes its lock with try_lock and gives up rather than wait, so a frame the renderer happens to
  // be holding costs a skipped publish and not a stalled dispatch.
  //
  // It is called whether or not a dashboard is running. With no reader the copy still happens, and
  // one code path that is always taken is worth more than a branch that is only exercised under a
  // flag nobody sets when it matters.
  auto publish_run = [&](uint64_t generated, uint64_t target, uint64_t prompt_tokens, uint64_t pos,
                         double decode_tok_s, double prefill_tok_s, double ms_block) {
    if (!dash.running()) return;
    dash.bus().publish([&](aff::ui::Snapshot& w) {
      const Model::PhaseProfile& p = model.profile();
      w.generated = generated;
      w.target = target;
      w.prompt_tokens = prompt_tokens;
      w.pos = pos;
      w.kv_capacity = kMaxKvPositions;
      w.decode_tok_s = decode_tok_s;
      w.prefill_tok_s = prefill_tok_s;
      w.ms_per_block = ms_block;
      w.blocks = p.blocks;
      w.drafted = p.drafted;
      w.accepted = p.accepted;
      // Marginal, one entry per position the draft has actually proposed at. Positions with no
      // observations are left out rather than drawn as zero, which reads as "never accepted".
      w.pos_valid = 0;
      for (uint32_t j = 0; j < kSpecTok && j < (uint32_t)aff::ui::kMaxSpecPos; ++j) {
        if (!p.pos_n[j]) break;
        w.pos_rate[j] = (double)p.pos_hit[j] / (double)p.pos_n[j];
        w.pos_valid = j + 1;
      }
      const PlacementEngineStats& es = placement.engine_stats();
      w.promotions = es.promotions;
      w.demotions = es.demotions;
      w.readmits = es.readmits;
      w.ssd_staged = es.ssd_staged;
      w.refused = es.refused;
      w.starved_slab = es.starved_slab;
      w.starved_pool = es.starved_pool;
      w.h2d_bytes = es.h2d_bytes;
      w.d2h_bytes = es.d2h_bytes;
      placement.snapshot_plane(&w.heat, &w.tier, &w.resident_per_layer);
      w.resident = 0;
      for (uint32_t r : w.resident_per_layer) w.resident += r;
      w.pool_gib = host_pool.stats().bytes / 1073741824.0;
      w.rss_gib = peak_rss_gib();

      w.phases.clear();
      if (aff::block_timed() && p.tokens) {
        const double n = (double)p.tokens;
        for (const auto& e : phase_rows(p))
          if (e.second.t > 0)
            w.phases.push_back({e.first, 1000.0 * e.second.t / n, e.second.wait_frac()});
        w.blocked_ms_per_token = 1000.0 * p.total_wait() / n;
      }
      w.drain_count = aff::drain_count().load(std::memory_order_relaxed);
      w.drain_max_ms = 1e-6 * (double)aff::drain_max_ns().load(std::memory_order_relaxed);
    });
  };
  publish_run(0, 0, 0, 0, 0, 0, 0);
  dash.stage(aff::ui::Stage::Ready, "loaded");

  // Prefill takes the prompt in one call and returns once; at a million positions that is minutes
  // with nothing on the screen, and the chunk hook is the only progress there is.
  //
  // ARMED ONLY AROUND A REAL PREFILL. The verify block is a forward_prefill too — one call a block,
  // a handful of tokens wide — so a hook left installed fires all through decode and leaves the
  // header reading "prefill 0 / 6 tokens" for the whole run.
  // A prompt jumps the position by thousands at once and can outrun any watermark, so prefill
  // commits for itself. This is the one place growth is synchronous, and it is between chunks
  // rather than inside a block.
  auto kv_reserve_for = [&](uint64_t upto) {
    std::string kerr;
    if (!dense_gpu.kv_commit_positions(upto, &kerr))
      aff::ui::fatal("kv cache: cannot back %llu positions: %s\n", (unsigned long long)upto,
                     kerr.c_str());
    dense_gpu.kv_note_position(upto);
  };

  auto watch_prefill = [&](bool on) {
    if (!on) { model.set_prefill_progress(nullptr, nullptr); return; }
    model.set_prefill_progress([](void* ctx, uint32_t done, uint32_t total) {
      auto* d = static_cast<aff::ui::Dashboard*>(ctx);
      d->stage(aff::ui::Stage::Prefill,
               std::to_string(done) + " / " + std::to_string(total) + " tokens",
               total ? (double)done / (double)total : -1.0);
    }, &dash);
  };

  // After the engine, so the residency map it writes is the one the first dispatch sees. Fatal if
  // the file cannot be opened: a --route-trace run that silently produces no trace is an hour of
  // GPU time spent on nothing.
  if (!route_trace_path.empty()) {
    if (!route_trace().open(route_trace_path, placement.n_layer(), placement.n_expert(),
                            c.n_expert_used)) {
      aff::ui::err("route trace: cannot open %s\n", route_trace_path.c_str());
      return 1;
    }
    std::vector<uint32_t> res;
    for (uint32_t l = 0; l < placement.n_layer(); ++l) {
      res.clear();
      for (uint32_t e = 0; e < placement.n_expert(); ++e)
        if (placement.resident(l, e)) res.push_back(e);
      route_trace().resident(l, res.data(), (uint32_t)res.size());
    }
    aff::ui::out("route trace: %s (%u layers, %u experts, top-%u)\n", route_trace_path.c_str(),
                placement.n_layer(), placement.n_expert(), c.n_expert_used);
  }

  // Not earlier: during the upload above the cards are already saturated and a heartbeat would only
  // take dispatch slots from it. See gpu/keepalive.h.
  GpuKeepalive keepalive;
  if (placement.enabled() && keepalive_us > 0 && !placement.devices().empty()) {
    keepalive.start(placement.devices(), keepalive_us);
    aff::ui::out("keepalive: %zu card(s), %.0f us heartbeat\n",
                placement.devices().size(), keepalive_us);
  }
#endif

  // Feeds an explicit token list and prints the top-k logits after the last one. This is what
  // tools/ref_forward.py compares against: an exact-id interface, so a tokenizer difference cannot
  // confound the comparison.
  if (dump_topk > 0) {
    std::vector<uint32_t> ids;
    for (size_t i = 0; i < token_list.size(); ) {
      const size_t c = token_list.find(',', i);
      ids.push_back((uint32_t)std::strtoul(token_list.substr(i, c - i).c_str(), nullptr, 10));
      if (c == std::string::npos) break;
      i = c + 1;
    }
    SeqState st;
    model.init_state(&st, ids.size() + 8);
    std::vector<float> logits;
    if (!model.forward_prefill(ids.data(), (uint32_t)ids.size(), &st, nullptr, nullptr, &logits,
                               !no_routed))
      for (uint32_t id : ids) model.forward_token(id, &st, nullptr, nullptr, &logits, !no_routed);
    std::vector<uint32_t> order(logits.size());
    for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
    std::partial_sort(order.begin(), order.begin() + dump_topk, order.end(),
                      [&](uint32_t x, uint32_t y) { return logits[x] > logits[y]; });
    double sum = 0.0, amax = 0.0;
    for (float v : logits) { sum += v; amax = std::max(amax, (double)std::fabs(v)); }
    // Truncated: the id list runs to thousands when the point is to reach the indexer, and echoing
    // it in full buries the numbers the comparison is about.
    aff::ui::out("engine layers=%u tokens=%zu (%.40s%s) routed=%d topk=%u\n", c.n_layer, ids.size(),
                token_list.c_str(), token_list.size() > 40 ? "..." : "", no_routed ? 0 : 1,
                c.index_topk);
    for (int i = 0; i < dump_topk; ++i)
      aff::ui::out("%6u %.6f\n", order[i], logits[order[i]]);
    aff::ui::out("checksum %.6f %.6f\n", sum, amax);
    if (!logits_out.empty()) {
      FILE* f = std::fopen(logits_out.c_str(), "wb");
      if (f) { std::fwrite(logits.data(), 4, logits.size(), f); std::fclose(f); }
    }
    return 0;
  }

  // The phase breakdown, for whichever half of the run just finished. A single dump at the end of
  // a run only ever describes decode, because prefill's counters are reset unread, which leaves the
  // longer of the two phases undecomposed.
  auto dump_profile = [&](const char* label) {
    const Model::PhaseProfile& p = model.profile();
    const double n = (double)std::max<uint64_t>(1, p.tokens), tot = p.total();
    aff::ui::out("\n%s phase breakdown (%llu tokens, %.2f ms/token measured, "
                "%.0f%% of it blocked on the cards)\n", label,
                (unsigned long long)p.tokens, 1000.0 * tot / n,
                tot > 0 ? 100.0 * p.total_wait() / tot : 0.0);
    const std::vector<std::pair<const char*, Model::PhaseProfile::Ph>> ph = phase_rows(p);
#ifdef AFF_WITH_HIP
    // Under static placement a non-zero ssd figure means the RAM budget did not cover the
    // complement. Under the placement engine it means the strategy put experts there by design.
    // Either way it should never be a surprise, which is what the counter is for.
    if (host_pool.enabled() && host_pool.stats().ssd_reads)
      aff::ui::out("  %-28s %8llu demand reads from disk (%llu experts on the ssd tier)\n",
                  "ssd tier:", (unsigned long long)host_pool.stats().ssd_reads,
                  (unsigned long long)host_pool.stats().ssd);
#endif
    // How many drains, and how long each waited: this separates "the host round trip is the
    // problem" from "the card has that much work to do", and the two want opposite fixes. A drain
    // averaging microseconds is a synchronisation to remove; one averaging milliseconds is a
    // dependency, and the only way to shorten it is to give the card less to do.
    //
    // The obvious reading of a 100%-wait phase is "stop synchronising", but layer L's routing
    // depends on layer L's activation, which depends on layer L-1's routed experts — the wait is
    // the model's own chain. Device-side dispatch removes the round trip on top (one host wakeup, a
    // 48 KB copy and seven launches, per layer per block rather than per token), not the chain.
    //
    // A delta since the previous dump: the counter is process-global and reset_profile() between
    // prefill and decode does not touch it.
    {
      static uint64_t seen = 0;
      const uint64_t all = aff::drain_count().load(std::memory_order_relaxed);
      const uint64_t nd = all - seen;
      seen = all;
      if (nd) {
        // The max and the over-10 ms count, not just the mean. A decode can contain a handful of
        // whole-machine stalls of hundreds of milliseconds each, which move the average by a
        // fraction of a millisecond and are invisible in it. Whether the engine has
        // them without a profiler attached decides whether they are the largest item in decode or
        // an artefact of measuring, and the mean cannot say which.
        aff::ui::out("  %-32s %9llu drains, %.3f ms each on average, worst %.1f ms at #%llu, "
                    "%llu over 10 ms\n",
                    "stream drains:", (unsigned long long)nd, 1000.0 * p.total_wait() / (double)nd,
                    1e-6 * (double)aff::drain_max_ns().load(std::memory_order_relaxed),
                    (unsigned long long)aff::drain_max_at().load(std::memory_order_relaxed),
                    (unsigned long long)aff::drain_slow().load(std::memory_order_relaxed));
        aff::drain_max_ns().store(0, std::memory_order_relaxed);
        aff::drain_slow().store(0, std::memory_order_relaxed);
        aff::drain_max_at().store(0, std::memory_order_relaxed);
      }
    }
    // ---- and the same total, split by WHERE ------------------------------------------------------
    //
    // The aggregate above answers "how much"; this answers "which", and only the second one can say
    // whether a wait was removed or merely moved. Two stalls under one name absorb each other:
    // remove either and the other takes on the identical wait, which against a single counter reads
    // as no change at all.
    //
    // A stream drain is one of at least eight ways this thread can stop. The rows that are not
    // drains are the ones a drain counter cannot see: the pinned ring's event sync, the SSD
    // tier's pread, and a `hipMemcpyAsync` from pageable memory, which is host-synchronous at these
    // sizes while looking like an async copy that cost nothing.
    //
    // Deltas since the previous dump, and zero rows are omitted — the target for each row is zero,
    // so a short table is the result and a long one is the work list.
    {
      static uint64_t seen_ns[(size_t)aff::BlockSite::kCount] = {};
      static uint64_t seen_ct[(size_t)aff::BlockSite::kCount] = {};
      bool any = false;
      for (size_t i = 0; i < (size_t)aff::BlockSite::kCount; ++i) {
        aff::BlockStat& b = aff::block_stats()[i];
        const uint64_t ct = b.count.load(std::memory_order_relaxed) - seen_ct[i];
        if (!ct) continue;
        const uint64_t ns = b.ns.load(std::memory_order_relaxed) - seen_ns[i];
        if (!any) {
          aff::ui::out("  %-32s %10s %9s %10s %9s\n", "blocked, by site:", "calls", "a token",
                      "ms/token", "worst ms");
          any = true;
        }
        aff::ui::out("  %-32s %10llu %9.1f %9.3f %9.1f\n", aff::block_site_name((aff::BlockSite)i),
                    (unsigned long long)ct, (double)ct / n, 1000.0 * 1e-9 * (double)ns / n,
                    1e-6 * (double)b.max_ns.load(std::memory_order_relaxed));
        b.max_ns.store(0, std::memory_order_relaxed);
      }
      for (size_t i = 0; i < (size_t)aff::BlockSite::kCount; ++i) {
        seen_ct[i] = aff::block_stats()[i].count.load(std::memory_order_relaxed);
        seen_ns[i] = aff::block_stats()[i].ns.load(std::memory_order_relaxed);
      }
    }
    // Read the wait column first. A row that is nearly all wall time is code to make faster; a row
    // that is nearly all wait is a queue, and what to fix is whatever is in it.
    aff::ui::out("  %-32s %10s %7s %10s\n", "", "ms/token", "share", "of it wait");
    for (const auto& e : ph)
      aff::ui::out("  %-32s %9.2f ms %6.1f%% %9.0f%%\n", e.first, 1000.0 * e.second.t / n,
                  tot > 0 ? 100.0 * e.second.t / tot : 0.0, 100.0 * e.second.wait_frac());
  };

  // What the block loop needs to turn its own counters into the header's numbers, and cannot know
  // for itself: which call it is inside, what that call is aiming at, and when it started. Set by
  // whichever caller is about to run a decode — the one-shot path and the server both.
  struct RunView {
    uint64_t prompt_tokens = 0, target = 0, base_hist = 0;
    double prefill_rate = 0;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  } run_view;
  // Assigned once the listener has a port; read by the request handler when it returns to idle.
  std::string serve_addr;

  // ---- greedy speculative decode, shared by one-shot and the server ------------------------------
  //
  // ONE COPY, shared by one-shot and the server. A second loop drifts from this one: the
  // single-token `forward_token` path aborts in the dense matvec the moment a shard is preshuffled
  // for the W8A8 GEMM, which every shipped container is, so a caller left on it fails on its first
  // generated token while a caller on the block path is fine. Two loops that are "the same decode"
  // until one of them is left behind is exactly that hazard.
  //
  // GREEDY, and not by omission: `BlockOut` returns each position's argmax and deliberately not its
  // logits (the block head's plane is kNarrowTok wide, and a block of logits is megabytes over
  // PCIe). Sampling would need the head to return them. Greedy speculation is bit-identical to
  // greedy decode by construction, so this is exactly temperature 0 — the caller is told rather
  // than quietly given something else.
  //
  // ---- the engine step lock ---------------------------------------------------------------------
  //
  // Sequence state is per SLOT since --slots, but the per-step scratch is not: one set of staging
  // buffers, one prefill arena, one indexer score plane. So requests may be in flight together and
  // their STEPS may not overlap. This is held across one step — a prefill chunk, or one speculative
  // block — and released around anything that waits on the network, so a slow reader cannot stall
  // the cards.
  std::mutex engine_mu;
  // `emit` takes each accepted token and returns false to stop. Returns how many were emitted.
  auto spec_decode = [&](uint32_t slot, SeqState& st, const Model::BlockOut& seed_blk, uint32_t first,
                         std::vector<uint32_t>& hist, uint32_t max_new, bool ignore_eos_,
                         const std::function<bool(uint32_t)>& emit,
                         const SamplerConfig* sc = nullptr,
                         // Teacher forcing, and a PARAMETER rather than a capture because this
                         // lambda is defined above where `force_ids` is parsed. Only ever non-null
                         // from the one-shot benchmark path; the server never forces.
                         const std::vector<uint32_t>* force = nullptr) -> uint32_t {
    const uint32_t B = dspark_on ? c.dspark_block : 0u;
    const bool draft_on = dspark_on && c.dspark_block > 0;
    // Sized from the checkpoint, not from B: dspark_draft writes dspark_block_size ids through this
    // pointer and takes no count, so the buffer's length must come from where the writer's does.
    std::vector<uint32_t> draft(c.dspark_block ? c.dspark_block : 1u), fed(B + 1);
    std::vector<uint8_t> match(B ? B : 1u);
    Model::BlockOut vb;
    // Sampling is the model card's operating point (temperature 1.0, top_p 0.95); greedy is what a
    // caller gets by passing nothing, and is still what the benchmarks that need a fixed stream use.
    std::mt19937_64 rng(sc ? sc->seed : 0ull);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    if (sc) {
      vb.sample = true;
      vb.temperature = sc->temperature;
      vb.top_p = sc->top_p;
      vb.top_k = sc->top_k;
      vb.min_p = sc->min_p;
    }
    const bool slot_dbg = std::getenv("AFF_SLOT_DEBUG") != nullptr;
    const bool n_slots_gt1 = dense_gpu.n_slots() > 1;
    uint64_t tap_hash_prev = 0, kv_hash_prev = 0;
    const uint32_t dbg_layer = dense_gpu.kv_layers_debug() ? dense_gpu.kv_layers_debug() - 1u : 0u;
    uint32_t seed_n = seed_blk.tap_rows, seed_pos0 = seed_blk.tap_pos0;
    uint32_t next = first, generated = 0;
    bool stop = false;
    while (generated < max_new && !stop) {
      // ONE BLOCK is one step. Taken here and dropped right after `rollback`, which is the last
      // thing in the iteration that touches the device: everything below it is host bookkeeping and
      // the emit callback, and emit writes to a socket.
      std::unique_lock<std::mutex> step(engine_mu);
      if (!dense_gpu.set_slot(slot)) {
        aff::ui::fatal("fatal: slot %u out of range\n", slot);
        std::abort();
      }
      if (slot_dbg && draft_on && seed_n) {
        const uint64_t now = dense_gpu.tap_hash_debug();
        if (tap_hash_prev && now != tap_hash_prev)
          aff::ui::err("slot %u: TAP CHANGED between blocks (%016llx -> %016llx)\n", slot,
                       (unsigned long long)tap_hash_prev, (unsigned long long)now);
        else if (tap_hash_prev)
          aff::ui::err("slot %u: tap intact across the gap (%016llx)\n", slot,
                       (unsigned long long)now);
        const uint64_t kvnow = dense_gpu.kv_hash_debug(dbg_layer);
        if (kv_hash_prev && kvnow != kv_hash_prev)
          aff::ui::err("slot %u: DRAFT KV CHANGED between blocks (%016llx -> %016llx)\n", slot,
                       (unsigned long long)kv_hash_prev, (unsigned long long)kvnow);
        else if (kv_hash_prev)
          aff::ui::err("slot %u: draft kv intact\n", slot);
      }
      if (draft_on && seed_n && !model.dspark_seed(seed_n, seed_pos0)) {
        aff::ui::fatal("fatal: dspark_seed refused %u positions at %u\n", seed_n, seed_pos0);
        std::abort();
      }
      // Position 0 leaves the draft's window nothing to look at, which happens only for a
      // single-token prompt. One unspeculated block gets past it.
      uint32_t nd = 0;
      if (draft_on && st.pos) {
        if (!model.dspark_draft(next, &st, draft.data())) {
          aff::ui::fatal("fatal: dspark_draft refused at position %u\n", st.pos);
          std::abort();
        }
        nd = B;
      }
      const uint32_t P = st.pos;
      if (slot_dbg && nd) {
        std::string ds;
        for (uint32_t i = 0; i < nd; ++i) ds += std::to_string(draft[i]) + (i + 1 < nd ? "," : "");
        aff::ui::err("slot %u DRAFT P=%u next=%u -> [%s]\n", slot, P, next, ds.c_str());
      }
      fed[0] = next;
      for (uint32_t i = 0; i < nd; ++i) fed[1 + i] = draft[i];
      // The proposals the target is asked to price, and the uniforms its draws use. Position j is
      // the target's prediction for fed[j+1], so that is the proposal at j; the last position has
      // none and is the bonus token.
      std::vector<double> accept_u(nd + 1);
      if (sc) {
        vb.query.assign(nd + 1, 0xFFFFFFFFu);
        vb.uniforms.assign(2u * (nd + 1), 0.0f);
        for (uint32_t j = 0; j < nd; ++j) vb.query[j] = fed[j + 1];
        for (uint32_t j = 0; j <= nd; ++j) {
          vb.uniforms[2 * j + 0] = (float)uni(rng);
          vb.uniforms[2 * j + 1] = (float)uni(rng);
          accept_u[j] = uni(rng);
        }
      }
      if (!model.forward_prefill(fed.data(), nd + 1, &st, nullptr, nullptr, nullptr, !no_routed,
                                 &vb)) {
        aff::ui::fatal("fatal: the verify block refused %u tokens at %u\n", nd + 1, P);
        std::abort();
      }
      // ---- how far the block is accepted ---------------------------------------------------------
      //
      // Greedy: the longest prefix the target would have produced itself, which makes the whole
      // thing bit-identical to greedy decode.
      //
      // Sampling: REJECTION SAMPLING, because "equals the argmax" is only the right test at
      // temperature 0. The draft proposes greedily, so its distribution is a point mass and the
      // proposal is accepted with probability p(x) under the target's own filtered distribution —
      // not whenever the two agree. On rejection the emitted token comes from the residual, which
      // for a point mass is the same distribution with the proposal removed. Together those make the
      // emitted stream the target's distribution exactly, which is the only reason speculation is
      // allowed to change nothing.
      //
      // Acceptance therefore falls against the greedy arm, and that is the price of sampling rather
      // than a regression.
      uint32_t k = 0;
      uint32_t reject_tok = 0;
      bool rejected = false;
      if (!sc) {
        while (k < nd && vb.greedy[k] == fed[k + 1]) ++k;
        for (uint32_t j = 0; j < nd; ++j) match[j] = vb.greedy[j] == fed[j + 1];
      } else {
        while (k < nd && accept_u[k] < (double)vb.draws[k].p_query) ++k;
        if (k < nd) { rejected = true; reject_tok = vb.draws[k].tok_excl; }
        for (uint32_t j = 0; j < nd; ++j) match[j] = accept_u[j] < (double)vb.draws[j].p_query;
      }
      if (slot_dbg && nd) {
        std::string gs;
        for (uint32_t i = 0; i < nd; ++i)
          gs += std::to_string(!sc ? vb.greedy[i] : vb.draws[i].tok) + (i + 1 < nd ? "," : "");
        aff::ui::err("slot %u VERIFY P=%u k=%u target=[%s]\n", slot, P, k, gs.c_str());
      }
      model.note_block(nd, k, match.data());
      // Positions P..P+k are real; P+k+1 holds vb.greedy[k] and is fed by the next block.
      model.rollback(&st, P + k + 1);
      if (n_slots_gt1) dense_gpu.sync_devices();   // no async work crosses into the next slot
      if (slot_dbg) { tap_hash_prev = dense_gpu.tap_hash_debug();
                      kv_hash_prev = dense_gpu.kv_hash_debug(dbg_layer); }
      step.unlock();                    // the cards are free from here; the rest is host-side
      seed_n = vb.tap_rows ? k + 1 : 0u;
      seed_pos0 = P;
      // The token at position k: the target's own, from whichever rule applied above.
      uint32_t tail = !sc ? vb.greedy[k] : (rejected ? reject_tok : vb.draws[k].tok);
      // ---- TEACHER FORCING, AND ONLY AT WIDTH 1 --------------------------------------------------
      //
      // Decode throughput is a property of the token stream as much as of the code: the token
      // decides which experts a step needs, and the resident fraction swings widely with it. Across
      // two RESIDENCIES that is not a nuisance, it is fatal to the measurement — a different
      // resident set changes the summation order, hence the emitted text, hence how well the draft
      // predicts it. Between two placements a few per cent apart in block cost, that drift in
      // tokens a block can be LARGER than the effect under test and point the other way.
      //
      // `!draft_on` is the whole safety argument and it is not a conservatism. With a draft running,
      // the block's KV was computed from the ids the draft PROPOSED; overriding the committed token
      // leaves the cache holding a continuation the model never produced, and every later position
      // attends to it. With `--dspark off` the block is one token wide, nd is 0, k is 0, and nothing
      // was proposed — so the forced token is simply the token, exactly as the (dead) single-token
      // loop below intended. Forcing `tail` rather than the emit expression covers the `next = tail`
      // that feeds the following block as well, which is the same value at width 1.
      if (force && !force->empty() && !draft_on) tail = (*force)[generated % force->size()];
      for (uint32_t i = 0; i <= k && generated < max_new; ++i) {
        next = (i < k) ? fed[i + 1] : tail;
        hist.push_back(next);
        ++generated;
        if (next == kEosToken && !ignore_eos_) { stop = true; break; }
        if (!emit(next)) { stop = true; break; }
      }
      // `next` must be the token at s->pos whatever the loop above did with the count.
      next = tail;
      // Once a block, not once a token: the placement plane is 44 KiB and a block is where every
      // counter on the screen actually moves. Both callers get it, so the server's dashboard is the
      // one-shot one.
      if (dash.running()) {
        const double el =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - run_view.t0).count();
        const uint64_t g = hist.size() > run_view.base_hist ? hist.size() - run_view.base_hist : 0;
        const uint64_t nb = model.profile().blocks;
        publish_run(g, run_view.target, run_view.prompt_tokens, st.pos,
                    el > 0 ? (double)g / el : 0.0, run_view.prefill_rate,
                    nb ? 1000.0 * el / (double)nb : 0.0);
      }
      // Where the sequence has reached, for the KV grower. A relaxed store of one word — the
      // growing happens on its own thread precisely so that none of it lands here.
      dense_gpu.kv_note_position(st.pos);
      // ...and the check that it kept up. Writing past the backed range is a memory violation
      // inside an attention kernel with no useful diagnostic, so this refuses instead. It cannot
      // fire in normal operation: the margin is thousands of blocks of lookahead.
      if (st.pos + B + 2 > dense_gpu.kv_backed_positions()) {
        aff::ui::err("kv cache: sequence reached %u and only %llu positions are backed; stopping\n",
                     st.pos, (unsigned long long)dense_gpu.kv_backed_positions());
        stop = true;
      }
      // Stopping BETWEEN blocks, never inside one: the fingerprint, the acceptance counters and the
      // placement report are the result of the run and all of them still get printed.
      if (dash.quit_requested()) stop = true;
    }
    return generated;
  };

  // One-shot: the same code path the server uses, minus the socket.
  if (one_shot) {
    std::vector<uint32_t> ids = tok.vocab_size() ? tok.encode(prompt, true)
                                                 : std::vector<uint32_t>{1u};
    std::vector<uint32_t> force_ids;
    for (size_t i = 0; i < force_list.size(); ) {
      const size_t c = force_list.find(',', i);
      force_ids.push_back((uint32_t)std::strtoul(force_list.substr(i, c - i).c_str(), nullptr, 10));
      if (c == std::string::npos) break;
      i = c + 1;
    }
    if (!force_ids.empty())
      aff::ui::out("forcing %zu token ids (decode is teacher-forced; tok/s compares BUILDS)\n",
                  force_ids.size());
    SeqState st;
    model.init_state(&st, ids.size() + (uint64_t)n_predict + 8);
    std::vector<float> logits;
    std::vector<uint32_t> hist = ids;

    // The draft's window is seeded from the prompt before the first block can be drafted: its
    // attention conditions on the target's tap over the last `sliding` positions, and prefill is
    // where those were computed. Prefill hands over the taps only — not a per-position head, which
    // a prompt chunk is far too wide for.
    //
    // The block width is the checkpoint's and is not configurable: dspark_draft writes
    // dspark_block_size ids unconditionally (its Markov chain is sequential, so step i's winner is
    // step i+1's input and there is no shorter run of it), so any other width sizes the receiving
    // buffer for fewer tokens than get written.
    // `spec` selects the BLOCK VERIFY path, which is the only decode path there is; `draft_on`
    // decides whether that block carries drafted tokens or is one token wide.
    const uint32_t B = dspark_on ? c.dspark_block : 0u;
    const bool draft_on = dspark_on && c.dspark_block > 0;
    // A property of the CONTAINER, and deliberately so: `--dspark off` means "a one-token-wide
    // verify block", not "a token loop". There is no token loop.
    const bool spec = c.dspark_block > 0;
    // `spec` and `draft_on` differ on exactly the configuration that matters, so do not simplify
    // them together: `spec` is true whenever the checkpoint declares a draft, which is always.
    // Teacher forcing lives in the block path (see spec_decode), so the refusal is on `draft_on` —
    // the condition under which it is actually unsafe.
    if (draft_on && !force_ids.empty()) {
      aff::ui::err("--force-tokens needs --dspark off: with a draft running, the block's KV "
                   "is computed from the ids\n"
                   "       the draft proposed, and overriding the committed token leaves the cache "
                   "holding a continuation\n"
                   "       the model never produced. At width 1 nothing is proposed and the stream "
                   "is simply forced.\n");
      return 1;
    }
    Model::BlockOut blk;
    blk.per_position_head = false;
    blk.tap_keep = c.sliding;

    const auto t0 = std::chrono::steady_clock::now();
    kv_reserve_for(st.pos + (uint64_t)ids.size() + 2);
    watch_prefill(true);
    // --no-routed-experts has to reach here too, not only --dump-logits, or the one flag for
    // isolating the dense path is a silent no-op in the mode the benchmark runs in.
    if (ids.size() > 1 &&
        !model.forward_prefill(ids.data(), (uint32_t)ids.size() - 1, &st, nullptr, nullptr, &logits,
                               !no_routed, spec ? &blk : nullptr)) {
      // With the draft on this is not a slower prefill but one that produced no tap, after which
      // the draft attends to 128 rows of zeros and proposes nothing anyone accepts.
      if (spec) {
        aff::ui::err("fatal: speculation needs the batched prefill and it refused\n");
        return 1;
      }
      for (size_t i = 0; i + 1 < ids.size(); ++i)
        model.forward_token(ids[i], &st, nullptr, nullptr, &logits, !no_routed);
    }
    watch_prefill(false);
    const auto t1 = std::chrono::steady_clock::now();
    // Held for the header, which keeps showing what prefill achieved once decode has started: it is
    // the same prompt's number and the two are read together.
    const double pre_s = std::chrono::duration<double>(t1 - t0).count();
    const double pre_rate = pre_s > 0 && ids.size() > 1 ? (double)(ids.size() - 1) / pre_s : 0.0;

    // Prefill's own counters, before they are cleared. Its phase mix differs from decode's — cold
    // caches, a compressor firing every fourth position, page faults — and averaging the two gives
    // a ms/token that describes neither.
    if (aff::block_timed()) dump_profile("prefill");
    model.reset_profile();

    // ---- --prefill-reps: take the LOAD out of the prefill measurement ---------------------------
    //
    // One prefill a process times a few seconds of work behind a cold load of tens of gigabytes,
    // whose own rate varies run to run. That variance lands inside the arm, where it is wider than
    // the differences worth changing anything for.
    //
    // Prefill is DETERMINISTIC given the prompt: no sampler, no draft, no text in the loop. So
    // unlike decode the noise is not the workload, it is the machine, and the fix is to stop paying
    // the load per measurement rather than to average over it. The prefill above is the warm-up and
    // is deliberately not counted.
    //
    // `init_state` is the whole reset. It zeroes `pos`, the hyper-connection carry and every layer's
    // counts; the compressor's cross-chunk state needs nothing because it is a RING indexed by
    // position rather than an accumulator, so a prefill starting at 0 overwrites every slot it will
    // later read (the argument is spelled out on Model::rollback).
    //
    // The logits checksum is the control, and it is not decoration — but read it against the
    // strategy. Under `--placement-strategy static` nothing moves and it MUST be constant, which is
    // what proves the reset above is complete. Under `heat` it drifts on purpose: the mover changes
    // which experts are resident, which changes the order the dispatch sums them in. A constant
    // checksum under heat would mean the mover was not running.
    //
    // WHAT THIS FLAG EXISTS TO SHOW: under `heat` the rate climbs across reps and flattens only
    // at the end, where the `static` arm is flat throughout at a lower rate. Same process, same page
    // cache, same clocks — so the climb is the placement engine converging and nothing else.
    // **A one-prefill-per-process figure is therefore a COLD-PLACEMENT figure**, and it understates
    // a long-lived server substantially. Quote the plateau, or say "cold" explicitly.
    if (prefill_reps > 1 && ids.size() > 1) {
      // ---- ONE PROMPT IS THE FRIENDLIEST CASE, SO OFFER MORE THAN ONE ---------------------------
      //
      // Re-prefilling a single prompt converges the mover onto exactly the expert set that prompt
      // needs, which is the strongest attractor there is and an optimistic bound on what a server
      // sees. `--prefill-alt FILE` rotates a second prompt in, so the mover has to hold a union
      // instead. The gap between the two is the part of the convergence gain that is real for
      // varied traffic.
      std::vector<std::vector<uint32_t>> sets{ids};
      if (!prefill_alt.empty()) {
        std::FILE* af = std::fopen(prefill_alt.c_str(), "rb");
        if (!af) { aff::ui::err("fatal: --prefill-alt %s: cannot open\n", prefill_alt.c_str()); return 1; }
        std::string txt;
        char buf[8192]; size_t got;
        while ((got = std::fread(buf, 1, sizeof buf, af)) > 0) txt.append(buf, got);
        std::fclose(af);
        std::vector<uint32_t> alt = tok.vocab_size() ? tok.encode(txt, true) : std::vector<uint32_t>{1u};
        if (alt.size() < 2) { aff::ui::err("fatal: --prefill-alt tokenised to %zu ids\n", alt.size()); return 1; }
        sets.push_back(std::move(alt));
        aff::ui::out("\nrotating %zu prompts (%zu and %zu ids)\n", sets.size(), sets[0].size(), sets[1].size());
      }
      aff::ui::out("\nprefill x%u (the run above was the warm-up and is not counted)\n", prefill_reps);
      aff::ui::out("  %-4s %4s %10s %10s %12s\n", "rep", "set", "ms", "tok/s", "logits-sum");
      // Median over tok/s, not ms: rotating prompts of different lengths makes ms incomparable
      // between reps while the rate stays the thing being measured.
      std::vector<double> rate_all;
      uint64_t cap = 0;
      for (const auto& s : sets) cap = std::max<uint64_t>(cap, s.size());
      for (uint32_t r = 0; r < prefill_reps; ++r) {
        const std::vector<uint32_t>& cur = sets[r % sets.size()];
        // The phase breakdown is only worth reading at the PLATEAU. Accumulated over every rep it
        // averages a converging series — the first reps are a cold placement and describe a state
        // the engine leaves and never returns to. Reset here so the dump below is the last rep
        // alone, which is the steady state a live server actually runs in.
        if (r + 1 == prefill_reps) model.reset_profile();
        model.init_state(&st, cap + (uint64_t)n_predict + 8);
        const auto a0 = std::chrono::steady_clock::now();
        kv_reserve_for(st.pos + (uint64_t)cur.size() + 2);
        watch_prefill(true);
        const bool ok = model.forward_prefill(cur.data(), (uint32_t)cur.size() - 1, &st, nullptr,
                                              nullptr, &logits, !no_routed, spec ? &blk : nullptr);
        watch_prefill(false);
        const auto a1 = std::chrono::steady_clock::now();
        if (!ok) { aff::ui::err("fatal: prefill rep %u refused\n", r); return 1; }
        const double ms = std::chrono::duration<double>(a1 - a0).count() * 1e3;
        const double rate = (double)(cur.size() - 1) * 1e3 / ms;
        rate_all.push_back(rate);
        // Sum rather than a hash: the point is to catch a DIFFERENT prefill, and a sum over 129280
        // floats does that while staying readable when it drifts by a little rather than a lot.
        double s = 0.0;
        for (float v : logits) s += (double)v;
        aff::ui::out("  %-4u %4zu %10.1f %10.1f %12.4f\n", r, r % sets.size(), ms, rate, s);
      }
      std::sort(rate_all.begin(), rate_all.end());
      const double med = rate_all[rate_all.size() / 2];
      aff::ui::out("  median %.1f tok/s   spread %.1f%% (%.1f to %.1f)\n", med,
                  100.0 * (rate_all.back() - rate_all.front()) / med, rate_all.front(),
                  rate_all.back());
      // The last rep only — see the reset in the loop. This is the converged breakdown, and it is
      // the one worth optimising against; a profile taken from a single prefill is a cold-placement
      // profile.
      if (aff::block_timed()) dump_profile("prefill (converged, last rep)");
      // Explicitly, because returning here skips the normal exit path that would have printed it —
      // and this is the mode where the placement counters matter MOST, since the whole point of the
      // flag is to watch the engine converge.
      placement.engine_report();
      return 0;
    }

    uint32_t next = ids.empty() ? 1u : ids.back();
    // Through the token sink, not the log sink: it is the head of the same stream the model is
    // about to continue, and it belongs in the pane the continuation goes to.
    if (echo_prompt) aff::ui::tok("\n" + prompt);
    int generated = 0;
    // Speculative decode. A cycle hands the draft whatever the last block committed, drafts B
    // tokens, runs the target over [next, draft...] as one chunk, and keeps the longest prefix the
    // target agrees with.
    //
    // Accept means exact agreement with the target's argmax, which makes this bit-identical to
    // greedy decode: same text, same fingerprint, only the time differs. That is the check as well
    // as the point — a subtly wrong draft cannot corrupt the output, only stop being accepted. So
    // acceptance rate is the correctness signal for the draft, and the md5 for the target.
    if (!spec) {
      aff::ui::fatal("error: this container declares dspark_block = 0, and there is no single-token\n"
                     "       decode path to fall back to — the dense FP8 shards are preshuffled for\n"
                     "       the W8A8 GEMM, which the fused one-token matvec cannot read.\n");
      return 2;
    }
    dash.stage(aff::ui::Stage::Decode);
    run_view = {ids.size() ? ids.size() - 1 : 0, (uint64_t)n_predict, hist.size(), pre_rate, t1};
    generated += (int)spec_decode(/*slot=*/0u, st, blk, next, hist, (uint32_t)n_predict, ignore_eos,
                                    [&](uint32_t id) {
        aff::ui::tok(tok.vocab_size() ? tok.decode_one(id) : std::to_string(id) + " ");
        return true;
      }, sampling_set ? &sample_cfg : nullptr,
         force_ids.empty() ? nullptr : &force_ids);
    dash.stage(aff::ui::Stage::Done);
    // Down before anything below prints. The summary, the fingerprint and the placement report are
    // the run's result and belong on a restored terminal, in the order and on the streams they have
    // always been on; the dashboard replays whatever the console pane still held first.
    dash.stop();
    const auto t2 = std::chrono::steady_clock::now();
    const double pre = pre_s;
    const double gen = std::chrono::duration<double>(t2 - t1).count();
    // `generated`, not n_predict: an EOS break makes them differ, and dividing the requested count
    // by a shorter run's elapsed time reports a throughput that was never achieved.
    // The exact fingerprint of what was generated, on stderr and unconditional. It is what a bench
    // hashes: the generated TEXT is not usable for that, because a short continuation is shorter
    // than the load banners and any "longest line" rule silently hashes a line carrying load times
    // instead — which reads as a nondeterministic engine and is not one.
    aff::ui::err("\ngenerated ids: ");
    for (size_t i = ids.size(); i < hist.size(); ++i)
      aff::ui::err("%u%s", hist[i], i + 1 < hist.size() ? "," : "");
    aff::ui::err("\n");
    if (aff::block_timed()) dump_profile("decode per-token");
    // Always: the speedup is linear in acceptance and everything else here is downstream of it.
    if (draft_on) {
      const auto& p = model.profile();
      aff::ui::out("\n\ndspark: %llu blocks, %llu drafted, %llu accepted (%.1f%%, %.2f of %u a "
                  "block), %.2f tokens a block\n",
                  (unsigned long long)p.blocks, (unsigned long long)p.drafted,
                  (unsigned long long)p.accepted,
                  p.drafted ? 100.0 * (double)p.accepted / (double)p.drafted : 0.0,
                  p.blocks ? (double)p.accepted / (double)p.blocks : 0.0, B,
                  p.blocks ? (double)(p.accepted + p.blocks) / (double)p.blocks : 0.0);
      // The marginal rate per position, which decides whether the block is too long. Not the
      // accepted fraction: acceptance stops at the first miss, so position 4's contribution to the
      // total is conditioned on 0..3 all being right, while this is not.
      aff::ui::out("        marginal hit by position:");
      for (uint32_t j = 0; j < B && j < kSpecTok; ++j)
        aff::ui::out(" %u:%.0f%%", j, p.pos_n[j] ? 100.0 * (double)p.pos_hit[j] / (double)p.pos_n[j]
                                                : 0.0);
      aff::ui::out("\n");
    }
#ifdef AFF_WITH_HIP
    // Always, not only under AFF_PROFILE: the engine moves bytes over the same link the hybrid tier
    // uses, so a run that does not say how many is one whose throughput cannot be explained.
    placement.engine_report();
    // The draft's too, and for the same reason: its three stages build their dispatch on the card
    // as well, so "how many of them actually did" is a number the run has to print. Without it a
    // draft that silently kept the host build for every block would look exactly like one that did
    // not, and a flag that never fires reads as a noise floor.
    if (draft_placement.enabled()) draft_placement.engine_report();
    // How many times the two cards were actually compared against each other. Printed rather than
    // assumed, because a check that reports zero checks is a failed check and reads exactly like a
    // check that passed — and this is the one instrument that can name a field diverging between
    // the cards.
    if (dense_gpu.ranks() > 1) {
      // ...and ZERO says something different from "all equal", which is the whole point of printing
      // it. The sampler fires every kXrEvery-th call, so a run shorter than that interval compares
      // nothing — correct, but it must not read as a pass.
      const unsigned long long xr = (unsigned long long)aff::xrank_compared();
      if (xr) aff::ui::err("cross-rank: %llu replicated-activation samples compared, "
                           "all equal\n", xr);
      else aff::ui::err("cross-rank: NOT CHECKED — this run was shorter than the sampling "
                        "interval, so the two cards were never compared.\n");
    }
#endif
    aff::ui::out("\n\nprefill %zu tok in %.2fs (%.1f tok/s), decode %d tok in %.2fs (%.2f tok/s)%s\n",
                ids.size() ? ids.size() - 1 : 0, pre, pre > 0 ? (double)(ids.size() - 1) / pre : 0.0,
                generated, gen, gen > 0 ? generated / gen : 0.0,
                generated < n_predict ? "  [stopped at EOS]" : "");
    // The pool is the budget the operator sets; this is what the process actually took, and the
    // difference between them is what --host-pool-reserve-mib has to cover.
    if (const double hwm = peak_rss_gib(); hwm > 0.0)
      aff::ui::out("peak host RSS %.1f GiB, of which %.1f GiB is the expert pool\n",
                  hwm, host_pool.stats().bytes / 1073741824.0);
    return 0;
  }

  Server srv;
  // The MODEL, not the engine. A client picks a model by this string and routes on it, so naming
  // the server here made every deployment claim to serve something called "affinity" and made two
  // engines serving the same weights look like two different models. `owned_by` is where the engine
  // belongs, and that still says affinity.
  srv.set_model_name("DeepSeek-V4-Flash");
  // The container's own encoder, not a hand-rolled template: this release ships no Jinja
  // chat_template and points at `encoding/` instead. src/engine/chat_encode.cpp transcribes it and
  // tests/test_chat_encode.cpp holds it to the container's fixtures byte for byte.
  //
  // Defaults are the model card's: temperature 1.0, top_p 0.95 (its agentic recommendation, and this
  // server exists to be driven by agents), and a random seed per request so two identical requests
  // are two samples rather than one answer twice.
  srv.set_slots(kSlots);
  const bool ok = srv.start(host, port, [&](const CompletionRequest& req, const TokenSink& sink,
                                            GenResult* res, uint32_t slot) {
    std::vector<ChatMsg> msgs;
    if (req.messages.empty()) {
      // The legacy completions endpoint: a raw prompt with no roles, so it is encoded as a bare
      // user turn rather than smuggled past the template.
      ChatMsg m; m.role = "user"; m.content = req.prompt;
      msgs.push_back(m);
    } else {
      for (const ChatMessage& in : req.messages) {
        ChatMsg m;
        m.role = in.role;
        m.content = in.content;
        m.reasoning_content = in.reasoning_content;
        m.tool_call_id = in.tool_call_id;
        for (const ToolCallIO& t : in.tool_calls) m.tool_calls.push_back({t.id, t.name, t.arguments});
        msgs.push_back(m);
      }
    }
    // Tools ride on the first system message, which is where the encoder renders them. With no
    // system message there is one to carry them, because a tools block in a user turn is a
    // different prompt.
    // `tool_choice: "none"` means the model must not call a tool, and the way to mean it is to not
    // offer them. "required" and a named function need constrained decoding, which this engine does
    // not do; the request is refused rather than served as "auto", because a caller that asked for a
    // guaranteed call and got prose has no way to tell that the guarantee was dropped.
    // The encoder renders the roles it knows and emits nothing for anything else, so an unrecognised
    // role would drop that message out of the prompt without a word. The caller is told instead.
    for (const ChatMsg& m : msgs) {
      if (m.role != "system" && m.role != "user" && m.role != "assistant" &&
          m.role != "tool" && m.role != "latest_reminder") {
        res->error = "unsupported message role \"" + m.role + "\"";
        return;
      }
    }

    // One sequence is decoded per request, so `n` above 1 would return fewer choices than asked for
    // and the caller could not tell. It is refused rather than under-served.
    if (req.n > 1) {
      res->error = "n > 1 is not supported: this engine decodes one sequence per request";
      return;
    }
    const std::string& tc = req.tool_choice;
    if (!tc.empty() && tc != "auto" && tc != "none") {
      res->error = "tool_choice \"" + tc + "\" requires constrained decoding, which this engine does "
                   "not implement. Send \"auto\" or \"none\".";
      return;
    }
    // Tools and the response schema both ride on the first system message, which is where the
    // encoder renders them. With no system message there is one to carry them, because either block
    // in a user turn is a different prompt.
    const bool want_tools = !req.tools.empty() && tc != "none";
    if (want_tools || !req.response_format.empty()) {
      if (msgs.empty() || msgs[0].role != "system") {
        ChatMsg sys; sys.role = "system";
        msgs.insert(msgs.begin(), sys);
      }
      if (want_tools)
        for (const std::string& t : req.tools) msgs[0].tools.push_back(ToolDef{t});
      msgs[0].response_format = req.response_format;
    }

    EncodeOpts eo;
    eo.thinking_mode = req.thinking_mode == "chat" ? ThinkingMode::Chat : ThinkingMode::Thinking;
    // `max` by default, which is what the card evaluates its agentic scores at. The level is only a
    // text prefix telling the model how hard to think, so it costs nothing to set and everything to
    // get wrong; a client that wants a faster, shallower answer sends "low".
    eo.reasoning_effort = req.reasoning_effort == "low"  ? ReasoningEffort::Low
                        : req.reasoning_effort == "high" ? ReasoningEffort::High
                                                         : ReasoningEffort::Max;
    // A conversation with nothing in it is refused rather than encoded. The prompt would be four
    // control tokens, which the model answers by observing that it has not been asked anything --
    // and a degenerate sequence has, separately, been seen to fault the engine downstream. Neither
    // is a good way to report that a client sent no text.
    bool any_text = false;
    for (const ChatMsg& m : msgs)
      any_text = any_text || !m.content.empty() || !m.tool_calls.empty() || !m.tools.empty();
    if (!any_text) {
      res->error = "every message is empty — if the client sends content as typed blocks, this build "
                   "reads only the \"text\" type";
      return;
    }

    const std::string prompt = encode_messages(msgs, eo);
    // add_bos false: the encoder put the BOS in the string, and a second one is a different prompt.
    std::vector<uint32_t> ids = tok.vocab_size() ? tok.encode(prompt, false)
                                                 : std::vector<uint32_t>{1};

    // ---- what max_tokens bounds ------------------------------------------------------------------
    //
    // EVERY generated token counts, reasoning included. That is what the OpenAI schema says for a
    // reasoning model — `max_completion_tokens` is defined over reasoning and answer together — and
    // more importantly it is the only thing a caller can bound its own latency and spend with.
    //
    // Charging only the answer, and letting the reasoning run to the context, reads as generous and
    // is not: a request with a small `max_tokens` then generates to the end of the context, because
    // nothing in the shape of it can stop it. A budget the client cannot set is not a budget.
    //
    // The cost of this is that a request whose reasoning fills the allowance returns
    // finish_reason "length" with empty content. That is the documented behaviour of every reasoning
    // model and it is honest — the tokens were spent — where an unbounded run is neither. A caller
    // that wants the whole allowance spent on the answer sends thinking_mode "chat".
    //
    // --max-tokens-floor is the operator's answer to a client that hardcodes a `max_tokens` too
    // small for a reasoning turn and offers no way to change it. It raises the request's budget and
    // never lowers it, and the context still bounds the result.
    const uint64_t ctx_room = kMaxKvPositions > ids.size() + 32
                            ? kMaxKvPositions - ids.size() - 32 : 0;
    const uint64_t asked = req.has_max_tokens ? (uint64_t)req.max_tokens : ctx_room;
    const bool floored = req.has_max_tokens && asked < (uint64_t)max_tokens_floor;
    const uint32_t total_budget = (uint32_t)std::min<uint64_t>(
        ctx_room, std::max<uint64_t>(asked, (uint64_t)max_tokens_floor));

    SeqState st;
    // The budget covers the whole generation, so it is also exactly what the state has to hold.
    model.init_state(&st, ids.size() + total_budget + 8);
    std::vector<float> logits;
    std::vector<uint32_t> hist = ids;

    Model::BlockOut blk;
    blk.per_position_head = false;
    blk.tap_keep = c.sliding;
    // The prompt feeds every position but the last, which the first verify block feeds.
    const uint32_t n_pre = ids.empty() ? 0u : (uint32_t)ids.size() - 1;
    uint32_t from = 0;
    uint32_t matched_blocks = 0;              // read again below, by the promotion boundary
    if (prefix_cache.ready() && n_pre) {
      uint32_t& matched = matched_blocks;
      std::string rerr;
      uint32_t hit = 0;
      {
        // lookup reads the cache index and restore writes this slot's KV; both must be serialized
        // against the other slot's publish, which mutates the same index and stage pool. lookup
        // was previously outside the lock, racing publish's hash-table writes.
        std::unique_lock<std::mutex> step(engine_mu);
        dense_gpu.set_slot(slot);
        hit = prefix_cache.lookup(ids.data(), n_pre, &matched);   // index only, no I/O
        if (hit && prefix_cache.restore(ids.data(), hit, &rerr)) {
          model.restore_state(&st, hit);
          from = hit;
        }
      }
      // One line a request. What a prefix cache does is invisible from the outside — a miss and a
      // hit differ only in how long the reply took — so the hit length is reported rather than left
      // to be inferred from a latency that also moves with residency and acceptance.
      // A digest of the leading tokens, because "0 blocks matched" has two very different causes:
      // a store that never got them, and a render that does not begin the same way twice.
      uint64_t idh = 1469598103934665603ull;
      const uint32_t nlead = std::min<uint32_t>(n_pre, prefix_cache.block_tokens());
      for (uint32_t i = 0; i < nlead; ++i)
        for (int k = 0; k < 4; ++k) { idh ^= (ids[i] >> (8*k)) & 0xFF; idh *= 1099511628211ull; }
      aff::ui::err("prefix cache: %u/%u prompt tokens restored, %u to prefill (%u blocks of %u "
                   "matched, lead %u tok %016llx)\n", from, n_pre, n_pre - from, matched,
                   n_pre / prefix_cache.block_tokens(), nlead, (unsigned long long)idh);
      if (hit && !from) {
        // Loud, and then a full prefill. A cache that cannot be read is a slow request rather than
        // a wrong one — the restore either completed or wrote rows that the prefill about to run
        // overwrites before anything reads them — but it is still a fault, and a silent one would
        // read for months as a cache that simply never helps.
        aff::ui::err("prefix cache: %s — prefilling %u tokens from scratch\n",
                     rerr.c_str(), n_pre);
      }
    }
    // ---- the promotion boundary --------------------------------------------------------------
    //
    // `lookup` returns two different numbers and both of them matter. `from` is where a
    // RESUME POINT exists, and one exists only where some request ended; `matched` is how many
    // leading BLOCKS the store holds, and blocks are shared between conversations, so a fresh
    // conversation whose opening is a common system prompt routinely matches many blocks and
    // resumes at none of them. Those blocks are then re-derived on this prefill and thrown away.
    //
    // Ending a prefill exactly on that boundary and publishing there converts them into a resume
    // point for every later request through the same prefix. It costs one extra `publish` — the
    // blocks below it are already held, so what is written is the checkpoint alone — and no extra
    // model work: the same tokens are prefilled either way, in two calls instead of one.
    //
    // GUARDED BY THE TAP. `forward_prefill` hands back the DSpark taps for the last `sliding`
    // positions of ITS OWN call, so a split that leaves the second call shorter than the window
    // seeds the draft with fewer rows and changes what it proposes. A boundary that close to the
    // end is also worth nothing, so it is simply not taken.
    uint32_t promote = 0;
    if (prefix_cache.ready()) {
      const uint64_t b = (uint64_t)matched_blocks * prefix_cache.block_tokens();
      if (b > from && b + c.sliding <= n_pre) promote = (uint32_t)b;
    }
    if (promote) {
      std::unique_lock<std::mutex> step(engine_mu);
      dense_gpu.set_slot(slot);
      kv_reserve_for(st.pos + (uint64_t)ids.size() + 2);
      watch_prefill(true);
      const bool pre_ok = model.forward_prefill(ids.data() + from, promote - from, &st, nullptr,
                                                nullptr, nullptr, true, &blk);
      if (dense_gpu.n_slots() > 1) dense_gpu.sync_devices();
      watch_prefill(false);
      if (!pre_ok) {
        res->finish_reason = "stop";
        sink("", Delta::Content, true);
        return;
      }
      std::string perr;
      if (!prefix_cache.publish(ids.data(), promote, &perr))
        aff::ui::err("prefix cache: %s\n", perr.c_str());
      else
        aff::ui::err("prefix cache: promoted a resume point at %u (%u blocks held)\n",
                     promote, matched_blocks);
      from = promote;
    }
    if (n_pre > from) {
      std::unique_lock<std::mutex> step(engine_mu);
      dense_gpu.set_slot(slot);
      kv_reserve_for(st.pos + (uint64_t)ids.size() + 2);
      watch_prefill(true);
      const bool pre_ok = model.forward_prefill(ids.data() + from, n_pre - from, &st, nullptr,
                                                nullptr, &logits, true, &blk);
      if (dense_gpu.n_slots() > 1) dense_gpu.sync_devices();
      watch_prefill(false);
      if (!pre_ok) {
        res->finish_reason = "stop";
        sink("", Delta::Content, true);
        return;
      }
    }
#ifdef AFF_WITH_HIP
    if (pcache_verify && n_pre) {
      // Every live row, in the region order the store uses. The append-only parts are hashed only
      // as far as the position reaches, because rows past it are whatever the last conversation
      // left and are never read; the rings are hashed whole, because all of them are live.
      //
      // Both ranks, separately. The cache is replicated and a save reads rank 0 alone, so if the
      // two ever disagreed a restore would quietly make them agree — and every claim about the
      // cache being one thing would have been wrong for as long as it had been running.
      static const char* kPartName[] = {"raw", "comp", "idx8", "idxscale", "hist", "idxhist"};
      // One column a rank, and every rank compared against rank 0. Comparing against rank 0 rather
      // than pairwise is what keeps this complete at any rank count: the cache is meant to be one
      // replicated thing, so agreeing with rank 0 is agreeing with each other, and a rank that
      // drifted shows up whichever one it is.
      const size_t nrank = dense_gpu.ranks();
      uint64_t h[kMaxRanks];
      for (size_t r = 0; r < kMaxRanks; ++r) h[r] = 1469598103934665603ull;
      uint32_t split = 0;
      std::string first_split;
      for (const KvRegion& r : pcache_regions) {
        // What attention will actually READ, not what the buffer holds. The raw ring is 383 slots so
        // that a whole sub-batch can commit before any of it attends, but only the last `sliding`
        // of them are ever in a window; the rest hold positions that have scrolled out, and rows
        // that speculation wrote and rolled back. Hashing those compares the engine's litter and
        // says nothing about the state.
        const bool raw = (DenseGpu::KvPart)r.part == DenseGpu::KvPart::Raw;
        const uint64_t rows = r.ratio ? n_pre / r.ratio
                                      : (raw ? std::min<uint64_t>(n_pre, c.sliding) : r.rows);
        if (!rows) continue;
        const uint64_t lo = raw ? (n_pre - rows) % r.rows : 0;
        // The window wraps, so it is up to two runs of slots.
        const uint64_t n0 = std::min(rows, r.rows - lo), n1 = rows - n0;
        uint64_t rh[kMaxRanks];
        for (size_t rank = 0; rank < kMaxRanks; ++rank) rh[rank] = 1469598103934665603ull;
        for (size_t rank = 0; rank < nrank; ++rank)
          for (uint64_t* dst : {&rh[rank], &h[rank]}) {
            dense_gpu.kv_part_hash((DenseGpu::KvPart)r.part, r.layer, lo, n0, rank, dst);
            if (n1) dense_gpu.kv_part_hash((DenseGpu::KvPart)r.part, r.layer, 0, n1, rank, dst);
          }
        bool region_split = false;
        for (size_t rank = 1; rank < nrank; ++rank) region_split |= rh[rank] != rh[0];
        if (region_split) {
          if (!split++)
            first_split = std::string(kPartName[r.part < 6 ? r.part : 0]) + " layer " +
                          std::to_string(r.layer);
        }
        // Per region, so two runs can be diffed to say WHICH state a restore failed to reproduce.
        // A single total tells you only that something moved.
        std::string cols;
        for (size_t rank = 0; rank < nrank; ++rank) {
          char buf[24];
          std::snprintf(buf, sizeof buf, " %016llx", (unsigned long long)rh[rank]);
          cols += buf;
        }
        aff::ui::err("kv-region pos %u %-8s L%-2u rows %-7llu%s\n", n_pre,
                     kPartName[r.part < 6 ? r.part : 0], r.layer, (unsigned long long)rows,
                     cols.c_str());
        // ...and, for the append-only parts, at quarter points of the row order. Rows are laid down
        // in position order, so the first quarter to disagree between two runs says which POSITIONS
        // a restore reproduced and which it did not. Without this a whole-region mismatch cannot be
        // told apart from a mismatch in the last row.
        if (r.ratio)
          for (int q = 1; q <= 4; ++q) {
            const uint64_t n = rows * q / 4;
            if (!n) continue;
            uint64_t qh = 1469598103934665603ull;
            dense_gpu.kv_part_hash((DenseGpu::KvPart)r.part, r.layer, 0, n, 0, &qh);
            aff::ui::err("kv-quart pos %u %-8s L%-2u q%d rows %-7llu %016llx\n", n_pre,
                         kPartName[r.part < 6 ? r.part : 0], r.layer, q,
                         (unsigned long long)n, (unsigned long long)qh);
          }
      }
      std::string totals;
      for (size_t rank = 0; rank < nrank; ++rank) {
        char buf[40];
        std::snprintf(buf, sizeof buf, " rank%zu %016llx", rank, (unsigned long long)h[rank]);
        totals += buf;
      }
      aff::ui::err("kv-hash: pos %u restored %u prefilled %u %s"
                   "  cross-rank splits %u/%zu%s%s\n",
                   n_pre, from, n_pre - from, totals.c_str(),
                   split, pcache_regions.size(), split ? " first at " : "", first_split.c_str());
    }
#endif

    // The raw turn, kept whole. Tool markup is not content and cannot be understood a piece at a
    // time, so what streams is text and what is parsed is the whole thing.
    std::string raw;
    const std::string marker = kToolCallsOpen;
    const std::string think_end = kThinkEnd;
    size_t streamed = 0;
    bool saw_marker = false;
    // Every string that must be recognised whole before any of it is emitted.
    std::vector<std::string> sentinels = req.stop;
    sentinels.push_back(kToolCallsOpen);
    // In thinking mode the reasoning streams on its OWN channel. It must not go out as
    // `delta.content` — that puts the model's deliberation, and then a bare </think>, into the
    // client's transcript as if it were the reply — but dropping it is just as wrong, because a
    // client that renders reasoning has nothing to render. So it goes out as `reasoning_content`
    // until the block closes, and as content after.
    bool open_think = eo.thinking_mode == ThinkingMode::Thinking;
    size_t think_streamed = 0;
    // A token boundary is not a character boundary: an emoji arrives as two tokens, and cutting
    // between them puts half a character in a JSON string, which is not valid UTF-8 and not valid
    // JSON. Back the cut up to the last complete character; the rest goes out with the next token.
    auto utf8_end = [](const std::string& s, size_t end) -> size_t {
      if (end == 0 || (uint8_t)s[end - 1] < 0x80) return end;
      size_t lead = end - 1;
      while (lead && ((uint8_t)s[lead] & 0xC0) == 0x80) --lead;
      const uint8_t c = (uint8_t)s[lead];
      const size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
      return lead + need <= end ? end : lead;
    };

    // The model card's settings unless the request overrides them, and a random seed so two
    // identical requests are two samples rather than one answer twice. `temperature: 0` still
    // reaches the greedy path inside the sampler, so a caller can ask for determinism.
    SamplerConfig sc;
    sc.temperature = req.has_temperature ? req.temperature : 1.0f;
    sc.top_p = req.has_top_p ? req.top_p : 0.95f;
    sc.top_k = req.top_k;
    sc.min_p = req.min_p;
    sc.seed = req.seed ? req.seed : (uint64_t)std::random_device{}();


    run_view = {ids.size(), total_budget, hist.size(), 0.0, std::chrono::steady_clock::now()};
    // Per-request speculation accounting. The counters are global and cumulative, so a snapshot
    // either side of the call is this request's share — which is the only way to see acceptance
    // collapse for ONE sequence while another is interleaved with it.
    const Model::PhaseProfile p_before = model.profile();
    const auto t_dec0 = std::chrono::steady_clock::now();
    const uint32_t made = spec_decode(slot, st, blk, ids.empty() ? 1u : ids.back(), hist, total_budget,
                                      /*ignore_eos_=*/false, [&](uint32_t id) {
      if ((int32_t)id == tok.eos()) return false;
      raw += tok.vocab_size() ? tok.decode_one(id) : std::string();
      if (open_think) {
        const size_t te = raw.find(think_end);
        // Everything up to the closing marker is reasoning. Hold back any tail that could still
        // BECOME that marker, or a partial "</thi" reaches the client as reasoning text.
        size_t safe_r = te == std::string::npos ? raw.size() : te;
        if (te == std::string::npos)
          for (size_t back = std::min(think_end.size() - 1, safe_r - think_streamed); back; --back)
            if (raw.compare(safe_r - back, back, think_end, 0, back) == 0) { safe_r -= back; break; }
        if (te == std::string::npos) safe_r = utf8_end(raw, safe_r);
        if (safe_r > think_streamed) {
          const std::string piece = raw.substr(think_streamed, safe_r - think_streamed);
          think_streamed = safe_r;
          if (!sink(piece, Delta::Reasoning, false)) return false;
        }
        if (te == std::string::npos) return true;        // still reasoning
        open_think = false;
        streamed = te + think_end.size();
      }
      if (saw_marker) return true;
      // Hold back anything that might still become the tool-call marker. Without this the opening
      // "\n\n<" of a tool call is streamed as content and the client renders markup.
      const size_t at = raw.find(marker, streamed);
      if (at != std::string::npos) { saw_marker = true; }
      size_t safe = saw_marker ? at : raw.size();
      // A stop string is sought in the accumulated text, not in the piece: one that straddles two
      // tokens appears in neither. It also CUTS the reply — the stop text is not part of the answer
      // and OpenAI does not return it — so `raw` is truncated and not merely abandoned.
      size_t stop_at = std::string::npos;
      for (const std::string& sp : req.stop) {
        if (sp.empty()) continue;
        const size_t s = raw.find(sp, streamed);
        if (s < stop_at) stop_at = s;
      }
      const bool hit_stop = stop_at != std::string::npos && stop_at <= safe;
      if (hit_stop) safe = stop_at;
      // Neither a marker nor a stop string may be emitted in halves, so any tail that could still
      // become one waits for the next token — as does a partial UTF-8 character.
      if (!saw_marker && !hit_stop) {
        for (const std::string& sen : sentinels)
          for (size_t back = std::min(sen.size() - 1, safe - streamed); back; --back)
            if (raw.compare(safe - back, back, sen, 0, back) == 0) { safe -= back; break; }
        safe = utf8_end(raw, safe);
      }
      if (safe > streamed) {
        const std::string piece = raw.substr(streamed, safe - streamed);
        streamed = safe;
        if (!sink(piece, Delta::Content, false)) return false;
      }
      if (hit_stop) { raw.resize(stop_at); return false; }
      return true;
    }, &sc);
    res->prompt_tokens = (uint32_t)ids.size();
    res->completion_tokens = made;
    // Back to the idle banner: the stage was moved to prefill and then decode by this request, and
    // a server that stayed on "decode" between requests would report the last one forever.
    dash.stage(aff::ui::Stage::Serve, serve_addr);
    // Offer the whole turn to the store while the cards still hold it.
    //
    // The caches can be AHEAD of the transcript here. A verify block runs `nd + 1` tokens through
    // the model and only then emits them, so a budget that runs out mid-block leaves positions fed
    // that no client ever saw. Publishing `st.pos` of them would store a prefix that does not match
    // its own token ids and could never be looked up again; refusing to publish when they disagree
    // silently disables the cache for exactly those requests, and whether a request is one of them
    // depends on where acceptance happens to land.
    //
    // So the state is wound back to the transcript first. That is the same rollback speculation
    // already does after a rejected tail, and it is safe for the same reason: the rows above `upto`
    // are overwritten by the next tokens before anything reads them.
    if (prefix_cache.ready() && !hist.empty()) {
      const uint32_t upto = (uint32_t)std::min<size_t>(st.pos, hist.size());
      if (st.pos > upto) model.rollback(&st, upto);
      std::string perr;
      if (upto && !prefix_cache.publish(hist.data(), upto, &perr))
        aff::ui::err("prefix cache: %s\n", perr.c_str());
    }
    // Truncation has to be distinguishable from a finished answer: a caller that sees "stop" on a
    // reply cut at max_tokens has no way to know it was cut, and will hand the fragment on as whole.
    {
      const Model::PhaseProfile& pa = model.profile();
      const uint64_t nb = pa.blocks - p_before.blocks;
      const uint64_t nd = pa.drafted - p_before.drafted;
      const uint64_t na = pa.accepted - p_before.accepted;
      const double el =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t_dec0).count();
      aff::ui::err("slot %u: %u tok in %.2fs (%.1f t/s), %llu blocks, %llu drafted, %llu accepted "
                   "(%.1f%%, %.2f tok/block)\n", slot, made, el, el > 0 ? made / el : 0.0,
                   (unsigned long long)nb, (unsigned long long)nd, (unsigned long long)na,
                   nd ? 100.0 * (double)na / (double)nd : 0.0,
                   nb ? (double)made / (double)nb : 0.0);
    }
    // The budget is spent inside spec_decode rather than in the sink below it, so the only thing
    // that says which way the run ended is whether it produced the whole allowance -- EOS and every
    // stop string return early and leave `made` short.
    const bool hit_budget = total_budget && made >= total_budget;
    // Whatever a hold-back is still holding. Without this the last characters are dropped whenever
    // the turn ends on something that could have begun a marker -- a trailing newline is enough,
    // and the tool marker starts with two. A block that never closed stays on its own channel: the
    // whole turn is reasoning, and flushing it as content would hand the caller deliberation as an
    // answer.
    if (open_think) {
      if (think_streamed < raw.size()) sink(raw.substr(think_streamed), Delta::Reasoning, false);
    } else if (!saw_marker && streamed < raw.size()) {
      sink(raw.substr(streamed), Delta::Content, false);
    }

    const ThinkingMode tm = eo.thinking_mode;
    ParsedMessage pm = parse_completion(raw + kEosStr, tm);
    if (tm == ThinkingMode::Thinking && raw.find(kThinkEnd) == std::string::npos) {
      // The reasoning allowance ran out before the block closed. That text is reasoning, not an
      // answer, and returning it as `content` hands the caller the model's deliberation as if it had
      // replied. Said on stderr too, because an empty content with finish_reason "length" is easy
      // for a client to render as "the model stopped" when the cause is a budget.
      res->reasoning_content = raw;
      res->finish_reason = "length";
      aff::ui::err("server: the reasoning block did not close within the %u token budget (%s), so "
                   "there is no answer to return. Raise it with --max-tokens-floor, or send "
                   "thinking_mode \"chat\" to spend the whole budget on the answer.\n",
                   total_budget,
                   floored              ? "--max-tokens-floor"
                   : req.has_max_tokens ? "max_tokens"
                                        : "the context left after the prompt");
    } else if (!pm.ok) {
      // A generation the parser cannot frame is still the model's output, and the client gets it
      // rather than an error: the alternative is discarding a turn because its markup was odd.
      res->content = raw;
      res->finish_reason = "stop";
    } else {
      res->content = pm.content;
      res->reasoning_content = pm.reasoning_content;
      for (size_t i = 0; i < pm.tool_calls.size(); ++i)
        res->tool_calls.push_back(ToolCallIO{"call_" + std::to_string(i), pm.tool_calls[i].name,
                                             pm.tool_calls[i].arguments});
      res->finish_reason = pm.tool_calls.empty() ? "stop" : "tool_calls";
    }
    if (hit_budget && res->finish_reason == "stop") res->finish_reason = "length";
    sink("", Delta::Content, true);
  }, &err);
  if (!ok) { aff::ui::err("server: %s\n", err.c_str()); return 1; }

  aff::ui::out("listening on http://%s:%u  (hit rate will be reported on exit)\n", host.c_str(), srv.port());
  serve_addr = sfmt("http://%s:%u", host.c_str(), srv.port());
  dash.stage(aff::ui::Stage::Serve, serve_addr);

  // Interactively, "q" or EOF on stdin quits. Detached, stdin is closed or /dev/null and waiting on
  // it would exit immediately, so the server waits for a signal instead.
  if (dash.running()) {
    // The dashboard holds stdin in raw mode with VMIN=0 and reads `q` itself. An fgets against that
    // returns nothing and would take the EOF branch below, so the server would exit the instant it
    // started listening.
    while (!dash.quit_requested() && !g_stop)
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
  } else if (::isatty(STDIN_FILENO)) {
    for (;;) {
      char line[64];
      if (!std::fgets(line, sizeof(line), stdin)) break;
      if (!std::strncmp(line, "q", 1)) break;
    }
  } else {
    std::signal(SIGINT, [](int) { g_stop = 1; });
    std::signal(SIGTERM, [](int) { g_stop = 1; });
    while (!g_stop) ::pause();
  }
  srv.stop();
  if (prefix_cache.ready()) {
    const PrefixCacheStats& p = prefix_cache.stats();
    aff::ui::out("prefix cache: %llu/%llu requests resumed, %llu tokens restored, "
                "%llu blocks written (%llu evicted), %llu checkpoints, %.1f GiB read / %.1f GiB "
                "written\n",
                (unsigned long long)p.hits, (unsigned long long)p.lookups,
                (unsigned long long)p.tokens_restored, (unsigned long long)p.blocks_written,
                (unsigned long long)p.blocks_evicted, (unsigned long long)p.ckpts_written,
                p.bytes_read / 1073741824.0, p.bytes_written / 1073741824.0);
    if (p.verify_misses)
      aff::ui::out("prefix cache: %llu entries failed verification and were not used\n",
                  (unsigned long long)p.verify_misses);
  }
  return 0;
}
