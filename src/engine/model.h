// Model runtime: configuration, weight binding, and the layer loop.
//
// Layer schedule is read from `compress_ratios` in the checkpoint config (44 entries = 43 layers
// + the MTP block): 0 on layers 0-1 (uncompressed), then alternating 4 (CSA, indexed) and
// 128 (HCA, unindexed). Hash routing applies to layers 0-2 only.

#pragma once

#include "attention.h"
#include "ops.h"
#include "expert_kernel.h"
#include "format/aff_reader.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aff {

// Every field below is overwritten by parse_model_config() from the container's own config; the
// initialisers are documentation of the shipped model's shape, not defaults anything relies on.
struct ModelConfig {
  uint32_t n_layer = 43;
  uint32_t n_embd = 4096;
  uint32_t n_head = 64;
  uint32_t head_dim = 512;
  uint32_t rope_dim = 64;          // qk_rope_head_dim
  uint32_t n_expert = 256;
  uint32_t n_expert_used = 6;
  uint32_t n_shared_expert = 1;
  uint32_t moe_inter = 2048;
  uint32_t vocab = 129280;
  uint32_t sliding = 128;
  uint32_t index_topk = 512;
  uint32_t n_hash_layer = 3;       // layers 0..n_hash_layer-1 use the token-id lookup table
  uint32_t q_lora_rank = 1024;
  uint32_t o_lora_rank = 1024;
  uint32_t hc_mult = 4;
  uint32_t hc_sinkhorn_iters = 20;
  uint32_t index_head_dim = 128;
  uint32_t index_n_heads = 64;
  float    rope_factor = 16.0f;         // YaRN scaling factor
  uint32_t rope_orig_ctx = 65536;
  float    rope_beta_fast = 32.0f;
  float    rope_beta_slow = 1.0f;
  float    hc_eps = 1e-6f;
  float    rms_eps = 1e-6f;
  float    rope_theta = 10000.0f;
  float    compress_rope_theta = 160000.0f;
  float    routed_scaling = 1.5f;
  float    swiglu_limit = 10.0f;
  bool     norm_topk_prob = true;
  std::vector<int32_t> compress_ratios;   // per layer, PLUS one per mtp stage at the tail

  // ---- DSpark, the speculative draft (config keys `dspark_*`) ---------------------------------
  //
  // Zero when the checkpoint has no draft module, and then nothing below is read. `dspark_block`
  // is how many tokens one draft pass emits (5 here, so a verified pass covers 6 positions);
  // `dspark_taps` names the TARGET layers whose hidden state the draft is conditioned on, averaged
  // over the hc lanes and concatenated; `dspark_noise_token` is what the draft's positions 1..n-1
  // are seeded with, position 0 taking the token the target just produced.
  uint32_t dspark_block = 0;
  uint32_t dspark_noise_token = 0;
  uint32_t dspark_markov_rank = 256;
  std::vector<int32_t> dspark_taps;

  // ---- DeepSeek-V4.1 (`model_type` `deepseek_v41`) --------------------------------------------
  //
  // Same block shape, different attention schedule. V4 gave every compressing layer its own
  // compressor and indexer; V4.1 names SOURCE layers: a kv-source layer builds the compressed KV
  // and the indexer keys, an index-source layer builds the top-k, and the layers between reuse
  // them (`SharedAttentionRuntime` in the reference). `compress_ratios` are 1 and 2 here, not
  // 4 and 128, so `ratio == 4` no longer means "indexed" — ask `is_index_source`.
  bool     v41 = false;
  uint32_t o_groups = 8;                  // wo_a is block-diagonal over this many groups
  std::vector<int32_t> kv_source_layers;      // build compressed KV + indexer keys
  std::vector<int32_t> index_source_layers;   // build the indexer top-k
  // Candidate pre-filter: one source layer scores blocks of `candidate_block` positions and keeps
  // `candidate_topk_blocks` of them; < 0 disables it and the other two are unused.
  int32_t  candidate_source_layer = -1;
  uint32_t candidate_topk_blocks = 0;
  uint32_t candidate_block = 0;
  // Engram: n-gram hash lookups added to the residual stream between layers. The tables are ~95 GiB
  // each and stay in the CHECKPOINT (the container holds only q/k/wkv), so `engram_rows` is what a
  // loader needs to find them.
  std::vector<int32_t>  engram_layers;
  std::vector<uint64_t> engram_rows;
  uint32_t engram_max_ngram = 0;
  uint32_t engram_vocab = 0;              // bucket count each (n-gram size, head) searches primes from
  uint32_t engram_n_heads = 0;
  uint32_t engram_head_dim = 0;
  uint32_t engram_pad_token = 2;
  uint32_t engram_compressed_vocab = 0;
  // Routing. V4 was sigmoid + a token-id hash table on layers 0-2; V4.1 is sqrt-softplus with no
  // hash layers, and carries a SECOND selection bias used inside image spans.
  enum class ScoreFunc : uint8_t { Sigmoid, Softmax, SqrtSoftplus };
  ScoreFunc score_func = ScoreFunc::Sigmoid;
  float     gate_temp = 1.0f;
  // The draft's own expert pool, which V4.1 sizes below the target's (128 of 384, 3 active).
  uint32_t dspark_n_expert = 0;
  uint32_t dspark_n_expert_used = 0;

  uint32_t ratio_for(uint32_t layer) const {
    return layer < compress_ratios.size() ? (uint32_t)compress_ratios[layer] : 0u;
  }
  bool uses_hash_routing(uint32_t layer) const { return layer < n_hash_layer; }
  static bool in_list(const std::vector<int32_t>& v, uint32_t l) {
    for (int32_t x : v) if ((uint32_t)x == l) return true;
    return false;
  }
  // V4 has no source layers, so every compressing layer is its own source and the V4 paths below
  // keep behaving exactly as they did.
  bool is_kv_source(uint32_t layer) const {
    return v41 ? in_list(kv_source_layers, layer) : ratio_for(layer) != 0;
  }
  bool is_index_source(uint32_t layer) const {
    return v41 ? in_list(index_source_layers, layer) : ratio_for(layer) == 4;
  }
  bool is_engram_layer(uint32_t layer) const { return in_list(engram_layers, layer); }
  // head_dim in the config is the FULL latent width (512); AttnConfig splits it into the
  // unrotated lead and the rotated tail.
  AttnConfig attn() const {
    AttnConfig a;
    a.n_head = n_head; a.head_dim = head_dim - rope_dim; a.rope_dim = rope_dim;
    a.sliding = sliding; a.rope_theta = rope_theta;
    return a;
  }
};

// Parses the subset of a DeepSeek-V4 config.json the runtime needs. Tolerant of unknown keys.
bool parse_model_config(const std::string& json, ModelConfig* c, std::string* err);

// Per-layer weight bindings. Expert matrices resolve through the .aff expert pool (O(1)
// addressing); everything else is a named tensor. Matrices are bf16, vectors f32.
struct LayerWeights {
  const AffLayerDesc* experts = nullptr;
  const float* attn_norm = nullptr;
  const float* ffn_norm  = nullptr;
  const float* attn_sinks = nullptr;      // [n_head] — seeds the softmax, never a key
  const uint16_t* router_w = nullptr;     // [n_expert, n_embd], bf16
  int32_t      router_gpu = -1;           // handle from DenseOps::reg_router, or -1
  const float* router_b = nullptr;        // aux-loss-free correction bias, may be null
  std::vector<int32_t> hash_table;        // [vocab, n_expert_used], hash layers only

  // Attention. There is no kv_b_proj: the absorbed-weight transform ships baked in, so `wkv`
  // produces the shared latent directly.
  DenseW wq_a ;         // [q_lora_rank, n_embd]
  DenseW wq_b ;         // [n_head*qk_width, q_lora_rank]
  DenseW wkv  ;         // [qk_width, n_embd]
  DenseW wo_a ;         // [8*o_lora_rank, n_embd] — 8 independent groups
  DenseW wo_b ;         // [n_embd, 8*o_lora_rank]
  int32_t q_norm_gpu = -1;
  int32_t attn_norm_gpu = -1, ffn_norm_gpu = -1;
  int32_t hc_attn_scale_gpu = -1, hc_attn_base_gpu = -1;
  int32_t hc_ffn_scale_gpu = -1, hc_ffn_base_gpu = -1;
  int32_t kv_norm_gpu = -1;
  const float* q_norm = nullptr;
  const float* kv_norm = nullptr;

  // Compressor (present when the layer's ratio is non-zero) and lightning indexer (ratio 4 only).
  DenseW comp_wkv ;
  DenseW comp_wgate ;
  const float*    comp_ape = nullptr;
  const float*    comp_norm = nullptr;
  DenseW idx_wq_b ;
  DenseW idx_proj ;
  DenseW idx_comp_wkv ;
  DenseW idx_comp_wgate ;
  const float*    idx_comp_ape = nullptr;
  const float*    idx_comp_norm = nullptr;
  // V4.1 only. The indexer's keys come from the layer's own projection rather than a second
  // compressor, and only kv-source layers have them; `idx_wq_b`/`idx_proj` sit on the (larger)
  // set of index-source layers.
  DenseW idx_wk ;
  const float*    idx_k_norm = nullptr;
  // The routing bias used inside image spans. Null on a text-only load and on V4.
  const float*    router_b_vl = nullptr;
  // Engram, on `engram_layers` only: q/k are [hc_mult, n_embd] mixes and wkv projects the looked-up
  // n-gram rows. The TABLE itself is not in the container.
  DenseW engram_wkv ;
  const float*    engram_q = nullptr;
  const float*    engram_k = nullptr;

  // Shared expert — dense, runs on every token alongside the routed six.
  DenseW shexp_gate ;
  DenseW shexp_up ;
  DenseW shexp_down ;

  // Hyper-connection control projections, one pair per sublayer.
  DenseW hc_attn_fn ;
  const float*    hc_attn_base = nullptr;
  const float*    hc_attn_scale = nullptr;
  DenseW hc_ffn_fn ;
  const float*    hc_ffn_base = nullptr;
  const float*    hc_ffn_scale = nullptr;

  RopeParams rope;                        // resolved once at load: theta and YaRN per layer
  uint32_t   ratio = 0;
  // Whose compressed cache this layer reads: itself on V4, the nearest kv-source on V4.1.
  uint32_t   kv_owner = 0;
};

// Per-layer rolling state for one sequence — COUNTS ONLY. Every key this names lives in VRAM: the
// raw ring is written by kv_commit, the compressed rows by compress/compress_one, and the indexer
// scores them where they are. Do not add host mirrors back: at a 1M window they are tens of GiB of
// first-touched zeroes that nothing reads, taken out of the RAM the expert pool wants. `comp_rows`
// stays because the batched mask plane is strided by the compressed capacity.
struct LayerState {
  uint32_t n_raw = 0, raw_head = 0;
  uint64_t comp_rows = 0;                 // compressed capacity, in rows
  uint32_t n_comp = 0, n_idx_comp = 0;
};

struct SeqState {
  std::vector<LayerState> layer;
  std::vector<float> hc;                  // [hc_mult][n_embd] — the residual carrier
  uint32_t pos = 0;
  // ---- WHETHER `hc` ABOVE MEANS ANYTHING ------------------------------------------------------
  //
  // forward_token carries the residual through `hc` on the host. The BLOCK path does not: the lanes
  // live in VRAM for the whole block and only the winners come back, so `hc` is whatever the last
  // pass left there.
  //
  // Writing it anyway, from the block's last position, does not make the contract hold: `rollback`
  // discards the positions after the accepted prefix and does not touch `hc`, so any partially
  // accepted block — which is most of them — leaves `hc` holding a position the sequence has rewound
  // past, and a caller mixing the two paths gets a plausible wrong continuation rather than an
  // error. So the flag says which path owns the lanes, and forward_token refuses on a state a block
  // last touched.
  bool hc_host = true;
  // V4.1 only: the mix the LAST layer's FFN produced, which is what the head collapses the lanes
  // with (V4 computes its own from hc_head_fn). Written by the FFN sublayer of the final layer;
  // `valid` guards a head that would otherwise collapse with zeros and emit confident nonsense.
  float last_ffn_pre[8] = {0};
  bool  last_ffn_pre_valid = false;
};

// The DSpark draft, loaded from a companion container.
//
// A stage IS a LayerWeights — the mtp blocks are structurally ordinary layers with ratio 0, no
// hash table and no compressor — so the whole per-layer binder is reused and only the pieces that
// are the draft's own live here. `main_proj` takes the target's tapped hidden states (hc lanes
// averaged, `dspark_taps` of them concatenated) down to one n_embd vector; the embedding and the
// output head are the TARGET's, shared exactly as `Transformer.__init__` shares them.
struct DsparkWeights {
  std::vector<LayerWeights> stage;
  DenseW main_proj;                     // [n_embd, n_embd * taps]
  // main_norm runs on the CARD, right after main_proj's GEMM, so it needs a registered handle and
  // not the container's mapping — a host pointer reaches the kernel as an unmapped address and the
  // run dies with a page fault nowhere near here. `norm` is the opposite: it is applied on the host,
  // to five collapsed vectors, just before the vocabulary head.
  const float* main_norm = nullptr;
  int32_t      main_norm_gpu = -1;
  // ...and `norm` is applied on the card too now, by the same kernel family that gates the lanes.
  // The host pointer stays bound because the reference epilogue in dspark_draft is what the device
  // one was checked against and is still the thing to fall back to when reading this code.
  const float* norm = nullptr;
  int32_t      norm_gpu = -1;
  DenseW hc_head_fn;
  const float* hc_head_base = nullptr;
  const float* hc_head_scale = nullptr;
  int32_t      hc_head_base_gpu = -1;
  int32_t      hc_head_scale_gpu = -1;
  // The rank-`dspark_markov_rank` bigram correction. w1 is an EMBEDDING (token id -> rank), w2 a
  // head (rank -> vocab); its bias on the logits is what makes the block autoregressive without
  // running the three stages again, and its embedding is half the confidence head's input.
  //
  // Both are dense weights and not host pointers: w2 is 66 MB and the loop reads it once per drafted
  // token, so on the host a five-token block would be 330 MB of DRAM — more than the three stages
  // and the vocabulary head together.
  DenseW markov_w1;                     // [vocab][rank] bf16
  DenseW markov_w2;                     // [vocab][rank] bf16
  // Bound and checked, never evaluated. The confidence head scores a drafted token so a caller can
  // stop a block early; the accept rule here is exact agreement with the target's argmax, which is
  // what makes speculative decode bit-identical to greedy decode, and that rule has no use for a
  // score. Left in the container's contract because the moment a variable block length is worth
  // measuring, the missing piece should be the policy and not the weight.
  const uint16_t* confidence = nullptr; // [1][n_embd + rank] bf16
  bool ready() const {
    return !stage.empty() && main_proj && markov_w1 && markov_w2 && main_norm_gpu >= 0 && norm;
  }
};

class Model {
public:
  bool load(const std::string& aff_path, std::string* err);
  // The companion container written by `aff-quantize --dspark`. Call AFTER load(): it binds
  // against the config the main container carried, and its stages register their weights through
  // the same DenseOps.
  //
  // Returns false with a reason on ANY problem, and the caller is expected to exit. There is no
  // degraded mode: a half-bound draft still emits token ids, they are simply never accepted, and
  // the run then reports a decode rate that looks like an honest negative result about
  // speculation. `dspark_` is left empty so nothing downstream can half-run.
  bool load_dspark(const std::string& path, std::string* err);
  const DsparkWeights& dspark() const { return dspark_; }
  // The draft's container, for the second StaticPlacement its expert pool needs.
  const AffReader& dspark_reader() const { return dspark_aff_; }
  uint64_t dspark_bytes() const { return dspark_.stage.empty() ? 0 : dspark_aff_.header().file_size; }

  const ModelConfig& config() const { return cfg_; }
  const AffReader& reader() const { return aff_; }
  // Non-const, for `set_profile_override` alone. The container's bytes stay read-only — the only
  // mutable state on the reader is which expert-popularity profile it hands out, which is a
  // property of the RUN and not of the file. Kept as a separate name so that reaching for it is a
  // deliberate act rather than the default way to get at the reader.
  AffReader& reader_mut() { return aff_; }
  const Codebook& codebook() const { return cb_; }

  // ---- oracle knobs, both set BEFORE load() ---------------------------------------------------
  //
  // Neither is reachable from serving. They exist because the lightning indexer is the one part of
  // the forward pass no oracle could reach: it engages only once the compressed cache outgrows
  // `index_topk`, and n_comp is pos/ratio, so at the shipped 512 nothing below ~2048 positions runs
  // it at all. A numpy reference over 2048 positions and 43 layers is hours; over 300 positions and
  // 4 layers it is a minute. Lowering index_topk moves the SAME code path into that range.
  //
  // The comparison is only honest if both sides are lowered together — `ref_forward.py` takes the
  // matching --index-topk.
  void set_index_topk(uint32_t k) { index_topk_override_ = k; }
  void set_max_layers(uint32_t n) { max_layers_ = n; }

  void init_state(SeqState* s, uint64_t max_pos) const;

  // `expert_fetch` is the hook the residency manager owns: given (layer, expert, which) it hands
  // back a view of those weights, from VRAM or RAM. That indirection is the whole hybrid design —
  // the layer loop never learns where an expert lives.
  using ExpertFetch = ExpertMatrixView (*)(void* ctx, uint32_t layer, uint32_t expert, int which);

  // Everything one layer's two compressors need for one chunk. The KV compressor and the indexer's
  // run as one call because they read the same activation and their only difference is head_dim and
  // what the emitted row is encoded as.
  //
  // `slot0`/`idx_slot0` are the caller's row counts before this chunk; `n_out` rows are appended.
  // The caller derives n_out itself — it is just the positions in [pos0, pos0+n) divisible by ratio
  // — because it needs the same count to size the per-token key counts attention will be given.
  struct CompressArgs {
    uint32_t layer = 0, n = 0, pos0 = 0, ratio = 0, n_out = 0;
    int32_t h_kv = -1, h_sc = -1;             // comp_wkv, comp_wgate
    int32_t h_ikv = -1, h_isc = -1;           // idx_comp_wkv, idx_comp_wgate
    int32_t h_iproj = -1;                     // idx_proj, or -1 when the indexer will not run
    const float* ape = nullptr;    const float* norm = nullptr;
    const float* iape = nullptr;   const float* inorm = nullptr;
    uint32_t width = 0, idx_dim = 0, n_rot = 0, n_embd = 0;
    uint32_t slot0 = 0, idx_slot0 = 0;
    uint32_t idx_heads = 0;                   // rows of idx_proj; 0 skips it
    float iscale = 0.0f;                      // 1/sqrt(idx_dim * idx_heads), folded in on device
    RopeDerived rope{};
    float eps = 0.0f;
  };

  // Device execution of the dense path. Registered before load() so the loader can hand every
  // dense tensor to the GPU as it binds it; `mv`/`mvg` then run those weights on device and the
  // bytes never cross the host bus again. Kept as a hook so the engine core stays free of HIP.
  struct DenseOps {
    int32_t (*reg)(void* ctx, const char* name, const void* host, const uint8_t* scales,
                   uint32_t quant, uint64_t rows, uint64_t cols) = nullptr;
    void (*mv)(void* ctx, int32_t h, const float* x, uint64_t rows, uint64_t cols, float* y) = nullptr;
    void (*mvg)(void* ctx, int32_t h, const float* x, uint32_t n_groups, uint64_t group_dim,
                uint64_t rank, float* y) = nullptr;
    // Several matrices against the same x in one round trip. The compressor's four matvecs all read
    // `norm`, and issuing them separately cost four synchronisations per layer for 10 MB of weights.
    bool (*mv_multi)(void* ctx, const int32_t* h, const uint64_t* rows, uint64_t cols, uint32_t n,
                     const float* x, float* const* y) = nullptr;
    // Every matvec in the attention sublayer that reads `norm` and feeds nothing else inside it, in
    // ONE launch: wq_a and wkv on top of the compressor's four. Those two are the FIRST link of the
    // attn_q and attn_kv chains, so they are independent of everything until their own rms_norm —
    // and issued separately they are also the two worst-shaped launches in the layer, 1024 and 512
    // rows asking for a fraction of a 64-CU card each while carrying its full dispatch ramp.
    //
    // Leaves both projections in VRAM for attn_q/attn_kv to pick up, which is why those two must be
    // called next and on this same context. The compressor's outputs are left IN FLIGHT to the host;
    // comp_collect finishes them and returns what mv_multi would have. Returns false if it could not
    // run, in which case the caller uses mv_multi and the two chains run whole.
    bool (*attn_pre)(void* ctx, int32_t qa, int32_t kv, uint64_t qr, uint32_t width,
                     const int32_t* h, const uint64_t* rows, uint64_t cols, uint32_t n,
                     float* const* y) = nullptr;
    bool (*comp_collect)(void* ctx) = nullptr;
    // Fused SwiGLU FFN, entirely device-resident. Returns false when it cannot run and the caller
    // should use the unfused path.
    // `defer` issues the chain and returns without synchronising; shexp_collect finishes it. The
    // shared expert is independent of the routed experts, and whatever share of those runs on the
    // CPU is dead time on the card — deferring is what lets the two overlap instead of queueing.
    bool (*shexp)(void* ctx, int32_t g, int32_t u, int32_t d, const float* x, uint64_t hidden,
                  uint64_t inter, float limit, float* out, bool defer) = nullptr;
    bool (*shexp_collect)(void* ctx, float* out, uint64_t hidden) = nullptr;
    // Registers a small f32 vector (a norm weight) on device. Returns a handle or -1.
    int32_t (*reg_vec)(void* ctx, const float* host, uint64_t n) = nullptr;
    // ---- the router gate ------------------------------------------------------------------------
    // [n_expert, n_embd] f32, one per layer. On the host this is read out of the SAME DRAM the
    // expert kernel is saturating, and it holds the cards idle while it runs — which costs more than
    // the arithmetic does, because a burst after an idle gap is slow to recover (gpu/keepalive.h).
    //
    // Only the logits come back; top-k stays on the host, where it is trivial.
    int32_t (*reg_router)(void* ctx, const uint16_t* w, const float* bias, uint32_t n_expert,
                          uint64_t n_embd) = nullptr;
    // `x` null means "use the norm already in VRAM". Returns false if it could not run, in which
    // case the caller does the matvec on the host.
    bool (*router)(void* ctx, int32_t h, const float* x, uint32_t n_expert, uint64_t n_embd,
                   float* logits) = nullptr;
    // Q path: wq_a, learned norm, wq_b, weightless per-head norm, RoPE — one round trip.
    // qrnorm_out is written back because the indexer consumes it.
    bool (*attn_q)(void* ctx, int32_t qa, int32_t qb, int32_t qnorm, const float* x,
                   uint64_t hidden, uint64_t qr, uint32_t n_head, uint32_t width, uint32_t n_rot,
                   uint32_t pos, RopeDerived rope, float eps, float* q_out,
                   float* qrnorm_out) = nullptr;
    // KV path: wkv, learned norm, RoPE. The FP8 QAT round-trip stays on the host — it is 512
    // floats and moving it would buy nothing.
    bool (*attn_kv)(void* ctx, int32_t kv, int32_t kvnorm, const float* x, uint64_t hidden,
                    uint32_t width, uint32_t n_rot, uint32_t pos, RopeDerived rope, float eps,
                    float* kv_out) = nullptr;
    // ---- KV cache in VRAM ----------------------------------------------------------------------
    void (*kv_sinks)(void* ctx, uint32_t layer, const float* sinks, uint32_t n) = nullptr;
    // `pos` is the ABSOLUTE position, not a ring slot. The raw ring is wider than the sliding
    // window (attn_ring_slots() in gpu/attention_gpu.h) and only the device knows how much wider,
    // so the slot rule lives there and nowhere else. The latent never crosses the bus: attn_kv
    // leaves it in VRAM and this encodes it in place.
    bool (*kv_commit)(void* ctx, uint32_t layer, uint32_t pos, uint32_t width, uint32_t n_rot) = nullptr;
    // Both compressors for ONE token, and the lightning indexer for one token. Decode must use
    // these once prefill has: the cross-chunk pooling window lives in VRAM, so a host-side pool
    // would run over a state nothing has updated since the prompt.
    // `indexer_one` leaves the admission mask on device, so `attend` is then called with a null
    // `allowed` and picks it up there.
    bool (*compress_one)(void* ctx, const CompressArgs& a) = nullptr;
    // Which layer's compressed cache each layer READS (V4.1 shares one per kv-source group; V4
    // gives every compressing layer its own, and then this is the identity). Handed over once, at
    // load: the raw sliding window stays per layer, so only the compressed half is redirected.
    void (*set_kv_owner)(void* ctx, const uint32_t* owner, uint32_t n) = nullptr;
    // V4.1 shifts the hyper-connection mixes by half a sublayer — each sublayer collapses with what
    // the PREVIOUS one computed. V4's own reference collapses with its own, which is what affinity
    // has always done, so this is off there and nothing about that path changes.
    void (*set_shift_pre)(void* ctx, bool v) = nullptr;
    // `n_keys` is what the indexer SCORES and `n_mask` what the attention will READ — the two
    // compressors have separate capacities, so the second can be the larger and the rows between
    // them must come back "not admitted" rather than stale.
    bool (*indexer_one)(void* ctx, uint32_t layer, int32_t h, uint64_t qr, uint32_t n_head,
                        uint32_t dim, uint32_t n_keys, uint32_t n_mask, uint32_t topk, uint32_t pos,
                        uint32_t n_rot, RopeDerived rope) = nullptr;
    // Tells attn_q to keep qr_norm in VRAM for indexer_one rather than copying it to the host.
    void (*want_qrn_dev)(void* ctx, bool v) = nullptr;
    bool (*attend)(void* ctx, uint32_t layer, const float* q, const uint8_t* allowed,
                   uint32_t pos, uint32_t n_comp, float scale, float* heads) = nullptr;
    // Inverse RoPE + grouped wo_a + wo_b on the heads attention left in VRAM.
    bool (*attn_out)(void* ctx, int32_t woa, int32_t wob, uint32_t n_head, uint32_t width,
                     uint32_t n_rot, uint32_t pos, RopeDerived rope, uint32_t n_groups,
                     uint64_t rank, uint64_t hidden, float* block) = nullptr;
    // Whether the fused device-resident chains can run at all. This has to be ASKED, not assumed:
    // the caller decides on that basis whether to leave attention's heads in VRAM, and if it
    // assumes wrongly the heads are never brought back and the fallback rotates a stale buffer into
    // the output projection. That produced fluent, wrong text on a two-rank split.
    bool (*fused_ok)(void* ctx) = nullptr;
    // ---- the hidden state, resident in VRAM ------------------------------------------------
    // The 4-lane hyper-connection state IS the model's hidden state. While it lived on the host
    // every layer dragged it back across the bus; keeping it on device is what makes a sublayer
    // device-resident. `norm` still returns once per sublayer because the router, the expert
    // dispatch, the compressor and the indexer have not been ported yet.
    bool (*hc_ready)(void* ctx) = nullptr;
    void (*hc_seed)(void* ctx, const float* embed, uint32_t n_embd, uint32_t n_hc) = nullptr;
    // `defer` enqueues the copy-back of `norm` and returns without synchronising; norm_collect
    // finishes it. The FFN-side call is the most expensive device entry point in the engine, once a
    // layer, and the sync is pure waste on the common path because the device router runs next on
    // the same stream and synchronises anyway. See the call site in model.cpp.
    bool (*hc_pre)(void* ctx, int32_t fn, int32_t scale_v, int32_t base_v, int32_t norm_w,
                   uint32_t n_embd, uint32_t n_hc, uint32_t iters, float hc_eps, float rms_eps,
                   float* norm_host, bool defer) = nullptr;
    // Idempotent: a no-op when no copy is outstanding, so every host read of `norm` can call it.
    bool (*norm_collect)(void* ctx, float* norm_host, uint32_t n_embd) = nullptr;
    bool (*hc_post)(void* ctx, bool has_control, uint32_t n_embd, uint32_t n_hc) = nullptr;
    bool (*block_add)(void* ctx, const float* src, uint32_t n_embd) = nullptr;
    // Sum the per-card blocks so every card holds the sublayer's whole contribution. Every matrix
    // that is split along its INPUT dimension — the output projection, the shared expert's `down`,
    // every routed expert's `down` — leaves a PARTIAL behind, and this is the one place per
    // sublayer where those become a value. Must be called after the last deposit and before
    // hc_post, which folds the block into the hidden state. Skipping it is not a slow path, it is
    // half an answer that still looks like text.
    bool (*block_reduce)(void* ctx, uint32_t n_embd) = nullptr;
    void (*hc_read)(void* ctx, float* dst, uint32_t n_embd, uint32_t n_hc) = nullptr;
    void* ctx = nullptr;
  };
  void set_dense_ops(const DenseOps& d) { dops_ = d; }

  // ---- the same forward pass, for a CHUNK of tokens ---------------------------------------------
  //
  // Prefill was a loop over forward_token, so it read every dense weight once per token and paid a
  // stream synchronise per sublayer per token. In the profile the router bucket alone ran orders of
  // magnitude above its own bandwidth cost. Nothing there is a slow kernel; it is a batch size of
  // one.
  //
  // The pass batches cleanly because THE RESIDUAL STREAM IS PER TOKEN. Within one layer, every
  // token's hidden state is already final from the layer below, so the norms, all the projections,
  // the router and the shared expert are `nb` independent problems and become GEMMs. Only three
  // things are ordered inside the chunk — the KV ring, the compressor's window and attention's
  // causal mask — and those stay a loop over tokens, on device, without a synchronise.
  //
  // Buffers are dim-major ([dim][nb]) throughout; see gpu/batch_kernels.h for why. `nb` is the
  // padded stride, `n_live` the tokens that actually exist. Everything crossing to the host is
  // token-major, because that is what the routed-expert dispatch and the compressor want.
  // One position's outcome under sampling. `p_query` is the probability the proposal is accepted
  // with; `tok` is the draw to emit when nothing was proposed or everything was accepted; `tok_excl`
  // is the draw for the rejection branch, from the same distribution with the proposal removed.
  struct PosDraw { uint32_t tok = 0, tok_excl = 0; float p_query = 0.0f; };

  struct BatchOps {
    // Largest chunk the device was sized for. 0 means there is no batched path and the caller
    // should loop forward_token.
    uint32_t (*cap)(void* ctx) = nullptr;
    // Open a chunk. `emb` is [nb][n_embd], the dequantised embedding of every token in it, and
    // seeds all n_hc lanes. Columns past n_live are zeroed so padding can never carry a NaN into
    // an arithmetic that would otherwise be finite.
    bool (*begin)(void* ctx, uint32_t n_live, uint32_t pos0, const float* emb, uint32_t n_embd,
                  uint32_t n_hc) = nullptr;
    // control + reduce + rms_norm for every token. `norm_host`, when non-null, receives the result
    // token-major — the router's top-k, the compressor and the expert dispatch all read it there.
    bool (*hc_pre)(void* ctx, int32_t fn, int32_t sv, int32_t bv, int32_t nw, uint32_t n_embd,
                   uint32_t n_hc, uint32_t iters, float hc_eps, float rms_eps,
                   float* norm_host) = nullptr;
    // Q and KV in one call: wq_a, the learned norm, wq_b, the per-head norm and RoPE; wkv, its
    // norm and RoPE. Both results stay in VRAM for kv_commit and attend to pick up. `qrnorm_host`
    // is the lightning indexer's only input and is copied back only when it asks.
    bool (*attn_qkv)(void* ctx, int32_t qa, int32_t qb, int32_t qnorm, int32_t kv, int32_t kvnorm,
                     uint64_t hidden, uint64_t qr, uint32_t n_head, uint32_t width, uint32_t n_rot,
                     RopeDerived rope, float eps, float* qrnorm_host) = nullptr;
    // Batched matvecs over an activation the chunk already has in VRAM, results token-major on the
    // host. `src` is 0 for the sublayer norm and 1 for the query path's qr_norm. This is what the
    // compressor and the lightning indexer use: their pooling and their top-k are sequential in
    // position and stay on the host, but the projections that feed them are ordinary GEMMs.
    bool (*mv_host)(void* ctx, int src, const int32_t* h, const uint64_t* rows, uint64_t cols,
                    uint32_t n, float* const* y) = nullptr;
    // Both of the layer's compressors for the whole chunk: the four projections, the softmax pool,
    // the RoPE, the QAT and the cache write. This replaces the sequential host walk that pass one
    // used to be — the pool is a window over eight positions, not an accumulator, so every emitted
    // row is independent. See gpu/compressor_gpu.h.
    bool (*compress)(void* ctx, const CompressArgs& a) = nullptr;
    // Discards the cross-chunk window for every layer. Called when a sequence is reset, because the
    // device owns the pooling state.
    bool (*compress_reset)(void* ctx) = nullptr;
    // Encode tokens [b0, b0+n) of the chunk into the layer's raw ring, and then attend for the
    // same range — one launch each. Committing the whole sub-batch before any of it attends is
    // only safe because the ring holds sliding + kAttnBatch - 1 rows; see attention_gpu.h.
    bool (*kv_commit)(void* ctx, uint32_t layer, uint32_t b0, uint32_t n, uint32_t pos0,
                      uint32_t width, uint32_t n_rot) = nullptr;
    // `allowed` is a host [n][mask_stride] plane or null and `n_comp` an array of n per-token
    // counts. The stride is the CALLER's, because the device plane's is the compressed cache's
    // capacity and the two have no reason to agree.
    bool (*attend)(void* ctx, uint32_t layer, uint32_t b0, uint32_t n, uint32_t pos0,
                   const uint8_t* allowed, uint32_t mask_stride, const uint32_t* n_comp,
                   float scale) = nullptr;
    // The indexer's query projection, left on device for `indexer` to consume.
    bool (*indexer_q)(void* ctx, int32_t h, uint64_t rows, uint64_t cols, uint32_t n_head,
                      uint32_t dim) = nullptr;
    // The lightning indexer for a whole chunk. Queries, head weights and keys are all already in
    // VRAM — indexer_q and compress put them there — so only the per-token counts come from here.
    // Leaves the admission mask on device, so `attend` is called with a null `allowed`.
    bool (*indexer)(void* ctx, uint32_t layer, uint32_t n, const uint32_t* n_comp, uint32_t n_keys,
                    uint32_t n_head, uint32_t dim, uint32_t topk, uint32_t pos0, uint32_t n_rot,
                    RopeDerived rope) = nullptr;
    bool (*attn_out)(void* ctx, int32_t woa, int32_t wob, uint32_t n_head, uint32_t width,
                     uint32_t n_rot, RopeDerived rope, uint32_t n_groups, uint64_t rank,
                     uint64_t hidden) = nullptr;
    bool (*hc_post)(void* ctx, bool has_control, uint32_t n_embd, uint32_t n_hc) = nullptr;
    // `norm` on demand. hc_pre no longer hands it back every FFN sublayer, because in the
    // all-resident case nothing on the host reads it — only hash routing and the host expert tier
    // do, and both know they need it before they ask.
    bool (*norm_fetch)(void* ctx, float* norm) = nullptr;
    // The gate GEMM and the top-k together, with only the selection coming back. `bias` and
    // `resident` are host arrays of n_expert; `resident` null means "do not restrict".
    bool (*router_topk)(void* ctx, int32_t h, uint32_t n_expert, uint64_t n_embd, uint32_t k,
                        const float* bias, const uint8_t* resident, int norm_topk_prob,
                        float routed_scaling, uint32_t* sel, float* wt, uint32_t* nsel) = nullptr;
    // The same gate GEMM for a selection the caller already made — hash-routed layers, whose
    // experts come from a table keyed on the token id. `sel` [live][k] and `nsel` [live] go up,
    // `wt` [live][k] comes back, and `norm` never leaves the card.
    bool (*router_hash)(void* ctx, int32_t h, uint32_t n_expert, uint64_t n_embd, uint32_t k,
                        const uint32_t* sel, const uint32_t* nsel, int norm_topk_prob,
                        float routed_scaling, float* wt) = nullptr;
    // [n_expert, nb] logits back token-major; top-k stays on the host where it is 256 elements.
    bool (*router)(void* ctx, int32_t h, uint32_t n_expert, uint64_t n_embd,
                   float* logits) = nullptr;
    bool (*shexp)(void* ctx, int32_t g, int32_t u, int32_t d, uint64_t hidden, uint64_t inter,
                  float limit) = nullptr;
    // The routed experts still run partly on the host, so their contribution arrives token-major
    // and is added into the device block.
    bool (*block_add)(void* ctx, const float* src, uint32_t n_embd) = nullptr;
    // See DenseOps::block_reduce. Same contract, over the whole chunk.
    bool (*block_reduce)(void* ctx, uint32_t n_embd) = nullptr;
    // One token's collapsed lanes, for the head. PREFILL ONLY — it needs the last position's on
    // the host to run the reference epilogue there. The block path does not use this: `collapse`
    // below does the same arithmetic for every position at once, on the card.
    bool (*hc_read)(void* ctx, uint32_t b, float* dst, uint32_t n_embd, uint32_t n_hc) = nullptr;
    // The head's epilogue for the whole block, in three kernels, leaving the normed hidden state
    // in VRAM in both layouts the head can consume. This is `draft_collapse` with the target's
    // handles rather than the draft's — literally the same function, because the reference
    // epilogue they transcribe is the same one.
    bool (*collapse)(void* ctx, int32_t fn, int32_t sv, int32_t bv, int32_t nw, uint32_t n_embd,
                     uint32_t n_hc, float hc_eps, float rms_eps) = nullptr;
    // V4.1: its head has no mixer, so the epilogue reuses the mix the last layer's FFN left on the
    // device rather than computing one from `fn` — same kernel, same order, one input fewer.
    bool (*collapse_pre)(void* ctx, int32_t nw, uint32_t n_embd, uint32_t n_hc, float hc_eps,
                         float rms_eps) = nullptr;
    // The lanes AVERAGED — `h.mean(dim=2)`, which is what DSpark conditions its draft on — written
    // into the card's own tap plane rather than handed back. Nothing crosses the bus in either
    // direction: draft_proj reads the same plane on the same card. `slot` is which of the tap
    // layers this is, `b_lo`/`n_keep` which of the live tokens are inside the trailing window the
    // caller asked to keep, and `row0` where the first of them lands.
    bool (*hc_tap)(void* ctx, uint32_t slot, uint32_t n_taps, uint32_t b_lo, uint32_t n_keep,
                   uint32_t row0, uint32_t rows_cap, uint32_t n_embd, uint32_t n_hc) = nullptr;
    // The vocabulary head for every token of the chunk, with only the argmax coming back. `x` is
    // [n][n_embd] token-major — the collapsed, normed hidden state — and `ids` [n].
    //
    // Prefill runs the head once, through DenseOps::mv, because it throws away every position but
    // the last. A speculative block needs all of them, and the weight is 529 MB a card: six matvecs
    // read it six times, one GEMM over six columns reads it once.
    bool (*head)(void* ctx, int32_t h, uint64_t vocab, uint64_t n_embd, uint32_t n, const float* x,
                 uint32_t* ids) = nullptr;
    // The same head, sampling instead of taking the argmax, and returning what a rejection-sampled
    // speculative step needs at each position: the accept probability of the token the draft
    // proposed, a draw, and a draw with that proposal removed. See gpu/sampler_gpu.h for why those
    // three, and BlockOut::sample for how a caller asks.
    bool (*head_sample)(void* ctx, int32_t h, uint64_t vocab, uint64_t n_embd, uint32_t n,
                        const float* x, const uint32_t* query, const float* u,
                        float temperature, float top_p, uint32_t top_k, float min_p,
                        PosDraw* out) = nullptr;
    // ---- the DSpark draft --------------------------------------------------------------------
    // The stages are structurally ordinary ratio-0 layers, so everything above runs them unchanged
    // with the stage's handles. These five are what a draft has and a layer does not.
    // Reads the tap plane the target left in VRAM, at row `tap_row0`. There is no host pointer:
    // the bytes were computed on this card and never left it.
    bool (*draft_proj)(void* ctx, int32_t h, int32_t norm_w, uint64_t rows, uint64_t cols,
                       uint32_t n, uint32_t tap_row0, float eps) = nullptr;
    bool (*draft_kv)(void* ctx, uint32_t layer, int32_t kv, int32_t kvnorm, uint32_t n,
                     uint32_t pos0, uint64_t hidden, uint32_t width, uint32_t n_rot,
                     RopeDerived rope, float eps) = nullptr;
    bool (*draft_attend)(void* ctx, uint32_t layer, uint32_t n, uint32_t pos0, uint32_t win_lo,
                         float scale) = nullptr;
    // The four lanes gated, collapsed and normed on the card, leaving the head's activation where
    // the vocabulary GEMM already reads. With this bound, draft_head takes x = nullptr.
    bool (*draft_collapse)(void* ctx, int32_t fn, int32_t sv, int32_t bv, int32_t nw,
                           uint32_t n_embd, uint32_t n_hc, float hc_eps, float rms_eps) = nullptr;
    bool (*draft_head)(void* ctx, int32_t h_head, uint64_t vocab, uint64_t n_embd, uint32_t n,
                       const float* x, int32_t w1, int32_t w2, uint32_t rank, uint32_t first_id,
                       uint32_t* ids) = nullptr;
    void* ctx = nullptr;
  };
  void set_batch_ops(const BatchOps& b) { bops_ = b; }

  // Where the token's time actually goes. Enabled by AFF_PROFILE=1; zero cost otherwise (one
  // predictable branch per phase). It exists because the decomposition is otherwise done with
  // bandwidth arithmetic, which cannot see a queue.
  struct PhaseProfile {
    // ---- WORK AND WAIT ARE DIFFERENT NUMBERS AND A PHASE COUNTER CANNOT TELL THEM APART --------
    //
    // `t` is wall time in the phase. `wait` is how much of that `t` was spent blocked inside a
    // device stream drain (engine/instrument.h counts them; gpu/dense_gpu.hip's AFF_DRAIN bumps it).
    //
    // Without the split, a phase named for host work reads as that work when it is really a stream
    // synchronise for something queued behind it. The fix for a phase that is nearly all `t` is to
    // make its code faster; the fix for one that is nearly all `wait` is to go and find what it is
    // waiting FOR, which is somewhere else entirely.
    struct Ph {
      double t = 0, wait = 0;
      // Of this phase's wall time, the fraction the host spent blocked. Read this column FIRST.
      double wait_frac() const { return t > 0 ? wait / t : 0.0; }
    };
    Ph attn_proj, compressor, indexer, attention;
    Ph ffn_dense, ffn_routed, hyper, head, embed, attn_out;
    // Split out of ffn_routed: the gate matvec plus top-k. It is host DRAM traffic per layer, and
    // inside the expert bucket it made the experts look bigger than they are.
    Ph router;
    // Split out of ffn_routed for the same reason, one level down: prefill's expert bucket is the
    // per-token host top-k, the per-layer descriptor build and the GEMM launches all at once, and
    // the three want completely different fixes. `route` is the host top-k loop, `dispatch` the
    // batched device pass, `ffn_routed` whatever is left over for the host/bus tier.
    Ph route, dispatch;
    // The DSpark draft, kept out of the target's buckets entirely. A drafted token and a verified
    // one are different work at different batch sizes, and averaging them gives a "ms/token" that
    // describes neither — the same reason prefill's counters are dumped and cleared before decode.
    Ph draft_kv, draft_attn, draft_ffn, draft_head;
    // Split out of the three above, each of which otherwise hides a host round trip inside a bucket
    // named after a sublayer. `draft_seed` is the block's embedding build and upload, `draft_route`
    // the router drains, `draft_ep` the per-token epilogue on the CPU before the vocab projection.
    Ph draft_seed, draft_route, draft_ep;
    uint64_t tokens = 0;
    // Blocks issued, tokens drafted, and tokens the target agreed with. The ratio of the last two
    // is the only number that decides whether speculation pays, and it is a property of the model
    // and the stream rather than of the code — so it is reported, not assumed.
    uint64_t blocks = 0, drafted = 0, accepted = 0;
    // Marginal hit rate per drafted position: pos_hit[j]/pos_n[j] is how often the draft's j-th
    // token was the target's argmax, independent of whether the prefix before it survived.
    uint64_t pos_hit[kSpecTok] = {}, pos_n[kSpecTok] = {};
    double total() const {
      return attn_proj.t + compressor.t + indexer.t + attention.t + ffn_dense.t + ffn_routed.t
           + hyper.t + head.t + embed.t + attn_out.t + router.t + route.t + dispatch.t
           + draft_kv.t + draft_attn.t + draft_ffn.t + draft_head.t
           + draft_seed.t + draft_route.t + draft_ep.t;
    }
    // Of the whole token, how much the host spent blocked on the cards. Near 100% means the engine
    // is device-bound and no amount of host work will help; near 0% means the opposite. It is the
    // one number to read before believing any row of the table.
    double total_wait() const {
      return attn_proj.wait + compressor.wait + indexer.wait + attention.wait + ffn_dense.wait
           + ffn_routed.wait + hyper.wait + head.wait + embed.wait + attn_out.wait + router.wait
           + route.wait + dispatch.wait + draft_kv.wait + draft_attn.wait + draft_ffn.wait
           + draft_head.wait + draft_seed.wait + draft_route.wait + draft_ep.wait;
    }
  };
  const PhaseProfile& profile() const { return prof_; }
  void reset_profile() { prof_ = {}; }

  // Called once per prefill CHUNK with (tokens done, tokens in this call). A display's only way to
  // show progress through a long prompt: forward_prefill takes the whole thing and returns once,
  // and at a million positions that is minutes with nothing to show for it.
  //
  // A function pointer rather than std::function so the null check is a load and a branch, and per
  // chunk rather than per layer so it cannot become a cost. The engine does nothing with it.
  using PrefillProgress = void (*)(void* ctx, uint32_t done, uint32_t total);
  void set_prefill_progress(PrefillProgress f, void* ctx) { pp_ = f; pp_ctx_ = ctx; }
  // The acceptance counters, which the decode loop owns because only it knows what was accepted.
  // Always recorded, not just under AFF_PROFILE: acceptance is the term the whole speculative
  // speedup is linear in, so a run that does not report it has not been measured.
  // `match[j]` is whether the draft's token j equals the target's argmax at that position, for
  // EVERY j — not just the accepted prefix. The aggregate rate cannot tell "right everywhere, some
  // of the time" from "right at the first position and worthless at the fourth", and those want
  // opposite fixes: a better draft against a shorter block. The prefix length censors the later
  // positions, so the marginal rate has to be counted separately — and the target verifies all of
  // them whatever it accepts, so the information is already there.
  void note_block(uint32_t drafted, uint32_t accepted, const uint8_t* match) const {
    ++prof_.blocks; prof_.drafted += drafted; prof_.accepted += accepted;
    for (uint32_t j = 0; j < drafted && j < kSpecTok; ++j) {
      ++prof_.pos_n[j];
      if (match[j]) ++prof_.pos_hit[j];
    }
  }

  // THE ROUTED EXPERTS HAVE ONE DISPATCH: `set_expert_ffn_batch` below. There is deliberately no
  // second one — that is the last place where two engines could compute the same expert.

  // Prefill only: a whole chunk's routed experts for one layer, grouped BY EXPERT so each one's
  // weights are read once for all the tokens that chose it instead of once per token. `sel` and `w`
  // are [nb_tok][k] row-major. Writes 1 into `handled[b*k + j]` for every pair it took; the caller
  // runs the rest through the per-token path, so the two partition rather than duplicate.
  //
  // Returns how many pairs it took, 0 if it could do nothing.
  using ExpertFfnBatch = uint32_t (*)(void* ctx, uint32_t layer, const uint32_t* sel, const float* w,
                                      uint32_t nb_tok, uint32_t k, uint8_t* handled);
  void set_expert_ffn_batch(ExpertFfnBatch fn, void* ctx) { ffnb_ = fn; ffnb_ctx_ = ctx; }

  // Says whether (layer, expert) is in VRAM. With AFF_FORCE_RESIDENT=1 routing then OBEYS it: the
  // top-k is taken over the resident experts only, so nothing is fetched over PCIe. The
  // continuation is wrong on purpose — the model takes the second-best expert whenever the best did
  // not fit — but every other cost is untouched, so it measures what the engine would do if the
  // whole expert pool fit in VRAM, which is what separates a slow kernel from a badly placed
  // weight. Both the flag and the predicate are required; the flag alone does nothing.
  using ResidentPred = bool (*)(void* ctx, uint32_t layer, uint32_t expert);
  void set_resident_pred(ResidentPred fn, void* ctx) { res_ = fn; res_ctx_ = ctx; }

  // Will this layer's routed dispatch be built on the device? When it will, and nothing else in the
  // layer wants the host copy of the selection, the router is asked not to bring it back — which
  // deletes the drain that was waiting for the whole previous layer.
  //
  // The dispatch consults the SAME predicate, so the two cannot disagree. See
  // StaticPlacement::device_dispatch.
  using DeviceDispatchPred = bool (*)(void* ctx, uint32_t layer);
  void set_device_dispatch_pred(DeviceDispatchPred fn, void* ctx) { ddp_ = fn; ddp_ctx_ = ctx; }
  // True when the override is armed, so the caller can say so once at startup rather than let a
  // benchmark number be quietly incomparable.
  static bool force_resident_routing();

  // Dense matvec, on device when the weight was registered there. Every dense projection in the
  // forward pass goes through these two.
  void dense_mv(const DenseW& w, const float* x, uint64_t rows, uint64_t cols, float* y) const {
    if (dops_.mv && w.gpu >= 0) dops_.mv(dops_.ctx, w.gpu, x, rows, cols, y);
    else                        matvec_dense_mt(w, x, rows, cols, y);
  }
  void dense_mvg(const DenseW& w, const float* x, uint32_t n_groups, uint64_t group_dim,
                 uint64_t rank, float* y) const {
    if (dops_.mvg && w.gpu >= 0) dops_.mvg(dops_.ctx, w.gpu, x, n_groups, group_dim, rank, y);
    else                         matvec_dense_grouped_mt(w, x, n_groups, group_dim, rank, y);
  }

  // `routed_experts = false` zeroes the routed-expert contribution, leaving only dense weights in
  // play. That makes the pass bit-comparable against a reference that has not been quantised.
  void forward_token(uint32_t token_id, SeqState* s,
                     ExpertFetch fetch, void* fetch_ctx,
                     std::vector<float>* logits, bool routed_experts = true) const;

  // What a SPECULATIVE BLOCK needs from the same pass and prefill does not.
  //
  // Prefill computes 129280 logits for every token and throws all but the last away. A block cannot:
  // position j's argmax is the token the draft's j+1 is checked against, so every position's winner
  // is the answer. And the DSpark tap — the hc lanes averaged at each of `dspark_taps`,
  // concatenated — is what the draft is conditioned on.
  //
  // Passing one turns the head from "the last token" into "all of them" and requires the whole
  // block to fit one chunk, which a block of six always does.
  // The PROMPT's prefill asks for one of these too, with `per_position_head` off: the draft's
  // attention conditions on a sliding window of the target's tap, so the last 128 positions of the
  // prompt have to be handed over before the first block can be drafted at all.
  struct BlockOut {
    // Every position's argmax, through the block head. Off leaves `logits` filled from the last
    // token as usual — the block head's output plane is kNarrowTok columns wide and a prompt chunk
    // is hundreds, so this is not a preference.
    bool     per_position_head = true;
    // How many TRAILING positions of the tap to keep; 0 is all of them. The draft's window is 128.
    uint32_t tap_keep = 0;
    std::vector<uint32_t> greedy;         // [n] — argmax of each position's logits
    // ---- sampling, when the caller wants the model's actual distribution ------------------------
    //
    // Off leaves everything above exactly as it was, which is what prefill and the draft want: the
    // draft proposes greedily by design (`draft_sample_method: greedy`), and prefill has no token to
    // emit. On, `query`/`uniforms` come in and `draws` goes out, and `greedy` is not filled.
    bool     sample      = false;
    float    temperature = 1.0f;
    float    top_p       = 0.95f;
    float    min_p       = 0.0f;
    uint32_t top_k       = 0;
    std::vector<uint32_t> query;      // [n] the proposal at each position, 0xFFFFFFFF for none
    std::vector<float>    uniforms;   // [2n] two per position, for the two draws
    std::vector<PosDraw>  draws;      // [n]
    // The tap itself is not here and never comes back: it is written into a plane in VRAM on both
    // cards, which the draft reads on the card that wrote it. These two say WHERE it landed —
    // `tap_rows` is 0 when no draft is loaded and nothing was tapped.
    uint32_t tap_pos0 = 0;                // the absolute position the plane's row 0 belongs to
    uint32_t tap_rows = 0;
  };

  // Runs `n` tokens through the model in chunks, leaving `s` exactly as `n` calls to forward_token
  // would have and filling `logits` from the LAST of them. Returns false when the batched path is
  // unavailable (no device, no fusion), in which case the caller should loop forward_token.
  //
  // With `blk` non-null `logits` is left alone: the caller wanted every position, and the block head
  // returns ids rather than 3.1 MB of logits a block.
  bool forward_prefill(const uint32_t* ids, uint32_t n, SeqState* s,
                       ExpertFetch fetch, void* fetch_ctx,
                       std::vector<float>* logits, bool routed_experts = true,
                       BlockOut* blk = nullptr) const;

  // The draft's 768 experts are 6.48 GiB in a SECOND container with its own pool, so they are a
  // second StaticPlacement and cannot go through ffnb_/res_, which address the target's.
  //
  // There is no host tier and no `handled` fallback: an expert the card cannot reach would stream
  // 8.64 MiB over PCIe for one of five drafted tokens, which is slower than not drafting at all. A
  // draft whose experts do not all fit is a configuration error and says so at load.
  struct DraftOps {
    uint32_t (*ffn_batch)(void* ctx, uint32_t stage, const uint32_t* sel, const float* w,
                          uint32_t nb_tok, uint32_t k, uint8_t* handled) = nullptr;
    // The draft placement's own `device_dispatch`, and it must be the DRAFT's: the two placements
    // latch independently, so asking the target's would let the draft's router skip a readback its
    // dispatch still wanted. Same contract as `ddp_` — the router asks it to decide whether to bring
    // the selection back, and the dispatch asks it to decide which build runs.
    bool (*dev_plan)(void* ctx, uint32_t stage) = nullptr;
    void* ctx = nullptr;
  };
  void set_draft_ops(const DraftOps& d) { drops_ = d; }

  // Hands the draft the target's tap for `n` real positions starting at `pos0`, one KV row per
  // position in each stage's own ring. That IS the conditioning: the draft's residual stream starts
  // from a noise token and everything it knows of the target arrives through these keys.
  //
  // Called for the tail of the prompt once, and then for whatever the last block committed.
  bool dspark_seed(uint32_t n, uint32_t pos0) const;

  // Draft `dspark_block` tokens. `first_id` is the token the target has already decided for
  // position s->pos — the one not yet fed through it — and every real position up to s->pos - 1
  // must already have been handed over by dspark_seed.
  //
  // On return `out` holds `dspark_block` ids, for positions s->pos + 1 .. s->pos + dspark_block. The
  // target's state is untouched: the draft has its own ring, its own stages and its own experts.
  bool dspark_draft(uint32_t first_id, SeqState* s, uint32_t* out) const;

  // Unwinds the KV cache to `keep` positions, after a speculative block whose tail was rejected.
  //
  // Cheaper than it looks, and the reason is a property of both caches rather than an optimisation:
  // the raw ring is indexed by position and the compressor's cross-chunk window is a RING indexed by
  // `position & (hist-1)`, not an accumulator. A rejected position's slot is therefore only ever
  // read again after a real token has been written to it, because the next block starts at exactly
  // the first rejected position and covers at least as many. So nothing has to be restored — only
  // the COUNTS, which are what tells attention how much of each cache is real.
  void rollback(SeqState* s, uint32_t keep) const;

  // Declares `pos` positions of every cache real, for a state whose CONTENTS have just been written
  // from outside — a prefix cache restoring a checkpoint. The mirror of rollback(), which can only
  // ever lower a count and so cannot be reused for this: it clamps against what the state already
  // claims, and a freshly initialised state claims nothing.
  //
  // The caller is responsible for having filled every cache this then vouches for. Getting that
  // wrong does not fault — attention reads whatever the rows happen to hold — so the check that
  // matters is a hash of the caches against a straight prefill, not anything assertable here.
  void restore_state(SeqState* s, uint32_t pos) const;

private:
  ExpertMatrixView expert_view(uint32_t layer, uint32_t expert, int which) const;
  // Rewrites a routed selection in place so every expert in it is VRAM-resident. `logits` is the
  // router's output for this token, or null on the hash-routed layers, which have no logits to
  // re-rank and get the next resident id instead. No-op unless the flag and the predicate are both
  // there.
  void force_resident(uint32_t layer, const float* logits, uint32_t* sel, float* w,
                      uint32_t* n) const;

  AffReader aff_;
  AffReader dspark_aff_;
  DsparkWeights dspark_;
  ModelConfig cfg_;
  std::vector<LayerWeights> layers_;
  const uint16_t* embed_ = nullptr;
  DenseW head_;
  const float* out_norm_ = nullptr;
  ExpertFfnBatch ffnb_ = nullptr;
  void* ffnb_ctx_ = nullptr;
  DeviceDispatchPred ddp_ = nullptr;
  void* ddp_ctx_ = nullptr;
  ResidentPred res_ = nullptr;
  void* res_ctx_ = nullptr;
  DenseOps  dops_{};
  BatchOps  bops_{};
  DraftOps  drops_{};
  mutable PhaseProfile prof_{};
  PrefillProgress pp_ = nullptr;
  void* pp_ctx_ = nullptr;
  // The draft block's seed embeddings, sized once on the first block and then only row 0 rewritten.
  // See dspark_draft: rows 1..B-1 are the noise token in every block there will ever be.
  mutable std::vector<float> dspark_emb_;
  // layer -> the layer whose compressed KV it reads; see LayerWeights::kv_owner.
  std::vector<uint32_t> kv_owner_;
  DenseW hc_head_fn_;
  const float* hc_head_base_ = nullptr;
  const float* hc_head_scale_ = nullptr;
  // ...and the same three as device handles, for the block path's collapse.
  int32_t out_norm_gpu_ = -1, hc_head_base_gpu_ = -1, hc_head_scale_gpu_ = -1;
  Codebook cb_;
  // Router gates converted from an older container's f32. Owned here because the pointers handed to
  // reg_router and to the CPU path have to outlive load(); empty for a container already in bf16.
  std::vector<std::vector<uint16_t>> router_bf16_;
  uint32_t index_topk_override_ = 0;      // 0 = use the container's value
  uint32_t max_layers_ = 0;               // 0 = every layer the container holds
};

} // namespace aff
