// Dense weights resident in VRAM, executed on both GPUs and never on the CPU.
//
// Every dense byte is read every token, and host DRAM is an order of magnitude slower than VRAM, so
// these are unconditionally resident — no admission, no eviction. The routed experts stay hybrid
// because they are far too large to fit. Upload must happen BEFORE StaticPlacement::init so
// placement sizes the expert slabs from the VRAM that is left.
//
// ---- tensor parallel across however many cards there are ----------------------------------------
//
// Every op runs across EVERY rank. The count is whatever gfx12_devices() finds, one rank a card, and
// it is a runtime value throughout — kMaxRanks below only sizes tables. One card is a supported
// configuration and simply has no collective to run; the design point is still two or more, because
// what TP buys is dividing the dense weight read.
//
// Ordinary Megatron shape, one allreduce per sublayer:
//
//   Row   output dim split; rank r owns its band of the rows. First matrix of every chain, and
//         every matrix whose result goes to the host (compressor, indexer, router, vocab head) —
//         each rank writes its own slice and nothing is exchanged.
//   Col   input dim split, so rank r consumes Row's shard and yields a partial sum over the full
//         output. Second matrix of every chain: attn_out_b, shexp_down, every expert's `down`.
//   Rep   every card holds and computes the whole tensor. For matrices small enough that splitting
//         would save less reading than the exchange costs: wq_a (4 MiB), wkv (2.25 MiB), the
//         hyper-connection control (0.4 MiB). A Col matrix whose columns do not divide the rank
//         count on the alignment boundary is demoted to Rep rather than refused — see reg().
//
// Attention is head-parallel against a REPLICATED KV cache. MQA gives one KV head shared by all 64
// query heads, so there is no head axis in the cache to split along; replication costs a copy per
// card and keeps the sublayer communication-free until wo_b. That trade gets worse as ranks are
// added — the cache is paid N times, and a card's share of the heads shrinks until the flash
// kernel's arithmetic intensity suffers — so past two cards the right shape is to split the cache by
// POSITION and merge the partial attentions by log-sum-exp. That is unbuilt; see attn_flash.hip.
//
// The exchange is a few KiB twice a layer at decode, and latency-bound at that size.

#pragma once

#include "engine/model.h"
#include "gpu/attention_gpu.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace aff {

struct TpContext;

// A SIZING BOUND for per-rank tables, not the rank count. The live count is DenseGpu::ranks(),
// resolved from gfx12_devices() at init, and every loop over ranks must use it — a loop that runs to
// kMaxRanks touches entries no device backs.
//
// Kept equal to hip_common.h's kMaxGpuDevices; dense_gpu.h is included by CPU translation units that
// must not pull in the HIP headers, so the two are asserted equal in dense_gpu.hip rather than
// shared.
constexpr size_t kMaxRanks = 8;

struct DenseGpuStats {
  uint64_t tensors = 0;
  uint64_t bytes = 0;            // summed across ranks, i.e. the whole model
  uint64_t calls = 0;
  size_t   ranks = 0;
  double   upload_seconds = 0.0;
  uint64_t kv_bytes = 0;
  // Dense tensors that would not fit and were left on the CPU. In a supported configuration this is
  // ZERO: the single-token CPU path aborts on a preshuffled fp8 shard anyway, so a tensor that
  // stayed behind is not a slower engine, it is one that will fault later. The caller refuses to
  // start rather than run degraded — see the check in tools/affinity.cpp.
  uint64_t cpu_fallbacks = 0;
};

class DenseGpu {
public:
  ~DenseGpu();

  // Brings up one rank per gfx12 device found, or `want_ranks` of them when that is smaller.
  // Returns false — having allocated nothing — when there is no supported device at all.
  bool init(uint64_t max_rows, uint64_t max_cols, std::string* err, size_t want_ranks = 0);

  // The hook the model binds to. Pass `ops()` to Model::set_dense_ops BEFORE Model::load.
  Model::DenseOps ops();

  // Shared expert: gate and up from x, SwiGLU, then down. gate/up are Row, down is Col. `defer`
  // leaves the chain in flight for shexp_collect(), which is the window the routed experts' CPU
  // share runs in.
  bool shexp(int32_t h_gate, int32_t h_up, int32_t h_down, const float* x, uint64_t hidden,
             uint64_t inter, float limit, float* out, bool defer = false);
  bool shexp_collect(float* out, uint64_t hidden);

  // Several registered matrices against the SAME x in one round trip — the compressor's four all
  // read `norm`, and four separate calls meant four host syncs for 10 MB of weights. All Row.
  bool mv_multi(const int32_t* h, const uint64_t* rows, uint64_t cols, uint32_t n, const float* x,
                float* const* y);

  // mv_multi with the attention sublayer's two first projections folded in. Both land in VRAM and
  // are picked up by the next attn_q / attn_kv; the compressor's outputs stay in flight until
  // comp_collect. Makes every check those two make, so a true return promises they find their
  // operand rather than reading stale scratch.
  bool attn_pre(int32_t qa, int32_t kv, uint64_t qr, uint32_t width, const int32_t* h,
                const uint64_t* rows, uint64_t cols, uint32_t n, float* const* y);
  bool comp_collect();

  int32_t reg_vec(const float* host, uint64_t n);

  // ---- the router gate, resident in VRAM -------------------------------------------------------
  // One gate per layer, small enough to replicate on both cards, and it buys back host DRAM traffic
  // that contended with the expert kernel every token. The decode matvec is row-split over the
  // replica: decode sends logits to the host for top-k either way, so each card produces half and
  // nothing is exchanged. `bias` is the aux-loss-free correction, kept in VRAM so the device top-k
  // never needs an H2D; null for hash-routed layers, which do not register at all.
  int32_t reg_router(const uint16_t* w, const float* bias, uint32_t n_expert, uint64_t n_embd);
  bool router(int32_t h, const float* x, uint32_t n_expert, uint64_t n_embd, float* logits);


  // ---- hyper-connections, the model's hidden state, resident in VRAM --------------------------
  // Replicated: 64 KiB of state and a one-block control kernel, so splitting would be pure
  // communication.
  bool hc_init(uint32_t n_embd, uint32_t n_hc, std::string* err);
  bool hc_ready() const { return hc_ok_; }
  // Seed all lanes from the token embedding at the start of a token.
  void hc_seed(const float* embed_host, uint32_t n_embd, uint32_t n_hc);
  // control + reduce + rms_norm, leaving `norm` on device on both cards and copied to `norm_host`
  // for the router top-k, the expert dispatch and the indexer — the one sync per sublayer this
  // design is left with. `defer` leaves that copy in flight for norm_collect().
  bool hc_pre(int32_t fn, int32_t scale_v, int32_t base_v, int32_t norm_w,
              uint32_t n_embd, uint32_t n_hc, uint32_t iters, float hc_eps, float rms_eps,
              float* norm_host, bool defer = false);
  // Idempotent: false and no work when nothing is outstanding.
  bool norm_collect(float* norm_host, uint32_t n_embd);
  // expand, or a plain residual add when this layer has no control.
  bool hc_post(bool has_control, uint32_t n_embd, uint32_t n_hc);
  // The routed experts' host contribution, added to both cards' blocks so they stay identical.
  bool block_add_host(const float* src, uint32_t n_embd);
  // Reduce both ranks' partial blocks into one value present on both cards, once per sublayer
  // after every Col matrix has deposited. Idempotent: depositing sets the flag, this clears it.
  bool block_reduce(uint32_t n_embd);
  // Read the collapsed lanes back for the head at the end of a token.
  void hc_read(float* dst, uint32_t n_embd, uint32_t n_hc);

  // ---- what the routed-expert dispatch needs to reach both cards ------------------------------
  size_t ranks() const { return st_.ranks; }
  int    device(size_t r) const;
  void*  stream(size_t r) const;
  float* batch_norm_dev(size_t r) const;
  float* batch_block_dev(size_t r) const;
  uint32_t batch_nb() const { return b_nb_; }

  // The router's selection as a CARD can read it, which is what a device-built dispatch needs and
  // the host path never did. `sel` and `wt` are [nb_tok][k] on rank `r`'s own memory — rank 0's own
  // block, and a mirror pushed over the peer link for every other rank. `ready` is the event that
  // marks the instant both exist; a reader on any stream must wait on it before the first load.
  //
  // Null pointers mean the mirror could not be built (one card, or no peer access), and the caller
  // keeps the host path. It does not mean "route from stale memory".
  struct RouteDev { const uint32_t* sel = nullptr; const float* wt = nullptr; void* ready = nullptr; };
  RouteDev route_dev(size_t r) const;

  // Take everything the compressor and indexer would otherwise grab on first use, at final size.
  // AFTER attn_init and batch_init (needs a_maxcomp_ and b_cap_), BEFORE StaticPlacement, which
  // sizes the expert slabs from hipMemGetInfo and keeps back only a fixed reserve.
  bool reserve_runtime(uint32_t n_layer, const uint32_t* ratios, uint32_t width, uint32_t idx_dim,
                       uint32_t idx_heads, std::string* err);

  // ---- attention state, resident in VRAM ------------------------------------------------------
  // Per layer and bounded: `sliding` raw rows in a ring (attention is over a SET, so ring order is
  // irrelevant once RoPE is applied) and `comp_rows[l]` compressed rows. comp_rows is per layer —
  // `kv_positions / ratio_for(l) + 2`, 0 where the layer does not compress — because handing every
  // layer the ratio-4 count costs 2.76 GiB a card at a 1M cache.
  //
  // Replicated across both cards: MQA, so a row is 2 KiB and the duplicate costs about one expert,
  // and each card then attends its own query heads with no exchange until wo_b.
  // Backs every layer's compressed cache far enough to hold `positions`, growing it if needed.
  //
  // WATERMARK-DRIVEN, FROM THE PLACEMENT THREAD, NEVER THE DISPATCH PATH. It calls into the driver
  // to map pages, and those pages come out of the expert slab — both of which are things a decode
  // block must never wait for. Decode advances ~4.4 positions a block, so calling this a chunk
  // ahead of the sequence leaves thousands of blocks of slack. Cheap and a no-op once the range is
  // already backed, so the caller can ask on every tick.
  bool kv_commit_positions(uint64_t positions, std::string* err);
  // Physical bytes the compressed cache currently holds, summed over cards.
  uint64_t kv_committed_bytes() const;

  // Starts a thread that keeps the cache backed `margin` positions ahead of wherever the sequence
  // has reached, and stops it. THIS is what keeps growth off the dispatch path: the alternative —
  // checking at a block boundary — still blocks the thread that issues, and hipMemCreate is a
  // driver call. Decode advances ~4.4 positions a block, so a 16384 margin fires roughly once every
  // 3700 blocks and always thousands of blocks before the position that needs it.
  void kv_autogrow_start(uint64_t margin);
  void kv_autogrow_stop();
  // Told to the grower by the dispatch, once a block. A relaxed store of one word.
  void kv_note_position(uint64_t pos) { kv_pos_.store(pos, std::memory_order_relaxed); }
  // The high-water mark the grower has actually backed. A caller about to write past it must stop
  // rather than fault: unmapped VA reads as a memory violation with no useful diagnostic.
  uint64_t kv_backed_positions() const { return kv_backed_.load(std::memory_order_acquire); }

  bool attn_init(uint32_t n_layer, uint32_t n_head, uint32_t width, uint32_t n_rot,
                 uint32_t sliding, const uint32_t* comp_rows, KvDtype dt,
                 uint64_t max_pos, uint64_t init_pos, std::string* err);
  KvDtype kv_dtype() const { return a_dt_; }
  // Safe to call before attn_init, and always is: the loader binds `attn_sinks` while walking the
  // layers, and attn_init needs the layer count that walk produces. Held on the host until then.
  void attn_set_sinks(uint32_t layer, const float* sinks, uint32_t n);
  // Finish the latent attn_kv left on device — QAT, f16 round, encode — straight into the cache
  // slot on both cards. No sync: attend() runs on the same streams and is ordered behind it.
  bool kv_commit(uint32_t layer, uint32_t pos, uint32_t width, uint32_t n_rot);
  bool attn_run(uint32_t layer, const float* q, const uint8_t* allowed, uint32_t pos,
                uint32_t n_comp, float scale, float* heads_out);

  // Tail of the attention sublayer, on the heads attention left in VRAM: inverse RoPE, the grouped
  // wo_a in one launch, then wo_b — the sublayer's one Col matrix, so this is where the attention
  // half deposits its partials.
  bool attn_out(int32_t h_woa, int32_t h_wob, uint32_t n_head, uint32_t width, uint32_t n_rot,
                uint32_t pos, RopeDerived rope, uint32_t n_groups, uint64_t rank, uint64_t hidden,
                float* block_out);
  bool attn_q(int32_t qa, int32_t qb, int32_t qnorm, const float* x, uint64_t hidden, uint64_t qr,
              uint32_t n_head, uint32_t width, uint32_t n_rot, uint32_t pos, RopeDerived rope,
              float eps, float* q_out, float* qrnorm_out);
  bool attn_kv(int32_t kv, int32_t kvnorm, const float* x, uint64_t hidden, uint32_t width,
               uint32_t n_rot, uint32_t pos, RopeDerived rope, float eps, float* kv_out);

  // ---- checkpointing the attention state ------------------------------------------------------
  //
  // The whole of what a sequence has accumulated on the cards, addressed as (part, layer, rows) so
  // that a caller can move it to storage without knowing the compress schedule. Two kinds, and the
  // difference is what a prefix cache is built around:
  //
  //   APPEND-ONLY   Comp, Idx8, IdxScale. Row g belongs to a fixed span of positions and is never
  //                 rewritten, so rows [a, b) are exactly what the tokens in [a*ratio, b*ratio)
  //                 produced. Two sequences sharing a token prefix share these rows bit for bit,
  //                 which is what makes them worth content-addressing and sharing on disk.
  //
  //   ROLLING       Raw, Hist, IdxHist. Rings indexed by position, overwritten every `sliding` (or
  //                 every `comp_hist(ratio)`) positions. They describe ONE position and cannot be
  //                 accumulated, so they are saved whole and belong to the checkpoint rather than
  //                 to a block.
  //
  // Raw covers the DSpark stages too — `attn_init` is called with n_layer + n_stage and the draft
  // commits its rows into a_raw[n_layer + stage], so saving every layer of it saves the draft's
  // conditioning as well and there is nothing separate to keep for the draft.
  enum class KvPart : uint8_t { Raw = 0, Comp, Idx8, IdxScale, Hist, IdxHist };

  // Row width and row count of `part` on `layer`. Both zero when this layer has no such buffer —
  // a ratio-0 layer compresses nothing, and only the ratio-4 layers carry an indexer — which is how
  // a caller enumerates the state without a copy of the ratio schedule.
  //
  // Hist and IdxHist report TWICE the ring depth: the pooled value ring and the score ring are one
  // allocation each and are laid out one after the other, values first.
  bool kv_part_geometry(KvPart part, uint32_t layer, uint64_t* row_bytes, uint64_t* rows) const;

  // Rows [row0, row0 + nrows) of `part` on ONE rank, to the host and back.
  //
  // Per rank, not rank 0 broadcast to all, because the cards do not hold the same bytes. Each keeps
  // a whole copy of the cache, but the hidden state feeding wkv has been through a cross-rank
  // reduce, and from layer 1 on the two copies differ in their last bits over nearly every region.
  // Both are synchronous on the rank's own stream: the copy has to be
  // ordered behind the kernels that produced the rows, and the caller's next launch behind it.
  bool kv_part_save(KvPart part, uint32_t layer, uint32_t rank, uint64_t row0, uint64_t nrows,
                    void* dst) const;
  bool kv_part_load(KvPart part, uint32_t layer, uint32_t rank, uint64_t row0, uint64_t nrows,
                    const void* src);

  // FNV-1a over rows [row0, row0 + nrows) as `rank` holds them, mixed into `h`.
  //
  // For proving a restored state is the state a straight prefill would have left, and for the one
  // question the replication claim above cannot answer from the outside: whether the two cards
  // really do hold the same bytes. Reads the device, so it is a diagnostic and not a fast path.
  bool kv_part_hash(KvPart part, uint32_t layer, uint64_t row0, uint64_t nrows, size_t rank,
                    uint64_t* h) const;

  // ---- batched prefill ------------------------------------------------------------------------
  //
  // A second arena, dim-major and `max_tokens` wide, in which a whole prompt chunk makes one pass
  // through the model. The single-token buffers above are untouched by it, so decode after a
  // batched prefill runs exactly the code it always did — the two paths share only the weights and
  // the KV cache. ~170 MiB a card at 256 tokens, so it too must precede StaticPlacement.
  bool batch_init(uint32_t max_tokens, uint32_t n_embd, uint32_t n_hc, uint32_t n_head,
                  uint32_t width, uint32_t max_mid, uint32_t vocab, std::string* err);
  Model::BatchOps batch_ops();
  uint32_t batch_cap() const { return b_cap_; }

  // Everything the DSpark draft allocates, after batch_init and before static placement. `tap_in`
  // is n_embd * the tap count — main_proj's input width. Until this runs every draft entry point
  // refuses, which is what a run with no draft wants.
  bool draft_init(uint32_t n_embd, uint32_t tap_in, uint32_t vocab, uint32_t markov_rank,
                  std::string* err);

  const DenseGpuStats& stats() const { return st_; }

private:
  // How a tensor is cut. See the header comment for which tensors get which and why.
  enum class Split : uint8_t { Row, Col, Rep };

  struct Shard {
    void*    data = nullptr;        // device, this rank's slice
    uint8_t* scales = nullptr;      // device, FP8 only
    uint64_t row0 = 0, nrows = 0;   // absolute row range this rank owns  (Col/Rep: 0, rows)
    uint64_t col0 = 0, ncols = 0;   // absolute col range this rank owns  (Row/Rep: 0, cols)
    // `data` is in WMMA fragment order, not row-major. Only the W8A8 GEMM can read it; every other
    // reader is a wrong answer that still decodes fluently, so they abort instead.
    bool     ps = false;
  };
  struct Entry {
    Shard sh[kMaxRanks];
    uint32_t quant = 0;
    uint64_t rows = 0, cols = 0;
    Split split = Split::Row;
  };
  static Split split_for(const char* name);
  // True for the fp8 tensors stored in WMMA fragment order.
  static bool ps_for(const char* name);
  // The row band rank `d` owns of a matrix whose caller wants `want` rows. Row: its own slice.
  // Rep: all of them, on both cards. Col does not partition the output and has no meaning here.
  static void row_band(const Shard& s, uint64_t want, uint64_t* r0, uint64_t* nr);
  // Row `local` of a shard, and the scale tile row that covers it.
  static const void* shard_rows(const Shard& s, uint32_t quant, uint64_t local);
  static const uint8_t* shard_scales(const Shard& s, uint64_t local);

  static int32_t reg_cb(void* ctx, const char* name, const void* host, const uint8_t* scales,
                        uint32_t quant, uint64_t rows, uint64_t cols);
  static void mv_cb(void* ctx, int32_t h, const float* x, uint64_t rows, uint64_t cols, float* y);
  static void mvg_cb(void* ctx, int32_t h, const float* x, uint32_t n_groups, uint64_t group_dim,
                     uint64_t rank, float* y);
  static bool shexp_cb(void* ctx, int32_t g, int32_t u, int32_t d, const float* x, uint64_t hidden,
                       uint64_t inter, float limit, float* out, bool defer);
  static bool shexp_collect_cb(void* ctx, float* out, uint64_t hidden);
  static bool mv_multi_cb(void* ctx, const int32_t* h, const uint64_t* rows, uint64_t cols,
                          uint32_t n, const float* x, float* const* y);
  static bool attn_pre_cb(void* ctx, int32_t qa, int32_t kv, uint64_t qr, uint32_t width,
                          const int32_t* h, const uint64_t* rows, uint64_t cols, uint32_t n,
                          float* const* y);
  static bool comp_collect_cb(void* ctx);
  static int32_t reg_vec_cb(void* ctx, const float* host, uint64_t n);
  static int32_t reg_router_cb(void* ctx, const uint16_t* w, const float* bias, uint32_t n_expert,
                               uint64_t n_embd);
  static bool router_cb(void* ctx, int32_t h, const float* x, uint32_t n_expert, uint64_t n_embd,
                        float* logits);
  static bool attn_q_cb(void* ctx, int32_t qa, int32_t qb, int32_t qn, const float* x,
                        uint64_t hidden, uint64_t qr, uint32_t n_head, uint32_t width,
                        uint32_t n_rot, uint32_t pos, RopeDerived rope, float eps, float* q_out,
                        float* qrnorm_out);
  static bool attn_kv_cb(void* ctx, int32_t kv, int32_t kvn, const float* x, uint64_t hidden,
                         uint32_t width, uint32_t n_rot, uint32_t pos, RopeDerived rope, float eps,
                         float* kv_out);
  static void kv_sinks_cb(void* ctx, uint32_t l, const float* s, uint32_t n);
  static bool kv_commit_cb(void* ctx, uint32_t l, uint32_t pos, uint32_t w, uint32_t nr);
  static bool attend_cb(void* ctx, uint32_t l, const float* q, const uint8_t* allowed,
                        uint32_t n_raw, uint32_t n_comp, float scale, float* heads);
  static bool hc_ready_cb(void* ctx);
  static void hc_seed_cb(void* ctx, const float* e, uint32_t n, uint32_t h);
  static bool hc_pre_cb(void* ctx, int32_t fn, int32_t sv, int32_t bv, int32_t nw,
                        uint32_t n, uint32_t h, uint32_t it, float he, float re, float* nh,
                        bool defer);
  static bool norm_collect_cb(void* ctx, float* nh, uint32_t n);
  static bool hc_post_cb(void* ctx, bool hasc, uint32_t n, uint32_t h);
  static bool block_add_cb(void* ctx, const float* src, uint32_t n);
  static bool block_reduce_cb(void* ctx, uint32_t n);
  static bool fused_ok_cb(void* ctx);
  static void hc_read_cb(void* ctx, float* dst, uint32_t n, uint32_t h);
  static bool attn_out_cb(void* ctx, int32_t woa, int32_t wob, uint32_t n_head, uint32_t width,
                          uint32_t n_rot, uint32_t pos, RopeDerived rope, uint32_t n_groups,
                          uint64_t rank, uint64_t hidden, float* block);

  int32_t reg(const char* name, const void* host, const uint8_t* scales, uint32_t quant,
              uint64_t rows, uint64_t cols);
  void mv(int32_t h, const float* x, uint64_t rows, uint64_t cols, float* y);
  void mvg(int32_t h, const float* x, uint32_t n_groups, uint64_t group_dim, uint64_t rank,
           float* y);
  // n floats at in[r] on rank r's stream, summed, left in out[r] on both. Enqueued, never waited on.
  void reduce(float* const in[kMaxRanks], float* const out[kMaxRanks], uint64_t n);

  // Issue the shared expert's two launches that shexp() held back, on rank `d`'s side stream,
  // gated behind whatever the main stream has enqueued. Idempotent: clears the pending pointers.
  void shexp_release(size_t d);

  // Device and pinned-host buffers the two batched router paths share: selection, weights and
  // per-token count. Taken on first use and never resized — sized from `b_cap_`, not the chunk in
  // hand, because the first chunk of a run can be 21 tokens and the next 1536. False if the
  // reservation failed, so the caller falls back rather than proceeding with half-null pointers.
  bool router_scratch(size_t rank, uint32_t n_expert, uint32_t k);
  // Allocate every non-zero rank's copy of the selection block and rank 0's publish event. Latched
  // once through `route_ready_`: 0 untried, 1 built, 2 refused. Three states rather than a bool so
  // a refusal is not retried per layer AND is distinguishable from "not yet" at the reader.
  bool route_mirror(uint32_t n_expert, uint32_t k);
  void route_publish();
  uint8_t route_ready_ = 0;
  uint64_t route_seq_ = 0;   // monotone; the ring slot is seq % kRouteSlots
  uint64_t tp_calls_ = 0;    // collectives issued, for the periodic tp_check

  struct Impl;
  Impl* impl_ = nullptr;
  std::vector<Entry> ent_;
  void upload_sink(uint32_t layer);
  // The allocations a KvPart spans on one rank, in row order: one for the append-only parts, two
  // for the rings, whose pooled values and scores are separate allocations of the same shape. The
  // ONLY place that maps a part to a buffer — geometry, save, load and hash are each this plus a
  // copy, so none of them can drift from the others. False when the layer has no such buffer.
  bool kv_part_bufs(KvPart part, uint32_t layer, size_t rank, uint64_t* row_bytes,
                    uint64_t* rows_each, void* bufs[2], uint32_t* n_buf) const;

  // Host copy of every sink vector handed to attn_set_sinks — see attn_set_sinks.
  std::vector<std::vector<float>> a_sink_host_;
  // attn_pre's handoff to attn_q / attn_kv. The batch writes into `d_y` at fixed offsets: wq_a's
  // `qr` rows at 0, wkv's `width` at `qr`, the compressor's after those. Consumed exactly once
  // each and cleared there, so a second chain in the same layer cannot read an overwritten buffer.
  bool     pre_q_ = false, pre_kv_ = false;
  uint64_t pre_qr_ = 0, pre_width_ = 0;
  // attn_kv's handoff to kv_commit: the learned norm and the RoPE tail ride on the commit kernel
  // instead of a dim3(1) launch of their own. Set only once attn_kv has established that this same
  // kv_commit will accept the shape, so a deferral cannot leave the latent normalised by nobody.
  bool         kvn_pending_ = false;
  const float* kvn_src_[kMaxRanks] = {};   // the RAW wkv output, per rank
  const float* kvn_w_[kMaxRanks] = {};     // the learned norm weight, per rank
  float        kvn_eps_ = 0.0f;
  uint32_t     kvn_pos_ = 0;
  RopeDerived  kvn_rope_{};
  // Destinations of the compressor half, outstanding until comp_collect. Those matrices are Row so
  // each rank's d_y layout differs; wq_a and wkv are Rep and sit at the same offsets on both.
  uint32_t pre_n_ = 0;
  float*   pre_y_[8] = {nullptr};
  uint64_t pre_r0_[kMaxRanks][8] = {};
  uint64_t pre_nr_[kMaxRanks][8] = {};
  uint64_t pre_total_[kMaxRanks] = {0};

  // Bytes outstanding in the pinned norm buffer, 0 when nothing is deferred.
  uint32_t norm_pending_ = 0, norm_cap_ = 0;
  // The stream's drain count when the deferred `norm` copy was enqueued, so `norm_collect` can
  // tell "already landed" from "must wait" without asking HIP.
  uint64_t norm_epoch_ = 0;
  KvDtype  a_dt_ = KvDtype::FP8R;
  // a_head_ is the model's head count; a_hpr_ is what one card owns. Rank r has heads
  // [r*a_hpr_, (r+1)*a_hpr_) and stores them at offset 0 of its own a_q_ / a_out_.
  uint32_t a_head_ = 0, a_hpr_ = 0, a_width_ = 0, a_rot_ = 0, a_sliding_ = 0, a_maxcomp_ = 0;
  // Compressed rows a layer actually holds. a_maxcomp_ is the maximum over these and is what the
  // per-token scratch is sized by — the admission mask plane, the indexer's score plane and its
  // key cache are all "widest layer a token will meet", not "this layer".
  std::vector<uint32_t> a_comp_rows_;
  uint32_t comp_rows_for(uint32_t l, uint64_t pos) const;
  // The context a_comp_rows_ was sized for, and how much of it is backed at load. The gap between
  // them is address space that costs nothing until the sequence reaches it — which is 962 expert
  // slots at the shipped --kv-size, worth 19% of decode. See gpu/vmem.h.
  uint64_t a_max_pos_ = 0, a_init_pos_ = 0;
  std::atomic<uint64_t> kv_pos_{0}, kv_backed_{0}, kv_margin_{0};
  std::atomic<bool> kv_grow_run_{false};
  std::thread kv_grow_;
  std::string kv_grow_err_;
  int64_t  a_idx_layer_ = -1;        // the layer whose admissions b_allow currently holds
  bool     a_idx_gather_ = false;    // ...and whether it also left the compacted list beside it
  uint32_t a_idx_topk_ = 0;          // that layer's index_topk, the list's largest possible length
  uint32_t i_tile_ = 1;              // tokens a pass through the indexer's score plane; see idx_grow
  bool     idx_want_qrn_ = false;    // attn_q keeps qr_norm in VRAM for the single-token indexer
  uint32_t b_norm_rows_ = 0;            // n_embd, remembered by hc_pre for the lazy norm fetch
  uint32_t a_ring_ = 0;                 // raw slots = sliding + kAttnBatch - 1; see attention_gpu.h
  bool     attn_ok_ = false;
  bool     hc_ok_ = false;
  // Length of an exchange block_reduce deferred to the next hc_post, or 0. Carries no buffer and
  // no ordering — d_block is where it always was — only the fact that the sum has not happened
  // yet. hc_post clears it on every path, including the ones that decline the fold.
  uint32_t ar_pending_ = 0;
  size_t   n_vec_ = 0, n_router_ = 0;
  DenseGpuStats st_;

  // ---- the batch arena -------------------------------------------------------------------------
  // Everything here is dim-major with stride b_nb_, which is b_live_ rounded up to kBatchTok. The
  // padding columns are computed and discarded; they are zeroed at begin() so they can never carry
  // a NaN into an arithmetic that would otherwise be finite.
  bool     batch_begin(uint32_t n_live, uint32_t pos0, const float* emb, uint32_t n_embd,
                       uint32_t n_hc);
  bool     batch_hc_pre(int32_t fn, int32_t sv, int32_t bv, int32_t nw, uint32_t n_embd,
                        uint32_t n_hc, uint32_t iters, float hc_eps, float rms_eps,
                        float* norm_host);
  bool     batch_attn_qkv(int32_t qa, int32_t qb, int32_t qnorm, int32_t kv, int32_t kvnorm,
                          uint64_t hidden, uint64_t qr, uint32_t n_head, uint32_t width,
                          uint32_t n_rot, RopeDerived rope, float eps, float* qrnorm_host);
  bool     batch_mv_host(int src, const int32_t* h, const uint64_t* rows, uint64_t cols, uint32_t n,
                         float* const* y);
  bool     batch_kv_commit(uint32_t layer, uint32_t b0, uint32_t n, uint32_t pos0, uint32_t width,
                           uint32_t n_rot);
  // The indexer's query projection straight into the device arena the indexer reads, skipping the
  // drain-and-copy-back batch_mv_host does. Must run before batch_indexer for the same chunk.
  bool     batch_indexer_q(int32_t h, uint64_t rows, uint64_t cols, uint32_t n_head, uint32_t dim);
  // `devp` is an Impl::Dev*, which this header cannot name — Impl is opaque here on purpose.
  bool     idx_grow(void* devp, uint32_t n, uint32_t n_head, uint32_t dim);
  // Both of the layer's compressors for a whole chunk: five GEMMs, four transposes, two pooling
  // kernels and two ring updates. See gpu/compressor_gpu.h.
  bool     batch_compress(const Model::CompressArgs& a);
  bool     compress_grow(void* devp, uint32_t layer, uint32_t ratio, uint32_t width,
                         uint32_t idx_width);
  bool     compress_reset();
  // The same two compressors and indexer for one token — decode must run the device
  // implementation because the device owns the cross-chunk window prefill left behind.
  bool     compress_one(const Model::CompressArgs& a);
  bool     indexer_one(uint32_t layer, int32_t h, uint64_t qr, uint32_t n_head, uint32_t dim,
                       uint32_t n_keys, uint32_t n_mask, uint32_t topk, uint32_t pos,
                       uint32_t n_rot, RopeDerived rope);
  // Tells attn_q to keep qr_norm in VRAM for indexer_one instead of copying it to the host.
  void     set_idx_want_qrn(bool v) { idx_want_qrn_ = v; a_idx_layer_ = -1; }
  // Scores every compressed row against every head, ReLU-weights them and takes the top `topk`,
  // leaving the admission mask on device where batch_attend already reads it. Must be followed by
  // batch_attend for the same tokens.
  bool     batch_indexer(uint32_t layer, uint32_t n, const uint32_t* n_comp, uint32_t n_keys,
                         uint32_t n_head, uint32_t dim, uint32_t topk, uint32_t pos0,
                         uint32_t n_rot, RopeDerived rope);
  // One launch for tokens [b0, b0+n), which start at absolute position pos0. `allowed` is a host
  // [n][mask_stride] plane or null; `n_comp` is a host array of n counts.
  bool     batch_attend(uint32_t layer, uint32_t b0, uint32_t n, uint32_t pos0,
                        const uint8_t* allowed, uint32_t mask_stride, const uint32_t* n_comp,
                        float scale);
  // `norm` on demand rather than every FFN sublayer: 13.4 MB a layer out of pageable memory at 818
  // tokens, which in the all-resident case nothing on the host reads.
  bool     batch_norm_fetch(float* norm_host);
  // The gate GEMM and the top-k in one call, with only the selection coming back: 48 KB a layer
  // instead of 13.4 MB of `norm` plus a host pass over 256 logits per token.
  bool     batch_router_topk(int32_t h, uint32_t n_expert, uint64_t n_embd, uint32_t k,
                             const float* bias, const uint8_t* resident, int norm_topk_prob,
                             float routed_scaling, uint32_t* sel, float* wt, uint32_t* nsel);
  // The same gate GEMM where the caller brought its own selection: hash-routed layers pick their
  // experts from a table and need only the weights, so `norm` is never touched on the host.
  bool     batch_router_hash(int32_t h, uint32_t n_expert, uint64_t n_embd, uint32_t k,
                             const uint32_t* sel, const uint32_t* nsel, int norm_topk_prob,
                             float routed_scaling, float* wt);
  bool     batch_attn_out(int32_t woa, int32_t wob, uint32_t n_head, uint32_t width,
                          uint32_t n_rot, RopeDerived rope, uint32_t n_groups, uint64_t rank,
                          uint64_t hidden);
  bool     batch_hc_post(bool has_control, uint32_t n_embd, uint32_t n_hc);
  bool     batch_router(int32_t h, uint32_t n_expert, uint64_t n_embd, float* logits);
  bool     batch_shexp(int32_t g, int32_t u, int32_t d, uint64_t hidden, uint64_t inter,
                       float limit);
  bool     batch_block_add(const float* src, uint32_t n_embd);
  bool     batch_block_reduce(uint32_t n_embd);
  bool     batch_hc_read(uint32_t b, float* dst, uint32_t n_embd, uint32_t n_hc);
  bool     batch_hc_tap(uint32_t slot, uint32_t n_taps, uint32_t b_lo, uint32_t n_keep,
                        uint32_t row0, uint32_t rows_cap, uint32_t n_embd, uint32_t n_hc);
  // The same head, sampled. The vocabulary is Row-split, so a nucleus over one card's band is not a
  // nucleus: the peer's half is gathered first and the draw happens over the whole distribution.
  // Scratch for the sampler's small transfers, sized once at the block width.
  void*    smp_q_ = nullptr;
  void*    smp_u_ = nullptr;
  void*    smp_out_ = nullptr;
  uint32_t smp_cap_ = 0;
  bool     batch_head_sample(int32_t h, uint64_t vocab, uint64_t n_embd, uint32_t n, const float* x,
                             const uint32_t* query, const float* u, float temperature, float top_p,
                             uint32_t top_k, float min_p, Model::PosDraw* out);
  bool     batch_head(int32_t h, uint64_t vocab, uint64_t n_embd, uint32_t n, const float* x,
                      uint32_t* ids);
  bool     draft_proj(int32_t h, int32_t norm_w, uint64_t rows, uint64_t cols, uint32_t n,
                      uint32_t tap_row0, float eps);
  bool     draft_kv(uint32_t layer, int32_t kv, int32_t kvnorm, uint32_t n, uint32_t pos0,
                    uint64_t hidden, uint32_t width, uint32_t n_rot, RopeDerived rope, float eps);
  bool     draft_attend(uint32_t layer, uint32_t n, uint32_t pos0, uint32_t win_lo, float scale);
  // The block's four lanes collapsed and normed into b_norm, where the vocabulary head reads its
  // activation. Call it, then pass x = nullptr to draft_head.
  bool     draft_collapse(int32_t fn, int32_t sv, int32_t bv, int32_t nw, uint32_t n_embd,
                          uint32_t n_hc, float hc_eps, float rms_eps);
  bool     draft_head(int32_t h_head, uint64_t vocab, uint64_t n_embd, uint32_t n, const float* x,
                      int32_t h_w1, int32_t h_w2, uint32_t rank, uint32_t first_id, uint32_t* ids);
  // One dim-major [n][b_nb_] buffer on rank r to a token-major host array, staged through pinned
  // memory at `off` rows into the arena. Enqueued, not waited on.
  void     batch_stage_out(size_t r, const float* src_dm, uint32_t n, uint32_t off);
  // The trailing aff_file/aff_line here and on batch_gemm_ent_q8 are AFF_PROFILE call-site
  // attribution, and must be on these two rather than only on batch_kernels.h's entry points:
  // both are shared wrappers, so a default argument there binds inside the wrapper and every
  // caller collapses onto one row.
  void     batch_gemm_ent(size_t r, int32_t h, const float* x, float* y, uint64_t rows,
                          uint64_t cols, const char* aff_file = __builtin_FILE(),
                          uint32_t aff_line = __builtin_LINE());
  // ---- W8A8: quantise an activation once, then run every GEMM that reads it -------------------
  // Separate because one quantise serves several GEMMs, and because a cache keyed on the source
  // pointer would be silently wrong — b_norm keeps its address and changes every layer.
  bool     quant_act(size_t r, const float* src, uint32_t dim);
  // ...and with the shared expert's SwiGLU as the source, so `b_y` is never written: that f32
  // intermediate was 6.3 MB out and back a layer a card at chunk 1536, for a value only this
  // quantisation reads.
  bool     quant_act_swiglu(size_t r, const float* g, const float* u, uint32_t dim, float limit);
  bool     q_slot(size_t r, uint32_t dim);      // the destination the three of them share
  // `accum` makes the epilogue add rather than store — the shared expert's down projection writing
  // straight into the sublayer block instead of through a scratch.
  void     batch_gemm_ent_q8(size_t r, int32_t h, float* y, uint64_t rows, uint64_t cols,
                             uint32_t x_off = 0, uint32_t w_row_off = 0, bool accum = false,
                             const char* aff_file = __builtin_FILE(),
                             uint32_t aff_line = __builtin_LINE());
  bool     q8_ok(int32_t h, uint64_t cols) const;

  uint32_t b_cap_ = 0, b_nb_ = 0, b_live_ = 0, b_pos0_ = 0, b_mid_ = 0;
  uint32_t b_vocab_ = 0;                        // 0 until batch_init
  uint32_t b_tapin_ = 0;                        // 0 until draft_init
};

// Per-card VRAM as the driver reports it. Declared here rather than in hip_common.h so a caller
// needs no HIP headers.
struct VramSample {
  double used_gib = 0.0, total_gib = 0.0;
};
std::vector<VramSample> vram_sample();

// The cards this build will actually use, in the same order and by the same filter as everything
// else in gpu/ — never 0..hipGetDeviceCount(), which enumerates the iGPU alongside them.
//
// Here for the same reason VramSample is: a caller that only wants to NAME the devices should not
// have to include the HIP headers to do it. `bus_id` is the PCI address in the driver's own form
// ("0000:03:00.0"), which is what the sysfs nodes hang off, and `cus` is compute units rather than
// the WGP count hipDeviceProp_t reports.
struct GpuDeviceDesc {
  std::string name, bus_id, arch;
  int cus = 0;
  double vram_total_gib = 0;
};
std::vector<GpuDeviceDesc> gpu_devices();

} // namespace aff
