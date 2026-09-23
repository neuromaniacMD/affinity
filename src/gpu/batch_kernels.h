// The same forward pass, for a CHUNK of tokens at a time.
//
// WHY THIS FILE EXISTS. Prefill as a loop over `forward_token` costs the same per token as decode
// does, and that equality is the whole problem: decode has one token and must read every weight for
// it, while prefill has the entire prompt available at once and was reading every weight per token
// anyway. The profile says the rest: the router bucket ran orders of magnitude above its own
// bandwidth cost, almost all of it stream synchronisations, and the routed-expert bucket spent its
// time streaming non-resident weights that the very next token would ask for again.
//
// Both of those divide by the chunk size, and neither needed a faster kernel.
//
// ---- the one design decision here ------------------------------------------------------------
//
// LANE = TOKEN. The single-token kernels in expert_kernel.hip put a wave on an output ROW and split
// that row's columns across the wave's lanes, which costs a cross-lane reduction per row and reads
// the weight once per token. Here each lane owns four whole dot products — four tokens, same row —
// so there is no reduction at all, the weight value is wave-uniform and is read once for 128
// tokens, and what streams through the lanes is the activation.
//
// That inverts the storage order: batched activations are [dim][nb], dim-major, so that the 32
// lanes of a wave read 32 (or 4x32) CONSECUTIVE floats. Every buffer crossing between these
// kernels is in that layout; `batch_transpose_hip` converts at the two edges where the host wants
// one token's vector contiguous (the routed-expert dispatch and the compressor).
//
// Sizing, from the balance point rather than taste: the card's FMA rate against its bandwidth puts
// the roofline knee at a flop-per-byte this shape reaches at a fairly small token count. The chunk
// tile is deliberately well past that knee — compute-side — which leaves read bandwidth for the
// routed experts streaming from host RAM at the same time, and those are prefill's real bottleneck.

#pragma once

#include "engine/ops.h"

#include <cstdint>

namespace aff {

// Tokens per pass: 32 lanes x 4 tokens per lane. A chunk larger than this costs one extra pass over
// the weights per 128 tokens, which is the ratio the whole file is about — so chunk sizes should be
// multiples of 128 to avoid paying for a pass that is mostly padding.
constexpr uint32_t kBatchTok = 128;

// The smallest batch the W8A8 GEMM has a kernel for -- wmma::kN, the NT=1 instantiation. It is
// engine/ops.h's kSpecTok, which the engine core sizes its speculative runs against without being
// able to include this header.
constexpr uint32_t kNarrowTok = kSpecTok;

// THE batch stride for `n` live tokens. Every site that sizes a batch buffer and every site that
// strides one must call this and no other expression: the arena's rounding and the per-chunk
// rounding disagreeing by one tile is not a slow path, it is `batch_begin` refusing the chunk and
// the caller silently looping single tokens instead -- which reads as a real measurement, because
// the answer stays correct and only the speed says anything.
//
// Below one 128-token pass, rounding up to 128 is the entire cost rather than a rounding: a DSpark
// verify pass is six tokens, and at a 128-wide stride every GEMM in it computes 128 columns of
// which 122 are padding.
inline uint32_t batch_round(uint32_t n) {
  return n <= kNarrowTok ? kNarrowTok : ((n + kBatchTok - 1) / kBatchTok) * kBatchTok;
}

// Several weight matrices against ONE activation, in ONE launch.
//
// A small weight read gets a fraction of the streaming rate and the same bytes fused get all of it:
// `bytes/peak + a fixed launch cost` fits every shape, so a small GEMM spends most of itself on
// ramp. The batched dense path issues hundreds of launches a rank a speculative cycle.
//
// Every job shares `cols`, the activation and nb; each has its own weight, scales, row count and
// destination, and the destinations must be CONSECUTIVE bands of one buffer — `jobs[0].y` is the
// base and each job's band follows the previous one's rows, which is how the split-K partial plane
// stays a single [z][total][nb] block that one reduce covers. Narrow (nb <= 16) only. Returns false
// if the group cannot be served, and the caller must then issue the jobs one at a time.
// At most this many matrices in one grouped launch. Eight covers every call site.
constexpr uint32_t kGemmJobsMax = 8;
struct BatchGemmJob {
  const void* w;
  const uint8_t* s;                     // fp8 tile scales, null for bf16
  float* y;                             // [rows][nb], dim-major
  uint32_t rows;
  // For a group that slices one weight and one activation rather than holding several: the column
  // of X this job starts at, and the row of W (and of its scale plane) it starts at.
  uint32_t x_off;
  uint32_t w_row_off;
};
// The trailing `aff_file`/`aff_line` on this and the three below are the AFF_PROFILE call-site
// attribution and nothing else. A default argument is evaluated AT THE CALL SITE, so declaring them
// here gives every caller in the engine its own row in the profile table without one of them being
// edited — and without a macro, which would have hidden the real signature from every reader.
// Ignored entirely unless AFF_PROFILE is set. See `GemmTally` in batch_kernels.hip.
bool batch_gemm_multi_hip(uint32_t quant, const BatchGemmJob* jobs, uint32_t n, const void* X,
                          uint32_t cols, uint32_t nb, bool x_bf16, void* stream,
                          const char* aff_file = __builtin_FILE(),
                          uint32_t aff_line = __builtin_LINE());

// The same for the W8A8 path, whose activation is already E4M3 with a per-(token, 128-col) scale.
// `preshuffled` is one flag for the whole group, so every job's weight must be in the same layout —
// which is what the caller has anyway, since a group is several shards of the same registration.
bool batch_gemm_fp8_multi_hip(const BatchGemmJob* jobs, uint32_t n, const uint8_t* X,
                              const float* Sx, uint32_t cols, uint32_t nb, uint32_t x_ld,
                              uint32_t sx_ld, void* stream, bool preshuffled = false,
                              const char* aff_file = __builtin_FILE(),
                              uint32_t aff_line = __builtin_LINE());

// Y[row][b] = sum_c W[row][c] * X[c][b], for b < nb. W is row-major in the container's own format;
// `quant` is AFF_BF16, AFF_FP8_E4M3 or AFF_F32. `S` is the FP8 tile-scale plane and is ignored
// otherwise. X and Y are dim-major with stride `nb`.
void batch_gemm_hip(uint32_t quant, const void* W, const uint8_t* S, const float* X, float* Y,
                    uint32_t rows, uint32_t cols, uint32_t nb, void* stream,
                    const char* aff_file = __builtin_FILE(),
                    uint32_t aff_line = __builtin_LINE());

// The same, with a bf16 activation. Not a precision trade: the f32 form has always truncated X to
// bf16 while staging it into LDS for the matrix core, so this hands the WMMA the identical operand
// and merely halves the bytes read to get there. False for an AFF_F32 weight, which takes a
// different kernel entirely.
bool batch_gemm_bf16x_hip(uint32_t quant, const void* W, const uint8_t* S, const uint16_t* X,
                          float* Y, uint32_t rows, uint32_t cols, uint32_t nb, void* stream,
                          const char* aff_file = __builtin_FILE(),
                          uint32_t aff_line = __builtin_LINE());

// Quantise a dim-major f32 activation into the token-major E4M3 + per-(token, 128-column) scale
// that `batch_gemm_fp8_hip` consumes. `dim % 128 == 0`. This is a transpose as well as a quantise,
// so it goes through LDS; it exists as its own launch until it can be folded into the epilogue of
// whichever kernel produced `x`.
void batch_quant_fp8_hip(const float* x, uint8_t* out, float* scales, uint32_t dim, uint32_t nb,
                         uint32_t x_off, void* stream = nullptr);
// ...and the same with the dense SwiGLU as its source, which is `batch_swiglu_hip` followed by
// `batch_quant_fp8_hip` with the f32 intermediate deleted rather than written and read back. Same
// expression in the same order, so it is bit-identical to the pair. This is the one of prefill's
// six activations where the producer is elementwise over dim-major arenas and therefore folds; the
// rest need a reduction across a token and are rewrites into this kernel's tile shape, not folds.
void batch_swiglu_quant_fp8_hip(const float* g, const float* u, uint8_t* out, float* scales,
                                uint32_t dim, uint32_t nb, float limit, void* stream = nullptr);

// Permute an E4M3 weight into the order the WMMA fragment wants, once at load. This is
// vllm-radiance's shuffle_weight(w, (16,16)) and it is what `batch_gemm_fp8_hip` expects `W` to be
// in. `dst` is the same size as `src`; rows and cols must both be multiples of 16.
void preshuffle_fp8_hip(const uint8_t* src, uint8_t* dst, uint32_t rows, uint32_t cols,
                        void* stream);

// W8A8. Both operands are E4M3 and NOTHING is converted in the loop -- the weights come straight
// from the container and the activation straight from the pipeline.
//
//   W   PRESHUFFLED by preshuffle_fp8_hip, with `Sw` the UE8M0 plane, one byte per 128x128 tile
//       of the ORIGINAL row-major shape.
//   X   [nb][cols]   E4M3, TOKEN-major (not dim-major like the f32 buffers), with `Sx` one f32 per
//                    (token, 128-column block). Token-major is what makes staging a pure byte copy.
//   Y   [rows][nb]   f32, dim-major. The GEMM cannot compute its own output scale without a
//                    reduction across every M block, so it does not try; whichever elementwise op
//                    consumes Y next already reads a whole token vector and re-quantises for free.
//
// Requires cols % 128 == 0.
// `preshuffled` says which layout `W` is in. True is the vllm order that `preshuffle_fp8_hip`
// produces, read straight to registers with no LDS. False is the ORDINARY row-major weight, staged
// through the same `stage8` as the activation — possible only because both operands are [row][k]
// byte planes once the activation is fp8 and token-major, and the reason it matters is that the
// decode matvecs read the same bytes row-major and there is no VRAM for a second copy.
// `x_ld`/`sx_ld` are the row strides of `X` and `Sx`; 0 means "cols" and "ceil(cols/128)", i.e. the
// activation is exactly this GEMM's operand. They differ when one quantised buffer serves a whole
// sublayer and a grouped GEMM reads a `cols`-wide window of it at `x_off`.
//
// `nb <= 16` — the speculative-decode batch, DSPark's gamma+1 — takes a separate instantiation with
// one 16-wide token tile and a narrower M tile. `mw` forces the wave count of that instantiation
// (1, 2, 4 or 8) and exists for `bench/spec_batch.hip`; 0 lets `fp8_narrow_mw` pick and is what the
// engine passes. It is ignored for nb > 16.
void batch_gemm_fp8_hip(const uint8_t* W, const uint8_t* Sw, const uint8_t* X, const float* Sx,
                        float* Y, uint32_t rows, uint32_t cols, uint32_t nb, uint32_t x_off,
                        uint32_t w_row_off, void* stream, bool preshuffled = true,
                        uint32_t x_ld = 0, uint32_t sx_ld = 0, bool accum = false,
                        uint32_t mw = 0, const char* aff_file = __builtin_FILE(),
                        uint32_t aff_line = __builtin_LINE());
uint32_t fp8_narrow_mw(uint32_t rows);

// Grouped form, for wo_a: Y[g*rank + r][b] = sum_c W[(g*rank+r)*group_dim + c] * X[g*group_dim+c][b].
// Requires rank % 128 == 0 for FP8 so each group starts on a scale-tile row.
void batch_gemm_grouped_hip(uint32_t quant, const void* W, const uint8_t* S, const void* X,
                            float* Y, uint32_t n_groups, uint32_t group_dim, uint32_t rank,
                            uint32_t nb, void* stream, bool x_bf16 = false,
                            const char* aff_file = __builtin_FILE(),
                            uint32_t aff_line = __builtin_LINE());

// ---- the elementwise half ---------------------------------------------------------------------
// All of these are one thread per token: dim-major storage puts consecutive tokens in consecutive
// addresses, so a thread walking `dim` for its own token reads a fully coalesced column each step
// and the per-token reductions need no cross-lane work at all.

// out[i][b] = x[i][b] * rsqrt(mean(x[.][b]^2) + eps) * w[i]. Exactly one of `w32`/`wbf16` may be
// non-null; both null means no weight.
void batch_rms_norm_hip(const float* x, const float* w32, const uint16_t* wbf16, float* out,
                        uint32_t n, uint32_t nb, float eps, void* stream);

// The same at a speculative block's width — one block a TOKEN rather than kBlkTok tokens a block,
// which is 16 workgroups instead of two — and emitting the E4M3 its consumer would otherwise need
// a whole launch to produce. Bit-identical. False if the shape does not fit, and then the caller
// must issue the pair.
bool batch_rms_norm_q8_hip(const float* x, const float* w32, const uint16_t* wbf16, float* out,
                           uint8_t* q8, float* qs, uint32_t n, uint32_t nb, float eps,
                           void* stream);

// Per-head RMS (no weight), the RoPE tail, and the dim-major -> token-major transpose the query
// path needs anyway, in two passes over `x` instead of five. `inv` is [n_head][nb] scratch for the
// per-head scale; `dst` is [nb][n_head*head_dim]. `x` is left UNSCALED — its only reader was this
// transpose. Position is pos0 + b, exact for prefill because a chunk is contiguous in position.
//
// False if the model's head_dim or n_rot is not a multiple of 64, which is the alignment the tile
// depends on. `dst` is bf16: its only consumer is the attention, which quantises Q to E4M3.
// `qrms` false (V4.1) skips the per-head RMS: `inv` is filled with 1.0 and only rope + transpose run.
bool batch_head_rms_rope_tr_hip(const float* x, uint16_t* dst, float* inv, uint32_t n_head,
                                uint32_t head_dim, uint32_t n_rot, uint32_t pos0, RopeDerived rope,
                                float eps, uint32_t nb, void* stream, bool qrms = true);

// Learned RMS then the RoPE tail — the KV path.
void batch_rms_norm_rope_hip(const float* x, const float* w, float* out, uint32_t n, uint32_t n_rot,
                             uint32_t pos0, RopeDerived rope, float eps, uint32_t nb, void* stream);

// The same in place on a bf16 buffer — the attention output heads.
void batch_rope_tail_bf16_hip(uint16_t* x, uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
                              uint32_t pos0, RopeDerived rope, float sin_sign, uint32_t nb,
                              void* stream);

void batch_swiglu_hip(const float* g, const float* u, float* out, uint32_t n, uint32_t nb,
                      float limit, void* stream);

// dst[b][n] <- src[n][b], or the reverse. The only two places a batched buffer changes layout: the
// host expert dispatch and the compressor both want one token's vector contiguous.
// Up to four dim-major -> token-major transposes in one launch. The compressor issues exactly four
// back to back, and each as its own dispatch pays a full launch gap before the kernel even starts.
// Same arithmetic as batch_transpose_hip with to_token_major=1.
void batch_transpose_multi_hip(const float* const* src, float* const* dst, const uint32_t* n,
                               uint32_t n_jobs, uint32_t nb, void* stream);

void batch_transpose_hip(const float* src, float* dst, uint32_t n, uint32_t nb, int to_token_major,
                         void* stream);
// The same, rounding to bf16 on the way out.
void batch_transpose_bf16_hip(const float* src, uint16_t* dst, uint32_t n, uint32_t nb,
                              int to_token_major, void* stream);

// The same, times a constant. The indexer's per-token head weights carry a 1/sqrt(dim*heads) that
// the host used to apply in a loop over 64 floats a token; here it costs nothing at all.
void batch_scale_transpose_hip(const float* src, float* dst, uint32_t n, uint32_t nb,
                               int to_token_major, float scale, void* stream);

// dst[i*cap + k] <- src[i]: one decoded token's activation written into column `k` of a dim-major
// [n][cap] staging block, which is the layout batch_gemm_hip reads. This is the whole per-token
// cost of the deferred compressor -- 16 KiB of stores in place of a 20 MiB weight read.
void batch_stage_col_hip(const float* src, float* dst, uint32_t n, uint32_t cap, uint32_t k,
                         void* stream);

// ---- hyper-connections ------------------------------------------------------------------------
// Same arithmetic as hc_fused_kernel, one block per token instead of one block for the model.

void batch_hc_fused_hip(const uint16_t* hc, const float* mix, const float* scale, const float* base,
                        const float* nw, float* pre, float* post, float* comb, float* cur,
                        float* norm, uint32_t n_embd, uint32_t n_hc, uint32_t nb, uint32_t iters,
                        float hc_eps, float rms_eps, int has_control, void* stream, const float* pre_in);

// Dim chunks in the narrow form below, and therefore both its block count and the row count of the
// partial plane the caller has to own. Here rather than beside the kernels because the arena that
// holds `part` is sized in dense_gpu.hip.
constexpr uint32_t kHcnChunks = 32;
// Threads a block in that family; the norm kernel's block owns this many CONTIGUOUS dims, which is
// what lets it emit the E4M3 scale for the 128-column blocks it covers.
constexpr uint32_t kHcnThreads = 256;

// The same op at a speculative block's width, where the kernel above has nb*64 threads to cover the
// machine with and that is 1024. Parallel over `n_embd` instead, in four launches around a grid-wide
// reduction, with `part` a [kHcnChunks][nb] scratch plane. `nb` MUST be kNarrowTok.
//
// NOT bit-identical to batch_hc_fused_hip: the reduction is a tree rather than 64 serial stripes.
void batch_hc_fused_narrow_hip(const uint16_t* hc, const float* mix, const float* scale,
                               const float* base, const float* nw, float* pre, float* post,
                               float* comb, float* cur, float* norm, float* part, uint8_t* q8,
                               float* qs, uint32_t n_embd, uint32_t n_hc, uint32_t nb,
                               uint32_t iters, float hc_eps, float rms_eps, int has_control,
                               void* stream, const float* pre_in);

// The same family at the END of the stream: gate the four lanes, sum them into one vector, rms_norm
// it against `nw`. No post gate, no Sinkhorn, and `mix` is [n_hc][nb] rather than the sublayer's
// [2*n_hc + n_hc^2][nb] — the head expands nothing. `nb` MUST be kNarrowTok; `part` is the same
// two-plane [2][kHcnChunks][nb] scratch batch_hc_fused_narrow_hip wants.
// `norm_tm`, when non-null, is the same result written token-major — `nb` contiguous n_embd-float
// vectors, which is the layout matvec_multix_hip reads and the dim-major plane is not.
void batch_hc_head_narrow_hip(const uint16_t* hc, const float* mix, const float* scale,
                              const float* base, const float* nw, float* cur, float* norm,
                              float* norm_tm, float* part, uint32_t n_embd, uint32_t n_hc,
                              uint32_t nb, float hc_eps, float rms_eps, void* stream, const float* pre_in);

// out_hc[dst][d][b] = block[d][b]*post[dst][b] + sum_src comb[dst + src*n_hc][b] * hc[src][d][b]
//
// `zero_block` leaves `block` zeroed on the way out. Both of these visit every element of the arena
// exactly once, so the sublayer accumulator's memset is theirs to absorb — see the kernel. The flag
// exists rather than being unconditional so the memset can be restored as a control; the two arms
// are bit-identical.
void batch_hc_expand_hip(uint16_t* out_hc, float* block, const uint16_t* hc,
                         const float* post,
                         const float* comb, uint32_t n_embd, uint32_t n_hc, uint32_t nb,
                         bool zero_block, void* stream);
void batch_hc_add_hip(float* dst, const float* src, uint64_t n, void* stream);
// The same into the bf16 lanes, for a layer with no hyper-connection control.
void batch_hc_add_bf16_hip(uint16_t* dst, float* src, uint64_t n, bool zero_src, void* stream);

// Every lane of the 4-wide hidden state starts as the token embedding. `emb` arrives token-major
// (the host dequantised it out of the container's bf16 table) and is transposed into place.
void batch_hc_seed_hip(const float* emb, uint16_t* hc, uint32_t n_embd, uint32_t n_hc, uint32_t nb,
                       void* stream);

// The hc lanes averaged — `h.mean(dim=2)`, which is what DSpark conditions its draft on — written
// TOKEN-MAJOR straight into the tap plane the draft's projection reads, so the value never leaves
// VRAM. `tap` is [rows_cap][row_stride] and this call fills one layer's `n_embd`-wide column at
// `slot_off`, for the `n_keep` tokens starting at `b_lo`, into rows `row0` upward.
void batch_hc_tap_hip(const uint16_t* hc, float* tap, uint32_t n_embd, uint32_t n_hc, uint32_t nb,
                      uint32_t b_lo, uint32_t n_keep, uint32_t row0, uint32_t rows_cap,
                      uint32_t row_stride, uint32_t slot_off, void* stream);

// Per-token argmax over the head's [rows][nb] plane, one block a token — dim-major when a GEMM
// wrote it, token-major [nb][rows] when the multi-x matvec did. `out` is 2*nb floats: the band's
// best value in the first nb and `row0 +` its row in the second, so a caller with the matrix split
// across ranks compares two of these and never moves a logit.
//
// The row index goes back as a float on purpose — the vocabulary is 129280 and every integer below
// 2^24 is exact, so this stays one buffer and one copy instead of two of each.
void batch_argmax_hip(const float* y, float* out, uint32_t rows, uint32_t nb, uint32_t live,
                      uint32_t row0, void* stream, int y_token_major = 0);

// ONE step of DSpark's Markov correction: the bias, the add and the argmax, in a single launch.
//
//   logits[v] += dot(w2[v], w1[token])   for v in this card's band,   then argmax over the band.
//
// It is fused because it cannot be pipelined: step i's argmax IS step i+1's token, so the five
// steps of a block are serial and every launch between them is on the critical path. Splitting it
// into a gather, a GEMM and an argmax would be three launches and an 8.3 MB scratch for a value
// nothing reads — only the winner leaves.
//
// The chain is serial on the DEVICE, which is not the same as serial through the host, and it used
// to be both. `ids` is the chain itself, in device memory: `ids[0]` is the block's first token,
// step `col` reads `ids[col]` and the combine writes `ids[col+1]`. Nothing is a launch argument, so
// the host issues all five steps without ever learning what any of them chose.
//
// `w1` and `w2` are both [vocab][rank] bf16 and REPLICATED, so `row0`/`rows` slice w2 to whatever
// band the vocabulary head produced. Only column `col` of `logits` is read: dim-major [rows][nb]
// when a GEMM wrote it, token-major [nb][rows] when the multi-x matvec did — say which with
// `logits_token_major`. `best` is one uint64 of device scratch, owned by the caller and zeroed here;
// it holds this card's band's packed (value, ABSOLUTE row) maximum.
void markov_step_hip(const uint16_t* w1, const uint16_t* w2, const uint32_t* ids,
                     const float* logits, uint64_t* best, uint32_t rows, uint32_t row0,
                     uint32_t rank, uint32_t nb, uint32_t col, void* stream,
                     int logits_token_major = 0);

// The cross-card half of one step: `own` is this card's packed key and `peers` the other cards',
// staged contiguously; the largest wins, which is the same "higher value, then lower id" the host
// used to apply after a readback per card. Writes the winner into `ids[col + 1]`, where the next
// step will read it. npeer == 0 is the identity.
void markov_combine_hip(const uint64_t* own, const uint64_t* peers, uint32_t npeer, uint32_t* ids,
                        uint32_t col, void* stream);

// The same exchange for the TARGET's head: `own` and each of the `npeer` planes in `peers` are
// [2][nb] argmax planes (value at [t], absolute id at [nb + t]) and the winner per position lands
// in `ids[t]`. A card with no band of the vocabulary contributes no plane.
void batch_head_combine_hip(const float* own, const float* peers, uint32_t npeer, uint32_t* ids,
                            uint32_t n, uint32_t nb, void* stream);

// Gather one token's column out of a dim-major buffer into a contiguous vector, so the per-token
// attention kernel can consume it unchanged.
void batch_column_bf16_hip(const uint16_t* src, float* dst, uint32_t n, uint32_t nb, uint32_t b,
                           void* stream);

// The router's top-k, on the device where the logits already are. One block per token.
//
// This is route_topk() and Model::force_resident()'s masked re-rank, transcribed. It has to be
// transcribed exactly rather than approximated: the selection is a DISCRETE choice over 256
// near-tied scores, so a different rounding does not perturb a number, it swaps a whole expert and
// with it 9 MB of weights. The two rules that are easy to lose are the tie-break (lower expert id
// wins) and that the aux-loss-free `bias` is added AFTER sqrtsoftplus and affects only WHICH
// experts win, never how much they contribute.
//
// `logits` is [n_expert][nb] dim-major, as the gate GEMM leaves it. `resident`, when non-null, is
// an [n_expert] byte mask and a zero excludes that expert from the selection entirely — the
// AFF_FORCE_RESIDENT measurement mode, and the only place the two implementations may diverge: if
// fewer than k experts are resident this emits what it found, where the host walks forward and
// duplicates. A layer that thin is not what the flag is for.
void router_topk_hip(const float* logits, const float* bias, const uint8_t* resident,
                     uint32_t* sel, float* wt, uint32_t* nsel, uint32_t n_expert, uint32_t nb,
                     uint32_t live, uint32_t k, int norm_topk_prob, float routed_scaling,
                     void* stream);

// The WEIGHTING half of the router, for a selection somebody else already made.
//
// Hash-routed layers pick their experts from a table keyed on the token id rather than from the
// logits — but they still weight each contribution by the router's own probability, and that is a
// `router_w[e] . norm` dot per selected expert. On the host that is `live * k` of them, scalar bf16,
// single-threaded, with the card idle throughout — the single largest stall in prefill. The gate
// GEMM that feeds router_topk_kernel already produces every logit for the whole chunk in a fraction
// of the time, so this just gathers the chosen ones out of it.
//
// `logits` is [n_expert][nb] dim-major, as the gate GEMM leaves it. `sel`/`nsel` are the caller's
// selection, on device. Same sqrtsoftplus, same fp16 min-normal floor, same divide-then-scale and
// the same order as the host loop it replaces — only the dot product itself is computed
// differently, and it is computed by the same GEMM the non-hash layers have used all along.
void router_hash_wt_hip(const float* logits, const uint32_t* sel, const uint32_t* nsel, float* wt,
                        uint32_t nb, uint32_t live, uint32_t k, int norm_topk_prob,
                        float routed_scaling, void* stream);

} // namespace aff
