// GPU expert kernel — the same fused dequant+GEMV as the CPU path, on gfx1201.
//
// The hybrid executor's other half: experts resident in VRAM are evaluated here, experts resident
// in RAM by the AVX-512 path in src/engine/expert_kernel.cpp. Both consume the identical bytes
// straight out of the mmap'd container — no repack, no transcode.
//
// Per matrix, three planes:
//
//   data     10 bits per QUAD of weights, packed contiguously  (2.5 bits/weight)
//   scales   6 bits per 16-weight block: [variant:1][rot:1][sign:1][scale:3]
//   rscales  one bf16 per row
//
//   w[4i..4i+3]  =  rscale[row] * 2^scale * (+-) rot(LUT[variant][code])
//
// AFF_Q2P875, the format before the requantise, spends the same bits as 5-bit PAIR codes over a
// 32-entry 2-D codebook with a 4-bit integer scale — same plane sizes, different decode, which is
// what `expert_wire_format` is for.
//
// The binding constraint is the DECODE budget, not bandwidth: the GPU allows about an order of
// magnitude fewer ops per weight than the CPU before decode stops being free. Hence a 2-D LUT in LDS
// rather than anything computed.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "format/aff_format.h"

namespace aff {

// ---- device weight layout -------------------------------------------------------------------
//
// The container stores each plane ROW-MAJOR, which suits the CPU path and anything walking one row
// at a time, and is the worst possible layout for a GEMM walking 32 rows at a time: a wave's lanes
// ask for 32 separate short spans a row-stride apart, every one a partial cache line, which reads at
// a fraction of the streaming rate. Changing nothing but the addresses puts it back on the ceiling.
//
// So the device slab is colgroup-major: `[c0/64][row][20 B]` for the codes and `[c0/64][row][3 B]`
// for the fields. A column group is 64 columns = 4 blocks = exactly 20 and exactly 3 whole bytes,
// which makes the reorder a pure permutation of spans with no bit surgery. Same size, same bytes,
// and the container on disk does not change — the permute happens once per expert at load.
//
// EVERY device reader of those planes must agree. The CPU path reads the mmap'd container directly
// and is unaffected. Defaults to true; only a test or bench building its own row-major planes
// turns it off.
bool expert_planes_are_colgroup_major();

// Force the layout, for tests and tools that build their own planes. Call it BEFORE any expert
// kernel or upload: nothing re-reads a plane once it is on the card.
void expert_set_plane_layout_colgroup(bool on);

// Which decode the packed codes are, as an `expert_kernel.hip` kFmt* value:
//
//   0  kFmtPair16   5-bit codes over a 32-entry 2-D codebook, 4 variants x 16 integer block scales
//                   folded into a 4 KB LUT. The format the container shipped before the requantise.
//   4  kFmtQuad4    10-bit codes over a 1024-entry 4-D codebook, 4 variants, 16 KB LUT.
//   5  kFmtQuad4S   kFmtQuad4 plus a per-block sign bit. 16 KB, and slower than pair16 — the table
//                   costs a block per WGP, which is more than the decode saves.
//   6  kFmtQuad2S   TWO variants (8 KB, so occupancy holds) plus a sign bit and a 2-byte rotate.
//                   AFF_Q2P875_Q4, what ships: faster and lower error at the same 2.875 bpw.
//   8  kFmtQuad8    kFmtQuad4 with 8 variants, 32 KB LUT. Clearly slower; a control.
//
// UNLIKE the plane layout this is not a preference — it is a property of the container, and reading
// one as the other is in-bounds, right-sized and completely wrong. The model loader calls the setter
// from the layer's quant id; a bench with no loader must call it for itself, via
// `expert_wire_format_for_quant`, BEFORE it builds a plane.
uint32_t expert_wire_format();
void     expert_set_wire_format(uint32_t fmt);
// The wire format a container's quant id implies, in one place so the loader and the checkers
// cannot pick different answers. Takes a AffQuant as uint32_t (what AffMatrixDesc::quant holds).
uint32_t expert_wire_format_for_quant(uint32_t aff_quant);

// Permute ONE matrix's code and field planes, row-major -> colgroup-major. src and dst must not
// overlap. This is the single definition of the layout; everything that writes a device plane goes
// through it, so the loader and the checkers cannot drift apart.
void expert_transpose_plane_host(const uint8_t* src_codes, const uint8_t* src_fields,
                                 uint8_t* dst_codes, uint8_t* dst_fields,
                                 uint32_t rows, uint32_t cols);

// Permute one expert's three matrices in place. `buf` is the expert's `stride` bytes as they came
// off disk; `scratch` is reused across calls.
void expert_transpose_planes_host(uint8_t* buf, const AffLayerDesc& L, std::vector<uint8_t>& scratch);

// ---- tensor-parallel expert sharding -----------------------------------------------------------
//
// One expert split across `parts` cards so both do half the work regardless of which experts a
// token picks. Standard FFN tensor parallelism:
//
//   gate, up   rows = inter, cols = hidden   split by ROWS  -> part p holds rows [p*inter/P, ..)
//   down       rows = hidden, cols = inter   split by COLS  -> part p holds cols [p*inter/P, ..)
//
// so part p produces its slice of the intermediate, feeds it to its slice of `down`, and emits a
// PARTIAL sum over the full hidden dimension. The MoE all-reduce already sums exactly that, which
// is why this needs no new collective.
//
// The colgroup-major plane — `plane[(g * rows + row) * K]`, kEK=64 columns a group — makes both
// slices whole-byte spans and each shard byte-for-byte identical to a natively-transposed matrix
// of the halved shape. So `DeviceMatrix` needs no stride and no kernel changes: a shard is just a
// smaller matrix.
//
// `rscales` is per row: sliced for gate/up, and kept WHOLE for down, since every part writes every
// row. Scaling each partial by the row scale and adding is the same number as scaling the sum.
struct ExpertShardLayout {
  uint32_t data_off[3] = {}, scale_off[3] = {}, rscale_off[3] = {};   // gate, up, down
  uint32_t rows[3] = {}, cols[3] = {};
  uint32_t stride = 0;                        // bytes of one shard, 256-aligned per plane
};

// Fills `out` for a `parts`-way split of `L`'s expert shape. False when the shape does not divide:
// gate/up need rows % parts == 0, down needs its column groups (cols / 64) % parts == 0 and the
// resulting cols still a multiple of 512, which is what the gemv kernels require.
bool expert_shard_layout(const AffLayerDesc& L, uint32_t parts, ExpertShardLayout* out);

// Copy part `part` of one expert out of `whole` — which must ALREADY be colgroup-major, i.e. after
// expert_transpose_planes_host — into `dst`, which must hold `S.stride` bytes.
void expert_shard_host(const uint8_t* whole, const AffLayerDesc& L, const ExpertShardLayout& S,
                       uint32_t parts, uint32_t part, uint8_t* dst);

// Check a shard against the container's ROW-MAJOR bytes — `raw` is one expert exactly as it came
// off disk. Returns the number of mismatching bytes, 0 for a correct shard.
//
// Shares nothing with the code it checks but the three layout constants: it addresses the row-major
// plane directly and works out where each span must have ended up, rather than replaying the
// permutation and the slice.
size_t expert_shard_check_host(const uint8_t* raw, const AffLayerDesc& L,
                               const ExpertShardLayout& S, uint32_t parts, uint32_t part,
                               const uint8_t* shard);

// A quantised expert matrix in device memory. Pointers are device addresses; the layout is byte
// for byte what the container holds.
struct DeviceMatrix {
  const uint8_t*  data    = nullptr;
  const uint8_t*  scales  = nullptr;
  const uint16_t* rscales = nullptr;   // bf16 per row
  uint32_t rows = 0, cols = 0;
};

// y[0..rows) += w * (W . x[0..cols))
//
// `cb` is the fitted codebook in device memory: 4 variants x 32 entries x 2 floats, exactly as
// stored in the container's `__codebook` tensor. `stream` may be null for the default stream.
//
// Requires cols % 512 == 0 (one wave of 32 lanes covers a whole row, 16 weights per block, and at
// least 4 blocks per lane so the 6-bit scale fields start on byte boundaries).
void expert_gemv_hip(const DeviceMatrix& m, const float* cb, const float* x, float w, float* y,
                     void* stream = nullptr);

// Same, but evaluates several experts into one accumulator in a single launch — the shape the
// decode path actually wants, since all six routed experts sum into the same vector.
void expert_gemv_batch_hip(const DeviceMatrix* mats, const float* weights, uint32_t n_mats,
                           const float* cb, const float* x, float* y, void* stream = nullptr);

// Like expert_gemv_batch_hip, but each matrix carries its own activation. The six down projections
// of a routed layer share an output vector while having different inputs, which is what lets them
// collapse into a single launch.
void expert_gemv_multix_hip(const DeviceMatrix* mats, const float* const* xs, const float* weights,
                            uint32_t n_mats, const float* cb, float* y, void* stream = nullptr);

// SPLIT OUTPUT: n matrices of the SAME shape, each reading its own x and writing its own y. gate
// and up cannot share an output the way `down` does — each writes its own intermediate — but they
// can share a LAUNCH: at decode that is ~5.8 launches of 3.02 MB against one of 17.4.
//
// Stores rather than accumulates, so `ys` need not be zeroed. Returns false if the planes are not
// colgroup-major (no split kernel for row-major) or the shapes differ; the caller falls back.
bool expert_gemv_split_hip(const DeviceMatrix* mats, const float* const* xs, float* const* ys,
                           const float* weights, uint32_t n_mats, const float* cb,
                           void* stream = nullptr);

// ---- the batched form: one expert, T token columns, ONE pass over the weights ------------------
//
// `y[rows][nt] += wscale * m * x[cols][nt]`, both tiles dim-major and compact. This is the prefill
// shape: a chunk of C tokens gives each expert `C * top_k / n_expert` = C/42.7 of them, so at chunk
// 2048 one pass over an expert's 9 MB serves ~48 tokens instead of one. `nt` must be a power of two
// up to 32 (it sizes a register array); pad and ignore the spare columns.
//
// `y` is STORED, not accumulated: a tile belongs to one expert for the length of one call, so
// there was never anything to accumulate onto and the memsets that used to zero it are gone.
bool expert_gemm_tokens_hip(const DeviceMatrix& m, const float* cb, const float* x, float* y,
                            uint32_t nt, float wscale, void* stream = nullptr);

// Two matrices of the SAME shape against the same `x`, in one launch (blockIdx.y picks one). gate
// and up are exactly that pair, and they were two of the nine launches an expert used to cost.
// `b` may be null, which is the single-matrix form.
bool expert_gemm_tokens_pair_hip(const DeviceMatrix& a, float* ya, const DeviceMatrix* b, float* yb,
                                 const float* cb, const float* x, uint32_t nt, float wscale,
                                 void* stream = nullptr);

// The widest token block one tile may carry. This is a CLIFF, not a ceiling: one step wider and the
// accumulators alone exceed the wave32 register file — `raw` and `out` are a double accumulator for
// the 128-column block-scale fold — so they spill to scratch and occupancy collapses. Splitting a
// double-width tile into two of these pays one extra read of the weight and still wins by more than
// a factor of two.
constexpr uint32_t kExpertWmmaTok = 64;

// One expert's slice of a batched W8A8 dispatch: its weight planes, the chunk tokens it was routed
// and where its rows go. All experts in a layer share `rows`, `cols`, the activation arena and the
// codebook, so only these differ.
struct ExpertTile {
  const uint8_t*  data    = nullptr;
  const uint8_t*  fields  = nullptr;
  const uint16_t* rscales = nullptr;
  const uint32_t* tok     = nullptr;   // nt token ids into the chunk; null means identity
  // [rows][ld], this tile's nt columns. f32 or bf16 — expert_gemm_wmma_hip's y_bf16 decides,
  // and the consumer must agree: the engine writes bf16, bench/expert_wmma keeps f32 so its
  // correctness check still resolves a layout bug from a rounding difference.
  void*           y       = nullptr;
  uint32_t        nt      = 0;
  // Column stride of `y`; 0 means the tile owns a compact [rows][nt] block of its own. A non-zero
  // `ld` lets every tile in a layer write into ONE arena at its own column offset, which is what
  // turns the swiglu, the requantise and the scatter into single whole-arena launches instead of
  // one per tile. Without it they would have to walk a ragged list of per-tile blocks.
  uint32_t        ld      = 0;
};

// W8A8 form. The weights stay packed at 2.875 bpw and the codebook decode is fused into the K-loop
// via a 4 KB LDS table that absorbs the 4-bit block scale, so the inner loop does one lookup a
// weight PAIR and no arithmetic. `x8` is the activation, E4M3 and TOKEN-major ([nt][cols]), with
// `xs` one f32 scale per (token, 128-column block). `y` is [rows][nt] f32 with the row rscale
// already applied.
// `tiles` is a DEVICE array of n_tiles descriptors; the grid is (rows/128, n_tiles) so every expert
// in a layer goes out in one launch. That is 4096 blocks against 16 — a single expert does not fill
// the card, and the kernel measures nearly linear in block count.
void expert_gemm_wmma_hip(const ExpertTile* tiles, uint32_t n_tiles, uint32_t max_nt,
                          const uint8_t* x8, const float* xs, const float* cb, uint32_t rows,
                          uint32_t cols, float wscale, void* stream = nullptr,
                          bool y_bf16 = false);

// SwiGLU over a [dim][total] arena AND the requantise to E4M3, in one pass. `g` and `u` are the
// gate and up GEMM outputs, `w` the per-slot router weight, `out` is token-major [total][dim] with
// `scales` one f32 per (slot, 128-column block) — exactly what the down GEMM consumes.
//
// Fused because the f32 intermediate is used once: separate kernels would write [dim][total] f32
// and read it straight back, 25 MB each way a layer a card at this shape. The router weight rides
// here rather than in the scatter because the scale is picked from this data, so a per-slot factor
// applied before the amax is absorbed into it exactly.
void expert_swiglu_quant_hip(const uint16_t* g, const uint16_t* u, const float* w, uint8_t* out,
                             float* scales, uint32_t dim, uint32_t total, float limit,
                             void* stream = nullptr);

// Reduce a [dim][total] slot arena into the chunk's [dim][nb] accumulator.
//
// `ord` lists the slots sorted by token and `run` is its nb+1 prefix, so token t owns
// ord[run[t] .. run[t+1]). One block a dim: the whole `total`-wide row it reads is contiguous, so
// the scattered order inside it is served by cache, and the writes it makes are consecutive tokens.
//
// The reduction order is FIXED, unlike the per-tile scatter's atomicAdd. Same arithmetic every run.
void expert_scatter_slots_hip(const uint16_t* y, float* block, const uint32_t* ord,
                              const uint32_t* run, uint32_t dim, uint32_t nb_tok,
                              uint32_t nb_stride, uint32_t total, void* stream = nullptr);

// Gather `nt` token columns out of a dim-major [dim][nb] chunk arena into a compact [dim][nt] tile,
// which is what the above consumes.
void expert_gather_tokens_hip(const float* src, float* dst, const uint32_t* tok, uint32_t dim,
                              uint32_t nb, uint32_t nt, void* stream = nullptr);

// One expert's three matrices, as they sit in VRAM.
struct DeviceExpert {
  DeviceMatrix gate, up, down;
};

// The complete routed-expert FFN for one token, entirely on device:
//
//   out += sum_k  weight[k] * down_k( swiglu( gate_k(x), up_k(x) ) )
//
// `scratch` must hold 2 * n_experts * inter floats. Keeping the intermediate on device matters:
// at 2048-wide intermediate and six experts it is 48 KiB per token, and round-tripping it over
// PCIe would cost more than the arithmetic it feeds.
//
// The SwiGLU clamp is asymmetric BETWEEN its two operands, matching Expert.forward in the
// checkpoint's own inference/model.py: the gate is bounded above only, `up` on both sides.
void expert_ffn_hip(const DeviceExpert* experts, const float* weights, uint32_t n_experts,
                    const float* cb, const float* x, float swiglu_limit,
                    float* scratch, float* out, void* stream = nullptr);

// Largest token tile the batched form takes in one call. It is the register array's size, so this
// is a hardware limit, not a policy: a chunk giving an expert more than this is split.
constexpr uint32_t kExpertTokMax = 32;

// ONE expert's complete FFN over `nt` of the chunk's token columns, entirely on device:
//
//   block[:, tok[t]] += w[t] * down( swiglu( gate(x[:, tok[t]]), up(x[:, tok[t]]) ) )
//
// `x_arena` and `d_block` are the chunk's dim-major [dim][nb] buffers and are NOT copied; the
// gather and the scatter are the only two places the token list is touched. `d_tok` and `d_w` are
// device arrays of `nt` entries. `scratch` must hold (2*hidden + 2*inter) * T floats, where T is
// `nt` rounded up to a power of two.
//
// This is the prefill shape. The decode path keeps expert_ffn_hip: at one token the two are the
// same arithmetic and this one only adds a gather.
bool expert_ffn_tokens_hip(const DeviceExpert& e, const float* cb, const float* x_arena,
                           const uint32_t* d_tok, const float* d_w, uint32_t nt, uint32_t nb,
                           uint32_t hidden, uint32_t inter, float limit, float* scratch,
                           float* d_block, void* stream = nullptr);

// The same FFN in ONE cooperative launch. Removes thirteen of the fourteen launches and the gaps
// between them; requires a device with cooperativeLaunch (gfx1201 has it, the iGPU does not).
void expert_ffn_mono_hip(const DeviceExpert* experts, const float* weights, uint32_t n_experts,
                         const float* cb, const float* x, float swiglu_limit,
                         float* scratch, float* out, void* stream = nullptr);

// ---------------------------------------------------------------------------------------------
// Dense matvec — most of the traffic. Per token the dense side moves several times what the routed
// experts do, and it is also the part that FITS in VRAM.
//
// FP8-E4M3 with one UE8M0 scale per 128x128 tile. `S` holds ceil(rows/128) * ceil(cols/128) scale
// bytes, row-major, exactly as the checkpoint ships them.
void matvec_fp8_hip(const uint8_t* W, const uint8_t* S, const float* x, uint32_t rows,
                    uint32_t cols, float* out, void* stream);

// The same matvec with the RMSNorm of its own input folded in as a prologue: out = W * rmsnorm(x,
// nw), one launch instead of two. Every block redoes the reduction over all `cols` — 4 KB out of L1
// — because the alternative is a one-block kernel that owns 1 of 32 CUs and costs a launch.
// Bit-identical to rms_norm_f32_kernel followed by matvec_fp8_hip; see the kernel for why.
// `xn` may be null; when it is not, the normalised vector is written there as well.
void matvec_fp8_rmsnorm_hip(const uint8_t* W, const uint8_t* S, const float* x, const float* nw,
                            float eps, uint32_t rows, uint32_t cols, float* out, float* xn,
                            void* stream);

// Grouped FP8 matvec in ONE launch: out[g*rank + r] = sum_c W[(g*rank+r)*group_dim + c]
// * x[g*group_dim + c]. Requires rank % 128 == 0 so each group starts on a scale-tile row.
void matvec_fp8_grouped_hip(const uint8_t* W, const uint8_t* S, const float* x, uint32_t n_groups,
                            uint32_t group_dim, uint32_t rank, float* out, void* stream);

// The same grouped matvec with the INVERSE RoPE of its own input folded in: x is read as if
// rope_tail_kernel(..., sin_sign = -1) had already run over it, but x is not written. Rotates one
// group's tail into LDS per block, so the transcendentals are paid once per block and not once per
// row. Returns false — having launched nothing — when any of its shape preconditions fails, and the
// caller must then run the separate rope kernel; see the definition for the list.
bool matvec_fp8_grouped_rope_hip(const uint8_t* W, const uint8_t* S, const float* x,
                                 uint32_t n_groups, uint32_t group_dim, uint32_t rank,
                                 uint32_t head_dim, uint32_t n_rot, uint32_t pos,
                                 float freq_scale, float theta_scale, float ext_factor, float lo,
                                 float hi, float* out, void* stream);

// `want_tall` forces the block-per-row kernel that the row-count heuristic below would only pick
// for a short matrix. It is a NUMERIC request: the two kernels reduce a row in different orders and
// agree only to the last bit, which is enough to swap a near-tied router selection. Any two callers
// whose results must be comparable have to make the same request.
void matvec_bf16_hip(const uint16_t* W, const float* x, uint32_t rows, uint32_t cols,
                     float* out, void* stream = nullptr, bool want_tall = false);

// Several dense matvecs that share one `x`, in ONE launch. Two problems, one fix: independent
// projections issued back to back each pay a dispatch ramp (see gpu/keepalive.h), and a per-matrix
// grid of `rows / 8` blocks leaves the card nearly empty for the small ones. Sizing the grid from
// the TOTAL row count fixes both.
//
// Every job must have the same `cols` — they share `x`. Quant may differ per job; a wave handles
// one row of one matrix, so the dispatch on it is wave-uniform and costs nothing.
struct MvJob {
  const void* data   = nullptr;
  const uint8_t* scales = nullptr;      // fp8 only
  uint32_t quant = 0;                   // AFF_FP8_E4M3 or AFF_BF16
  uint32_t rows  = 0;
  float* out     = nullptr;
  // Folded into the store, so a caller that would otherwise follow the matvec with a scale kernel
  // does not need one. `x * 1.0f == x` for every finite float, so leaving it at the default is
  // BIT-IDENTICAL to the version that had no scale at all — verified against the frozen text.
  float scale    = 1.0f;
};
// An optional side copy folded into the same launch: dst[i*stride] = src[i] for i < n, done by
// block 0 alongside its share of the rows. For the compressor's staging column, which is a small
// copy of a vector the matvec is already reading and not worth a launch of its own. Nothing is
// ordered against the matvec: src is read-only to both and dst is touched by neither.
struct MvStage {
  const float* src = nullptr;
  float* dst       = nullptr;
  uint32_t n       = 0;
  uint32_t stride  = 1;
};
void matvec_batch_hip(const MvJob* jobs, uint32_t n_jobs, const float* x, uint32_t cols,
                      void* stream = nullptr, const MvStage* stage = nullptr);

// The same batch against `nt` activation vectors instead of one, and the reason it is not just a
// GEMM: the weight row is loaded once for the whole group but each (row, token) accumulator keeps
// the per-lane order `bf16_row_dot` uses, so the result is BIT-IDENTICAL to `nt` separate
// matvec_batch_hip calls. A WMMA GEMM is not — and at a small `nt` it is also slower, because kWN
// is 128 and it computes a full N tile whichever four columns are live.
//
//   x        TOKEN-major, `nt` vectors of `cols` floats, token t at x + t*x_stride.
//   out[j]   TOKEN-major [nt][jobs[j].rows] -- which is what the compressor's pool indexes by.
//
// BF16 ONLY. Returns false and does nothing if any job is FP8 or nt is out of range, so the caller
// keeps whatever path it would otherwise have taken; there is no silent half-answer.
constexpr uint32_t kMvMultiXMax = 8;
// ...and the width above which it stops being the faster of the two. A wave re-reads all `nt`
// activation vectors for every row it owns, so the working set is nt * cols * 4 bytes — and one
// column past this it stops fitting in cache, at which point the matvec falls well behind the GEMM.
//
// LDS staging fixes that cliff and loses below it, since it pays barriers per row-group to save
// reads that were already hitting cache. So no single kernel is best at both, and carrying two was
// not worth changing the arithmetic of the head that decides the output. Callers with a choice take
// the GEMM above this width.
constexpr uint32_t kMvMultiXFast = 5;
bool matvec_multix_hip(const MvJob* jobs, uint32_t n_jobs, const float* x, uint32_t cols,
                       uint32_t nt, uint32_t x_stride, void* stream = nullptr);

// out[row] = swiglu(gate[row] . x, up[row] . x), one wave per row, ONE launch. The gate/up pair and
// the elementwise SwiGLU over them used to be two launches plus a 8 KB round trip through VRAM.
void mv_gate_up_swiglu_hip(const void* wg, const uint8_t* sg, uint32_t quant_g, const void* wu,
                           const uint8_t* su, uint32_t quant_u, const float* x, float* out,
                           uint32_t rows, uint32_t cols, float limit, void* stream = nullptr);
constexpr uint32_t kMvBatchMax = 8;


// True when at least one gfx12 device is visible. Note this excludes the 7950X3D's integrated
// gfx1036, which HIP enumerates but which has no code objects here. The engine runs CPU-only when
// this is false.
bool hip_available();

// Human-readable device summary, or the reason there is none.
std::string hip_device_summary();

} // namespace aff
