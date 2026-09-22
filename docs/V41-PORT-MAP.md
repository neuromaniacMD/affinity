# PORT MAP — DeepSeek-V4.1 on the affinity engine (what changes vs the V4 code)

Sources: the checkpoint's own `~/hf/DeepSeek-V4.1-Flash/inference/{model.py,engram.py,kernel.py,convert.py}`
(read 2026-09-22) and affinity fork `amdnas:~/src/affinity-fork` (V4). Container side is DONE
(`PLAN-AFFINITY-V41-2026-09-22.md`); this is the engine side. ⚠️ Items marked **[verify]** are read from the
reference but not yet checked line-by-line against affinity's C++.

## Same as V4 — no work expected
- Block shape: hc stream of `hc_mult`=4 copies, `hc_mixes` → (pre, post, comb) with Sinkhorn on comb,
  `hc_pre` → norm → sublayer → `hc_post`; each sublayer's mixes feed the NEXT one (`model.py:968`).
  Same constants: hc_mult 4, sinkhorn_iters 20, hc_eps 1e-6. affinity already implements all of it.
- Attention skeleton: low-rank q (`wq_a`→`q_norm`→`wq_b`), single shared KV head (`wkv`, MQA), grouped
  low-rank output (`wo_a` block-diagonal over `o_groups`=8 + `wo_b`), learned `attn_sink` added to the
  softmax DENOMINATOR only, sliding window + optional compressed KV in ONE `sparse_attn` call.
- Expert FFN: gate/up/down with SwiGLU (`swiglu_limit` 10.0), 1 shared expert, routed experts through the
  affinity codebook. Geometry per expert unchanged in kind (hidden 5120, inter 2304).
- RoPE: YaRN on compressed KV (`compress_rope_theta` 160000), plain `rope_theta` 10000 for window-only
  layers; rotation applied to the last `qk_rope_head_dim`=64 of q, and INVERSELY to the attention output.

## Changed — must be edited
| # | Thing | V4 | V4.1 |
|---|---|---|---|
| 1 | hidden / inter / experts | 4096 / 2048 / 256, 6 active | **5120 / 2304 / 384, 6 active** |
| 2 | layers | 43 | **40** (+3 mtp stages) |
| 3 | `compress_ratios` | 0 / 4 (CSA) / 128 (HCA) | **0,0, then 2 ×18, then 1 ×20, 0,0,0 (mtp)** — the ratio-4/128 special cases become ratios 1 and 2 |
| 4 | compressor | per compressing layer, has `ape` | **only on `kv_source_layer_ids` [2,8,14,20]; NO `ape`; `wgate` only on 2,8,14** (layer 20, ratio 1, has none) |
| 5 | indexer | per layer at ratio 4, own compressor | **`wq_b`+`weights_proj` on `index_source_layer_ids` [2,8,14,20,24,28,32,36]; `wk`+`k_norm` only on the 4 kv-source layers; no indexer-compressor** |
| 6 | KV/index sharing | none | ★ **`SharedAttentionRuntime` (`model.py:1166`): a kv-source layer computes compress_kv + index_k; an index-source layer computes topk_idxs; consumers reuse them. One slot each, written before read** |
| 7 | candidate pre-filter | none | ★ **`select_candidate_blocks` (`model.py:583`), source layer 20, `candidate_topk_blocks` 2048, block 8** |
| 8 | router | softmax/sigmoid + hash routing on layers 0-2 + aux-loss-free bias | **`scoring_func = sqrtsoftplus`, `gate_temp`, `norm_topk_prob` true, `routed_scaling_factor` 1.5; NO hash layers; bias PLUS `bias_vl` (image-span routing bias, `Gate.forward(x, image_mask)`)** |
| 9 | head | `hc_head_fn/base/scale` collapse the hc copies | **no `hc_head_*` tensors in the checkpoint — `ParallelHead` (`model.py:997`) must be re-read [verify]** |
| 10 | dense FP8 | one UE8M0 scale per 128×128 | 32×32 in the checkpoint — **already handled: the container refolds to 128×128, lossless (rel ~1e-7)** |
| 11 | `wo_a` | FP8 | **checkpoint FP8 but `convert.py` dequantises it to bf16 and DROPS its scale** — the reference expects bf16. Our container keeps it FP8 (fine for the engine, but the engine must not assume convert.py semantics) |

## New — must be written
- ★ **Engram** (`engram.py`, `model.py:296-368`): n-gram hash lookups added to the residual stream at layers
  **1 and 14**, between the layers (`Transformer.forward:1262`), NOT inside the block. Per layer: a table of
  ~384 M rows × 256 FP8 (~95 GiB each, **left in the checkpoint shards — the engine mmaps them**), plus
  `q_weight`/`k_weight` [4,5120] and `wkv` [25600,6144] FP8 in the container. Hashing is
  `NgramHashState` (n-gram sizes to 4, `engram_vocab_size` 16 M buckets, primes via sympy, pad id 2,
  compressed vocab 99092) — the hash must match the reference exactly or the lookups are noise.
- **DSpark draft**: 3 stages × 128 experts (top-3), `main_proj` over 3 tapped layers [37,38,39],
  `markov_head.{embed,head}` (V4 called them `markov_w1/w2`), `confidence_head`, block size 5.
  `DSparkAttention` (`model.py:1032`) is window-only (ratio 0 asserted).
- **Vision**: out of scope (text-only lane), but `image_mask` reaches the router (`bias_vl`) — with no
  images the mask is None and the plain bias is used.

## Verification harness (BUILT TODAY, works)
`amdnas:~/aff-v41/{kernel_shim.py,refblock.py}` (copies in `deployed/v41-quant/`) runs the checkpoint's own
`model.py` on CPU: the TileLang kernels are replaced by torch stand-ins — arithmetic ones by
dequantise-then-matmul in float32, and the two that are model math (`hc_split_sinkhorn`, `sparse_attn`)
reimplemented statement-for-statement from `kernel.py`. Weights load straight from the safetensors by
module path; `wo_a` is folded to float per `convert.py`.
```
venv/bin/python refblock.py ~/hf/DeepSeek-V4.1-Flash <layer> ~/aff-v41/golden [seqlen]
```
Writes `golden/L<n>/{x,pre_mix,out,ffn_pre}.npy` + meta. **Verified: deterministic (bit-identical reruns);
layer 0 (window-only), layer 2 (kv+index source, ratio 2) and layer 20 (ratio 1) all run.** ~7 s a layer.
Not yet covered: consumer layers that need a source layer's shared state (run the source first into the same
process), the Engram layers, and the mtp stages.

## Suggested order
1. Loader: read the V4.1 container + config into the engine's model struct (names above), no forward yet.
2. Block forward for a **window-only** layer (0/1) → match `golden/L0` within tolerance. This exercises hc,
   attention, router, experts and already proves most of the stack.
3. Compressor/indexer on a kv-source layer (2) → `golden/L2`; then a consumer layer via the shared runtime.
4. Candidate pre-filter (layer 20 source) → `golden/L20`.
5. Engram (layers 1, 14) — hash state first, checked against the reference's own hashes.
6. Whole-model forward, then the gates (KL vs the FP4 teacher on Lucebox's 8,184 tokens, pv_bench v3,
   agentic replay).
