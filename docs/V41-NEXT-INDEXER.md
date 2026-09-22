# NEXT: the V4.1 indexer (and the candidate pre-filter)

Written 2026-09-22 at session close, while it was fresh. Read `V41-REPRO.md` first for how to run
what already works.

## Why this is the next thing

V4.1 keeps **`index_topk` = 512** compressed rows per query and attends to those. This engine builds
no index and attends over **every** compressed row. Below ~1K tokens of context those are the same
set, which is why short prompts look right. Above it we are doing more work than the reference AND
computing a different attention — so long-document behaviour is unverified, and long documents are
the workload this model was quantised for. It is a correctness item that presents as "slightly odd
answers on long inputs", not as a crash.

## What V4.1's indexer is

`inference/model.py`, class `Indexer` (line ~488) and `select_candidate_blocks` (~583); shapes in
`config.json`: `index_n_heads` 32, `index_head_dim` 128, `index_topk` 512,
`index_source_layer_ids` [2,8,14,20,24,28,32,36], `kv_source_layer_ids` [2,8,14,20],
`candidate_source_layer_id` 20, `candidate_topk_blocks` 2048, `candidate_block_size` 8.

The split is the thing to get right, and it differs from V4:

| piece | tensor | lives on | note |
|---|---|---|---|
| keys | `attn.indexer.wk` + `k_norm` | the 4 **kv-source** layers | V4 built keys from a SECOND compressor (`idx_comp_*`); V4.1 projects the layer's own latent. Those tensors do not exist here |
| queries | `attn.indexer.wq_b` + `weights_proj` | the 8 **index-source** layers | per-head weights, `1/sqrt(dim*heads)` folded in |
| top-k | — | index-source layers | the consumers between sources REUSE it, exactly like the compressed KV |

So the sharing already built for the compressed cache (`kv_owner_`, `Model::is_index_source`,
`DenseGpu::kvl`) is the same shape the index needs — an `idx_owner_` beside it, mapping a layer to
the index-source that produced its top-k.

## Where it goes in the engine

- The container already carries all four tensors: `idx_wk`, `idx_k_norm`, `idx_wq_b`, `idx_proj`,
  bound per layer and required exactly where the checkpoint has them (`model.cpp`, the V4.1 branch
  of the layer loader). Nothing reads them yet.
- `has_idx` is currently `w.ratio == 4 && w.idx_comp_wkv && ...`, which is false for every V4.1
  layer — that single predicate is what turns the whole path off. It needs a V4.1 form:
  keys on `is_kv_source`, queries on `is_index_source`.
- The scoring and the admission mask already exist and are good: `indexer_scores_hip`,
  `DenseGpu::indexer_one` (decode) and the batched one (prefill), both already routed through
  `kvl()` so they read the owner's key plane. What is missing is BUILDING the keys from `idx_wk`
  rather than from the indexer's own compressor, and the query side on the 8 source layers.
- The candidate pre-filter (layer 20, 2048 blocks of 8) is a second, coarser stage in front of the
  indexer. Leave it until the indexer is right; it only matters at long context, where its whole
  purpose is to cut what the indexer scores.

## How to know it works

1. `tools/v41/refblock.py` on layer 2 (a kv+index source) and layer 3 (a consumer) gives golden
   vectors; the layer-3 case needs layer 2 run first in the same process so the shared state exists.
2. End to end, the honest check is a long prompt: `tools/v41/mkprompt.py` makes 4K and 16K fixtures.
   Before/after the indexer, the 16K answer should not change in kind — and prefill should get
   FASTER, because scoring 512 rows beats attending to all ~8000.
3. The standing numbers to beat, two R9700s, `--dspark off`, 136 GiB pool:
   3,806 tok → 829 tok/s prefill; 15,159 tok → 1,105 tok/s; decode 17.7 short / ~10.3 at depth.

## The other open items, in order

1. **Engram** (layers 1 and 14). Tables stay in the checkpoint (2 × ~95 GiB FP8, `engram.embed.*`);
   the container holds only `engram_q`/`k`/`wkv`. Needs the n-gram hash to match `engram.py`
   exactly — `NgramHashState`, sizes to 4, 16M buckets, primes, pad id 2 — or the lookups are noise.
   Until this lands, this is not the released model.
2. **Quality gates**: KL vs the FP4 teacher on Lucebox's 8,184 fineweb-edu tokens (their `eval/`
   holds the teacher log-probs, so no need to run the full model), then pv_bench v3, then the
   agentic replay. Note the standing lesson: sub-4-bit DS-V4 quants have passed pv_bench and still
   failed multi-turn tool use.
3. **The draft**: 12 % acceptance, costs 7.35 GiB of expert slab a card, so it is off. The tap fix
   was right but insufficient; next suspects are the window index set `get_dspark_topk_idxs` builds
   and the Markov/confidence heads.
