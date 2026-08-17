# Affinity — multi-sequence concurrency: design + ROI analysis

Branch `feat/concurrency-slots` (isolated tree `/root/affinity-conc`, production `/root/affinity`
untouched). Base: `3ff1359` (7892d58 + #179 fix).

## 1. What blocks concurrency today

| Layer | State | Blocker |
|---|---|---|
| Server | `server.cpp` | one global `gen_mu_` around all generation |
| Host seq state | `SeqState` {layer, hc, pos, hc_host} | already per-sequence ✅ |
| Device KV | `DenseGpu::Impl::Dev.a_raw[layer]`, `.a_comp[layer]` | **single set** — one ring, one comp cache |
| Device indexer keys | `.a_idx8[layer]`, `.a_idxs[layer]` | single set |
| Compressor window | `.c_hkv[l]`, `.c_hsc[l]`, `.ci_hkv[l]`, `.ci_hsc[l]` | single set (`compress_reset()` = "sequence reset") |
| KV grower | `kv_pos_`, `kv_backed_` atomics | single sequence |
| Prefix cache | keyed to the one live sequence | single |
| Attention kernel | `batch_attend(layer, b0, n, pos0, …)`, flash `pos = pos0 + tk` | scalar `pos0`, scalar KV base |

## 2. The good news: the batched compute primitive already exists

`forward_prefill(ids, n, SeqState*, …, BlockOut*)` runs **n tokens through the model in one
batched step** and, with `BlockOut`, returns every position's argmax. That *is* the speculative
block verify (`tools/affinity.cpp`: *"same block verify at width 1: one token a cycle instead of
1+B, through the same code"*). DSpark therefore already runs the engine at **B ≈ 6** every step.

The flash kernel already takes **per-token** `n_comp_v[tk]` and per-token admissions
(`adm` + `adm_stride`), so per-token indirection is an established pattern there.

## 3. Design (if built)

**`--slots N`** — split the KV budget N ways; each slot gets its own KV ring, comp cache, indexer
key cache and compressor window. VRAM is the hard constraint: at 512K there is ~290 MiB/card free,
so slots must divide context (2×256K ≈ 1×512K in VRAM terms, verified from the measured curve).

- **M1 — slotting.** Make the per-sequence device arrays 2-D `[slot][layer]`; add `cur_slot_` to
  `DenseGpu`; `kv_pos_`/`kv_backed_` per slot. Mechanical refactor of accessors, no kernel change.
  Execution still serialized → **no throughput gain**, purely the foundation.
- **M2 — batched decode across slots.** Replace scalar `pos0`/KV base with per-token arrays:
  `pos_v[]` and `slot_v[]` (base offset = `slot·stride`), threaded through `batch_attend`,
  `kv_commit`, `batch_indexer` and the flash kernel. This is where throughput comes from.
- **M3 — scheduler.** Replace `gen_mu_` with a slot manager + a step loop that gathers pending
  sequences into one forward. Prefix cache per slot.

## 4. ROI — why this is a worse trade than it first looks

Decompose a decode step as `F + B·m` (fixed + marginal), plus `d` per draft pass.

From measurement: dspark **off** = 1 token/step at 33 ms → `F + m = 33`. dspark **on** (code) =
3.94 tokens per block at 95.1 t/s → 41.4 ms per block = one 6-wide verify + 5 draft passes →
`F + 6m + 5d = 41.4`. Hence `m + d ≈ 1.68 ms`, and with a plausible draft cost `d ≈ 1.2 ms`:
**F ≈ 32.5 ms, m ≈ 0.5 ms/token — the step is ~93% fixed cost.**

That is exactly the quantity batching amortises — **but DSpark already amortises it.** Projecting
N concurrent sequences, each running its own block (`B = 6N`, `5N` draft passes):

| N slots | ctx per slot | aggregate | vs today | per-request |
|---:|---:|---:|---:|---:|
| 1 (today) | 512K | 95 t/s | 1.00× | 95 t/s |
| 2 | 256K | ~157 t/s | ~1.65× | ~78 t/s |
| 4 | 128K | ~232 t/s | ~2.4× | ~58 t/s |
| 8 | 64K | ~304 t/s | ~3.2× | ~38 t/s |

(Upper bounds: assumes drafts don't batch and ignores per-sequence KV read growth, which erodes
it at longer context — precisely where this lane is valuable.)

So the honest offer is **~2.4× aggregate at N=4, paid for with a 4-way context split (512K→128K)
and ~40% slower per-request decode**, for M1+M2+M3 — weeks of kernel surgery in an upstream
codebase, against the author's explicit design decision D1.

The naive "~4× headroom" figure from the earlier analysis was measured against **B=1 (DSpark off)**.
Against the config we actually run, most of that headroom is already banked.

## 5. The better target: reduce F

`F ≈ 32.5 ms` across 43 layers = **~0.76 ms per layer of fixed cost**, and the engine has **no HIP
graph capture** (`grep hipGraph` → 0 hits) — every decode step re-issues every kernel of all 43
layers from the host, plus per-layer collectives and syncs.

Halving F would take single-stream decode from ~95 → ~150 t/s **on every request**, with no context
split, no per-request slowdown, and no scheduler. It also multiplies with DSpark rather than
competing with it, and would benefit every Affinity user (2-card included) — a far better upstream
contribution than a batching fork.

**Recommended order:**
1. `AFF_PROFILE=1` / `AFF_BLOCK=1` decode breakdown (~10 min, lane down ~3 min) → localise F.
2. If F is launch/sync-dominated: prototype HIP graph capture for the decode step.
3. Revisit slot-based batching only if aggregate throughput is still the binding constraint.
