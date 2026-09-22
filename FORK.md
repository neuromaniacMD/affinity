# This is a fork

Upstream: **[StillDeadcode/affinity](https://codeberg.org/StillDeadcode/affinity)** — an inference engine
for DeepSeek-V4-Flash on one or two RDNA4 GPUs, by Yoshi Exeler. MIT licensed; `LICENSE` and the copyright
are kept as they are. Everything good in the engine is upstream's work.

This fork exists to run **DeepSeek-V4.1-Flash** on a pair of Radeon AI PRO R9700s, and to hold the fixes
found while doing it. Upstream is unchanged since 2026-08-18; nothing here has been offered to it yet.

## Branches

| branch | what |
|---|---|
| `main` | plain mirror of upstream `main` |
| `v41` | **the working branch**: DeepSeek-V4.1 support (default branch) |
| `amdnas-fixes` | the poison-bug work: everything below in one line of history |
| `pr/slab-budget-own-allocs` | placement: count the engine's own allocations inside the slab budget |
| `pr/hugepage-first-host-pool` | host pool: `MADV_HUGEPAGE` **before** populating, so it registers on 2 MiB pages |
| `pr/sampler-oob-token` | sampler: never return an out-of-range token id on inverse-CDF drift |
| `pr/check-kernel-launches` | gpu: check every kernel launch with `hipGetLastError` |
| `pr/compress-reset-per-seq` | engine: clear the compressor/indexer history rings on sequence reset |

The five `pr/*` branches are single commits on upstream `main`, meant to be offered upstream as they are.

## The bug they came from

At default flags on two 32 GiB cards, the server ran fine until the first long streamed request and then
emitted fluent token soup for the life of the process — no HIP error, no abort, DSpark acceptance pinned at
0.0 %. Two things combined: the slab left ~122 MiB free rather than the intended 262 (the shadow slots and
codebook are allocated *after* the budget), and the 51 GiB host expert pool was registered with the GPUs on
4 KiB pages and only then hinted `MADV_HUGEPAGE`, so khugepaged collapsed pages under a live GPU mapping.
Each collapse makes the driver rebuild GPU page-table entries, and with little free VRAM the rebuild left
stale ones: the expert GEMM streamed the wrong bytes. At 62 MiB free it faults instead; at ≥162 MiB it
survives. Either fix alone clears it (10/10 clean runs each, against 9/9 poisoned before).

## DeepSeek-V4.1 status

The **container side is done**: `aff-quantize` writes V4.1 containers (191 GiB at 2.875 bpw, imatrix-weighted)
and a draft container, both verified. The **engine side is not** — see `docs/V41-PORT-MAP.md` for what changes
and `tools/v41/` for the CPU reference harness that produces golden vectors to port against.

## Pushing here

Work happens on `amdnas`, whose `origin` is a bare repo on macserver; this GitHub remote is pushed from the
StrixHalo laptop, which is where the `gh` credentials live.
