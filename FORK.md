# This is a fork

Upstream: **[StillDeadcode/affinity](https://codeberg.org/StillDeadcode/affinity)**, an inference engine for
DeepSeek-V4-Flash on one or two RDNA4 GPUs, by Yoshi Exeler. MIT licensed; `LICENSE` and the copyright are kept
as they are. The engine is upstream's work. This fork adds robustness fixes, DeepSeek-V4.1-Flash support, and a
reproducible container build, all measured on **two Radeon AI PRO R9700s** (gfx1201, 32 GiB each).

## Branches

| branch | what | use it for |
|---|---|---|
| `main` | plain mirror of upstream `main` (`a48b72e`) | reference |
| `v4-fixes` | upstream + the fixes below | **DeepSeek-V4-Flash and DeepSeek-V4-Flash-0731** |
| `v41` | `v4-fixes` + DeepSeek-V4.1-Flash support | **DeepSeek-V4.1-Flash** (not yet run with V4 / 0731 models) |

## `v4-fixes`: what it fixes

At default flags on two 32 GiB cards, the server ran fine until the first long streamed request and then emitted
fluent token soup for the life of the process: no HIP error, no abort, draft acceptance pinned at 0 %. Two things
combined. The slab left ~122 MiB of VRAM free instead of the intended 262, because the shadow slots and codebook
were allocated after the budget. And the 51 GiB host expert pool was registered with the GPUs on 4 KiB pages and
only then hinted `MADV_HUGEPAGE`, so khugepaged collapsed pages under a live GPU mapping, and with little free
VRAM the page-table rebuild left stale entries. Either fix alone clears it (10/10 clean runs each, against 9/9
poisoned before).

| commit | change |
|---|---|
| `f87aaac` | placement: shadow slots, codebook and pad counted inside the slab budget |
| `2882edf` | host pool: `MADV_HUGEPAGE` **before** populating, so it registers on 2 MiB pages (`AFF_POOL_THP=0` = upstream order) |
| `0dad822` | sampler: never return an out-of-range token id on inverse-CDF drift |
| `a4f1fec` | engine: clear the compressor/indexer history rings on every sequence reset |
| `1f00675` | gpu: check every kernel launch (`hipGetLastError`) |
| `6dc42ae` | placement: swallowed HIP failures in the slab load are fatal (upstream PR #5) |
| `9585712`, `921ca87`, `f369a38`, `a12ee7b`, `a473c28` | opt-in verifiers: `AFF_MALLOC_TRACE`, `AFF_WEIGHT_VERIFY`, `AFF_PLC_VERIFY`, `AFF_MOVE_VERIFY` |

The first five are also offered upstream on Codeberg (issue #6, PRs #7–#11).

## `v41`: DeepSeek-V4.1-Flash

V4.1 adds a two-level sparse-attention indexer, engram n-gram layers (1 and 14, with 91.6 GiB of lookup tables
read from the checkpoint), 384 experts, and a new chat encoding. The port is correct against the reference model
(a CPU reference harness is in `tools/v41/`). Four bugs produced fluent text that ignored the prompt until they
were fixed; every V4.1 number measured before them is void:

| commit | bug |
|---|---|
| `e03a05b` | the engine applied V4's per-head RMS on q; V4.1 has none |
| `774aa6a` | the engram gate indexed the residual with the token count, not the batch stride |
| `4165fce` | engram `q_weight`/`k_weight` are bf16 and were read as f32 |
| `4bc938b` | engram buffers were allocated after placement and ate the driver reserve (OOM at larger prefill chunks) |

Plus `e6070f5` (V4.1 reasoning effort and `<｜System｜>` token, byte-identical to the reference `encoding.py`),
`2dbb093` (stop retrying a KV growth the card cannot hold, the 64K decode cliff) and `5252ba5` (re-seed the engram
look-back when a prefix-cache checkpoint is restored).

**More than two ranks is refused** (`AFF_ALLOW_NRANK=1` overrides it for debugging only). A fix for the 4-rank
split is being validated on four real CUDA cards. The `AFF_LOGICAL_RANKS` oversubscription harness (several ranks
per card) was found non-deterministic on CUDA, so a two-card box cannot verify it.

## Measured (two R9700s, container below, 2026-09-22/23)

| model | branch | container | ctx | prefill | decode | quality |
|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash-0731 | `v4-fixes` | 101 GiB + 7.0 GiB draft (bundled DSpark) | 262K | 1,264 / 1,356 / 1,324 t/s @4K/16K/64K | 50.6 / 49.8 / 40.5 t/s (192-token long answers); 31–35 t/s on free-form prose and sustained reasoning | needle 3/3 @4K/16K/64K; 44/47 on an internal 47-item pharmacovigilance benchmark at temp 1.0 / top_p 1.0; 7/7 on a 15-turn agentic tool-use task |
| DeepSeek-V4-Flash | `v4-fixes` | 101 GiB + draft | 262K | 1,354 @3.9K, 1,563 @10.9K, ~1,150 @32K | 25–50 t/s (draft-dependent) | 45/47 on the same benchmark; passed a 65-turn agentic conversion task |
| DeepSeek-V4.1-Flash | `v41` | 191 GiB, engram on | 131K | ~1,000 t/s @16K–48K | ~14 t/s | needle 3/3 @4K/16K/64K; no full benchmark yet |

Decode tracks the DSpark draft's acceptance rate: predictable output (code, tool calls, tables) is fastest,
open-ended prose slowest.

Observed with DeepSeek-V4-Flash-0731 at this quant: **temperature 0.6 makes it fall into reasoning loops**
(runs to the token cap at >90 % draft acceptance, the signature of repetition). Use the model card's sampling.

## Build

`docker/Dockerfile` builds the engine on AMD's TheRock ROCm 7.14 image for gfx1201; the host needs no ROCm.

```sh
docker build -f docker/Dockerfile -t affinity:v4-fixes .    # from the repo root, on the branch you want
```

## Quantise

`aff-quantize` (built alongside the engine; a CPU job, no GPU needed) converts the Hugging Face FP8 checkpoint
into an affinity container. **Only the routed experts are quantised**: 2.875 bpw with the engine's codebook quant
(2 variants x 1,024 entries), imatrix-weighted. Dense weights stay in the checkpoint's FP8 (6.2 GiB) and the
embedding / head in bf16 (2.0 GiB). No quantised weights are published here; you build your own container.

**DeepSeek-V4-Flash and V4-Flash-0731** (single machine):
```sh
# verify the download first: every shard's sha256 against the Hugging Face API (lfs.sha256)
aff-quantize --src <checkpoint> --out model-0731.aff --imatrix <imatrix> --threads 24
aff-quantize --src <checkpoint> --out model-0731.dspark.aff --dspark --codebook model-0731.aff
aff-info model-0731.aff
```
- Imatrix: a wikitext imatrix for the original V4-Flash. 0731 is the same architecture, so the V4 imatrix was
  reused as-is.
- Cost: main container **101.05 GiB in ~2 h 30 min** (1.4 experts/s, 4.7 GB RSS, 24 threads); draft **7.0 GiB in
  11 min**. The draft is 0731's own bundled 3-layer DSpark head, quantised against the main container's codebook.
  Disk: ~156 GiB checkpoint + ~108 GiB output.

**DeepSeek-V4.1-Flash** (`v41`: 15,360 experts; about 5 h on one machine, or split by layer across several):
```sh
aff-quantize --src <ckpt> --cb-spread --codebook-out v41cb.raw                 # codebook, once (~20 min)
aff-quantize --src <ckpt> --codebook-raw v41cb.raw --imatrix <imatrix.gguf> \
             --experts-dir experts --layer-range 0:40 --threads N             # experts; split --layer-range across machines
aff-quantize --src <ckpt> --dspark --codebook-raw v41cb.raw --experts-dir experts-dspark
aff-quantize --src <ckpt> --codebook-raw v41cb.raw --imatrix <imatrix.gguf> --experts-from experts --out model.aff
aff-quantize --src <ckpt> --dspark --codebook-raw v41cb.raw --experts-from experts-dspark --out model.dspark.aff
python3 tools/v41/engram_prep.py <ckpt> model.engram                           # engram sidecar (~3 s; needs tokenizers)
```
`tools/v41/worker.sh` drives the distributed expert encode. Every worker must use the same binary, codebook and
imatrix. Result: `model.aff` 191.32 GiB, `model.dspark.aff` 5.22 GiB. Checks that should all pass:
`aff-info`, `aff-cfgcheck`, `aff-fp8check model.aff <ckpt>` (FP8 tensors bit-faithful), `aff-exphash | sort -u`
(15,360 distinct experts), and `aff-gpucheck --dq` (GPU vs CPU on the real bytes).

## Run (DeepSeek-V4-Flash-0731, two cards, OpenAI-compatible server)

```sh
docker run --rm --device=/dev/kfd --device=/dev/dri --security-opt label=disable --network host \
  --ipc host --ulimit memlock=-1:-1 --shm-size 16g -e HIP_VISIBLE_DEVICES=0,1 \
  -v /path/to/models:/aff affinity:v4-fixes \
    -m /aff/model-0731.aff --tokenizer /aff/DeepSeek-V4-Flash-0731/tokenizer.json \
    --gpus 2 --host-pool-mib 56000 --dspark on \
    --kv-size 262144 --kv-commit 262144 \
    --ui plain --host 127.0.0.1 --port 8080
```

- `--security-opt label=disable` is required on SELinux hosts, or the model read is denied.
- `--kv-commit` equal to `--kv-size` backs the whole context at load. Growing it mid-request instead comes out of
  the small driver reserve.
- Thinking is on by default, and the server raises any request's `max_tokens` to at least 256,000 so reasoning
  is never cut off. Pass `--max-tokens-floor 0` if clients must be able to cap output, e.g. for benchmarks.
- Sampling defaults when a request sends none: temperature 1.0, top_p 0.95.

DeepSeek-V4.1-Flash (`v41`): `-m model.aff --engram model.engram --engram-tables <checkpoint dir>
--kv-size 131072 --kv-commit 131072 --prefill-chunk-size 2816 --host-pool-mib 138000 --dspark off`, with the
checkpoint mounted read-only (the engram tables are read from it).
