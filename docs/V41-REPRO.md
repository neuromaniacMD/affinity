# DeepSeek-V4.1-Flash on affinity — how to reproduce it

Everything below was run on **amdnas** (Ryzen 9 7950X, 192 GB, 2× Radeon AI PRO R9700 32 GB + a W7900
that this engine cannot use — gfx1201 only) on 2026-09-22. The host has no ROCm; every build and run
goes through the TheRock container.

## 1. Inputs

| | |
|---|---|
| checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash`, revision `dba1be0a40aa45a94ad051997016db3960a90277`, 48 shards, ~510 GB, sha256-verified on download (`hf/dl-v41-capped.sh` in the workspace copy) |
| imatrix | `smalinin/DeepSeek-V4.1-Flash-GGUF` → `imatrix-dsv41-262144.gguf`, 789,154,336 B (262K tokens; its own corpus is undocumented) |
| engine | this branch, `v41` |

## 2. Quantise (≈2 h across three machines, or ~5 h on one)

```sh
# codebook, once: 8 experts spread over all layers, ~20 min single-threaded
aff-quantize --src <ckpt> --cb-spread --codebook-out v41cb.raw      # hash 539439d8cc1b65a3

# experts: every machine runs the SAME binary against the SAME codebook + imatrix.
# A layer's experts live in shard l+3, so a worker needs one shard at a time.
aff-quantize --src <ckpt> --codebook-raw v41cb.raw --imatrix <imatrix.gguf> \
             --experts-dir experts --layer-range 0:40 --threads N

# draft: 3 stages x 128 experts
aff-quantize --src <ckpt> --dspark --codebook-raw v41cb.raw --experts-dir experts-dspark

# assemble (minutes; re-runnable whenever the dense map changes, with no re-encode)
aff-quantize --src <ckpt> --codebook-raw v41cb.raw --imatrix <imatrix.gguf> \
             --experts-from experts --out model.aff
aff-quantize --src <ckpt> --dspark --codebook-raw v41cb.raw \
             --experts-from experts-dspark --out model.dspark.aff
```

`tools/v41/worker.sh` is the distributed driver (layers claimed by atomic `mkdir` on the hub; a
crashed worker's layer is picked up by another). Result: **model.aff 191.32 GiB**, 15,360 experts at
2.875 bpw; **model.dspark.aff 5.22 GiB**.

Checks, all of which passed:
```sh
aff-info model.aff                      # geometry + tensor directory
aff-cfgcheck model.aff                  # the parsed V4.1 layer schedule
aff-fp8check model.aff <ckpt>           # 330 FP8 tensors, worst rel 3.92e-06, 0 bad
aff-exphash model.aff | sort -u -k3 | wc -l   # 15360 distinct expert hashes
aff-gpucheck -m model.aff --experts 8 --layer 0 --dq   # GPU vs CPU on the real bytes: 2.4e-07
```

## 3. Build the engine

```sh
# build context = this repo minus .git, beside the Dockerfile
docker build -t local/affinity:v41 .        # tools/v41/Dockerfile, 54 steps, ~1 min warm
```
The image is `kyuz0/vllm-therock-gfx1201:rocm7.14.0-torch2.11.0-vllm0.27.1` + this source; hipcc comes
from the image's `_rocm_sdk_devel` tree (`AFF_HIPCC` must be passed, or CMake derives the wrong root).

## 4. Run

```sh
docker run --rm --device=/dev/kfd --device=/dev/dri --security-opt label=disable \
  --network host --ipc host --ulimit memlock=-1:-1 --shm-size 16g \
  -e HIP_VISIBLE_DEVICES=0,1 -v <dir>:/m -v <ckpt>:/ck:ro local/affinity:v41 \
  -m /m/model.aff --tokenizer /ck/tokenizer.json \
  --dspark off --host-pool-mib 138000 --host-pool-reserve-mib 512 --ui plain \
  --prompt-file /m/p4k.txt -n 24
```

⚠️ `--security-opt label=disable` is not optional on this SELinux host: without it the container's
read of the model is `Permission denied`.

⚠️ **Size the host pool.** 182 GiB of experts, ~46 in VRAM: anything under ~136 GiB of pool leaves
experts on the SSD tier, and a layer holding ONE addressless expert keeps the slow host dispatch for
that whole layer. It cost 2x decode (8.3 vs 17.7 tok/s) and the log says so if you read it.

## 5. What it does (2 cards, 136 GiB pool, greedy, `--dspark off`)

| prompt | prefill | decode |
|---|---|---|
| 19 tok | — (too short to mean anything) | 17.7 tok/s |
| 3,806 tok | **828.9 tok/s** (4.6 s) | 10.2 tok/s |
| 15,159 tok | **1,105.0 tok/s** (13.7 s) | 10.4 tok/s |

One card, for scale: 4.0 tok/s decode. Prefill improves with length (batching amortises), so a 16K
clinical document is ~14 s of prompt processing.

Sanity, same prompt every time: *"A patient taking warfarin develops a severe headache and bruising.
The most likely explanation is:"* → intracranial haemorrhage / intracerebral haemorrhage / subdural
haematoma. Clinically right, which is the cheapest evidence that 2.875-bit experts, the shared KV and
the routing are all behaving.

## 6. What is NOT done

- **The indexer and the candidate pre-filter.** V4.1 selects `index_topk`=512 compressed rows per
  query; this engine attends over all of them. Below ~1K tokens of context that is identical
  behaviour; above it the engine is doing MORE work than the reference and diverging from it. This is
  the next correctness item, and it is the one that matters for long documents.
- **Engram** (layers 1 and 14): the tables stay in the checkpoint and are never read. The model runs
  without them, but it is not the released model until they are.
- **The DSpark draft** loads and drafts, at 12 % acceptance where V4's gets 37-50 %, and costs 7.35
  GiB of expert slab a card — net 17.7 -> 6.9 tok/s, so it is off. The tap fix (V4.1 conditions its
  draft on the attention INPUT of layers 37-39, not their output) moved acceptance 10.8 -> 12.0 % and
  first-position hits 38 -> 48 %, so it was right but is not the whole story. Next suspects: the
  window index set `get_dspark_topk_idxs` builds, and the Markov/confidence heads.
- **Quality is ungated.** No KL against the FP4 teacher, no pv_bench, no agentic replay. The golden
  vectors for that are in `tools/v41/` and unused so far.

## 7. The reference harness (what settles disagreements)

`tools/v41/refblock.py` runs the checkpoint's own `model.py` on CPU with torch stand-ins for the
TileLang kernels (`tools/v41/kernel_shim.py`), loading real weights for one layer and dumping
`{x,pre_mix,out,ffn_pre}.npy`. Deterministic across runs. It is how the half-sublayer mix shift was
confirmed, and it is what any "which layer diverges" question should go through.
