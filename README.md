# Affinity

An inference engine for DeepSeek-V4-Flash on **one or two** RDNA4 GPUs. Single C++/HIP codebase, no
framework.

The model is 284 B parameters and does not fit in VRAM. At 2.875 bpw its routed experts are ~93 GiB
against the ~64 GiB two 32 GiB cards give you, or half that on one, so on every token some of the
experts a layer needs are in host RAM. An expert on a card reads at ~630 GB/s; one in host RAM
crosses PCIe at about a tenth of that. Decode speed is therefore mostly one question: how often is
the expert this token wants already resident? The engine is built around moving experts to keep that
number high.

## Requirements

| | |
|---|---|
| GPU | **One or two AMD RDNA4 `gfx1201` cards.** Radeon AI PRO R9700 (32 GiB) is the reference card; two of them is the design point |
| Runtime | ROCm 7.x with `hipcc`. Two cards additionally need peer access between them |
| RAM | 64 GB. The non-resident experts live here; more is better, and on one card it is what makes the model runnable at all |
| Disk | ~107 GiB container, ~7 GiB draft, plus the source checkpoint |
| Build | CMake 3.24+, C++20 |

The kernels assume RDNA4 wave32 and its WMMA fragment layout; the configure refuses anything that is
not `gfx12xx`.

### Card count

**Two cards** is the design point and what every number in this README was measured on. Both cards
are tensor-parallel ranks: every GPU op except the routed experts is split across them, each holds
half of every resident expert, and the two exchange through a peer-push all-reduce.

**One card** is supported and runs the whole model with no collective at all. It is worth having if
you have the RAM, since the experts that do not fit spill to the host pool and then to the container
on disk. It works, and it is slow: on a 32 GiB card with 64 GB of RAM, roughly 8 tok/s against 110
on two cards, because most of the experts end up on disk. More RAM is the fix; the pool is what
keeps them off the SSD tier.

Every `gfx1201` card the runtime reports becomes a rank, so on a machine with exactly one or two
there is nothing to configure. `--gpus N` caps the count and `HIP_VISIBLE_DEVICES` chooses which.

**Three or more cards is refused at startup.** Nothing in the engine is written for two in
particular. The rank count is a runtime value throughout, and the pieces that were most likely to
break are tested past two: the N-way expert split byte-for-byte and arithmetically at four ways, the
all-reduce bit-identical across ranks at 3, 4 and 8, attention at every per-card head count. The
engine as a whole still emits degenerate text there, and the cause is not yet known. Refusing beats
shipping a configuration whose output is fluent and wrong. On a bigger box, run it with `--gpus 2`.

A CPU-only configure builds the quantiser but not `affinity`.

## 1. Build

```sh
git clone https://codeberg.org/StillDeadcode/affinity.git && cd affinity
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Builds `affinity`, `aff-quantize`, `aff-info` and `aff-gpucheck`.

| Option | Default | |
|---|---|---|
| `-DAFF_HIP=ON\|OFF\|AUTO` | `AUTO` | GPU kernels; `AUTO` enables them if `hipcc` is found |
| `-DAFF_GPU_ARCH=` | `gfx1201` | Target architecture |
| `-DAFF_NATIVE=ON\|OFF` | `ON` | `-march=native` |

## 2. Get the checkpoint

```sh
pip install -U "huggingface_hub[cli]"
hf download deepseek-ai/DeepSeek-V4-Flash --local-dir ./DeepSeek-V4-Flash
```

You need the whole directory: `config.json`, the safetensors shards and `tokenizer.json`. The
speculative-decode draft is in the same checkpoint (the `mtp.*` tensors), not a separate download.

Then an importance matrix. This is not ours. It is a llama.cpp GGUF imatrix published alongside a
community GGUF of the same checkpoint:

```sh
hf download teamblobfish/DeepSeek-V4-Flash-GGUF imatrix/imatrix-v4-flash.dat --local-dir .
```

470 MB, calibrated on wikitext (2000 × 512-token chunks). Any llama.cpp imatrix for this model works
as long as it carries **per-expert** `in_sum2` and `counts`. For MoE tensors llama.cpp writes one row
per expert, and both jobs below need that per-expert form rather than a tensor-wide average.

It does two jobs, and it is honest to say both are modest.

**It weights the quantiser's error.** `in_sum2/counts` is `E[x_j²]` per input column per expert, which
turns the rounding objective from plain L2 into a weighted one, so precision goes to the columns the
model actually drives hard. Worth about 2% lower error under the activation distribution the imatrix
itself measures. This is the job that justifies the download.

**Its activation counts become the container's expert-popularity profile**, which the placement
engine ranks against to choose which experts start resident. Against no profile at all that is worth
1.2 points of hit rate. Imatrix popularity is not decode popularity, the calibration corpus is not
your workload, and the heat engine re-ranks from real routing within a few dispatches regardless. It
is a warm start, not a policy, and `--expert-profile` overrides it with one fitted from real traces.

Both are optional. Quantising without `--imatrix` gives you an unweighted container and no profile;
it works. `--imatrix` is ignored under `--dspark`, which is keyed by target layer.

## 3. Quantise

```sh
./build/aff-quantize --src ./DeepSeek-V4-Flash --out model.aff \
                     --imatrix imatrix/imatrix-v4-flash.dat --threads 12
```

Produces 106.6 GiB: 92.9 GiB of routed experts, ~11.6 GiB of dense weights kept in the checkpoint's
own FP8, 2.0 GiB of embedding and head. Around three hours at twelve threads. The output is a pure
function of the checkpoint and the imatrix, so the same inputs give a byte-identical file at any
`--threads`. It exits non-zero if any expert fails to write, because a partial container loads
cleanly and decodes garbage.

Then the draft, which must share the main container's codebook:

```sh
./build/aff-quantize --src ./DeepSeek-V4-Flash --out model.dspark.aff \
                     --dspark --codebook model.aff
```

~6.8 GiB. Skipping it is fine: run with `--dspark off`, which gives the draft's VRAM to the experts.

| `aff-quantize` | |
|---|---|
| `--src DIR` | Checkpoint directory |
| `--out FILE` | Container to write |
| `--imatrix FILE` | Importance weights, llama.cpp format |
| `--layers N` | First N layers only. Exercises the pipeline; the output is not the model |
| `--threads N` | Default one per core. Does not affect the output |
| `--dspark` | Quantise the `mtp.*` draft stages instead |
| `--codebook FILE` | Take the codebook from this container. Required with `--dspark` |

## 4. Run

```sh
./build/aff-info model.aff                    # check the container first

./build/affinity -m model.aff --tokenizer ./DeepSeek-V4-Flash/tokenizer.json \
                 -p 'The capital of France is' -n 32

./build/affinity -m model.aff --tokenizer ./DeepSeek-V4-Flash/tokenizer.json --port 8080
```

The server does `/v1/chat/completions` (streaming and not), `/v1/completions` and `/v1/models`, with
tool calls in and out. `--dspark-model` defaults to `<model>.dspark.aff`, so step 3's draft is picked
up automatically.

If output looks wrong, `aff-gpucheck -m model.aff` runs the GPU expert path against the CPU
reference on the real container.

On a terminal both modes draw a live view: per-card utilisation and VRAM, throughput, draft
acceptance, and the expert plane coloured by tier. Piped or redirected it prints ordinary lines
instead, unchanged. `--ui plain` forces lines on a terminal; `--ui dash` fails rather than downgrade.

## Configuration

`--help` prints all of this. Defaults in brackets.

**Model and memory**

| | |
|---|---|
| `-m, --model FILE` | The `.aff` container. Required |
| `--tokenizer FILE` | `tokenizer.json` |
| `--gpus N` | [0 = every supported card] Tensor-parallel ranks, 1 or 2. `HIP_VISIBLE_DEVICES` chooses which cards; this takes a prefix of them |
| `--dspark on\|off` | [on] Speculative decode. Off does not load the draft and gives its VRAM to the experts; the text differs between the two |
| `--dspark-model FILE` | [`<model>.dspark.aff`] |
| `--dspark-placement vram\|ram` | [vram] `ram` pools the draft's slab. A trade, not a saving: the draft rereads its experts several times a block |
| `--gpu-mib M` | Cap on VRAM per card for the expert slabs. Default exceeds any card, so free VRAM binds |
| `--host-pool-mib M` | RAM for the non-VRAM experts [from MemAvailable]. Pin it for anything you compare |
| `--gpu-headroom-mib M` | [0 = derive] VRAM the slab leaves alone. Too small and the slab's own allocation fails |
| `--host-pool-reserve-mib M` | [1536] RAM left free when auto-sizing. What does not fit the pool falls to SSD, and a layer with even one expert on SSD reverts to the slower host dispatch |
| `--kv-size N` | [1048576] KV positions. Address space, not VRAM; lowering it does not buy residency |
| `--kv-commit N` | [65536] Positions actually backed at load. The rest costs nothing until reached, and the difference goes to the expert slab |
| `--kv-margin N` | [16384] How far ahead the grower keeps the cache backed |

**Prefix cache.** Keeps served attention state on SSD so an agent session does not re-prefill its
whole transcript every turn.

| | |
|---|---|
| `--prefix-cache-dir D` | Off unless given. `D` needs `O_DIRECT` |
| `--prefix-cache-gib N` | [from free space, max 64] |
| `--prefix-cache-block N` | [512] Tokens a block. Must divide by the largest compress ratio |
| `--persistent-prefix-cache` | Keep the store across restarts. **It holds conversation content on disk unencrypted** |
| `--prefix-cache-verify` | Hash the attention state once the prompt is in. A check, not a serving mode |

**Serving**

| | |
|---|---|
| `--host ADDR` | [127.0.0.1] |
| `--port N` | [8080] |
| `--max-tokens-floor N` | [256000] Raise a request's `max_tokens` to at least this. Reasoning and answer share one budget, so a small hardcoded value truncates reasoning and returns no answer. 0 honours the caller |

**Generation**

| | |
|---|---|
| `-p, --prompt TEXT` | Generate once to stdout instead of serving |
| `--prompt-file F` | Needed past ~30K tokens, where `-p "$(cat F)"` hits `ARG_MAX` |
| `-n, --n-predict N` | [32] |
| `--ignore-eos` | Keep going past EOS |
| `--no-echo` | Do not echo the prompt |
| `--ui auto\|dash\|plain` | [auto] |
| `--temp F` | [0, greedy] Greedy degenerates and changes which experts route; use `--seed` and real settings |
| `--top-k N` | [0 = off] |
| `--top-p F` | [1.0] |
| `--min-p F` | [0] |
| `--seed N` | Reproducible only under `--placement-strategy static`: the mover changes residency, and residency sets the summation order |

**Performance**

| | |
|---|---|
| `--prefill-chunk-size N` | [1408] Tokens a prefill chunk, rounded to 128. Bigger amortises the experts further but leaves fewer resident |
| `--placement-strategy S` | [heat] `heat`, `demand` (promotes what the dispatch just streamed) or `static` (nothing moves) |
| `--placement-moves N` | [8] Swaps started a layer dispatch |
| `--placement-decay F` | [0.98] Heat decay a dispatch |
| `--placement-hysteresis F` | [1.25] Promote only at this heat ratio |
| `--placement-min-gain F` | [1.5] …and this many activations. Link traffic is U-shaped in it; raising it moves more bytes, not fewer |
| `--placement-lead N` | [2] Hold the host within N layer dispatches of the cards. 0 is unbounded, which starves the mover of slots to move into. Decode only |
| `--placement-shadow N` | [32] Spare slab slots for in-flight promotions |
| `--keepalive-us U` | [0 = off] Per-card heartbeat |

**Diagnostics.** Several of these deliberately produce output that is not the model's.

| | |
|---|---|
| `--prefill-reps N` | Re-run the prefill N times in one process, median and spread. Placement converges across reps, so one prefill is a cold number |
| `--prefill-alt FILE` | Rotate a second prompt through `--prefill-reps` |
| `--route-trace FILE` | Record every layer dispatch. Format at the top of `src/engine/route_trace.h`. Forces the host dispatch, so its tok/s is not the shipping rate |
| `--expert-profile FILE` | Use this popularity profile instead of the container's |
| `--dump-logits K` / `--tokens a,b,c` / `--logits-out F` | Top-K logits for explicit token ids |
| `--dump-tokens` | Print the token ids for `-p` and exit |
| `--force-tokens a,b,c` | Decode these ids instead of the sampler's. Needed to compare two builds at all |
| `--no-routed-experts` | Zero the routed experts to isolate the dense path |
| `--index-topk K` | Override the indexer's top-k |
| `--max-layers N` | Stop after N layers |

**Environment**

| | |
|---|---|
| `AFF_PROFILE=1` | Per-phase breakdown. **Not a passive observer.** It restores a readback and the drain behind it, which changes the mover's slot supply. Do not compare a profiled run against an unprofiled one |
| `AFF_BLOCK=1` | The host blocking table alone, without that readback |
| `AFF_FORCE_RESIDENT=1` | Restrict routing to resident experts. Measures the all-resident ceiling; the output is wrong on purpose |
| `AFF_TRACE=dense[,events][,deep][,sync][,evsync]` | Per-entry-point device timing |
| `AFF_MOVER_DUP=K` / `AFF_MOVER_DUP_DIR=both\|h2d\|d2h` | Re-issue each mover copy K times. Same decisions and same text, K times the link bytes. Prices a move without changing what the engine does |
| `AFF_XRANK_INJECT=N` | Perturb one rank's cross-rank digest, so the check comparing the cards can be shown to fire |

## How it works

Every routed expert is in one of three places: a flat per-card **VRAM slab**, a pinned **host pool**,
or left in the container's mmap on **SSD**. A missing expert is not copied into VRAM and then
multiplied. The GEMM reads it straight out of host memory over PCIe while it runs, so the transfer
is inside the kernel rather than in front of it. The SSD tier is a cliff; size `--host-pool-mib` to
avoid it.

A placement engine moves experts between slab and pool while decoding. It keeps a heat per expert
(LFU with decay, credited per distinct expert per dispatch and weighted by how many of the block's
tokens chose it) and swaps the hottest pooled expert for the coldest resident one when the gain
clears the two guards. The ranking is global rather than per layer: a slab slot is layer-agnostic,
so capacity moves to the layers that need it.

Two things about that are not obvious. A swap moves two shards and a miss moves one, and the two
PCIe directions share one budget, so a promotion has to repay about 1.4 future hits, which is why
total link traffic is U-shaped in `--placement-min-gain`. And what usually limits the mover is not
the policy but slot supply: a freed slot returns to the free list only when its quarantine event
signals, and that event is on the compute stream, so an unbounded host lead leaves the mover with
nowhere to move into. That is what `--placement-lead` exists for.

The rest: every GPU op except the routed experts is tensor-parallel across every card, with a
one-shot peer-push all-reduce. Dense weights keep the checkpoint's own FP8 rather than being
dequantised, because VRAM is the scarce resource. Attention is MLA over a compressed latent KV at
8.5 KB a token, which is why a 1M-position cache costs ~4.3 GiB a card instead of tens of gigabytes,
and above ~2048 positions an indexer picks which positions each query attends to. Speculative decode
runs the checkpoint's own MTP stages as a draft, verified in one batched pass with acceptance
checked on the device.

## Limitations

- DeepSeek-V4-Flash only. The geometry is compiled in; there is no architecture abstraction.
- One or two cards; more is refused at startup. See Requirements.
- Batch 1 to 4, latency-first. No continuous batching and no paged scheduler: batching an MoE that
  reads weights from host RAM makes the tokens in a batch pay for the union of their experts.
- Throughput depends heavily on the prompt, because which experts a token routes to decides how much
  host memory the step reads. Two prompts differ by more than most code changes do, so compare
  builds with `--force-tokens` and quote milliseconds a block rather than tok/s.
- An expert that fits neither VRAM nor the host pool falls to an SSD tier with no dispatch path, and
  routing to one aborts. Pin `--host-pool-mib` rather than trusting the auto-size.

## Acknowledgements

The all-reduce is adapted from `vllm-radiance`'s `radiance_ar_ext`, and the FP8 weight preshuffle
follows its `shuffle_weight` layout.

## License

MIT. See [`LICENSE`](LICENSE).
