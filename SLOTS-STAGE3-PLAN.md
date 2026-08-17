# slots(3/3) — implementation plan: interleaved execution

## The finding that shapes this

Latency isolation does **not** need the flash-kernel changes. The serve path touches the device
at only four points (`tools/affinity.cpp`, the `gen_` lambda):

| entry point | what it does | interleave granularity |
|---|---|---|
| `forward_prefill` (prefix-cache promote) | chunked prefill | per chunk (driver must loop) |
| `forward_prefill` (main prefill) | chunked prefill | per chunk (driver must loop) |
| `spec_decode(...)` | loops DSpark blocks internally | **per block — already a loop** |
| `kv_reserve_for` | KV growth (no-op at full commit) | n/a |

Today `server.cpp` holds `gen_mu_` across the **whole generation**. Moving the lock to a single
**engine step** (one prefill chunk, or one DSpark block) lets two request threads interleave:
each already runs on its own thread ("one thread per connection"), and each has its own
`SeqState` on the host and — since slots(2/3) — its own KV, compressor window and indexer keys
on the device.

Crucially, the **per-call scratch stays shared and safe** (`b_qtm`, `b_allow`, `i_sc`, the
prefill arena): only one step executes at a time, so nothing overlaps. That is why no kernel
change is required. Batched decode (real throughput multiplication) still needs per-token
`(slot,pos)`, but that is a *separate, later* milestone.

## What this delivers

- **Real concurrency in the sense that matters for a worker/planner lane**: a long request no
  longer blocks a short one. Two agents progress together, each at ~half decode speed.
- **Aggregate throughput unchanged** (~90 t/s on code) — steps are still serialized.
- Cost: none in VRAM (2x256K == 1x512K, measured), no kernel risk.

## Work items

1. **Slot pool** (`server.h/.cpp`): free-list of `n_slots` slots; a request acquires one on entry
   and releases on completion; block when none free (that is the natural admission control, and
   it caps concurrency at the slot count).
2. **Pass the slot to `gen_`**: extend `GenerateFn` with a slot index. Replace the
   whole-generation `gen_mu_` with a per-step `engine_mu_`.
3. **Driver-side chunked prefill**: loop `forward_prefill` over `prefill_chunk_size` tokens,
   taking `engine_mu_` + `set_slot()` per chunk, so a 256K prefill yields ~180 times instead of
   holding the engine for minutes.
4. **Per-block lock in `spec_decode`**: take `engine_mu_` + `set_slot()` around each block.
5. **Per-slot prefix cache**: the store is keyed to one live sequence today.
6. **Isolation test**: two sequences interleaved across slots 0/1; each must produce the same
   tokens as a solo reference run. This is the correctness gate — it is what proves a slot's KV
   is not being clobbered by its neighbour.

Estimated ~150 lines across `server.{h,cpp}` and `tools/affinity.cpp`, plus the test harness.
The risk is concentrated in (3) and (4) — everything else is bookkeeping.

## Then, separately, for throughput

7. per-token `pos_v[]`/`slot_v[]` through `batch_attend`/`kv_commit`/`batch_indexer` + flash
   kernel (base = `slot*stride`; the kernel already indirects per token for `n_comp_v` and
   admissions), and a gather step that collects N sequences into one forward. Projected
   ~1.5x (N=2) / ~2.1x (N=4) aggregate, per-request decode ~67 / ~46 t/s.
