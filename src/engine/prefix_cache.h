// A prefix cache for the attention state, backed by the SSD.
//
// An agent session re-sends its whole transcript every turn, so without a cache the engine
// re-prefills every earlier turn on every request and a session costs O(N^2) prefill. The render is
// append-only in the configuration this engine is built for — a conversation carrying tools keeps
// every turn's reasoning, so nothing before the new suffix is ever rewritten — which means the
// previous request's state is exactly the prefix of this one's.
//
// ---- why the SSD and not RAM -------------------------------------------------------------------
//
// There is no host RAM to put it in. The expert pool takes all but a few GiB, and the pool is what
// decides decode speed, so a RAM-backed cache would be paid for out of residency. The SSD is not a
// compromise here: restoring is a linear read of a few GiB against a prefill that is quadratic in
// the transcript, and the read wins by three orders of magnitude at any context worth caching.
//
// ---- what is stored, and why it is two things --------------------------------------------------
//
// The device state splits cleanly (see DenseGpu::KvPart):
//
//   BLOCK       The append-only rows — compressed KV, indexer keys and their scales. Row g is
//               produced by a fixed span of positions and never rewritten, so two conversations
//               that share a token prefix produce these bytes identically. They are therefore
//               content-addressed and shared, exactly as vLLM shares its KV blocks.
//
//   CHECKPOINT  The rolling rings — the sliding raw window and the compressor's cross-chunk
//               windows. They describe ONE position and cannot be accumulated, so they are stored
//               whole. This is the part a paged cache does not need and this engine does.
//
// Restoring to position B is then: load every block below B, load the checkpoint at B, set the
// counts, and prefill the rest of the prompt as usual. There is no separate replay path — the
// suffix prefill IS the replay.
//
// A resume point therefore exists exactly where some request ENDED, because only there were the
// rings ever saved. That is what makes the agent loop free — each turn resumes at the previous
// turn's last token — and it is also the shape of rollback: returning to any earlier TURN is exact
// and costs nothing, while returning to an arbitrary token inside a turn costs a re-prefill from
// the turn boundary below it. Blocks below a resume point are still shared between conversations
// and still worth keeping; they just cannot be resumed from on their own.
//
// ---- the index ---------------------------------------------------------------------------------
//
// vLLM's, transcribed: a block's key is SHA-256 over its parent's key and its own token ids, so a
// key names an entire prefix and not just a span; only whole blocks are stored; and eviction is LRU
// over a doubly-linked free list with sentinels, freeing a request's blocks from the tail backwards
// because the last block of a request hashes the most tokens and is the least likely to be reused.
//
// Two things are added to it. A block records its own token ids and they are compared on load, so a
// hash collision cannot serve one conversation another's context — probability is not a good enough
// argument when the store is persistent and shared across sessions. And every key is salted with a
// fingerprint of the model and cache geometry, so two containers, two KV formats or two block sizes
// can never read each other's rows.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace aff {

// One (part, layer) buffer the engine holds, as the cache sees it: opaque rows. `part` is a
// DenseGpu::KvPart and is only ever handed straight back to the callbacks, which is what keeps this
// file free of HIP and testable without a card.
struct KvRegion {
  uint8_t  part = 0;
  uint32_t layer = 0;
  uint64_t row_bytes = 0;
  uint64_t rows = 0;         // what the buffer holds, i.e. the ceiling on row0 + nrows
  // Positions per row, for the append-only parts: block [t0, t1) owns rows [t0/ratio, t1/ratio).
  // 0 marks a rolling ring, which belongs to the checkpoint and is always stored whole.
  uint32_t ratio = 0;
};

// The device side. Both move rows [row0, row0+nrows) of (part, layer) on ONE rank, and return false
// on any refusal — there is no partial transfer to recover from.
//
// ---- ONE RANK IS STORED AND EVERY RANK IS RESTORED FROM IT ----------------------------------------
//
// The replicated state is bit-identical across cards, checked two ways: `xrank_check` samples it on
// the shipping path and aborts on a mismatch, and `--prefix-cache-verify` hashes every region per
// rank and reports the cross-rank splits per request. So the store holds rank 0 and hands the same
// bytes to every card, which divides the file, the restore I/O and the D2H a publish costs by the
// rank count.
//
// If the ranks ever part again this becomes a silent, valid-looking state change that moves the
// emitted text. That is what `xrank_check` is for, and it is fatal rather than reported.
struct KvIo {
  bool (*save)(void* ctx, uint8_t part, uint32_t layer, uint32_t rank, uint64_t row0,
               uint64_t nrows, void* dst) = nullptr;
  bool (*load)(void* ctx, uint8_t part, uint32_t layer, uint32_t rank, uint64_t row0,
               uint64_t nrows, const void* src) = nullptr;
  void* ctx = nullptr;
};

struct PrefixCacheStats {
  uint64_t lookups = 0, hits = 0;
  uint64_t tokens_restored = 0, tokens_prefilled = 0;
  uint64_t blocks_written = 0, blocks_evicted = 0, ckpts_written = 0;
  uint64_t bytes_read = 0, bytes_written = 0;
  uint64_t verify_misses = 0;      // key matched, token ids did not — a collision, caught
};

class PrefixCache {
 public:
  struct Options {
    std::string dir;                  // empty disables the cache entirely
    uint64_t    budget_bytes = 0;     // total on-disk footprint, blocks and checkpoints together
    uint32_t    block_tokens = 512;   // must be a positive multiple of the largest compress ratio
    bool        persistent = false;   // survive process exit
    uint32_t    queue_depth = 4;      // staged publishes in flight
    // Of the budget, the share held back for checkpoints. Blocks are shared between conversations
    // and checkpoints are not, so most of the space belongs to blocks; this only has to keep enough
    // resume points that the sessions in flight each have one.
    uint32_t    ckpt_percent = 25;
    uint32_t    ranks = 1;            // cards to restore to; ONE of them is stored — see KvIo
  };

  PrefixCache() = default;
  ~PrefixCache();
  PrefixCache(const PrefixCache&) = delete;
  PrefixCache& operator=(const PrefixCache&) = delete;

  // `regions` is every buffer the engine holds. `fingerprint` must cover everything that decides
  // the bytes — container, KV dtype, widths, sliding window, ratio schedule — because a key is
  // computed from token ids alone and nothing else would stop two geometries sharing a block.
  //
  // Returns false with a reason and leaves the cache closed. There is no degraded mode: a cache
  // that half-opened would restore some regions and not others, which is silent corruption.
  bool open(const Options& o, const std::vector<KvRegion>& regions, const KvIo& io,
            uint64_t fingerprint, std::string* err);
  void close();
  bool ready() const { return ready_; }

  // Tokens of `ids` that the store holds a resume point for — always a multiple of block_tokens,
  // and 0 when there is nothing to reuse. Reads the index only; no I/O.
  // `blocks_matched`, when given, receives how many leading blocks the store held. It separates
  // the two ways a lookup returns 0 — a prompt that shares no block at all, and one that shares
  // many but ends at no resume point — which look identical from the hit length alone.
  uint32_t lookup(const uint32_t* ids, uint32_t n, uint32_t* blocks_matched = nullptr) const;

  // Restore the state for the first `n_restore` tokens, which must be a value `lookup` returned for
  // the same ids. On success the engine is at position `n_restore` and the caller prefills the
  // rest. On failure nothing is left half-restored: the caller must prefill from 0.
  bool restore(const uint32_t* ids, uint32_t n_restore, std::string* err);

  // Offer positions [0, n) to the store: every full block that is not already held, and a
  // checkpoint at the last block boundary so the next turn can resume there.
  //
  // The device reads happen here, on the caller's thread, because they have to be ordered against
  // the kernels that produced the rows. Hashing, eviction and the writes are handed to the writer
  // thread, so what the caller pays is one D2H and nothing else.
  bool publish(const uint32_t* ids, uint32_t n, std::string* err);

  // Block until the writer has drained. For tests and for shutdown.
  void drain();

  const PrefixCacheStats& stats() const { return st_; }
  uint32_t block_tokens() const { return opt_.block_tokens; }
  // Bytes one block and one checkpoint occupy on disk, payload and header together.
  uint64_t block_bytes() const { return blk_slot_; }
  uint64_t ckpt_bytes() const { return ck_slot_; }
  uint32_t block_slots() const { return (uint32_t)blk_.size(); }
  uint32_t ckpt_slots() const { return (uint32_t)ck_.size(); }

 private:
  struct Impl;
  Impl* impl_ = nullptr;

  Options opt_{};
  KvIo    io_{};
  std::vector<KvRegion> app_;    // append-only, in block order
  std::vector<KvRegion> roll_;   // rolling, in checkpoint order
  uint64_t fp_ = 0;
  // Ranks whose rows are actually on disk. One, and the reason is in KvIo's header; it is a named
  // constant rather than a literal so the two places that would have to change together — the
  // layout and the restore fan-out — cannot drift.
  static constexpr uint32_t kSavedRanks = 1;
  uint64_t blk_payload_ = 0, ck_payload_ = 0;   // bytes of rows
  uint64_t blk_slot_ = 0, ck_slot_ = 0;         // header + payload, rounded to the I/O alignment
  bool ready_ = false;
  PrefixCacheStats st_{};

  // Slot tables, preallocated at open() and never resized — see the engine's no-allocation rule.
  //
  // Written to the index file verbatim, so it is a plain aggregate on purpose: `prev`/`next` thread
  // the eviction list and `hnext` the hash chain, both as indices, so nothing in a saved slot points
  // at an address that will not exist next time.
  static constexpr uint32_t kNoSlot = 0xFFFFFFFFu;
  struct Slot {
    uint8_t  key[32] = {};
    uint8_t  parent[32] = {};
    uint32_t prev = kNoSlot, next = kNoSlot;
    uint32_t hnext = kNoSlot;
    uint32_t n_tokens = 0;
    uint32_t pos = 0;
    bool     used = false;
  };
  std::vector<Slot> blk_, ck_;

  bool  build_layout(std::string* err);
  bool  open_files(std::string* err);
  bool  load_index();
  bool  save_index() const;
  // Fills `keys` with the chained key of every full block of `ids`, and returns how many there are.
  uint32_t chain(const uint32_t* ids, uint32_t n, std::vector<uint8_t>* keys) const;
  int32_t  find(const uint8_t* key) const;
  // The longest checkpoint hanging off `parent` whose tail is a prefix of `ids[0, avail)`. Several
  // can share a parent — one per turn of the same conversation — so the chain is walked rather than
  // probed, and a candidate is confirmed by recomputing its key over this prompt's own tokens,
  // which verifies the tail without reading the slot back off the disk.
  int32_t  find_ckpt(const uint8_t* parent, const uint32_t* ids, uint32_t avail,
                     uint32_t* tail) const;
  void     touch(int32_t slot, bool ckpt);
  // Take a slot back out of the index and put it at the front of the free list. Used when a slot's
  // payload turns out not to match its key — a short write, a truncated file, an index that
  // outlived its data. Without it a single bad slot poisons its prefix PERMANENTLY: `find` keeps
  // matching the key, `restore` keeps failing verification, and every future request through that
  // prefix prefills from scratch while reporting a hit.
  void     drop(int32_t slot, bool ckpt);
  int32_t  claim(bool ckpt);
  void     thread_free_lists();
  uint32_t acquire_stage();
  void     release_stage(uint32_t buf);
};

}  // namespace aff
