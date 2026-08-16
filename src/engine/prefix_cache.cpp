#include "engine/prefix_cache.h"

#include "ui/log.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>

namespace aff {

namespace {

// ---- SHA-256 -----------------------------------------------------------------------------------
//
// vLLM keys its blocks with it, and the reason to follow rather than use something cheaper is that
// this store is persistent: a key that survives a restart and is shared between conversations is a
// key whose collisions serve one caller another caller's context. The token ids are compared on
// load as well, so a collision costs a miss rather than a wrong answer — but the two guards are
// cheap and the failure they prevent is silent.
struct Sha256 {
  uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                   0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  uint8_t  buf[64] = {};
  uint64_t len = 0;
  uint32_t n = 0;

  static uint32_t ror(uint32_t x, uint32_t c) { return (x >> c) | (x << (32 - c)); }

  void block(const uint8_t* p) {
    static const uint32_t K[64] = {
      0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
      0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
      0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
      0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
      0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
      0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
      0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
      0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
    uint32_t w[64];
    for (uint32_t i = 0; i < 16; ++i)
      w[i] = ((uint32_t)p[4*i] << 24) | ((uint32_t)p[4*i+1] << 16) |
             ((uint32_t)p[4*i+2] << 8) | (uint32_t)p[4*i+3];
    for (uint32_t i = 16; i < 64; ++i) {
      const uint32_t s0 = ror(w[i-15],7) ^ ror(w[i-15],18) ^ (w[i-15] >> 3);
      const uint32_t s1 = ror(w[i-2],17) ^ ror(w[i-2],19) ^ (w[i-2] >> 10);
      w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for (uint32_t i = 0; i < 64; ++i) {
      const uint32_t S1 = ror(e,6) ^ ror(e,11) ^ ror(e,25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
      const uint32_t S0 = ror(a,2) ^ ror(a,13) ^ ror(a,22);
      const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = S0 + mj;
      hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
  }

  void update(const void* data, size_t bytes) {
    const uint8_t* p = (const uint8_t*)data;
    len += bytes;
    while (bytes) {
      const size_t take = std::min(bytes, (size_t)(64 - n));
      std::memcpy(buf + n, p, take);
      n += (uint32_t)take; p += take; bytes -= take;
      if (n == 64) { block(buf); n = 0; }
    }
  }

  void final(uint8_t out[32]) {
    const uint64_t bits = len * 8;
    const uint8_t pad = 0x80;
    update(&pad, 1);
    const uint8_t z = 0;
    while (n != 56) update(&z, 1);
    uint8_t be[8];
    for (uint32_t i = 0; i < 8; ++i) be[i] = (uint8_t)(bits >> (56 - 8*i));
    update(be, 8);
    for (uint32_t i = 0; i < 8; ++i) {
      out[4*i+0] = (uint8_t)(h[i] >> 24); out[4*i+1] = (uint8_t)(h[i] >> 16);
      out[4*i+2] = (uint8_t)(h[i] >> 8);  out[4*i+3] = (uint8_t)h[i];
    }
  }
};

// key = SHA256(fingerprint || parent || token ids). The fingerprint rides in every link rather than
// only the first, so a store written by a different container or cache geometry cannot match at any
// depth. `parent` null is the root.
void block_key(uint64_t fp, const uint8_t* parent, const uint32_t* ids, uint32_t n,
               uint8_t out[32]) {
  Sha256 s;
  s.update(&fp, sizeof fp);
  static const uint8_t kRoot[32] = {};
  s.update(parent ? parent : kRoot, 32);
  s.update(ids, (size_t)n * sizeof(uint32_t));
  s.final(out);
}

constexpr uint32_t kNil = 0xFFFFFFFFu;
constexpr uint64_t kAlign = 4096;              // O_DIRECT wants offsets, lengths and buffers on it
constexpr uint32_t kBlockMagic = 0x52424C4Bu;  // "RBLK"
constexpr uint32_t kCkptMagic  = 0x52434B50u;  // "RCKP"
constexpr uint32_t kIndexMagic = 0x52494458u;  // "RIDX"
constexpr uint64_t kRestoreArenaBytes = 64ull << 20;

uint64_t round_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

uint32_t next_pow2(uint32_t v) {
  uint32_t p = 1;
  while (p < v) p <<= 1;
  return p;
}

// The header at the front of every slot. Fixed size so a slot's payload starts on an aligned
// offset, and self-describing so a persistent store can be validated rather than trusted.
struct SlotHeader {
  uint32_t magic = 0;
  uint32_t n_tokens = 0;       // block: block_tokens. checkpoint: the partial tail, 0..B-1
  uint64_t fingerprint = 0;
  uint64_t payload_bytes = 0;
  uint32_t pos = 0;            // checkpoint: the position it restores to
  uint32_t reserved = 0;
  uint8_t  key[32] = {};
  uint8_t  parent[32] = {};    // checkpoint: the block key it hangs off, so lookup is one probe
};

}  // namespace

struct PrefixCache::Impl {
  int blk_fd = -1, ck_fd = -1;

  // Hash chains over the slot tables. Bucket arrays are preallocated to twice the slot count and
  // the chain pointers live in the slots, so a lookup, an insert and an eviction all allocate
  // nothing.
  std::vector<uint32_t> blk_bucket, ck_bucket;
  uint32_t blk_mask = 0, ck_mask = 0;
  uint32_t blk_head = kNil, blk_tail = kNil;
  uint32_t ck_head = kNil, ck_tail = kNil;

  // ---- the writer -------------------------------------------------------------------------------
  struct Job { bool ckpt = false; uint32_t slot = 0; uint32_t buf = 0; };
  std::thread          writer;
  std::mutex           m;
  std::condition_variable cv_job, cv_buf;
  std::deque<Job>      jobs;
  std::vector<uint8_t*> stage;      // queue_depth aligned buffers, each ck_slot_ bytes
  std::vector<bool>    stage_free;
  uint32_t             in_flight = 0;
  bool                 stop = false;

  // Restore staging: `gblocks` slot images plus one gather buffer, so a batch of blocks reaches the
  // cards as one transfer per region instead of one per region per block.
  std::vector<uint8_t*> rbuf;
  uint8_t*             gather = nullptr;
  uint64_t             gather_bytes = 0;
  uint32_t             gblocks = 0;
  uint8_t*             ckbuf = nullptr;

  ~Impl() {
    for (uint8_t* p : stage) std::free(p);
    for (uint8_t* p : rbuf) std::free(p);
    std::free(gather);
    std::free(ckbuf);
    if (blk_fd >= 0) ::close(blk_fd);
    if (ck_fd >= 0) ::close(ck_fd);
  }
};

PrefixCache::~PrefixCache() { close(); }

// ---- layout --------------------------------------------------------------------------------------

bool PrefixCache::build_layout(std::string* err) {
  const uint32_t B = opt_.block_tokens;
  uint64_t app = 0;
  for (const KvRegion& r : app_) {
    if (!r.ratio || B % r.ratio) {
      if (err) *err = "block_tokens " + std::to_string(B) + " is not a multiple of compress ratio " +
                      std::to_string(r.ratio);
      return false;
    }
    app += (uint64_t)kSavedRanks * (B / r.ratio) * r.row_bytes;
  }
  uint64_t roll = 0;
  for (const KvRegion& r : roll_) roll += (uint64_t)kSavedRanks * r.rows * r.row_bytes;
  if (!app || !roll) {
    if (err) *err = "no regions to cache";
    return false;
  }
  blk_payload_ = app;
  // A checkpoint carries the rings AND everything the partial block has emitted so far, which is
  // what lets it restore to an arbitrary position rather than only to a block boundary. Sized for
  // the worst case, B-1 tokens past the boundary.
  ck_payload_ = roll + app;
  blk_slot_ = round_up(sizeof(SlotHeader) + (uint64_t)B * sizeof(uint32_t), kAlign) +
              round_up(blk_payload_, kAlign);
  ck_slot_  = round_up(sizeof(SlotHeader) + (uint64_t)B * sizeof(uint32_t), kAlign) +
              round_up(ck_payload_, kAlign);
  return true;
}

// Where a block's rows for `region` sit inside a slot's payload, and how many there are for the
// span [t0, t1). The same walk serves the block payload and the checkpoint's partial tail, which is
// why it takes the span rather than assuming a whole block.
static void region_span(const std::vector<KvRegion>& regs, uint32_t ranks, uint32_t t0, uint32_t t1,
                        uint32_t idx, uint64_t* off, uint64_t* row0, uint64_t* nrows) {
  uint64_t o = 0;
  for (uint32_t i = 0; i < idx; ++i) {
    const KvRegion& r = regs[i];
    o += (uint64_t)ranks * (t1 / r.ratio - t0 / r.ratio) * r.row_bytes;
  }
  const KvRegion& r = regs[idx];
  *off = o;
  *row0 = t0 / r.ratio;
  *nrows = t1 / r.ratio - t0 / r.ratio;
}

// ---- open / close --------------------------------------------------------------------------------

bool PrefixCache::open(const Options& o, const std::vector<KvRegion>& regions, const KvIo& io,
                       uint64_t fingerprint, std::string* err) {
  close();
  if (o.dir.empty()) return true;                    // not configured is not an error
  if (!io.save || !io.load) { if (err) *err = "no KV transfer callbacks"; return false; }
  if (!o.block_tokens) { if (err) *err = "block_tokens must be positive"; return false; }
  opt_ = o; io_ = io; fp_ = fingerprint;
  if (!opt_.ranks) opt_.ranks = 1;
  for (const KvRegion& r : regions) {
    if (!r.row_bytes || !r.rows) continue;
    (r.ratio ? app_ : roll_).push_back(r);
  }
  if (!build_layout(err)) { close(); return false; }
  // ---- THE LAYOUT IS PART OF THE IDENTITY ---------------------------------------------------------
  //
  // The caller's fingerprint covers the container and the model geometry. It does NOT cover how this
  // file arranges the bytes — the region set, the rank policy, the block size — and a slot written
  // under one arrangement and read under another has the right key, the right token ids, and rows at
  // the wrong offsets. Verification would pass, because verification reads the header.
  //
  // Folding the payload sizes in makes every such change invalidate the store automatically, which
  // is the behaviour a version constant only approximates: it covers layout changes nobody
  // remembered to bump.
  fp_ = fp_ * 1099511628211ull ^ blk_payload_;
  fp_ = fp_ * 1099511628211ull ^ ck_payload_;

  // Parents included: the flag names a path, and requiring the caller to have made its parents is a
  // startup failure that says nothing about what to do next.
  for (size_t i = 1; i <= o.dir.size(); ++i) {
    if (i != o.dir.size() && o.dir[i] != '/') continue;
    const std::string part = o.dir.substr(0, i);
    if (::mkdir(part.c_str(), 0700) && errno != EEXIST) {
      if (err) *err = "cannot create " + part + ": " + std::strerror(errno);
      close(); return false;
    }
  }
  // The budget split. Blocks are shared between conversations and checkpoints are not, so most of
  // the space belongs to blocks; the checkpoint share only has to hold a resume point per session
  // in flight.
  const uint32_t pc = std::min<uint32_t>(o.ckpt_percent, 90);
  uint64_t ck_bytes = o.budget_bytes / 100 * pc;
  uint32_t n_ck = (uint32_t)std::min<uint64_t>(ck_bytes / ck_slot_, 1u << 20);
  if (n_ck < 2) n_ck = 2;
  uint64_t blk_bytes = o.budget_bytes > n_ck * ck_slot_ ? o.budget_bytes - n_ck * ck_slot_ : 0;
  uint32_t n_blk = (uint32_t)std::min<uint64_t>(blk_bytes / blk_slot_, 1u << 22);
  if (n_blk < 8) {
    if (err) *err = "prefix cache budget " + std::to_string(o.budget_bytes >> 20) +
                    " MiB holds only " + std::to_string(n_blk) + " blocks of " +
                    std::to_string(blk_slot_ >> 20) + " MiB; raise --prefix-cache-gib";
    close(); return false;
  }
  blk_.assign(n_blk, Slot{});
  ck_.assign(n_ck, Slot{});

  impl_ = new Impl();
  impl_->blk_bucket.assign(next_pow2(n_blk * 2), kNil);
  impl_->ck_bucket.assign(next_pow2(n_ck * 2), kNil);
  impl_->blk_mask = (uint32_t)impl_->blk_bucket.size() - 1;
  impl_->ck_mask = (uint32_t)impl_->ck_bucket.size() - 1;
  thread_free_lists();

  if (!open_files(err)) { close(); return false; }

  // Host arenas, all of them here so nothing allocates once serving starts.
  const uint32_t qd = o.queue_depth ? o.queue_depth : 1u;
  impl_->stage.assign(qd, nullptr);
  impl_->stage_free.assign(qd, true);
  for (uint32_t i = 0; i < qd; ++i) {
    if (posix_memalign((void**)&impl_->stage[i], kAlign, (size_t)ck_slot_)) {
      if (err) *err = "cannot allocate the prefix cache staging arena";
      close(); return false;
    }
  }
  impl_->gblocks = (uint32_t)std::max<uint64_t>(1, kRestoreArenaBytes / blk_slot_);
  impl_->rbuf.assign(impl_->gblocks, nullptr);
  for (uint32_t i = 0; i < impl_->gblocks; ++i) {
    if (posix_memalign((void**)&impl_->rbuf[i], kAlign, (size_t)blk_slot_)) {
      if (err) *err = "cannot allocate the prefix cache restore arena";
      close(); return false;
    }
  }
  // One region's rows across a whole restore batch, which is the unit a restore hands to the cards.
  uint64_t g = 0;
  for (const KvRegion& r : app_)
    g = std::max(g, (uint64_t)kSavedRanks * impl_->gblocks * (opt_.block_tokens / r.ratio) * r.row_bytes);
  impl_->gather_bytes = g;
  if (posix_memalign((void**)&impl_->gather, kAlign, (size_t)g) ||
      posix_memalign((void**)&impl_->ckbuf, kAlign, (size_t)ck_slot_)) {
    if (err) *err = "cannot allocate the prefix cache gather arena";
    close(); return false;
  }

  if (opt_.persistent) load_index();
  impl_->writer = std::thread([this] {
    for (;;) {
      Impl::Job j;
      {
        std::unique_lock<std::mutex> lk(impl_->m);
        impl_->cv_job.wait(lk, [&] { return impl_->stop || !impl_->jobs.empty(); });
        if (impl_->stop && impl_->jobs.empty()) return;
        j = impl_->jobs.front();
        impl_->jobs.pop_front();
      }
      const uint64_t slot_bytes = j.ckpt ? ck_slot_ : blk_slot_;
      const int fd = j.ckpt ? impl_->ck_fd : impl_->blk_fd;
      const uint8_t* src = impl_->stage[j.buf];
      const uint64_t base = (uint64_t)j.slot * slot_bytes;
      // ---- THE HEADER GOES LAST, AND THAT IS THE WHOLE COMMIT PROTOCOL ---------------------------
      //
      // Verification on the way back reads the header: magic, fingerprint, token count, token ids.
      // Written front to back, a short write leaves that header COMPLETE and the payload truncated,
      // so the slot verifies and the engine loads whatever was in the file underneath it — the one
      // failure this cache must not have, because it is silent and it moves the emitted text.
      // Written payload first, a torn write leaves the old header in place; its key belongs to some
      // other prefix, so the token compare rejects it and `restore` drops the slot.
      const uint64_t hbytes = round_up(sizeof(SlotHeader) + (uint64_t)opt_.block_tokens *
                                       sizeof(uint32_t), kAlign);
      auto span = [&](uint64_t off, uint64_t bytes) -> uint64_t {
        uint64_t done = 0;
        while (done < bytes) {
          const ssize_t w = ::pwrite(fd, src + off + done, (size_t)(bytes - done),
                                     (off_t)(base + off + done));
          if (w <= 0) {
            aff::ui::err("prefix cache: write failed at slot %u+%llu: %s\n", j.slot,
                         (unsigned long long)(off + done), std::strerror(errno));
            break;
          }
          done += (uint64_t)w;
        }
        return done;
      };
      uint64_t done = span(hbytes, slot_bytes - hbytes);
      if (done == slot_bytes - hbytes) done += span(0, hbytes);
      {
        std::lock_guard<std::mutex> lk(impl_->m);
        impl_->stage_free[j.buf] = true;
        st_.bytes_written += done;
        --impl_->in_flight;
      }
      impl_->cv_buf.notify_all();
    }
  });
  ready_ = true;
  return true;
}

// Every slot unused and in index order, which is what makes the eviction list hand out the
// never-written slots before it evicts anything real.
void PrefixCache::thread_free_lists() {
  const uint32_t nb = (uint32_t)blk_.size(), nc = (uint32_t)ck_.size();
  for (uint32_t i = 0; i < nb; ++i) {
    blk_[i].prev = i ? i - 1 : kNil;
    blk_[i].next = i + 1 < nb ? i + 1 : kNil;
  }
  impl_->blk_head = nb ? 0 : kNil; impl_->blk_tail = nb ? nb - 1 : kNil;
  for (uint32_t i = 0; i < nc; ++i) {
    ck_[i].prev = i ? i - 1 : kNil;
    ck_[i].next = i + 1 < nc ? i + 1 : kNil;
  }
  impl_->ck_head = nc ? 0 : kNil; impl_->ck_tail = nc ? nc - 1 : kNil;
}

bool PrefixCache::open_files(std::string* err) {
  const std::string bp = opt_.dir + "/blocks.bin";
  const std::string cp = opt_.dir + "/ckpts.bin";
  // O_DIRECT, and a refusal rather than a buffered fallback. The store is sized in gigabytes and
  // read in megabyte runs; going through the page cache would evict the expert pool's own file
  // mappings, which is paid for in decode and would never show up as a cache problem.
  const int flags = O_RDWR | O_CREAT | O_DIRECT;
  impl_->blk_fd = ::open(bp.c_str(), flags, 0600);
  impl_->ck_fd = ::open(cp.c_str(), flags, 0600);
  if (impl_->blk_fd < 0 || impl_->ck_fd < 0) {
    if (err) *err = "cannot open " + bp + " with O_DIRECT: " + std::strerror(errno) +
                    " (the cache directory must be on a filesystem that supports it)";
    return false;
  }
  const uint64_t bb = (uint64_t)blk_.size() * blk_slot_, cb = (uint64_t)ck_.size() * ck_slot_;
  if (::fallocate(impl_->blk_fd, 0, 0, (off_t)bb) || ::fallocate(impl_->ck_fd, 0, 0, (off_t)cb)) {
    if (err) *err = "cannot reserve " + std::to_string((bb + cb) >> 20) + " MiB in " + opt_.dir +
                    ": " + std::strerror(errno);
    return false;
  }
  if (!opt_.persistent) {
    // No name, so the space is returned the moment the process exits however it exits.
    ::unlink(bp.c_str());
    ::unlink(cp.c_str());
  }
  return true;
}

void PrefixCache::close() {
  if (impl_) {
    {
      std::lock_guard<std::mutex> lk(impl_->m);
      impl_->stop = true;
    }
    impl_->cv_job.notify_all();
    if (impl_->writer.joinable()) impl_->writer.join();
    // The writer drains before it returns, so both of these are zero unless something changed that
    // contract. It is worth an abort rather than a warning: a queued job that never ran leaves a
    // slot in the index — and, under --persistent-prefix-cache, in the index FILE — naming a
    // payload that was never written, which the next process reads back as a hit.
    if (!impl_->jobs.empty() || impl_->in_flight) {
      aff::ui::fatal("fatal: prefix cache closed with %zu queued and %u staged writes\n",
                   impl_->jobs.size(), impl_->in_flight);
      std::abort();
    }
    if (opt_.persistent && ready_) save_index();
    delete impl_;
    impl_ = nullptr;
  }
  blk_.clear(); ck_.clear(); app_.clear(); roll_.clear();
  ready_ = false;
}

// ---- the eviction list and the index ---------------------------------------------------------------

void PrefixCache::touch(int32_t slot, bool ckpt) {
  std::vector<Slot>& t = ckpt ? ck_ : blk_;
  uint32_t& head = ckpt ? impl_->ck_head : impl_->blk_head;
  uint32_t& tail = ckpt ? impl_->ck_tail : impl_->blk_tail;
  Slot& s = t[slot];
  if (tail == (uint32_t)slot) return;
  if (s.prev != kNil) t[s.prev].next = s.next; else head = s.next;
  if (s.next != kNil) t[s.next].prev = s.prev;
  s.prev = tail; s.next = kNil;
  if (tail != kNil) t[tail].next = (uint32_t)slot; else head = (uint32_t)slot;
  tail = (uint32_t)slot;
}

void PrefixCache::drop(int32_t slot, bool ckpt) {
  std::vector<Slot>& t = ckpt ? ck_ : blk_;
  std::vector<uint32_t>& bucket = ckpt ? impl_->ck_bucket : impl_->blk_bucket;
  const uint32_t mask = ckpt ? impl_->ck_mask : impl_->blk_mask;
  uint32_t& head = ckpt ? impl_->ck_head : impl_->blk_head;
  uint32_t& tail = ckpt ? impl_->ck_tail : impl_->blk_tail;
  const uint32_t i = (uint32_t)slot;
  Slot& s = t[i];
  if (s.used) {
    uint32_t b;
    std::memcpy(&b, ckpt ? s.parent : s.key, 4);
    b &= mask;
    uint32_t* p = &bucket[b];
    while (*p != kNil && *p != i) p = &t[*p].hnext;
    if (*p == i) *p = s.hnext;
  }
  s.hnext = kNil;
  s.used = false;
  std::memset(s.key, 0, sizeof(s.key));
  std::memset(s.parent, 0, sizeof(s.parent));
  s.n_tokens = 0; s.pos = 0;
  // To the FRONT, not the back: a slot with nothing valid in it is the first one that should be
  // handed out again, which is the same place `thread_free_lists` puts a never-written slot.
  if (head == i) return;
  if (s.prev != kNil) t[s.prev].next = s.next;
  if (s.next != kNil) t[s.next].prev = s.prev; else tail = s.prev;
  s.prev = kNil; s.next = head;
  if (head != kNil) t[head].prev = i;
  head = i;
  if (tail == kNil) tail = i;
}

// Take the least reusable slot. vLLM's rule: the never-written slots sit at the front and go first,
// and after that it is plain LRU. Evicting drops the old key from the chain, which is the only
// place a stale key could otherwise survive.
int32_t PrefixCache::claim(bool ckpt) {
  std::vector<Slot>& t = ckpt ? ck_ : blk_;
  std::vector<uint32_t>& bucket = ckpt ? impl_->ck_bucket : impl_->blk_bucket;
  const uint32_t mask = ckpt ? impl_->ck_mask : impl_->blk_mask;
  uint32_t& head = ckpt ? impl_->ck_head : impl_->blk_head;
  if (head == kNil) return -1;
  const uint32_t i = head;
  Slot& s = t[i];
  if (s.used) {
    uint32_t b;
    std::memcpy(&b, ckpt ? s.parent : s.key, 4);
    b &= mask;
    uint32_t* p = &bucket[b];
    while (*p != kNil && *p != i) p = &t[*p].hnext;
    if (*p == i) *p = s.hnext;
    ++st_.blocks_evicted;
  }
  s.hnext = kNil;
  s.used = true;
  return (int32_t)i;
}

int32_t PrefixCache::find(const uint8_t* key) const {
  uint32_t b;
  std::memcpy(&b, key, 4);
  for (uint32_t i = impl_->blk_bucket[b & impl_->blk_mask]; i != kNil; i = blk_[i].hnext)
    if (blk_[i].used && !std::memcmp(blk_[i].key, key, 32)) return (int32_t)i;
  return -1;
}

// Checkpoints are indexed by their PARENT block's key, because a checkpoint sits at an arbitrary
// position inside a block and a caller cannot know how far in until it has found one. Several
// share a parent — one per turn of the same conversation — so the chain is walked and the longest
// verified tail wins.
int32_t PrefixCache::find_ckpt(const uint8_t* parent, const uint32_t* ids, uint32_t avail,
                               uint32_t* tail) const {
  uint32_t b;
  std::memcpy(&b, parent, 4);
  int32_t best = -1;
  uint32_t best_tail = 0;
  for (uint32_t i = impl_->ck_bucket[b & impl_->ck_mask]; i != kNil; i = ck_[i].hnext) {
    const Slot& s = ck_[i];
    if (!s.used || std::memcmp(s.parent, parent, 32)) continue;
    if (s.n_tokens > avail) continue;
    if (best >= 0 && s.n_tokens <= best_tail) continue;
    // Recomputing the key over THIS prompt's tokens is what proves the tail matches. Without it a
    // checkpoint from a conversation that shares a block boundary and then diverges would be
    // restored, and the divergence would never surface as anything but wrong output.
    uint8_t probe[32];
    block_key(fp_, parent, ids, s.n_tokens, probe);
    if (std::memcmp(probe, s.key, 32)) continue;
    best = (int32_t)i; best_tail = s.n_tokens;
  }
  if (best >= 0 && tail) *tail = best_tail;
  return best;
}

uint32_t PrefixCache::chain(const uint32_t* ids, uint32_t n, std::vector<uint8_t>* keys) const {
  const uint32_t B = opt_.block_tokens;
  const uint32_t nb = n / B;
  keys->assign((size_t)nb * 32, 0);
  for (uint32_t i = 0; i < nb; ++i)
    block_key(fp_, i ? keys->data() + (size_t)(i - 1) * 32 : nullptr, ids + (size_t)i * B, B,
              keys->data() + (size_t)i * 32);
  return nb;
}

uint32_t PrefixCache::lookup(const uint32_t* ids, uint32_t n, uint32_t* blocks_matched) const {
  if (blocks_matched) *blocks_matched = 0;
  if (!ready_ || !ids || !n) return 0;
  const uint32_t B = opt_.block_tokens;
  std::vector<uint8_t> keys;
  const uint32_t nb = chain(ids, n, &keys);
  // How many leading blocks the store actually holds. A gap stops the walk: restoring needs every
  // block below the resume point, not a subset.
  uint32_t have = 0;
  while (have < nb && find(keys.data() + (size_t)have * 32) >= 0) ++have;
  if (blocks_matched) *blocks_matched = have;
  // Then the deepest checkpoint hanging off one of those boundaries. Its parent key names the whole
  // prefix, so this is one bucket walk per level and the common case — the previous turn of this
  // same conversation — hits at the first.
  static const uint8_t kRoot[32] = {};
  for (int32_t k = (int32_t)have; k >= 0; --k) {
    const uint8_t* parent = k ? keys.data() + (size_t)(k - 1) * 32 : kRoot;
    uint32_t tail = 0;
    if (find_ckpt(parent, ids + (size_t)k * B, n - (uint32_t)k * B, &tail) >= 0)
      return (uint32_t)k * B + tail;
  }
  return 0;
}

// ---- restore ---------------------------------------------------------------------------------------

namespace {
bool read_exact(int fd, void* dst, uint64_t bytes, uint64_t off) {
  uint64_t done = 0;
  while (done < bytes) {
    const ssize_t r = ::pread(fd, (uint8_t*)dst + done, (size_t)(bytes - done), (off_t)(off + done));
    if (r <= 0) return false;
    done += (uint64_t)r;
  }
  return true;
}
}  // namespace

bool PrefixCache::restore(const uint32_t* ids, uint32_t n_restore, std::string* err) {
  if (!ready_ || !n_restore) return false;
  drain();                       // a slot published moments ago may still be in the writer's queue
  const uint32_t B = opt_.block_tokens;
  const uint32_t nfull = n_restore / B, tail = n_restore % B;
  std::vector<uint8_t> keys;
  chain(ids, n_restore, &keys);

  ++st_.lookups;
  const uint32_t hdr = (uint32_t)round_up(sizeof(SlotHeader) + (uint64_t)B * sizeof(uint32_t),
                                          kAlign);
  // Every block this restore read, so the eviction order can be set once at the end and deep-end
  // first — see the note in publish(). Touching them as they are read does it the other way round.
  std::vector<uint32_t> used(nfull, kNil);
  // ---- the full blocks, a batch at a time -------------------------------------------------------
  for (uint32_t b0 = 0; b0 < nfull; b0 += impl_->gblocks) {
    const uint32_t g = std::min(impl_->gblocks, nfull - b0);
    for (uint32_t i = 0; i < g; ++i) {
      const int32_t s = find(keys.data() + (size_t)(b0 + i) * 32);
      if (s < 0) { if (err) *err = "prefix cache: a block vanished between lookup and restore"; return false; }
      if (!read_exact(impl_->blk_fd, impl_->rbuf[i], blk_slot_, (uint64_t)s * blk_slot_)) {
        if (err) *err = std::string("prefix cache: block read failed: ") + std::strerror(errno);
        return false;
      }
      const SlotHeader* h = (const SlotHeader*)impl_->rbuf[i];
      const uint32_t* tok = (const uint32_t*)(impl_->rbuf[i] + sizeof(SlotHeader));
      // The key matched; this is what makes that enough. A collision now costs a miss instead of
      // serving one conversation another's context.
      if (h->magic != kBlockMagic || h->fingerprint != fp_ || h->n_tokens != B ||
          std::memcmp(tok, ids + (size_t)(b0 + i) * B, (size_t)B * sizeof(uint32_t))) {
        ++st_.verify_misses;
        drop(s, false);          // or this prefix reports a hit and prefills from scratch forever
        if (err) *err = "prefix cache: a block failed verification";
        return false;
      }
      st_.bytes_read += blk_slot_;
      used[b0 + i] = (uint32_t)s;
    }
    // Region by region across the whole batch, so the cards see one transfer per region rather than
    // one per region per block. At a 1M restore that is the difference between ten thousand
    // transfers and ten million.
    for (uint32_t ri = 0; ri < app_.size(); ++ri) {
      const KvRegion& r = app_[ri];
      const uint64_t rpb = B / r.ratio, seg = rpb * r.row_bytes;
      uint64_t off, row0, nrows;
      region_span(app_, kSavedRanks, 0, B, ri, &off, &row0, &nrows);
      // The batch's rows for this region, contiguous: the slots hold one block each and the cards
      // want one run across all of them.
      for (uint32_t i = 0; i < g; ++i)
        std::memcpy(impl_->gather + (uint64_t)i * seg, impl_->rbuf[i] + hdr + off, (size_t)seg);
      // The SAME bytes to every card. They are bit-identical on the machine — see KvIo — so this is
      // a fan-out and not a broadcast of one card's opinion.
      for (uint32_t rk = 0; rk < opt_.ranks; ++rk)
        if (!io_.load(io_.ctx, r.part, r.layer, rk, (uint64_t)b0 * rpb, (uint64_t)g * rpb,
                      impl_->gather)) {
          if (err) *err = "prefix cache: the engine refused a restored block";
          return false;
        }
    }
  }

  // ---- the checkpoint: the rings, and whatever the partial block has emitted ---------------------
  static const uint8_t kRoot[32] = {};
  const uint8_t* parent = nfull ? keys.data() + (size_t)(nfull - 1) * 32 : kRoot;
  uint32_t got = 0;
  const int32_t cs = find_ckpt(parent, ids + (size_t)nfull * B, tail, &got);
  if (cs < 0 || got != tail) {
    if (err) *err = "prefix cache: the checkpoint vanished between lookup and restore";
    return false;
  }
  if (!read_exact(impl_->ck_fd, impl_->ckbuf, ck_slot_, (uint64_t)cs * ck_slot_)) {
    if (err) *err = std::string("prefix cache: checkpoint read failed: ") + std::strerror(errno);
    return false;
  }
  const SlotHeader* h = (const SlotHeader*)impl_->ckbuf;
  const uint32_t* tok = (const uint32_t*)(impl_->ckbuf + sizeof(SlotHeader));
  if (h->magic != kCkptMagic || h->fingerprint != fp_ || h->pos != n_restore ||
      (tail && std::memcmp(tok, ids + (size_t)nfull * B, (size_t)tail * sizeof(uint32_t)))) {
    ++st_.verify_misses;
    drop(cs, true);
    if (err) *err = "prefix cache: the checkpoint failed verification";
    return false;
  }
  st_.bytes_read += ck_slot_;
  uint64_t off = hdr;
  for (const KvRegion& r : roll_) {
    for (uint32_t rk = 0; rk < opt_.ranks; ++rk)
      if (!io_.load(io_.ctx, r.part, r.layer, rk, 0, r.rows, impl_->ckbuf + off)) {
        if (err) *err = "prefix cache: the engine refused a restored ring";
        return false;
      }
    off += (uint64_t)kSavedRanks * r.rows * r.row_bytes;
  }
  if (tail) {
    const uint32_t t0 = nfull * B;
    for (uint32_t ri = 0; ri < app_.size(); ++ri) {
      uint64_t o, row0, nrows;
      region_span(app_, kSavedRanks, t0, t0 + tail, ri, &o, &row0, &nrows);
      if (!nrows) continue;
      for (uint32_t rk = 0; rk < opt_.ranks; ++rk)
        if (!io_.load(io_.ctx, app_[ri].part, app_[ri].layer, rk, row0, nrows,
                      impl_->ckbuf + off + o)) {
          if (err) *err = "prefix cache: the engine refused a restored partial block";
          return false;
        }
    }
  }
  touch(cs, true);
  for (uint32_t i = nfull; i-- > 0; )
    if (used[i] != kNil) touch((int32_t)used[i], false);
  ++st_.hits;
  st_.tokens_restored += n_restore;
  return true;
}

// ---- publish -----------------------------------------------------------------------------------------

// Takes a staging buffer, waiting for the writer if all of them are out. The wait is bounded by the
// SSD, which is an order of magnitude faster than the prefill that produced the rows, so it is a
// backstop rather than a cost.
uint32_t PrefixCache::acquire_stage() {
  std::unique_lock<std::mutex> lk(impl_->m);
  impl_->cv_buf.wait(lk, [&] {
    return std::find(impl_->stage_free.begin(), impl_->stage_free.end(), true) !=
           impl_->stage_free.end();
  });
  for (uint32_t i = 0; i < impl_->stage_free.size(); ++i)
    if (impl_->stage_free[i]) { impl_->stage_free[i] = false; ++impl_->in_flight; return i; }
  return 0;
}

// Hands one back without a write having happened, which is the path every refusal below takes.
void PrefixCache::release_stage(uint32_t buf) {
  {
    std::lock_guard<std::mutex> lk(impl_->m);
    impl_->stage_free[buf] = true;
    --impl_->in_flight;
  }
  impl_->cv_buf.notify_all();
}

bool PrefixCache::publish(const uint32_t* ids, uint32_t n, std::string* err) {
  if (!ready_ || !ids || !n) return true;
  const uint32_t B = opt_.block_tokens;
  const uint32_t nfull = n / B, tail = n % B;
  std::vector<uint8_t> keys;
  chain(ids, n, &keys);
  const uint32_t hdr = (uint32_t)round_up(sizeof(SlotHeader) + (uint64_t)B * sizeof(uint32_t),
                                          kAlign);

  for (uint32_t i = 0; i < nfull; ++i) {
    if (find(keys.data() + (size_t)i * 32) >= 0) continue;   // already held, and identical
    const int32_t s = claim(false);
    if (s < 0) break;
    const uint32_t bi = acquire_stage();
    uint8_t* buf = impl_->stage[bi];
    SlotHeader* h = (SlotHeader*)buf;
    *h = SlotHeader{};
    h->magic = kBlockMagic; h->n_tokens = B; h->fingerprint = fp_;
    h->payload_bytes = blk_payload_; h->pos = (i + 1) * B;
    std::memcpy(h->key, keys.data() + (size_t)i * 32, 32);
    std::memcpy(buf + sizeof(SlotHeader), ids + (size_t)i * B, (size_t)B * sizeof(uint32_t));
    for (uint32_t ri = 0; ri < app_.size(); ++ri) {
      uint64_t off, row0, nrows;
      region_span(app_, kSavedRanks, i * B, (i + 1) * B, ri, &off, &row0, &nrows);
      for (uint32_t rk = 0; rk < kSavedRanks; ++rk)
        if (!io_.save(io_.ctx, app_[ri].part, app_[ri].layer, rk, row0, nrows,
                      buf + hdr + off + (uint64_t)rk * nrows * app_[ri].row_bytes)) {
          if (err) *err = "prefix cache: the engine refused to hand over a block";
          release_stage(bi);
          return false;
        }
    }
    Slot& sl = blk_[s];
    std::memcpy(sl.key, h->key, 32);
    sl.n_tokens = B; sl.pos = h->pos;
    uint32_t b; std::memcpy(&b, sl.key, 4); b &= impl_->blk_mask;
    sl.hnext = impl_->blk_bucket[b];
    impl_->blk_bucket[b] = (uint32_t)s;
    touch(s, false);                              // claim() hands back the head and does not unlink
    ++st_.blocks_written;
    {
      std::lock_guard<std::mutex> lk(impl_->m);
      impl_->jobs.push_back(Impl::Job{false, (uint32_t)s, bi});
    }
    impl_->cv_job.notify_one();
  }

  // ---- WHICH END OF A REQUEST AGES OUT FIRST, AND WHY IT IS THE DEEP END -------------------------
  //
  // The eviction list is LRU and the head goes first, so the last thing touched is the last thing
  // evicted. Touching the blocks in ASCENDING order — which is what writing them in order did —
  // put the DEEPEST block furthest from the head, i.e. protected it best. That is backwards: a
  // block's key chains every token before it, so block i+1 is strictly more specific than block i
  // and strictly less likely to be shared with another conversation. It is also the one whose loss
  // costs least, because `lookup` walks the chain from the front and stops at the first gap — so
  // evicting from the deep end leaves a shorter usable prefix, while evicting from the shallow end
  // strands every block above it.
  //
  // Reusing a block is a use. The `continue` above skips a block that was already held, and skipping
  // it meant a hot shared prefix aged while the cold new blocks stacked on top of it stayed fresh —
  // exactly inverted. So this walks the keys rather than the slots the loop above claimed: `find`
  // covers held and freshly written alike, and it also drops the blocks that a pool too small for
  // this prompt has already evicted out from under their own publish.
  for (uint32_t i = nfull; i-- > 0; ) {
    const int32_t s = find(keys.data() + (size_t)i * 32);
    if (s >= 0) touch(s, false);
  }

  // The checkpoint. One per publish, at exactly the position the engine is at, which is what lets
  // the next turn resume without replaying anything.
  static const uint8_t kRoot[32] = {};
  const uint8_t* parent = nfull ? keys.data() + (size_t)(nfull - 1) * 32 : kRoot;
  const int32_t cs = claim(true);
  if (cs < 0) return true;
  const uint32_t bi = acquire_stage();
  uint8_t* buf = impl_->stage[bi];
  SlotHeader* h = (SlotHeader*)buf;
  *h = SlotHeader{};
  h->magic = kCkptMagic; h->n_tokens = tail; h->fingerprint = fp_;
  h->payload_bytes = ck_payload_; h->pos = n;
  std::memcpy(h->parent, parent, 32);
  block_key(fp_, parent, ids + (size_t)nfull * B, tail, h->key);
  if (tail) std::memcpy(buf + sizeof(SlotHeader), ids + (size_t)nfull * B,
                        (size_t)tail * sizeof(uint32_t));
  uint64_t off = hdr;
  for (const KvRegion& r : roll_) {
    for (uint32_t rk = 0; rk < kSavedRanks; ++rk)
      if (!io_.save(io_.ctx, r.part, r.layer, rk, 0, r.rows,
                    buf + off + (uint64_t)rk * r.rows * r.row_bytes)) {
        if (err) *err = "prefix cache: the engine refused to hand over a ring";
        release_stage(bi);
        return false;
      }
    off += (uint64_t)kSavedRanks * r.rows * r.row_bytes;
  }
  if (tail) {
    const uint32_t t0 = nfull * B;
    for (uint32_t ri = 0; ri < app_.size(); ++ri) {
      uint64_t o, row0, nrows;
      region_span(app_, kSavedRanks, t0, t0 + tail, ri, &o, &row0, &nrows);
      if (!nrows) continue;
      for (uint32_t rk = 0; rk < kSavedRanks; ++rk)
        if (!io_.save(io_.ctx, app_[ri].part, app_[ri].layer, rk, row0, nrows,
                      buf + off + o + (uint64_t)rk * nrows * app_[ri].row_bytes)) {
          if (err) *err = "prefix cache: the engine refused to hand over a partial block";
          release_stage(bi);
          return false;
        }
    }
  }
  Slot& sl = ck_[cs];
  std::memcpy(sl.key, h->key, 32);
  std::memcpy(sl.parent, parent, 32);
  sl.n_tokens = tail; sl.pos = n;
  uint32_t b; std::memcpy(&b, sl.parent, 4); b &= impl_->ck_mask;
  sl.hnext = impl_->ck_bucket[b];
  impl_->ck_bucket[b] = (uint32_t)cs;
  touch(cs, true);
  ++st_.ckpts_written;
  {
    std::lock_guard<std::mutex> lk(impl_->m);
    impl_->jobs.push_back(Impl::Job{true, (uint32_t)cs, bi});
  }
  impl_->cv_job.notify_one();
  return true;
}

void PrefixCache::drain() {
  if (!impl_) return;
  std::unique_lock<std::mutex> lk(impl_->m);
  impl_->cv_buf.wait(lk, [&] { return impl_->jobs.empty() && impl_->in_flight == 0; });
}

// ---- persistence -------------------------------------------------------------------------------------

bool PrefixCache::save_index() const {
  const std::string p = opt_.dir + "/index.bin";
  FILE* f = std::fopen(p.c_str(), "wb");
  if (!f) return false;
  // The eviction lists go with the slots. They are threaded through prev/next, so restoring the
  // slots without their heads would leave two lists whose ends nothing points at.
  struct IndexHeader {
    uint32_t magic, block_tokens, n_blk, n_ck;
    uint64_t fp, blk_slot, ck_slot;
    uint32_t blk_head, blk_tail, ck_head, ck_tail;
  } hd{kIndexMagic, opt_.block_tokens, (uint32_t)blk_.size(), (uint32_t)ck_.size(),
       fp_, blk_slot_, ck_slot_,
       impl_->blk_head, impl_->blk_tail, impl_->ck_head, impl_->ck_tail};
  bool ok = std::fwrite(&hd, sizeof hd, 1, f) == 1;
  for (uint32_t i = 0; i < blk_.size() && ok; ++i) ok = std::fwrite(&blk_[i], sizeof(Slot), 1, f) == 1;
  for (uint32_t i = 0; i < ck_.size() && ok; ++i) ok = std::fwrite(&ck_[i], sizeof(Slot), 1, f) == 1;
  std::fclose(f);
  return ok;
}

bool PrefixCache::load_index() {
  const std::string p = opt_.dir + "/index.bin";
  FILE* f = std::fopen(p.c_str(), "rb");
  if (!f) return false;
  struct IndexHeader {
    uint32_t magic, block_tokens, n_blk, n_ck;
    uint64_t fp, blk_slot, ck_slot;
    uint32_t blk_head, blk_tail, ck_head, ck_tail;
  } hd{};
  bool ok = std::fread(&hd, sizeof hd, 1, f) == 1;
  // Any of these differing means the slots on disk are not the slots this process would write, so
  // the store is dropped rather than reinterpreted. That is the whole point of the fingerprint.
  if (!ok || hd.magic != kIndexMagic || hd.fp != fp_ || hd.block_tokens != opt_.block_tokens ||
      hd.n_blk != blk_.size() || hd.n_ck != ck_.size() || hd.blk_slot != blk_slot_ ||
      hd.ck_slot != ck_slot_) {
    std::fclose(f);
    return false;
  }
  for (uint32_t i = 0; i < blk_.size() && ok; ++i) ok = std::fread(&blk_[i], sizeof(Slot), 1, f) == 1;
  for (uint32_t i = 0; i < ck_.size() && ok; ++i) ok = std::fread(&ck_[i], sizeof(Slot), 1, f) == 1;
  std::fclose(f);
  if (!ok) {
    // A truncated index is not a partial index. Everything goes back to empty, with both eviction
    // lists rethreaded, because half a store is indistinguishable from a store that lies.
    blk_.assign(blk_.size(), Slot{});
    ck_.assign(ck_.size(), Slot{});
    thread_free_lists();
    return false;
  }
  impl_->blk_head = hd.blk_head; impl_->blk_tail = hd.blk_tail;
  impl_->ck_head = hd.ck_head;   impl_->ck_tail = hd.ck_tail;
  // Rebuild both chains from the slots themselves: the bucket arrays are derived state and storing
  // them would be one more thing that can disagree with what it indexes.
  std::fill(impl_->blk_bucket.begin(), impl_->blk_bucket.end(), kNil);
  std::fill(impl_->ck_bucket.begin(), impl_->ck_bucket.end(), kNil);
  for (uint32_t i = 0; i < blk_.size(); ++i) {
    blk_[i].hnext = kNil;
    if (!blk_[i].used) continue;
    uint32_t b; std::memcpy(&b, blk_[i].key, 4); b &= impl_->blk_mask;
    blk_[i].hnext = impl_->blk_bucket[b];
    impl_->blk_bucket[b] = i;
  }
  for (uint32_t i = 0; i < ck_.size(); ++i) {
    ck_[i].hnext = kNil;
    if (!ck_[i].used) continue;
    uint32_t b; std::memcpy(&b, ck_[i].parent, 4); b &= impl_->ck_mask;
    ck_[i].hnext = impl_->ck_bucket[b];
    impl_->ck_bucket[b] = i;
  }
  return true;
}

}  // namespace aff
