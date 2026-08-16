#include "threadpool.h"

#include <algorithm>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define AFF_PAUSE() _mm_pause()
#else
#define AFF_PAUSE() ((void)0)
#endif

namespace aff {

namespace {

// Set while a worker is executing a region, so a nested parallel_for runs inline instead of
// deadlocking on a pool that is already fully occupied.
thread_local bool t_in_pool = false;

} // namespace

ThreadPool::ThreadPool(unsigned n_threads) : nthreads_(n_threads < 1 ? 1u : n_threads) {
  n_threads = nthreads_;
  // Workers do not spin before sleeping. The gap between two regions is the GPU dense chain, far
  // longer than any spin budget, so every worker would burn the whole budget every time — and with
  // SMT a spinning worker holds its sibling's issue slot away from a thread doing real work.
  // Throughput falls monotonically as the budget grows.
  workers_.reserve(n_threads - 1);
  for (unsigned i = 1; i < n_threads; ++i) workers_.emplace_back([this, i] { worker_loop(i); });
}

ThreadPool::~ThreadPool() {
  stop_.store(true, std::memory_order_release);
  epoch_.fetch_add(1, std::memory_order_release);
  { std::lock_guard<std::mutex> g(mu_); }            // pair with a worker mid-way into wait()
  cv_start_.notify_all();
  for (auto& t : workers_) if (t.joinable()) t.join();
}

void ThreadPool::worker_loop(unsigned id) {
  const unsigned nt = size();
  uint64_t seen = 0;
  for (;;) {
    uint64_t e = epoch_.load(std::memory_order_acquire);

    if (e == seen) {
      // Announce before re-reading the epoch: the dispatcher publishes the epoch and then reads
      // this counter, so between the two seq_cst pairs at least one side sees the other.
      sleepers_.fetch_add(1, std::memory_order_seq_cst);
      {
        std::unique_lock<std::mutex> lk(mu_);
        // seq_cst on this load, NOT acquire. It is the other half of the Dekker pair above: under
        // acquire the dispatcher may read sleepers_ as 0 and skip the wake while this worker reads
        // a stale epoch and blocks, and then `remaining_` never reaches zero and the caller spins
        // for ever. A hang, not a slowdown, so it is not a tuning question.
        cv_start_.wait(lk, [&] {
          return stop_.load(std::memory_order_acquire) ||
                 epoch_.load(std::memory_order_seq_cst) != seen;
        });
      }
      sleepers_.fetch_sub(1, std::memory_order_seq_cst);
      e = epoch_.load(std::memory_order_acquire);
    }

    if (stop_.load(std::memory_order_acquire)) return;
    seen = e;

    // fn_ and n_ were written before the epoch's release store.
    const auto* fn = fn_;
    const uint64_t n = n_;
    const uint64_t chunk = (n + nt - 1) / nt;
    const uint64_t lo = std::min(n, (uint64_t)id * chunk);
    const uint64_t hi = std::min(n, lo + chunk);
    if (hi > lo) {
      t_in_pool = true;
      (*fn)(lo, hi);
      t_in_pool = false;
    }
    remaining_.fetch_sub(1, std::memory_order_acq_rel);
  }
}

void ThreadPool::parallel_for(uint64_t n, const std::function<void(uint64_t, uint64_t)>& fn) {
  if (n == 0) return;
  const unsigned nt = size();
  if (nt == 1 || t_in_pool) { fn(0, n); return; }

  fn_ = &fn;
  n_ = n;
  remaining_.store((unsigned)workers_.size(), std::memory_order_relaxed);
  epoch_.fetch_add(1, std::memory_order_seq_cst);    // publishes fn_/n_/remaining_

  // Only pay the futex when somebody is actually asleep. The seq_cst on both this load and the
  // epoch bump is what makes "nobody is asleep" safe to believe.
  if (sleepers_.load(std::memory_order_seq_cst)) {
    // Under the lock: a worker that has evaluated the predicate but not yet blocked would
    // otherwise miss the notification and sleep until the next region.
    { std::lock_guard<std::mutex> g(mu_); }
    cv_start_.notify_all();
  }

  // Thread 0 is the caller: it takes the first chunk rather than idling at the barrier.
  const uint64_t chunk = (n + nt - 1) / nt;
  const uint64_t hi = std::min(n, chunk);
  if (hi > 0) {
    t_in_pool = true;
    fn(0, hi);
    t_in_pool = false;
  }

  // Join by spinning on the counter, then yielding. There is no condition variable on this side on
  // purpose: notifying it would put every worker back through the same mutex the rewrite removed,
  // and the caller has nothing else to do anyway. The yield is for the case where a worker was
  // descheduled, which is rare and must not become a busy wait.
  unsigned spins = 0;
  while (remaining_.load(std::memory_order_acquire) != 0) {
    if (++spins < 4096) AFF_PAUSE();
    else std::this_thread::yield();
  }
}

ThreadPool& global_pool() {
  static ThreadPool pool([] {
    // Every logical core, not every physical one. SMT siblings do contend for the load/store unit,
    // but on this kernel that is a wash rather than a loss, and halving the count leaves half the
    // machine out of the expert share. The pool sleeps when idle, so the extra threads cost nothing
    // when the engine is not decoding.
    const unsigned hc = std::thread::hardware_concurrency();
    return hc > 1 ? hc : 1u;
  }());
  return pool;
}

void parallel_for(uint64_t n, uint64_t min_parallel, const std::function<void(uint64_t, uint64_t)>& fn) {
  if (n < min_parallel) { fn(0, n); return; }
  global_pool().parallel_for(n, fn);
}

} // namespace aff
