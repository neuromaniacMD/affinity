// A persistent worker pool for the decode path.
//
// Why a pool rather than std::async per op: a single token touches ~40 parallel regions per layer
// (projections, experts, attention heads, the head), 43 layers deep. At ~1700 regions per token,
// thread creation would cost more than the arithmetic.
//
// The hot path is ATOMICS ONLY. As a condition-variable pool, `notify_all` woke every sleeper, each
// of which had to reacquire one mutex to leave `wait()` and take it again on the way out to
// decrement the counter — two N-way convoys through a single lock per region, which at this region
// count costs milliseconds a token for regions that may be empty. Publishing the region with one
// release store and joining on one atomic counter removes both.
//
// Deliberately NOT nested: `parallel_for` called from inside a worker runs serially on the calling
// thread. Every parallel region in the engine is over rows of one matrix, so nesting would only
// oversubscribe.

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace aff {

class ThreadPool {
public:
  explicit ThreadPool(unsigned n_threads);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  unsigned size() const { return nthreads_; }

  // Splits [0, n) into `size()` contiguous chunks and runs `fn(begin, end)` on each. Blocks until
  // all chunks finish. Contiguous rather than strided so each worker sweeps a contiguous span of
  // weights — the access pattern the prefetchers and the 2 MiB pages are set up for.
  void parallel_for(uint64_t n, const std::function<void(uint64_t, uint64_t)>& fn);

private:
  void worker_loop(unsigned id);

  // Fixed before the first worker is spawned, and read by every worker on entry. NOT derived from
  // `workers_.size()`: that vector is still being emplaced into while the threads it has already
  // created are running, so worker 1 would read 2 and worker 15 would read 16, and each would keep
  // its wrong value for life. The result is a correct answer computed with wildly uneven chunks —
  // worker 1 taking half the index space — which shows up as a large bandwidth loss on
  // a wide parallel region and not at all on the empty-region case.
  const unsigned nthreads_;
  std::vector<std::thread> workers_;

  // --- published under `epoch_`'s release; read after its acquire -------------------------------
  const std::function<void(uint64_t, uint64_t)>* fn_ = nullptr;
  uint64_t n_ = 0;

  // --- the hot path ----------------------------------------------------------------------------
  alignas(64) std::atomic<uint64_t> epoch_{0};
  alignas(64) std::atomic<unsigned> remaining_{0};
  // Read by the dispatcher to decide whether a futex wake is needed at all. Sequentially consistent
  // against the epoch bump on both sides: this is Dekker, and acquire/release alone lets a worker
  // that has decided to sleep and a dispatcher that has decided not to wake it both be right.
  alignas(64) std::atomic<unsigned> sleepers_{0};

  // --- the sleep transition --------------------------------------------------------------------
  std::mutex mu_;
  std::condition_variable cv_start_;
  std::atomic<bool> stop_{false};
};

// The engine's pool, created on first use and sized from `hardware_concurrency()`.
//
// There is no environment override: a knob that silently does nothing is worse than no knob.
// Construct a ThreadPool directly to vary the count.
ThreadPool& global_pool();

// Convenience wrapper: runs serially for small `n`, where the barrier costs more than the work.
void parallel_for(uint64_t n, uint64_t min_parallel, const std::function<void(uint64_t, uint64_t)>& fn);

} // namespace aff
