#include "gpu/keepalive.h"
#include "gpu/hip_common.h"

#include <atomic>
#include <chrono>
#include <ctime>
#include <thread>

namespace aff {

namespace {
// Pure arithmetic: no loads, no stores on the taken path. A memory-touching keepalive would work
// equally well but would compete for the bandwidth decode is entirely made of.
//
// The store is unreachable — `a` starts non-negative and only grows — but the compiler cannot prove
// it, so the loop survives optimisation. `kIters` sets its duration.
__global__ void keepalive_kernel(float* __restrict__ sink, uint32_t iters) {
  float a = (float)threadIdx.x;
  for (uint32_t i = 0; i < iters; ++i) a = fmaf(a, 1.0000001f, 1.0f);
  if (a < 0.0f) sink[0] = a;
}

constexpr uint32_t kIters = 8000;

// nanosleep rather than a spin: the CPU is doing expert arithmetic on every core during exactly the
// window this covers, and taking a core from it to save a few microseconds of GPU dispatch would be
// a straight loss. Overshoot is fine: the interval is set below the threshold with margin.
void sleep_us(double us) {
  if (us <= 0) return;
  timespec ts{};
  ts.tv_sec = 0;
  ts.tv_nsec = (long)(us * 1000.0);
  nanosleep(&ts, nullptr);
}
} // namespace

struct GpuKeepalive::Impl {
  std::atomic<bool> stop{false};
  std::vector<std::thread> th;
};

GpuKeepalive::~GpuKeepalive() { stop(); }

bool GpuKeepalive::start(const std::vector<int>& devices, double interval_us) {
  if (running_ || devices.empty()) return false;
  impl_ = new Impl();
  for (int dev : devices) {
    impl_->th.emplace_back([this, dev, interval_us] {
      if (hipSetDevice(dev) != hipSuccess) return;
      // Lowest priority the driver offers, so the heartbeat can never take a turn ahead of real
      // work. It exists to fill idle time, not to compete for the device.
      int lo = 0, hi = 0;
      (void)hipDeviceGetStreamPriorityRange(&lo, &hi);
      hipStream_t st;
      if (hipStreamCreateWithPriority(&st, hipStreamNonBlocking, lo) != hipSuccess) return;
      float* sink = nullptr;
      if (hipMalloc(&sink, sizeof(float)) != hipSuccess) { (void)hipStreamDestroy(st); return; }
      // Sleep for the interval MINUS what the launch and synchronise already cost. Sleeping the
      // full interval on top of the round trip would drift past the threshold on a loaded machine,
      // which is the one failure mode that makes this useless.
      while (!impl_->stop.load(std::memory_order_relaxed)) {
        const auto t0 = std::chrono::steady_clock::now();
        hipLaunchKernelGGL(keepalive_kernel, dim3(1), dim3(256), 0, st, sink, kIters);
        if (hipStreamSynchronize(st) != hipSuccess) break;
        const double spent =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e6;
        sleep_us(interval_us - spent);
      }
      (void)hipFree(sink);
      (void)hipStreamDestroy(st);
    });
  }
  running_ = true;
  return true;
}

void GpuKeepalive::stop() {
  if (!impl_) return;
  impl_->stop.store(true, std::memory_order_relaxed);
  for (auto& t : impl_->th) if (t.joinable()) t.join();
  delete impl_;
  impl_ = nullptr;
  running_ = false;
}

} // namespace aff
