// A heartbeat that keeps each card's dispatch path warm.
//
// THE EFFECT. A kernel dispatched to a card that has received no dispatch for roughly half a
// millisecond costs several times what the same kernel costs on a card dispatched to recently.
// Decode at batch 1 hands the CPU tens of milliseconds of expert arithmetic per token, so a card can
// cross that threshold several times a token and pay on the next call. A heartbeat recovers most of
// the penalty for a small fraction of one compute unit; see `bench/roundtrip.hip`.
//
// IT IS OFF BY DEFAULT, because in this engine it is a small net loss: the idle windows sit just
// UNDER the threshold, so there is little to recover, while the heartbeat threads wake thousands of
// times a second and contend with the expert pool on a machine whose cores are already busy.
// Anything that lengthens the host's gap between dispatches pushes it over, and then it is worth
// switching on with `--keepalive-us`.
//
// The cost tracks HOST DISPATCH traffic, not clocks and not occupancy, which is also why CONTINUOUS
// IS WORSE THAN PERIODIC: past the point where the path is warm a spinning block only competes with
// the kernel it exists to accelerate. The signal wanted is "a dispatch happened recently".

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace aff {

class GpuKeepalive {
public:
  ~GpuKeepalive();
  GpuKeepalive() = default;
  GpuKeepalive(const GpuKeepalive&) = delete;
  GpuKeepalive& operator=(const GpuKeepalive&) = delete;

  // One thread and one lowest-priority stream per device. Call AFTER the weights are uploaded:
  // during load the cards are saturated anyway and this would only take bandwidth from the upload.
  // `interval_us` must stay under the dispatch-idle threshold with room for scheduling jitter.
  // Returns false if already running or `devices` is empty; a device that refuses is skipped, since
  // failing to keep a card warm is a performance matter and never a correctness one.
  bool start(const std::vector<int>& devices, double interval_us = 250.0);
  void stop();

  bool running() const { return running_; }

private:
  struct Impl;
  Impl* impl_ = nullptr;
  bool running_ = false;
};

} // namespace aff
