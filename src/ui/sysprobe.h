// Machine state for the dashboard: per-card utilisation from amdgpu's sysfs nodes, and host memory
// from /proc.
//
// NO HIP IN HERE, on purpose. Everything on this path is read by the UI thread, and a HIP call from
// a second thread takes the driver's lock — which the dispatch thread is holding whenever it is
// doing anything. Ten of those a second is not much, but "the instrument perturbs the thing" is a
// mistake this project has already paid for, and the kernel exports every number the panel wants.
// The driver hands over each device's PCI address once at startup and nothing else crosses.
//
// GPU BUSY IS SAMPLED FAST AND AVERAGED SLOWLY. `gpu_busy_percent` is an instantaneous read of the
// hardware's activity bit, not an interval average, so its value depends entirely on how often it
// is asked: at 1 Hz decode reads as ~92% device-bound, at 20 Hz as ~58%. So `tick()` runs an order
// of magnitude faster than the frame rate and `roll()` returns the mean since the last frame.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace aff::ui {

struct DeviceInfo {
  std::string name;      // as the driver reports it
  std::string bus_id;    // "0000:03:00.0"
  std::string arch;      // gfx1201
  int cus = 0;           // compute units, not WGPs
  double vram_total_gib = 0;
};

struct DeviceSample {
  bool present = false;   // a sysfs node was found and read at least once
  double busy = -1;       // percent, mean over the frame; negative when the node is absent
  double mem_busy = -1;
  double vram_used_gib = -1;
  double temp_c = -1;
  double power_w = -1;
  double sclk_mhz = -1;
  double mclk_mhz = -1;
};

struct HostSample {
  double mem_total_gib = 0, mem_avail_gib = 0;
  double rss_gib = 0, rss_peak_gib = 0;
};

class Probe {
 public:
  ~Probe();
  // Resolves the card's sysfs directory from its PCI address. A device that cannot be resolved is
  // still added — the panel shows its name and geometry with the meters blank, which is the honest
  // rendering on a kernel that does not export them.
  void add_device(const DeviceInfo& d);
  size_t devices() const { return dev_.size(); }
  const DeviceInfo& info(size_t i) const { return dev_[i].info; }

  // Cheap: one pread per open node. Safe to call at 50 Hz.
  void tick();
  // Mean since the last call, then reset. Sizes `out` to devices().
  void roll(std::vector<DeviceSample>* out);

  // /proc, re-read at most every 250 ms however often this is called.
  const HostSample& host();

 private:
  struct Node {
    int fd = -1;
    double scale = 1.0;
  };
  struct Dev {
    DeviceInfo info;
    std::string sysfs;
    Node busy, mem_busy, vram_used, temp, power, sclk, mclk;
    // Accumulators, reset by roll(). `n_*` differs per node because a card may export some and not
    // others, and dividing a present node's sum by an absent one's count reports zero.
    double s_busy = 0, s_mem = 0, s_vram = 0, s_temp = 0, s_power = 0, s_sclk = 0, s_mclk = 0;
    uint32_t n_busy = 0, n_mem = 0, n_vram = 0, n_temp = 0, n_power = 0, n_sclk = 0, n_mclk = 0;
  };
  std::vector<Dev> dev_;
  HostSample host_{};
  uint64_t host_at_ms_ = 0;
};

}  // namespace aff::ui
