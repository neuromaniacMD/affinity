// HIP device plumbing for RDNA4 (gfx12). Deliberately thin: there is one target architecture, so
// there is no device abstraction and no backend dispatch.
//
// Three gfx12 facts that shape everything in src/gpu/:
//
//   * wave32 deliberately. RDNA can run wave64, and several HIP samples use width-less `__shfl_*`,
//     which silently means "whatever the wave is". Every cross-lane op here passes an explicit
//     width of 32 and every launch assumes 32 lanes.
//   * No async global->LDS copy. There is no `cp.async` equivalent; staging through LDS costs a
//     real round trip, so it only pays for data reused many times (the codebook, the activation).
//   * 256 B cachelines, not 128. Coalescing targets are twice as wide as CUDA intuition suggests.
//
// NEVER iterate 0..hipGetDeviceCount(); always go through gfx12_devices(). An integrated GPU is
// enumerated alongside the discrete cards and reports system RAM as its "VRAM", so a capacity check
// will not rule it out — and a launch that lands on a device this binary has no code objects for
// dies inside libamdhip64 with a SEGV rather than a clean error.
//
// One naming trap: hipDeviceProp_t::multiProcessorCount counts Work Group Processors on RDNA, and a
// WGP is two CUs. A part that rocminfo calls 64 CUs reports 32 here. See device_cus().

#pragma once

#include <hip/hip_runtime.h>

#include "ui/log.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace aff {

// Upper bound for per-device static tables. Sized above any configuration the engine supports so
// the tables are never resized; the live device count comes from gfx12_devices().
constexpr int kMaxGpuDevices = 8;

#define AFF_HIP_CHECK(expr)                                                              \
  do {                                                                                   \
    const hipError_t aff_err_ = (expr);                                                  \
    if (aff_err_ != hipSuccess) {                                                        \
      aff::ui::fatal("%s:%d: HIP error: %s (%s)\n", __FILE__, __LINE__,                  \
                     hipGetErrorString(aff_err_), #expr);                                \
      std::abort();                                                                      \
    }                                                                                    \
  } while (0)

// Non-fatal variant for probing: returns false and fills `err` instead of aborting.
inline bool hip_try(hipError_t e, const char* what, std::string* err) {
  if (e == hipSuccess) return true;
  if (err) *err = std::string(what) + ": " + hipGetErrorString(e);
  return false;
}

inline constexpr int kWave = 32;   // RDNA4 wave32; checked against the device, never inferred

// COMPUTE UNITS, WHICH IS NOT WHAT hipDeviceProp_t REPORTS. On RDNA `multiProcessorCount` counts
// Work Group Processors and a WGP is two CUs, so a 64-CU part reports 32. Taking that field as a
// CU count halves every budget derived from it and, worse, halves the width of a CU mask — a
// 64-CU part silently gets one covering only its first 32. The doubling is RDNA-specific: on CDNA
// the field already is the CU count.
inline uint32_t device_cus(int dev) {
  hipDeviceProp_t pr{};
  if (hipGetDeviceProperties(&pr, dev) != hipSuccess || pr.multiProcessorCount <= 0) {
    // No fallback. Guessing a CU count here silently mis-sizes every grid derived from it, and the
    // wrong answer is a slow run rather than an error anyone would notice.
    aff::ui::fatal("device %d: cannot read multiProcessorCount\n", dev);
    std::abort();
  }
  return (uint32_t)pr.multiProcessorCount * 2u;
}

// Total HIP devices, INCLUDING the iGPU. Rarely what you want — see gfx12_devices().
inline int hip_device_count() {
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess) return 0;
  return n;
}

// The architecture this binary holds code objects for. CMake defines it; the fallback keeps the
// header usable in a translation unit compiled without it.
#ifndef AFF_GPU_ARCH
#define AFF_GPU_ARCH "gfx1201"
#endif

// Indices of the devices this build can run on. This is the only enumeration the engine should
// use: `hipSetDevice(0)` is right only by luck, and breaks the moment an integrated GPU sorts
// first.
//
// The match is against the compiled arch exactly, not the gfx12 family. A family prefix accepts
// cards this binary has no code objects for — gfx1200 is a shipping RDNA4 part — and the failure
// is a SEGV inside libamdhip64 on the first launch rather than a diagnosable error.
inline std::vector<int> gfx12_devices() {
  std::vector<int> out;
  const int n = hip_device_count();
  for (int i = 0; i < n; ++i) {
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, i) != hipSuccess) continue;
    // gcnArchName carries feature suffixes (":sramecc+:xnack-"); compare only the base name.
    std::string arch(p.gcnArchName);
    arch = arch.substr(0, arch.find(':'));
    if (arch != AFF_GPU_ARCH) continue;
    // Every cross-lane op in src/gpu/ passes an explicit width of kWave and every launch assumes
    // that many lanes. A wave64 device would compute the wrong thing quietly, so refuse it here.
    if (p.warpSize != kWave) {
      aff::ui::err("device %d (%s) reports wave%d; this build assumes wave%d. Skipping.\n",
                   i, arch.c_str(), p.warpSize, kWave);
      continue;
    }
    out.push_back(i);
  }
  return out;
}

// Supported devices found, and every device that was rejected and why. Called once at startup so a
// user whose card is skipped is told which one and what the binary was built for, rather than
// meeting "no gfx12 device" with two cards installed.
inline void report_rejected_devices() {
  const int n = hip_device_count();
  for (int i = 0; i < n; ++i) {
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, i) != hipSuccess) continue;
    std::string arch(p.gcnArchName);
    arch = arch.substr(0, arch.find(':'));
    if (arch != AFF_GPU_ARCH)
      aff::ui::err("device %d (%s) skipped: this build targets %s. Rebuild with "
                   "-DAFF_GPU_ARCH=%s to use it.\n", i, arch.c_str(), AFF_GPU_ARCH, arch.c_str());
  }
}

// Makes the first gfx12 device current. Returns false when there is none.
inline bool hip_select_device(int which = 0) {
  const std::vector<int> d = gfx12_devices();
  if ((size_t)which >= d.size()) return false;
  return hipSetDevice(d[which]) == hipSuccess;
}

struct DeviceInfo {
  int         index = 0;
  std::string arch;
  size_t      vram_total = 0;
  size_t      vram_free  = 0;
  int         cus = 0;
  int         lds_per_block = 0;
};

// Queries device `i`. Restores the previously-current device: reading free VRAM requires making a
// device current, and leaving that in place turned an informational summary into a trap that
// silently retargeted every subsequent allocation and launch at the iGPU.
inline bool hip_device_info(int i, DeviceInfo* out, std::string* err) {
  hipDeviceProp_t p{};
  if (!hip_try(hipGetDeviceProperties(&p, i), "hipGetDeviceProperties", err)) return false;
  out->index = i;
  out->arch = p.gcnArchName;
  out->cus = (int)device_cus(i);
  out->lds_per_block = (int)p.sharedMemPerBlock;
  int prev = 0;
  const bool have_prev = hipGetDevice(&prev) == hipSuccess;
  if (hipSetDevice(i) == hipSuccess) (void)hipMemGetInfo(&out->vram_free, &out->vram_total);
  if (have_prev) (void)hipSetDevice(prev);
  return true;
}

// RAII device buffer. No pooling: allocation happens at load time, not on the decode path.
template <typename T>
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t n) { alloc(n); }
  ~DeviceBuffer() { free(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }

  // `pad_bytes` over-allocates so a kernel may read a few bytes past the logical end. The packed
  // formats are read through 64-bit windows at arbitrary bit offsets, so the last window of a
  // buffer legitimately straddles the end.
  void alloc(size_t n, size_t pad_bytes = 0) {
    free();
    AFF_HIP_CHECK(hipMalloc(&p_, n * sizeof(T) + pad_bytes));
    AFF_HIP_CHECK(hipMemset(p_, 0, n * sizeof(T) + pad_bytes));
    n_ = n;
  }
  void upload(const T* src, size_t n) {
    AFF_HIP_CHECK(hipMemcpy(p_, src, n * sizeof(T), hipMemcpyHostToDevice));
  }
  void download(T* dst, size_t n) const {
    AFF_HIP_CHECK(hipMemcpy(dst, p_, n * sizeof(T), hipMemcpyDeviceToHost));
  }
  void free() {
    if (p_) { (void)hipFree(p_); p_ = nullptr; n_ = 0; }
  }
  T* get() const { return p_; }
  size_t size() const { return n_; }

private:
  T* p_ = nullptr;
  size_t n_ = 0;
};

} // namespace aff
