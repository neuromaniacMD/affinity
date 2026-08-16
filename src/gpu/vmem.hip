#include "gpu/vmem.h"

#include <algorithm>
#include <utility>

namespace aff {
namespace {

hipMemAllocationProp device_prop(int dev) {
  hipMemAllocationProp p = {};
  p.type = hipMemAllocationTypePinned;
  p.location.type = hipMemLocationTypeDevice;
  p.location.id = dev;
  return p;
}

std::string hip_err(const std::string& what, hipError_t e) {
  return what + ": " + hipGetErrorString(e);
}

}  // namespace

bool VirtualSpan::granularity_for(int dev, size_t* out, std::string* err) {
  const hipMemAllocationProp p = device_prop(dev);
  const hipError_t e =
      hipMemGetAllocationGranularity(out, &p, hipMemAllocationGranularityRecommended);
  if (e != hipSuccess) {
    if (err) *err = hip_err("hipMemGetAllocationGranularity", e);
    return false;
  }
  if (!*out) {
    if (err) *err = "hipMemGetAllocationGranularity returned 0";
    return false;
  }
  return true;
}

VirtualSpan::VirtualSpan(VirtualSpan&& o) noexcept { *this = std::move(o); }

VirtualSpan& VirtualSpan::operator=(VirtualSpan&& o) noexcept {
  if (this != &o) {
    release();
    dev_ = o.dev_; base_ = o.base_; reserved_ = o.reserved_;
    committed_ = o.committed_; gran_ = o.gran_; chunks_ = std::move(o.chunks_);
    o.dev_ = -1; o.base_ = nullptr; o.reserved_ = o.committed_ = o.gran_ = 0;
    o.chunks_.clear();
  }
  return *this;
}

VirtualSpan::~VirtualSpan() { release(); }

bool VirtualSpan::reserve(int dev, size_t bytes, std::string* err) {
  release();
  if (!granularity_for(dev, &gran_, err)) return false;
  reserved_ = (bytes + gran_ - 1) / gran_ * gran_;
  if (!reserved_) reserved_ = gran_;
  void* p = nullptr;
  // Address 0 and flags 0: let the driver place it. A fixed hint buys nothing here and fails on a
  // busy address space for no reason.
  const hipError_t e = hipMemAddressReserve(&p, reserved_, 0, nullptr, 0);
  if (e != hipSuccess) {
    if (err) *err = hip_err("hipMemAddressReserve of " + std::to_string(reserved_ >> 20) + " MiB", e);
    reserved_ = 0;
    return false;
  }
  dev_ = dev;
  base_ = static_cast<char*>(p);
  committed_ = 0;
  return true;
}

bool VirtualSpan::commit_to(size_t bytes, hipStream_t zero_on, std::string* err) {
  if (!base_) { if (err) *err = "commit_to on an unreserved span"; return false; }
  size_t want = (bytes + gran_ - 1) / gran_ * gran_;
  if (want > reserved_) want = reserved_;
  if (want <= committed_) return true;

  const hipMemAllocationProp prop = device_prop(dev_);
  hipMemAccessDesc access = {};
  access.location = prop.location;
  access.flags = hipMemAccessFlagsProtReadWrite;

  // THE WHOLE GROWTH IN ONE MAPPING, not one per granularity unit. hipMemMap and hipMemSetAccess
  // rebuild page tables over the range, so a loop that adds one unit at a time is quadratic in the
  // unit count: at 65536 positions that is ~135 units and unnoticeable, at 1M it is thousands and
  // the process sits at 100% of one core for a quarter of an hour with the cards idle. hipMemCreate
  // takes any multiple of the granularity, so there is no reason to subdivide.
  const size_t off = committed_, add = want - committed_;
  hipMemGenericAllocationHandle_t h{};
  hipError_t e = hipMemCreate(&h, add, &prop, 0);
  if (e != hipSuccess) {
    if (err) *err = hip_err("hipMemCreate of " + std::to_string(add >> 20) + " MiB", e) +
                    " with " + std::to_string(off >> 20) + " MiB already backed";
    return false;
  }
  e = hipMemMap(base_ + off, add, 0, h, 0);
  if (e != hipSuccess) {
    (void)hipMemRelease(h);
    if (err) *err = hip_err("hipMemMap", e);
    return false;
  }
  // OVER THE WHOLE MAPPED PREFIX, not just the new part. This driver refuses hipMemSetAccess on a
  // second range inside a reservation — every growth of an already-mapped span returns "invalid
  // argument", at any offset and any alignment (measured). Restating access for the whole
  // contiguous prefix is what it accepts, and it is one call per growth rather than per page.
  e = hipMemSetAccess(base_, want, &access, 1);
  if (e != hipSuccess) {
    (void)hipMemUnmap(base_ + off, add);
    (void)hipMemRelease(h);
    if (err) *err = hip_err("hipMemSetAccess", e);
    return false;
  }
  // Zeroed on arrival, because the buffers this backs are read before they are fully written —
  // attention masks a partially filled cache by row count, not by content. ASYNC, on this span's
  // own non-blocking stream: the blocking form synchronises the device, and 86 of those a growth
  // from the grower thread is a stall on the dispatch. Waited for here, so the memory is zero
  // before commit_to returns and the caller can publish it.
  e = hipMemsetD8Async(reinterpret_cast<hipDeviceptr_t>(base_ + off), 0, add, zero_on);
  if (e == hipSuccess) e = hipStreamSynchronize(zero_on);
  if (e != hipSuccess) {
    (void)hipMemUnmap(base_ + off, add);
    (void)hipMemRelease(h);
    if (err) *err = hip_err("hipMemsetD8Async", e);
    return false;
  }
  chunks_.push_back({h, add});
  committed_ = want;
  return true;
}

bool VirtualSpan::decommit_to(size_t bytes, std::string* err) {
  if (!base_) return true;
  // Whole mappings only: a chunk is released when the target falls below its base, never split.
  // The caller asking for a size inside a chunk keeps that chunk, so this is conservative — it
  // never unmaps memory the caller might still be reading.
  const size_t want = (bytes + gran_ - 1) / gran_ * gran_;
  while (!chunks_.empty() && committed_ - chunks_.back().second >= want) {
    const size_t sz = chunks_.back().second;
    const size_t off = committed_ - sz;
    const hipError_t e = hipMemUnmap(base_ + off, sz);
    if (e != hipSuccess) { if (err) *err = hip_err("hipMemUnmap", e); return false; }
    (void)hipMemRelease(chunks_.back().first);
    chunks_.pop_back();
    committed_ = off;
  }
  return true;
}

void VirtualSpan::release() {
  if (base_) {
    std::string ignored;
    (void)decommit_to(0, &ignored);
    (void)hipMemAddressFree(base_, reserved_);
  }
  dev_ = -1;
  base_ = nullptr;
  reserved_ = committed_ = gran_ = 0;
  chunks_.clear();
}

}  // namespace aff
