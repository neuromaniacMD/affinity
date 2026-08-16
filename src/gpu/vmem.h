// A device allocation whose ADDRESS is fixed and whose physical backing is not.
//
// The KV cache is sized for the context the engine can serve, not the one it is serving. At
// --kv-size 1048576 that reservation is ~4.3 GiB a card, and because placement measures free VRAM
// after it, those bytes come straight out of the expert slab: 4334 resident experts instead of
// 5296. Measured on the card, that costs 19% of decode — and handing the same VRAM back with
// --gpu-headroom-mib restores the loss exactly, so it is residency and not a cheaper attention.
//
// The obvious fix — allocate the cache in pages and address it through a block table — is a rewrite
// of every attention kernel. This is the other one: reserve the whole virtual range up front so the
// kernels keep their linear addressing and their pointer arithmetic, and map physical pages under
// it as the sequence grows. Nothing above this file learns that the memory arrived late.
//
// GROWTH IS NEVER ON THE DISPATCH PATH. hipMemCreate and hipMemMap are driver calls that
// synchronise, and the pages they need have to come from somewhere — which means evicting experts.
// So growth is watermark-driven from the placement thread, far enough ahead that the position
// needing the memory arrives long after it does. `commit_to` is safe to call from that thread while
// the dispatch thread reads the span, because a mapping only ever appears ABOVE the committed
// high-water mark and nothing below it moves.

#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>

namespace aff {

class VirtualSpan {
 public:
  VirtualSpan() = default;
  ~VirtualSpan();
  VirtualSpan(const VirtualSpan&) = delete;
  VirtualSpan& operator=(const VirtualSpan&) = delete;
  VirtualSpan(VirtualSpan&&) noexcept;
  VirtualSpan& operator=(VirtualSpan&&) noexcept;

  // Reserves `bytes` of address space on `dev` and maps nothing. Rounds up to the allocation
  // granularity, which is what every offset below is a multiple of.
  bool reserve(int dev, size_t bytes, std::string* err);

  // Grows the mapped prefix to at least `bytes`, zeroing whatever is newly mapped on `zero_on`.
  // Cheap and a no-op when the prefix already covers it, so a watermark check can call it every
  // tick.
  //
  // THE STREAM IS THE CALLER'S, one per device and not one per span. It only needs to be a stream
  // that is not the null one — hipMemsetD8 is blocking and the null stream synchronises with every
  // other, so zeroing there from the grower thread stalls the dispatch. Owning a stream per span
  // instead cost 640 MiB of host RAM across 86 spans, which came straight out of the expert pool
  // and stranded experts on the SSD tier.
  bool commit_to(size_t bytes, hipStream_t zero_on, std::string* err);

  // Shrinks the mapped prefix, releasing the physical pages above `bytes` back to the driver. The
  // caller owns the claim that nothing is reading up there.
  bool decommit_to(size_t bytes, std::string* err);

  void* ptr() const { return base_; }
  size_t reserved() const { return reserved_; }
  size_t committed() const { return committed_; }
  size_t granularity() const { return gran_; }
  void release();

  // The driver's minimum mapping unit for device memory. Every reservation and commitment is a
  // multiple of it, so a caller sizing chunks should ask rather than assume a page size.
  static bool granularity_for(int dev, size_t* out, std::string* err);

 private:
  int dev_ = -1;
  char* base_ = nullptr;
  size_t reserved_ = 0, committed_ = 0, gran_ = 0;
  // One handle per mapped chunk with its size, in address order, so decommit releases exactly
  // the tail. A chunk is a whole growth step, not a granularity unit — see commit_to.
  std::vector<std::pair<hipMemGenericAllocationHandle_t, size_t>> chunks_;
};

}  // namespace aff
