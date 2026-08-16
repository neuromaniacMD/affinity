// aff-info — dump the structure of a .aff container.

#include "format/aff_reader.h"

#include <cstdio>
#include <cinttypes>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <cstdlib>

using namespace aff;

static const char* human(uint64_t b, char* buf, size_t n) {
  const char* u[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double v = (double)b; int i = 0;
  while (v >= 1024.0 && i < 4) { v /= 1024.0; ++i; }
  std::snprintf(buf, n, "%.2f %s", v, u[i]);
  return buf;
}

// Replicates the decode hot path: per step, top-k random experts per layer, touched end to end.
// This is the access pattern that makes page size matter — experts are megabytes apart and chosen
// at random, so at 4 KiB pages one expert needs thousands of TLB entries.
static void bench_access(AffReader& r, int steps, int topk) {
  const uint64_t layers = r.layer_count();
  std::mt19937_64 rng(20260729);
  volatile uint64_t sink = 0;
  uint64_t bytes = 0;

  // Warm-up pass so mmap mode is not measured cold against a pre-read huge-page copy.
  for (uint64_t l = 0; l < layers; ++l) {
    const AffLayerDesc& L = r.layer((uint32_t)l);
    for (int k = 0; k < topk; ++k) {
      const uint8_t* p = r.expert_ptr((uint32_t)l, (uint32_t)(rng() % L.n_experts));
      for (uint64_t o = 0; o < L.stride; o += 4096) sink += p[o];
    }
  }

  const auto t0 = std::chrono::steady_clock::now();
  for (int s = 0; s < steps; ++s) {
    for (uint64_t l = 0; l < layers; ++l) {
      const AffLayerDesc& L = r.layer((uint32_t)l);
      for (int k = 0; k < topk; ++k) {
        const uint8_t* p = r.expert_ptr((uint32_t)l, (uint32_t)(rng() % L.n_experts));
        // Touch every cacheline — this is a full weight read, as the GEMV would do.
        for (uint64_t o = 0; o < L.stride; o += 64) sink += p[o];
        bytes += L.stride;
      }
    }
  }
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  (void)sink;

  std::printf("\naccess benchmark (%d steps x %" PRIu64 " layers x top-%d)\n", steps, layers, topk);
  std::printf("  read        %.2f GiB in %.3f s\n", bytes / 1073741824.0, sec);
  std::printf("  bandwidth   %.1f GB/s\n", bytes / sec / 1e9);
  std::printf("  per step    %.2f ms  -> %.1f tok/s if this were the only cost\n",
              sec / steps * 1e3, steps / sec);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
      "usage: aff-info <file.aff> [--experts] [--bench N] [--topk K]\n");
    return 2;
  }
  bool show_experts = false;
  int  bench_steps = 0;
  int  topk = 6;                 // experts a token routes to; --bench reads that many per layer
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if      (a == "--experts")   show_experts = true;
    else if (a == "--bench")     bench_steps = (i + 1 < argc) ? std::atoi(argv[++i]) : 10;
    else if (a == "--topk")      topk = (i + 1 < argc) ? std::atoi(argv[++i]) : 6;
  }
  if (topk < 1) topk = 1;

  AffReader r;
  std::string err;
  if (!r.open(argv[1], &err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  const AffHeader& h = r.header();
  char b[64];

  std::printf("file            %s\n", argv[1]);
  std::printf("version         %u\n", h.version);
  std::printf("size            %s\n", human(h.file_size, b, sizeof(b)));
  std::printf("flags           %s\n", (h.flags & AFF_FLAG_HAS_PROFILE) ? "profile" : "none");
  std::printf("metadata        %" PRIu64 " bytes\n", h.meta_size);
  if (h.meta_size && h.meta_size < 2048)
    std::printf("                %.*s\n", (int)h.meta_size, r.meta().data());

  std::printf("\nnon-expert tensors (%" PRIu64 ")\n", h.tensor_count);
  std::printf("  %-40s %12s %12s %-10s %10s %14s\n", "name", "rows", "cols", "quant", "bytes",
              "file offset");
  const auto* tents = reinterpret_cast<const AffTensorEntry*>(
      reinterpret_cast<const uint8_t*>(&h) + h.tensor_dir_off);
  uint64_t tensor_bytes = 0;
  for (uint64_t i = 0; i < h.tensor_count; ++i) {
    const auto& d = tents[i].desc;
    const uint64_t bytes = d.data_size + d.scale_size;
    tensor_bytes += bytes;
    // The offset a `dd` would use, so a tensor can be extracted or two of them compared without
    // going through the reader.
    std::printf("  %-40.128s %12" PRIu64 " %12" PRIu64 " %-10s %10s %14" PRIu64 "\n",
                tents[i].name, d.rows, d.cols,
                quant_name(static_cast<AffQuant>(d.quant)), human(bytes, b, sizeof(b)),
                h.tensor_data_off + tents[i].blob_off + d.data_off);
  }
  std::printf("  total non-expert: %s\n", human(tensor_bytes, b, sizeof(b)));

  std::printf("\nexpert pool: %s at offset %" PRIu64 " (%s aligned)\n",
              human(h.expert_pool_size, b, sizeof(b)), h.expert_pool_off,
              (h.expert_pool_off % kPoolAlign == 0) ? "2MiB" : "UNALIGNED!");
  std::printf("  %-6s %8s %-10s %-10s %14s %14s\n",
              "layer", "experts", "gate/up", "down", "stride", "layer total");
  uint64_t total_experts = 0;
  for (uint64_t i = 0; i < h.layer_count; ++i) {
    const AffLayerDesc& L = r.layer((uint32_t)i);
    total_experts += L.n_experts;
    char b2[64];
    std::printf("  %-6" PRIu64 " %8u %-10s %-10s %14s %14s\n", i, L.n_experts,
                quant_name(static_cast<AffQuant>(L.gate.quant)),
                quant_name(static_cast<AffQuant>(L.down.quant)),
                human(L.stride, b, sizeof(b)),
                human(L.stride * L.n_experts, b2, sizeof(b2)));
  }
  std::printf("  %" PRIu64 " experts across %" PRIu64 " layers\n", total_experts, h.layer_count);

  if (const float* p = r.profile()) {
    std::printf("\nexpert popularity profile: present\n");
    // Report skew — the number that decides the whole residency strategy.
    for (uint64_t l = 0; l < h.layer_count && l < 4; ++l) {
      const AffLayerDesc& L = r.layer((uint32_t)l);
      double sum = 0, mx = 0;
      for (uint32_t e = 0; e < L.n_experts; ++e) { sum += p[e]; if (p[e] > mx) mx = p[e]; }
      std::printf("  layer %" PRIu64 ": max/mean = %.2fx\n", l,
                  sum > 0 ? mx / (sum / L.n_experts) : 0.0);
      p += L.n_experts;
    }
  }

  if (show_experts) {
    std::printf("\nexpert 0 layout (layer 0)\n");
    const AffLayerDesc& L = r.layer(0);
    auto row = [&](const char* n, const AffMatrixDesc& m) {
      std::printf("  %-6s [%5" PRIu64 " x %5" PRIu64 "] %-8s  data@%-9" PRIu64 "%9" PRIu64 "B"
                  "  scale@%-9" PRIu64 "%7" PRIu64 "B  rscale@%-9" PRIu64 "%6" PRIu64 "B\n",
                  n, m.rows, m.cols, quant_name(static_cast<AffQuant>(m.quant)),
                  m.data_off, m.data_size, m.scale_off, m.scale_size,
                  m.rscale_off, m.rscale_size);
    };
    row("gate", L.gate); row("up", L.up); row("down", L.down);
    std::printf("  stride %" PRIu64 " B\n", L.stride);
  }

  if (bench_steps > 0) bench_access(r, bench_steps, topk);
  return 0;
}
