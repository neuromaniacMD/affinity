// aff-exphash — FNV-64 of every routed expert span in a container, "L E hash" per line, for the
// first N layers (default all). Used to prove a distributed quantise reproduces a reference.
#include "format/aff_reader.h"
#include <cstdio>
#include <cstdlib>
#include <string>
using namespace aff;
int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "aff-exphash model.aff [layers]\n"); return 2; }
  AffReader r; std::string err;
  if (!r.open(argv[1], &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
  uint32_t nl = (uint32_t)r.layer_count();
  if (argc > 2) nl = std::min<uint32_t>(nl, (uint32_t)std::atoi(argv[2]));
  for (uint32_t l = 0; l < nl; ++l) {
    const AffLayerDesc& L = r.layer(l);
    for (uint32_t e = 0; e < L.n_experts; ++e) {
      const uint8_t* p = r.expert_ptr(l, e);
      uint64_t h = 1469598103934665603ull;
      for (uint64_t i = 0; i < L.stride; ++i) { h ^= p[i]; h *= 1099511628211ull; }
      std::printf("%u %u %016llx\n", l, e, (unsigned long long)h);
    }
  }
  return 0;
}
