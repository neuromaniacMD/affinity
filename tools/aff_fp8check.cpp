// aff-fp8check — every FP8 tensor in a container, dequantised through the engine's own
// fp8_block_dequant (128x128), against its checkpoint source dequantised on the source's grid
// (32 or 128). Prints the relative error per tensor. Proves the 32 -> 128 refold as WRITTEN.
#include "format/aff_reader.h"
#include "model/safetensors.h"
#include "quant/source_dtypes.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
using namespace aff;
int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "aff-fp8check model.aff <ckpt-dir>\n"); return 2; }
  AffReader r; std::string err; const std::string src = argv[2];
  if (!r.open(argv[1], &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
  std::unordered_map<std::string, std::string> wmap;
  if (!load_safetensors_index(src + "/model.safetensors.index.json", &wmap, &err)) {
    std::fprintf(stderr, "%s\n", err.c_str()); return 1;
  }
  const std::map<std::string, std::string> ren = {
    {"attn_q_a", "attn.wq_a"}, {"attn_q_b", "attn.wq_b"}, {"attn_kv", "attn.wkv"},
    {"attn_out_a", "attn.wo_a"}, {"attn_out_b", "attn.wo_b"}, {"shexp_gate", "ffn.shared_experts.w1"},
    {"shexp_up", "ffn.shared_experts.w3"}, {"shexp_down", "ffn.shared_experts.w2"},
    {"idx_wq_b", "attn.indexer.wq_b"}, {"engram_wkv", "engram.wkv"}};
  std::map<std::string, std::unique_ptr<SafeTensorsFile>> files;
  int bad = 0, n = 0; double worst = 0;
  for (uint64_t i = 0; i < r.tensor_count(); ++i) {
    const AffTensorEntry& e = r.tensor_at(i);
    if (e.desc.quant != AFF_FP8_E4M3) continue;
    const std::string name = e.name;                     // blk.<l>.<kind>
    const size_t d1 = name.find('.', 4);
    if (name.rfind("blk.", 0) != 0 || d1 == std::string::npos) { std::printf("skip %s\n", name.c_str()); continue; }
    auto it = ren.find(name.substr(d1 + 1));
    if (it == ren.end()) { std::printf("skip %s (no source rule)\n", name.c_str()); continue; }
    const std::string sname = "layers." + name.substr(4, d1 - 4) + "." + it->second;
    auto wi = wmap.find(sname + ".weight");
    if (wi == wmap.end()) { std::printf("MISSING source %s\n", sname.c_str()); ++bad; continue; }
    auto& f = files[wi->second];
    if (!f) {
      f = std::make_unique<SafeTensorsFile>();
      if (!f->open(src + "/" + wi->second, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    }
    const StTensor* sw = f->find(sname + ".weight");
    const StTensor* ss = f->find(sname + ".scale");
    const uint64_t R = e.desc.rows, C = e.desc.cols;
    const uint64_t g = (uint64_t)ss->shape[0] * 32 == R ? 32 : 128;
    std::vector<float> a(R * C), b(R * C);
    fp8_block_dequant(r.tensor_data(e), r.tensor_scales(e), R, C, 128, 128, a.data());
    fp8_block_dequant(sw->data, ss->data, R, C, g, g, b.data());
    double e2 = 0, t2 = 0, mx = 0;
    for (uint64_t k = 0; k < R * C; ++k) {
      const double d = (double)a[k] - b[k];
      e2 += d * d; t2 += (double)b[k] * b[k]; mx = std::max(mx, std::fabs(d));
    }
    const double rel = std::sqrt(e2 / t2);
    worst = std::max(worst, rel); ++n;
    if (rel > 1e-5) ++bad;
    std::printf("%-26s grid %3llu  rel %.2e  maxabs %.2e%s\n", name.c_str(), (unsigned long long)g, rel, mx,
                rel > 1e-5 ? "  <-- BAD" : "");
  }
  std::printf("fp8check: %d tensors, worst rel %.2e, %d bad\n", n, worst, bad);
  return bad ? 1 : 0;
}
