// aff-quantize — turn a DeepSeek-V4-Flash safetensors checkpoint into a .aff container.
//
// Reads the native dtypes (MXFP4 experts, FP8-E4M3 128x128-block attention/shared, BF16 routers),
// quantises routed experts to AFF_Q2P875 (D11) and writes everything into the expert-oriented
// container. Streams shard-at-a-time so peak RSS stays bounded regardless of model size.
//
//   aff-quantize --src <checkpoint-dir> --out model.aff [--layers N] [--imatrix f.dat] [--threads N]

#include "format/aff_reader.h"
#include "format/aff_writer.h"
#include "model/safetensors.h"
#include "quant/source_dtypes.h"
#include "quant/blockquant.h"
#include "quant/imatrix.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

using namespace aff;
using clk = std::chrono::steady_clock;

namespace {

std::string slurp(const std::string& p) {
  FILE* f = std::fopen(p.c_str(), "rb");
  if (!f) return "";
  std::string s; char b[65536]; size_t n;
  while ((n = std::fread(b, 1, sizeof(b), f)) > 0) s.append(b, n);
  std::fclose(f);
  return s;
}

int cfg_int(const std::string& j, const char* k, int dflt) {
  const std::string pat = std::string("\"") + k + "\"";
  size_t p = j.find(pat);
  if (p == std::string::npos) return dflt;
  p = j.find(':', p);
  return p == std::string::npos ? dflt : (int)std::strtol(j.c_str() + p + 1, nullptr, 10);
}


// True when a checkpoint tensor ships as block-scaled FP8 and both dimensions are tile-aligned, so
// it can be stored verbatim instead of dequantised.
bool fp8_passthrough_ok(const StTensor* t, const StTensor* sc) {
  if (!t || !sc || t->dtype != StDType::F8_E4M3 || t->shape.size() != 2) return false;
  return (t->shape[0] % 128) == 0 && (t->shape[1] % 128) == 0;
}

// A declared tensor's source name in the checkpoint. An empty name means synthesised.
struct SrcRef { std::string name; uint64_t rows, cols; };

// Parses a flat integer array out of config.json, e.g. "compress_ratios": [0, 0, 4, 128, ...].
std::vector<int> cfg_int_array(const std::string& j, const char* k) {
  std::vector<int> v;
  const std::string pat = std::string("\"") + k + "\"";
  size_t p = j.find(pat);
  if (p == std::string::npos) return v;
  p = j.find('[', p);
  if (p == std::string::npos) return v;
  const size_t e = j.find(']', p);
  for (size_t i = p + 1; i < e; ) {
    while (i < e && (j[i] == ' ' || j[i] == ',' || j[i] == '\n')) ++i;
    if (i >= e) break;
    char* end = nullptr;
    v.push_back((int)std::strtol(j.c_str() + i, &end, 10));
    i = (size_t)(end - j.c_str());
  }
  return v;
}

// Materialises any source dtype we might meet as f32. FP8 weights carry a separate UE8M0 scale
// tensor on a 128x128 grid; bf16/f32/i64 are direct.
bool load_as_f32(const StTensor* t, const StTensor* scale, uint64_t n, std::vector<float>* out) {
  if (!t || t->numel() != n) return false;
  out->resize(n);
  switch (t->dtype) {
    case StDType::BF16:
      bf16_dequant(reinterpret_cast<const uint16_t*>(t->data), n, out->data());
      return true;
    case StDType::F32:
      std::memcpy(out->data(), t->data, n * 4);
      return true;
    case StDType::I64: {
      const int64_t* s = reinterpret_cast<const int64_t*>(t->data);
      for (uint64_t i = 0; i < n; ++i) (*out)[i] = (float)s[i];
      return true;
    }
    case StDType::F8_E4M3: {
      if (!scale || t->shape.size() != 2) return false;
      return fp8_block_dequant(t->data, scale->data,
                               (uint64_t)t->shape[0], (uint64_t)t->shape[1], 128, 128, out->data());
    }
    default:
      return false;
  }
}

// The routed-expert format, at 2.875 bpw:
//
//   block 16 | 10-bit QUAD code | 3-bit pow2 scale | sign | rotate-by-two | 1 variant bit
//
// LDS decides how the six field bits split, not the bit rate. A 4-wide table is 1024 entries x 4
// packed E4M3 bytes = 4 KB a variant, and the expert GEMM's staging has 8 KB to spare, so the
// table is two variants and the block scale has to be a power of two added to each byte's exponent
// rather than a dimension inside the table. That frees two bits for symmetries costing no entries
// at all: one negates the block, one rotates the quad by two bytes. Without them a codebook fitted
// to weights with no sign or positional preference spends its entries reproducing both.
//
// ONE definition, shared with everything in the engine that reads the result back.
constexpr AffQuant kExpertQuant = AFF_Q2P875_Q4;
QuantSpec expert_spec() { return expert_quant_spec((uint32_t)kExpertQuant); }

} // namespace

int main(int argc, char** argv) {
  std::string src, out = "model.aff", imat_path, cb_from;
  int want_layers = -1, nthread = (int)std::thread::hardware_concurrency();
  bool dspark = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto nx = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--src") src = nx();
    else if (a == "--out") out = nx();
    else if (a == "--layers") want_layers = std::atoi(nx());
    else if (a == "--imatrix") imat_path = nx();
    else if (a == "--threads") nthread = std::atoi(nx());
    else if (a == "--dspark") dspark = true;
    else if (a == "--codebook") cb_from = nx();
    else if (a == "-h" || a == "--help") {
      std::printf(
          "aff-quantize --src <ckpt-dir> --out model.aff [--imatrix f.dat] [--layers N]\n"
          "aff-quantize --src <ckpt-dir> --out model.dspark.aff --dspark --codebook model.aff\n"
          "\n"
          "  --src DIR        checkpoint directory: config.json and the safetensors shards\n"
          "  --out FILE       container to write (default model.aff)\n"
          "  --imatrix FILE   importance weights, in llama.cpp's imatrix format. Also supplies the\n"
          "                   expert popularity profile the placement engine reads at load\n"
          "  --layers N       quantise only the first N layers. A truncated container loads and\n"
          "                   decodes fluent nonsense; it is for exercising the pipeline, not for\n"
          "                   inference\n"
          "  --threads N      worker threads (default: one per core). Does not affect the output\n"
          "  --dspark         quantise the mtp.* draft stages instead of the model\n"
          "  --codebook FILE  read the expert codebook from this container rather than fitting\n"
          "                   one. Required under --dspark\n"
          "\n"
          "The container is a pure function of the checkpoint and the imatrix: the same inputs\n"
          "give a byte-identical file. Thread count, the order the shards are read in, and\n"
          "whether the host CPU has AVX-512 do not enter into it.\n");
      return 0;
    }
    // No silent typos. A misspelled --imatrix would otherwise produce a container that loads,
    // reports the right geometry, and carries neither importance weighting nor a profile.
    else {
      std::fprintf(stderr, "error: unrecognised argument '%s'. Try --help.\n", a.c_str());
      return 2;
    }
  }
  if (src.empty()) { std::fprintf(stderr, "error: --src is required\n"); return 2; }
  if (nthread < 1) nthread = 1;
  // ---- --dspark: the mtp.* stages into a container of their OWN ------------------------------
  //
  // A companion file rather than an addition to the model, so that adding the draft cannot change
  // what the target emits: the engine's placement decides which experts fit in VRAM, and that
  // choice sets the summation order, so a container that grew by the draft's experts would produce
  // different text from the same weights.
  //
  // `--codebook` is required and not optional. Every routed expert in the engine decodes through
  // ONE global table held in LDS, so both containers must agree on it exactly. Refitting would very
  // likely reproduce it, and "very likely" is not a property a silent 4-bit lookup can have.
  if (dspark && cb_from.empty()) {
    std::fprintf(stderr, "error: --dspark needs --codebook <the main .aff>, because both\n"
                         "       containers' experts decode through one global table\n");
    return 2;
  }

  const std::string cfg_json = slurp(src + "/config.json");
  if (cfg_json.empty()) { std::fprintf(stderr, "error: no config.json in %s\n", src.c_str()); return 1; }
  const int n_expert = cfg_int(cfg_json, "n_routed_experts", 256);
  const int hidden = cfg_int(cfg_json, "hidden_size", 4096);
  const int inter = cfg_int(cfg_json, "moe_intermediate_size", 2048);

  std::unordered_map<std::string, std::string> wmap;
  std::string err;
  if (!load_safetensors_index(src + "/model.safetensors.index.json", &wmap, &err)) {
    std::fprintf(stderr, "index: %s\n", err.c_str()); return 1;
  }

  // Counted, not read from config: `dspark_block_size` says how many tokens a draft emits and
  // `num_nextn_predict_layers` is MTP-1's count and is 1 here, so neither of them is the number of
  // stages. The checkpoint's own index is.
  int n_mtp = 0;
  while (wmap.count("mtp." + std::to_string(n_mtp) + ".attn.wkv.weight")) ++n_mtp;
  if (dspark && n_mtp == 0) {
    std::fprintf(stderr, "error: --dspark but no mtp.* tensors in %s\n", src.c_str());
    return 1;
  }
  // The stage's own prefix in the checkpoint, and the one it gets in the container.
  const std::string SP = dspark ? "mtp." : "layers.";
  const std::string DP = dspark ? "mtp." : "blk.";
  const int n_layer_total = dspark ? n_mtp : cfg_int(cfg_json, "num_hidden_layers", 43);
  const int n_layer = (want_layers > 0 && want_layers < n_layer_total) ? want_layers : n_layer_total;

  ImatrixFile imat;
  // Not in --dspark mode, and this is a correctness guard rather than a shortcut: the imatrix is
  // keyed by the TARGET's gguf layer names, so `gguf_name_for_expert(l, "w1")` would hand mtp
  // stage 0 the importance vector measured on the target's layer 0. Weighting a quantisation by
  // the wrong tensor's activations is invisible in every shape and size check there is.
  const bool have_imat = !dspark && !imat_path.empty() && imat.open(imat_path, &err);
  if (!imat_path.empty() && !have_imat && !dspark) std::fprintf(stderr, "warning: imatrix: %s\n", err.c_str());
  if (!imat_path.empty() && dspark)
    std::fprintf(stderr, "note   : --imatrix ignored under --dspark (it is keyed by target layer)\n");

  const QuantSpec spec = expert_spec();
  std::printf("source   : %s (%d layers, %d experts, hidden %d, inter %d)\n",
              src.c_str(), n_layer_total, n_expert, hidden, inter);
  std::printf("quantise : %d layers -> %s at %.4f bpw%s\n",
              n_layer, out.c_str(), spec.bpw(), have_imat ? " (imatrix-weighted)" : "");

  // ---- declare the layout up front so experts can be written in any order --------------------
  std::vector<LayerDecl> ld(n_layer);
  for (auto& L : ld) {
    L.n_experts = (uint32_t)n_expert;
    L.geom = { (uint64_t)hidden, (uint64_t)inter, kExpertQuant, kExpertQuant, kExpertQuant };
  }
  // Non-expert tensors: matrices in bf16, vectors in f32. They are a small fraction of the model
  // and every token reads all of them, so they stay off the routed experts' bandwidth-bound path.
  // Quantising them further is possible and buys little.
  std::vector<TensorDecl> td;
  std::vector<SrcRef> srcs;
  // Matrices whose source is FP8 keep FP8; everything else stays bf16 or f32. Passing AFF_FP8_E4M3
  // here is a REQUEST — the writer below falls back to bf16 if the source turns out not to be FP8
  // or not tile-aligned, so a checkpoint that changes dtype cannot silently produce garbage.
  auto add = [&](const std::string& n, const std::string& src, uint64_t r, uint64_t c, AffQuant q) {
    td.push_back({n, r, c, q});
    srcs.push_back({src, r, c});
  };
  const uint64_t vocab = (uint64_t)cfg_int(cfg_json, "vocab_size", 129280);
  const uint64_t n_hc = (uint64_t)cfg_int(cfg_json, "hc_mult", 4);
  const uint64_t hc_dim = n_hc * (uint64_t)hidden;
  const uint64_t qr = (uint64_t)cfg_int(cfg_json, "q_lora_rank", 1024);
  const uint64_t orank = (uint64_t)cfg_int(cfg_json, "o_lora_rank", 1024);
  const uint64_t hd = (uint64_t)cfg_int(cfg_json, "head_dim", 512);
  const uint64_t nh = (uint64_t)cfg_int(cfg_json, "num_attention_heads", 64);
  const uint64_t ihd = (uint64_t)cfg_int(cfg_json, "index_head_dim", 128);
  const uint64_t inh = (uint64_t)cfg_int(cfg_json, "index_n_heads", 64);
  const uint64_t n_used = (uint64_t)cfg_int(cfg_json, "num_experts_per_tok", 6);
  const int n_hash = cfg_int(cfg_json, "num_hash_layers", 3);
  const std::vector<int> ratios = cfg_int_array(cfg_json, "compress_ratios");

  // ONE codebook for every routed expert. It has to live in the container: the kernel cannot
  // reconstruct a fitted codebook from the packed codes. Per-matrix codebooks are better on paper,
  // but the container holds one `__codebook` and the expert GEMM holds one table in LDS, so a
  // global fit is what the format and the kernel support.
  // A 4-wide codebook is stored ALREADY PACKED as E4M3 quads, one dword an entry, because the GEMM
  // would otherwise build 4096 entries from floats in every block. So the declared element count is
  // one dword per entry, not `pair` floats — the tensor is a byte blob that happens to be declared
  // f32-wide. The pair form keeps its 4 x 32 x 2 floats.
  add("__codebook", "", 1,
      (uint64_t)spec.variants() * spec.lut_size() * (spec.pair == 4 ? 1u : spec.pair), AFF_F32);
  const uint64_t mrank = (uint64_t)cfg_int(cfg_json, "dspark_markov_rank", 256);
  const std::vector<int> taps = cfg_int_array(cfg_json, "dspark_target_layer_ids");
  if (!dspark) {
    add("token_embd",  "embed.weight",   vocab, (uint64_t)hidden, AFF_BF16);
    add("output",      "head.weight",    vocab, (uint64_t)hidden, AFF_BF16);
    add("output_norm", "norm.weight",    1, (uint64_t)hidden, AFF_F32);
    add("hc_head_fn",    "hc_head_fn",    n_hc, hc_dim, AFF_BF16);
    add("hc_head_base",  "hc_head_base",  1, n_hc, AFF_F32);
    add("hc_head_scale", "hc_head_scale", 1, 1, AFF_F32);
  } else {
    // The draft shares the TARGET's embedding and output head — `Transformer.__init__` assigns
    // `self.mtp[-1].embed = self.embed` and `.head = self.head` — so neither is copied here. What
    // the draft owns is the projection of the tapped hidden states on the way in, and its own
    // hc_head / norm / markov / confidence on the way out.
    if (taps.empty()) {
      std::fprintf(stderr, "error: --dspark but config has no dspark_target_layer_ids\n");
      return 1;
    }
    const std::string first = SP + "0.";
    const std::string last  = SP + std::to_string(n_mtp - 1) + ".";
    add("mtp_main_proj",  first + "main_proj.weight", (uint64_t)hidden,
        (uint64_t)hidden * taps.size(), AFF_FP8_E4M3);
    add("mtp_main_norm",  first + "main_norm.weight", 1, (uint64_t)hidden, AFF_F32);
    add("mtp_norm",       last + "norm.weight",       1, (uint64_t)hidden, AFF_F32);
    add("mtp_hc_head_fn",    last + "hc_head_fn",    n_hc, hc_dim, AFF_BF16);
    add("mtp_hc_head_base",  last + "hc_head_base",  1, n_hc, AFF_F32);
    add("mtp_hc_head_scale", last + "hc_head_scale", 1, 1, AFF_F32);
    // The Markov head is a rank-`mrank` bigram correction: w1 is an EMBEDDING (token -> mrank) and
    // w2 a head (mrank -> vocab), and its logit bias is what makes the block autoregressive without
    // running the transformer again. Both are [vocab][mrank] in the checkpoint.
    add("mtp_markov_w1", last + "markov_head.markov_w1.weight", vocab, mrank, AFF_BF16);
    add("mtp_markov_w2", last + "markov_head.markov_w2.weight", vocab, mrank, AFF_BF16);
    add("mtp_confidence", last + "confidence_head.proj.weight", 1, (uint64_t)hidden + mrank,
        AFF_BF16);
  }

  for (int l = 0; l < n_layer; ++l) {
    const std::string p = DP + std::to_string(l) + ".";
    const std::string q = SP + std::to_string(l) + ".";
    // The mtp stages have their own entries at the tail of `compress_ratios` — the array is 46
    // long for 43 layers — and DSparkAttention asserts compress_ratio == 0. Checked rather than
    // assumed: a nonzero one would need the compressor and the indexer, which this does not write.
    const size_t ri = dspark ? (size_t)cfg_int(cfg_json, "num_hidden_layers", 43) + l : (size_t)l;
    const int ratio = ri < ratios.size() ? ratios[ri] : 0;
    if (dspark && ratio != 0) {
      std::fprintf(stderr, "error: mtp stage %d has compress_ratio %d, expected 0\n", l, ratio);
      return 1;
    }
    const uint64_t coff = (ratio == 4) ? 2 : 1;

    add(p + "attn_norm",   q + "attn_norm.weight",     1, (uint64_t)hidden, AFF_F32);
    add(p + "ffn_norm",    q + "ffn_norm.weight",      1, (uint64_t)hidden, AFF_F32);
    add(p + "attn_sinks",  q + "attn.attn_sink",       1, nh, AFF_F32);
    add(p + "attn_q_a",    q + "attn.wq_a.weight",     qr, (uint64_t)hidden, AFF_FP8_E4M3);
    add(p + "attn_q_b",    q + "attn.wq_b.weight",     nh * hd, qr, AFF_FP8_E4M3);
    add(p + "attn_kv",     q + "attn.wkv.weight",      hd, (uint64_t)hidden, AFF_FP8_E4M3);
    add(p + "attn_out_a",  q + "attn.wo_a.weight",     8 * orank, (uint64_t)hidden, AFF_FP8_E4M3);
    add(p + "attn_out_b",  q + "attn.wo_b.weight",     (uint64_t)hidden, 8 * orank, AFF_FP8_E4M3);
    add(p + "attn_q_norm", q + "attn.q_norm.weight",   1, qr, AFF_F32);
    add(p + "attn_kv_norm",q + "attn.kv_norm.weight",  1, hd, AFF_F32);
    // The router gate in bf16. Decode reads all of it every token, once a layer, so the halving is
    // worth having: its output is a softmax then a top-k, so what has to survive is the ORDER of
    // the top few, and bf16 weights against an f32 activation and accumulator keep 8 mantissa bits
    // of each product where the router's own logit gaps are far wider. fp16 would keep 10 bits and
    // lose exponent range this model does not need.
    add(p + "ffn_gate_inp",q + "ffn.gate.weight",      (uint64_t)n_expert, (uint64_t)hidden, AFF_BF16);
    if (!dspark && l < n_hash) {
      add(p + "ffn_gate_hash", q + "ffn.gate.tid2eid", vocab, n_used, AFF_F32);
    } else {
      // The aux-loss-free correction bias. Present only on non-hash layers, and it shifts expert
      // SELECTION without touching the routing weights, so its absence is invisible in the output
      // shapes and changes only which six experts run.
      add(p + "ffn_gate_bias", q + "ffn.gate.bias", 1, (uint64_t)n_expert, AFF_F32);
    }
    add(p + "shexp_gate",  q + "ffn.shared_experts.w1.weight", (uint64_t)inter, (uint64_t)hidden, AFF_FP8_E4M3);
    add(p + "shexp_up",    q + "ffn.shared_experts.w3.weight", (uint64_t)inter, (uint64_t)hidden, AFF_FP8_E4M3);
    add(p + "shexp_down",  q + "ffn.shared_experts.w2.weight", (uint64_t)hidden, (uint64_t)inter, AFF_FP8_E4M3);
    for (const char* w : {"attn", "ffn"}) {
      const std::string b = std::string("hc_") + w;
      add(p + b + "_fn",    q + b + "_fn",    2 * n_hc + n_hc * n_hc, hc_dim, AFF_BF16);
      add(p + b + "_base",  q + b + "_base",  1, 2 * n_hc + n_hc * n_hc, AFF_F32);
      add(p + b + "_scale", q + b + "_scale", 1, 3, AFF_F32);
    }
    if (ratio != 0) {
      add(p + "comp_wkv",   q + "attn.compressor.wkv.weight",   coff * hd, (uint64_t)hidden, AFF_BF16);
      add(p + "comp_wgate", q + "attn.compressor.wgate.weight", coff * hd, (uint64_t)hidden, AFF_BF16);
      add(p + "comp_ape",   q + "attn.compressor.ape",          (uint64_t)ratio, coff * hd, AFF_F32);
      add(p + "comp_norm",  q + "attn.compressor.norm.weight",  1, hd, AFF_F32);
    }
    if (ratio == 4) {
      add(p + "idx_wq_b",   q + "attn.indexer.wq_b.weight",         inh * ihd, qr, AFF_FP8_E4M3);
      add(p + "idx_proj",   q + "attn.indexer.weights_proj.weight", inh, (uint64_t)hidden, AFF_BF16);
      add(p + "idx_comp_wkv",   q + "attn.indexer.compressor.wkv.weight",   coff * ihd, (uint64_t)hidden, AFF_BF16);
      add(p + "idx_comp_wgate", q + "attn.indexer.compressor.wgate.weight", coff * ihd, (uint64_t)hidden, AFF_BF16);
      add(p + "idx_comp_ape",   q + "attn.indexer.compressor.ape",          (uint64_t)ratio, coff * ihd, AFF_F32);
      add(p + "idx_comp_norm",  q + "attn.indexer.compressor.norm.weight",  1, ihd, AFF_F32);
    }
  }

  // ---- fit the global codebook from a spread of experts --------------------------------------
  Codebook gcb;
  if (dspark) {
    // Read it, do not refit. Both containers' experts are decoded by one table held in LDS, so
    // "a codebook fitted the same way" is not good enough — it has to be the same bytes.
    AffReader base;
    if (!base.open(cb_from, &err)) {
      std::fprintf(stderr, "codebook %s: %s\n", cb_from.c_str(), err.c_str()); return 1;
    }
    const AffTensorEntry* e = base.find_tensor("__codebook");
    if (!e) { std::fprintf(stderr, "error: %s has no __codebook\n", cb_from.c_str()); return 1; }
    const uint64_t want = (uint64_t)spec.variants() * spec.lut_size() * (spec.pair == 4 ? 1u : spec.pair);
    if (e->desc.rows * e->desc.cols != want) {
      std::fprintf(stderr, "error: %s __codebook is %llu entries, this build wants %llu\n",
                   cb_from.c_str(), (unsigned long long)(e->desc.rows * e->desc.cols),
                   (unsigned long long)want);
      return 1;
    }
    const uint8_t* v = base.tensor_data(*e);
    if (spec.pair >= 4) {
      unpack_codebook_e4m3_quads(reinterpret_cast<const uint32_t*>(v), spec.variants(),
                                 spec.lut_size(), &gcb);
    } else {
      gcb.variants = spec.variants();
      gcb.k = spec.lut_size();
      gcb.pair = spec.pair;
      gcb.v.assign((size_t)gcb.variants * gcb.k * gcb.pair, 0.0f);
      std::memcpy(gcb.v.data(), v, gcb.v.size() * 4);
    }
    std::printf("codebook : %u variants x %u entries read from %s\n", gcb.variants, gcb.k,
                cb_from.c_str());
  } else {
    std::vector<float> samples, sw, deq;
    auto it0 = wmap.find("layers.0.ffn.experts.0.w1.weight");
    if (it0 == wmap.end()) { std::fprintf(stderr, "error: no expert tensors in index\n"); return 1; }
    SafeTensorsFile st;
    if (!st.open(src + "/" + it0->second, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    std::vector<float> all;
    const char* which[3] = {"w1", "w3", "w2"};
    for (int e = 0; e < 8; ++e) {
      for (int m = 0; m < 3; ++m) {
        const std::string base = "layers.0.ffn.experts." + std::to_string(e * 13) + "." + which[m];
        const StTensor* wt = st.find(base + ".weight");
        const StTensor* sc = st.find(base + ".scale");
        if (!wt || !sc) continue;
        const uint64_t rows = (uint64_t)wt->shape[0], cols = (uint64_t)wt->shape[1] * 2;
        deq.assign(rows * cols, 0.0f);
        if (!mxfp4_dequant(wt->data, sc->data, rows, cols, deq.data(), NibbleOrder::LowFirst)) continue;
        samples.clear();
        // The budget scales with the table, so each centroid keeps a few hundred quads to fit
        // against. At 1024 entries x 4 variants the pair form's 2^17 would leave about thirty,
        // which is noise.
        const uint64_t budget = spec.pair >= 4 ? (1u << 19) : (1u << 17);
        collect_normalized_samples(deq.data(), rows, cols, spec, &samples, budget, nullptr, nullptr);
        all.insert(all.end(), samples.begin(), samples.end());
      }
    }
    if (all.empty()) { std::fprintf(stderr, "error: could not sample experts for the codebook\n"); return 1; }
    const auto t_fit = clk::now();
    gcb = fit_codebook(all.data(), all.size(), spec, nullptr);
    // Round the fit onto exactly what the container will store and the kernel will hold. Doing it
    // BEFORE the encode matters: quantising against the floats and then rounding the table would
    // leave every code chosen against a value the GPU never sees.
    if (spec.pair >= 4) project_codebook_e4m3(&gcb, spec.pow2_block_scale);
    std::printf("codebook : global fit over %zu samples (%u variants x %u entries) in %.0f s\n",
                all.size() / spec.pair, gcb.variants, gcb.k,
                std::chrono::duration<double>(clk::now() - t_fit).count());
  }

  // Brute force is 1024 distance evaluations a quad, against tens of billions of quads. Build the
  // acceleration grid ONCE here and hand it to every quantize_matrix call: it is read-only, so all
  // the worker threads below share it.
  CodebookSearch gsearch;
  if (spec.pair >= 4) {
    const auto t_ix = clk::now();
    build_codebook_search(gcb, &gsearch);
    std::printf("index    : %s (%zu candidate slots) in %.1f s\n",
                gsearch.ready() ? "grid built" : "NOT BUILT — falling back to brute force",
                gsearch.cand.size(), std::chrono::duration<double>(clk::now() - t_ix).count());
  }

  // An exclusive lock on the output. Two quantisers writing one container interleave their expert
  // payloads and produce a file that still LOADS — right header, right geometry, garbage weights.
  // Nothing downstream detects that, so the collision is refused here.
  const int lockfd = ::open((out + ".lock").c_str(), O_CREAT | O_RDWR, 0644);
  if (lockfd < 0) { std::fprintf(stderr, "cannot create %s.lock\n", out.c_str()); return 1; }
  if (::flock(lockfd, LOCK_EX | LOCK_NB) != 0) {
    std::fprintf(stderr, "error: another aff-quantize is already writing %s\n"
                         "       (holding %s.lock). Refusing to interleave.\n",
                 out.c_str(), out.c_str());
    return 1;
  }

  AffWriter w;
  if (!w.begin(out, cfg_json, ld, td, &err)) { std::fprintf(stderr, "begin: %s\n", err.c_str()); return 1; }
  std::printf("container: %.2f GiB reserved (%zu dense tensors)\n",
              w.file_size() / 1073741824.0, td.size());

  // ---- non-expert tensors: grouped by shard so each file is mapped exactly once --------------
  {
    std::map<std::string, std::vector<size_t>> by_shard;
    for (size_t i = 0; i < td.size(); ++i) {
      if (srcs[i].name.empty()) { by_shard[""].push_back(i); continue; }
      auto it = wmap.find(srcs[i].name);
      if (it == wmap.end()) { by_shard[""].push_back(i); continue; }
      by_shard[it->second].push_back(i);
    }
    std::vector<float> f32;
    std::vector<uint8_t> blob;
    size_t missing = 0;
    for (const auto& [shard, idxs] : by_shard) {
      SafeTensorsFile st;
      if (!shard.empty() && !st.open(src + "/" + shard, &err)) {
        std::fprintf(stderr, "open %s: %s\n", shard.c_str(), err.c_str()); return 1;
      }
      for (size_t i : idxs) {
        const TensorDecl& t = td[i];
        const uint64_t n = t.rows * t.cols;
        const StTensor* src_t = shard.empty() ? nullptr : st.find(srcs[i].name);
        const StTensor* sc_t  = (src_t && src_t->dtype == StDType::F8_E4M3)
                                ? st.find(srcs[i].name.substr(0, srcs[i].name.rfind(".weight")) + ".scale")
                                : nullptr;
        (void)sc_t;
        // FP8 passthrough: copy the checkpoint's own bytes and scales, no conversion at all.
        // Dequantising these to bf16 doubles the largest traffic source in the engine for no
        // quality gain, since FP8+UE8M0 is what the model was trained and shipped in.
        if (t.quant == AFF_FP8_E4M3) {
          if (!fp8_passthrough_ok(src_t, sc_t)) {
            std::fprintf(stderr, "error: %s was declared FP8 but its source is %s%s\n", t.name.c_str(),
                         src_t ? st_dtype_name(src_t->dtype) : "absent",
                         src_t && !sc_t ? " (no .scale)" : "");
            return 1;
          }
          MatrixData mdf;
          mdf.data = src_t->data;
          mdf.data_size = quant_data_bytes(AFF_FP8_E4M3, n);
          mdf.scales = sc_t->data;
          mdf.scale_size = quant_scale_bytes(AFF_FP8_E4M3, n);
          if (!w.write_tensor(t.name, mdf, &err)) {
            std::fprintf(stderr, "tensor %s: %s\n", t.name.c_str(), err.c_str()); return 1;
          }
          continue;
        }
        if (t.name == "__codebook" && spec.pair >= 4) {
          // Packed E4M3 quads, one dword an entry — a byte blob in an f32-declared tensor. The
          // decoder in model.cpp unpacks it back to floats for the CPU path and hands the dwords
          // straight to the GPU, and E4M3 -> float -> nearest E4M3 is the identity, so the two can
          // never disagree.
          const std::vector<uint32_t> packed = pack_codebook_e4m3_quads(gcb);
          if (packed.size() != n) {
            std::fprintf(stderr, "error: __codebook declared %llu dwords, packed %zu\n",
                         (unsigned long long)n, packed.size());
            return 1;
          }
          f32.assign(n, 0.0f);
          std::memcpy(f32.data(), packed.data(), n * 4);
        } else if (t.name == "__codebook") {
          f32.assign(gcb.v.begin(), gcb.v.end());
          f32.resize(n, 0.0f);
        } else if (!load_as_f32(src_t, sc_t, n, &f32)) {
          f32.assign(n, 0.0f);
          ++missing;
          std::fprintf(stderr, "warning  : missing %s (source %s)\n",
                       t.name.c_str(), srcs[i].name.c_str());
        }
        if (t.quant == AFF_BF16) {
          blob.resize(n * 2);
          uint16_t* d = reinterpret_cast<uint16_t*>(blob.data());
          for (uint64_t k = 0; k < n; ++k) d[k] = float_to_bf16(f32[k]);
        } else {
          blob.resize(n * 4);
          std::memcpy(blob.data(), f32.data(), n * 4);
        }
        MatrixData md;
        md.data = blob.data(); md.data_size = blob.size();
        if (!w.write_tensor(t.name, md, &err)) {
          std::fprintf(stderr, "tensor %s: %s\n", t.name.c_str(), err.c_str()); return 1;
        }
      }
    }
    if (missing) std::fprintf(stderr, "warning: %zu dense tensors not found in the checkpoint\n", missing);
    std::printf("dense    : %zu tensors written\n", td.size());
  }

  // ---- routed experts, streamed shard-at-a-time ----------------------------------------------
  std::atomic<uint64_t> done{0};
  const uint64_t total = (uint64_t)n_layer * n_expert;
  const auto t0 = clk::now();
  std::mutex wmu;

  for (int l = 0; l < n_layer; ++l) {
    // Group by shard so each file is mapped once.
    std::map<std::string, std::vector<int>> by_shard;
    for (int e = 0; e < n_expert; ++e) {
      const std::string k = SP + std::to_string(l) + ".ffn.experts." + std::to_string(e) + ".w1.weight";
      auto it = wmap.find(k);
      if (it != wmap.end()) by_shard[it->second].push_back(e);
    }
    for (auto& [shard, experts] : by_shard) {
      SafeTensorsFile st;
      if (!st.open(src + "/" + shard, &err)) { std::fprintf(stderr, "shard %s: %s\n", shard.c_str(), err.c_str()); return 1; }

      std::atomic<size_t> next{0};
      std::vector<std::thread> th;
      for (int t = 0; t < nthread; ++t) th.emplace_back([&] {
        std::vector<float> deq, samples, sw;
        for (;;) {
          const size_t i = next.fetch_add(1);
          if (i >= experts.size()) break;
          const int e = experts[i];
          const std::string base = SP + std::to_string(l) + ".ffn.experts." + std::to_string(e) + ".";
          ExpertData ed{};
          std::vector<uint8_t> hold[9];
          int slot = 0;
          bool ok = true;
          // w1 = gate, w3 = up, w2 = down (DeepSeek naming)
          const char* which[3] = {"w1", "w3", "w2"};
          MatrixData* dst[3] = {&ed.gate, &ed.up, &ed.down};
          for (int m = 0; m < 3 && ok; ++m) {
            const StTensor* wt = st.find(base + which[m] + ".weight");
            const StTensor* sc = st.find(base + which[m] + ".scale");
            if (!wt || !sc) { ok = false; break; }
            const uint64_t rows = (uint64_t)wt->shape[0];
            const uint64_t cols = (uint64_t)wt->shape[1] * 2;   // nibble-packed
            deq.assign(rows * cols, 0.0f);
            if (!mxfp4_dequant(wt->data, sc->data, rows, cols, deq.data(), NibbleOrder::LowFirst)) { ok = false; break; }

            const float* imp = nullptr;
            std::vector<float> impv;
            if (have_imat) {
              const std::string gg = ImatrixFile::gguf_name_for_expert((uint32_t)l, which[m]);
              if (imat.importance(gg, (uint64_t)e, &impv) && impv.size() == cols) imp = impv.data();
            }
            QuantizedMatrix q;
            std::string e2;
            if (!quantize_matrix(deq.data(), rows, cols, spec, gcb, &q, &e2, imp, &gsearch)) {
              ok = false; break;
            }
            hold[slot] = std::move(q.data);
            hold[slot + 1] = std::move(q.scales);
            hold[slot + 2] = std::move(q.rscales);
            dst[m]->data = hold[slot].data();     dst[m]->data_size = hold[slot].size();
            dst[m]->scales = hold[slot+1].data(); dst[m]->scale_size = hold[slot+1].size();
            dst[m]->rscales = hold[slot+2].data();dst[m]->rscale_size = hold[slot+2].size();
            slot += 3;
          }
          if (!ok) continue;
          {
            std::lock_guard<std::mutex> g(wmu);
            std::string e3;
            if (!w.write_expert((uint32_t)l, (uint32_t)e, ed, &e3))
              std::fprintf(stderr, "\nwrite L%d E%d: %s\n", l, e, e3.c_str());
          }
          const uint64_t d = ++done;
          if ((d % 64) == 0) {
            const double sec = std::chrono::duration<double>(clk::now() - t0).count();
            std::printf("\r  %llu/%llu experts  %.1f%%  %.1f/s  eta %.0f min   ",
                        (unsigned long long)d, (unsigned long long)total,
                        100.0 * d / total, d / sec, (total - d) / (d / sec) / 60.0);
            std::fflush(stdout);
          }
        }
      });
      for (auto& t : th) t.join();
    }
  }
  std::printf("\r  %llu/%llu experts  100.0%%%30s\n", (unsigned long long)done.load(),
              (unsigned long long)total, "");

  // Expert popularity from the imatrix counts. The placement engine reads this at load to decide
  // which experts start resident, so a container built without --imatrix has to learn the same
  // ranking from its own routing before it places well.
  if (have_imat) {
    std::vector<float> prof((size_t)n_layer * n_expert, 0.0f);
    for (int l = 0; l < n_layer; ++l) {
      const std::string gg = ImatrixFile::gguf_name_for_expert((uint32_t)l, "w1");
      for (int e = 0; e < n_expert; ++e) prof[(size_t)l * n_expert + e] = (float)imat.count(gg, (uint64_t)e);
    }
    if (!w.write_profile(prof.data(), prof.size(), &err))
      std::fprintf(stderr, "profile: %s\n", err.c_str());
    else std::printf("profile  : expert popularity written (%d layers)\n", n_layer);
  }

  // A partially-written container is worse than none: it loads, reports the right geometry, and
  // decodes garbage. So this is an error and a nonzero exit, never a warning.
  ::flock(lockfd, LOCK_UN);
  ::close(lockfd);
  ::unlink((out + ".lock").c_str());

  if (w.missing_experts()) {
    std::fprintf(stderr, "error    : %llu experts unwritten — container is unusable\n",
                 (unsigned long long)w.missing_experts());
    return 1;
  }
  if (!w.finalize(&err)) { std::fprintf(stderr, "finalize: %s\n", err.c_str()); return 1; }
  const double sec = std::chrono::duration<double>(clk::now() - t0).count();
  std::printf("done     : %.1f GiB in %.0f s\n", w.file_size() / 1073741824.0, sec);
  return 0;
}
