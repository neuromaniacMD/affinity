// aff-cfgcheck — parse a container's stored config through the engine's own parse_model_config and
// print the layer schedule it produces. CPU-only: it is how the V4.1 config path is checked without
// a GPU, and how you see at a glance which layers build compressed KV, which build the index, and
// which carry Engram.
#include "engine/model.h"
#include "format/aff_reader.h"
#include <cstdio>
#include <string>
using namespace aff;
int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "aff-cfgcheck model.aff\n"); return 2; }
  AffReader r;
  std::string err;
  if (!r.open(argv[1], &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
  ModelConfig c;
  if (!parse_model_config(std::string(r.meta()), &c, &err)) { std::fprintf(stderr, "config: %s\n", err.c_str()); return 1; }
  std::printf("arch            %s\n", c.v41 ? "deepseek_v41" : "deepseek_v4");
  std::printf("layers/embd     %u / %u   heads %u x %u (rope %u)  o_groups %u\n",
              c.n_layer, c.n_embd, c.n_head, c.head_dim, c.rope_dim, c.o_groups);
  std::printf("experts         %u routed, %u active, %u shared, inter %u\n",
              c.n_expert, c.n_expert_used, c.n_shared_expert, c.moe_inter);
  std::printf("router          %s, temp %.3f, scale %.3f, norm_topk %d, hash layers %u\n",
              c.score_func == ModelConfig::ScoreFunc::SqrtSoftplus ? "sqrtsoftplus"
              : c.score_func == ModelConfig::ScoreFunc::Softmax ? "softmax" : "sigmoid",
              (double)c.gate_temp, (double)c.routed_scaling, (int)c.norm_topk_prob, c.n_hash_layer);
  std::printf("indexer         topk %u, %u heads x %u\n", c.index_topk, c.index_n_heads, c.index_head_dim);
  std::printf("candidate       source layer %d, %u blocks of %u\n", c.candidate_source_layer,
              c.candidate_topk_blocks, c.candidate_block);
  std::printf("engram          layers");
  for (size_t i = 0; i < c.engram_layers.size(); ++i)
    std::printf(" %d(%llu rows)", c.engram_layers[i], (unsigned long long)c.engram_rows[i]);
  std::printf(", %u heads x %u, ngram<=%u, vocab %u\n", c.engram_n_heads, c.engram_head_dim,
              c.engram_max_ngram, c.engram_vocab);
  std::printf("dspark          block %u, %u experts (%u active), taps", c.dspark_block,
              c.dspark_n_expert, c.dspark_n_expert_used);
  for (int32_t t : c.dspark_taps) std::printf(" %d", t);
  std::printf("\n\nlayer schedule (ratio / kv-source / index-source / engram)\n");
  for (uint32_t l = 0; l < c.n_layer; ++l)
    std::printf("  %2u  ratio %-3u %s%s%s\n", l, c.ratio_for(l),
                c.is_kv_source(l) ? "KV " : "   ", c.is_index_source(l) ? "IDX " : "    ",
                c.is_engram_layer(l) ? "ENGRAM" : "");
  return 0;
}
