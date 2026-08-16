#include "model.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace aff {

namespace {
// Minimal scanner for the flat keys we need. DeepSeek configs are machine-generated and shallow,
// so a full JSON parser is not warranted — but unknown keys must be tolerated, since the config
// carries plenty we deliberately ignore.
const char* find_key(const std::string& s, const char* key) {
  const std::string pat = std::string("\"") + key + "\"";
  size_t p = s.find(pat);
  if (p == std::string::npos) return nullptr;
  p = s.find(':', p + pat.size());
  if (p == std::string::npos) return nullptr;
  return s.c_str() + p + 1;
}
bool get_u32(const std::string& s, const char* k, uint32_t* v) {
  const char* p = find_key(s, k); if (!p) return false;
  *v = (uint32_t)std::strtoul(p, nullptr, 10); return true;
}
bool get_f32(const std::string& s, const char* k, float* v) {
  const char* p = find_key(s, k); if (!p) return false;
  *v = std::strtof(p, nullptr); return true;
}
bool get_bool(const std::string& s, const char* k, bool* v) {
  const char* p = find_key(s, k); if (!p) return false;
  while (*p == ' ') ++p;
  *v = (std::strncmp(p, "true", 4) == 0); return true;
}
} // namespace

bool parse_model_config(const std::string& j, ModelConfig* c, std::string* err) {
  get_u32(j, "num_hidden_layers", &c->n_layer);
  get_u32(j, "hidden_size", &c->n_embd);
  get_u32(j, "num_attention_heads", &c->n_head);
  get_u32(j, "head_dim", &c->head_dim);
  get_u32(j, "qk_rope_head_dim", &c->rope_dim);
  get_u32(j, "n_routed_experts", &c->n_expert);
  get_u32(j, "num_experts_per_tok", &c->n_expert_used);
  get_u32(j, "n_shared_experts", &c->n_shared_expert);
  get_u32(j, "moe_intermediate_size", &c->moe_inter);
  get_u32(j, "vocab_size", &c->vocab);
  get_u32(j, "sliding_window", &c->sliding);
  get_u32(j, "index_topk", &c->index_topk);
  get_u32(j, "num_hash_layers", &c->n_hash_layer);
  get_u32(j, "q_lora_rank", &c->q_lora_rank);
  get_u32(j, "o_lora_rank", &c->o_lora_rank);
  get_u32(j, "hc_mult", &c->hc_mult);
  get_u32(j, "index_head_dim", &c->index_head_dim);
  get_u32(j, "index_n_heads", &c->index_n_heads);
  // YaRN lives in a nested object; the flat scan finds the keys regardless of nesting.
  get_f32(j, "factor", &c->rope_factor);
  get_u32(j, "original_max_position_embeddings", &c->rope_orig_ctx);
  get_f32(j, "beta_fast", &c->rope_beta_fast);
  get_f32(j, "beta_slow", &c->rope_beta_slow);
  get_u32(j, "hc_sinkhorn_iters", &c->hc_sinkhorn_iters);
  get_f32(j, "hc_eps", &c->hc_eps);
  get_f32(j, "rms_norm_eps", &c->rms_eps);
  get_f32(j, "rope_theta", &c->rope_theta);
  get_f32(j, "compress_rope_theta", &c->compress_rope_theta);
  get_f32(j, "routed_scaling_factor", &c->routed_scaling);
  get_f32(j, "swiglu_limit", &c->swiglu_limit);
  get_bool(j, "norm_topk_prob", &c->norm_topk_prob);

  // compress_ratios drives the whole layer schedule: 0 = uncompressed, 4 = CSA, 128 = HCA.
  // NOTE it is longer than n_layer: the tail entries belong to the DSpark mtp stages, which is why
  // `ratio_for` is indexed by absolute layer and the draft stages read the entries past n_layer.
  auto int_array = [&](const char* key, std::vector<int32_t>* out) {
    const char* p = find_key(j, key);
    if (!p) return;
    while (*p && *p != '[') ++p;
    if (*p != '[') return;
    ++p;
    out->clear();
    while (*p && *p != ']') {
      while (*p == ' ' || *p == ',' || *p == '\n') ++p;
      if (*p == ']' || !*p) break;
      out->push_back((int32_t)std::strtol(p, (char**)&p, 10));
    }
  };
  int_array("compress_ratios", &c->compress_ratios);
  // DSpark. Absent from a container without the draft module, and then dspark_block is 0 and every
  // consumer is off.
  get_u32(j, "dspark_block_size", &c->dspark_block);
  get_u32(j, "dspark_noise_token_id", &c->dspark_noise_token);
  get_u32(j, "dspark_markov_rank", &c->dspark_markov_rank);
  int_array("dspark_target_layer_ids", &c->dspark_taps);
  // ASCENDING, whatever the container said. The reference builds `main_hidden` by appending inside
  // `for i, layer in enumerate(self.layers)`, so the concatenation order is the layer order and not
  // the order the ids happen to be listed in — and `main_proj` was trained against that. The engine
  // writes slot i for taps[i], so an unsorted list would pair every tap with the wrong slice of a
  // 3*n_embd projection. Nothing faults; the draft simply stops predicting, which reads as a
  // checkpoint that drafts badly rather than as a config that was read wrongly.
  if (!std::is_sorted(c->dspark_taps.begin(), c->dspark_taps.end())) {
    std::sort(c->dspark_taps.begin(), c->dspark_taps.end());
    std::fprintf(stderr, "config: dspark_target_layer_ids was not ascending; sorted to match the "
                         "reference's concatenation order\n");
  }
  if (c->n_layer == 0 || c->n_embd == 0) {
    if (err) *err = "config missing num_hidden_layers / hidden_size";
    return false;
  }
  return true;
}

} // namespace aff
