#!/usr/bin/env python3
"""refblock.py — run ONE real DeepSeek-V4.1 block on CPU through the checkpoint's own model.py and
dump golden vectors for the affinity port.

    refblock.py <ckpt-dir> <layer> <out-dir> [seqlen]

Writes <out-dir>/L<layer>/{x,pre_mix,out,ffn_pre}.npy (float32, C order) + meta.json. The C++ port
feeds x/pre_mix through its own block and must reproduce out/ffn_pre within tolerance.
"""
import json, struct, sys, os
import numpy as np
import torch

CK, LAYER, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
SEQ = int(sys.argv[4]) if len(sys.argv) > 4 else 16
sys.path.insert(0, os.path.join(CK, "inference"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kernel_shim                      # noqa: E402
sys.modules["kernel"] = kernel_shim     # model.py imports `kernel` -> our CPU stand-ins
torch.set_default_dtype(torch.float32)
torch.manual_seed(0)

import model as M                       # noqa: E402

cfg = json.load(open(f"{CK}/config.json"))
t = cfg["text_config"]
rs = t.get("rope_scaling", {})
args = M.ModelArgs(
    max_batch_size=1, max_seq_len=max(4096, SEQ * 4), dtype="fp8", expert_dtype="fp4",
    vocab_size=t["vocab_size"], dim=t["hidden_size"], moe_inter_dim=t["moe_intermediate_size"],
    n_layers=t["num_hidden_layers"], n_mtp_layers=t.get("num_nextn_predict_layers", 0),
    n_heads=t["num_attention_heads"], n_routed_experts=t["n_routed_experts"],
    n_shared_experts=t["n_shared_experts"], n_activated_experts=t["num_experts_per_tok"],
    score_func=t["scoring_func"], norm_topk_prob=t["norm_topk_prob"],
    route_scale=t["routed_scaling_factor"], swiglu_limit=t.get("swiglu_limit", 0.0),
    q_lora_rank=t["q_lora_rank"], head_dim=t["head_dim"], rope_head_dim=t["qk_rope_head_dim"],
    norm_eps=t["rms_norm_eps"], o_groups=t["o_groups"], o_lora_rank=t["o_lora_rank"],
    window_size=t["sliding_window"], compress_ratios=tuple(t["compress_ratios"]),
    kv_source_layers=tuple(t["kv_source_layer_ids"]), index_source_layers=tuple(t["index_source_layer_ids"]),
    compress_rope_theta=t["compress_rope_theta"], original_seq_len=rs.get("original_max_position_embeddings", 0),
    rope_theta=t["rope_theta"], rope_factor=rs.get("factor", 1), beta_fast=rs.get("beta_fast", 32),
    beta_slow=rs.get("beta_slow", 1), index_n_heads=t["index_n_heads"], index_head_dim=t["index_head_dim"],
    index_topk=t["index_topk"], candidate_source_layer=t.get("candidate_source_layer_id", -1),
    candidate_topk_blocks=t.get("candidate_topk_blocks", 0), candidate_block_size=t.get("candidate_block_size", 0),
    hc_mult=t["hc_mult"], hc_sinkhorn_iters=t["hc_sinkhorn_iters"], hc_eps=t["hc_eps"],
    dspark_block_size=t.get("dspark_block_size", 0), dspark_n_routed_experts=t.get("dspark_n_routed_experts", 0),
    dspark_n_activated_experts=t.get("dspark_num_experts_per_tok", 0),
)
M.default_dtype = torch.float8_e4m3fn
M.fp8_block_size = t_block = cfg["quantization_config"]["weight_block_size"][0]
M.fp4_block_size = 32
M.scale_fmt = cfg["quantization_config"]["scale_fmt"]
M.scale_dtype = torch.float8_e8m0fnu

ST = {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu, "BF16": torch.bfloat16,
      "F32": torch.float32, "I8": torch.int8, "I64": torch.int64}
index = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
_hdr = {}


def load(name):
    f = f"{CK}/{index[name]}"
    if f not in _hdr:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            _hdr[f] = (json.loads(fh.read(n)), 8 + n)
    h, base = _hdr[f]
    e = h[name]
    a, b = e["data_offsets"]
    with open(f, "rb") as fh:
        fh.seek(base + a)
        buf = bytearray(fh.read(b - a))
    x = torch.frombuffer(buf, dtype=ST[e["dtype"]])
    return x.view(*e["shape"]) if e["dtype"] != "I8" else x.view(torch.float4_e2m1fn_x2).view(*e["shape"])


with torch.no_grad():
    block = M.Block(LAYER, args)
    missing = 0
    for name, p in block.named_parameters():
        key = f"layers.{LAYER}.{name}"
        if key not in index:
            print(f"  MISSING {key}"); missing += 1; continue
        w = load(key)
        # convert.py dequantises wo_a to bf16 and drops its scale; the module declares it bf16, so
        # the checkpoint's FP8 pair has to be folded here or the einsum in Attention.forward gets
        # an fp8 tensor it cannot multiply.
        if name.endswith("wo_a.weight"):
            w = kernel_shim.dequant_fp8(w, load(key.replace("weight", "scale")),
                                        w.size(0) // load(key.replace("weight", "scale")).size(0)).float()
        # The real model runs bf16 throughout; this reference runs float32 (more precise, and the
        # comparison is by tolerance), so every non-quantised weight is widened to match.
        if w.dtype in (torch.bfloat16, torch.float32):
            w = w.float()
        if w.shape != p.shape:
            print(f"  SHAPE {key}: ckpt {tuple(w.shape)} vs module {tuple(p.shape)}"); missing += 1; continue
        p.data = w
    if missing:
        sys.exit(f"{missing} parameters unloaded")
    # `.scale` is read off `.weight` by model.linear(), and assigning p.data above detached them.
    for mod in block.modules():
        if isinstance(mod, M.Linear) and getattr(mod, "scale", None) is not None:
            mod.weight.scale = mod.scale

    hc, dim = args.hc_mult, args.dim
    x = (torch.randn(1, SEQ, hc, dim) * 0.02).float()
    pre_mix = M.make_identity_pre_mix(x, hc)
    out, ffn_pre = block(x, 0, pre_mix, None)

os.makedirs(f"{OUT}/L{LAYER}", exist_ok=True)
for n, v in [("x", x), ("pre_mix", pre_mix), ("out", out), ("ffn_pre", ffn_pre)]:
    np.save(f"{OUT}/L{LAYER}/{n}.npy", v.float().numpy())
json.dump({"layer": LAYER, "seqlen": SEQ, "hc_mult": hc, "dim": dim,
           "compress_ratio": args.compress_ratios[LAYER], "seed": 0,
           "out_absmax": float(out.abs().max()), "out_rms": float(out.float().square().mean().sqrt())},
          open(f"{OUT}/L{LAYER}/meta.json", "w"), indent=1)
print(f"L{LAYER}: out absmax {out.abs().max():.4f} rms {out.float().square().mean().sqrt():.4f}, "
      f"ffn_pre[0,0] {ffn_pre[0, 0].tolist()}")
