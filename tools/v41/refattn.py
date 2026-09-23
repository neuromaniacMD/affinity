#!/usr/bin/env python3
"""Layer-0 attention internals in the REFERENCE, as sums of squares over all tokens — to line up
against the engine's `woa` / `blk` trace and find which step of attention diverges.
    refattn.py <ckpt> <ids> [layer]
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
CK, IDS = sys.argv[1], [int(v) for v in sys.argv[2].split(",")]
LAYER = int(sys.argv[3]) if len(sys.argv) > 3 else 0
sys.argv = [sys.argv[0], CK, "0", "/tmp/_refattn_unused", "4"]
sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "refblock.py")).read().split("with torch.no_grad():")[0]
g = {"__name__": "refblock_setup", "__file__": os.path.join(HERE, "refblock.py")}
exec(compile(src, "refblock.py", "exec"), g)
M, args, load, torch = g["M"], g["args"], g["load"], g["torch"]


def fill(mod, prefix):
    for name, p in mod.named_parameters():
        key = f"{prefix}{name}"; w = load(key)
        if name.endswith("wo_a.weight"):
            sc = load(key.replace("weight", "scale"))
            w = g["kernel_shim"].dequant_fp8(w, sc, w.size(0) // sc.size(0)).float()
        if w.dtype in (torch.bfloat16, torch.float32): w = w.float()
        p.data = w
    for m in mod.modules():
        if isinstance(m, M.Linear) and getattr(m, "scale", None) is not None:
            m.weight.scale = m.scale


ss = lambda t: float(t.double().square().sum())
with torch.no_grad():
    x_ids = torch.tensor([IDS])
    h = load("embed.weight").float()[x_ids].unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
    pre_mix = M.make_identity_pre_mix(h, args.hc_mult)
    for L in range(LAYER + 1):
        blk = M.Block(L, args); fill(blk, f"layers.{L}.")
        if L == LAYER:
            A = blk.attn
            real_sparse = M.sparse_attn
            def spy(q, kv, sink, idx, scale):
                o = real_sparse(q, kv, sink, idx, scale)
                print(f"L{L} q(roped) {ss(q):.6e}  kv {ss(kv):.6e}  o(core) {ss(o):.6e}  scale {scale:.6f}  "
                      f"sink {A.attn_sink[:4].tolist()}  q.shape {tuple(q.shape)} kv.shape {tuple(kv.shape)}")
                return o
            M.sparse_attn = spy
            A.wq_b.register_forward_hook(lambda m, a, o: print(f"L{L} wq_b(q, pre-rope) {ss(o):.6e}  qr {ss(a[0]):.6e}"))
            A.wkv.register_forward_hook(lambda m, a, o: print(f"L{L} wkv(raw kv) {ss(o):.6e}"))
            A.kv_norm.register_forward_hook(lambda m, a, o: print(f"L{L} kv_norm {ss(o):.6e}"))
            A.wo_b.register_forward_pre_hook(lambda m, a: print(f"L{L} woa (wo_b input) {ss(a[0]):.6e}"))
            A.wo_b.register_forward_hook(lambda m, a, o: print(f"L{L} blk (wo_b out) {ss(o):.6e}"))
        h, pre_mix = blk(h, 0, pre_mix, None)
        if L == LAYER:
            M.sparse_attn = real_sparse
