#!/usr/bin/env python3
"""refprefix.py — the REFERENCE forward over a short real prompt, layer by layer on CPU, printing the
same numbers the engine's AFF_NORM_TRACE prints, so the first sublayer where they part names the bug.

    refprefix.py <ckpt-dir> "<prompt text>" [n_layers|all] [--ids 1,2,3]

The engine side (one-shot, raw text + BOS, engram OFF on both sides):
    AFF_NORM_TRACE=1 run.sh once <image> <prompt-file> 1 ... 2>&1 | grep norm-trace

Per block it prints, in the engine's order and units (SUM OF SQUARES over the whole [tokens x dim]
buffer, float64): `nrm` = the attention sublayer input (hc_pre + attn_norm), `blk` = the attention
output, `nrm` = the FFN input, `blk` = the FFN output. Weights are streamed one layer at a time, so
the whole model runs in the RAM of one layer. At the end: the top-5 next tokens for the last position.

The engine traces only the batched prefill, which covers every prompt token but the LAST (the last
one is fed by the first decode/verify step). So the per-block numbers here are over tokens[:-1] too,
and the final logits come from a separate full-length pass... no: from the same pass, last position.
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CK, TEXT = sys.argv[1], sys.argv[2]
NL = sys.argv[3] if len(sys.argv) > 3 else "all"
ids_arg = None
ENG = None                      # aff-engramcheck output for these ids: engram ON in the reference
if "--engram-ids" in sys.argv:
    ENG = sys.argv[sys.argv.index("--engram-ids") + 1]
if "--ids" in sys.argv:
    ids_arg = [int(v) for v in sys.argv[sys.argv.index("--ids") + 1].split(",")]

# Reuse refblock's setup (args, load(), globals) without running its block: import it as a module
# with a stub argv, then build our own blocks.
sys.argv = [sys.argv[0], CK, "0", "/tmp/_refprefix_unused", "4"]
sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "refblock.py")).read()
setup = src.split("with torch.no_grad():")[0]          # everything up to the block run
g = {"__name__": "refblock_setup", "__file__": os.path.join(HERE, "refblock.py")}
exec(compile(setup, "refblock.py", "exec"), g)
M, args, load, index = g["M"], g["args"], g["load"], g["index"]

if ids_arg is None:
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(f"{CK}/tokenizer.json")
    bos = json.load(open(f"{CK}/config.json")).get("bos_token_id",
          json.load(open(f"{CK}/config.json"))["text_config"].get("bos_token_id", 0))
    ids = [bos] + tk.encode(TEXT, add_special_tokens=False).ids
else:
    ids = ids_arg
print("ids", ",".join(map(str, ids)), flush=True)
n_layers = args.n_layers if NL == "all" else int(NL)


def fill(mod, prefix):
    for name, p in mod.named_parameters():
        key = f"{prefix}{name}"
        w = load(key)
        if name.endswith("wo_a.weight"):
            sc = load(key.replace("weight", "scale"))
            w = g["kernel_shim"].dequant_fp8(w, sc, w.size(0) // sc.size(0)).float()
        if w.dtype in (torch.bfloat16, torch.float32):
            w = w.float()
        assert w.shape == p.shape, (key, tuple(w.shape), tuple(p.shape))
        p.data = w
    for m in mod.modules():
        if isinstance(m, M.Linear) and getattr(m, "scale", None) is not None:
            m.weight.scale = m.scale


ENG_LAYERS = (1, 14)            # engram_layer_ids; hash band = position in this tuple


def engram_hash_ids(path, n):
    """[1, n, 2, 24] int64 from aff-engramcheck's `t band c0..c23` lines (proved equal to the
    reference's own NgramHashState on 2026-09-22)."""
    out = torch.zeros(1, n, len(ENG_LAYERS), 24, dtype=torch.int64)
    for line in open(path):
        v = line.split()
        if len(v) < 26: continue
        out[0, int(v[0]), int(v[1])] = torch.tensor([int(x) for x in v[2:26]])
    return out


def table_rows(L, ids):
    """Rows of layers.L.engram.embed dequantised the reference's way (F8 * E8M0 per 32, then bf16),
    read by byte offset so the 91.6 GiB table is never loaded."""
    import struct as _s
    res = {}
    for part in ("weight", "scale"):
        name = f"layers.{L}.engram.embed.{part}"
        f = f"{CK}/{index[name]}"
        with open(f, "rb") as fh:
            n = _s.unpack("<Q", fh.read(8))[0]
            e = json.loads(fh.read(n))[name]; base = 8 + n + e["data_offsets"][0]
            width = e["shape"][1]
            buf = []
            for r in ids.flatten().tolist():
                fh.seek(base + r * width); buf.append(fh.read(width))
        raw = torch.frombuffer(bytearray(b"".join(buf)), dtype=torch.uint8).view(-1, width)
        res[part] = raw
    w = res["weight"].view(torch.float8_e4m3fn).float()                     # [k, 256]
    sc = torch.pow(2.0, res["scale"].float() - 127.0)                       # [k, 8]
    v = (w.unflatten(-1, (-1, 32)) * sc.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
    return v.view(*ids.shape, -1)


def make_engram(L, band):
    eng = M.Engram.__new__(M.Engram); torch.nn.Module.__init__(eng)
    eng.layer_id, eng.layer_hash_index, eng.dim, eng.hc_mult = L, band, args.dim, args.hc_mult
    eng.clamp_value, eng.eps = 1e-6, args.norm_eps
    eng.q_weight = torch.nn.Parameter(load(f"layers.{L}.engram.q_weight").float())
    eng.k_weight = torch.nn.Parameter(load(f"layers.{L}.engram.k_weight").float())
    wkv = M.Linear(24 * 256, args.dim * (args.hc_mult + 1))
    wkv.weight.data = load(f"layers.{L}.engram.wkv.weight")
    wkv.scale.data = load(f"layers.{L}.engram.wkv.scale"); wkv.weight.scale = wkv.scale
    eng.wkv = wkv
    def emb(ids, L=L):
        r = table_rows(L, ids)
        print(f"ref L{L:02d} erow {float(r.float().double().square().sum()):.9e}", flush=True)
        return r
    eng.embed = emb
    def wkv_spy(m, a, o):
        k, val = o.split([args.hc_mult * args.dim, args.dim], dim=-1)
        print(f"ref L{L:02d} ekey {float(k.float().double().square().sum()):.9e} "
              f"eval {float(val.float().double().square().sum()):.9e}", flush=True)
    eng.wkv.register_forward_hook(wkv_spy)
    return eng


def ss(t, ntok):   # sum of squares over tokens[:ntok], like the engine's l2_kernel over its batch
    return float(t[:, :ntok].double().square().sum())


with torch.no_grad():
    x_ids = torch.tensor([ids])
    n_pre = len(ids) if os.environ.get("ALLTOK") else len(ids) - 1
    emb = load("embed.weight").float()
    h = emb[x_ids]                                            # [1, s, dim]
    del emb
    h = h.unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
    pre_mix = M.make_identity_pre_mix(h, args.hc_mult)
    hashes = engram_hash_ids(ENG, len(ids)) if ENG else None
    for L in range(n_layers):
        t0 = time.time()
        if hashes is not None and L in ENG_LAYERS:
            b = ENG_LAYERS.index(L)
            before = ss(h.flatten(2), n_pre)
            h = make_engram(L, b)(h, hashes[:, :, b, :])
            print(f"ref L{L:02d} engram resid {before:.9e} -> {ss(h.flatten(2), n_pre):.9e}", flush=True)
        blk = M.Block(L, args)                                # engram_layout None: engram OFF
        fill(blk, f"layers.{L}.")
        rec = []
        blk.attn.register_forward_pre_hook(lambda m, a: rec.append(("nrm", ss(a[0], n_pre))))
        blk.attn.wo_b.register_forward_pre_hook(lambda m, a: rec.append(("woa", ss(a[0], n_pre))))
        blk.attn.register_forward_hook(lambda m, a, o: rec.append(("blk", ss(o, n_pre))))
        blk.ffn.register_forward_pre_hook(lambda m, a: rec.append(("nrm", ss(a[0], n_pre))))
        blk.ffn.register_forward_hook(lambda m, a, o: rec.append(("blk", ss(o, n_pre))))
        h, pre_mix = blk(h, 0, pre_mix, None)
        for tag, v in rec:
            print(f"ref L{L:02d} {tag} {v:.9e}", flush=True)
        print(f"ref L{L:02d} resid {ss(h.flatten(2), n_pre):.9e}  ({time.time()-t0:.0f}s)", flush=True)
        del blk
    if n_layers == args.n_layers:
        last = M.Block(args.n_layers - 1, args)                # only for hc_pre's arithmetic
        y = torch.sum(pre_mix.unsqueeze(-1) * h.float(), dim=2)
        norm = M.RMSNorm(args.dim, args.norm_eps); norm.weight.data = load("norm.weight").float()
        W = load("head.weight").float()
        logits = torch.nn.functional.linear(norm(y)[:, -1], W)
        if os.environ.get("REF_LOGITS_OUT"):
            logits[0].float().numpy().astype("float32").tofile(os.environ["REF_LOGITS_OUT"])
            print("wrote", os.environ["REF_LOGITS_OUT"], flush=True)
        top = torch.topk(logits[0], 5)
        try:
            names = [tk.decode([i]) for i in top.indices.tolist()]
        except Exception:
            names = ["?"] * 5
        print("top5", [(int(i), n, round(float(v), 3)) for i, n, v in
                       zip(top.indices.tolist(), names, top.values.tolist())], flush=True)
