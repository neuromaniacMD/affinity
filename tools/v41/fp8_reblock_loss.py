#!/usr/bin/env python3
# fp8_reblock_loss.py — can V4.1's FP8 (UE8M0 scale per 32x32 block) be stored as affinity's FP8 format
# (one UE8M0 scale per 128x128 tile) without loss? Folding = keep the tile's MAX sub-exponent and shift
# each element's FP8 value down by (max - own) exponent steps, re-encoded to the nearest E4M3.
# Exact unless a shifted value falls below E4M3's precision (subnormal range / zero).
import json, struct, sys, re, numpy as np
CK = sys.argv[1]; LAYERS = [int(x) for x in sys.argv[2].split(",")]
idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]

def e4m3_table():
    t = np.zeros(256, np.float64)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0; e = (b >> 3) & 0xF; m = b & 7
        if e == 0xF and m == 7: t[b] = np.nan
        elif e == 0: t[b] = s * (m / 8.0) * 2.0 ** -6
        else: t[b] = s * (1 + m / 8.0) * 2.0 ** (e - 7)
    return t
LUT = e4m3_table()
POS = np.unique(np.abs(LUT[~np.isnan(LUT)]))           # sorted representable magnitudes incl 0

def nearest_e4m3(x):
    a = np.abs(x); i = np.clip(np.searchsorted(POS, a), 1, len(POS) - 1)
    lo, hi = POS[i - 1], POS[i]
    r = np.where(a - lo <= hi - a, lo, hi)          # ties to lower magnitude: close enough for a loss bound
    return np.sign(x) * r

def load(name):
    f = f"{CK}/{idx[name]}"
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))[name]
        a, b = h["data_offsets"]; fh.seek(8 + n + a); buf = fh.read(b - a)
    return np.frombuffer(buf, np.uint8).reshape(h["shape"])

agg = {}
for name in sorted(idx):
    if not name.endswith(".weight") or ".experts." in name or ".engram.embed" in name: continue
    m = re.match(r"(layers|mtp)\.(\d+)\.", name)
    if not m or (m.group(1) == "layers" and int(m.group(2)) not in LAYERS) or (m.group(1) == "mtp" and m.group(2) != "0"): continue
    sc_name = name[:-len(".weight")] + ".scale"
    if sc_name not in idx: continue
    W = load(name); S = load(sc_name).astype(np.int32) - 127
    R, C = W.shape
    v = LUT[W]                                           # fp8 value (without block scale)
    esub = np.repeat(np.repeat(S, 32, 0), 32, 1)[:R, :C]  # per-element sub-block exponent
    Et = S.reshape(R // 128, 4, C // 128, 4).max(axis=(1, 3))
    etile = np.repeat(np.repeat(Et, 128, 0), 128, 1)
    d = etile - esub
    x = v * np.exp2(-d.astype(np.float64))
    q = nearest_e4m3(x)
    true = v * np.exp2(esub); new = q * np.exp2(etile)
    changed = np.count_nonzero(q != x); zeroed = np.count_nonzero((q == 0) & (v != 0))
    rel = np.linalg.norm(new - true) / np.linalg.norm(true)
    key = re.sub(r"\.\d+\.", ".N.", name, count=1)
    a = agg.setdefault(key, [0, 0, 0, 0.0, 0.0, 0])
    a[0] += W.size; a[1] += changed; a[2] += zeroed; a[3] += float(np.sum((new - true) ** 2)); a[4] += float(np.sum(true ** 2)); a[5] = max(a[5], int(d.max()))
    print(f"{name:48s} changed {changed / W.size:8.4%}  zeroed {zeroed / W.size:8.4%}  relerr {rel:.2e}  max shift {d.max()}", flush=True)

print("\n== aggregate by tensor type (layers %s + mtp.0)" % LAYERS)
for k, (n, c, z, e2, t2, dm) in sorted(agg.items()):
    print(f"{k:48s} changed {c / n:8.4%}  zeroed {z / n:8.4%}  relerr {np.sqrt(e2 / t2):.2e}  max shift {dm}")
