import re, sys
eng, ref = sys.argv[1], sys.argv[2]
# engine: per rank, the sequence of (tag, value); group into layers by the attn-side 'nrm'
seq = {0: [], 1: []}
for l in open(eng):
    m = re.match(r"norm-trace r(\d) (\S+)\s+\d+ (\S+)", l)
    if m: seq[int(m.group(1))].append((m.group(2), float(m.group(3))))
def layers(s):
    out, cur, half = [], None, 0
    for tag, v in s:
        if tag == "nrm":
            if cur is None or half == 1:          # attention-side nrm opens a new layer
                if cur: out.append(cur)
                cur, half = {"nrm_a": v}, 0
            else:
                cur["nrm_f"] = v; half = 1
        elif tag in ("woa",): cur["woa"] = v
        elif tag == "blk": cur["blk_a" if half == 0 else "blk_f"] = v
    if cur: out.append(cur)
    return out
e0, e1 = layers(seq[0]), layers(seq[1])
R = {}
for l in open(ref):
    m = re.match(r"ref L(\d+) (nrm|woa|blk) (\S+)", l)
    if not m: continue
    L, tag, v = int(m.group(1)), m.group(2), float(m.group(3))
    d = R.setdefault(L, {"_n": 0})
    if tag == "nrm": d["nrm_a" if "nrm_a" not in d else "nrm_f"] = v
    elif tag == "woa": d["woa"] = v
    elif tag == "blk": d["blk_a" if "blk_a" not in d else "blk_f"] = v
print(f"{'L':>3} {'nrm_a':>8} {'woa':>8} {'blk_a':>8} {'nrm_f':>8} {'blk_f':>8}   (engine/ref ratio of sum of squares)")
for L in range(min(len(e0), len(R))):
    a, b, r = e0[L], e1[L], R[L]
    def rat(k, summed=False):
        if k not in r or k not in a: return "   -    "
        ev = a[k] + (b.get(k, 0) if summed else 0)
        return f"{ev / r[k]:8.3f}"
    print(f"{L:>3} {rat('nrm_a')} {rat('woa', True)} {rat('blk_a')} {rat('nrm_f')} {rat('blk_f')}")
