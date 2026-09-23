#!/usr/bin/env python3
"""Emit the constants Engram's n-gram hash needs, straight out of the checkpoint's own code.

The hash has to match `engram.py` exactly or every lookup is a different row of a 384-million-row
table -- which is not a crash, it is noise added to the residual stream. So nothing here is
reimplemented: this IMPORTS the reference's `build_compressed_token_map`, `EngramLayout` and
`compute_hash_multipliers` and writes down what they produce.

That matters because two of the three are unreasonable to reproduce in C++:

  - the compressed token map runs the HF `tokenizers` Rust normalizer stack (NFKC, NFD, strip
    accents, lowercase, whitespace regexes) over all 129,280 ids;
  - the multipliers come out of numpy's PCG64 `default_rng(10007 * layer_id).integers(...)`, whose
    bounded-integer draw is Lemire rejection over that specific bit stream.

Both are deterministic, and both are a few hundred KB once evaluated, so they belong in a file
rather than in the engine. The primes are `sympy.isprime` walks, which is reproducible but pointless
to redo.

Usage:
    engram_prep.py <checkpoint-dir> <out.engram>
    engram_prep.py <checkpoint-dir> <out.engram> --dump-hashes <ids.txt> <out.npy>

The second form writes the reference hash ids for a token sequence, which is what the engine's own
implementation is checked against.

File format, all little-endian, which `aff_engram.cpp` reads:

    magic   "AFFENGRM"                       8 B
    version u32 = 2
    n_layer u32                              engram layers (2)
    max_ngram u32                            4
    n_heads u32                              8
    head_dim u32                             256
    n_cols u32                               (max_ngram-1)*n_heads = 24
    vocab u32                                tokenizer vocab (129280)
    cvocab u32                               compressed vocab (99092)
    pad u32                                  the COMPRESSED pad id, already mapped
    rows[n_layer]     u64                    table rows per layer
    multipliers[n_layer][max_ngram]  i64
    primes[n_layer][max_ngram-1][n_heads] i64
    offsets[n_layer][n_cols]         i64
    token_map[vocab]                 i32
    then, per layer, WHERE THE TABLE IS (version 2):
      name_len u32, name[name_len]            shard file name, relative to the checkpoint dir
      w_off u64, s_off u64                    byte offsets of embed.weight / embed.scale in it
      w_bytes u64, s_bytes u64
The engine mmaps those two ranges rather than parsing safetensors: the table is 91.6 GiB a layer and
the only thing it needs from the container's JSON is where it starts.
"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]) / "inference"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from engram import (  # noqa: E402
    EngramLayout,
    build_compressed_token_map,
    compute_hash_multipliers,
)

MAGIC = b"AFFENGRM"


class Args:
    """The handful of fields EngramLayout.from_args reads, out of config.json."""

    def __init__(self, cfg):
        self.engram_layer_ids = tuple(cfg["engram_layer_ids"])
        self.engram_num_embeddings = tuple(cfg["engram_num_embeddings"])
        self.engram_max_ngram_size = cfg["engram_max_ngram_size"]
        self.engram_vocab_size = cfg["engram_vocab_size"]
        self.engram_n_heads = cfg["engram_n_heads"]
        self.engram_head_dim = cfg["engram_head_dim"]
        self.engram_pad_id = cfg.get("engram_pad_token_id", cfg.get("engram_pad_id", 2))
        self.engram_compressed_vocab_size = cfg["engram_compressed_vocab_size"]


def table_locations(ckpt: Path, layer_ids):
    """Where each layer's embed.weight / embed.scale actually sit, from the shard headers.

    Read here rather than in the engine: safetensors is a JSON blob in front of the data, and the
    engine needs exactly two byte ranges out of it per layer.
    """
    import json
    import struct

    index = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    out = []
    for l in layer_ids:
        wk, sk = f"layers.{l}.engram.embed.weight", f"layers.{l}.engram.embed.scale"
        shard = index[wk]
        assert index[sk] == shard, (wk, sk, "expected both in one shard")
        with open(ckpt / shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        base = 8 + n
        w, sc = hdr[wk], hdr[sk]
        assert w["dtype"] == "F8_E4M3", w["dtype"]
        assert sc["dtype"] == "F8_E8M0", sc["dtype"]
        out.append(
            dict(
                shard=shard,
                w_off=base + w["data_offsets"][0],
                w_bytes=w["data_offsets"][1] - w["data_offsets"][0],
                s_off=base + sc["data_offsets"][0],
                s_bytes=sc["data_offsets"][1] - sc["data_offsets"][0],
                shape=w["shape"],
                sshape=sc["shape"],
            )
        )
    return out


def build(ckpt: Path):
    import json

    cfg = json.loads((ckpt / "config.json").read_text())["text_config"]
    args = Args(cfg)
    layout = EngramLayout.from_args(args)
    assert layout is not None, "config has no engram layers"

    tok = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
    token_map, cvocab = build_compressed_token_map(tok)
    # The reference asserts this, and so should we: every multiplier is derived from cvocab, so a
    # mismatch silently rehashes the whole table rather than failing.
    assert cvocab == args.engram_compressed_vocab_size, (cvocab, args.engram_compressed_vocab_size)

    mult = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, cvocab).numpy()
    primes = np.array(layout.primes, dtype=np.int64)          # [L][ng-1][heads]
    flat = primes.reshape(len(layout.layer_ids), -1)          # [L][n_cols], the order hashes land in
    offsets = np.concatenate(
        [np.zeros((flat.shape[0], 1), np.int64), np.cumsum(flat[:, :-1], axis=1)], axis=1
    )
    loc = table_locations(ckpt, layout.layer_ids)
    return args, layout, np.array(token_map, np.int32), cvocab, mult, primes, offsets, loc


def write(path: Path, args, layout, token_map, cvocab, mult, primes, offsets, loc):
    n_cols = (layout.max_ngram_size - 1) * layout.n_heads
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(
            struct.pack(
                "<9I",
                2,
                len(layout.layer_ids),
                layout.max_ngram_size,
                layout.n_heads,
                layout.head_dim,
                n_cols,
                len(token_map),
                cvocab,
                int(token_map[args.engram_pad_id]),
            )
        )
        np.array(layout.num_embeddings, np.uint64).tofile(f)
        mult.astype(np.int64).tofile(f)
        primes.astype(np.int64).tofile(f)
        offsets.astype(np.int64).tofile(f)
        token_map.astype(np.int32).tofile(f)
        for d in loc:
            name = d["shard"].encode()
            f.write(struct.pack("<I", len(name)))
            f.write(name)
            f.write(struct.pack("<4Q", d["w_off"], d["s_off"], d["w_bytes"], d["s_bytes"]))


def main():
    ckpt = Path(sys.argv[1])
    out = Path(sys.argv[2])
    args, layout, token_map, cvocab, mult, primes, offsets, loc = build(ckpt)
    write(out, args, layout, token_map, cvocab, mult, primes, offsets, loc)
    print(f"{out}: layers={list(layout.layer_ids)} max_ngram={layout.max_ngram_size} "
          f"heads={layout.n_heads} head_dim={layout.head_dim} cols={(layout.max_ngram_size-1)*layout.n_heads}")
    print(f"  compressed vocab {cvocab} (asserted against config), pad -> {token_map[args.engram_pad_id]}")
    print(f"  rows {list(layout.num_embeddings)}")
    print(f"  multipliers {mult.tolist()}")
    print(f"  primes[0][0][:3] {primes[0][0][:3].tolist()}  offsets[0][:4] {offsets[0][:4].tolist()}")
    for l, d in zip(layout.layer_ids, loc):
        print(f"  layer {l} table {d['shape']} in {d['shard']} at {d['w_off']} "
              f"({d['w_bytes'] / 2**30:.1f} GiB) + scale {d['sshape']} at {d['s_off']}")
    print(f"  wrote {out.stat().st_size} bytes")

    if len(sys.argv) > 4 and sys.argv[3] == "--dump-hashes":
        # The reference's own NgramHashState over a token sequence, for the engine to match.
        from engram import NgramHashState
        from transformers import AutoTokenizer as AT

        ids = [int(x) for x in Path(sys.argv[4]).read_text().split()]

        class A2(Args):
            max_batch_size = 1
            max_seq_len = max(len(ids), 8)

        cfg = __import__("json").loads((ckpt / "config.json").read_text())["text_config"]
        a2 = A2(cfg)
        a2.max_batch_size, a2.max_seq_len = 1, max(len(ids), 8)
        st = NgramHashState(a2, layout, AT.from_pretrained(str(ckpt), trust_remote_code=True))
        h = st(torch.tensor([ids]), 0)          # [1, L, n_layers, n_cols]
        np.save(sys.argv[5], h[0].numpy().astype(np.int64))
        print(f"  hashes {tuple(h.shape)} -> {sys.argv[5]}")


if __name__ == "__main__":
    main()
