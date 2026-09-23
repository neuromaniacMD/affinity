// aff-engramcheck — the engine's n-gram hash, printed, so it can be diffed against the reference.
//
// The hash has to match `engram.py` exactly: a mismatch reads a different row of a 384-million-row
// table, which adds noise to the residual stream rather than failing. So it gets checked directly,
// against `tools/v41/engram_prep.py --dump-hashes`, before anything is built on top of it.
//
//   aff-engramcheck <model.engram> <ids.txt> [--decode | --resume K | --resume-noseed K]
//                   [--gather <ckpt-dir> <out.bin>]
//
// `--decode` pushes the ids ONE at a time instead of as one chunk, which is the path a decode step
// takes: the look-back then has to reach into what earlier pushes cached. Both forms must print the
// same table, and that equality is the real test — it is the prefill/decode split that a rolling
// hash state gets wrong.
//
// `--resume K` is a prefix-cache resume at position K: the look-back is first filled by a DIFFERENT
// sequence (the ids reversed — what another request leaves behind), then re-seeded the way
// Model::restore_state does, then the suffix is pushed from position K. It must print the same
// table as a straight run. `--resume-noseed K` skips the re-seed and must NOT, which is what shows
// the check can see the bug it guards against. Rows below K are taken from a straight run.
//
// `--gather` additionally mmaps the real tables out of the checkpoint and writes the dequantised
// lookup for every (token, layer) as f32, which is what checks the two things the hash cannot: the
// byte offsets, and the FP8 + E8M0 decode.
#include "engine/engram.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: aff-engramcheck <model.engram> <ids.txt> [--decode]\n");
    return 2;
  }
  const bool decode = argc > 3 && std::string(argv[3]) == "--decode";
  const bool resume = argc > 4 && (std::string(argv[3]) == "--resume" ||
                                   std::string(argv[3]) == "--resume-noseed");
  const bool noseed = resume && std::string(argv[3]) == "--resume-noseed";
  const size_t resume_at = resume ? (size_t)std::strtoul(argv[4], nullptr, 10) : 0;

  aff::EngramConsts c;
  std::string err;
  if (!c.load(argv[1], &err)) {
    std::fprintf(stderr, "fatal: %s\n", err.c_str());
    return 1;
  }
  std::fprintf(stderr,
               "engram: %u layers, %u-gram, %u heads, head_dim %u, %u cols/layer; vocab %u -> %u, "
               "pad %u; rows %llu/%llu\n",
               c.n_layer, c.max_ngram, c.n_heads, c.head_dim, c.n_cols, c.vocab, c.cvocab, c.pad,
               (unsigned long long)c.rows[0], (unsigned long long)(c.rows.size() > 1 ? c.rows[1] : 0));

  std::vector<uint32_t> ids;
  {
    FILE* f = std::fopen(argv[2], "r");
    if (!f) { std::fprintf(stderr, "fatal: cannot open %s\n", argv[2]); return 1; }
    long v;
    while (std::fscanf(f, "%ld", &v) == 1) ids.push_back((uint32_t)v);
    std::fclose(f);
  }
  if (ids.empty()) { std::fprintf(stderr, "fatal: no ids\n"); return 1; }

  const char* gather_dir = nullptr;
  const char* gather_out = nullptr;
  for (int i = 3; i + 2 < argc + 1 && i < argc; ++i)
    if (std::string(argv[i]) == "--gather" && i + 2 < argc) {
      gather_dir = argv[i + 1];
      gather_out = argv[i + 2];
    }

  aff::EngramHash h;
  h.init(&c, ids.size() + 8);
  std::vector<int64_t> out((size_t)ids.size() * c.n_layer * c.n_cols);
  if (decode)
    for (size_t t = 0; t < ids.size(); ++t)
      h.push(&ids[t], 1, t, out.data() + t * c.n_layer * c.n_cols);
  else
    h.push(ids.data(), (uint32_t)ids.size(), 0, out.data());
  if (resume) {
    if (resume_at == 0 || resume_at >= ids.size()) {
      std::fprintf(stderr, "fatal: --resume K needs 0 < K < %zu\n", ids.size());
      return 2;
    }
    const size_t row = (size_t)c.n_layer * c.n_cols;
    std::vector<uint32_t> other(ids.rbegin(), ids.rend());
    std::vector<int64_t> scratch(other.size() * row);
    aff::EngramHash r;
    r.init(&c, ids.size() + 8);
    r.push(other.data(), (uint32_t)other.size(), 0, scratch.data());   // the stale look-back
    if (!noseed) {
      const size_t k = std::min<size_t>(resume_at, r.lookback());
      r.seed(ids.data() + (resume_at - k), (uint32_t)k, resume_at - k);
    }
    r.push(ids.data() + resume_at, (uint32_t)(ids.size() - resume_at), resume_at,
           out.data() + resume_at * row);
  }

  if (gather_dir) {
    aff::EngramTables t;
    if (!t.open(&c, gather_dir, &err)) {
      std::fprintf(stderr, "fatal: %s\n", err.c_str());
      return 1;
    }
    std::fprintf(stderr, "engram tables: %.1f GiB mapped\n", (double)t.bytes() / (1024.0*1024*1024));
    std::vector<float> row((size_t)c.n_cols * c.head_dim);
    FILE* g = std::fopen(gather_out, "wb");
    if (!g) { std::fprintf(stderr, "fatal: cannot write %s\n", gather_out); return 1; }
    for (size_t tt = 0; tt < ids.size(); ++tt)
      for (uint32_t l = 0; l < c.n_layer; ++l) {
        t.gather(l, out.data() + (tt * c.n_layer + l) * c.n_cols, row.data());
        std::fwrite(row.data(), sizeof(float), row.size(), g);
      }
    std::fclose(g);
    std::fprintf(stderr, "wrote %zu x %u x %zu f32 to %s\n", ids.size(), c.n_layer, row.size(),
                 gather_out);
  }

  // One line a (token, layer): the ids the reference's [L][n_layers][n_cols] tensor holds.
  for (size_t t = 0; t < ids.size(); ++t)
    for (uint32_t l = 0; l < c.n_layer; ++l) {
      std::printf("%zu %u", t, l);
      const int64_t* r = out.data() + (t * c.n_layer + l) * c.n_cols;
      for (uint32_t i = 0; i < c.n_cols; ++i) std::printf(" %lld", (long long)r[i]);
      std::printf("\n");
    }
  return 0;
}
