#include "aff_format.h"

namespace aff {

const AffQuantInfo* quant_info(AffQuant q) noexcept {
  for (const auto& e : kQuantTable)
    if (e.id == q) return &e;
  return nullptr;
}

const char* quant_name(AffQuant q) noexcept {
  const auto* i = quant_info(q);
  return i ? i->name : "unknown";
}

uint64_t quant_data_bytes(AffQuant q, uint64_t n) noexcept {
  const auto* i = quant_info(q);
  if (!i) return 0;
  // Round up to whole super-blocks — a partial block still occupies a full block on disk.
  const uint64_t blocks = (n + i->block - 1) / i->block;
  return blocks * i->data_bytes;
}

uint64_t quant_scale_bytes(AffQuant q, uint64_t n) noexcept {
  const auto* i = quant_info(q);
  if (!i || i->scale_bytes == 0) return 0;
  const uint64_t blocks = (n + i->block - 1) / i->block;
  return blocks * i->scale_bytes;
}

uint64_t quant_row_scale_bytes(AffQuant q, uint64_t rows) noexcept {
  const auto* i = quant_info(q);
  if (!i || i->row_scale_bytes == 0) return 0;
  return rows * i->row_scale_bytes;
}

double quant_bpw(AffQuant q, uint64_t rows, uint64_t cols) noexcept {
  const uint64_t n = rows * cols;
  if (n == 0) return 0.0;
  const uint64_t bytes = quant_data_bytes(q, n) + quant_scale_bytes(q, n) +
                         quant_row_scale_bytes(q, rows);
  return static_cast<double>(bytes) * 8.0 / static_cast<double>(n);
}

uint64_t compute_expert_layout(const ExpertGeometry& g,
                               AffMatrixDesc& gate,
                               AffMatrixDesc& up,
                               AffMatrixDesc& down) noexcept {
  // gate: [intermediate, hidden]   up: [intermediate, hidden]   down: [hidden, intermediate]
  //
  // Layout order inside an expert:
  //   [gate quants][up quants][down quants] [gate blk-scales][up blk-scales][down blk-scales]
  //   [gate row-scales][up row-scales][down row-scales]
  //
  // Quant data is grouped first so the dense weight stream is one contiguous run — what the CPU
  // streaming path and the GPU coalesced loads both want. The two scale tiers form a small SoA
  // tail; row scales are tiny and land last.
  auto lay = [](AffMatrixDesc& m, uint64_t rows, uint64_t cols, AffQuant q) {
    m = {};
    m.rows  = rows;
    m.cols  = cols;
    m.quant = static_cast<uint32_t>(q);
    m.data_size   = quant_data_bytes(q, rows * cols);
    m.scale_size  = quant_scale_bytes(q, rows * cols);
    m.rscale_size = quant_row_scale_bytes(q, rows);
  };

  lay(gate, g.intermediate, g.hidden,       g.gate_q);
  lay(up,   g.intermediate, g.hidden,       g.up_q);
  lay(down, g.hidden,       g.intermediate, g.down_q);

  uint64_t off = 0;
  auto place = [&off](uint64_t& dst_off, uint64_t size) {
    dst_off = off;
    off = align_up(off + size, kBlockAlign);
  };

  place(gate.data_off, gate.data_size);
  place(up.data_off,   up.data_size);
  place(down.data_off, down.data_size);

  place(gate.scale_off, gate.scale_size);
  place(up.scale_off,   up.scale_size);
  place(down.scale_off, down.scale_size);

  place(gate.rscale_off, gate.rscale_size);
  place(up.rscale_off,   up.rscale_size);
  place(down.rscale_off, down.rscale_size);

  return align_up(off, kExpertAlign);
}

} // namespace aff
